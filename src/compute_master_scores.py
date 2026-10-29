import csv
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "src":
    PROJECT_ROOT = PROJECT_ROOT.parent
elif (PROJECT_ROOT / "src").is_dir():
    pass
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_engine.vector_store import FinancialVectorStore
from src.rag_engine.re_ranker import RiskReRanker
from src.models.risk_drift import MasterRiskPipeline


DB_PATH = PROJECT_ROOT / "data" / "04_vector_db_v2"
WEIGHTS = PROJECT_ROOT / "src" / "models" / "saved_weights"
NVDM_WEIGHTS = WEIGHTS / "nvdm_weights.pt"
VECTORIZER_PATH = WEIGHTS / "nvdm_vectorizer.pkl"
FINBERT_WEIGHTS = PROJECT_ROOT / "src" / "models" / "finbert_tncrm_weights"
BIGRU_WEIGHTS = WEIGHTS / "bigru_weights.pt"


def compute_all_scores(
    target_json_path,
    output_csv_path,
    max_nodes=200,
    baseline_document_tag=None,
    baseline_year_tag="Baseline",
    collection_name="financial_reports",
):
    if not baseline_document_tag:
        raise ValueError(
            "Specify baseline_document_tag, e.g. 'Mahindra_FY24' "
            "or 'RIL_IAR_2024'."
        )

    target_path = Path(target_json_path)
    output_path = Path(output_csv_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with target_path.open("r", encoding="utf-8") as f:
        target_data = json.load(f)

    paragraphs = target_data.get("dual_stream_paragraphs", [])
    target_name = target_data.get("document_name", target_path.stem)

    store = FinancialVectorStore(
        db_path=str(DB_PATH),
        collection_name=collection_name,
    )
    ranker = RiskReRanker()
    pipeline = MasterRiskPipeline(
        str(NVDM_WEIGHTS),
        str(VECTORIZER_PATH),
        str(FINBERT_WEIGHTS),
        str(BIGRU_WEIGHTS),
    )

    rows = []
    selected = paragraphs[:max_nodes] if max_nodes else paragraphs

    for i, item in enumerate(selected):
        query_text = " ".join(
            str(item.get(key, "") or "")
            for key in ("stream_1_nvdm", "stream_2_finbert")
        ).strip()

        row = {
            "target_document": target_name,
            "target_para_id": i,
            "baseline_document": baseline_document_tag,
            "baseline_para_id": "",
            "target_text": query_text,
            "baseline_text": "",
            "reranker_score": "",
            "fres_score": "",
            "attention": "",
            "confidence": "",
            "dispersion": "",
            "semantic_similarity": "",
            "category": "",
            "status": "",
            "error": "",
        }

        if len(query_text.split()) < 8:
            row["status"] = "SKIPPED_SHORT_TARGET"
            rows.append(row)
            continue

        try:
            candidates, metadata = store.search_baseline_context(
                query_text=query_text,
                target_year=baseline_year_tag,
                n_results=5,
                document_tag=baseline_document_tag,
            )

            ranked = ranker.rank_candidates(query_text, candidates)
            if not ranked:
                row["status"] = "NO_BASELINE_MATCH"
                rows.append(row)
                continue

            best = ranked[0]
            baseline_text = best["text"]
            candidate_index = best["candidate_index"]
            baseline_meta = metadata[candidate_index]

            row["baseline_para_id"] = baseline_meta.get("para_id", "")
            row["baseline_text"] = baseline_text
            row["reranker_score"] = best["reranker_score"]

            fres, attention, confidence, dispersion, similarity, category = (
                pipeline.evaluate_temporal_risk(baseline_text, query_text)
            )

            row.update({
                "fres_score": fres,
                "attention": attention,
                "confidence": confidence,
                "dispersion": dispersion,
                "semantic_similarity": similarity,
                "category": category,
                "status": "OK",
            })

        except Exception as exc:
            row["status"] = "ERROR"
            row["error"] = type(exc).__name__

        rows.append(row)

    columns = list(rows[0].keys()) if rows else [
        "target_document", "target_para_id", "baseline_document",
        "baseline_para_id", "target_text", "baseline_text",
        "reranker_score", "fres_score", "attention", "confidence",
        "dispersion", "semantic_similarity", "category", "status", "error",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    print(f"Target document: {target_name}")
    print(f"Baseline document: {baseline_document_tag}")
    print(f"Rows written: {len(rows)}")
    print(f"Status counts: {counts}")
    print(f"Diagnostics CSV: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare target annual-report paragraphs to a named baseline."
    )
    parser.add_argument("target_json")
    parser.add_argument("output_csv")
    parser.add_argument("--baseline-document-tag", required=True)
    parser.add_argument("--max-nodes", type=int, default=200)
    args = parser.parse_args()

    compute_all_scores(
        args.target_json,
        args.output_csv,
        max_nodes=args.max_nodes,
        baseline_document_tag=args.baseline_document_tag,
    )
