import argparse
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_prep.pdf_extractor import extract_financial_narrative
from src.data_prep.text_pipeline import DualStreamProcessor
from src.compute_master_scores import compute_all_scores


RAW_DIR = PROJECT_ROOT / "data" / "01_raw"
PARSED_DIR = PROJECT_ROOT / "data" / "02_parsed"
PROCESSED_DIR = PROJECT_ROOT / "data" / "03_processed"
CSV_OUT = PROJECT_ROOT / "data" / "master_fres_scores_pipeline_200.csv"


def ensure_processed(pdf_path: Path, expected_document: str, force: bool = False) -> Path:
    processed_path = PROCESSED_DIR / f"{expected_document}_dual_stream.json"

    if processed_path.is_file() and not force:
        print(f"Reusing processed file: {processed_path.name}")
        return processed_path

    if not pdf_path.is_file():
        raise FileNotFoundError(f"Required PDF not found: {pdf_path}")

    parsed_path = extract_financial_narrative(str(pdf_path), str(PARSED_DIR))
    processor = DualStreamProcessor()
    output_path = Path(processor.run_pipeline(parsed_path, str(PROCESSED_DIR)))

    if not output_path.is_file():
        raise FileNotFoundError(f"Preprocessing did not create: {output_path}")

    import json
    with output_path.open("r", encoding="utf-8") as f:
        document = json.load(f)

    if document.get("document_name") != expected_document:
        raise ValueError(
            f"Expected document identity {expected_document!r}, "
            f"got {document.get('document_name')!r} in {output_path.name}"
        )

    return output_path


def execute_full_pipeline(base_pdf: Path, target_pdf: Path, force_preprocess: bool = False):
    baseline_json = ensure_processed(base_pdf, "Mahindra_FY24", force_preprocess)
    target_json = ensure_processed(target_pdf, "Mahindra_FY25", force_preprocess)

    # The baseline index is maintained separately. Do not delete or rebuild it here.
    compute_all_scores(
        str(target_json),
        str(CSV_OUT),
        baseline_document_tag="Mahindra_FY24",
    )

    print(f"\nScoring completed. Output: {CSV_OUT}")
    return CSV_OUT


def main():
    parser = argparse.ArgumentParser(
        description="Run FinRisk-Evo scoring for Mahindra FY24 versus FY25."
    )
    parser.add_argument(
        "--force-preprocess",
        action="store_true",
        help="Re-extract and preprocess both PDFs even if processed JSON files exist.",
    )
    args = parser.parse_args()

    base_pdf = RAW_DIR / "Mahindra_FY24.pdf"
    target_pdf = RAW_DIR / "Mahindra_FY25.pdf"

    start_time = time.time()
    print("=== FINRISK-EVO: MAHINDRA FY24 -> FY25 ===")

    try:
        output = execute_full_pipeline(
            base_pdf,
            target_pdf,
            force_preprocess=args.force_preprocess,
        )
    except Exception as exc:
        print(f"Pipeline failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    print(f"Completed in {time.time() - start_time:.2f} seconds.")
    print(f"CSV: {output}")


if __name__ == "__main__":
    main()
