
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag_engine.vector_store import FinancialVectorStore


PROCESSED_DIR = PROJECT_ROOT / "data" / "03_processed"
DB_PATH = PROJECT_ROOT / "data" / "04_vector_db_v2"

# Only verified same-company pairs belong here.
# The Kotak file is an error-clarification filing, so it is excluded.
BASELINES = [
    {
        "path": "Mahindra_FY24_dual_stream.json",
        "year_tag": "Baseline",
        "document_tag": "Mahindra_FY24",
    },
    {
        "path": "RIL_IAR_2024_dual_stream.json",
        "year_tag": "Baseline",
        "document_tag": "RIL_IAR_2024",
    },
]


def main():
    DB_PATH.mkdir(parents=True, exist_ok=True)

    store = FinancialVectorStore(
        db_path=str(DB_PATH),
        collection_name="financial_reports",
    )

    for entry in BASELINES:
        json_path = PROCESSED_DIR / entry["path"]
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing baseline JSON: {json_path}")

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        actual_name = data.get("document_name")
        if actual_name != entry["document_tag"]:
            raise ValueError(
                f"Document identity mismatch in {json_path.name}: "
                f"expected {entry['document_tag']!r}, got {actual_name!r}"
            )

        store.ingest_parsed_json(
            str(json_path),
            year_tag=entry["year_tag"],
            document_tag=entry["document_tag"],
        )

    print("\n=== INDEX VERIFICATION ===")
    print(f"Database: {DB_PATH}")
    print(f"Collection count: {store.collection.count()}")

    for entry in BASELINES:
        result = store.collection.get(
            where={"document_tag": entry["document_tag"]},
            limit=1,
            include=["metadatas"],
        )
        print(
            f"{entry['document_tag']}: "
            f"{len(result['metadatas'])} sample record(s)"
        )


if __name__ == "__main__":
    main()
