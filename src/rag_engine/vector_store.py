import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions


class FinancialVectorStore:
    def __init__(self, db_path: str, collection_name: str = "financial_reports"):
        print(f"Initializing Vector Store at: {db_path}")
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path)
        self.embedding_fn = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_parsed_json(
        self,
        json_path: str,
        year_tag: str,
        *,
        document_tag: str | None = None,
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        resolved_tag = document_tag or data.get("document_name")
        if not resolved_tag:
            raise ValueError(f"Missing document_name in {json_path}")

        documents, metadatas, ids = [], [], []

        for i, item in enumerate(data.get("dual_stream_paragraphs", [])):
            text = " ".join(
                str(item.get(key, "") or "")
                for key in ("stream_1_nvdm", "stream_2_finbert")
            ).strip()

            if len(text.split()) <= 10:
                continue

            documents.append(text)
            metadatas.append({
                "year": year_tag,
                "doc_name": data.get("document_name", resolved_tag),
                "document_tag": resolved_tag,
                "para_id": i,
            })
            ids.append(f"{resolved_tag}_{i}")

        if documents:
            # Upsert makes rerunning indexing safe for existing document IDs.
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        print(f"Indexed {len(documents)} paragraphs for {resolved_tag}.")

    def search_baseline_context(
        self,
        query_text: str,
        target_year: str = "Baseline",
        n_results: int = 2,
        *,
        document_tag: str | None = None,
    ):
        if self.collection.count() == 0:
            return [], []

        where = {"year": target_year}
        if document_tag:
            where = {
                "$and": [
                    {"year": target_year},
                    {"document_tag": document_tag},
                ]
            }

        result = self.collection.query(
            query_texts=[query_text],
            n_results=min(n_results, self.collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return (
            (result.get("documents") or [[]])[0],
            (result.get("metadatas") or [[]])[0],
        )
