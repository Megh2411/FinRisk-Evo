import os
import json
import chromadb
from chromadb.utils import embedding_functions

class FinancialVectorStore:
    def __init__(self, db_path: str, collection_name: str = "financial_reports"):
        print(f"Initializing Vector Store at: {db_path}")
        # Initialize a persistent ChromaDB client so data is saved to disk
        self.client = chromadb.PersistentClient(path=db_path)
        
        # We use a lightweight, highly efficient embedding model perfect for CPU inference
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Get or create the collection, optimized for Cosine Similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"} 
        )

    def ingest_parsed_json(self, json_path: str, year_tag: str):
        """Reads the dual-stream JSON and loads the raw text into the vector database."""
        print(f"Ingesting {year_tag} data from {json_path} into ChromaDB...")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        metadatas = []
        ids = []
        
        for i, item in enumerate(data["dual_stream_paragraphs"]):
            # We use stream_2 (the preserved sequence) for human-readable RAG context
            text = item["stream_2_finbert"]
            
            # Filter out extremely short/useless phrases (like table headers)
            if len(text.split()) > 10:
                documents.append(text)
                # Metadata is crucial for the RAG engine to filter by year
                metadatas.append({"year": year_tag, "doc_name": data["document_name"], "para_id": i})
                ids.append(f"{year_tag}_{i}")
        
        # ChromaDB handles batching automatically, but we chunk it for clean terminal output
        batch_size = 500
        for i in range(0, len(documents), batch_size):
            print(f"Adding batch {i} to {min(i + batch_size, len(documents))}...")
            self.collection.add(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
        print(f"Successfully ingested {len(documents)} high-quality paragraphs for {year_tag}.")

    def search_baseline_context(self, query_text: str, target_year: str, n_results: int = 2):
        """Finds the most semantically similar paragraphs from a specific baseline year."""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"year": target_year} # Filter specifically for the baseline year
        )
        
        # Return the retrieved documents and their metadata
        return results["documents"][0], results["metadatas"][0]

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DB_PATH = os.path.join(BASE_DIR, "data", "04_vector_db")
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "03_processed")
    
    input_file = os.path.join(PROCESSED_DIR, "Mahindra_FY24_dual_stream.json")
    
    if os.path.exists(input_file):
        # 1. Initialize the database
        vector_store = FinancialVectorStore(db_path=DB_PATH)
        
        # 2. Ingest the FY24 baseline data
        vector_store.ingest_parsed_json(input_file, year_tag="FY24")
        
        # 3. Simulate the pipeline looking for a match
        print("\n" + "="*50)
        print("SIMULATING RAG RETRIEVAL")
        print("="*50)
        
        # Imagine the parser just extracted this sentence from the FY25 report:
        fake_fy25_sentence = "Due to the global economic slowdown, our tractor exports dropped significantly in key markets."
        print(f"FY25 Query: '{fake_fy25_sentence}'\n")
        
        print("Searching Vector DB for the FY24 Baseline...")
        docs, metas = vector_store.search_baseline_context(query_text=fake_fy25_sentence, target_year="FY24")
        
        for i in range(len(docs)):
            print(f"\n--- MATCH {i+1} (ID: {metas[i]['para_id']}) ---")
            print(f"Text: {docs[i]}")
        print("="*50)
    else:
        print(f"Cannot find processed data at: {input_file}")