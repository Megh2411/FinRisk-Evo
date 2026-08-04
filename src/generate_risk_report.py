import os
import sys
import json
import torch
import warnings
import random

# Suppress HuggingFace warnings for clean output
warnings.filterwarnings("ignore")

# Import your custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.risk_drift import MasterRiskPipeline
from src.rag_engine.vector_store import FinancialVectorStore
from src.data_prep.tncrm_dataset import TNCRMDataBuilder

def clean_corporate_noise(text: str) -> bool:
    """
    Returns True if the text is actual narrative. 
    Returns False if it's boilerplate corporate/legal noise.
    """
    # Expanded list of legal/administrative stopwords
    noise_keywords = [
        "regd. office", "cin no", "fax:", "tel:", "email:", "sebi circular",
        "annual general meeting", "agm", "compliance under regulation", 
        "board of directors", "dear sir", "madam", "enclosed",
        "ratification", "resolved that", "adoption of", "din:",
        "ordinary (equity)", "exchange plaza", "bandra-kurla",
        "appointment of", "re-appointment", "rupees", "lakhs fifty"
    ]
    text_lower = text.lower()
    
    if any(keyword in text_lower for keyword in noise_keywords):
        return False
        
    words = text.split()
    numbers = sum(1 for word in words if any(char.isdigit() for char in word))
    if numbers / len(words) > 0.2: # Stricter number ratio (max 20% numbers)
        return False
        
    return True

def generate_fres_report():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    FY25_JSON = os.path.join(BASE_DIR, "data", "03_processed", "Mahindra_FY25_dual_stream.json")
    DB_PATH = os.path.join(BASE_DIR, "data", "04_vector_db")
    NVDM_WEIGHTS = os.path.join(BASE_DIR, "src", "models", "saved_weights", "nvdm_weights.pt")
    VECTORIZER = os.path.join(BASE_DIR, "src", "models", "saved_weights", "nvdm_vectorizer.pkl")
    FINBERT_DIR = os.path.join(BASE_DIR, "src", "models", "finbert_tncrm_weights")
    BIGRU_WEIGHTS = os.path.join(BASE_DIR, "src", "models", "saved_weights", "bigru_weights.pt")

    print("\n" + "="*60)
    print("FINRISK-EVO: AUTOMATED RISK ESCALATION PIPELINE")
    print("="*60)

    print("\nBooting up Neural Engines...")
    vector_db = FinancialVectorStore(db_path=DB_PATH)
    pipeline = MasterRiskPipeline(NVDM_WEIGHTS, VECTORIZER, FINBERT_DIR, BIGRU_WEIGHTS)
    tncrm_filter = TNCRMDataBuilder()

    print(f"\nLoading FY25 Data from {os.path.basename(FY25_JSON)}...")
    with open(FY25_JSON, 'r', encoding='utf-8') as f:
        fy25_data = json.load(f)

    # 3. Filter for High-Value Sentences
    test_candidates = []
    
    # NEW LOGIC: Skip the first 400 paragraphs (Administrative section)
    paragraphs_to_scan = fy25_data["dual_stream_paragraphs"][400:]
    
    for item in paragraphs_to_scan:
        text = item["stream_2_finbert"]
        if tncrm_filter.is_tncrm_candidate(text) and len(text.split()) > 20 and clean_corporate_noise(text):
            test_candidates.append(text)
            if len(test_candidates) == 10:
                break

    print(f"\nProcessing {len(test_candidates)} high-value FY25 paragraphs for temporal drift...")
    print("-" * 60)

    results = []
    
    for i, fy25_text in enumerate(test_candidates):
        print(f"Analyzing Paragraph {i+1}...")
        
        fy24_docs, metadata = vector_db.search_baseline_context(query_text=fy25_text, target_year="FY24", n_results=1)
        fy24_baseline = fy24_docs[0] 
        
        fres_score, attn_25, conf, disp, sim, category = pipeline.evaluate_temporal_risk(fy24_baseline, fy25_text)
        
        results.append({
            "fy24": fy24_baseline,
            "fy25": fy25_text,
            "fres": fres_score,
            "attn_25": attn_25,
            "confidence": conf,
            "category": category
        })

    print("\n" + "="*60)
    print("FINAL RISK ESCALATION REPORT (FRES)")
    print("="*60)
    
    results.sort(key=lambda x: x["fres"], reverse=True)
    
    for rank, res in enumerate(results):
        print(f"\n[ RANK {rank+1} | FRES: {res['fres']:.2f}/100 | {res['category']} ]")
        print(f"FY24 Baseline : {res['fy24']}")
        print(f"FY25 Update   : {res['fy25']}")
        print(f"Inference     : {res['attn_25']:.1f}% BiGRU attention | {res['confidence']:.1f}% confidence")
        print("-" * 60)

if __name__ == "__main__":
    generate_fres_report()