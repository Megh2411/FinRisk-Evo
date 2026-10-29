import os
import sys
import json
import pandas as pd
import warnings

warnings.filterwarnings("ignore")
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)

from src.models.risk_drift import MasterRiskPipeline
from src.rag_engine.vector_store import FinancialVectorStore
from src.rag_engine.re_ranker import RiskReRanker

def compute_all_scores(target_json_path, output_csv_path):
    DB_PATH = os.path.join(PROJECT_ROOT, "data", "04_vector_db")
    N_W = os.path.join(PROJECT_ROOT, "src", "models", "saved_weights", "nvdm_weights.pt")
    V_P = os.path.join(PROJECT_ROOT, "src", "models", "saved_weights", "nvdm_vectorizer.pkl")
    F_D = os.path.join(PROJECT_ROOT, "src", "models", "finbert_tncrm_weights")
    G_W = os.path.join(PROJECT_ROOT, "src", "models", "saved_weights", "bigru_weights.pt")
    
    pipe = MasterRiskPipeline(N_W, V_P, F_D, G_W)
    vdb = FinancialVectorStore(DB_PATH)
    ranker = RiskReRanker()

    with open(target_json_path, 'r', encoding='utf-8') as f: data = json.load(f)
    nodes = [item["stream_2_finbert"] for item in data["dual_stream_paragraphs"]]
    
    results = []
    # Taking top 200 nodes for speed during execution, can be customized
    print(f"Beginning temporal scoring for up to 200 nodes...")
    success_count = 0
    error_count = 0
    
    for i, t_text in enumerate(nodes[:200]):
        try:
            cands, _ = vdb.search_baseline_context(t_text, "Baseline", 5)
            b_text, _ = ranker.pick_best_match(t_text, cands)
            fres, attn, conf, disp, sim, category = pipe.evaluate_temporal_risk(b_text, t_text)
            
            results.append({
                "fy24": b_text, 
                "fy25": t_text, 
                "fres": fres, 
                "attn_25": attn, 
                "confidence": conf, 
                "dispersion": disp,
                "similarity": sim,
                "category": category
            })
            success_count += 1
        except Exception as e:
            error_count += 1
            continue

    print(f"Scoring completed. Success: {success_count} | Errors: {error_count}")
    
    if not results:
        print("Warning: No matching results generated.")
        df = pd.DataFrame(columns=["fy24", "fy25", "fres", "attn_25", "confidence", "dispersion", "similarity", "category"])
    else:
        # Keep pair-local FRES values from MasterRiskPipeline. Batch-wise min-max
        # rescaling makes a pair's score depend on unrelated rows and breaks the
        # documented fixed-normalization formula. Preserve zero-score gated pairs.
        df = pd.DataFrame(results)
        
    df.sort_values("fres", ascending=False).to_csv(output_csv_path, index=False)
    print(f"Saved master scores to: {output_csv_path}")