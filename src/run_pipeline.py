import os
import sys

# Reconfigure stdout to use UTF-8 to prevent emoji encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path: sys.path.append(PROJECT_ROOT)

from src.data_prep.pdf_extractor import extract_financial_narrative
from src.data_prep.text_pipeline import DualStreamProcessor
from src.rag_engine.vector_store import FinancialVectorStore
from src.compute_master_scores import compute_all_scores

def execute_full_pipeline(base_pdf, tgt_pdf):
    PARSED_DIR = os.path.join(PROJECT_ROOT, "data", "02_parsed")
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "03_processed")
    DB_PATH = os.path.join(PROJECT_ROOT, "data", "04_vector_db")
    CSV_OUT = os.path.join(PROJECT_ROOT, "data", "master_fres_scores.csv")
    
    # 1. Parse
    b_parse = extract_financial_narrative(base_pdf, PARSED_DIR)
    t_parse = extract_financial_narrative(tgt_pdf, PARSED_DIR)
    
    # 2. Process
    proc = DualStreamProcessor()
    proc.run_pipeline(b_parse, PROCESSED_DIR)
    proc.run_pipeline(t_parse, PROCESSED_DIR)
    
    b_json = os.path.join(PROCESSED_DIR, f"{Path(base_pdf).stem}_dual_stream.json")
    t_json = os.path.join(PROCESSED_DIR, f"{Path(tgt_pdf).stem}_dual_stream.json")
    
    # 3. Vector DB
    v_db = FinancialVectorStore(db_path=DB_PATH)
    try: v_db.client.delete_collection("financial_reports")
    except: pass
    v_db = FinancialVectorStore(db_path=DB_PATH)
    v_db.ingest_parsed_json(b_json, year_tag="Baseline")
    
    # 4. Crunch
    compute_all_scores(t_json, CSV_OUT)

if __name__ == "__main__":
    import time
    start_time = time.time()
    print("==================================================")
    print("    FINRISK-EVO: RUNNING END-TO-END PIPELINE     ")
    print("==================================================")
    
    base_pdf = os.path.join(PROJECT_ROOT, "data", "01_raw", "Mahindra_FY24.pdf")
    tgt_pdf = os.path.join(PROJECT_ROOT, "data", "01_raw", "Mahindra_FY25.pdf")
    
    if os.path.exists(base_pdf) and os.path.exists(tgt_pdf):
        execute_full_pipeline(base_pdf, tgt_pdf)
        print(f"\n✅ Pipeline completed successfully in {time.time() - start_time:.2f} seconds.")
    else:
        print(f"Error: Mahindra PDFs not found. Please verify paths:\n- {base_pdf}\n- {tgt_pdf}")