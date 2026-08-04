import os
import json
import re
import fitz
from pathlib import Path

def extract_financial_narrative(pdf_path: str, output_dir: str) -> str:
    print(f"Extracting blocks for: {os.path.basename(pdf_path)}")
    doc = fitz.open(pdf_path)
    paragraphs = []
    
    # We maintain statistics for verification logging
    total_blocks_scanned = 0
    margin_blocks_filtered = 0
    short_blocks_filtered = 0
    valid_paragraphs_saved = 0
    
    for page in doc:
        page_height = page.rect.height
        blocks = page.get_text("blocks")
        for b in blocks:
            total_blocks_scanned += 1
            # Check block_type is text (0)
            if b[6] == 0: 
                x0, y0, x1, y1, text, block_no, block_type = b
                
                # Scientific Layout Constraint: Filter out page margins (headers in top 8%, footers in bottom 8%)
                if y0 < page_height * 0.08 or y1 > page_height * 0.92:
                    margin_blocks_filtered += 1
                    continue
                
                clean_text = text.strip().replace('\n', ' ')
                
                # Filter out pure page numbers or common header metadata chunks
                if re.match(r'^\d+$', clean_text) or re.match(r'^page \d+( of \d+)?$', clean_text.lower()):
                    margin_blocks_filtered += 1
                    continue
                
                words = clean_text.split()
                # Exclude brief fragments (headers, section indices, single words)
                if len(words) > 12: 
                    paragraphs.append({"text": clean_text})
                    valid_paragraphs_saved += 1
                else:
                    short_blocks_filtered += 1

    print(f"Extraction stats | Scanned: {total_blocks_scanned} | Margin filtered: {margin_blocks_filtered} | Short filtered: {short_blocks_filtered} | Saved: {valid_paragraphs_saved}")

    file_name = Path(pdf_path).stem
    payload = {"document_name": file_name, "nodes": paragraphs}
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(output_dir, f"{file_name}_parsed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path