import os
import json
import re
from pathlib import Path

class DualStreamProcessor:
    def process_for_nvdm(self, text):
        return re.sub(r'[^a-z\s]', ' ', text.lower()).strip()

    def process_for_finbert(self, text):
        return re.sub(r'\s+', ' ', text).strip()

    def run_pipeline(self, input_json_path, output_dir):
        with open(input_json_path, 'r', encoding='utf-8') as f: data = json.load(f)
        processed = {"document_name": data["document_name"], "dual_stream_paragraphs": []}
        
        for node in data.get("nodes", []):
            raw = node["text"]
            processed["dual_stream_paragraphs"].append({
                "stream_1_nvdm": self.process_for_nvdm(raw),
                "stream_2_finbert": self.process_for_finbert(raw)
            })
            
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_file = os.path.join(output_dir, f"{data['document_name']}_dual_stream.json")
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(processed, f, indent=2)
        return out_file