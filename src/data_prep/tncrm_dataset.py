import os
import json
import re
import nltk
from pathlib import Path
from nltk.tokenize import sent_tokenize

# Download sentence tokenizer
nltk.download('punkt', quiet=True)

class TNCRMDataBuilder:
    def __init__(self):
        # The specific magnitude and trend words derived from financial research
        self.magnitude_words = {
            "increase", "increased", "increasing", "surge", "surged",
            "decrease", "decreased", "decreasing", "plunge", "plunged",
            "drop", "dropped", "dip", "dipped", "rise", "rose", "rising",
            "fall", "fell", "falling", "grow", "grew", "growth", "decline", 
            "declined", "skyrocket", "plummet", "plummeted", "jump", "jumped"
        }
        
        # Regex to detect any numerical presence (e.g., 40%, 1.5L, 100, 50bps)
        self.num_regex = re.compile(r'\d+')

    def is_tncrm_candidate(self, sentence: str) -> bool:
        """
        Determines if a sentence contains numerical values OR magnitude words.
        If it does, it is a valid candidate for Targeted Masking.
        """
        sentence_lower = sentence.lower()
        words = set(re.findall(r'\b\w+\b', sentence_lower))
        
        # Check for magnitude words
        has_magnitude = bool(self.magnitude_words.intersection(words))
        # Check for numbers
        has_number = bool(self.num_regex.search(sentence_lower))
        
        return has_magnitude or has_number

    def build_dataset(self, input_json_path: str, output_dir: str):
        print("Extracting TNCRM valid sentences from FinBERT stream...")
        
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        tncrm_sentences = []
        total_sentences = 0

        for item in data["dual_stream_paragraphs"]:
            # We strictly use Stream 2 (Sequence Preserved) for Transformer models
            paragraph = item["stream_2_finbert"]
            
            # Break paragraph into individual sentences
            sentences = sent_tokenize(paragraph)
            total_sentences += len(sentences)
            
            for sent in sentences:
                if len(sent.split()) > 5 and self.is_tncrm_candidate(sent):
                    tncrm_sentences.append(sent)

        # Save as a plain text file (one sentence per line) which is the standard 
        # format expected by HuggingFace for custom MLM training.
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        out_file = os.path.join(output_dir, f"{data['document_name']}_tncrm_corpus.txt")
        
        with open(out_file, 'w', encoding='utf-8') as f:
            for sent in tncrm_sentences:
                f.write(sent + '\n')

        print(f"Total Sentences Scanned: {total_sentences}")
        print(f"High-Value TNCRM Sentences Extracted: {len(tncrm_sentences)}")
        print(f"Saved custom training corpus to: {out_file}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "03_processed")
    
    input_file = os.path.join(PROCESSED_DIR, "Mahindra_FY24_dual_stream.json")
    
    if os.path.exists(input_file):
        builder = TNCRMDataBuilder()
        builder.build_dataset(input_file, PROCESSED_DIR)
    else:
        print(f"Cannot find processed data at: {input_file}")