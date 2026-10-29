import os
import sys
import torch
import pickle
from torch.nn.functional import cosine_similarity
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.nvdm import NVDM
from src.data_prep.text_pipeline import DualStreamProcessor
from src.models.bigru_attn import TemporalRiskBiGRU
from src.models.taxonomy import classify_risk_type
from src.models.fres_scoring import SIMILARITY_GATE, calculate_fres

class MasterRiskPipeline:
    def __init__(self, nvdm_weights, vectorizer_path, finbert_weights, bigru_weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = DualStreamProcessor()
        
        with open(vectorizer_path, 'rb') as f: self.vectorizer = pickle.load(f)
        self.vocab_size = len(self.vectorizer.get_feature_names_out())
        
        self.nvdm = NVDM(vocab_size=self.vocab_size, n_topics=20).to(self.device)
        self.nvdm.load_state_dict(torch.load(nvdm_weights, map_location=self.device))
        self.nvdm.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(finbert_weights)
        self.finbert = AutoModel.from_pretrained(finbert_weights).to(self.device)
        self.finbert.eval()

        self.bigru = TemporalRiskBiGRU(input_dim=788, hidden_dim=256).to(self.device)
        self.bigru.load_state_dict(torch.load(bigru_weights, map_location=self.device))
        self.bigru.eval()
        
        # Precompute Semantic Projection Anchor Vectors for Sentiment Polarity Fusion
        esc_anchors = [
            "financial risk escalated", 
            "severe losses intensified", 
            "supply chain disruption", 
            "debt capacity constraints", 
            "regulatory noncompliance penalty"
        ]
        mit_anchors = [
            "financial performance improved", 
            "losses declined", 
            "successful product launch", 
            "stable compliance operations", 
            "strong strategic recovery"
        ]
        
        with torch.no_grad():
            esc_vecs = []
            for text in esc_anchors:
                inp = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
                esc_vecs.append(self.finbert(**inp).last_hidden_state[:, 0, :])
            self.esc_anchor_vector = torch.cat(esc_vecs, dim=0).mean(dim=0, keepdim=True)
            
            mit_vecs = []
            for text in mit_anchors:
                inp = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
                mit_vecs.append(self.finbert(**inp).last_hidden_state[:, 0, :])
            self.mit_anchor_vector = torch.cat(mit_vecs, dim=0).mean(dim=0, keepdim=True)

    def get_fused_vector(self, raw_text):
        s1 = self.processor.process_for_nvdm(raw_text)
        bow = torch.FloatTensor(self.vectorizer.transform([s1]).toarray()).to(self.device)
        with torch.no_grad(): _, top_vec, logvar = self.nvdm(bow)

        s2 = self.processor.process_for_finbert(raw_text)
        inp = self.tokenizer(s2, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad(): sem_vec = self.finbert(**inp).last_hidden_state[:, 0, :]
        
        # NVDM.forward returns its latent mean as the second value; use it directly.
        return torch.cat((top_vec, sem_vec), dim=1), top_vec, logvar, sem_vec

    def evaluate_temporal_risk(self, fy24_text, fy25_text):
        v24, _, _, sem_24 = self.get_fused_vector(fy24_text)
        v25, mu_25, logvar_25, sem_25 = self.get_fused_vector(fy25_text)
        
        sem_sim = cosine_similarity(sem_24, sem_25).item()
        dispersion = self.nvdm.get_topic_dispersion(mu_25, logvar_25)
        
        seq = torch.stack([v24, v25], dim=1)
        mean_score, conf, attn = self.bigru.predict_with_uncertainty(seq)
        
        # Calculate continuous semantic projection score along the Risk axis
        sim_esc = cosine_similarity(sem_25, self.esc_anchor_vector).item()
        sim_mit = cosine_similarity(sem_25, self.mit_anchor_vector).item()
        
        # Severity projection is normalized to [0, 1]
        severity_projection = (sim_esc - sim_mit) / 2.0 + 0.5
        severity_projection = max(0.0, min(1.0, severity_projection))
        
        # Bayesian Confidence scaling (penalize high-uncertainty predictions)
        conf_weight = conf / 100.0
        
        # FRES Score Calculation: Gated by semantic similarity threshold,
        # scaled by semantic similarity gap, semantic severity projection, and Bayesian confidence
        if sem_sim > SIMILARITY_GATE:
            fres = 0.0
            category = "GATED / STATUS QUO"
        else:
            fres = calculate_fres(sem_sim, mean_score, severity_projection, conf_weight)
            # The index-to-category mapping in taxonomy.py is manually specified,
            # and the input is the NVDM latent mean (not a topic probability vector).
            category = classify_risk_type(fres, fy25_text, mu_25)
            
        return fres, attn[0, 1, 0].item() * 100, conf, dispersion, sem_sim, category