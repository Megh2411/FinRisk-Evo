import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pickle
from transformers import AutoTokenizer, AutoModel

# Ensure Python can find the src module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.nvdm import NVDM
from src.data_prep.text_pipeline import DualStreamProcessor
from src.models.bigru_attn import TemporalRiskBiGRU

def generate_synthetic_training_data(pipeline_processor, nvdm, finbert, tokenizer, vectorizer, device):
    """
    Generates synthetic temporal pairs with Gaussian noise to prevent instant overfitting.
    Label 0 = Status Quo (Low Drift)
    Label 1 = Escalating Risk (High Drift)
    """
    print("Generating Contrastive Training Pairs with Data Augmentation...")
    
    base_pairs = [
        # --- ESCALATING RISKS (Label 1.0) ---
        ("Supply chain was stable.", "Supply chain collapsed causing a 40% surge in costs.", 1.0),
        ("Revenues grew by 5%.", "Revenues plummeted by 30% due to market conditions.", 1.0),
        ("Cybersecurity protocols are functional.", "A massive data breach exposed 2 million records.", 1.0),
        ("Operations are normal.", "Operations halted due to severe labor strikes.", 1.0),
        ("Demand is steady.", "Demand dropped off a cliff, falling 50%.", 1.0),
        
        # --- STATUS QUO / IMPROVEMENT (Label 0.0) ---
        ("Logistics costs remained flat.", "Logistics costs saw a minor 1% variance.", 0.0),
        ("Revenue dipped slightly by 2%.", "Revenue recovered and grew by 3%.", 0.0),
        ("Regulatory compliance is ongoing.", "We remain fully compliant with all new regulations.", 0.0),
        ("Debt levels are stable.", "Debt was reduced by 5% this quarter.", 0.0),
        ("Market share is maintained.", "Market share increased marginally by 0.5%.", 0.0)
    ]

    sequence_tensors = []
    labels = []

    nvdm.eval()
    finbert.eval()

    with torch.no_grad():
        # We loop 40 times to create 400 total training samples
        for _ in range(40): 
            for text_t1, text_t2, label in base_pairs:
                # --- Get T1 Vector ---
                s1_t1 = pipeline_processor.process_for_nvdm(text_t1)
                bow_t1 = torch.FloatTensor(vectorizer.transform([s1_t1]).toarray()).to(device)
                _, top_t1, _ = nvdm(bow_t1)
                
                s2_t1 = pipeline_processor.process_for_finbert(text_t1)
                inp_t1 = tokenizer(s2_t1, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
                inp_t1 = {k: v.to(device) for k, v in inp_t1.items()}
                sem_t1 = finbert(**inp_t1).last_hidden_state[:, 0, :]
                
                vec_t1 = torch.cat((top_t1, sem_t1), dim=1)

                # --- Get T2 Vector ---
                s1_t2 = pipeline_processor.process_for_nvdm(text_t2)
                bow_t2 = torch.FloatTensor(vectorizer.transform([s1_t2]).toarray()).to(device)
                _, top_t2, _ = nvdm(bow_t2)
                
                s2_t2 = pipeline_processor.process_for_finbert(text_t2)
                inp_t2 = tokenizer(s2_t2, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
                inp_t2 = {k: v.to(device) for k, v in inp_t2.items()}
                sem_t2 = finbert(**inp_t2).last_hidden_state[:, 0, :]
                
                vec_t2 = torch.cat((top_t2, sem_t2), dim=1)

                # --- DATA AUGMENTATION ---
                # Inject a small amount of random Gaussian noise (mean 0, std 0.05) to the vectors.
                # This simulates the natural phrasing differences in real annual reports 
                # and prevents the BiGRU from memorizing the exact base pairs.
                noise_t1 = torch.randn_like(vec_t1) * 0.05
                noise_t2 = torch.randn_like(vec_t2) * 0.05
                
                vec_t1_noisy = vec_t1 + noise_t1
                vec_t2_noisy = vec_t2 + noise_t2

                # Stack into time sequence: [1, 2, 788]
                seq = torch.stack([vec_t1_noisy, vec_t2_noisy], dim=1).squeeze(0)
                sequence_tensors.append(seq)
                labels.append(label)

    X = torch.stack(sequence_tensors)
    y = torch.FloatTensor(labels).unsqueeze(1)
    
    return TensorDataset(X, y)

def train_bigru_engine(nvdm_path, vectorizer_path, finbert_path, output_dir):
    import numpy as np
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print("Seed initialized to 42 for reproducible BiGRU training run.")
    print("Initializing BiGRU Training Sequence...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = DualStreamProcessor()
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
        
    nvdm = NVDM(vocab_size=len(vectorizer.get_feature_names_out()), n_topics=20).to(device)
    nvdm.load_state_dict(torch.load(nvdm_path, map_location=device))
    
    tokenizer = AutoTokenizer.from_pretrained(finbert_path)
    finbert = AutoModel.from_pretrained(finbert_path).to(device)

    dataset = generate_synthetic_training_data(processor, nvdm, finbert, tokenizer, vectorizer, device)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    bigru = TemporalRiskBiGRU(input_dim=788, hidden_dim=256).to(device)
    bigru.train()

    criterion = nn.BCEWithLogitsLoss()
    # Lowered learning rate to ensure smooth convergence
    optimizer = optim.Adam(bigru.parameters(), lr=1e-4) 

    EPOCHS = 30
    print("\nTraining BiGRU Evolution Engine on CPU...")
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            risk_logits, _ = bigru(batch_x)
            loss = criterion(risk_logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | BiGRU Loss: {total_loss/len(dataloader):.4f}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "bigru_weights.pt")
    torch.save(bigru.state_dict(), out_path)
    print(f"\nTraining Complete. BiGRU weights saved to {out_path}")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    NVDM_WEIGHTS = os.path.join(BASE_DIR, "src", "models", "saved_weights", "nvdm_weights.pt")
    VECTORIZER = os.path.join(BASE_DIR, "src", "models", "saved_weights", "nvdm_vectorizer.pkl")
    FINBERT_DIR = os.path.join(BASE_DIR, "src", "models", "finbert_tncrm_weights")
    BIGRU_OUT_DIR = os.path.join(BASE_DIR, "src", "models", "saved_weights")
    
    train_bigru_engine(NVDM_WEIGHTS, VECTORIZER, FINBERT_DIR, BIGRU_OUT_DIR)