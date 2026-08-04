import os
import sys
import json
import pickle
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer

# Ensure Python can find the src module when running from the terminal
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models.nvdm import NVDM, nvdm_loss_function

def train_topic_model(json_path: str, output_model_dir: str):
    import numpy as np
    import random
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print("Seed initialized to 42 for reproducible training run.")
    print("Loading dual-stream data...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 1. Extract only Stream 1 (NVDM) text
    corpus = [item["stream_1_nvdm"] for item in data["dual_stream_paragraphs"]]

    # 2. Vectorize the text (Bag of Words)
    # We restrict the vocabulary to the top 2000 most frequent financial words to avoid sparse noise
    print("Vectorizing corpus...")
    VOCAB_SIZE = 2000
    vectorizer = CountVectorizer(max_features=VOCAB_SIZE, min_df=2)
    bow_matrix = vectorizer.fit_transform(corpus).toarray()
    
    # Save the vectorizer so we can use the exact same vocabulary for the FY25 report later
    Path(output_model_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(output_model_dir, 'nvdm_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)

    # 3. Prepare PyTorch DataLoaders
    bow_tensor = torch.FloatTensor(bow_matrix)
    dataset = TensorDataset(bow_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 4. Initialize the NVDM Model
    # We will use 20 latent topics. For a single annual report, 20 topics provides dense, readable clusters.
    N_TOPICS = 20
    model = NVDM(vocab_size=VOCAB_SIZE, hidden_dim=500, n_topics=N_TOPICS)
    
    # Use Adam optimizer (standard for VAEs)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5. The Training Loop
    EPOCHS = 50
    print(f"\nBeginning Training for {EPOCHS} Epochs...")
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0]
            
            optimizer.zero_grad()
            log_probs, mu, logvar = model(x)
            
            # Calculate ELBO Loss
            loss = nvdm_loss_function(log_probs, x, mu, logvar)
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Average ELBO Loss: {avg_loss:.4f}")

    # 6. Save the trained model weights
    torch.save(model.state_dict(), os.path.join(output_model_dir, 'nvdm_weights.pt'))
    print("\nModel weights and vectorizer saved successfully.")

    # 7. Extract and Print the Discovered Topics
    print("\n--- EXTRACTED LATENT FINANCIAL TOPICS ---")
    vocab = vectorizer.get_feature_names_out()
    # The decoder weights map the latent topics back to the vocabulary
    decoder_weights = model.decoder.weight.data.cpu().numpy() 
    
    for k in range(N_TOPICS):
        # Sort weights in descending order to find the most heavily associated words for topic k
        top_word_indices = decoder_weights[:, k].argsort()[::-1][:8]
        top_words = [vocab[i] for i in top_word_indices]
        print(f"Topic {k+1}: {', '.join(top_words)}")

if __name__ == "__main__":
    from pathlib import Path
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    PROCESSED_DIR = os.path.join(BASE_DIR, "data", "03_processed")
    MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "src", "models", "saved_weights")
    
    # Target the processed dual-stream JSON
    input_file = os.path.join(PROCESSED_DIR, "Mahindra_FY24_dual_stream.json")
    
    if os.path.exists(input_file):
        train_topic_model(input_file, MODEL_OUTPUT_DIR)
    else:
        print(f"Cannot find processed data at: {input_file}")