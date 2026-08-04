import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1, bias=False)

    def forward(self, gru_outputs):
        scores = self.attention(gru_outputs)
        attention_weights = F.softmax(scores, dim=1)
        context_vector = torch.sum(gru_outputs * attention_weights, dim=1)
        return context_vector, attention_weights

class TemporalRiskBiGRU(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.3):
        super(TemporalRiskBiGRU, self).__init__()
        self.bigru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, 
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = AttentionLayer(hidden_dim)
        self.fc_risk = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def forward(self, x):
        gru_out, _ = self.bigru(x)
        context, attn_weights = self.attention(gru_out)
        return self.fc_risk(context), attn_weights

    def predict_with_uncertainty(self, x, n_samples=10):
        """Monte Carlo Dropout for Bayesian Confidence."""
        self.train() # Keep dropout ON
        scores, attns = [], []
        for _ in range(n_samples):
            with torch.no_grad():
                logit, attn = self.forward(x)
                scores.append(torch.sigmoid(logit))
                attns.append(attn)
        scores = torch.stack(scores)
        mean_score = scores.mean().item()
        confidence = (1.0 - scores.std().item()) * 100
        mean_attn = torch.stack(attns).mean(dim=0)
        return mean_score, confidence, mean_attn