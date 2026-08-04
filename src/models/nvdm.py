import torch
import torch.nn as nn
import torch.nn.functional as F

class NVDM(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 500, n_topics: int = 50):
        super(NVDM, self).__init__()
        self.vocab_size = vocab_size
        self.n_topics = n_topics
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, n_topics)
        self.fc_logvar = nn.Linear(hidden_dim, n_topics)
        self.decoder = nn.Linear(n_topics, vocab_size)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            return mu + torch.randn_like(std) * std
        return mu

    def forward(self, bow_vector):
        hidden = self.encoder(bow_vector)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)
        logits = self.decoder(z)
        return F.log_softmax(logits, dim=-1), mu, logvar

    def get_topic_dispersion(self, mu, logvar):
        """Calculates KL Divergence for the Z-Axis Entropy."""
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        return kl.mean().item()

def nvdm_loss_function(log_probs, target, mu, logvar):
    """
    Computes the Variational ELBO loss: Reconstruction loss + KL Divergence.
    - target is the Bag-of-Words document counts (batch_size, vocab_size).
    - log_probs is the log-softmax outputs from the decoder (batch_size, vocab_size).
    - mu and logvar are the variational distribution parameters.
    """
    # Reconstruction loss (Multinomial log-likelihood approximation)
    reconstruction_loss = -torch.sum(target * log_probs, dim=-1)
    
    # KL Divergence from the normal prior
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    
    # Return average loss over the batch
    return (reconstruction_loss + kl_loss).mean()