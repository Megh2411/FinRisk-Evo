import pytest
import torch

from src.models.bigru_attn import TemporalRiskBiGRU


def test_uncertainty_sampling_restores_eval_mode():
    model = TemporalRiskBiGRU(input_dim=4, hidden_dim=3, num_layers=2, dropout=0.2)
    model.eval()
    x = torch.randn(1, 2, 4)

    mean_score, confidence, attention = model.predict_with_uncertainty(x, n_samples=3)

    assert model.training is False
    assert 0.0 <= mean_score <= 1.0
    assert 0.0 <= confidence <= 100.0
    assert attention.shape == (1, 2, 1)


def test_uncertainty_requires_at_least_two_samples():
    model = TemporalRiskBiGRU(input_dim=4, hidden_dim=3, num_layers=2, dropout=0.2)
    with pytest.raises(ValueError, match="at least 2"):
        model.predict_with_uncertainty(torch.randn(1, 2, 4), n_samples=1)
