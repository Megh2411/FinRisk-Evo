import pytest

from src.models.fres_scoring import calculate_fres, normalize_fres


def test_similarity_gate_returns_zero():
    assert calculate_fres(0.931, 0.9, 0.8, 0.95) == 0.0


def test_score_matches_documented_formula():
    similarity = 0.80
    probability = 0.70
    projection = 0.60
    confidence = 0.90
    raw = probability * (1.1 - similarity) * projection * confidence
    expected = min(100.0 * raw / 0.229, 100.0)
    assert calculate_fres(similarity, probability, projection, confidence) == pytest.approx(expected)


def test_score_is_capped_at_100():
    assert calculate_fres(0.0, 1.0, 1.0, 1.0) == 100.0


def test_normalization_rejects_invalid_scale():
    with pytest.raises(ValueError):
        normalize_fres(0.1, phi_max=0.0)


def test_normalization_rejects_negative_raw_score():
    with pytest.raises(ValueError):
        normalize_fres(-0.1)


def test_pair_score_does_not_depend_on_batch_minimum_or_maximum():
    pair = (0.82, 0.75, 0.62, 0.91)
    score_alone = calculate_fres(*pair)
    # Deliberately calculate unrelated scores; the pair's result remains fixed.
    _ = [calculate_fres(0.20, 0.1, 0.1, 0.5), calculate_fres(0.50, 0.9, 0.9, 0.9)]
    assert calculate_fres(*pair) == score_alone
