"""Pure scoring utilities shared by inference and regression tests.

FRES is a heuristic display index, not a calibrated probability or a validated
financial-risk measure. Constants intentionally match the current implementation.
"""

SIMILARITY_GATE = 0.93
PHI_MAX = 0.229
SHIFT_OFFSET = 1.1


def normalize_fres(raw_score: float, phi_max: float = PHI_MAX) -> float:
    """Map a non-negative raw FRES value to the documented 0--100 display scale."""
    if phi_max <= 0:
        raise ValueError("phi_max must be positive")
    if raw_score < 0:
        raise ValueError("raw_score must be non-negative")
    return min(100.0 * raw_score / phi_max, 100.0)


def calculate_fres(
    semantic_similarity: float,
    mean_risk_probability: float,
    severity_projection: float,
    confidence_weight: float,
    *,
    similarity_gate: float = SIMILARITY_GATE,
    phi_max: float = PHI_MAX,
) -> float:
    """Calculate FRES using the same fixed normalization for every paragraph pair.

    Scores are pair-local: the result must not be rescaled using the other rows
    in a batch, because that makes a paragraph's score depend on batch contents.
    """
    if semantic_similarity > similarity_gate:
        return 0.0
    raw = (
        mean_risk_probability
        * (SHIFT_OFFSET - semantic_similarity)
        * severity_projection
        * confidence_weight
    )
    return normalize_fres(max(0.0, raw), phi_max=phi_max)
