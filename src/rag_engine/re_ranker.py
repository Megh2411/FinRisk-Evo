import re
from sentence_transformers import CrossEncoder


class RiskReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        print(f"Loading Neural Re-Ranker: {model_name}...")
        self.model = CrossEncoder(model_name)

    @staticmethod
    def is_usable_candidate(text):
        if not isinstance(text, str):
            return False

        text = text.strip()
        words = re.findall(r"\b[\w'-]+\b", text)
        alpha_words = [w for w in words if re.search(r"[A-Za-z]", w)]

        if len(words) < 8 or len(alpha_words) < 6:
            return False

        lower = text.lower()
        boilerplate = (
            "table of contents",
            "contents",
            "dear sirs",
            "dear sir/madam",
            "notice is hereby given",
        )
        if any(lower.startswith(p) for p in boilerplate):
            return False

        # Filter highly numeric fragments, but retain short financial disclosures.
        digit_count = sum(ch.isdigit() for ch in text)
        alpha_count = sum(ch.isalpha() for ch in text)
        if alpha_count and digit_count / alpha_count > 1.5:
            return False

        return True

    def rank_candidates(self, query_text, candidates):
        usable = []
        for i, candidate in enumerate(candidates):
            if self.is_usable_candidate(candidate):
                usable.append((i, candidate))

        if not usable:
            return []

        pairs = [[query_text, candidate] for _, candidate in usable]
        scores = self.model.predict(pairs)

        ranked = [
            {
                "candidate_index": original_index,
                "text": candidate,
                "reranker_score": float(score),
            }
            for (original_index, candidate), score in zip(usable, scores)
        ]
        return sorted(
            ranked,
            key=lambda item: item["reranker_score"],
            reverse=True,
        )

    def pick_best_match(self, query_text, candidates):
        ranked = self.rank_candidates(query_text, candidates)
        if not ranked:
            return None, None
        return ranked[0]["text"], ranked[0]["reranker_score"]


if __name__ == "__main__":
    examples = [
        "The company reported revenue of Rs. 1,000 crore during FY2025.",
        "Our emissions declined after improvements to the manufacturing process.",
        "Dear Sirs, please find attached the annual report.",
        "Table of Contents Financial Highlights Risk Factors",
    ]
    for example in examples:
        print(RiskReRanker.is_usable_candidate(example), example)
