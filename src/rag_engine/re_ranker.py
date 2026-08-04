from sentence_transformers import CrossEncoder

class RiskReRanker:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        # This model is specifically trained to calculate how well two sentences answer each other
        print(f"Loading Neural Re-Ranker: {model_name}...")
        self.model = CrossEncoder(model_name)

    def pick_best_match(self, query_fy25, candidates_fy24):
        """
        Takes the query and multiple results from ChromaDB.
        Returns the single best matching text based on deep semantic interaction.
        """
        # Create pairs: [[query, doc1], [query, doc2]...]
        pairs = [[query_fy25, doc] for doc in candidates_fy24]
        
        # Cross-encoders look at both sentences at once (much more accurate than vectors)
        scores = self.model.predict(pairs)
        
        # Find the index of the highest score
        best_idx = scores.argmax()
        return candidates_fy24[best_idx], scores[best_idx]

if __name__ == "__main__":
    # Test logic
    ranker = RiskReRanker()
    fy25 = "Mahindra ZEO is our new electric 4W vehicle with 21kWh battery."
    top_5_fuzzy_matches = [
        "Mahindra ESOP trust holds 1.04% shares.", # Bad match
        "Our electric vehicle journey started in 2020.", # Decent match
        "The company is focused on battery assembly lines.", # Good match
        "Tractor sales grew by 5%.", # Irrelevant
        "We are launching new passenger vehicles next year." # Generic
    ]
    best, score = ranker.pick_best_match(fy25, top_5_fuzzy_matches)
    print(f"\nBest Match Picked: {best}\nScore: {score}")