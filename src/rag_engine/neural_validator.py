from transformers import pipeline

class NeuralMarketValidator:
    def __init__(self):
        print("Loading Zero-Shot Entailment Model (BART-MNLI)...")
        self.classifier = pipeline("zero-shot-classification", 
                                   model="facebook/bart-large-mnli", 
                                   device=0) # use 0 for GPU

    def validate_risk(self, internal_risk_text, external_news_text):
        """
        Calculates the probability that the news 'Entails' (proves) the risk.
        """
        # We ask: Does this news text support the premise of 'financial risk'?
        result = self.classifier(external_news_text, 
                                 candidate_labels=["validates risk", "contradicts risk", "neutral"],
                                 hypothesis_template="This news {} based on the company report.")
        
        score = result['scores'][0] if result['labels'][0] == "validates risk" else 0.0
        return score # 0.0 to 1.0 probability of validation