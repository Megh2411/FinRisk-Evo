import os
from google import genai

class RiskExplainerLLM:
    def __init__(self, api_key: str):
        # Initialize the modern Google GenAI Client
        self.client = genai.Client(api_key=api_key)
        
        # Use the state-of-the-art 2.5 Flash model
        self.model_name = 'gemini-2.5-flash'

    def generate_executive_summary(self, fy24_text: str, fy25_text: str, fres_score: float) -> str:
        """
        Prompts the LLM to act as a quantitative risk analyst and explain the temporal drift.
        """
        prompt = f"""
        You are assisting an analyst comparing two corporate annual-report passages.
        Treat FRES as a heuristic display score, not a probability or proof of risk.
        Do not assume escalation: determine whether the text supports escalation,
        mitigation, no material change, or an uncertain/unrelated comparison.
        State only changes supported by the two passages; do not invent figures.

        FRES heuristic display score: {fres_score:.2f}/100.

        Previous-year baseline passage:
        "{fy24_text}"

        Target-year passage:
        "{fy25_text}"

        Write a concise 2-3 sentence explanation. If the passages do not support a
        directional conclusion, say so explicitly.
        """
        
        try:
            # The new generate_content syntax
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating summary: {e}"

# --- Optional local smoke test ---
if __name__ == "__main__":
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY in your environment to run this test.")
    explainer = RiskExplainerLLM(api_key=api_key)
    baseline = "Supply-chain costs remained stable, with a 2% variance."
    target = "Supply-chain costs increased by 8% during the reporting period."
    print(explainer.generate_executive_summary(baseline, target, 42.0))
