import os

from google import genai


class RiskExplainerLLM:
    def __init__(self, api_key: str | None = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Set GEMINI_API_KEY in your environment to use "
                "the optional Gemini risk explainer."
            )

        self.client = genai.Client(api_key=key)
        self.model_name = "gemini-2.5-flash"

    def generate_executive_summary(
        self,
        fy24_text: str,
        fy25_text: str,
        fres_score: float,
    ) -> str:
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
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            return (response.text or "").strip()
        except Exception as exc:
            return f"Summary generation failed: {type(exc).__name__}"


if __name__ == "__main__":
    explainer = RiskExplainerLLM()
    print("Gemini client initialized successfully.")
