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
        You are a Chief Risk Officer at a top-tier quantitative FinTech firm. 
        Your proprietary deep learning architecture has flagged a risk escalation in a company's Annual Report.
        
        The Financial Risk Escalation Score (FRES) is: {fres_score:.2f}/100.
        
        Here is the text from the previous year (Baseline):
        "{fy24_text}"
        
        Here is the newly extracted text from the current year (Escalation):
        "{fy25_text}"
        
        Your task:
        Write a highly concise, 2-to-3 sentence executive summary explaining exactly what the risk is and why it escalated from last year to this year. 
        Do not use filler words. Be objective, quantitative, and direct. 
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

# --- Quick Sanity Check ---
if __name__ == "__main__":
    # REPLACE WITH YOUR NEW API KEY
    API_KEY = "[REDACTED_REVOKED_API_KEY]" 
    
    if API_KEY != "YOUR_NEW_API_KEY_HERE":
        print("Booting Generative Risk Explainer...")
        explainer = RiskExplainerLLM(api_key=API_KEY)
        
        dummy_fy24 = "Supply chain operations remained stable with a 2% variance in logistics costs."
        dummy_fy25 = "Supply chain operations completely collapsed, causing a 40% surge in logistics costs and halting production."
        
        print("\nSending context to LLM...")
        summary = explainer.generate_executive_summary(dummy_fy24, dummy_fy25, 99.99)
        
        print("\n--- AI EXECUTIVE SUMMARY ---")
        print(summary)
        print("----------------------------")
    else:
        print("Please paste your API key into the script to test the LLM.")