from google import genai
from src.state.Agentstate import AgentState
from dotenv import load_dotenv
load_dotenv()


class SymptomExtractorGemini:
    def __init__(self):
        
        self.client = genai.Client(api_key=None)
        
    def extract_symptoms_node(self, state: AgentState) -> dict:
        query = state["user_query"].strip()
        prompt = (
            "You are a medical assistant.\n"
            "Extract *only* symptoms mentioned by the user as a comma-separated list.\n\n"
            f"User: \"{query}\"\n"
            "Symptoms:"
        )

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text.strip()

        parts = text.split("Symptoms:")
        symptom_part = parts[-1].strip().rstrip(".")
        extracted = [
            s.strip().lower()
            for s in symptom_part.replace(" and ", ", ").split(",")
            if s.strip()
        ]

        if not extracted:
            extracted = [query.lower()]

        state["extracted_symptoms"] = extracted
        return state


