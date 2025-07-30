import os
import requests
from src.state.Agentstate import AgentState

GEMINI_URL = ""
GEMINI_MODEL = ""

def call_gemini(prompt: str) -> str:
    api_key = os.getenv("")
    
    resp = requests.post(
        GEMINI_URL,
        
        json={"model": GEMINI_MODEL, "messages": [{"role": "user", "content": prompt}]}
    )
    resp.raise_for_status()
    return resp.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

def refine_query_node(state: AgentState) -> AgentState:
    """
    Regenerate a semantically similar question based on the user's original query
    and extracted symptoms—without altering meaning.
    """
    original = state.get("user_query", "")
    symptoms = state.get("extracted_symptoms", [])
    symptom_str = ", ".join(symptoms)

    prompt = (
    "You are an expert at query rewriting. Your task is to paraphrase the user's original text to create a fresh, natural-sounding alternative. "
    "Follow these rules strictly:\n\n"
    "1.  **Preserve Exact Intent:** The core meaning, intent, and all specific entities (like names, places, numbers, or technical terms) in the original query MUST be perfectly maintained. Do not add, omit, or change any factual information.\n"
    "2.  **Match the Format:** If the original is a question, the paraphrase must be a question. If it is a command or statement, the paraphrase must be a command or statement.\n"
    "3.  **Vary the Structure:** Use different vocabulary and sentence structure. Avoid simply swapping a few synonyms.\n"
    "4.  **Handle Ambiguity:** If the original query is ambiguous (e.g., uses subjective words like 'best' or 'interesting'), the paraphrase must retain that same ambiguity. Do not try to make it more specific.\n"
    "5.  **Output Cleanly:** Your response must ONLY be the paraphrased text. Do not include any preamble, explanation, or quotation marks.\n\n"
    f"Original Text: \"{original}\"\n\n"
    "Paraphrased Text:"
)

    try:
        refined = call_gemini(prompt)
        if not refined:
            raise ValueError("Empty response from Gemini")
    except Exception:
        # Fallback: simple templated paraphrase
        refined = f"Given symptoms ({symptom_str}), what condition matches \"{original}\"?"

    state["refined_query"] = refined
    state["user_query"] = refined
    state["retry_count"] = state.get("retry_count", 0) + 1
    return state
