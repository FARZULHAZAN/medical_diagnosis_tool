from typing import List, Dict, Any,TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    user_query: str
    extracted_symptoms: List[str]
    similarity_score: float
    retrieved_disease: Dict[str, Any]
    refined_query: str
    retry_count: int
    final_response: str
    medicine_request: bool
    medicines: List[str]
    conversation_history: List[Dict[str, str]]
    messages: Annotated[List[Dict[str, str]], add_messages]