
from langgraph.graph import StateGraph, END, START
from src.state.Agentstate import AgentState
from src.nodes.extract_symptoms_node import SymptomExtractorGemini
from src.nodes import refine_query_node
from src.nodes.vector_search_node import DiseaseRAG
from src.nodes.web_search_node import MedicalWebSearchAgent
from src.nodes import generate_response_node 
class decision:
    def decide_next_step(self, state: AgentState) -> str:
        """Decide next step based on similarity score"""
        score = state["similarity_score"]
        retry_count = state["retry_count"]
        
        if score >= 0.7:
            return "generate_response"
        elif retry_count < 3:
            return "refine_query"
        else:
            return "web_search"
    
    def decide_after_web_search(self, state: AgentState) -> str:
        """Decide next step after web search"""
        score = state["similarity_score"]
        retry_count = state["retry_count"]
        
        if score >= 0.7:
            return "generate_response"
        elif retry_count < 3:
            return "refine_query"
        else:
            return "generate_response"
    
    def check_medicine_request(self, state: AgentState) -> str:
        """Check if user requested medicine information"""
        if state.get("medicine_request", False):
            return "search_medicines"
        return "end"

def setup_graph(self):
        """Create LangGraph workflow"""
        workflow = StateGraph(AgentState)
        obj=decision()
        obj1=SymptomExtractorGemini()
        dis=DiseaseRAG()
        # Add nodes
        workflow.add_node("extract_symptoms", obj1.extract_symptoms_node)
        workflow.add_node("vector_search", dis.vector_search_node)
        workflow.add_node("refine_query", refine_query_node.refine_query_node)
        workflow.add_node("web_search", MedicalWebSearchAgent.search_disease)
        workflow.add_node("generate_response", generate_response_node.generate_response_node)
        workflow.add_node("search_medicines", MedicalWebSearchAgent.search_medicines)
        
        # Set entry point
        workflow.add_edge(START, "extract_symptoms")
        
        # Add edges
        workflow.add_edge("extract_symptoms", "vector_search")
        workflow.add_conditional_edges(
            "vector_search",
            obj.decide_next_step,
            {
                "refine_query": "refine_query",
                "web_search": "web_search", 
                "generate_response": "generate_response"
            }
        )
        workflow.add_edge("refine_query", "vector_search")
        workflow.add_conditional_edges(
            "web_search",
            obj.decide_after_web_search,
            {
                "refine_query": "refine_query",
                "generate_response": "generate_response"
            }
        )
        workflow.add_conditional_edges(
            "generate_response",
            obj.check_medicine_request,
            {
                "search_medicines": "search_medicines",
                "end": END
            }
        )
        workflow.add_edge("search_medicines", END)
        
        self.app = workflow.compile()