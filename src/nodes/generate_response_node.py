from src.state.Agentstate import AgentState


def generate_response_node(state: AgentState) -> AgentState:
        """Generate user-friendly response"""
        disease = state["retrieved_disease"]
        symptoms = state["extracted_symptoms"]
        
        response = f"""
Based on your symptoms ({', '.join(symptoms)}), here's what I found:

**Possible Condition:** {disease['name']}

**Description:** {disease['description']}

**Severity:** {disease['severity']}

**General Treatment Approach:** {disease['treatment']}

**Important Note:** This is for informational purposes only and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.

Would you like me to search for common medications used for this condition?
"""
        
        state["final_response"] = response
        return state