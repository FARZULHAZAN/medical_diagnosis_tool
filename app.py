import os
import streamlit as st
import torch
import warnings

from src.state.Agentstate import AgentState
from src.graph.graph_builder import setup_graph  # Importing setup_graph
from types import SimpleNamespace

# Suppress warnings
warnings.filterwarnings("ignore")
torch.set_default_device('cpu')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

st.set_page_config(
    page_title="Medical Diagnosis Assistant",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to detect medicine-related queries
def is_medicine_request(text: str) -> bool:
    keywords = ['medicine', 'medication', 'drug', 'treatment', 'pill']
    return any(k in text.lower() for k in keywords)


# 👇 Initialize LangGraph app only once
if 'diagnosis_graph' not in st.session_state:
    with st.spinner("Initializing LangGraph agent..."):
        # Dummy wrapper class to hold the graph object with setup_graph
        class GraphWrapper:
            def __init__(self):
                self.app = None
                self.setup_graph()

            def setup_graph(self):
                # Call your setup_graph method from graph_builder.py
                setup_graph(self)

        st.session_state.diagnosis_graph = GraphWrapper().app


# 👇 Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("🩺 Medical Diagnosis Assistant")

# Display past chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Describe your symptoms (e.g., 'I have chest pain and cough')")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your symptoms..."):
            medicine_req = is_medicine_request(user_input)

            # Create initial AgentState dictionary
            init_state: AgentState = {
                "user_query": user_input,
                "extracted_symptoms": [],
                "similarity_score": 0.0,
                "retrieved_disease": {},
                "refined_query": "",
                "retry_count": 0,
                "final_response": "",
                "medicine_request": medicine_req,
                "medicines": [],
                "conversation_history": [],
                "messages": []
            }

            # 🔗 Run LangGraph
            result: AgentState = st.session_state.diagnosis_graph.invoke(init_state)

            # 💬 Show final response
            st.markdown(result["final_response"])

            with st.expander("🔍 Processing Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("🧠 Extracted Symptoms")
                    for s in result["extracted_symptoms"]:
                        st.write(f"- {s}")
                with col2:
                    st.subheader("📊 Internal Info")
                    st.write(f"**Similarity Score:** {result['similarity_score']:.2f}")
                    st.write(f"**Retry Count:** {result['retry_count']}")
                    st.write(f"**Disease Found:** {result['retrieved_disease'].get('name', 'N/A')}")

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result["final_response"]
            })

    # Optional medicine follow-up
    if not medicine_req:
        if st.button("💊 Show Medicines"):
            with st.chat_message("assistant"):
                with st.spinner("Fetching medicine info..."):
                    init_state["medicine_request"] = True
                    med_result = st.session_state.diagnosis_graph.invoke(init_state)

                    if med_result["medicines"]:
                        med_response = "Here are some commonly recommended medicines:\n\n"
                        med_response += "\n".join(f"- {med}" for med in med_result["medicines"])
                    else:
                        med_response = "Sorry, couldn't find medicine info."

                    st.markdown(med_response)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": med_response
                    })

if st.button("🧹 Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
