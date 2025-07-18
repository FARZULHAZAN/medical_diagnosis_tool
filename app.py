 
import os
import json
import asyncio
import streamlit as st
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
import faiss
import spacy
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.schema import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
import requests
from datetime import datetime
import re
import pickle
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Set PyTorch to use CPU and avoid meta tensor issues
torch.set_default_device('cpu')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Configure page
st.set_page_config(
    page_title="Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main Streamlit application"""
    st.title("🏥 Medical Diagnosis Assistant")
    st.markdown("*AI-powered symptom analysis and medical information tool*")
    
    # Sidebar
    st.sidebar.header("About")
    st.sidebar.markdown("""
    This AI assistant helps analyze symptoms and provides medical information.
    
    **Features:**
    - Symptom extraction using NLP
    - Vector database similarity search
    - Query refinement with retry mechanism
    - Web search integration
    - Medicine information lookup
    
    **Disclaimer:** This tool is for informational purposes only and should not replace professional medical advice.
    """)
    
    # Initialize system
    if 'diagnosis_system' not in st.session_state:
        with st.spinner("Initializing Medical Diagnosis System..."):
            st.session_state.diagnosis_system = MedicalDiagnosisSystem()
    
    # Chat interface
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    user_input = st.chat_input("Describe your symptoms (e.g., 'I have headache, stomach pain, and fever')")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Process the query
        with st.chat_message("assistant"):
            with st.spinner("Analyzing symptoms..."):
                # Check if user is asking for medicines
                request_medicines = any(word in user_input.lower() for word in ['medicine', 'medication', 'drug', 'treatment', 'pill'])
                
                result = st.session_state.diagnosis_system.process_query(user_input, request_medicines)
                
                # Display results
                st.markdown(result["final_response"])
                
                # Show processing details in expander
                with st.expander("Processing Details"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Extracted Symptoms")
                        for symptom in result["extracted_symptoms"]:
                            st.write(f"• {symptom}")
                    
                    with col2:
                        st.subheader("Analysis Details")
                        st.write(f"**Similarity Score:** {result['similarity_score']:.2f}")
                        st.write(f"**Retry Count:** {result['retry_count']}")
                        st.write(f"**Retrieved Disease:** {result['retrieved_disease'].get('name', 'N/A')}")
                
                # Add assistant response to chat
                st.session_state.chat_history.append({"role": "assistant", "content": result["final_response"]})
        
        # Medicine request button
        if not any(word in user_input.lower() for word in ['medicine', 'medication', 'drug', 'treatment', 'pill']):
            if st.button("🔍 Get Medicine Information"):
                with st.spinner("Searching for medicine information..."):
                    medicine_result = st.session_state.diagnosis_system.process_query(user_input, True)
                    
                    with st.chat_message("assistant"):
                        if medicine_result["medicines"]:
                            st.markdown("Here's medicine information for your condition:")
                            for medicine in medicine_result["medicines"]:
                                st.write(f"• {medicine}")
                        else:
                            st.markdown("Please consult a healthcare provider for medication recommendations.")
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()