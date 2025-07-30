import os
import pandas as pd
import numpy as np
from typing import List, Tuple
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from src.state.Agentstate import AgentState

class DiseaseRAG:
    def __init__(
        self,
        csv_path: str = "Dataset_cleaned.csv",
        vector_db_path: str = "disease_db",
        groq_model: str = "Gemma2-9b-It",
    ):
        self.csv_path = csv_path
        self.vector_db_path = vector_db_path

        # 1️⃣ Initialize & store embeddings (same model for retrieval + scoring)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": "cpu", "trust_remote_code": True},
        )

        self._build_or_load_db()
        self._initialize_qa(groq_model)

    def _build_or_load_db(self):
        df = pd.read_csv(self.csv_path, dtype=str)
        texts = [f"{r['disease']}: {r['symptoms']}" for _, r in df.iterrows()]
        metadata = df.to_dict(orient="records")

        if not os.path.exists(self.vector_db_path):
            self.db = FAISS.from_texts(texts, self.embeddings, metadatas=metadata)
            self.db.save_local(self.vector_db_path)
        else:
            self.db = FAISS.load_local(
                self.vector_db_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    def _initialize_qa(self, model_name: str):
        llm = ChatGroq(
            model=model_name,
            groq_api_key=os.getenv("GROQ_API_KEY", ""),
            temperature=0.2
        )
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a precise medical information extractor. Follow these rules strictly:\n"
    "1. You MUST answer the question using ONLY the information provided in the CONTEXT. Do not use any external knowledge.\n"
    "2. Your task is to identify and list ALL disease names from the CONTEXT that directly answer the QUESTION.\n"
    "3. If you find one or more disease names, your answer MUST consist ONLY of the disease names, separated by a comma and a space. Do not add any other words or explanations (e.g., 'The diseases are...', 'Based on the context...').\n"
    "4. If the answer to the QUESTION cannot be found in the CONTEXT, you MUST reply with the exact phrase 'I don't know'. Your entire response must be 'I don't know' and nothing else.\n\n"
    "CONTEXT:\n"
    "{context}\n\n"
    "QUESTION:\n"
    "{question}\n\n"
    "Answer:"
            )
        )
        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    def vector_search_node(self, state: AgentState) -> AgentState:
        symptoms: List[str] = state.get("extracted_symptoms", [])
        query_text = ", ".join(symptoms)

        docs_and_scores = self.db.similarity_search_with_score(query_text, k=5)
        if not docs_and_scores:
            state["similarity_score"] = 0.0
            state["retrieved_disease"] = {
                "name": "Unknown",
                "symptoms": symptoms,
                "predicted_disease": "I don't know",
                "matched_symptoms": [],
                "description": "No match found",
                "severity": "Unknown",
                "treatment": "Consult healthcare provider"
            }
            return state

        top_doc, _ = docs_and_scores[0]
        disease_meta = top_doc.metadata
        disease_text = ", ".join(disease_meta.get("symptoms", []))

        predicted = self.qa({"query": "Which diseases matches these symptoms: " + query_text})["result"].strip()

        emb_q = np.array(self.embeddings.embed_query("Which diseases matches these symptoms: " + query_text))
        emb_d = np.array(self.embeddings.embed_query(f"This are the symptoms {disease_text} for the disease {predicted}"))
        sim_score = self._cosine_sim(emb_q, emb_d)
        if predicted.lower() == "i don't know":
            sim_score = 0.0

        matched = list(set(symptoms) & set(disease_meta.get("symptoms", [])))

        state["similarity_score"] = sim_score
        state["retrieved_disease"] = {
            **disease_meta,
            "predicted_disease": predicted,
            "matched_symptoms": matched
        }
        return state
