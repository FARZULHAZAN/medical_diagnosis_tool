# 🏥 AI Medical Diagnosis Assistant

An AI-powered medical diagnosis tool built using **Gemini models**, **semantic search**, and **Streamlit**. This assistant takes symptoms as input, extracts them using LLMs , searches a vector database of diseases, and provides potential conditions, treatments, and optional medicine information.

> ⚠️ **Disclaimer**: This tool is for informational and educational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 🚀 Features

- 🔍 Symptom extraction using **Gemini Flash** models (LLM-based)
- 🧠 Semantic search over a **vector database** of diseases
- 📊 Similarity threshold and **retry mechanism** for refining search
- 🌐 Optional **web search** fallback for external info
- 💊 Medicine information retrieval (if requested)
- 💬 Conversational chat interface using **Streamlit**

---

## 🛠️ Project Structure

```
├── app.py                     # Streamlit frontend
├── src/
│   ├── graph/                 # LangGraph workflow setup
│   ├── nodes/                 # All node logic (symptom extractor, generator, etc.)
│   ├── tools/                 # Optional tools (e.g., web search)
│   ├── vector/                # VectorDB loading and search logic
│   ├── state/Agentstate.py    # AgentState TypedDict (shared state)
├── data/                      # Disease dataset and embeddings
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/medical-diagnosis-assistant.git
cd medical-diagnosis-assistant
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate       # On Linux/Mac
venv\Scripts\activate          # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup environment
Create a `.env` file in the root and add your Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

---

## 🧪 Run the Application

```bash
streamlit run app.py
```

Then go to [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💡 Example Usage

Just type something like:

> _"I have fever, stomach pain and headache"_

And the app will:

1. Extract symptoms from your query.
2. Search for the closest disease using embeddings.
3. Show possible diagnosis, treatment, and severity.
4. Optionally suggest medicines on button click.

---

## 🧠 Technologies Used

- [Gemini Flash 2.5](https://deepmind.google/technologies/gemini/)
- [LangGraph](https://docs.langgraph.dev)
- [ChromaDB / FAISS](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io)
- Python 3.10+

---

## ✅ TODO / Roadmap

- [ ] Add feedback loop for user confirmations
- [ ] Integrate local symptom-to-medicine mapping DB
- [ ] Enable doctor/hospital suggestions based on user location
- [ ] Add multilingual support (e.g., Hindi, Malayalam)
- [ ] Deploy via Docker / Hugging Face Spaces

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

```bash
# Format code using Black
black .
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🧑‍💻 Author

Built by **Farzul Hazan** using LLMs and agentic workflows.  
Feel free to connect or contribute!
