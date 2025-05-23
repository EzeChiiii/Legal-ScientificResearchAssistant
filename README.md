
# 🧠 Legal Assistant Agent

An intelligent AI-powered legal assistant that helps users analyze legal PDFs by answering submitted questions using advanced Retrieval-Augmented Generation (RAG) techniques. The system rewrites vague legal queries, generates hypothetical answers (HyDE), and retrieves the most relevant context from the uploaded document using vector similarity.

## 🚀 Features

- 🔄 Query rewriting with LLaMA 3 to clarify vague legal questions
- ✨ HyDE (Hypothetical Document Embeddings) for speculative reasoning
- 📄 PDF ingestion using `langchain_unstructured`
- 🧠 Embeddings via HuggingFace (`all-mpnet-base-v2`)
- 🔍 Semantic search using Chroma vector database
- ⚙️ Interactive UI with Streamlit

## 📸 Demo Preview

*(You can include a GIF or screenshot here of your Streamlit app in action)*

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI (optional for expansion)
- **Frontend**: Streamlit
- **AI & RAG**: LangChain, LLaMA via Groq API
- **Embeddings**: HuggingFaceEmbeddings
- **Vector Store**: ChromaDB
- **PDF Parsing**: langchain_unstructured

## 📂 Project Structure

```
legal-assistant-agent/
│
├── main.py                     # Core Streamlit app with RAG pipeline
├── chroma_db/                  # Persisted Chroma vector database
├── requirements.txt
└── README.md
```

## ⚙️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/legal-assistant-agent.git
cd legal-assistant-agent
```

2. Set up your virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set your Groq API Key:

```bash
export GROQ_API_KEY=your_api_key  # or use .env file
```

5. Launch the app:

```bash
streamlit run main.py
```

## ✅ Strengths

- Handles vague input using intelligent rewriting
- Uses speculative answers to improve retrieval accuracy
- Modular, scalable, and easy to extend
- Fast and user-friendly with Streamlit UI

## ⚠️ Limitations

- Does not replace professional legal advice
- Limited by the content of the uploaded document
- No jurisdiction-specific filtering yet
- Requires human review for final legal interpretation

## 📄 License

MIT License — free to use and modify.

## 👤 Author

**Clinton Iwu**  
[LinkedIn]([https://www.linkedin.com/in/YOURNAME](https://www.linkedin.com/in/clintoniwu/))  
