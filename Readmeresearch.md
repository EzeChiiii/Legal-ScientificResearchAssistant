
# 🧪 Scientific Research Assistant

An AI-powered assistant designed to help researchers explore underdeveloped scientific questions by rewriting vague queries, generating hypothetical abstracts, and retrieving evidence-based answers from user-uploaded PDFs using advanced RAG techniques.

## 🚀 Features

- 🔁 SMART query rewriting using LLaMA 3
- 📄 Hypothetical abstract generation for vague or emerging topics
- 📚 Retrieval-Augmented Generation (RAG) with context-aware answers
- 🧠 HuggingFace embeddings + Chroma vector store for semantic search
- 📥 PDF ingestion with unstructured loader
- 🖥️ Streamlit-based interactive interface

## 📸 Demo Preview

<img width="1440" alt="Screenshot 2025-05-23 at 7 18 36 PM" src="https://github.com/user-attachments/assets/f14cadd3-9557-4e41-b6a8-6c49ef6e5364" />


## 🛠️ Tech Stack

- **AI Models**: LLaMA 3 (via Groq API), HuggingFaceEmbeddings
- **Retrieval**: Chroma Vector Store
- **Text Processing**: langchain_text_splitters, langchain_unstructured
- **Frontend**: Streamlit
- **Language**: Python

## 📂 Project Structure

```
scientific-research-assistant/
│
├── mini_project-2_Agent2.py                     # Streamlit app and AI pipeline
├── chroma_research/            # Persisted vector store
├── requirements.txt
└── README.md
```

## ⚙️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/scientific-research-assistant.git
cd scientific-research-assistant
```

2. Set up your virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set environment variable:

```bash
export GROQ_API_KEY=your_groq_api_key
```

5. Launch the Streamlit app:

```bash
streamlit run main.py
```

## ✅ Strengths

- Refines ambiguous research questions into structured, searchable topics
- Generates speculative research abstracts to guide exploration
- Combines semantic search with AI-generated summaries
- Encourages human-in-the-loop validation for academic integrity

## ⚠️ Limitations

- Effectiveness depends on content and quality of uploaded PDFs
- Hypothetical outputs should be reviewed by human experts
- No built-in citation formatting or journal export support

## 📄 License

MIT License — free to use, modify, and distribute.

## 👤 Author



