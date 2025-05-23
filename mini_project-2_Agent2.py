import os
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document

# --- Set Environment Key ---
os.environ["GROQ_API_KEY"] = ""

# --- Init Model ---
llm = ChatGroq(model="llama3-8b-8192", temperature=0.5)

# --- Streamlit Page Config ---
st.set_page_config(page_title="Scientific Research Assistant")
st.title("Scientific Research Assistant")

# --- File Upload & Load ---
def upload_and_load_file():
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        with open("research_temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = UnstructuredLoader("research_temp.pdf")
        docs = loader.load()
        st.success("PDF loaded successfully!")

        st.markdown("### Extracted Text Preview")
        st.write(docs[0].page_content[:1500] + "...")
        return docs
    return None

# --- Split Text ---
def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

# --- Create Chroma Vectorstore ---
def create_vectorstore(chunks):
    documents = [
        Document(page_content=chunk[0], metadata=chunk[1]) if isinstance(chunk, tuple) else chunk
        for chunk in chunks
    ]
    filtered_docs = filter_complex_metadata(documents)
    for doc in filtered_docs:
        doc.metadata = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                        for k, v in doc.metadata.items()}
        for key in ["points", "layout_width", "layout_height"]:
            doc.metadata.pop(key, None)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return Chroma.from_documents(filtered_docs, embeddings, persist_directory="./chroma_research")

# --- Refine Query ---
def rewrite_query(original_query: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        "You are a research assistant. Rephrase the vague scientific query below into a precise, searchable research question:\n\n"
        "Original: {query}\nRefined:")
    return (prompt | llm).invoke({"query": original_query}).content.strip()

# --- Generate Hypothetical Abstract ---
def generate_hypothetical_answer(query: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        "You are a research assistant. Generate a hypothetical abstract for the following emerging or underexplored topic:\n\n"
        "Query: {query}\n\nHypothetical Abstract:")
    return (prompt | llm).invoke({"query": query}).content.strip()

# --- Answer with RAG ---
def answer_query(query: str, db):
    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return None, []
    context = "\n\n".join([doc.page_content for doc in docs[:3]])
    prompt = ChatPromptTemplate.from_template(
        "As a research assistant, provide a clear, sourced response using the context below:\n\nContext:\n{context}\n\nQuery:\n{query}\n\nAnswer:")
    result = (prompt | llm).invoke({"context": context, "query": query})
    return result.content.strip(), docs[:3]

# --- Main Flow ---
documents = upload_and_load_file()

if documents:
    chunks = split_text(documents)
    st.info(f"Document split into {len(chunks)} chunks.")
    db = create_vectorstore(chunks)
    st.success("Embeddings created and stored.")

    query = st.text_input("Enter your research question:")
    if query:
        refined = rewrite_query(query)
        st.write("**Refined Query:**", refined)

        hypo = generate_hypothetical_answer(refined)
        st.markdown("###Hypothetical Abstract")
        st.write(hypo)

        answer, top_docs = answer_query(refined, db)
        if answer:
            st.markdown("### Answer with Sources")
            st.write(answer)

            st.markdown("### Top Sources")
            for doc in top_docs:
                st.markdown(f"- **Source:** `{doc.metadata.get('source', 'Unknown')}`")
                st.markdown(f"> {doc.page_content[:300]}...")
        else:
            st.warning("No relevant sources found. Consider using the hypothetical abstract above.")

        st.markdown("---")
        st.markdown("Please review findings before drawing conclusions (human-in-the-loop).")
