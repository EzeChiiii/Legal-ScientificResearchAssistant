import os

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_core.utils import get_from_env
from unstructured.partition.pdf import partition_pdf

 # Forces early initialization




import streamlit as st



os.environ["GROQ_API_KEY"] = ""


# Initialize the Groq LLM
llm = ChatGroq(model="llama3-8b-8192", temperature=0.5)

st.set_page_config(page_title="Legal Assistant Agent")
st.title("Legal Assistant Agent")


def upload_and_load_file():
    uploaded_file = st.file_uploader("Upload a legal PDF", type=["pdf"])
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = UnstructuredLoader("temp.pdf")
        docs = loader.load()
        st.success("PDF loaded successfully!")

        # Display extracted text
        st.markdown("### Extracted Text Preview")
        st.write(docs[0].page_content[:1500] + "...")  # Preview first 1500 characters

        return docs
    return None



def split_text(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]  # Removed regex pattern
    )
    return splitter.split_documents(docs)

def create_vectorstore(chunks):
    # Convert all items to Document instances
    documents = [
        Document(page_content=chunk[0], metadata=chunk[1])
        if isinstance(chunk, tuple) else chunk
        for chunk in chunks
    ]

    # Step 2: Filter complex metadata types
    filtered_docs = filter_complex_metadata(documents)

    # Step 3: Remove problematic metadata keys
    for doc in filtered_docs:
        doc.metadata = {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
                        for k, v in doc.metadata.items()}
        # Remove layout-related metadata entirely
        doc.metadata.pop("points", None)
        doc.metadata.pop("layout_width", None)
        doc.metadata.pop("layout_height", None)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return Chroma.from_documents(filtered_docs, embeddings, persist_directory="./chroma_db")

def rewrite_query(original_query: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        "You are a legal assistant. Rephrase the following vague or unclear legal question "
        "to be more specific and suitable for retrieving legal information:\n\n"
        "Original question: {query}\nRewritten question:"
    )
    chain = prompt | llm
    result = chain.invoke({"query": original_query})
    return result.content.strip()

    # Generate hypothetical answer (HyDE)
def generate_hypothetical_answer(query: str) -> str:
        prompt = ChatPromptTemplate.from_template(
            "You are a legal assistant. Given the question below, generate a hypothetical but reasonable answer "
            "as if you had access to the document:\n\n"
            "Question: {query}\n\nHypothetical Answer:"
        )
        chain = prompt | llm
        result = chain.invoke({"query": query})
        return result.content.strip()

        # RAG retrieval + answer
def answer_query(query: str, db):
    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs[:3]])  # top 3 results

    qa_prompt = ChatPromptTemplate.from_template(
        "You are a helpful legal assistant. Based on the following context, answer the question.\n\n"
        "Context:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    )
    chain = qa_prompt | llm
    result = chain.invoke({"context": context, "query": query})
    return result.content.strip()

documents = upload_and_load_file()

if documents:
    chunks = split_text(documents)
    st.info(f"Document split into {len(chunks)} chunks.")
    db = create_vectorstore(chunks)
    st.success("Embeddings created and stored in ChromaDB.")

    query = st.text_input("Ask your legal question:")
    if query:
        rewritten = rewrite_query(query)
        st.write("**Rewritten Query:**", rewritten)


        hypothetical = generate_hypothetical_answer(rewritten)
        st.markdown("### ✨ Hypothetical Answer (HyDE)")
        st.write(hypothetical)



        answer = answer_query(rewritten, db)
        st.markdown("### 🤖 LLM Answer")
        st.write(answer)


