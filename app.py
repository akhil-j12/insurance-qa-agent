
import os
import fitz  # PyMuPDF
import tempfile
import requests
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Read Together.ai API key from secrets
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"

# Function: Extract text from uploaded PDF
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    return "\n".join([page.get_text() for page in doc])

# Function: Chunk text for embedding
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    return splitter.split_text(text)

# Function: Build prompt for LLM
def build_prompt(query, docs):
    context = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
    return f"""You are a helpful insurance assistant. Use the context below to answer the user's question.

Context:
{context}

Question: {query}

Answer:"""

# Function: Call Together.ai API
def query_together_ai(prompt):
    url = "https://api.together.xyz/v1/completions"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    data = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": 300,
        "temperature": 0.3
    }
    response = requests.post(url, json=data, headers=headers)
    response.raise_for_status()
    return response.json()["choices"][0]["text"].strip()

# Streamlit UI
st.title("📄 Insurance Contract QA Agent")
st.write("Upload an insurance PDF and ask questions about the policy.")

uploaded_file = st.file_uploader("Upload a policy document (PDF)", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("File uploaded. Extracting text and creating index...")

    raw_text = extract_text_from_pdf(tmp_path)
    chunks = chunk_text(raw_text)

    docs_for_faiss = [Document(page_content=chunk, metadata={"source": uploaded_file.name}) for chunk in chunks]
    vectorstore = FAISS.from_documents(docs_for_faiss, embedding_model)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    query = st.text_input("Ask a question about the policy:")

    if query:
        docs = retriever.get_relevant_documents(query)
        prompt = build_prompt(query, docs)
        with st.spinner("Thinking..."):
            answer = query_together_ai(prompt)
        st.markdown("### 🧠 Answer")
        st.write(answer)

        st.markdown("### 📄 Source Chunks")
        for i, doc in enumerate(docs):
            st.markdown(f"**[{i+1}] From {doc.metadata['source']}:**\n{doc.page_content[:400]}...")

