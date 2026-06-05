import os
import streamlit as st
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import chromadb
import PyPDF2
from dotenv import load_dotenv

load_dotenv(r"C:\Users\anana\Downloads\Upgrad-AI-Course\.env")

# CONFIGURATION - CHAT
MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("OPENAI_ENDPOINT")

# Client
chat_client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

# Initialize persistent ChromaDB client
@st.cache_resource
def get_chroma_client():
    # Save the database in a folder named "chroma_db" in the current directory
    return chromadb.PersistentClient(path="./chroma_db")

chroma_client = get_chroma_client()
collection = chroma_client.get_or_create_collection(name="ChromaDB-Collection")

# We let ChromaDB handle the embeddings using its default built-in model
# (SentenceTransformers 'all-MiniLM-L6-v2') which runs entirely locally.

st.title("Simple RAG with OpenAI & ChromaDB")
st.write("Strictly no LangChain used!")

# Option to clear the database
if st.button("Clear Vector Database"):
    try:
        chroma_client.delete_collection("ChromaDB-Collection")
        collection = chroma_client.get_or_create_collection(name="ChromaDB-Collection")
        st.success("Database cleared!")
    except Exception:
        pass

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    # Check if we already processed this file in this session
    if "processed_file" not in st.session_state or st.session_state.processed_file != uploaded_file.name:

        # 1. Read and extract text
        pdf = PyPDF2.PdfReader(uploaded_file)
        text = "".join(page.extract_text() for page in pdf.pages)
        
        # 2. Chunk text (simple word-based chunking)
        words = text.split()
        chunk_size = 200
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        
        # 3. Store in Vector DB
        with st.spinner("Embedding and storing in vector DB..."):
            for i, chunk in enumerate(chunks):
                # We use the filename in the ID. If the exact same file is uploaded, Chroma will ignore/overwrite it
                collection.add(
                    ids=[f"{uploaded_file.name}_chunk_{i}"],
                    documents=[chunk]
                )
        
        st.session_state.processed_file = uploaded_file.name
        st.success("Document processed and stored persistently!")
    else:
        st.success("Document already processed and ready for querying!")

# 4. Query and Retrieve
query = st.text_input("Ask a question about the document:")
if query:
    with st.spinner("Searching and generating response..."):
        # Retrieve relevant chunks
        results = collection.query(
            query_texts=[query],
            n_results=3
        )
        context = "\n---\n".join(results["documents"][0])
        
        # Generate answer using LLM
        prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer based only on context."
        response = chat_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}]
        )
        
        st.write("### Answer")
        st.write(response.choices[0].message.content)
