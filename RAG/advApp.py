import os
import hashlib
import logging
from datetime import datetime

import streamlit as st
import chromadb
import PyPDF2

from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# CONFIG
# ============================================================

load_dotenv(r"C:\Users\anana\Downloads\Upgrad-AI-Course\.env")

MODEL_NAME = os.getenv("MODEL_NAME")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("OPENAI_ENDPOINT")

PDF_FOLDER = "../Policies"
CHROMA_DB_PATH = "./chroma_db"
LOG_FILE = "../app.log"

os.makedirs(PDF_FOLDER, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)


def log_step(message):
    logging.info(message)


# ============================================================
# OPENAI CLIENT
# ============================================================

chat_client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)


# ============================================================
# CHROMA
# ============================================================

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)


chroma_client = get_chroma_client()

collection = chroma_client.get_or_create_collection(
    name="documents"
)

processed_collection = chroma_client.get_or_create_collection(
    name="processed_files"
)


# ============================================================
# HELPERS
# ============================================================

def get_file_hash(filepath):
    """
    Hash file content to detect changes.
    """

    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def extract_pdf_text(filepath):

    text_parts = []

    pdf = PyPDF2.PdfReader(filepath)

    for page in pdf.pages:

        try:
            extracted = page.extract_text()

            if extracted:
                text_parts.append(extracted)

        except Exception as e:
            log_step(f"Page extraction failed: {e}")

    return "\n".join(text_parts)


def chunk_text(text, chunk_size=300):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks


# ============================================================
# PDF INGESTION
# ============================================================

def ingest_new_pdfs():

    pdf_files = [
        f for f in os.listdir(PDF_FOLDER)
        if f.lower().endswith(".pdf")
    ]

    for pdf_file in pdf_files:

        full_path = os.path.join(PDF_FOLDER, pdf_file)

        file_hash = get_file_hash(full_path)

        try:

            existing = processed_collection.get(
                ids=[file_hash]
            )

            if existing["ids"]:
                continue

            log_step(f"NEW PDF DETECTED: {pdf_file}")

            text = extract_pdf_text(full_path)

            if not text.strip():

                log_step(
                    f"Skipped empty PDF: {pdf_file}"
                )

                continue

            chunks = chunk_text(text)

            for idx, chunk in enumerate(chunks):

                collection.add(
                    ids=[f"{file_hash}_{idx}"],
                    documents=[chunk],
                    metadatas=[{
                        "source_file": pdf_file,
                        "chunk_number": idx
                    }]
                )

            processed_collection.add(
                ids=[file_hash],
                documents=[pdf_file]
            )

            log_step(
                f"Indexed {pdf_file} "
                f"with {len(chunks)} chunks"
            )

        except Exception as e:

            logging.exception(
                f"Failed processing {pdf_file}: {e}"
            )


# ============================================================
# DELETE DETECTION
# ============================================================

def cleanup_deleted_pdfs():
    """
    Optional cleanup.
    Removes tracking record if file disappeared.
    Does NOT delete document vectors because
    Chroma metadata filtering on ids can become expensive.
    """

    try:

        processed = processed_collection.get()

        if not processed["ids"]:
            return

        current_hashes = set()

        for pdf_file in os.listdir(PDF_FOLDER):

            if pdf_file.lower().endswith(".pdf"):

                full_path = os.path.join(
                    PDF_FOLDER,
                    pdf_file
                )

                current_hashes.add(
                    get_file_hash(full_path)
                )

        for file_hash in processed["ids"]:

            if file_hash not in current_hashes:

                log_step(
                    f"PDF removed from folder. "
                    f"Hash={file_hash}"
                )

    except Exception as e:

        logging.exception(
            f"Cleanup failed: {e}"
        )


# ============================================================
# AUTO INDEX
# ============================================================

ingest_new_pdfs()
cleanup_deleted_pdfs()


# ============================================================
# UI
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    layout="wide"
)

st.title("📚 Auto PDF RAG Assistant")

st.write(
    "Drop PDFs into the 'pdfs' folder. "
    "They will automatically be indexed."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Database")

    if st.button("Clear Vector Database"):

        try:

            chroma_client.delete_collection(
                "documents"
            )

            chroma_client.delete_collection(
                "processed_files"
            )

            st.cache_resource.clear()

            st.success("Database cleared.")

            st.rerun()

        except Exception as e:

            st.error(str(e))

    try:

        total_docs = collection.count()

        st.metric(
            "Indexed Chunks",
            total_docs
        )

    except:
        pass


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    ["Chat", "Logs"]
)


# ============================================================
# CHAT TAB
# ============================================================

with tab1:

    query = st.text_input(
        "Ask a question:"
    )

    if query:

        try:

            log_step(
                f"USER QUERY: {query}"
            )

            with st.spinner(
                "Searching knowledge base..."
            ):

                log_step(
                    "Running vector search"
                )

                results = collection.query(
                    query_texts=[query],
                    n_results=5
                )

                documents = results["documents"][0]
                metadatas = results["metadatas"][0]

                log_step(
                    f"Retrieved {len(documents)} chunks"
                )

                context = "\n\n---\n\n".join(
                    documents
                )

                source_files = sorted(
                    list(
                        set(
                            m["source_file"]
                            for m in metadatas
                        )
                    )
                )

                prompt = f"""
You are a RAG assistant.

Answer ONLY using the supplied context.

If the answer is not present in
the context, say:

"I could not find that information
in the indexed documents."

CONTEXT:
{context}

QUESTION:
{query}
"""

                log_step(
                    "Sending prompt to LLM"
                )

                response = (
                    chat_client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=1 # low values like 0 not supported for this model
                    )
                )

                answer = (
                    response
                    .choices[0]
                    .message.content
                )

                log_step(
                    "Response generated"
                )

            st.subheader("Answer")

            st.write(answer)

            st.subheader("Sources")

            for source in source_files:
                st.write(f"• {source}")

        except Exception as e:

            logging.exception(
                f"Query failed: {e}"
            )

            st.error(str(e))


# ============================================================
# LOG TAB
# ============================================================

with tab2:

    st.subheader("Application Logs")

    if os.path.exists(LOG_FILE):

        with open(
            LOG_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            logs = f.readlines()

        logs = logs[-500:]

        st.text_area(
            "Logs",
            "".join(logs),
            height=600
        )

    else:

        st.info(
            "No logs available yet."
        )