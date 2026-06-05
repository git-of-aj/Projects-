"""
Azure AI Foundry + Azure AI Search RAG Chatbot
==============================================

Streamlit RAG chatbot using:
- Azure AI Search for retrieval
- Azure AI Foundry / Azure OpenAI Responses API
- Microsoft Entra ID authentication
"""

import os
import time
import logging

import streamlit as st

from openai import OpenAI
from azure.search.documents import SearchClient
from azure.identity import (
    DefaultAzureCredential,
    get_bearer_token_provider,
)
from dotenv import load_dotenv

# ==========================================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================================

load_dotenv()

# ==========================================================
# LOGGING CONFIGURATION
# ==========================================================

LOG_FILE = "app.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

logger = logging.getLogger(__name__)

# ==========================================================
# STREAMLIT PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Foundry RAG Assistant",
    page_icon="🔍",
    layout="centered",
)

st.title("🔍 Azure AI Foundry + Search Assistant")

st.markdown(
    "Ask questions about your internal documents using "
    "Azure AI Search + Azure OpenAI."
)

logger.info("=" * 80)
logger.info("Application Started")
logger.info("=" * 80)

# ==========================================================
# AUTHENTICATION
# ==========================================================

credential = DefaultAzureCredential()

token_provider = get_bearer_token_provider(
    credential,
    "https://ai.azure.com/.default",
)

# ==========================================================
# CONFIGURATION
# ==========================================================

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT")
FOUNDRY_DEPLOYMENT = os.getenv("FOUNDRY_DEPLOYMENT")

required_configs = {
    "AZURE_SEARCH_ENDPOINT": SEARCH_ENDPOINT,
    "AZURE_SEARCH_INDEX": SEARCH_INDEX,
    "FOUNDRY_ENDPOINT": FOUNDRY_ENDPOINT,
    "FOUNDRY_DEPLOYMENT": FOUNDRY_DEPLOYMENT,
}

missing_configs = [
    key
    for key, value in required_configs.items()
    if not value
]

if missing_configs:
    st.error(
        "Missing environment variables:\n\n"
        + "\n".join(missing_configs)
    )
    st.stop()

# ==========================================================
# CLIENT FACTORIES
# ==========================================================


@st.cache_resource
def get_search_client():
    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX,
        credential=credential,
    )


@st.cache_resource
def get_openai_client():
    endpoint = FOUNDRY_ENDPOINT.rstrip("/")

    if not endpoint.endswith("/openai/v1"):
        endpoint = f"{endpoint}/openai/v1"

    return OpenAI(
        base_url=endpoint,
        api_key=token_provider,
    )


search_client = get_search_client()
openai_client = get_openai_client()

# ==========================================================
# SESSION STATE
# ==========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================================
# RETRIEVAL
# ==========================================================


def retrieve_search_context(
    query: str,
    top_k: int = 3,
):
    """
    Returns:
        context_text (str)
        citations (list[str])
    """

    citations = []

    try:
        logger.info(
            f"SEARCH START | Query='{query}' | top_k={top_k}"
        )

        search_start = time.time()

        results = list(
            search_client.search(
                search_text=query,
                top=top_k,
            )
        )

        context_parts = []

        for idx, doc in enumerate(results):
            doc_name = (
                doc.get("title")
                or doc.get("id")
                or f"Document {idx + 1}"
            )

            content = doc.get("content", "")

            if not content:
                continue

            marker = f"[{idx + 1}]"

            citations.append(
                f"{marker} **{doc_name}**"
            )

            context_parts.append(
                f"{marker}\n{content}"
            )

        search_time = round(
            time.time() - search_start,
            2,
        )

        logger.info(
            f"SEARCH COMPLETE | "
            f"Results={len(results)} | "
            f"Time={search_time}s"
        )

        context_text = "\n\n---\n\n".join(
            context_parts
        )

        logger.info(
            "RETRIEVED CONTEXT "
            f"(first 1000 chars):\n"
            f"{context_text[:1000]}"
        )

        return context_text, citations

    except Exception as ex:
        logger.exception(
            f"SEARCH ERROR: {str(ex)}"
        )

        st.error(
            f"Azure Search error: {str(ex)}"
        )

        return "", []

# ==========================================================
# DISPLAY CHAT HISTORY
# ==========================================================

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================================
# USER INPUT
# ==========================================================

user_query = st.chat_input(
    "Ask a question about your documents..."
)

if user_query:

    logger.info("=" * 80)
    logger.info("NEW USER REQUEST")
    logger.info(f"QUESTION:\n{user_query}")

    # ======================================================
    # USER MESSAGE
    # ======================================================

    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    # ======================================================
    # RETRIEVE CONTEXT
    # ======================================================

    with st.spinner(
        "Searching knowledge base..."
    ):
        (
            retrieved_context,
            citations,
        ) = retrieve_search_context(
            user_query,
            top_k=3,
        )

    # ======================================================
    # NO RESULTS
    # ======================================================

    if not retrieved_context:

        no_results_message = (
            "I could not find any relevant "
            "documents for your question."
        )

        with st.chat_message("assistant"):
            st.markdown(
                no_results_message
            )

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": no_results_message,
            }
        )

        st.stop()

    # ======================================================
    # BUILD PROMPT
    # ======================================================

    system_prompt = """
You are a helpful enterprise assistant.

Rules:
1. Answer ONLY using the supplied context.
2. Cite supporting documents using citation markers
   such as [1], [2], [3] when possible.
3. If the answer is not present in the context,
   respond exactly with:

   I could not find that information in the provided documents.

4. Do not make up information.
"""

    augmented_prompt = f"""
Context:
{retrieved_context}

Question:
{user_query}
"""

    logger.info(
        "PROMPT SENT TO LLM "
        "(first 3000 chars)"
    )

    logger.info(
        augmented_prompt[:3000]
    )

    # ======================================================
    # GENERATE RESPONSE
    # ======================================================

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        try:
            llm_start = time.time()

            logger.info(
                f"LLM REQUEST START | "
                f"Model={FOUNDRY_DEPLOYMENT}"
            )

            response = (
                openai_client.responses.create(
                    model=FOUNDRY_DEPLOYMENT,
                    input=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content": augmented_prompt,
                        },
                    ],
                )
            )

            bot_reply = (
                getattr(
                    response,
                    "output_text",
                    None,
                )
                or "No response returned."
            )

            if citations:
                bot_reply += (
                    "\n\n---\n\n"
                    "**Sources:**\n"
                    + "\n".join(citations)
                )

            llm_time = round(
                time.time() - llm_start,
                2,
            )

            logger.info(
                f"LLM RESPONSE RECEIVED | "
                f"Time={llm_time}s"
            )

            logger.info(
                f"LLM RESPONSE:\n{bot_reply}"
            )

            response_placeholder.markdown(
                bot_reply
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": bot_reply,
                }
            )

            logger.info(
                "REQUEST COMPLETED SUCCESSFULLY"
            )

        except Exception as ex:

            logger.exception(
                f"OPENAI ERROR: {str(ex)}"
            )

            error_message = (
                "Error communicating with "
                "Azure OpenAI:\n\n"
                f"{str(ex)}"
            )

            response_placeholder.error(
                error_message
            )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": error_message,
                }
            )

    logger.info("=" * 80)

# ==========================================================
# END
# ==========================================================
