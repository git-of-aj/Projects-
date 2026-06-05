"""
Azure AI Foundry + Azure AI Search RAG Chatbot
==============================================

This Streamlit application demonstrates a simple Retrieval-Augmented Generation (RAG) pattern:

1. User asks a question
2. Azure AI Search retrieves relevant document chunks
3. Retrieved content is injected into the prompt
4. Azure OpenAI (Azure AI Foundry) generates an answer using that context
5. Chat history is maintained in Streamlit session state

Prerequisites
-------------
pip install streamlit openai azure-search-documents azure-identity

Required Environment Variables
------------------------------
AZURE_SEARCH_ENDPOINT
AZURE_SEARCH_INDEX
FOUNDRY_ENDPOINT
FOUNDRY_DEPLOYMENT

Authentication
--------------
This example uses Microsoft Entra ID (DefaultAzureCredential)
instead of API keys.
"""

import os
import streamlit as st

from openai import OpenAI
from azure.search.documents import SearchClient
from azure.identity import (
    DefaultAzureCredential,
    get_bearer_token_provider
)
from dotenv import load_dotenv
load_dotenv()

import logging
import time
from datetime import datetime
# ==========================================================
# LOGGING CONFIGURATION
# ==========================================================
# Logs are written to app.log
# ==========================================================

LOG_FILE = "app.log"

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(message)s"
    ),
    encoding="utf-8"
)

logger = logging.getLogger(__name__)

# ==========================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==========================================================
# Must be called before most Streamlit UI commands.
# Configures browser tab title, icon and layout.
# ==========================================================

st.set_page_config(
    page_title="Foundry RAG Assistant",
    page_icon="🔍",
    layout="centered"
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
# 2. AUTHENTICATION SETUP
# ==========================================================
# DefaultAzureCredential automatically tries:
# - Managed Identity
# - Azure CLI login
# - Visual Studio login
# - Environment variables
# - Other supported credential providers
#
# This allows secure authentication without storing keys.
# ==========================================================

credential = DefaultAzureCredential()

# Token provider used by Azure OpenAI.
# The scope below is the recommended scope for
# Azure OpenAI / Azure AI Foundry resources.
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://ai.azure.com/.default"
)

# ==========================================================
# 3. LOAD CONFIGURATION
# ==========================================================
# Read settings from environment variables.
# Using environment variables is recommended over
# hardcoded secrets.
# ==========================================================

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

FOUNDRY_ENDPOINT = os.getenv("FOUNDRY_ENDPOINT")
FOUNDRY_DEPLOYMENT = os.getenv("FOUNDRY_DEPLOYMENT")

# ==========================================================
# 4. VALIDATE CONFIGURATION
# ==========================================================
# Fail fast if any required setting is missing.
# This prevents confusing runtime errors later.
# ==========================================================

required_configs = {
    "AZURE_SEARCH_ENDPOINT": SEARCH_ENDPOINT,
    "AZURE_SEARCH_INDEX": SEARCH_INDEX,
    "FOUNDRY_ENDPOINT": FOUNDRY_ENDPOINT,
    "FOUNDRY_DEPLOYMENT": FOUNDRY_DEPLOYMENT,
}

missing_configs = [
    key for key, value in required_configs.items()
    if not value
]

if missing_configs:
    st.error(
        "Missing environment variables:\n\n"
        + "\n".join(missing_configs)
    )
    st.stop()

# ==========================================================
# 5. CREATE AZURE AI SEARCH CLIENT
# ==========================================================
# Streamlit caches the client object so it is
# created only once per session.
#
# This improves performance because we don't need
# to reconnect on every interaction.
# ==========================================================

@st.cache_resource
def get_search_client():
    """
    Creates and returns an Azure AI Search client.
    """

    return SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=SEARCH_INDEX,
        credential=credential
    )

# ==========================================================
# 6. CREATE AZURE OPENAI CLIENT
# ==========================================================
# Uses Microsoft Entra ID authentication instead of
# API keys.
#
# azure_ad_token_provider automatically refreshes
# access tokens when required.
# ==========================================================

@st.cache_resource
def get_openai_client():
    """
    Creates and returns Azure OpenAI client.
    """

    return OpenAI(  
    base_url = FOUNDRY_ENDPOINT,  
    api_key=token_provider,
    )

# Initialize both clients
search_client = get_search_client()
openai_client = get_openai_client()

# ==========================================================
# 7. SESSION STATE
# ==========================================================
# Streamlit reruns the script after every user action.
#
# Session state allows us to preserve chat history
# between reruns.
# ==========================================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================================================
# 8. DOCUMENT RETRIEVAL FUNCTION
# ==========================================================
# This function performs a keyword search against
# Azure AI Search and returns the top matching chunks.
#
# IMPORTANT:
# Replace "content" with the actual field name in
# your Azure Search index.
# ==========================================================

def retrieve_search_context(
    query: str,
    top_k: int = 3
) -> str:
    """
    Retrieves relevant document snippets from Azure AI Search.
    """
    citations = []
    context_text = ""
    try:

        logger.info(
            f"SEARCH START | Query='{query}' | top_k={top_k}"
        )

        search_start = time.time()

        results = search_client.search(
            search_text=query,
            top=top_k
        )

        for idx, doc in enumerate(results):
        # Fallback to 'id' if 'title' is empty or None
            doc_name = doc.get("title") if doc.get("title") else doc.get("id")
            content = doc.get("content")
            
            # Create a citation marker like [1], [2]
            citation_marker = f"[{idx + 1}]"

            citations.append(f"{citation_marker} **{doc_name}**")
            
            # Feed the content to the LLM with the citation marker so the LLM knows how to cite it
            context_text += f"\nDocument {citation_marker}:\n{content}\n"
        context_snippets = []

        result_count = 0

        for doc in results:

            result_count += 1

            content = doc.get("content", "")

            if content:
                context_snippets.append(content)

        search_time = round(
            time.time() - search_start,
            2
        )

        logger.info(
            f"SEARCH COMPLETE | "
            f"Results={result_count} | "
            f"Time={search_time}s"
        )

        context = "\n\n---\n\n".join(
            context_snippets
        )

        logger.info(
            f"RETRIEVED CONTEXT (first 1000 chars):\n"
            f"{context[:1000]}"
        )

        return context

    except Exception as ex:

        logger.exception(
            f"SEARCH ERROR: {str(ex)}"
        )

        st.error(
            f"Azure Search error: {str(ex)}"
        )

        return ""

# ==========================================================
# 9. DISPLAY PREVIOUS CHAT MESSAGES
# ==========================================================
# Every time Streamlit reruns, redraw all messages.
# ==========================================================

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==========================================================
# 10. CHAT INPUT
# ==========================================================
# Waits for user question.
# ==========================================================

user_query = st.chat_input(
    "Ask a question about your documents..."
)
logger.info("=" * 80)
logger.info("NEW USER REQUEST")
logger.info(f"QUESTION:\n{user_query}")

if user_query:

    # ======================================================
    # DISPLAY USER MESSAGE
    # ======================================================

    with st.chat_message("user"):
        st.markdown(user_query)

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": user_query
        }
    )

    # ======================================================
    # STEP 1: RETRIEVE DOCUMENT CONTEXT
    # ======================================================

    with st.spinner("Searching knowledge base..."):

        retrieved_context = retrieve_search_context(
            user_query,
            top_k=3
        )

    # ======================================================
    # HANDLE NO RESULTS FOUND
    # ======================================================

    if not retrieved_context:

        no_results_message = (
            "I could not find any relevant documents "
            "for your question."
        )

        with st.chat_message("assistant"):
            st.markdown(no_results_message)

        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": no_results_message
            }
        )

        st.stop()

    # ======================================================
    # STEP 2: BUILD RAG PROMPT
    # ======================================================
    # We explicitly instruct the model to answer
    # only from the retrieved context.
    # ======================================================

    system_prompt = """
You are a helpful enterprise assistant.

Rules:
1. Answer ONLY using the supplied context.
2. If the answer is not present in the context,
   say:
   "I could not find that information in the
   provided documents."
3. Do not make up information.
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
    # STEP 3: GENERATE RESPONSE
    # ======================================================
    #
    # We use the Responses API.
    #
    # If your Azure region does not support it,
    # switch to chat.completions.create().
    # ======================================================

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        try:
            llm_start = time.time()

            logger.info(
                f"LLM REQUEST START | "
                f"Model={FOUNDRY_DEPLOYMENT}"
            )

            response = openai_client.responses.create(
                model=FOUNDRY_DEPLOYMENT,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": augmented_prompt
                    }
                ]
            )

            # Most recent SDK versions expose output_text
            bot_reply = response.output_text
            if citations:
                bot_reply += "\n\n---\n**Citations:**\n" + "\n".join(citations)
            else:
                bot_reply += "\n\n---\n*No relevant documents found to cite.*"
            llm_time = round(
                time.time() - llm_start,
                2
            )

            logger.info(
                f"LLM RESPONSE RECEIVED | "
                f"Time={llm_time}s"
            )

            logger.info(
                f"LLM RESPONSE:\n{bot_reply}"
            )

            response_placeholder.markdown(bot_reply)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": bot_reply
                }
            )

        except Exception as ex:

            error_message = (
                f"Error communicating with Azure OpenAI:\n\n"
                f"{str(ex)}"
            )
            logger.exception(
                f"OPENAI ERROR: {str(ex)}"
            )

            response_placeholder.error(error_message)

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": error_message
                }
            )
        logger.info(
                    "REQUEST COMPLETED SUCCESSFULLY"
                )

        logger.info("=" * 80)

# ==========================================================
# END OF APPLICATION
# ==========================================================
#
# Flow Summary:
#
# User Question
#       ↓
# Azure AI Search
#       ↓
# Retrieve Top Chunks
#       ↓
# Build Prompt
#       ↓
# Azure OpenAI / Foundry
#       ↓
# Generated Answer
#       ↓
# Display Response
#
# This architecture is known as:
# Retrieval-Augmented Generation (RAG)
#
# ==========================================================
