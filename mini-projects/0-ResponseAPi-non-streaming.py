import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

# Load environment variables
load_dotenv()

MODEL_NAME = "lower-model-4.1"
BASE_URL = os.getenv("OPENAI_ENDPOINT")

# Azure AD token provider
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default"
)

# OpenAI client
client = OpenAI(
    base_url=BASE_URL,
    api_key=token_provider,
)

# Streamlit UI
st.set_page_config(page_title="Azure OpenAI Demo")
st.title("Azure OpenAI Demo")

user_input = st.text_area(
    "Enter your prompt:",
    value="Summarize Azure OpenAI Responses API in one sentence."
)

if st.button("Generate Response"):

    try:
        # One-shot (non-streaming) response
        response = client.responses.create(
            model=MODEL_NAME,
            input=user_input,
        )

        # Display full response
        st.markdown(response.output_text)

    except Exception as e:
        st.error(f"Error: {str(e)}")