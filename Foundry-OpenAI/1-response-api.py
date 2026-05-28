from openai import OpenAI
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

load_dotenv()

MODEL_NAME="lower-model-4.1"
API_KEY=os.getenv("API_KEY")
BASE_URL=os.getenv("OPENAI_ENDPOINT")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://ai.azure.com/.default"
)

client = OpenAI(  
  base_url = BASE_URL,  
  api_key=token_provider,
)

stream = client.responses.create(
    model=MODEL_NAME,
    input="Summarize Azure OpenAI Responses API in one sentence.",
    stream=True,
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="")