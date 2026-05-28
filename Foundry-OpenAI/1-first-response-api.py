import os
from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
load_dotenv()

# CONFIGURATION
MODEL_NAME=os.getenv("MODEL_NAME")
API_KEY=os.getenv("API_KEY")
BASE_URL=os.getenv("OPENAI_ENDPOINT")

# API key authentication
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)
response = client.responses.create(
    model=MODEL_NAME,
    input="This is a test."
)
print(response.model_dump_json(indent=2))
# Access the total_tokens attribute directly
total_tokens = response.usage.total_tokens

print(f"Total tokens used: {total_tokens}")

# Microsoft Entra ID authentication (recommended)
token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://ai.azure.com/.default"
)
Entra_client = OpenAI(
    base_url=BASE_URL,
    api_key=token_provider(),
)
response = Entra_client.responses.create(
    model=MODEL_NAME,
    input="This is a test."
)
# print(response.model_dump_json(indent=2))

# -----2. --------
response = client.responses.retrieve("resp_0bbf221ee390f55c006a1711f6eb288196862159c797706736")
print(response.model_dump_json(indent=2))