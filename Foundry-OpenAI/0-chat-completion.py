from openai import OpenAI
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

load_dotenv()

MODEL_NAME=os.getenv("MODEL_NAME")
API_KEY=os.getenv("API_KEY")
BASE_URL=os.getenv("OPENAI_ENDPOINT")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://ai.azure.com/.default"
)

client = OpenAI(  
  base_url = BASE_URL,  
  api_key=token_provider,
)

response = client.chat.completions.create(
  model=MODEL_NAME,  # Replace with your model deployment name.
    messages=[
        {"role": "system", "content": "Start Every response with AJ's Training Demo Assistant: "},
        {"role": "user", "content": "Who were the founders of Microsoft? and who are you?"}
    ]
)

#print(response)
print(response.model_dump_json(indent=2))
print(response.choices[0].message.content)