from openai import OpenAI
import os
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

load_dotenv()

MODEL_NAME="DeepSeek-R1"
API_KEY=os.getenv("API_KEY")
BASE_URL=os.getenv("OPENAI_ENDPOINT")

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "https://ai.azure.com/.default"
)

client = OpenAI(  
  base_url = BASE_URL,  
  api_key=token_provider,
)
conversation=[{"role": "system", "content": "You are a helpful assistant created by OpenAI, hosted on Azure by AJ."}]
n = 1
#while n == 1:
while True:
    user_input = input("You: ")      
    conversation.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
      model=MODEL_NAME,  # Replace with your model deployment name.
        messages=conversation
    )
    
    print(response)

    conversation.append({"role": "assistant", "content": response.choices[0].message.content})
    print("\n" + response.choices[0].message.content + "\n")