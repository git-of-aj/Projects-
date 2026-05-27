import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
load_dotenv()


API_KEY=os.getenv("API_KEY")
BASE_URL=os.getenv("INFERENCE_ENDPOINT")

client = ChatCompletionsClient(
    endpoint=BASE_URL,
    credential=AzureKeyCredential(API_KEY),
)

# Use your specific deployment name here
Llama_deployment_name = "Llama" 
grok_deployment_name = "grok-4.3"

LLma_response = client.complete(
    model=Llama_deployment_name, 
    messages=[
        SystemMessage(content="You are a helpful AI assistant."),
        UserMessage(content="Who are you? Are you created by OpenAI or by Meta AI or by Mistral?")
    ],
    temperature=0.7,
    max_tokens=512
)

print(LLma_response.choices[0].message.content)

grok_response = client.complete(
    model=grok_deployment_name, 
    messages=[
        SystemMessage(content="You are a helpful AI assistant."),
        UserMessage(content="Who are you? Are you created by OpenAI or by Meta AI or by Mistral or anyother company?")
    ]
)
print('*'*100,grok_response.choices[0].message.content,sep='\n')

