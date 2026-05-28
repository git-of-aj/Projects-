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
conversation=[{"role": "system", "content": "You are a helpful assistant created by OpenAI, hosted on Azure by AJ."}]
n = 1
while n == 1:
    user_input = input("Q:")      
    conversation.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
      model=MODEL_NAME,  # Replace with your model deployment name.
        messages=conversation
    )
    
    print(response)

    conversation.append({"role": "assistant", "content": response.choices[0].message.content})
    print("\n" + response.choices[0].message.content + "\n")
    print("\n Token usage: ")
    # Print full usage object
    print("\nUsage:\n")
    print(response.usage)

    # Individual fields
    print("\nDetailed Token Usage:\n")

    print(f"Prompt Tokens: {response.usage.prompt_tokens}")
    print(f"Completion Tokens: {response.usage.completion_tokens}")
    print(f"Total Tokens: {response.usage.total_tokens}")

    # Prompt token details
    if response.usage.prompt_tokens_details:
        print("\nPrompt Token Details:")
        print(f"Cached Tokens: {response.usage.prompt_tokens_details.cached_tokens}")
        print(f"Audio Tokens: {response.usage.prompt_tokens_details.audio_tokens}")

    # Completion token details
    if response.usage.completion_tokens_details:
        print("\nCompletion Token Details:")
        print(
            f"Accepted Prediction Tokens: "
            f"{response.usage.completion_tokens_details.accepted_prediction_tokens}"
        )
        print(
            f"Rejected Prediction Tokens: "
            f"{response.usage.completion_tokens_details.rejected_prediction_tokens}"
        )
        print(
            f"Reasoning Tokens: "
            f"{response.usage.completion_tokens_details.reasoning_tokens}"
        )
        print(
            f"Audio Tokens: "
            f"{response.usage.completion_tokens_details.audio_tokens}"
        )

        n +=1