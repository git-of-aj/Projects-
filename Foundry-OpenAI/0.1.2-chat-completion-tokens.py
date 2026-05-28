import tiktoken
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

system_message = {"role": "system", "content": "You are a helpful assistant."}
max_response_tokens = 250
token_limit = 4096
conversation = []
conversation.append(system_message)

def num_tokens_from_messages(messages, model="gpt-4o"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    
    if model in {
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-5",
        "gpt-4.1",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
    }:
        tokens_per_message = 3
        tokens_per_name = 1

    elif any(model.startswith(prefix) for prefix in [
        "gpt-4o-", 
        "gpt-5-", 
        "gpt-4.1-",
        "o1-", 
        "o3-", 
        "o4-mini-",
    ]):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. """
        )
    
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  
    return num_tokens

while True:
    user_input = input("Q:")      
    conversation.append({"role": "user", "content": user_input})
    conv_history_tokens = num_tokens_from_messages(conversation, model="gpt-4o")

    while conv_history_tokens + max_response_tokens >= token_limit:
        del conversation[1] 
        conv_history_tokens = num_tokens_from_messages(conversation, model="gpt-4o")

    response = client.chat.completions.create(
      model=MODEL_NAME,
        messages=conversation,
        temperature=0.7,
        max_tokens=max_response_tokens
    )

    conversation.append({"role": "assistant", "content": response.choices[0].message.content})
    print("\n" + response.choices[0].message.content + "\n")