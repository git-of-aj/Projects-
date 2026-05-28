import base64
from urllib import response
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

# 2. Define a function to read and convert the image to Base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            # Read the binary data and encode it to base64
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file at {image_path} was not found.")
        return None

# 3. Path to your local image
image_path = r"C:\Users\anana\OneDrive\Pictures\Screenshots\no-prob-assurance.png"

def openAI_image_request(image_path):

    # 4. Get the Base64 string
    base64_image = encode_image(image_path)

    if base64_image:
        print("Image encoded successfully. Sending to OpenAI...")
        
        # 5. Send the request to the Chat Completion API
        response = client.chat.completions.create(
            model=MODEL_NAME, # gpt-4o is currently the standard multimodal model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Please describe what is in this image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                # Format the base64 string correctly for the API
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )

        # 6. Print the AI's response
        print("\nResponse:")
        print(response.choices[0].message.content)

if __name__ == "__main__":
    openAI_image_request(image_path)