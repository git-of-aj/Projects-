from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv
import concurrent
import os

# Load environment variables
load_dotenv()

# Initialize OpenAI client
# CONFIGURATION
MODEL_NAME=os.getenv("MODEL_NAME")
API_KEY=os.getenv("API_KEY")
BASE_URL=os.getenv("OPENAI_ENDPOINT")

# API key authentication
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# Directory containing PDF files
dir_pdfs = r"C:\Users\anana\Downloads\Upgrad-AI-Course\Foundry-OpenAI\Policies"  # Store PDFs locally in this folder

# Validate directory existence
if not os.path.exists(dir_pdfs):
    raise FileNotFoundError(f"Directory not found: {dir_pdfs}")

# Get all PDF files
pdf_files = [
    os.path.join(dir_pdfs, f)
    for f in os.listdir(dir_pdfs)
    if f.endswith(".pdf")
]

print(pdf_files)

"""
Creating Vector Store with our PDFs

Steps:
1. Create a Vector Store on OpenAI's servers
2. Upload files to the vector store
"""


def create_vector_store(store_name: str) -> dict:
    try:
        vector_store = client.vector_stores.create(name=store_name)

        details = {
            "id": vector_store.id,
            "name": vector_store.name,
            "created_at": vector_store.created_at,
            "file_count": vector_store.file_counts.completed,
        }

        print("Vector store created:", details)
        return details

    except Exception as e:
        print(f"Error creating vector store: {e}")
        return {}


store_name = "policies_vector_store"
vector_store_details = create_vector_store(store_name)


def upload_single_pdf(file_path: str, vector_store_id: str):
    file_name = os.path.basename(file_path)

    try:
        with open(file_path, "rb") as f:
            file_response = client.files.create(
                file=f,
                purpose="assistants"
            )

        client.vector_stores.files.create(
            vector_store_id=vector_store_id,
            file_id=file_response.id
        )

        return {
            "file": file_name,
            "status": "success"
        }

    except Exception as e:
        print(f"Error with {file_name}: {str(e)}")

        return {
            "file": file_name,
            "status": "failed",
            "error": str(e)
        }


def upload_pdf_files_to_vector_store(vector_store_id: str):
    pdf_files = [
        os.path.join(dir_pdfs, f)
        for f in os.listdir(dir_pdfs)
        if f.endswith(".pdf")
    ]

    stats = {
        "total_files": len(pdf_files),
        "successful_uploads": 0,
        "failed_uploads": 0,
        "errors": []
    }

    print(f"{len(pdf_files)} PDF files to process. Uploading in parallel...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(
                upload_single_pdf,
                file_path,
                vector_store_id
            ): file_path
            for file_path in pdf_files
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(pdf_files)
        ):
            result = future.result()

            if result["status"] == "success":
                stats["successful_uploads"] += 1
            else:
                stats["failed_uploads"] += 1
                stats["errors"].append(result)

    return stats


# Upload all PDFs to vector store
if vector_store_details:
    upload_stats = upload_pdf_files_to_vector_store(
        vector_store_details["id"]
    )

    print("\nUpload Summary:")
    print(upload_stats)

"""
Standalone Vector Search

Now that our vector store is ready, we can query the Vector Store
directly and retrieve relevant content for a specific query.
"""

# --------------------Unsupported By Foundry but supported by OpenAI--------------------
# Date: May 29, 2026
# Proof:  https://learn.microsoft.com/en-us/answers/questions/2261951/azure-openai-vector-store-attributes-retrieval
# OpenAPI support: https://developers.openai.com/api/docs/guides/retrieval
# ----------------------------------------------------------------------------------------
# # Query 1
# query = "What is the leave policy?"

# search_results = client.vector_stores.search(
#     vector_store_id=vector_store_details["id"],
#     query=query,

# )

# print("\nSearch Result:")
# print(search_results.data[0].content[0].text)

# # Query 2
# query = "What is the notice period for employees ?"

# search_results = client.vector_stores.search(
#     vector_store_id=vector_store_details["id"],
#     query=query
# )

# print("\nDetailed Results:")

# for result in search_results.data:
#     print(
#         f"{len(result.content[0].text)} characters "
#         f"from {result.filename} "
#         f"with relevance score {result.score}"
#     )

"""
Integrating search results with LLM in a single API call
"""

query = "What is the leave policy?"

response = client.responses.create(
    input=query,
    model=MODEL_NAME,
    tools=[
        {
            "type": "file_search",
            "vector_store_ids": [vector_store_details["id"]],
        }
    ]
)

# Safely extract response
try:
    annotations = response.output[1].content[0].annotations

    # Get retrieved filenames
    retrieved_files = {
        result.filename
        for result in annotations
    }

    print(f"\nFiles used: {retrieved_files}")

    print("\nResponse:")
    print(response.output[1].content[0].text)

except Exception as e:
    print(f"Error reading response output: {e}")


def get_response_from_vectorstore(query: str):
    """
    Query the vector store using file_search tool
    and print the response with source files.
    """

    try:
        response = client.responses.create(
            input=query,
            model=MODEL_NAME,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [
                        vector_store_details["id"]
                    ],
                }
            ]
        )

        annotations = response.output[1].content[0].annotations

        retrieved_files = {
            result.filename
            for result in annotations
        }

        print(f"\nFiles used: {retrieved_files}")

        print("\nResponse:")
        print(response.output[1].content[0].text)

    except Exception as e:
        print(f"Error querying vector store: {e}")


# Example usage
get_response_from_vectorstore(
    "Explain the employee leave policy."
)