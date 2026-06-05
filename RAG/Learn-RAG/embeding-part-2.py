# This code is doing semantic similarity search.
#The idea is:
#"Given a query, which document is most similar in meaning?"
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

# Think of these as documents stored in your knowledge base.
docs = [
    "Cats are animals",
    "Dogs are pets",
    "Python is a programming language"
]
#     "Tell me about cats", # 1 as it is exact same as query
#     "Tell me about Cats"
# ]
# The model creates one vector per document.

"""
Conceptually:

embeddings = [
    [384 numbers],   # Cats are animals
    [384 numbers],   # Dogs are pets
    [384 numbers]    # Python is a programming language
]

Shape:

(3, 384)

Meaning:

3 documents
each represented by 384 numbers
"""

embeddings = model.encode(docs)

query = "Tell me about cats"

query_embedding = model.encode([query])

scores = cosine_similarity(
    query_embedding,
    embeddings
)

print(scores)
# cosine similarity scores between the query and each document via matrix multiplication
# Query vs Doc 0
# Query vs Doc 1
# Query vs Doc 2

# Now Pick the document with the highest score (most similar in meaning to the query)
best_doc_index = scores.argmax()
print("Best matching document:", docs[best_doc_index])

"""
This is essentially the basic retrieval step used in many RAG pipelines:

Convert documents into embeddings.
Store embeddings.
Convert user query into an embedding.
Compute cosine similarity.
Return the most similar documents.
Send those documents to an LLM for answer generation.
"""
