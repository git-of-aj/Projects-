import chromadb

client = chromadb.PersistentClient(
    path="./chroma_data"
)

collection = client.get_or_create_collection(
    name="animals"
)

# ----- Insert data -----
collection.add(
    documents=[
        "Cats are animals",
        "Dogs are pets",
        "Python is a programming language"
    ],
    ids=["1","2","3"]
)
# chromadb uses miniLM embeddings by default, but you can specify your own embedding function

results = collection.get()

print(results, "\n")
# SELECT * FROM animals;
results = collection.get(
    include=["documents", "embeddings"] # to retrieve and display the actual vectors.
)
print(results)

# # SELECT *
# FROM animals
# WHERE category='pet';
collection.add(
    documents=["Dogs are pets"],
    ids=["4"],
    metadatas=[
        {"category":"pet"}
    ]
)

collection.get(
    where={
        "category":"pet"
    }
)

# --------- Similarity search ---------
# TOP 3 nearest neighbors
top3 =collection.query(
    query_texts=[
        "Tell me about dogs"
    ],
    n_results=3
)
print(f"Top 3 results:\n {top3}")
