# https://pypi.org/project/sentence-transformers/
# This code is generating a sentence embedding — 
# a numerical representation of the meaning of a piece of text AKA vector representation of text.
# The vector is meant to capture the sentence's meaning 
# so it can be compared with other text using similarity measures

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Cats are animals"
"""
The model:

Breaks the sentence into tokens
Passes them through a transformer neural network
Produces a dense vector representation
"""
embedding = model.encode(text)
"""
Why 384 dimensions?
Other embedding models may output:

384 dimensions
768 dimensions
depending on their design.

For all-MiniLM-L6-v2, the embedding size is 384.
"""
print(f"Length of embedding: {len(embedding)} \nType of embedding: {type(embedding)} \n")
print(f"First 10 elements: {embedding[:10]}")

"""
So "Loading weights" means:

Load the learned neural network parameters into memory.

Not necessarily downloading them.
"""

s1 = "Cats are animals"
s2 = "A cat is a type of animal"
s3 = "I like football"

e1 = model.encode(s1)
e2 = model.encode(s2)
e3 = model.encode(s3)

from sentence_transformers.util import cos_sim

print(f"Similarity between s1 and s2: {cos_sim(e1, e2).item():.4f}") # Near 1 means very similar
print(f"Similarity between s1 and s3: {cos_sim(e1, e3).item():.4f}")
print(f"Similarity between s2 and s3: {cos_sim(e2, e3).item():.4f}")