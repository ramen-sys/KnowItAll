from sentence_transformers import SentenceTransformer

model=SentenceTransformer("all-MiniLM-L6-v2")

sentences=[
    "How long does shipping take?",
    "What is the return policy?",
    "How can I track my order?",
    "What payment methods do you accept?",
]    


vectors=model.encode(sentences)

print("vector_shaope:", vectors.shape)
print("first_vector:", vectors[0])