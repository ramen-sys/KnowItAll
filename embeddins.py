import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from ingest import load_text_file,load_excel_file,chunk_text

txt_content=load_text_file("data/sample.txt")
xlsx_content=load_excel_file("data/sample.xlsx")  
all_text=txt_content+"\n"+xlsx_content
chunks=chunk_text(all_text)

print("chunks to embed: ", len(chunks))


model=SentenceTransformer("all-MiniLM-L6-v2")
vectors=model.encode(chunks)

print("vector shape:", vectors.shape)

dimension=vectors.shape[1]
index=faiss.IndexFlatL2(dimension)

index.add(np.array(vectors,dtype="float32"))

def search(query,top_k=2):
    query_vector=model.encode([query])
    query_vector=np.array(query_vector,dtype="float32")

    distances,indices=index.search(query_vector,top_k)
    print(f"\nQuery: {query}")
    print(f"Top {top_k} results:\n")
    for rank,idx in enumerate(indices[0]):
        print(f"Rank {rank+1}: {chunks[idx]} (Distance: {distances[0][rank]:.4f})")
        print(f" {chunks[idx]}\n")
        
    


search("What is the capital of Pakistan?")
search("What is the best model of the cell phone ever?")

