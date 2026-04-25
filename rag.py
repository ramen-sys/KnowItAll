import json
import os
import numpy as np
import faiss
import urllib.request
import urllib.error
from dotenv import load_dotenv
import requests
from sentence_transformers import SentenceTransformer
from typer import prompt
from ingest import load_text_file,load_excel_file,chunk_text
from openai import OpenAI
from langchain_ollama import OllamaLLM 
load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = OllamaLLM(model="phi3:mini", temperature=0.7)
txt_content=load_text_file("data/sample.txt")
xlsx_content=load_excel_file("data/sample.xlsx")
all_text=txt_content+"\n"+xlsx_content
chunks=chunk_text(all_text)

embed_model=SentenceTransformer("all-MiniLM-L6-v2")
vectors=embed_model.encode(chunks)  

dimension=vectors.shape[1]
index=faiss.IndexFlatL2(dimension)
index.add(np.array(vectors,dtype="float32"))
print(f"Vector store ready with {len(chunks)} chunks.")

def retrieve(query, top_k=3):
    query_vector=embed_model.encode([query])
    query_vector=np.array(query_vector,dtype="float32")
    distances,indices=index.search(query_vector,top_k)
    
    threshold=2.0
    relevant_chunks=[]
    for dist,idx in zip(distances[0], indices[0]):
        if dist<threshold:
            relevant_chunks.append(chunks[idx]) 
    return relevant_chunks
def generate_response(prompt):
    
   try:
       response=llm.invoke(prompt)
       return response.strip()
   except Exception as e:
       print(f"Error generating response: {e}")
       return "Sorry, I couldn't generate a response at this time."
        
       
        
    
def ask_question(question):
    relevant_chunks=retrieve(question)
    if not relevant_chunks:
        print(f"\nQ: {question}\nA: Sorry, I couldn't find relevant information.")
        return
    
    context="\n\n".join(relevant_chunks)

    prompt=f"""You are a helpful assistant. Answer the user's question based only context provided below.If the answer is not in the context, say "I don't have that information" Context:{context}
    Question: {question}
    Answer:"""

    print(f"Q: {question}\nA: {generate_response(prompt)}\n")

ask_question("How long does a refund take?")
ask_question("What is the return policy?")
ask_question("What is the company's mission statement?")
ask_question("What is the capital of France?")

    
