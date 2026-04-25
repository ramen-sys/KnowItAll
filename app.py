import streamlit as st
import numpy as np
import faiss
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM
from ingest import load_text_file, load_excel_file, chunk_text
import urllib.request
from sentence_transformers import SentenceTransformer
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM 
llm = OllamaLLM(model="phi3:mini", temperature=0.7)
st.set_page_config(page_title="Custom Knowledge Bot", layout="wide")
st.title("Custom Knowledge Bot")
st.write("Ask questions based on the uploaded documents.")

def chunk_text(text):
    splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks=splitter.split_text(text)
    return chunks

def load_text_file(uploaded_file):
    all_text=""

    for file in uploaded_file:
        if file.type=="text/plain":
            text=file.read().decode("utf-8")
            all_text+=text+"\n"
        elif file.type=="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            df=pd.read_excel(file)
            rows_as_text=[]
            for _,row in df.iterrows():
                row_text="|".join(f"{col}:{val}" for col,val in row.items() )
                rows_as_text.append(row_text)
            combined="\n".join(rows_as_text)
            all_text+=combined+"\n"
    return all_text
def build_vector_store(chunks):
    embed_model=SentenceTransformer("all-MiniLM-L6-v2")
    vectors=embed_model.encode(chunks)  
    dimension=vectors.shape[1]
    index=faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors,dtype="float32"))
    return index, embed_model

def retrieve(query, index, embed_model, chunks, top_k=3):
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
        
st.sidebar.header("Upload Documents")
uploaded_files=st.sidebar.file_uploader("Upload text or Excel files", type=["txt","xlsx"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing documents..."):
        all_text=load_text_file(uploaded_files)
        chunks=chunk_text(all_text)
        index, embed_model=build_vector_store(chunks)

        st.session_state.index=index
        st.session_state.embed_model=embed_model
        st.session_state.chunks=chunks
        st.session_state.ready=True
        st.success("Documents processed and vector store built!")
else:
    st.info("Please upload at least one text or Excel file to get started.")
    st.session_state.ready=False

if st.session_state.get("ready", False):
    if "messages"   not in st.session_state:
        st.session_state.messages=[]
    
    for message in st.session_state.messages:
       with st.chat_message(message["role"]):
           st.markdown(message["content"])
    if user_question:=st.chat_input("Ask a question about the documents..."):
        st.session_state.messages.append({"role":"user","content":user_question})
        with st.chat_message("user"):
           st.markdown(user_question)
        with st.chat_message("assistant"):
           with st.spinner("Generating response..."):
                relevant_chunks=retrieve(user_question, st.session_state.index, st.session_state.embed_model, st.session_state.chunks)
                if not relevant_chunks:
                     response="Sorry, I couldn't find relevant information."
                else:
                    context="\n\n".join(relevant_chunks)
                    prompt=f"""You are a helpful assistant. Use the following context to answer the question. Question: {user_question} Context:{context} Answer:"""
                    answer=generate_response(prompt)
                    st.write(answer)

        st.session_state.messages.append({"role":"assistant","content":answer})


