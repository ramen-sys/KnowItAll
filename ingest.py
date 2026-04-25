import streamlit as st
import pandas as pd
import os

import tempfile

from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_text_file(filepath):
    with open(filepath,"r",encoding="utf-8") as f:
        text= f.read()
    return text

def load_excel_file(filepath):
    df=pd.read_excel(filepath)
    rows_as_text=[]
    for _,row in df.iterrows():
        row_text="|".join(f"{col}:{val}" for col,val in row.items() )
        rows_as_text.append(row_text)
    combined="\n".join(rows_as_text)
    return combined

def chunk_text(text):
    splitter=RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks=splitter.split_text(text)
    return chunks

if __name__=="__main__":
    txt_content=load_text_file("data/sample.txt")
    xlsx_content=load_excel_file("data/sample.xlsx")

    all_text=txt_content+"\n"+xlsx_content
    chunk=chunk_text(all_text)

    print("total chunks:", len(chunk))
    for i, c in enumerate(chunk[:5]):
        print(f"--- Chunk {i+1} ---")
        print(c)
        print()

