import numpy as np
import faiss
import json
import urllib.request
from sentence_transformers import SentenceTransformer
from ingest import load_text_file, load_excel_file, chunk_text

# ── Setup ──────────────────────────────────────────────────────
txt_content  = load_text_file("data/sample.txt")
xlsx_content = load_excel_file("data/sample.xlsx")
all_text     = txt_content + "\n" + xlsx_content
chunks       = chunk_text(all_text)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
vectors     = embed_model.encode(chunks)

dimension = vectors.shape[1]
index     = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors, dtype="float32"))

print(f"Vector store ready — {index.ntotal} chunks indexed\n")

# ── Test Ollama connection ─────────────────────────────────────
def test_ollama():
    """Test if Ollama is responding"""
    try:
        payload = json.dumps({
            "model": "mistral",
            "prompt": "Say hello",
            "stream": False
        }).encode('utf-8')
        
        print(f"Sending payload: {payload}")
        
        req = urllib.request.Request(
            "http://localhost:11434/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode('utf-8'))
            print(f"✓ Ollama working! Response: {result.get('response', 'no response')[:50]}...")
            return True
    except Exception as e:
        print(f"✗ Ollama failed: {e}")
        return False

# ── Run test first ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing Ollama connection...")
    if test_ollama():
        print("\nOllama is working! Now you can use RAG.")
    else:
        print("\nFix Ollama before continuing.")