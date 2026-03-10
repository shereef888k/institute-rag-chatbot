import os
import pickle
import faiss
import numpy as np
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from text_utils import normalize_text, chunk_text
from web_loader import fetch_webpage_text

DOCS_DIR = "data/docs"
STORE_DIR = "data/store"
INDEX_PATH = os.path.join(STORE_DIR, "qtest_index.faiss")
META_PATH = os.path.join(STORE_DIR, "metadata.pkl")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

WEB_URLS = [
    "https://www.qtestsolutions.com/",
    "https://www.qtestsolutions.com/about-us/",
    "https://www.qtestsolutions.com/contact-us/",
    "https://www.qtestsolutions.com/software-testing-training-in-kozhikode/",
]

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = []
    for page in reader.pages:
        text.append(page.extract_text() or "")
    return "\n".join(text)

def read_docx(file_path):
    doc = Document(file_path)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".txt":
        return read_txt(file_path)
    elif ext == ".pdf":
        return read_pdf(file_path)
    elif ext == ".docx":
        return read_docx(file_path)

    return ""

def ingest_documents():
    os.makedirs(STORE_DIR, exist_ok=True)
    all_chunks = []

    # Read local files
    if os.path.exists(DOCS_DIR):
        for filename in os.listdir(DOCS_DIR):
            file_path = os.path.join(DOCS_DIR, filename)

            if not os.path.isfile(file_path):
                continue

            text = extract_text(file_path)
            text = normalize_text(text)

            if not text:
                continue

            chunks = chunk_text(text, chunk_size=500, overlap=100)

            for i, chunk in enumerate(chunks):
                all_chunks.append({
                    "chunk_id": f"{filename}_chunk_{i}",
                    "text": chunk,
                    "source": filename
                })

    # Read website pages
    for url in WEB_URLS:
        text = fetch_webpage_text(url)
        text = normalize_text(text)

        if not text:
            continue

        chunks = chunk_text(text, chunk_size=500, overlap=100)

        for i, chunk in enumerate(chunks):
         safe_name = url.replace("https://", "").replace("http://", "").replace("/", "_")
         all_chunks.append({
        "chunk_id": f"{safe_name}_chunk_{i}",
        "text": chunk,
        "source": url
    })

    if not all_chunks:
        return {"status": "error", "message": "No valid content found."}

    texts = [item["text"] for item in all_chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump({
            "metadata": all_chunks,
            "model_name": MODEL_NAME
        }, f)

    return {
        "status": "success",
        "chunks_added": len(all_chunks)
    }

if __name__ == "__main__":
    result = ingest_documents()
    print(result)