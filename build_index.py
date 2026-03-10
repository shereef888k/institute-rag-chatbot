import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

with open("qa_store.pkl", "rb") as f:
    store = pickle.load(f)

QUESTIONS = store["QUESTIONS"]
Q_TO_KEY = store["Q_TO_KEY"]
ANSWERS = store["ANSWERS"]

model = SentenceTransformer(MODEL_NAME)

# normalized embeddings = cosine similarity with IndexFlatIP
emb = model.encode(QUESTIONS, normalize_embeddings=True).astype("float32")

dim = emb.shape[1]
index = faiss.IndexFlatIP(dim)   # IP on normalized vectors == cosine
index.add(emb)

faiss.write_index(index, "qtest_index.faiss")

# Save back model name if needed
store["MODEL_NAME"] = MODEL_NAME
with open("qa_store.pkl", "wb") as f:
    pickle.dump(store, f)

print("✅ Cosine FAISS index rebuilt and saved.")