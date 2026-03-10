import os
import pickle
from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from text_utils import normalize_text, correct_spelling

STORE_DIR = "data/store"
INDEX_PATH = os.path.join(STORE_DIR, "qtest_index.faiss")
META_PATH = os.path.join(STORE_DIR, "metadata.pkl")

index = None
metadata = None
model = None

GREETING_WORDS = [
    "hi", "hello", "hey", "hii", "helo", "hy", "hyy", "hai"
]

TRAINING_WORDS = [
    "course", "courses", "class", "classes",
    "training", "trainings", "program", "programs",
    "learn", "study", "studies", "syllabus"
]

CONTACT_WORDS = [
    "contact", "phone", "email", "mail", "call", "reach", "number"
]

LOCATION_WORDS = [
    "location", "address", "place", "where", "located", "locate"
]

SERVICE_WORDS = [
    "service", "services", "offer", "offers",
    "provide", "provides", "providing"
]


def load_vector_store():
    global index, metadata, model

    if index is not None and metadata is not None and model is not None:
        return index, metadata, model

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError("FAISS index not found. Run ingest.py first.")

    if not os.path.exists(META_PATH):
        raise FileNotFoundError("Metadata file not found. Run ingest.py first.")

    index = faiss.read_index(INDEX_PATH)

    with open(META_PATH, "rb") as f:
        store = pickle.load(f)

    metadata = store["metadata"]
    model_name = store["model_name"]
    model = SentenceTransformer(model_name)

    return index, metadata, model


def contains_any(text: str, words: List[str]) -> bool:
    text = f" {text} "
    return any(f" {word} " in text for word in words)


def is_greeting(text: str) -> bool:
    words = text.split()
    if not words:
        return False

    if len(words) <= 3:
        return any(word in GREETING_WORDS for word in words)

    return False


def format_answer(text: str) -> str:
    text = text.strip()

    text = text.replace(
        "qtest solutions offers professional software testing services.",
        "QTest Solutions offers these software testing services:"
    )
    text = text.replace(
        "qtest solutions provides software testing training programs.",
        "QTest Solutions provides software testing training programs."
    )
    text = text.replace(
        "qtest solutions contact information.",
        "QTest Solutions contact information."
    )

    text = text.replace("services include:", "\nServices include:")
    text = text.replace("training programs include:", "\nAvailable training areas:")
    text = text.replace("available training areas include:", "\nAvailable training areas:")
    text = text.replace("office address:", "\nOffice Address:")
    text = text.replace("email:", "\nEmail:")
    text = text.replace("phone:", "\nPhone:")

    text = text.replace(" - ", "\n- ")
    text = text.replace(". ", ".\n")

    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")

    return text.strip()


def keyword_overlap_score(query: str, text: str) -> int:
    query_words = set(normalize_text(query).split())
    text_words = set(normalize_text(text).split())
    return len(query_words & text_words)


def search_documents(query: str, top_k: int = 5):
    corrected_query = correct_spelling(query)
    cleaned_query = normalize_text(corrected_query)

    if not cleaned_query:
        return []

    index, metadata, model = load_vector_store()

    query_embedding = model.encode([cleaned_query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype="float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        chunk_text = metadata[idx]["text"]

        results.append({
            "score": float(score),
            "keyword_score": keyword_overlap_score(cleaned_query, chunk_text),
            "text": chunk_text,
            "source": metadata[idx]["source"],
            "chunk_id": metadata[idx]["chunk_id"]
        })

    results.sort(key=lambda x: (x["keyword_score"], x["score"]), reverse=True)
    return results


def get_answer(query: str) -> str:
    original_query = (query or "").strip()
    cleaned_query = normalize_text(original_query)
    corrected_query = normalize_text(correct_spelling(cleaned_query))

    if not cleaned_query:
        return "Please type your question."

    # Greeting handling
    if is_greeting(corrected_query):
        return (
            "Hello 👋\n\n"
            "I am the QTest Solutions assistant.\n\n"
            "You can ask me about:\n"
            "- Services\n"
            "- Training programs\n"
            "- Location\n"
            "- Contact details"
        )

    # Rule-based fast answers
    if contains_any(corrected_query, CONTACT_WORDS):
        return (
            "You can contact QTest Solutions through:\n\n"
            "Email: info@qtestsolutions.com\n"
            "Phone: +91 9961544424"
        )

    if contains_any(corrected_query, LOCATION_WORDS):
        return (
            "QTest Solutions is located at:\n\n"
            "4th Floor, Emerald Mall\n"
            "Mavoor Road\n"
            "Kozhikode, Kerala, India"
        )

    if contains_any(corrected_query, TRAINING_WORDS):
        return (
            "QTest Solutions provides software testing training programs.\n\n"
            "Available training areas include:\n"
            "- Manual Testing\n"
            "- Automation Testing\n"
            "- Selenium Testing\n"
            "- API Testing\n"
            "- Software Testing Fundamentals"
        )

    if contains_any(corrected_query, SERVICE_WORDS):
        return (
            "QTest Solutions offers these software testing services:\n\n"
            "- Manual Testing\n"
            "- Automation Testing\n"
            "- Regression Testing\n"
            "- API Testing\n"
            "- Performance Testing\n"
            "- Load Testing\n"
            "- Security Testing"
        )

    # Vector search fallback
    results = search_documents(original_query, top_k=5)

    if not results:
        return (
            "Sorry, I couldn't find that information.\n\n"
            "Please ask about services, training, location, or contact details."
        )

    best = results[0]

    if best["score"] < 0.18 and best["keyword_score"] == 0:
        return (
            "Sorry, I couldn't clearly understand that question.\n\n"
            "Please try asking in another way."
        )

    return format_answer(best["text"])