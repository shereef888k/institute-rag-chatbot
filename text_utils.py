import re
from rapidfuzz import process

COMMON_TERMS = [
    "hi", "hello", "hey", "hii", "helo", "hy", "hyy", "hai",
    "service", "services",
    "training", "trainings",
    "program", "programs",
    "course", "courses",
    "class", "classes",
    "manual", "automation", "selenium",
    "testing", "api", "performance", "security", "regression",
    "location", "address", "place", "located", "locate", "where",
    "contact", "phone", "email", "mail", "call", "reach", "number",
    "qtest", "solutions",
    "what", "how", "do", "you", "provide", "provides", "offering", "offer", "offers",
    "have", "has", "can", "please"
]


def normalize_text(text: str) -> str:
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s@.+-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def correct_spelling(text: str) -> str:
    words = normalize_text(text).split()
    corrected = []

    for word in words:
        # keep very short words as they are
        if len(word) <= 2:
            corrected.append(word)
            continue

        # don't try to "correct" emails or phone-like values
        if "@" in word or any(ch.isdigit() for ch in word):
            corrected.append(word)
            continue

        match = process.extractOne(word, COMMON_TERMS, score_cutoff=60)

        if match:
            corrected.append(match[0])
        else:
            corrected.append(word)

    text = " ".join(corrected)

    # synonym normalization with whole words only
    text = re.sub(r"\bcourse\b", "training", text)
    text = re.sub(r"\bcourses\b", "training", text)
    text = re.sub(r"\bclass\b", "training", text)
    text = re.sub(r"\bclasses\b", "training", text)

    return text


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    words = text.split()
    chunks = []

    if not words:
        return chunks

    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        chunk = " ".join(chunk_words).strip()

        if chunk:
            chunks.append(chunk)

    return chunks