import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from institute_data import COURSES, SYLLABUS, INFO

# Multilingual model (English + Malayalam + Manglish)
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(MODEL_NAME)

# --- Create Q/A pairs (questions -> answer_key) ---
QUESTIONS = []
Q_TO_KEY = []

ANSWERS = {
    "manual_fee": f"✅ {COURSES['manual']['name']} fee is ₹{COURSES['manual']['fee']} and duration is {COURSES['manual']['duration']}.",
    "selenium_fee": f"✅ {COURSES['selenium']['name']} fee is ₹{COURSES['selenium']['fee']} and duration is {COURSES['selenium']['duration']}.",
    "combined_fee": f"✅ {COURSES['combined']['name']} fee is ₹{COURSES['combined']['fee']} and duration is {COURSES['combined']['duration']}.",

    "manual_syllabus": "✅ Manual syllabus:\n• " + "\n• ".join(SYLLABUS["manual"]),
    "selenium_syllabus": "✅ Selenium syllabus:\n• " + "\n• ".join(SYLLABUS["selenium"]),
    "combined_syllabus": "✅ Master syllabus:\n• " + "\n• ".join(SYLLABUS["combined"]),

    "demo": INFO["demo"],
    "placement": INFO["placement"],
    "certificate": INFO["certificate"],
    "online_offline": INFO["online_offline"],
    "location": INFO["location"],
}

# Fee questions
manual_q = ["manual fee", "manual course fee", "manual testing fee", "manual fee ethra", "മാനുവൽ ഫീസ് എത്ര", "മാനുവൽ കോഴ്സ് ഫീസ്"]
selenium_q = ["selenium fee", "automation fee", "java selenium fee", "selenium fee ethra", "ഓട്ടോമേഷൻ ഫീസ് എത്ര", "സെലെനിയം ഫീസ് എത്ര"]
combined_q = ["master fee", "combined fee", "both fee", "master fee ethra", "രണ്ടും ഫീസ് എത്ര", "മാസ്റ്റർ കോഴ്സ് ഫീസ്"]

for q in manual_q:
    QUESTIONS.append(q); Q_TO_KEY.append("manual_fee")
for q in selenium_q:
    QUESTIONS.append(q); Q_TO_KEY.append("selenium_fee")
for q in combined_q:
    QUESTIONS.append(q); Q_TO_KEY.append("combined_fee")

# Syllabus questions
for q in ["manual syllabus", "manual topics", "manual syllabus enthokke", "മാനുവൽ സിലബസ്"]:
    QUESTIONS.append(q); Q_TO_KEY.append("manual_syllabus")
for q in ["selenium syllabus", "automation syllabus", "selenium topics", "സെലെനിയം സിലബസ്"]:
    QUESTIONS.append(q); Q_TO_KEY.append("selenium_syllabus")
for q in ["master syllabus", "combined syllabus", "master topics", "മാസ്റ്റർ സിലബസ്"]:
    QUESTIONS.append(q); Q_TO_KEY.append("combined_syllabus")

# Info questions
for q in ["demo", "demo class", "free demo", "ഡെമോ ക്ലാസ് ഉണ്ടോ"]:
    QUESTIONS.append(q); Q_TO_KEY.append("demo")
for q in ["placement", "placement assistance", "job assistance", "പ്ലേസ്മെന്റ് ഉണ്ടോ"]:
    QUESTIONS.append(q); Q_TO_KEY.append("placement")
for q in ["certificate", "course certificate", "സർട്ടിഫിക്കറ്റ് കിട്ടുമോ"]:
    QUESTIONS.append(q); Q_TO_KEY.append("certificate")
for q in ["online", "offline", "online class", "offline class", "night batch", "sunday batch"]:
    QUESTIONS.append(q); Q_TO_KEY.append("online_offline")
for q in ["location", "address", "contact", "phone number", "evide", "എവിടെ", "kozhikode address"]:
    QUESTIONS.append(q); Q_TO_KEY.append("location")

# --- Build FAISS index ---
embeddings = model.encode(QUESTIONS, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "qtest_index.faiss")

with open("qa_store.pkl", "wb") as f:
    pickle.dump({"QUESTIONS": QUESTIONS, "Q_TO_KEY": Q_TO_KEY, "ANSWERS": ANSWERS, "MODEL_NAME": MODEL_NAME}, f)

print("✅ Vector store created successfully!")