from flask import Flask, request, jsonify
from flask_cors import CORS
from ingest import ingest_documents
from query_engine import get_answer

app = Flask(__name__)
CORS(app)


@app.get("/")
def home():
    return "QTest document RAG server running!"


@app.post("/ingest")
def ingest():
    result = ingest_documents()
    return jsonify(result)


@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")
    reply = get_answer(user_message)
    return jsonify({"reply": reply})


@app.get("/chat")
def chat_info():
    return 'Use POST /chat with JSON: {"message": "your text"}'


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)