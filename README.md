# Institute RAG Chatbot

## Overview
This project is an AI-powered chatbot that answers user queries based on institute-related data using Retrieval-Augmented Generation (RAG). It retrieves relevant information from stored documents and generates accurate, context-aware responses.

## Features
- Semantic search using FAISS
- Context-based answers using Sentence Transformers
- Supports custom documents (PDF, text)
- Fast and accurate response generation
- Backend API using Flask

## Tech Stack
- Python
- Flask
- FAISS (Vector Database)
- Sentence Transformers

## How it Works
1. User asks a question
2. Question is converted into embeddings
3. FAISS retrieves relevant data
4. System generates accurate response using context

👉 RAG works by retrieving relevant data first and then generating answers based on that data :contentReference[oaicite:0]{index=0}

## Setup
```bash
git clone https://github.com/shereef888k/institute-rag-chatbot
cd institute-rag-chatbot
pip install -r requirements.txt
python app.py
