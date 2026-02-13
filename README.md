# 🧠 RAG Conversational Agent (FastAPI + OpenAI + Pinecone)

This project implements a Retrieval-Augmented Generation (RAG)
conversational agent capable of answering user questions based only on
ingested PDF documents.

The system combines: - Information retrieval (vector search) - LLM text
generation - Web interface chat

Goal: Provide accurate answers grounded in company documents while
avoiding hallucinations.

------------------------------------------------------------------------

## 🏗 Architecture

3-tier architecture:

Frontend → FastAPI Backend → Vector DB + LLM

-   Frontend: HTML/CSS/JS + Streamlit
-   Backend: FastAPI API
-   Data Layer: SQLite + Pinecone + PDF storage

------------------------------------------------------------------------

## ⚙️ Technologies

-   FastAPI
-   OpenAI API
-   Pinecone Vector Database
-   SQLite
-   Streamlit
-   HTML / CSS / JavaScript
-   JWT Authentication

------------------------------------------------------------------------

## 🔎 How RAG Works

1.  User uploads PDF
2.  Text extracted and split into chunks
3.  Embeddings generated
4.  Stored in Pinecone
5.  User asks a question
6.  Relevant chunks retrieved
7.  LLM generates contextual answer

------------------------------------------------------------------------

## 📁 Project Files

-   app.py → Streamlit chat interface
-   preflight_rag.py → Environment & API tests
-   inspect_db.py → Inspect SQLite schema
-   requirements.txt → Dependencies

------------------------------------------------------------------------

## 🚀 Run the Project

### Install dependencies

pip install -r requirements.txt

### Run backend

uvicorn src.api:app --reload --port 8000

Swagger: http://127.0.0.1:8000/docs

### Run Streamlit UI

streamlit run app.py

------------------------------------------------------------------------

## 🔐 Features

-   User authentication (JWT)
-   Conversation history
-   PDF ingestion
-   Vector search retrieval
-   Contextual AI responses

------------------------------------------------------------------------

## 👨‍💻 Author

Ismail Boulaich

Academic internship project
