# 📺 YouTube Video RAG Chatbot

An interactive web application built with **Streamlit**, **LangChain**, and **Google Gemini 2.5 Flash** that allows users to chat with any YouTube video using its transcript.

## 🚀 Features
- **Dynamic URL Parsing:** Supports standard watch links, short links, and embeds.
- **Semantic Search:** Uses HuggingFace embeddings (`all-MiniLM-L6-v2`) and FAISS for fast retrieval.
- **LLM Intelligence:** Powered by Google's `gemini-2.5-flash` model.
- **Caching:** Optimizes processing by caching vector embeddings per video.

## 🛠️ Tech Stack
- Python, Streamlit, LangChain, Google GenAI API, FAISS, Hugging Face.