Certainly! Below is a detailed `README.md` file for your project. This file explains the setup and usage of your chatbot application, which utilizes FastAPI and various AI models for conversation and retrieval.

---

# Chatbot Application

## Overview
This chatbot application integrates AI models from OpenAI, ChatCohere, and HuggingFace FLAN, providing an interactive interface for users to ask questions and receive responses. It utilizes FastAPI for the web framework and FAISS for efficient similarity search in large datasets.

## Features
- **AI Chat Models:** Utilizes models like GPT-3.5-turbo (OpenAI) and ChatCohere for generating responses.
- **Conversational Context:** Maintains a conversation history to provide context-aware responses.
- **Efficient Search:** Uses FAISS (Facebook AI Similarity Search) for quick retrieval of relevant information.
- **Customizable Prompts:** Supports customizable prompt templates for different chat scenarios.

## Requirements
- Python 3.8+
- FastAPI
- Uvicorn (for running the server)
- Various libraries from `langchain`
- dotenv, FAISS, HuggingFace libraries

## Installation
1. **Clone the Repository:**
   ```sh
   git clone https://github.com/JonOlummy/firesafetybot
   cd [repository-directory]
   ```

2. **Install Dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Environment Setup:**
   - Create a `.env` file in the root directory.
   - Add your HuggingFace API token: `HUGGINGFACE_TOKEN=your_token_here`.

## Usage
1. **Start the Server:**
   ```sh
   uvicorn main:app --reload
   ```

2. **Interacting with the Chatbot:**
   - Send a POST request to `/ask` with a JSON payload containing the question.
   - Retrieve chat history by sending a GET request to `/chat/history`.

3. **Building the FAISS Index:**
   - Run the `create_vector_db()` function from `ingest.py` to build the FAISS index with your desired documents.

## API Reference
- `POST /ask`: Endpoint to ask a question to the chatbot. Expects a payload with the question.
- `GET /chat/history`: Endpoint to retrieve the history of the conversation.
