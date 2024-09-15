---
title: Rag Mini Wikipedia Demo
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: apache-2.0
---

# RAG Project: Enhanced Information Retrieval for LLMs
This project demonstrates a Retrieval-Augmented Generation (RAG) application for improved question answering using 
Large Language Models (LLMs). RAG overcomes LLM limitations by enabling access to specific information not included 
in their training data.

## Key Features:
* Uses `rag-mini-wikipedia` dataset for factual information retrieval.
* Employs `all-MiniLM-L6-v2` for sentence encoding and `FAISS` for efficient similarity search.
* Leverages `meta-llama/Llama-2-7b-chat-hf` model for response generation.
* Built with Gradio 4.38.1 (a user interface library for machine learning).

## Benefits:
* More informative responses by incorporating external knowledge.
* Ideal for applications like smart Q&A chatbots in corporate knowledge bases.

## Running the Application:
1. Ensure you have the required libraries installed (refer to the project's requirements).
2. Open a terminal and navigate to the project directory.
3. Login to your Hugging Face account with appropriate token:
```bash
huggingface-cli login
```
4. Run the application using the following command:
```bash
python app.py
```
This will launch the Gradio interface where you can interact with the RAG model.

## Further details:
Research notebook in the 'Research' folder explores chunking, prompt development, and future directions. Read it !