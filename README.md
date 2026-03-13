# Historical Figures RAG Chatbot

A **Retrieval-Augmented Generation (RAG) based AI chatbot** that answers questions about historical figures using a structured PDF knowledge base.
The system combines **Large Language Models (LLMs)** with **vector search retrieval** to generate accurate, context-aware responses.

Instead of relying purely on the model’s internal training data, this system retrieves relevant information from a custom knowledge base before generating answers, improving factual accuracy and reducing hallucinations.

---

# Project Overview

This project demonstrates how modern AI systems integrate **LLMs with external knowledge sources** through Retrieval-Augmented Generation.

The chatbot processes a PDF containing historical information, converts the text into embeddings, stores them in a vector database, and retrieves the most relevant content when a user asks a question.

The retrieved context is then passed to the language model to generate a precise response.

This architecture mirrors how many **production AI assistants and enterprise AI systems** operate.

---

# Key Features

• Retrieval-Augmented Generation (RAG) pipeline
• PDF document ingestion and processing
• Semantic search using vector embeddings
• Context-aware question answering
• Conversational interface using Gradio
• Local LLM inference using Ollama
• Vector database storage using ChromaDB

---

# System Architecture


<p align="center">
User Query<br>
↓<br>
Embedding-based Retrieval<br>
↓<br>
Relevant Context from Vector Database<br>
↓<br>
Prompt Construction<br>
↓<br>
LLM Response Generation
</p>

The pipeline ensures the language model generates answers **grounded in external knowledge** rather than relying solely on its internal parameters.

---

# Outputs

<img width="1919" height="869" alt="image" src="https://github.com/user-attachments/assets/b01e7fda-72db-440d-b9f7-3f5b2ac7d204" />

<img width="1919" height="869" alt="image" src="https://github.com/user-attachments/assets/4097dcb1-0dec-46b4-8521-7a36e6a8cedf" />

---

# Technologies Used

Python – Core programming language

LangChain – Framework for building LLM-powered applications

Ollama – Local inference engine for running LLMs

ChromaDB – Vector database for storing embeddings

Gradio – Interactive UI for the chatbot interface

PyPDFLoader – PDF document processing

CharacterTextSplitter – Chunking documents for retrieval

---

# Project Structure

```
historical-figures-rag-chatbot
│
├── history_chatbot.py        # Main application script
├── history_figures.pdf       # Knowledge base used for retrieval
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Files ignored by version control
```

---

# How the System Works

### 1. Document Loading

The system loads a PDF containing information about historical figures.

### 2. Text Chunking

The document is split into smaller chunks to improve retrieval accuracy.

### 3. Embedding Generation

Each chunk is converted into vector embeddings using Ollama embedding models.

### 4. Vector Database Storage

Embeddings are stored in ChromaDB to enable semantic search.

### 5. Query Processing

When the user asks a question, the system converts the query into embeddings and retrieves the most relevant document chunks.

## 6. Answer Generation

The retrieved context is passed to the language model which generates a final response.

---

# Installation

Clone the repository

```
git clone https://github.com/AnshulMane/historical-figures-rag-chatbot
```

Navigate to the project directory

```
cd historical-figures-rag-chatbot
```

Create a virtual environment

```
python -m venv myenv
```

Activate the environment

Windows

```
myenv\Scripts\activate
```

Install dependencies

```
pip install -r requirements.txt
```

---

# Model Setup

Pull required models using Ollama

```
ollama pull llama3
ollama pull granite-embedding:latest
```

These models will be used for:

• Language generation
• Vector embedding creation

---

# Running the Application

Start the chatbot

```
python history_chatbot.py
```

The Gradio interface will launch locally in your browser.

You can then interact with the chatbot and ask questions about historical figures.

---

# Example Questions

Who was Alexander the Great?

What were the major achievements of Napoleon Bonaparte?

Tell me about the life of Mahatma Gandhi.

When did Julius Caesar rule Rome?

---

# Why Retrieval-Augmented Generation?

Traditional LLM systems suffer from:

• Hallucinations
• Outdated knowledge
• Lack of domain-specific information

RAG addresses these limitations by allowing the model to **retrieve information from external knowledge sources before generating answers**, resulting in:

• Higher factual accuracy
• More relevant responses
• Domain-specific knowledge support

This architecture is widely used in modern AI applications including:

• Enterprise AI assistants
• Knowledge management systems
• AI research copilots
• Document question answering systems

---

# Author

Anshul Mane

