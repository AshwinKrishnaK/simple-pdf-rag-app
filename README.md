# Simple RAG Chat Application using GROQ

This project is a Streamlit-based web application that leverages Retrieval-Augmented Generation (RAG) to create a chatbot capable of answering questions based on the context provided in uploaded PDF documents. It integrates GROQ AI for language generation and OpenAI for embeddings to build a powerful context-aware chatbot.

## Features
- **Document Upload**: Users can upload a PDF document, which will be processed and embedded for contextual understanding.
- **Query Input**: Users can input questions, and the chatbot will respond based on the content of the uploaded document.
- **Vector Store Creation**: Uses FAISS to create and store vector embeddings for efficient document retrieval.
- **Language Model**: Utilizes GROQ AI's Gemma2-9B-IT model for generating accurate and context-driven responses.

## Prerequisites

To run this project, ensure you have the following:

1. Python 3.8+
2. The following Python libraries:
    - `streamlit`
    - `langchain`
    - `langchain-community`
    - `langchain-openai`
    - `langchain-groq`
    - `python-dotenv`
    - `faiss-cpu`
3. API keys for:
    - OpenAI (`OPENAI_API_KEY`)
    - GROQ AI (`GROQ_API_KEY`)
4. Environment variables for:
    - `LANGCHAIN_API_KEY`
    - `LANGCHAIN_PROJECT`
    - `LANGCHAIN_TRACING_V2`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-chat-app.git
   cd rag-chat-app
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up the `.env` file with your API keys and other environment variables:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   LANGCHAIN_PROJECT=SIMPLE RAG CHAT APP USING GROQ
   LANGCHAIN_TRACING_V2=true
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the URL displayed in the terminal (default: `http://localhost:8501`) in your browser.

3. Upload a PDF document and click on "Document Embedding" to process the document.

4. Enter your query in the input box, and the chatbot will respond based on the uploaded document's content.

## Code Overview

### Key Components

1. **Environment Setup**:
   - `.env` file for storing API keys and configurations.

2. **Document Processing**:
   - Uses `PyPDFLoader` to read PDF documents.
   - Splits text into chunks using `RecursiveCharacterTextSplitter`.
   - Embeds documents using OpenAI embeddings and stores them in a FAISS vector store.

3. **Chat Functionality**:
   - GROQ AI's `ChatGroq` for generating context-aware responses.
   - RAG pipeline with retriever and document chain for question answering.

4. **Streamlit Interface**:
   - File uploader for PDFs.
   - Text input for user queries.
   - Button to trigger document embedding.

## File Structure
```
├── app.py                  # Main application code
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
└── README.md               # Project documentation
```

## Dependencies

Install the required Python libraries using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

## Acknowledgments

- [LangChain](https://www.langchain.com/) for building the foundational components.
- [GROQ AI](https://groq.com/) for providing the language generation model.
- [Streamlit](https://streamlit.io/) for creating an easy-to-use frontend.
