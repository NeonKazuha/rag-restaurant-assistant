# RAG Restaurant Assistant Chatbot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) chatbot designed to answer user questions about restaurants based on a provided knowledge base (`knowledgebase.json`). It leverages semantic search (Sentence Transformers + FAISS) to find relevant information about menu items and restaurant details, and uses the Google Gemini API (`gemini-1.5-flash-latest`) to synthesize natural language answers based solely on the retrieved context.

The chatbot can be interacted with via a Streamlit web interface (`app.py`) or a command-line interface (`run.py`).

## Features

*   **Natural Language Queries:** Ask questions about dishes, prices, restaurants, ratings, dietary options, etc., in plain English.
*   **RAG Implementation:** Retrieves relevant text chunks from the knowledge base using semantic vector search (Sentence Transformers + FAISS).
*   **LLM-Powered Generation:** Uses the Google Gemini API (`gemini-1.5-flash-latest`) to generate accurate and context-aware answers based on the retrieved information.
*   **Web Interface:** An interactive chat interface built with Streamlit (`app.py`).
*   **Console Interface:** A simple command-line version for testing (`run.py`).
*   **(Optional) Query Parsing:** Includes a basic regex-based parser (`query_planner.py`) to identify specific query types (e.g., comparisons, price lookups) for potential specialized handling or logging.

## Technology Stack

*   **Language:** Python (3.10 / 3.11 Recommended)
*   **Core Libraries:**
    *   `google-generativeai`: For interacting with the Gemini API.
    *   `sentence-transformers`: For generating text embeddings (`all-MiniLM-L6-v2`).
    *   `faiss-cpu` / `faiss-gpu`: For efficient vector similarity search.
    *   `numpy`: For numerical operations.
    *   `python-dotenv`: For managing API keys via `.env` files.
    *   `torch`: Backend dependency for sentence-transformers.
*   **Web Framework:** `streamlit` (for `app.py`).
*   **Data Handling:** Standard Python `json` library.

## System Architecture

The system employs a RAG pipeline:

1.  **Offline Processing:**
    *   Load restaurant data from `knowledgebase.json` (`fetch_restaurant.py`).
    *   Chunk data into individual menu item descriptions, embedding restaurant/item metadata (`chunking.py`).
    *   Generate vector embeddings for each chunk using `SentenceTransformer` (`index_faiss.py`).
    *   Build a FAISS index for efficient vector search (`index_faiss.py`).
2.  **Online Querying:**
    *   Receive user query via UI (Streamlit/Console - `app.py`/`run.py`).
    *   **(Optional)** Parse the query using `query_planner.py` to identify intent or specific entities.
    *   Embed the user query using the same `SentenceTransformer` (`chatbot.py`).
    *   Retrieve top-k relevant chunks from the FAISS index based on semantic similarity (`index_faiss.py`).
    *   Format retrieved chunks and the original query into a prompt suitable for the Gemini API (`chatbot.py`).
    *   Call the Google Gemini API (`answer` function in `chatbot.py`) to generate a natural language answer based *only* on the provided context.
    *   Display the answer to the user (`app.py` / `run.py`).

Folder StructureRAG-RESTAURANT-ASSISTANT/
│
├── data/
│   ├── processed/
│   │   └── knowledgebase.json   # Cleaned/structured data used by the app
│   └── raw/
│       └── restaurant_data.json # (Optional) Original raw scraped data
│
├── rag-pipeline/             # Core chatbot logic and interface
│   ├── __pycache__/          # Python cache files (usually ignored)
│   ├── app.py                # Streamlit web application
│   ├── chatbot.py            # Main chatbot logic (RAG + Gemini call)
│   ├── chunking.py           # Data chunking logic
│   ├── fetch_restaurant.py   # Utility to load knowledgebase.json
│   ├── index_faiss.py        # Embedding generation and FAISS index logic
│   ├── query_planner.py      # (Optional) Query parsing/logging
│   └── run.py                # Command-line interface
│
├── scraper/                  # (Optional) Code used for data collection
│   ├── extraction.py         # Logic for extracting specific data from scraped content
│   ├── fetch_websites.py     # Logic for fetching website content
│   ├── process.py            # Logic for processing extracted data (e.g., feature extraction)
│   ├── utils.py              # Utility functions for the scraper
│   └── web_scraper.py        # Main script orchestrating the web scraping process
│
├── .env                      # Stores environment variables (like API keys - DO NOT COMMIT)
├── .gitignore                # Specifies intentionally untracked files for Git
├── LICENSE                   # Project license file
├── README.md                 # This file
└── requirements.txt          # Python package dependencies

### File Descriptions

*   **`data/processed/knowledgebase.json`**: The main structured dataset containing restaurant names, menus, attributes, ratings, etc., used by the RAG pipeline.
*   **`data/raw/restaurant_data.json`**: (Optional) Could store the original, less processed data before cleaning/structuring into `knowledgebase.json`.
*   **`rag-pipeline/app.py`**: Implements the user-friendly web interface using Streamlit. Handles user input, calls the chatbot logic, and displays the conversation.
*   **`rag-pipeline/chatbot.py`**: Contains the core `answer` function (or potentially `answer_with_rag_gemini` if refactored). It orchestrates the RAG process: embedding the query, retrieving chunks from FAISS, formatting the prompt, calling the Gemini API, and returning the response.
*   **`rag-pipeline/chunking.py`**: Defines how the raw JSON data is broken down into smaller text chunks suitable for embedding and retrieval. Includes logic for adding relevant metadata to each chunk.
*   **`rag-pipeline/fetch_restaurant.py`**: A simple utility function to load the `knowledgebase.json` file into a Python object.
*   **`rag-pipeline/index_faiss.py`**: Handles the creation of vector embeddings using SentenceTransformer and builds/searches the FAISS vector index. Contains the `create_faiss_index` function and logic for retrieving chunks.
*   **`rag-pipeline/query_planner.py`**: (Currently Optional/Simplified) Intended for parsing user queries to detect specific intents or extract parameters (like price comparisons, specific dishes). In the current simplified RAG-only approach, it might only be used for logging or pre-filtering.
*   **`rag-pipeline/run.py`**: Provides a basic command-line interface to interact with the chatbot, useful for testing and debugging.
*   **`scraper/extraction.py`**: (Optional) Contains functions specifically designed to extract structured information (like menu items, prices, descriptions) from the raw HTML or content fetched by the scraper.
*   **`scraper/fetch_websites.py`**: (Optional) Includes code responsible for downloading the content (HTML, JSON, etc.) from target restaurant websites or data sources.
*   **`scraper/process.py`**: (Optional) Contains scripts or functions for cleaning, transforming, and potentially enriching the extracted raw data (e.g., calculating features like spice level, identifying dietary tags) before it's saved in a structured format like `knowledgebase.json`.
*   **`scraper/utils.py`**: (Optional) Holds common utility functions used across different parts of the scraping process (e.g., handling requests, parsing helpers, logging setup).
*   **`scraper/web_scraper.py`**: (Optional) The main script or entry point for the data collection process, likely coordinating the fetching, extraction, and processing steps.
*   **`.env`**: Stores sensitive information like the `GEMINI_API_KEY` outside of the main codebase. Must not be committed to Git.
*   **`.gitignore`**: Lists files and directories (like `__pycache__`, `venv`, `.env`) that Git should ignore.
*   **`LICENSE`**: Contains the software license under which the project is distributed.
*   **`README.md`**: This file, providing documentation and instructions for the project.
*   **`requirements.txt`**: Lists all the Python libraries required to run the project, allowing for easy installation using `pip install -r requirements.txt`.

### Setup Instructions

**Prerequisites:**

*   Git
*   Python (3.10 or 3.11 recommended) & Pip

**Steps:**

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd RAG-RESTAURANT-ASSISTANT
    ```
2.  **Create and activate a virtual environment:**
    *   Linux/macOS:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   Windows:
        ```bash
        python -m venv venv
        venv\Scripts\activate.bat
        ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Create and configure `.env` file:**
    *   Create a file named `.env` in the project root directory (`RAG-RESTAURANT-ASSISTANT/`).
    *   Add your Google Gemini API key to it:
        ```
        GEMINI_API_KEY=YOUR_ACTUAL_API_KEY_HERE
        ```
    *   **Important:** Add `.env` to your `.gitignore` file to prevent accidentally committing your API key.

5.  **Place the knowledge base:**
    *   Ensure your `knowledgebase.json` file is located at `data/processed/knowledgebase.json`.
    *   Adjust the path in `rag-pipeline/app.py` (variable `DATA_PATH`) and potentially `rag-pipeline/fetch_restaurant.py` if your file is located elsewhere.

### Running the Application

1.  Activate your virtual environment (if not already active).
2.  Navigate to the `rag-pipeline` directory:
    ```bash
    cd rag-pipeline
    ```
3.  Choose an interface:
    *   **Streamlit Web App:**
        ```bash
        streamlit run app.py
        ```
        (If you encounter file watcher issues, try: `streamlit run app.py --server.fileWatcherType none`)
    *   **Console App:**
        ```bash
        python run.py
        ```
        (Note: Ensure `run.py` is updated to use the correct chatbot function, e.g., `answer`, and handles dependencies correctly as shown in `app.py`).

### Challenges Faced

*   **Initial RAG Inaccuracy:** Early attempts using a simpler RAG setup resulted in factually incorrect or irrelevant answers, prompting investigation into retrieval quality and consideration of alternative vector databases (like Qdrant).
*   **Implementing Specific Features:** Adding functionalities beyond basic Q&A (like price comparisons, rating comparisons, spice comparisons) required developing a hybrid approach with a regex-based query parser (`query_planner.py`) and specific logic branches in `chatbot.py`.
*   **Hybrid Logic Complexity & Errors:** The hybrid approach, while powerful, introduced complexity and became prone to errors, leading to the decision to simplify and rely more heavily on the LLM's capabilities with well-retrieved context.
*   **Data Schema Adaptation:** The code required significant adjustments to match the actual structure of `knowledgebase.json`, including handling string-based ratings, using the `veg_nonveg` field, and managing potentially missing or inconsistent data (prices, attributes).
*   **Environment/Compatibility:** Runtime errors occurred, including `TypeError` due to function signature mismatches during refactoring and a `RuntimeError: no running event loop` potentially caused by Python 3.13 incompatibility with Streamlit/PyTorch, necessitating environment adjustments or workarounds (like disabling the file watcher).
*   **API Integration & Key Management:** Switching from a local T5 model to the Gemini API required integrating the `google-generativeai` library and implementing secure API key management using `python-dotenv` and environment variables.
*   **Retrieval Tuning:** Finding the right number of chunks (`k`) to retrieve involved balancing providing enough context for the LLM versus overwhelming it with potentially irrelevant information.
*   **Model Loading Times:** Initializing the embedding model (SentenceTransformer) could impact application startup time, partially mitigated using Streamlit's caching (`@st.cache_resource`).

### Future Improvements

*   **Advanced Retrieval:** Implement strategies like Parent Document Retriever or re-ranking for better context relevance.
*   **Vector Database:** Migrate from FAISS to a dedicated vector DB (Qdrant, Milvus) for scalability and advanced filtering.
*   **Model Experimentation:** Test different embedding models or newer/larger Gemini models (e.g., Gemini 1.5 Pro).
*   **Data Quality:** Enhance `knowledgebase.json` with more structured fields (cuisine, specific dietary tags) and ensure data consistency.
*   **Conversational Memory:** Add context from previous turns for follow-up questions.
*   **Evaluation:** Implement a framework (e.g., RAGAs) to systematically evaluate retrieval and generation quality.
*   **Error Handling:** Improve robustness and provide more informative user feedback on errors.

**(For a visual flow diagram, refer to the Mermaid diagram below or in separate technical documentation).**

```mermaid
sequenceDiagram
    participant User
    participant UI (app.py / run.py)
    participant Chatbot (chatbot.py)
    participant QueryPlanner (query_planner.py)
    participant Index (index_faiss.py / FAISS)
    participant Data (knowledgebase.json / chunks list)
    participant Embedder (SentenceTransformer)
    participant Generator (Gemini API)

    %% Initialization Steps (Conceptual)
    Note over Data, Index: Load Data (fetch_restaurant.py)
    Note over Data, Index: Chunk Data (chunking.py)
    Note over Data, Index: Embed Chunks (index_faiss.py)
    Note over Data, Index: Build FAISS Index (index_faiss.py)

    %% Runtime Query Flow (RAG + Gemini)
    User->>+UI: Asks question
    UI->>+Chatbot: answer_with_rag_gemini(question, ...)
    Chatbot->>+QueryPlanner: parse_query(question) # Optional: for logging/pre-filtering
    QueryPlanner-->>-Chatbot: Return spec (optional use)
    Chatbot->>+Embedder: embed(question)
    Embedder-->>-Chatbot: Query Vector
    Chatbot->>+Index: retrieve_chunks(Query Vector, k)
    Index-->>-Chatbot: Top-k relevant chunks
    Chatbot->>Chatbot: Formulate context from chunk text
    Chatbot->>+Generator: generate_content(prompt with context & question)
    Generator-->>-Chatbot: Synthesized Answer
    Chatbot-->>-UI: Return generated answer
    UI-->>-User: Display answer
