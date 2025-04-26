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
    *   **(Scraping - Optional)** Fetch website URLs (`scraper/fetch_websites.py`), scrape data (`scraper/web_scraper.py`), extract relevant information (`scraper/extraction.py`), save raw data (`data/raw/restaurant_data.json`).
    *   **(Indexing)** Load structured data (`data/processed/knowledgebase.json` - potentially generated from raw data or provided directly).
    *   Process data, generate embeddings, and build FAISS index (`scraper/process.py` or `rag-pipeline/index_faiss.py` depending on workflow).
2.  **Online Querying:**
    *   Receive user query via UI (Streamlit/Console - `app.py`/`run.py`).
    *   **(Optional)** Parse the query using `query_planner.py` to identify intent or specific entities.
    *   Embed the user query using the same `SentenceTransformer` (`chatbot.py`).
    *   Retrieve top-k relevant chunks from the FAISS index based on semantic similarity (`index_faiss.py` or `chatbot.py`).
    *   Format retrieved chunks and the original query into a prompt suitable for the Gemini API (`chatbot.py`).
    *   Call the Google Gemini API (`answer` function in `chatbot.py`) to generate a natural language answer based *only* on the provided context.
    *   Display the answer to the user (`app.py` / `run.py`).

Folder StructureRAG-RESTAURANT-ASSISTANT/
```
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
│   ├── chunking.py           # Data chunking logic (if separate from indexing)
│   ├── fetch_restaurant.py   # Utility to load knowledgebase.json
│   ├── index_faiss.py        # Embedding generation and FAISS index logic (alternative location)
│   ├── query_planner.py      # (Optional) Query parsing/logging
│   └── run.py                # Command-line interface
│
├── scraper/                  # (Optional) Code used for data collection & preprocessing
│   ├── output/               # Output from scraper scripts
│   │   ├── raw_extracted_data.json # Raw data from extraction.py
│   │   ├── knowledge_base.json     # Processed data (alternative location)
│   │   ├── faiss_index.bin         # FAISS index file
│   │   └── metadata.pkl            # Metadata for FAISS index
│   ├── extraction.py         # Extracts data using the scraper and saves raw output
│   ├── fetch_websites.py     # (Placeholder) Script to fetch website URLs for scraping
│   ├── process.py            # Processes data, creates embeddings and FAISS index
│   ├── utils.py              # (Placeholder) Utility functions for the scraper module
│   └── web_scraper.py        # Core web scraping logic for restaurant data
│
├── .env                      # Stores environment variables (like API keys - DO NOT COMMIT)
├── .gitignore                # Specifies intentionally untracked files for Git
├── LICENSE                   # Project license file
├── README.md                 # This file
└── requirements.txt          # Python package dependencies
```

### File Descriptions

*   **`data/processed/knowledgebase.json`**: The main structured dataset containing restaurant names, menus, attributes, ratings, etc., used by the RAG pipeline.
*   **`data/raw/restaurant_data.json`**: (Optional) Could store the original, less processed data before cleaning/structuring into `knowledgebase.json`. Also see `scraper/output/raw_extracted_data.json`.

*   **`rag-pipeline/app.py`**: Implements the user-friendly web interface using Streamlit. Handles user input, calls the chatbot logic, and displays the conversation.
*   **`rag-pipeline/chatbot.py`**: Contains the core `answer` function. It orchestrates the RAG process: embedding the query, retrieving chunks from FAISS, formatting the prompt, calling the Gemini API, and returning the response.
*   **`rag-pipeline/chunking.py`**: (If used) Defines how the raw JSON data is broken down into smaller text chunks suitable for embedding and retrieval. Includes logic for adding relevant metadata to each chunk. (Note: Chunking logic might be integrated into `scraper/process.py` or `rag-pipeline/index_faiss.py`).
*   **`rag-pipeline/fetch_restaurant.py`**: A simple utility function to load the `knowledgebase.json` file into a Python object.
*   **`rag-pipeline/index_faiss.py`**: (If used for RAG pipeline) Handles the creation of vector embeddings using SentenceTransformer and builds/searches the FAISS vector index. Contains the `create_faiss_index` function and logic for retrieving chunks. (Note: Indexing logic might be handled by `scraper/process.py`).
*   **`rag-pipeline/query_planner.py`**: (Currently Optional/Simplified) Intended for parsing user queries to detect specific intents or extract parameters (like price comparisons, specific dishes). In the current simplified RAG-only approach, it might only be used for logging or pre-filtering.
*   **`rag-pipeline/run.py`**: Provides a basic command-line interface to interact with the chatbot, useful for testing and debugging.

*   **`scraper/extraction.py`**: Orchestrates the data extraction process. Loads website URLs, utilizes `web_scraper.py` to scrape data from each site, and saves the aggregated raw extracted data (e.g., to `scraper/output/raw_extracted_data.json`).
*   **`scraper/fetch_websites.py`**: (Placeholder) Intended script to gather or load the list of restaurant website URLs that need to be scraped.
*   **`scraper/process.py`**: Handles the preprocessing of data (e.g., loading from `raw_extracted_data.json` or `knowledge_base.json`), generates text embeddings for relevant fields (like menu items, descriptions) using SentenceTransformer, builds a FAISS index for vector search, and saves the index and associated metadata (e.g., to `scraper/output/`).
*   **`scraper/utils.py`**: (Placeholder) Intended location for helper functions used across different scripts within the `scraper` module.
*   **`scraper/web_scraper.py`**: Contains the core logic (`RestaurantScraper` class) for scraping data from individual restaurant websites.

*   **`.env`**: Stores sensitive information like the `GEMINI_API_KEY` outside of the main codebase. Must not be committed to Git.
*   **`.gitignore`**: Lists files and directories (like `__pycache__`, `venv`, `.env`, `scraper/output/`) that Git should ignore.
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

5.  **Prepare Data and Index:**
    *   **Option A: Use Pre-processed Data:** Ensure your `knowledgebase.json` file is located at `data/processed/knowledgebase.json`. Ensure the corresponding FAISS index (`faiss_index.bin`) and metadata (`metadata.pkl`) are present (e.g., in `scraper/output/` or wherever `app.py` expects them). Adjust paths in the code if necessary.
    *   **Option B: Run Scraper/Processing Pipeline:**
        *   (If applicable) Prepare website list for `scraper/fetch_websites.py`.
        *   (If applicable) Run `python scraper/extraction.py` to scrape and save raw data.
        *   Run `python scraper/process.py` to generate the `knowledge_base.json` (if needed), embeddings, FAISS index, and metadata. Ensure input/output paths in `scraper/process.py` are correct.
    *   Adjust paths in `rag-pipeline/app.py` (e.g., `DATA_PATH`, paths for index/metadata) and potentially `rag-pipeline/fetch_restaurant.py` or `rag-pipeline/chatbot.py` to point to the correct data, index, and metadata files.

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
        (Note: Ensure `run.py` is updated to use the correct chatbot function, e.g., `answer`, and handles dependencies/paths correctly as shown in `app.py`).

### Challenges Faced

*   **Initial RAG Inaccuracy:** Early attempts using a simpler RAG setup resulted in factually incorrect or irrelevant answers, prompting investigation into retrieval quality and consideration of alternative vector databases (like Qdrant).
*   **Implementing Specific Features:** Adding functionalities beyond basic Q&A (like price comparisons, rating comparisons, spice comparisons) required developing a hybrid approach with a regex-based query parser (`query_planner.py`) and specific logic branches in `chatbot.py`.
*   **Hybrid Logic Complexity & Errors:** The hybrid approach, while powerful, introduced complexity and became prone to errors, leading to the decision to simplify and rely more heavily on the LLM's capabilities with well-retrieved context.
*   **Data Schema Adaptation:** The code required significant adjustments to match the actual structure of `knowledgebase.json`, including handling string-based ratings, using the `veg_nonveg` field, and managing potentially missing or inconsistent data (prices, attributes).
*   **Environment/Compatibility:** Runtime errors occurred, including `TypeError` due to function signature mismatches during refactoring and a `RuntimeError: no running event loop` potentially caused by Python 3.13 incompatibility with Streamlit/PyTorch, necessitating environment adjustments or workarounds (like disabling the file watcher).
*   **API Integration & Key Management:** Switching from a local T5 model to the Gemini API required integrating the `google-generativeai` library and implementing secure API key management using `python-dotenv` and environment variables.
*   **Retrieval Tuning:** Finding the right number of chunks (`k`) to retrieve involved balancing providing enough context for the LLM versus overwhelming it with potentially irrelevant information.
*   **Model Loading Times:** Initializing the embedding model (SentenceTransformer) could impact application startup time, partially mitigated using Streamlit's caching (`@st.cache_resource`).
*   **Data Pipeline Management:** Coordinating the steps from scraping (`extraction.py`) to processing/indexing (`process.py`) and ensuring the RAG application (`app.py`) uses the correct, up-to-date index and data requires careful path management.

### Future Improvements

*   **Advanced Retrieval:** Implement strategies like Parent Document Retriever or re-ranking for better context relevance.
*   **Vector Database:** Migrate from FAISS to a dedicated vector DB (Qdrant, Milvus) for scalability and advanced filtering.
*   **Model Experimentation:** Test different embedding models or newer/larger Gemini models (e.g., Gemini 1.5 Pro).
*   **Data Quality:** Enhance `knowledgebase.json` with more structured fields (cuisine, specific dietary tags) and ensure data consistency. Implement robust cleaning in `process.py` or a dedicated script.
*   **Conversational Memory:** Add context from previous turns for follow-up questions.
*   **Evaluation:** Implement a framework (e.g., RAGAs) to systematically evaluate retrieval and generation quality.
*   **Error Handling:** Improve robustness and provide more informative user feedback on errors throughout the pipeline (scraping, processing, RAG).
*   **Refine Scraper:** Make `web_scraper.py` more robust to different website structures; implement `fetch_websites.py` and `utils.py`.
*   **Streamline Data Flow:** Clarify and potentially automate the data flow from raw extraction to indexed data used by the app.

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
