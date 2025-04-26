# Project Deliverables: RAG Restaurant Assistant

This document outlines the key deliverables for the RAG Restaurant Assistant project.

## 1. Code Repository

### Source Code Files:

* **`app.py`**: Streamlit web application interface for the chatbot.
* **`run.py`**: Command-line interface for running the chatbot.
* **`chatbot.py`**: Core logic for handling user queries, including intent-specific handling (comparisons, filtering) and the RAG fallback mechanism using a T5 generation model.
* **`query_planner.py`**: Implements regex-based parsing to detect user intent (e.g., price queries, comparisons, menu requests) and extract relevant parameters.
* **`index_faiss.py`**: Handles embedding generation using `SentenceTransformer` ('all-MiniLM-L6-v2') and FAISS vector index creation/retrieval. Includes retrieval logic with filtering capabilities.
* **`chunking.py`**: Defines the strategy for processing the input JSON data into smaller, indexed chunks (per menu item), including relevant metadata.
* **`fetch_restaurant.py`**: Utility script to load the restaurant data from the `knowledgebase.json` file.

### Setup and Running Instructions (`README.md` Outline):

* **Project Description:** A brief overview of the RAG chatbot for querying restaurant information.
* **Requirements:**
    * Python (Recommended: 3.10 or 3.11).
    * `pip` (Python package installer).
    * `git` (for cloning the repository).
    * List of libraries (should be generated into `requirements.txt`, e.g., `streamlit`, `transformers`, `torch`, `sentence-transformers`, `faiss-cpu` or `faiss-gpu`, `numpy`).
* **Setup:**

    **Using Bash (Linux/macOS/WSL):**

    ```bash
    # 1. Clone the repository (replace with actual repository URL)
    git clone <your-repository-url>
    cd <repository-directory-name>

    # 2. Create a Python virtual environment
    python3 -m venv venv

    # 3. Activate the environment
    source venv/bin/activate

    # 4. Install dependencies
    pip install -r requirements.txt

    # 5. Ensure knowledgebase.json is placed correctly
    # (e.g., in data/processed/knowledgebase.json relative to scripts)
    # mkdir -p data/processed
    # cp <path-to-your-knowledgebase.json> data/processed/knowledgebase.json
    ```

    **Using Command Prompt (Windows):**

    ```cmd
    :: 1. Clone the repository (replace with actual repository URL)
    git clone <your-repository-url>
    cd <repository-directory-name>

    :: 2. Create a Python virtual environment
    python -m venv venv

    :: 3. Activate the environment
    venv\Scripts\activate.bat

    :: 4. Install dependencies
    pip install -r requirements.txt

    :: 5. Ensure knowledgebase.json is placed correctly
    :: (e.g., in data\processed\knowledgebase.json relative to scripts)
    :: mkdir data\processed
    :: copy <path-to-your-knowledgebase.json> data\processed\knowledgebase.json
    ```

* **Running the Application:**
    * **Streamlit App:** `streamlit run app.py`
        * (Optional: If watcher issues persist) `streamlit run app.py --server.fileWatcherType none`
    * **Console App:** `python run.py`

## 2. Scraped Dataset

(Content remains the same as the previous version: Dataset File, Schema Documentation, Collection Methodology placeholder)

* **Dataset File:** `knowledgebase.json` - The primary data source containing restaurant details and menu items.
* **Data Schema Documentation:**
    * **Root:** A JSON list `[]` containing multiple `Restaurant` objects.
    * **`Restaurant` Object:**
        * `name` (String)
        * `menu_items` (List of `MenuItem` objects)
        * `dietary_options` (List of Strings, e.g., `"Veg"`)
        * `price_range` (String)
        * `address` (String)
        * `opening_hours` (String)
        * `image_url` (String)
        * `phone_number` (String)
        * `rating` (String)
    * **`MenuItem` Object:**
        * `name` (String)
        * `description` (String)
        * `price` (Float/Number)
        * `attributes` (Object)
            * `veg_nonveg` (String)
            * `category` (String)
            * `spice_level` (Number/Integer)
* **Collection Methodology:**
    * *(User to provide details on how `knowledgebase.json` was created - e.g., scraping tools, websites targeted, manual collection, data cleaning process).*

## 3. Technical Documentation

Explains the system architecture, design decisions, challenges, and potential future improvements.

### System Architecture:

The system implements a hybrid Retrieval-Augmented Generation (RAG) pipeline tailored for querying restaurant menu data.

**Core Components & Flow:**

* **Data Ingestion:**
    * The process starts by loading structured restaurant data from `knowledgebase.json` using the `load_restaurants` function in `fetch_restaurant.py`. This data includes restaurant details, menu items, prices, ratings, etc.

* **Data Preprocessing (Offline/Initialization):**
    * **Chunking (`chunking.py`):** The loaded data is processed into smaller, manageable units (chunks). Each chunk typically represents a single menu item, but importantly includes relevant context from the parent restaurant (like name, rating, price range) and item attributes (`veg_nonveg`, price, spice level). This contextual information is embedded within the chunk's text (`'text'`) or stored as metadata.
    * **Embedding (`index_faiss.py`):** The textual content (`'text'`) of each chunk is converted into a dense vector representation (embedding) using a pre-trained `SentenceTransformer` model (`all-MiniLM-L6-v2`). These embeddings capture the semantic meaning of the chunk.

* **Indexing (Offline/Initialization):**
    * **Vector Store (`index_faiss.py`):** The generated vector embeddings are stored in a FAISS index (`IndexFlatL2`). This index allows for efficient similarity searches, enabling retrieval of chunks whose embeddings are closest to a query embedding. The raw chunk data (including metadata) is kept alongside, mapped by index position.

* **Query Processing (Online/Runtime):**
    * **User Interface (`app.py` / `run.py`):** Receives the user's natural language question.
    * **Query Parsing (`query_planner.py`):** The question is analyzed using regular expressions to identify specific, pre-defined intents (e.g., compare ratings, filter by price, request menu). It extracts key parameters (restaurant names, dish names, price thresholds).
    * **Execution Strategy (`chatbot.py`):**
        * **Direct Handling:** If a specific intent is detected, the system might bypass semantic search. It can directly filter the original `restaurants` data (e.g., for ratings/price range) or the `chunks` list based on metadata (e.g., price < X, `veg_nonveg` == 'Veg', dish name lookup).
        * **Semantic Retrieval (RAG Fallback):** If no specific intent matches, the user's query is embedded using the same `SentenceTransformer` model. This query embedding is used to search the FAISS index (`retrieve_chunks` function) for the top-k most semantically similar dish chunks.

* **Context Formulation (`chatbot.py`):**
    * Relevant information retrieved from either direct handling (e.g., specific ratings, prices) or semantic search (text content of top-k chunks) is assembled into a context string.

* **Answer Generation (`chatbot.py`):**
    * The formulated context, along with the original user question, is inserted into a predefined prompt template (`PROMPT`).
    * This complete prompt is passed to a text generation model (T5-base `generator`) which synthesizes a natural language answer based *only* on the provided context and prompt instructions.

* **Response Delivery (`app.py` / `run.py`):** The generated answer is displayed back to the user through the interface.

**Visual Flow (Mermaid Diagram):**

```mermaid
sequenceDiagram
    participant User
    participant UI (app.py / run.py)
    participant Chatbot (chatbot.py)
    participant QueryPlanner (query_planner.py)
    participant Index (index_faiss.py / FAISS)
    participant Data (knowledgebase.json / chunks list / restaurants list)
    participant Embedder (SentenceTransformer)
    participant Generator (T5 Model)

    %% Initialization Steps (Conceptual)
    Note over Data, Index: Load Data (fetch_restaurant.py)
    Note over Data, Index: Chunk Data (chunking.py)
    Note over Data, Index: Embed Chunks (index_faiss.py)
    Note over Data, Index: Build FAISS Index (index_faiss.py)

    %% Runtime Query Flow
    User->>+UI: Asks question (e.g., "Compare ratings for A and B")
    UI->>+Chatbot: answer_dynamic(question, ...)
    Chatbot->>+QueryPlanner: parse_query(question)
    QueryPlanner-->>-Chatbot: Return spec (e.g., {'feature_compare': ... 'rating'})
    Chatbot->>Chatbot: Evaluate spec
    Note over Chatbot, Data: Intent is 'feature_compare' (rating) -> Use Direct Handling
    Chatbot->>Data: find_restaurant("A"), find_restaurant("B")
    Data-->>Chatbot: Return Restaurant A data, Restaurant B data
    Chatbot->>Chatbot: Extract ratings, formulate context string
    Chatbot->>+Generator: generator(prompt with context & question)
    Generator-->>-Chatbot: Synthesized Answer
    Chatbot-->>-UI: Return generated answer
    UI-->>-User: Display answer

    %% Alternative Flow: RAG Fallback
    User->>+UI: Asks question (e.g., "Any spicy paneer dishes?")
    UI->>+Chatbot: answer_dynamic(question, ...)
    Chatbot->>+QueryPlanner: parse_query(question)
    QueryPlanner-->>-Chatbot: Return spec (no specific intent matched)
    Chatbot->>Chatbot: Evaluate spec -> Fallback to RAG
    Chatbot->>+Embedder: embed(question)
    Embedder-->>-Chatbot: Query Vector
    Chatbot->>+Index: retrieve_chunks(Query Vector, k)
    Index-->>-Chatbot: Top-k relevant chunks
    Chatbot->>Chatbot: Formulate context from chunk text
    Chatbot->>+Generator: generator(prompt with context & question)
    Generator-->>-Chatbot: Synthesized Answer
    Chatbot-->>-UI: Return generated answer
    UI-->>-User: Display answer
