import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from fetch_restaurant import load_restaurants
from chunking import build_dish_chunks
from index_faiss import create_faiss_index
from chatbot import answer_hybrid

load_dotenv()

# --- Configuration ---
DATA_PATH = Path('D:/rag-restaurant-assistant/data/processed/knowledgebase.json')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# --- Data and Model Loading Functions ---

# Cache data loading and chunking to avoid reloading on every interaction
@st.cache_data(show_spinner="Loading restaurant data...")
def load_restaurants_and_chunk_data(json_path):
    """Loads restaurant data from JSON and creates structured chunks."""
    try:
        restaurants_data = load_restaurants(json_path=json_path)
        if not restaurants_data:
            st.error(f"No restaurants loaded from {json_path}.")
            return None, None
        chunks = build_dish_chunks(restaurants_data)
        if not chunks:
            st.error("Failed to build data chunks.")
            return restaurants_data, None
        return restaurants_data, chunks
    except FileNotFoundError:
        st.error(f"Error: knowledgebase.json not found at {json_path}.")
        return None, None
    except Exception as e:
        st.error(f"Error loading or chunking data: {e}")
        return None, None

# Cache resource loading (models, index) for efficiency
@st.cache_resource(show_spinner="Setting up embedding model and index...")
def load_models_and_index(_chunks):
    """Loads the embedding model and creates the FAISS index."""
    if _chunks is None:
        st.error("Cannot initialize models without data chunks.")
        return None, None

    try:
        embed_model, index = create_faiss_index(_chunks, model_name=EMBEDDING_MODEL)
        if embed_model is None or index is None:
             st.error("Failed to initialize embedding model or FAISS index.")
             return None, None
        st.success("Embedding model and index loaded successfully.")
        return embed_model, index
    except Exception as e:
        st.error(f"Error initializing embedding model or FAISS index: {e}")
        return None, None

# --- Streamlit App UI ---
st.set_page_config(page_title="Restaurant Chatbot (Hybrid)", layout="centered")
st.title("🍽️ Restaurant Chatbot (Hybrid)")
st.caption("Ask me about dishes, prices, comparisons, and more!")

restaurants_data, chunks = load_restaurants_and_chunk_data(DATA_PATH)

if restaurants_data and chunks:
    embed_model, index = load_models_and_index(chunks)

    if embed_model and index:
        # Initialize Streamlit chat history if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you find restaurant information today?"}]

        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask your question here..."):
            # Add user message to state and display it
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display the assistant's response
            with st.chat_message("assistant"):
                message_placeholder = st.empty() # For streaming-like effect
                full_response = ""
                with st.spinner("Thinking..."):
                    # Pass restaurants_data for direct lookups (e.g., ratings)
                    response = answer_hybrid(prompt, embed_model, index, chunks, restaurants_data)
                    full_response = response

                message_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.warning("Embedding model or index could not be initialized. The chatbot is unavailable.")
        st.stop()
else:
    st.warning("Data could not be loaded or chunked. The chatbot is unavailable.")
    st.stop()
