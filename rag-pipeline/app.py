import streamlit as st
from pathlib import Path
from fetch_restaurant import load_restaurants
from chunking import build_dish_chunks
from index_faiss import create_faiss_index
from chatbot import answer_dynamic, generator

DATA_PATH = Path('D:/rag-restaurant-assistant/data/processed/knowledgebase.json')
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

@st.cache_data(show_spinner="Loading restaurant data...")
def load_and_chunk_data(json_path):
    """Loads restaurants and builds dish chunks."""
    try:
        restaurants = load_restaurants(json_path=json_path)
        if not restaurants:
            st.error(f"No restaurants loaded from {json_path}. Please ensure the file exists and is not empty.")
            return None
        chunks = build_dish_chunks(restaurants)
        return chunks
    except FileNotFoundError:
        st.error(f"Error: knowledgebase.json not found at {json_path}. Please ensure the file exists in the 'data' subdirectory.")
        return None
    except Exception as e:
        st.error(f"Error loading or chunking data: {e}")
        return None

# Cache model loading and index creation
@st.cache_resource(show_spinner="Setting up AI models and index...")
def load_models_and_index(_chunks):
    """Loads embedding model, T5 generator, and creates FAISS index."""
    if _chunks is None:
        st.error("Cannot initialize models without data chunks.")
        return None, None, None

    # Check if the Hugging Face generator from chatbot.py loaded correctly
    if generator is None:
         st.error("Failed to load the T5 text generation model. Cannot proceed.")
         return None, None, None

    try:
        embed_model, index = create_faiss_index(_chunks, model_name=EMBEDDING_MODEL)
        return embed_model, index, generator
    except Exception as e:
        st.error(f"Error initializing models or FAISS index: {e}")
        return None, None, None

# --- Streamlit App UI ---
st.set_page_config(page_title="Restaurant Chatbot", layout="centered")
st.title("🍽️ Restaurant Chatbot")
st.caption("Ask me about dishes, prices, dietary options, and more!")

chunks = load_and_chunk_data(DATA_PATH)

# Only proceed if chunks were loaded successfully
if chunks:
    embed_model, index, text_generator = load_models_and_index(chunks)

    if embed_model and index and text_generator:
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you find restaurant information today?"}]

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask your question here..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                with st.spinner("Thinking..."):
                    response = answer_dynamic(prompt, embed_model, index, chunks)
                    full_response = response

                message_placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.warning("Models could not be initialized. The chatbot is unavailable.")
        st.stop()
else:
    st.warning("Data could not be loaded. The chatbot is unavailable.")
    st.stop()

