import streamlit as st
import os

from fetch_restaurant import load_restaurants
from chunking import build_dish_chunks
from index_faiss import create_faiss_index
from chatbot import answer_dynamic, generator

# --- Configuration ---
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'knowledgebase.json')
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
         # Raise an exception or handle this more gracefully
         # For now, returning None to indicate failure
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

    # Check if models loaded successfully before proceeding
    if embed_model and index and text_generator:

        # Initialize chat history in session state if it doesn't exist
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you find restaurant information today?"}]

        # Display existing chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Get user input using chat_input
        if prompt := st.chat_input("Ask your question here..."):
            # Add user message to chat history and display it
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                with st.spinner("Thinking..."):
                    # Call the main answering function from chatbot.py
                    try:
                        response = answer_dynamic(prompt, embed_model, index, chunks)
                        full_response = response
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        full_response = "Sorry, I encountered an error while processing your request."

                message_placeholder.markdown(full_response)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.warning("Models could not be initialized. The chatbot is unavailable.")
        st.stop()
else:
    st.warning("Data could not be loaded. The chatbot is unavailable.")
    st.stop()

