import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch # Import torch
import logging # Use logging for better messages
import os # Import os for cache path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Function to get cache directory ---
def get_sentence_transformer_cache_dir():
    """Gets the default sentence transformer cache directory."""
    cache_home = os.getenv('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
    return os.path.join(cache_home, 'torch', 'sentence_transformers')

def create_faiss_index(chunks, model_name='all-MiniLM-L6-v2'):
    """
    Compute embeddings for each chunk['text'] and build a FAISS L2 index.
    Embeddings are normalized, which is standard practice for L2 distance search.
    Returns (embed_model, faiss_index).
    """
    if not chunks:
        logging.error("No chunks provided to create_faiss_index.")
        return None, None

    texts = [c.get('text', '') for c in chunks] # Use .get for safety

    # --- Log Cache Directory ---
    cache_dir = get_sentence_transformer_cache_dir()
    logging.info(f"Expected Sentence Transformer cache directory: {cache_dir}")
    if not os.path.exists(cache_dir):
        logging.warning(f"Cache directory does not exist. Model will be downloaded.")
    else:
        logging.info(f"Cache directory exists.")

    # --- Device Selection ---
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logging.info(f"Attempting to load model on device: {device}")

    # --- Model Loading ---
    embed_model = None
    try:
        embed_model = SentenceTransformer(model_name, device=device)
        logging.info(f"SentenceTransformer '{model_name}' loaded successfully on {device}.")
    except Exception as e:
        logging.exception(f"Initial attempt to load SentenceTransformer '{model_name}' on device {device} FAILED. Original error:")
        if device != torch.device('cpu'):
            logging.warning("Attempting to load SentenceTransformer on CPU as fallback...")
            device = torch.device('cpu')
            try:
                embed_model = SentenceTransformer(model_name, device=device)
                logging.info(f"SentenceTransformer '{model_name}' loaded successfully on CPU (fallback).")
            except Exception as e_cpu:
                logging.exception(f"Fallback attempt to load SentenceTransformer '{model_name}' on CPU ALSO FAILED. Original error:")
                raise RuntimeError(f"Could not load SentenceTransformer '{model_name}' on any available device. See logged exceptions for details.") from e_cpu
        else:
             raise RuntimeError(f"Could not load SentenceTransformer '{model_name}' on CPU. See logged exception for details.") from e

    # --- Embedding Generation ---
    try:
        logging.info(f"Generating embeddings for {len(texts)} chunks using device {embed_model.device}...")
        # Ensure normalization is True (default for many sentence transformers, but good to be explicit)
        embeddings = embed_model.encode(
            texts,
            normalize_embeddings=True,
            device=embed_model.device,
            show_progress_bar=True
        )
        logging.info("Embeddings generated successfully.")
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}", exc_info=True)
        raise RuntimeError("Failed to generate embeddings.") from e

    # --- FAISS Index Creation (Using L2) ---
    try:
        dim = embeddings.shape[1]
        logging.info(f"Creating FAISS IndexFlatL2 with dimension {dim}.") # Changed to L2
        embeddings_np = np.array(embeddings, dtype='float32')
        if np.isnan(embeddings_np).any() or np.isinf(embeddings_np).any():
             logging.error("Embeddings contain NaN or Inf values. Cannot add to FAISS index.")
             raise ValueError("Invalid values (NaN/Inf) found in embeddings.")

        # Use IndexFlatL2 for Euclidean distance
        index = faiss.IndexFlatL2(dim)
        # FAISS expects L2 normalized vectors for efficient search with IndexFlatL2
        # The normalize_embeddings=True in encode() should handle this.
        index.add(embeddings_np)
        logging.info(f"FAISS IndexFlatL2 created and {index.ntotal} vectors added.")
    except Exception as e:
        logging.error(f"Error creating or adding to FAISS index: {e}", exc_info=True)
        raise RuntimeError("Failed to create or populate FAISS index.") from e

    return embed_model, index

def retrieve_chunks(query, embed_model, index, chunks, k=5, filter_indices=None):
    """
    Retrieve top-k chunks for a query using FAISS IndexFlatL2.
    Handles potential filtering and ensures safe index access.
    Assumes embeddings in the index and the query embedding are L2 normalized.
    """
    if not query or embed_model is None or index is None or chunks is None:
        logging.warning("retrieve_chunks called with invalid arguments.")
        return []

    device = embed_model.device
    logging.info(f"Retrieving chunks for query using device {device}")

    # --- Query Embedding ---
    try:
        # Ensure query embedding is also normalized
        q_emb = embed_model.encode([query], normalize_embeddings=True, device=device)
        q_emb_np = np.array(q_emb, dtype='float32')
        if np.isnan(q_emb_np).any() or np.isinf(q_emb_np).any():
             logging.error("Query embedding contains NaN or Inf values.")
             return []
    except Exception as e:
        logging.error(f"Error encoding query '{query}': {e}", exc_info=True)
        return []

    # --- Search ---
    selected_indices = []
    try:
        if filter_indices is not None:
            logging.info(f"Performing filtered L2 search within {len(filter_indices)} indices.")
            valid_filter_indices = [int(i) for i in filter_indices if isinstance(i, (int, np.integer)) and 0 <= i < index.ntotal]
            if not valid_filter_indices:
                logging.warning("No valid indices remaining after filtering.")
                return []
            try:
                 # Reconstruct requires the original index type, but we search the temp index
                 # We need to create a temporary L2 index with the filtered embeddings
                 sub_embs = np.array([index.reconstruct(i) for i in valid_filter_indices], dtype='float32')
                 if sub_embs.size == 0:
                     logging.warning("Filtered indices resulted in zero embeddings for reconstruction.")
                     return []

                 # Ensure sub_embs are L2 normalized if they weren't stored normalized (they should be)
                 faiss.normalize_L2(sub_embs)

                 temp_index = faiss.IndexFlatL2(sub_embs.shape[1]) # Use L2 for temp index too
                 temp_index.add(sub_embs)
                 k_search = min(k, len(valid_filter_indices))
                 distances, temp_indices = temp_index.search(q_emb_np, k_search) # L2 search returns squared L2 distances

                 if temp_indices.size > 0:
                     selected_indices = [valid_filter_indices[i] for i in temp_indices[0] if i >= 0]

            except Exception as e_rec:
                 logging.error(f"Error reconstructing/searching filtered embeddings: {e_rec}", exc_info=True)
                 return []
        else:
            logging.info(f"Performing L2 search on the main index for top {k} results.")
            # Search the main L2 index
            distances, direct_indices = index.search(q_emb_np, k)
            if direct_indices.size > 0:
                selected_indices = [i for i in direct_indices[0] if i >= 0]

    except Exception as e_search:
        logging.error(f"Error during FAISS search: {e_search}", exc_info=True)
        return []

    logging.info(f"FAISS search returned {len(selected_indices)} potential indices: {selected_indices}")

    # --- Result Consolidation ---
    final_results = []
    num_chunks = len(chunks)
    for idx in selected_indices:
        if 0 <= idx < num_chunks:
            final_results.append(chunks[idx])
        else:
            logging.warning(f"Index {idx} from search result is out of bounds for chunks list (size {num_chunks}). Skipping.")

    logging.info(f"Returning {len(final_results)} retrieved chunks.")
    return final_results
