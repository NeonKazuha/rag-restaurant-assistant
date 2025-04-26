import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def create_faiss_index(chunks, model_name='all-MiniLM-L6-v2'):
    """
    Compute embeddings for each chunk['text'] and build a FAISS Flat index.
    Returns (embed_model, faiss_index).
    """
    texts = [c['text'] for c in chunks]
    embed_model = SentenceTransformer(model_name, device='cpu')
    embeddings = embed_model.encode(texts, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype='float32'))
    return embed_model, index

def retrieve_chunks(query, embed_model, index, chunks, k=5, filter_indices=None):
    """
    Retrieve top-k chunks for a query.
    If filter_indices is provided, restrict search to those chunk positions.
    """
    q_emb = embed_model.encode([query], normalize_embeddings=True)
    if filter_indices is not None:
        # build a small index for just the filtered ones
        sub_embs = np.array([index.reconstruct(i) for i in filter_indices], dtype='float32')
        temp = faiss.IndexFlatIP(sub_embs.shape[1])
        temp.add(sub_embs)
        D, I = temp.search(q_emb, min(k, len(filter_indices)))
        selected = [filter_indices[i] for i in I[0]]
    else:
        D, I = index.search(q_emb, k)
        selected = I[0]
    return [chunks[i] for i in selected]
