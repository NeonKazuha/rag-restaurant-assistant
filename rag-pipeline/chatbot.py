import re
from transformers import pipeline
from index_faiss import retrieve_chunks
from query_planner import parse_query

# Text-generation model
generator = pipeline(
    "text2text-generation",
    model="t5-base",
    tokenizer="t5-base",
    device=-1
)

PROMPT = """
You are ZomatoBot. Use the following information to answer the question.
Context:
{context}

Question:
{question}

Answer concisely:
""".strip()

def list_dishes(resto, chunks):
    names = [c['dish_name'] for c in chunks if c['resto_id'] == resto]
    return names

def answer_dynamic(question, embed_model, index, chunks, k=5):
    spec = parse_query(question)

    # 1) Price filter
    if spec['price_lt'] is not None:
        thr = spec['price_lt']
        idxs = [i for i,c in enumerate(chunks)
                if c.get('price') is not None and c['price'] < thr]
        if not idxs:
            return f"No dishes under ₹{thr}."
        if len(idxs) > k:
            selected = retrieve_chunks(question, embed_model, index, chunks,
                                       k=k, filter_indices=idxs)
        else:
            selected = [chunks[i] for i in idxs]
        return f"Dishes under ₹{thr}:\n" + "\n".join(c['dish_name'] for c in selected)

    # 2) Dietary filters
    if spec['dietary']:
        tags = spec['dietary']
        idxs = [i for i,c in enumerate(chunks)
                if all(c['attributes'].get(t, False) for t in tags)]
        if not idxs:
            return f"No dishes matching {', '.join(tags)}."
        if len(idxs) > k:
            selected = retrieve_chunks(question, embed_model, index, chunks,
                                       k=k, filter_indices=idxs)
        else:
            selected = [chunks[i] for i in idxs]
        return f"Dishes with {', '.join(tags)}:\n" + "\n".join(c['dish_name'] for c in selected)

    # 3) Spice comparison
    if spec['spice_cmp']:
        dish = spec['spice_cmp']['dish']
        r1, r2 = spec['spice_cmp']['restaurants']
        # find matching chunks
        d1 = next((c for c in chunks if c['dish_name']==dish and c['resto_id']==r1), None)
        d2 = next((c for c in chunks if c['dish_name']==dish and c['resto_id']==r2), None)
        if not d1 or not d2:
            return "Couldn’t find that dish in one of the restaurants."
        ctx = (f"{r1} – {dish}: spice level {d1['attributes'].get('spice_level')}\n"
               f"{r2} – {dish}: spice level {d2['attributes'].get('spice_level')}")
        prompt = PROMPT.format(context=ctx, question=question)
        return generator(prompt, max_length=100, do_sample=False)[0]['generated_text']

    # 4) List all dishes at a restaurant
    if spec['restaurant']:
        names = list_dishes(spec['restaurant'], chunks)
        if not names:
            return f"No dishes found for '{spec['restaurant']}'."
        return f"{spec['restaurant']} dishes:\n" + "\n".join(names)

    # 5) Fallback full RAG
    top = retrieve_chunks(question, embed_model, index, chunks, k=k)
    if not top:
        return "Sorry, I couldn't find relevant info."
    ctx = "\n".join(c['text'] for c in top)
    prompt = PROMPT.format(context=ctx, question=question)
    return generator(prompt, max_length=150, do_sample=False)[0]['generated_text']
