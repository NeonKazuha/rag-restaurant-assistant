import re
from transformers import pipeline
from index_faiss import retrieve_chunks
from query_planner import parse_query

try:
    generator = pipeline(
        "text2text-generation",
        model="t5-base",
        tokenizer="t5-base",
        device=-1 # Use -1 for CPU, 0 for first GPU, etc.
    )
except Exception as e:
    print(f"Error initializing Hugging Face pipeline: {e}")
    generator = None # Set generator to None if initialization fails

# --- Prompt Template ---
PROMPT = """
You are ZomatoBot. Use ONLY the following context to answer the question.
Do not add information not present in the context.
If the question is subjective (e.g., asking for the 'best'), list relevant options from the context with their descriptions instead of making a judgment.

Context:
{context}

Question:
{question}

Answer based on the context:
""".strip()

# --- Helper Function ---
def list_dishes(resto_id, chunks):
    """
    Retrieves all dish names for a specific restaurant ID from the chunks.

    Args:
        resto_id (str): The ID of the restaurant.
        chunks (list): A list of dictionaries, where each dictionary represents a dish chunk.

    Returns:
        list: A list of dish names for the given restaurant.
    """

    names = [c.get('dish_name', 'Unknown Dish') for c in chunks if c.get('resto_id') == resto_id]
    return names

# --- Main Answering Function ---
def answer_dynamic(question, embed_model, index, chunks, k=10):
    """
    Answers a user question dynamically based on parsed query specifications
    or by using RAG as a fallback.

    Args:
        question (str): The user's question.
        embed_model: The sentence transformer model used for embeddings.
        index: The FAISS index containing dish embeddings.
        chunks (list): The list of dish data chunks.
        k (int): The number of chunks to retrieve for RAG/filtering.

    Returns:
        str: The generated answer to the question.
    """
    # Check if the generator was initialized successfully
    if generator is None:
        return "Sorry, the text generation model could not be loaded."

    # Parse the user's query to understand intent and extract parameters
    try:
        spec = parse_query(question)
    except Exception as e:
        print(f"Error parsing query: {e}")
        return "Sorry, I had trouble understanding your request."


    # --- Handling Specific Query Types ---
    md_newline = "  \n"
    # 1) Price Filter: Find dishes below a certain price
    if spec.get('price_lt') is not None:
        thr = spec['price_lt']
        # Find indices of chunks matching the price criteria
        idxs = [i for i, c in enumerate(chunks)
                if c.get('price') is not None and isinstance(c.get('price'), (int, float)) and c['price'] < thr]
        if not idxs:
            return f"Sorry, I couldn't find any dishes under ₹{thr}."
        # If many matches, use semantic search within the filtered set
        if len(idxs) > k:
            selected = retrieve_chunks(question, embed_model, index, chunks,
                                       k=k, filter_indices=idxs)
        # If few matches, just use all of them
        else:
            selected = [chunks[i] for i in idxs]
        # Safely get dish names
        dish_names = [c.get('dish_name', 'Unknown Dish') for c in selected]
        return f"Here are some dishes under ₹{thr}:{md_newline}" + md_newline.join(dish_names)

    # 2) Dietary Filters: Find dishes matching dietary tags (e.g., vegan, gluten-free)
    if spec.get('dietary'):
        tags = spec['dietary']
        # Find indices of chunks matching *all* specified dietary tags
        idxs = [i for i, c in enumerate(chunks)
                if all(c.get('attributes', {}).get(t, False) for t in tags)] # Check attributes safely
        if not idxs:
            return f"Sorry, I couldn't find any dishes matching all criteria: {', '.join(tags)}."
        # Retrieve relevant chunks if many matches, otherwise use all
        if len(idxs) > k:
            selected = retrieve_chunks(question, embed_model, index, chunks,
                                       k=k, filter_indices=idxs)
        else:
            selected = [chunks[i] for i in idxs]
        # Safely get dish names
        dish_names = [c.get('dish_name', 'Unknown Dish') for c in selected]
        return f"Here are some dishes under ₹{thr}:{md_newline}" + md_newline.join(dish_names)

    # 3) Spice Comparison: Compare spice levels of the same dish at two restaurants
    if spec.get('spice_cmp'):
        spice_info = spec['spice_cmp']
        dish = spice_info.get('dish')
        restaurants = spice_info.get('restaurants')

        # Basic validation
        if not dish or not restaurants or len(restaurants) != 2:
             return "Sorry, I need a dish name and exactly two restaurant names to compare spice levels."

        r1, r2 = restaurants
        # Find the specific dish chunk for each restaurant
        d1 = next((c for c in chunks if c.get('dish_name') == dish and c.get('resto_id') == r1), None)
        d2 = next((c for c in chunks if c.get('dish_name') == dish and c.get('resto_id') == r2), None)

        if not d1 or not d2:
            missing_restos = []
            if not d1: missing_restos.append(r1)
            if not d2: missing_restos.append(r2)
            return f"Sorry, I couldn’t find the dish '{dish}' in restaurant(s): {', '.join(missing_restos)}."

        # Get spice levels safely, providing a default message if missing
        spice1 = d1.get('attributes', {}).get('spice_level', 'Not specified')
        spice2 = d2.get('attributes', {}).get('spice_level', 'Not specified')

        # Prepare context for the language model
        ctx = (f"Restaurant '{r1}' serves '{dish}' with spice level: {spice1}.\n"
               f"Restaurant '{r2}' serves '{dish}' with spice level: {spice2}.")
        try:
            prompt = PROMPT.format(context=ctx, question=question)
            # Generate the comparison answer using the LLM
            result = generator(prompt, max_length=100, do_sample=False)
            return result[0]['generated_text']
        except Exception as e:
            print(f"Error during text generation for spice comparison: {e}")
            return f"I found the spice levels ({r1}: {spice1}, {r2}: {spice2}), but couldn't generate a comparison summary."


    # List Dishes at Restaurant: List all known dishes for a specific restaurant
    if spec.get('restaurant'):
        resto_id = spec['restaurant']
        names = list_dishes(resto_id, chunks)
        if not names:
            return f"Sorry, I couldn't find any dishes listed for the restaurant '{resto_id}'."
        return f"Dishes available at {resto_id}:\n" + "\n".join(names)

    # --- Fallback: Full RAG ---
    # If no specific query type matched, use general RAG
    print(f"Performing general RAG for question: {question}")
    try:
        top = retrieve_chunks(question, embed_model, index, chunks, k=k)
        if not top:
            return "Sorry, I couldn't find relevant information to answer your question."

        # Combine text from retrieved chunks to form the context
        ctx = "\n---\n".join(c.get('text', '') for c in top)
        prompt = PROMPT.format(context=ctx, question=question)

        # Generate the final answer
        result = generator(prompt, max_length=150, do_sample=False)
        return result[0]['generated_text']
    except Exception as e:
        print(f"Error during fallback RAG: {e}")
        return "Sorry, I encountered an error while trying to find an answer."

