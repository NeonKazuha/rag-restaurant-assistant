import re
from transformers import pipeline
from index_faiss import retrieve_chunks
from query_planner import parse_query
import numpy as np
from collections import defaultdict
import json

generator = pipeline(
        "text2text-generation",
        model="t5-base",
        tokenizer="t5-base",
        device=-1
    )

PROMPT = """
You are ZomatoBot. Use ONLY the following context to answer the question.
Do not add information not present in the context or the knowledge base structure.
If the question asks for comparison based on rating or price range, provide the comparison based on the context.
If the question is subjective (e.g., asking for the 'best' without specifying criteria), list relevant options from the context.
Mention if specific information like cuisine type, vegan, or gluten-free is not available in the data if relevant to the query.

Context:
{context}

Question:
{question}

Answer based on the context:
""".strip()

def find_restaurant(name, restaurants):
    """Finds the first matching restaurant dict by name."""
    for resto in restaurants:
        if resto.get('name', '').strip().lower() == name.strip().lower():
            return resto
    return None

def list_dishes(resto_id, chunks):
    """Retrieves unique dish names for a specific restaurant ID."""
    names = {c.get('dish_name', 'Unknown Dish') for c in chunks if c.get('resto_id', '').lower() == resto_id.lower()}
    return sorted(list(names))

def parse_rating(rating_str):
    """Attempts to extract a numerical rating."""
    if not isinstance(rating_str, str):
        return None
    match = re.search(r"(\d+(?:\.\d+)?)", rating_str)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None

def answer_dynamic(question, embed_model, index, chunks, restaurants, k=10):
    """
    Answers user questions dynamically based on parsed specs or RAG fallback.
    Handles rating/price comparisons, price range queries, etc.
    """
    if generator is None:
        return "Sorry, the text generation model could not be loaded."
    if not restaurants:
        return "Sorry, the restaurant data could not be loaded."

    try:
        spec = parse_query(question)
    except Exception as e:
        print(f"Error parsing query: {e}")
        return "Sorry, I had trouble understanding your request."

    md_newline = "  \n"

    # --- Handling Specific Query Types ---

    # 1) Dish Price Query (Existing logic okay)
    if spec.get('dish_price_query'):
        dish_name = spec['dish_price_query']
        idxs = [i for i, c in enumerate(chunks) if c.get('dish_name', '').lower() == dish_name.lower()]
        if not idxs:
            return f"Sorry, I couldn't find the dish '{dish_name}' in any restaurant."
        selected = [chunks[i] for i in idxs]
        results = []
        for chunk in selected:
            resto_id = chunk.get('resto_id', 'Unknown Restaurant')
            price = chunk.get('price', 'N/A')
            price_str = f"₹{price:.2f}" if isinstance(price, (int, float)) else str(price)
            results.append(f"- {resto_id}: {price_str}")
        if not results:
             return f"Sorry, I found the dish '{dish_name}' but couldn't retrieve price information."
        unique_results = sorted(list(set(results)))
        return f"Prices for '{dish_name}':{md_newline}" + md_newline.join(unique_results)


    # 2) Spice Comparison (Existing logic okay)
    elif spec.get('spice_cmp'):
        spice_info = spec['spice_cmp']
        dish = spice_info.get('dish')
        resto_names = spice_info.get('restaurants')
        if not dish or not resto_names or len(resto_names) != 2:
             return "Sorry, I need a dish name and exactly two restaurant names to compare spice levels."
        r1_name, r2_name = resto_names
        d1 = next((c for c in chunks if c.get('dish_name', '').lower() == dish.lower() and c.get('resto_id', '').lower() == r1_name.lower()), None)
        d2 = next((c for c in chunks if c.get('dish_name', '').lower() == dish.lower() and c.get('resto_id', '').lower() == r2_name.lower()), None)
        if not d1 or not d2:
            missing_restos = []
            if not d1: missing_restos.append(r1_name)
            if not d2: missing_restos.append(r2_name)
            return f"Sorry, I couldn’t find the dish '{dish}' in restaurant(s): {', '.join(missing_restos)}."
        spice1 = d1.get('attributes', {}).get('spice_level', 'Not specified')
        spice2 = d2.get('attributes', {}).get('spice_level', 'Not specified')
        ctx = (f"Restaurant '{r1_name}' serves '{dish}' with spice level: {spice1}.\n"
               f"Restaurant '{r2_name}' serves '{dish}' with spice level: {spice2}.")
        try:
            prompt = PROMPT.format(context=ctx, question=question)
            result = generator(prompt, max_length=100, do_sample=False)
            return result[0]['generated_text']
        except Exception as e:
            print(f"Error during text generation for spice comparison: {e}")
            return f"I found the spice levels ({r1_name}: {spice1}, {r2_name}: {spice2}), but couldn't generate a comparison summary."


    # 3) NEW: Feature Comparison (Rating & Price Range)
    elif spec.get('feature_compare'):
        compare_info = spec['feature_compare']
        resto_names = compare_info.get('restaurants')
        feature = compare_info.get('feature') # 'rating' or 'price_range'

        if not resto_names or len(resto_names) != 2:
            return "Sorry, I need exactly two restaurant names to compare features."
        r1_name, r2_name = resto_names

        r1_data = find_restaurant(r1_name, restaurants)
        r2_data = find_restaurant(r2_name, restaurants)

        if not r1_data or not r2_data:
            missing_restos = []
            if not r1_data: missing_restos.append(r1_name)
            if not r2_data: missing_restos.append(r2_name)
            return f"Sorry, I couldn't find information for restaurant(s): {', '.join(missing_restos)}."

        # Prepare context based on the requested feature
        ctx = ""
        r1_val_str, r2_val_str = "N/A", "N/A"

        if feature == 'rating':
            r1_val_str = r1_data.get('rating', 'N/A')
            r2_val_str = r2_data.get('rating', 'N/A')
            ctx = f"{r1_name} Rating: {r1_val_str}\n{r2_name} Rating: {r2_val_str}"

            # Attempt numerical comparison
            r1_rating_num = parse_rating(r1_val_str)
            r2_rating_num = parse_rating(r2_val_str)

            comparison_summary = ""
            if r1_rating_num is not None and r2_rating_num is not None:
                if r1_rating_num > r2_rating_num:
                    comparison_summary = f"{r1_name} has a higher rating than {r2_name}."
                elif r2_rating_num > r1_rating_num:
                    comparison_summary = f"{r2_name} has a higher rating than {r1_name}."
                else:
                    comparison_summary = f"Both restaurants have a similar rating ({r1_val_str})."
            elif r1_val_str == 'N/A' or r2_val_str == 'N/A':
                 comparison_summary = "Rating information is missing for at least one restaurant, so I cannot compare them numerically."
            else:
                 comparison_summary = "I couldn't numerically compare the ratings, but here they are."

            ctx += f"\nComparison: {comparison_summary}"


        elif feature == 'price_range':
            r1_val_str = r1_data.get('price_range', 'N/A')
            r2_val_str = r2_data.get('price_range', 'N/A')
            ctx = f"{r1_name} Price Range: {r1_val_str}\n{r2_name} Price Range: {r2_val_str}"
            # Numerical comparison of price ranges is complex, rely on LLM or basic string comparison if needed

        else: # Should not happen if parser is correct
             return "Sorry, I can only compare 'rating' or 'price_range'."

        try:
            prompt = PROMPT.format(context=ctx, question=question)
            result = generator(prompt, max_length=150, do_sample=False)
            return result[0]['generated_text']
        except Exception as e:
            print(f"Error during text generation for feature comparison: {e}")
            return f"Here's the info I found:\n{ctx}" # Fallback

    # 4) Price Range Query (Uses restaurant list)
    elif spec.get('price_range_query'):
        resto_name = spec['price_range_query']
        resto_data = find_restaurant(resto_name, restaurants)
        if not resto_data:
            return f"Sorry, I couldn't find information for restaurant '{resto_name}'."
        price_range = resto_data.get('price_range', 'N/A')
        if price_range == 'N/A':
            return f"Sorry, the price range information is not available for '{resto_name}'."
        else:
            return f"The approximate price range for '{resto_name}' is: {price_range}."



    # 5) Price Filter
    elif spec.get('price_lt') is not None:
        thr = spec['price_lt']
        idxs = [i for i, c in enumerate(chunks)
                if c.get('price') is not None and isinstance(c.get('price'), (int, float)) and c['price'] < thr]
        if not idxs:
            return f"Sorry, I couldn't find any dishes under ₹{thr}."

        if spec.get('dietary'):
             tags = spec['dietary'] 

        if not idxs:
             return f"Sorry, no dishes found matching all criteria under ₹{thr}."

        if len(idxs) > k:
            selected = retrieve_chunks(question, embed_model, index, chunks, k=k, filter_indices=idxs)
        else:
            selected = [chunks[i] for i in idxs]

        dish_details = [f"{c.get('dish_name', '?')} (at {c.get('resto_id', '?')}) - ₹{c.get('price', 'N/A'):.2f}" for c in selected]
        unique_details = sorted(list(set(dish_details)))
        filter_desc = f"under ₹{thr}"
        # if spec.get('dietary'): filter_desc += f" matching {spec['dietary']}" # Adapt description
        return f"Here are some dishes {filter_desc}:{md_newline}" + md_newline.join(unique_details)


    # 6) Dietary Filters (Needs rework for veg/nonveg)
    elif spec.get('dietary'): # Assuming spec['dietary'] is adapted to 'Veg'/'Non-Veg' or similar
        return "Dietary filtering needs to be updated based on the 'veg_nonveg' field in your data."


    # 7) List Dishes at Restaurant
    elif spec.get('restaurant'):
        resto_id = spec['restaurant']
        names = list_dishes(resto_id, chunks)
        if not names:
            return f"Sorry, I couldn't find any dishes listed for the restaurant '{resto_id}'."
        return f"Dishes available at {resto_id}:{md_newline}" + md_newline.join(names)

    # --- Fallback: Full RAG ---
    else:
        print(f"Performing general RAG for question: {question}")
        try:
            top = retrieve_chunks(question, embed_model, index, chunks, k=k)
            if not top:
                return "Sorry, I couldn't find relevant information to answer your question."
            ctx_parts = []
            seen_texts = set()
            for c in top:
                text = c.get('text', '')
                if text not in seen_texts:
                     ctx_parts.append(text)
                     seen_texts.add(text)
            ctx = "\n---\n".join(ctx_parts)
            prompt = PROMPT.format(context=ctx, question=question)
            result = generator(prompt, max_length=200, do_sample=False)
            final_answer = result[0]['generated_text']
            # Add note about unavailable info if relevant
            if any(f in question.lower() for f in ['cuisine', 'vegan', 'gluten']):
                 final_answer += "\n\n(Note: Specific details like cuisine types, vegan, or gluten-free options might not be available in the source data.)"
            return final_answer.strip()
        except Exception as e:
            print(f"Error during fallback RAG: {e}")
            return "Sorry, I encountered an error while trying to find an answer."