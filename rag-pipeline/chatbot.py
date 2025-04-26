import google.generativeai as genai
import os
import re
import numpy as np
from dotenv import load_dotenv
from index_faiss import retrieve_chunks
from query_planner import parse_query

load_dotenv()

# --- Configure Gemini API ---
gemini_model = None
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Warning: GEMINI_API_KEY environment variable not set. RAG fallback will be disabled.")
else:
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# --- Prompt Template for Gemini RAG Fallback ---
PROMPT_TEMPLATE = """
Use ONLY the following context chunks to answer the user's question accurately.
Do not add information not present in the context.
If the context doesn't contain the answer, state that the information is not available in the provided data.
If the question is subjective (e.g., "best dish"), list relevant options from the context instead of making a judgment.

Context Chunks:
---
{context}
---

User Question: {question}

Answer:
"""

# --- Helper Functions ---
def find_restaurant(name, restaurants):
    """Finds the first matching restaurant dict by name (case-insensitive)."""
    for resto in restaurants:
        if resto.get('name', '').strip().lower() == name.strip().lower():
            return resto
    return None

def list_dishes(resto_id, chunks):
    """Retrieves unique dish names for a specific restaurant ID (case-insensitive)."""
    names = {c.get('dish_name', 'Unknown Dish') for c in chunks if c.get('resto_id', '').lower() == resto_id.lower()}
    return sorted(list(names))

def parse_rating(rating_str):
    """Attempts to extract the first numerical rating found in a string."""
    if not isinstance(rating_str, str): return None
    match = re.search(r"(\d+(?:\.\d+)?)", rating_str) # Extracts number like 4.5 or 4
    if match:
        try: return float(match.group(1))
        except ValueError: return None
    return None

# --- Main Answering Function (Hybrid Approach) ---
def answer_hybrid(question, embed_model, index, chunks, restaurants, k=10):
    """
    Answers questions by checking specific intents first, then uses RAG+Gemini as fallback.
    """
    if not restaurants:
         return "Sorry, the restaurant data could not be loaded for specific queries."

    try:
        spec = parse_query(question)
    except Exception as e:
        print(f"Error parsing query: {e}")
        return "Sorry, I had trouble understanding your request."

    md_newline = "  \n" # Markdown newline for formatting lists

    # --- Handling Specific Intents Directly ---

    # 1) Dish Price Query
    if spec.get('dish_price_query'):
        dish_name = spec['dish_price_query']
        matching_chunks = [c for c in chunks if c.get('dish_name', '').lower() == dish_name.lower()]
        if not matching_chunks:
            return f"Sorry, I couldn't find the dish '{dish_name}' in any restaurant."

        results = []
        for chunk in matching_chunks:
            resto_id = chunk.get('resto_id', 'Unknown Restaurant')
            price = chunk.get('price', 'N/A')
            price_str = f"₹{price:.2f}" if isinstance(price, (int, float)) else str(price)
            results.append(f"- {resto_id}: {price_str}")

        if not results:
             return f"Sorry, I found the dish '{dish_name}' but couldn't retrieve price information."

        unique_results = sorted(list(set(results)))
        return f"Prices for '{dish_name}':{md_newline}" + md_newline.join(unique_results)

    # 2) Spice Comparison
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
            missing_restos = [name for name, data in [(r1_name, d1), (r2_name, d2)] if data is None]
            return f"Sorry, I couldn’t find the dish '{dish}' in restaurant(s): {', '.join(missing_restos)}."

        spice1 = d1.get('attributes', {}).get('spice_level', 'Not specified')
        spice2 = d2.get('attributes', {}).get('spice_level', 'Not specified')

        # Direct response format for spice comparison
        return (f"For the dish '{dish}':{md_newline}"
                f"- Spice level at {r1_name}: {spice1}{md_newline}"
                f"- Spice level at {r2_name}: {spice2}")

    # 3) Feature Comparison (Rating & Price Range)
    elif spec.get('feature_compare'):
        compare_info = spec['feature_compare']
        resto_names = compare_info.get('restaurants')
        feature = compare_info.get('feature')

        if not resto_names or len(resto_names) != 2:
            return "Sorry, I need exactly two restaurant names to compare features."
        r1_name, r2_name = resto_names

        r1_data = find_restaurant(r1_name, restaurants)
        r2_data = find_restaurant(r2_name, restaurants)

        if not r1_data or not r2_data:
            missing_restos = [name for name, data in [(r1_name, r1_data), (r2_name, r2_data)] if data is None]
            return f"Sorry, I couldn't find information for restaurant(s): {', '.join(missing_restos)}."

        response_parts = []
        if feature == 'rating':
            r1_val_str = r1_data.get('rating', 'N/A')
            r2_val_str = r2_data.get('rating', 'N/A')
            response_parts.append(f"Comparing ratings for '{r1_name}' and '{r2_name}':")
            response_parts.append(f"- {r1_name} Rating: {r1_val_str}")
            response_parts.append(f"- {r2_name} Rating: {r2_val_str}")

            r1_rating_num = parse_rating(r1_val_str)
            r2_rating_num = parse_rating(r2_val_str)
            if r1_rating_num is not None and r2_rating_num is not None:
                if r1_rating_num > r2_rating_num:
                    response_parts.append(f"-> {r1_name} has a higher rating.")
                elif r2_rating_num > r1_rating_num:
                    response_parts.append(f"-> {r2_name} has a higher rating.")
                else:
                    response_parts.append("-> Both restaurants have a similar rating.")
            else:
                 response_parts.append("-> Cannot numerically compare ratings due to missing or non-standard data.")

        elif feature == 'price_range':
            r1_val_str = r1_data.get('price_range', 'N/A')
            r2_val_str = r2_data.get('price_range', 'N/A')
            response_parts.append(f"Comparing price ranges for '{r1_name}' and '{r2_name}':")
            response_parts.append(f"- {r1_name}: {r1_val_str}")
            response_parts.append(f"- {r2_name}: {r2_val_str}")
        else:
             return "Sorry, I can only compare 'rating' or 'price_range'."

        return md_newline.join(response_parts)

    # 4) Price Range Query
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

    # 5) List Dishes at Restaurant
    elif spec.get('restaurant'):
        resto_id = spec['restaurant']
        names = list_dishes(resto_id, chunks)
        if not names:
            resto_data = find_restaurant(resto_id, restaurants)
            if resto_data:
                 return f"Sorry, I couldn't find specific dishes listed for '{resto_id}', but the restaurant exists in the data."
            else:
                 return f"Sorry, I couldn't find the restaurant '{resto_id}' or any dishes listed for it."
        return f"Dishes available at {resto_id}:{md_newline}" + md_newline.join(names)

    # 6) Price Filter (Standalone)
    elif spec.get('price_lt') is not None:
        thr = spec['price_lt']
        idxs = [i for i, c in enumerate(chunks)
                if c.get('price') is not None and isinstance(c.get('price'), (int, float)) and c['price'] < thr]
        if not idxs:
            return f"Sorry, I couldn't find any dishes under ₹{thr}."

        diet_filter = None

        # Retrieve/select chunks
        if len(idxs) > k:
             # Use semantic search on the filtered subset if it's large
             refined_query = f"Dishes under {thr}"
             if diet_filter: refined_query += f" that are {diet_filter}"
             selected = retrieve_chunks(refined_query, embed_model, index, chunks, k=k, filter_indices=idxs)
        elif idxs:
             selected = [chunks[i] for i in idxs]
        else:
             selected = []

        if not selected:
             return f"Sorry, no matching dishes found for your criteria."

        dish_details = [f"{c.get('dish_name', '?')} (at {c.get('resto_id', '?')}) - ₹{c.get('price', 'N/A'):.2f}" for c in selected]
        unique_details = sorted(list(set(dish_details)))
        filter_desc = f"under ₹{thr}"
        if diet_filter: filter_desc += f" ({diet_filter})"
        return f"Here are some dishes {filter_desc}:{md_newline}" + md_newline.join(unique_details)

    # --- Fallback: RAG with Gemini API ---
    else:
        if gemini_model is None:
            return "Sorry, I cannot answer this general question as the advanced AI service is unavailable."

        print(f"No specific intent matched. Performing RAG + Gemini fallback for: {question}")
        top_chunks = retrieve_chunks(question, embed_model, index, chunks, k=k)
        if not top_chunks:
            print("No relevant chunks found by semantic search for Gemini.")
            context_str = "No specific information found in the knowledge base for this question."
        else:
            # Prepare context from retrieved chunk text
            context_str = "\n---\n".join(chunk.get('text', '') for chunk in top_chunks)

        prompt = PROMPT_TEMPLATE.format(context=context_str, question=question)

        print("Sending request to Gemini API...")
        try:
            # Example safety settings (adjust thresholds as needed)
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
            response = gemini_model.generate_content(prompt, safety_settings=safety_settings)

            # Handle potential blocks or empty responses
            if not response.candidates:
                 print("Gemini API response blocked or empty.")
                 try:
                     feedback = response.prompt_feedback
                     print(f"Prompt Feedback: {feedback}")
                     block_reason = getattr(feedback, 'block_reason', 'Unknown')
                     return f"Sorry, the request could not be completed due to safety filters (Reason: {block_reason}). Please rephrase your question."
                 except Exception:
                     return "Sorry, the request could not be completed due to safety filters. Please rephrase your question."

            generated_text = response.text
            print("Received response from Gemini API.")
            return generated_text.strip()

        except ValueError as ve:
             # Handle potential API key errors during generation
             if "API_KEY" in str(ve):
                  print(f"Gemini API Key Error: {ve}")
                  return "Sorry, there's an issue with the AI service configuration (API Key)."
             else:
                  print(f"Gemini API Value Error: {ve}")
                  return f"Sorry, an error occurred while generating the response: {ve}"
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "Sorry, I encountered an error while generating the response."





