import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Query Parsing Function ---
def parse_query(question):
    '''
    Parses the user question to extract intent and parameters.

    Args:
        question (str): The user's question.

    Returns:
        dict: A dictionary containing parsed specifications like
              price_lt, dietary, spice_cmp, restaurant.
              Example: {'price_lt': None, 'dietary': [], 'spice_cmp': None, 'restaurant': None}
    '''
    spec = {'price_lt': None, 'dietary': [], 'spice_cmp': None, 'restaurant': None}

    # Price check
    price_match = re.search(r"(?:under|less than|below)\s*₹?(\d+)", question, re.IGNORECASE)
    if price_match:
        spec['price_lt'] = int(price_match.group(1))
        logging.info(f"Extracted price filter: < {spec['price_lt']}")

    # Dietary checks (add more as needed)
    if re.search(r'\bvegan\b', question, re.IGNORECASE):
        spec['dietary'].append('vegan')
    if re.search(r'gluten[\s-]?free\b', question, re.IGNORECASE):
        spec['dietary'].append('gluten_free')
    if re.search(r'\bvegetarian\b', question, re.IGNORECASE):
         spec['dietary'].append('vegetarian')
    # Add more dietary flags based on your JSON attributes if needed
    # Example: if 'keto' in question.lower(): spec['dietary'].append('keto_friendly')

    if spec['dietary']:
        logging.info(f"Extracted dietary filters: {spec['dietary']}")


    # Spice comparison check
    spice_match = re.search(r"compare spice.*? dish (.+?) between (.+?) and (.+)", question, re.IGNORECASE)
    if spice_match:
        spec['spice_cmp'] = {
            'dish': spice_match.group(1).strip(),
            'restaurants': [spice_match.group(2).strip(), spice_match.group(3).strip()]
        }
        logging.info(f"Extracted spice comparison: {spec['spice_cmp']}")


    # Restaurant Name Extraction (Improved Regex)
    # Looks for patterns like "menu at/in/from RESTAURANT_NAME", "dishes at/in/from RESTAURANT_NAME",
    # or just "RESTAURANT_NAME menu/dishes"
    # Assumes restaurant names don't contain "menu" or "dishes" and are reasonably distinct.
    # This is still a simplification and might need a proper Named Entity Recognition (NER) model for complex cases.
    resto_patterns = [
        r"(?:menu|dishes)\s+(?:at|in|from)\s+(.+)", # "menu at Restaurant Name"
        r"(.+?)\s+(?:menu|dishes)" # "Restaurant Name menu"
    ]
    for pattern in resto_patterns:
        resto_match = re.search(pattern, question, re.IGNORECASE)
        if resto_match:
            # Extract the potential name, remove trailing punctuation/question marks if any
            potential_name = resto_match.group(1).strip().rstrip('.?!')
            # Basic check to avoid capturing overly long/generic phrases
            if len(potential_name) > 2 and len(potential_name) < 50: # Adjust length limits as needed
                spec['restaurant'] = potential_name
                logging.info(f"Extracted potential restaurant name: '{spec['restaurant']}' using pattern: '{pattern}'")
                break # Stop after first match

    # If a restaurant name was found, clear other filters as the intent is likely just the menu
    # (Optional: You might want to allow combined queries like "vegan dishes at Restaurant X")
    # If you want to allow combined queries, remove this block.
    if spec.get('restaurant') and (spec.get('price_lt') or spec.get('dietary')):
         logging.warning("Query contains both restaurant name and other filters. Prioritizing restaurant menu listing.")
         spec['price_lt'] = None
         spec['dietary'] = []
         spec['spice_cmp'] = None # Also clear spice comparison if restaurant is named


    logging.info(f"Final parsed query spec: {spec}")
    return spec
