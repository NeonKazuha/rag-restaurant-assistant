import re
# --- Query Parsing Function ---
def parse_query(question):
    '''
    Parses the user question to extract intent and parameters.
    (This is a placeholder - the actual implementation should be here)

    Args:
        question (str): The user's question.

    Returns:
        dict: A dictionary containing parsed specifications like
              price_lt, dietary, spice_cmp, restaurant.
              Example: {'price_lt': None, 'dietary': ['vegan'], 'spice_cmp': None, 'restaurant': None}
    '''
    # Placeholder implementation - Replace with your actual parsing logic
    spec = {'price_lt': None, 'dietary': [], 'spice_cmp': None, 'restaurant': None}

    # Price check
    price_match = re.search(r"(?:under|less than|below)\s*₹?(\d+)", question, re.IGNORECASE)
    if price_match:
        spec['price_lt'] = int(price_match.group(1))

    # Dietary checks (add more as needed)
    if re.search(r'\bvegan\b', question, re.IGNORECASE):
        spec['dietary'].append('vegan')
    if re.search(r'gluten[\s-]?free\b', question, re.IGNORECASE):
        spec['dietary'].append('gluten_free')
    if re.search(r'\bvegetarian\b', question, re.IGNORECASE):
         spec['dietary'].append('vegetarian')

    # Spice comparison check
    spice_match = re.search(r"compare spice.*? dish (.+?) between (.+?) and (.+)", question, re.IGNORECASE)
    if spice_match:
        spec['spice_cmp'] = {
            'dish': spice_match.group(1).strip(),
            'restaurants': [spice_match.group(2).strip(), spice_match.group(3).strip()]
        }

    # Using NER
    resto_match = re.search(r"(?:dishes|menu) at (.+)", question, re.IGNORECASE)
    if resto_match:
        spec['restaurant'] = resto_match.group(1).strip()


    print(f"Parsed query spec: {spec}")
    return spec

