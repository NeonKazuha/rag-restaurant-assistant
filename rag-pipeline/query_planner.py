import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Query Parsing Function ---
def parse_query(question):
    '''
    Parses the user question to extract intent and parameters.
    Includes detection for feature comparison (rating, price_range).
    '''
    # Add 'feature_compare' and 'price_range_query' keys
    spec = {
        'price_lt': None,
        'dietary': [], # Keep this for potential future dietary tags
        'spice_cmp': None,
        'restaurant': None,
        'dish_price_query': None,
        'feature_compare': None, # NEW: For comparing restaurant features (rating, price_range)
        'price_range_query': None # NEW: For asking about restaurant price range
    }

    question_lower = question.lower()

    # --- Check for specific intents first ---

    # Specific dish price query
    dish_price_match = re.search(r"(?:price|prices|cost|how much is|how much are)\s+(?:of|for)?\s+(.+?)(?:\s+across|\s+at all|\s+in all|\?|$)", question, re.IGNORECASE)
    if dish_price_match:
        dish_name = dish_price_match.group(1).strip().rstrip('.?!')
        if 2 < len(dish_name) < 50:
            spec['dish_price_query'] = dish_name
            logging.info(f"Detected dish price query for: '{dish_name}'")
            return spec

    # Spice comparison
    spice_match = re.search(r"compare spice.*? dish (.+?) between (.+?) and (.+)", question, re.IGNORECASE)
    if spice_match:
        spec['spice_cmp'] = {
            'dish': spice_match.group(1).strip(),
            'restaurants': [spice_match.group(2).strip(), spice_match.group(3).strip()]
        }
        logging.info(f"Extracted spice comparison: {spec['spice_cmp']}")
        return spec

    # --- NEW: Restaurant Feature Comparison (Rating and Price Range) ---
    # Pattern: "compare [rating/price range] between/for [A] and [B]"
    # Pattern: "which is better/higher rated/cheaper/more expensive, [A] or [B]?"
    # Pattern: "what's the rating/price difference between [A] and [B]?"
    compare_match = re.search(r"(?:compare|difference)\s+(?:(?:rating|price range)\s+)?(?:between|for)\s+(.+?)\s+and\s+(.+)", question, re.IGNORECASE) \
                 or re.search(r"which is (better|higher rated|cheaper|more expensive)\s*,?\s*(.+?)\s+or\s+(.+)", question, re.IGNORECASE) \
                 or re.search(r"what'?s the (rating|price)\s*(?:difference|range)?\s*(?:between|for)\s+(.+?)\s+and\s+(.+)", question, re.IGNORECASE)

    if compare_match:
        groups = compare_match.groups()
        feature = None
        qualifier = "" # e.g., "better", "cheaper"
        # Determine feature and qualifier based on matched words
        matched_text = compare_match.group(0).lower()
        if 'rating' in matched_text or 'rated' in matched_text or 'better' in matched_text:
            feature = 'rating'
            if 'better' in matched_text or 'higher' in matched_text: qualifier = 'higher'
        elif 'price' in matched_text or 'cheaper' in matched_text or 'expensive' in matched_text:
            feature = 'price_range'
            if 'cheaper' in matched_text: qualifier = 'cheaper'
            if 'expensive' in matched_text: qualifier = 'more_expensive'

        r1, r2 = None, None
        # Extract restaurant names based on which regex pattern matched
        if len(groups) == 3: # Matched 2nd or 3rd pattern
             if groups[0].lower() in ['better', 'higher rated', 'cheaper', 'more expensive', 'rating', 'price']:
                 # Qualifier/Feature was group 1
                 r1 = groups[1].strip().rstrip('?.!')
                 r2 = groups[2].strip().rstrip('?.!')
             else: # Should not happen with these patterns, but maybe handle unexpected cases
                 pass
        elif len(groups) == 2: # Matched 1st pattern
             r1 = groups[0].strip().rstrip('?.!')
             r2 = groups[1].strip().rstrip('?.!')

        if r1 and r2 and feature:
            spec['feature_compare'] = {'restaurants': [r1, r2], 'feature': feature, 'qualifier': qualifier}
            logging.info(f"Detected feature comparison query: {spec['feature_compare']}")
            return spec # Return early for specific comparison

    # Price Range Inquiry (for a single restaurant)
    price_range_match = re.search(r"(?:is|are)\s+(.+?)\s+(expensive|cheap|pricey|mid-range|affordable)\b", question, re.IGNORECASE) \
                     or re.search(r"what'?s the price range for\s+(.+)", question, re.IGNORECASE) \
                     or re.search(r"how expensive is\s+(.+)", question, re.IGNORECASE)
    if price_range_match:
        # Avoid conflict with comparison query if both names were mentioned
        if not spec['feature_compare']:
            resto_name = price_range_match.group(1).strip().rstrip('?.!')
            # Simple check to avoid matching parts of comparison query
            if ' and ' not in resto_name.lower():
                if 2 < len(resto_name) < 50:
                    spec['price_range_query'] = resto_name
                    logging.info(f"Detected price range query for: '{resto_name}'")
                    return spec # Return early

    # --- Existing Filters (apply if no specific query matched AND returned early) ---

    # Price filter for dishes
    price_match = re.search(r"(?:under|less than|below)\s*₹?(\d+)", question, re.IGNORECASE)
    if price_match:
        spec['price_lt'] = int(price_match.group(1))
        logging.info(f"Extracted price filter: < {spec['price_lt']}")

    # Dietary checks (Based on 'veg_nonveg' - update if needed)
    # Kept original logic for now, adjust based on chatbot implementation needs
    if re.search(r'\bvegan\b', question, re.IGNORECASE): spec['dietary'].append('vegan') # May need re-evaluation based on data
    if re.search(r'gluten[\s-]?free\b', question, re.IGNORECASE): spec['dietary'].append('gluten_free') # May need re-evaluation
    if re.search(r'\bvegetarian\b', question, re.IGNORECASE): spec['dietary'].append('vegetarian') # May need re-evaluation
    if spec['dietary']: logging.info(f"Extracted dietary filters: {spec['dietary']}")

    # Restaurant Name Extraction (for menu listing)
    # Check if no other major spec is set
    is_other_spec_set = any(spec[key] for key in spec if key not in ['price_lt', 'dietary'] and spec[key] is not None)
    if not is_other_spec_set:
        resto_patterns = [
            r"(?:menu|dishes)\s+(?:at|in|from)\s+(.+)",
            r"(.+?)\s+(?:menu|dishes)"
        ]
        for pattern in resto_patterns:
            resto_match = re.search(pattern, question, re.IGNORECASE)
            if resto_match:
                potential_name = resto_match.group(1).strip().rstrip('.?!')
                if 2 < len(potential_name) < 50:
                    spec['restaurant'] = potential_name
                    logging.info(f"Extracted restaurant name for menu listing: '{spec['restaurant']}'")
                    spec['price_lt'] = None
                    spec['dietary'] = []
                    return spec

    logging.info(f"Final parsed query spec: {spec}")
    return spec