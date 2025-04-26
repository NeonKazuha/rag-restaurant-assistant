import re
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_query(question):

    spec = {
        'price_lt': None,
        'dietary': [],
        'spice_cmp': None,
        'restaurant': None,
        'dish_price_query': None,
        'feature_compare': None,
        'price_range_query': None
    }

    question_lower = question.lower()

    dish_price_match = re.search(r"(?:price|prices|cost|how much is|how much are)\s+(?:of|for)?\s+(.+?)(?:\s+across|\s+at all|\s+in all|\?|$)", question, re.IGNORECASE)
    if dish_price_match:
        dish_name = dish_price_match.group(1).strip().rstrip('.?!')
        if 2 < len(dish_name) < 50:
            spec['dish_price_query'] = dish_name
            logging.info(f"Detected dish price query for: '{dish_name}'")
            return spec

    spice_match = re.search(r"compare spice.*? dish (.+?) between (.+?) and (.+)", question, re.IGNORECASE)
    if spice_match:
        spec['spice_cmp'] = {
            'dish': spice_match.group(1).strip(),
            'restaurants': [spice_match.group(2).strip(), spice_match.group(3).strip()]
        }
        logging.info(f"Extracted spice comparison: {spec['spice_cmp']}")
        return spec

    compare_match = re.search(r"(?:compare|difference)\s+(?:(?:rating|price range)\s+)?(?:between|for)\s+(.+?)\s+and\s+(.+)", question, re.IGNORECASE) \
                  or re.search(r"which is (better|higher rated|cheaper|more expensive)\s*,?\s*(.+?)\s+or\s+(.+)", question, re.IGNORECASE) \
                  or re.search(r"what'?s the (rating|price)\s*(?:difference|range)?\s*(?:between|for)\s+(.+?)\s+and\s+(.+)", question, re.IGNORECASE)

    if compare_match:
        groups = compare_match.groups()
        feature = None
        qualifier = ""
        matched_text = compare_match.group(0).lower()
        if 'rating' in matched_text or 'rated' in matched_text or 'better' in matched_text:
            feature = 'rating'
            if 'better' in matched_text or 'higher' in matched_text: qualifier = 'higher'
        elif 'price' in matched_text or 'cheaper' in matched_text or 'expensive' in matched_text:
            feature = 'price_range'
            if 'cheaper' in matched_text: qualifier = 'cheaper'
            if 'expensive' in matched_text: qualifier = 'more_expensive'

        r1, r2 = None, None
        if len(groups) == 3:
              if groups[0].lower() in ['better', 'higher rated', 'cheaper', 'more expensive', 'rating', 'price']:
                  r1 = groups[1].strip().rstrip('?.!')
                  r2 = groups[2].strip().rstrip('?.!')
              else:
                  pass
        elif len(groups) == 2:
              r1 = groups[0].strip().rstrip('?.!')
              r2 = groups[1].strip().rstrip('?.!')

        if r1 and r2 and feature:
            spec['feature_compare'] = {'restaurants': [r1, r2], 'feature': feature, 'qualifier': qualifier}
            logging.info(f"Detected feature comparison query: {spec['feature_compare']}")
            return spec

    price_range_match = re.search(r"(?:is|are)\s+(.+?)\s+(expensive|cheap|pricey|mid-range|affordable)\b", question, re.IGNORECASE) \
                      or re.search(r"what'?s the price range for\s+(.+)", question, re.IGNORECASE) \
                      or re.search(r"how expensive is\s+(.+)", question, re.IGNORECASE)
    if price_range_match:
        if not spec['feature_compare']:
            resto_name = price_range_match.group(1).strip().rstrip('?.!')
            if ' and ' not in resto_name.lower():
                if 2 < len(resto_name) < 50:
                    spec['price_range_query'] = resto_name
                    logging.info(f"Detected price range query for: '{resto_name}'")
                    return spec

    is_other_spec_set = any(spec[key] for key in ['dish_price_query', 'spice_cmp', 'feature_compare', 'price_range_query'])
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

    price_match = re.search(r"(?:under|less than|below)\s*₹?(\d+)", question, re.IGNORECASE)
    if price_match:
        spec['price_lt'] = int(price_match.group(1))
        logging.info(f"Extracted price filter: < {spec['price_lt']}")

    if re.search(r'\bvegan\b', question_lower): spec['dietary'].append('vegan')
    if re.search(r'gluten[\s-]?free\b', question_lower): spec['dietary'].append('gluten_free')
    if re.search(r'\bvegetarian\b', question_lower): spec['dietary'].append('vegetarian')

    if spec['dietary']: logging.info(f"Extracted dietary filters: {spec['dietary']}")


    logging.info(f"Final parsed query spec (no specific intent matched or only filters): {spec}")
    return spec