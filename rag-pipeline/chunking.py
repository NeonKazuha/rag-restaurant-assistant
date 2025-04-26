import json # Import json module if not already imported

def build_dish_chunks(restaurants):
    """
    Turn each restaurant's menu items into individual chunks
    with metadata for hybrid filtering and RAG retrieval,
    including comprehensive restaurant details.
    """
    chunks = []
    for resto in restaurants:
        # --- Get restaurant-level info ---
        resto_name = resto.get('name', '')
        dietary_options = resto.get('dietary_options', [])
        price_range = resto.get('price_range', '')
        address = resto.get('address', '')
        opening_hours = resto.get('opening_hours', '')
        phone_number = resto.get('phone_number', '')
        rating = resto.get('rating', '')
        rating_count = resto.get('rating_count', None)
        features = resto.get('features', '').strip() # Use strip() to remove leading/trailing whitespace
        cuisine = resto.get('cuisine', '')
        # ---------------------------------

        # --- Create restaurant information string ---
        resto_info_parts = [
            f"Restaurant: {resto_name}",
            f"Address: {address}",
            f"Phone: {phone_number}",
            f"Cuisine: {cuisine}",
            f"Opening Hours: {opening_hours}",
            f"Price Range: {price_range}",
            f"Rating: {rating} ({rating_count} ratings)" if rating and rating_count else (f"Rating: {rating}" if rating else ""),
            f"Features: {features}" if features else "",
            f"Dietary Options: {', '.join(dietary_options)}" if dietary_options else ""
        ]
        # Filter out empty strings
        resto_info = ". ".join(part for part in resto_info_parts if part)
        # ------------------------------------------

        for item in resto.get('menu_items', []): # Use 'menu_items' based on knowledgebase.json
            dish_name = item['name']
            desc      = item.get('description', '')
            price     = item.get('price', None)
            attrs     = item.get('attributes', {})

            # --- Build the text blob for the chunk ---
            # Start with dish info
            dish_info = f"Dish: {dish_name}"
            if desc:
                 dish_info += f": {desc}"
            if price is not None:
                dish_info += f". Price ₹{price}"

            # Add dish attributes
            tag_parts = []
            if attrs:
                # Add veg/non-veg status if present
                if 'veg_nonveg' in attrs:
                     tag_parts.append(attrs['veg_nonveg'])
                # Add spice level if present
                if 'spice_level' in attrs:
                    tag_parts.append(f"spice level {attrs['spice_level']}")
                # Add category if present
                if 'category' in attrs:
                    tag_parts.append(f"category: {attrs['category']}")
                # Add other boolean attributes or specific string attributes if needed

            if tag_parts:
                dish_info += ". Attributes: " + ", ".join(tag_parts)

            # Combine restaurant info and dish info for the final text chunk
            txt = f"{resto_info}. {dish_info}."
            # ------------------------------------------

            # --- Create the chunk dictionary with metadata ---
            chunk_metadata = {
                'resto_name': resto_name,
                'dish_name': dish_name,
                'text': txt, # The combined text blob
                # Restaurant metadata
                'dietary_options': dietary_options,
                'price_range': price_range,
                'address': address,
                'opening_hours': opening_hours,
                'phone_number': phone_number,
                'rating': rating,
                'rating_count': rating_count,
                'features': features,
                'cuisine': cuisine,
                # Dish metadata
                'price': price,
                'attributes': attrs
            }
            chunks.append(chunk_metadata)
            # -------------------------------------------------

    return chunks

# Example usage (assuming 'knowledgebase.json' is in the same directory):
# try:
#     with open('knowledgebase.json', 'r', encoding='utf-8') as f:
#         restaurants_data = json.load(f)
#     all_chunks = build_dish_chunks(restaurants_data)
#     # Now you can use 'all_chunks' to feed into your RAG system's vector store/index
#     if all_chunks:
#        print(f"Generated {len(all_chunks)} chunks.")
#        # print("\nExample chunk:")
#        # print(json.dumps(all_chunks[0], indent=2)) # Print the first chunk as an example
# except FileNotFoundError:
#     print("Error: knowledgebase.json not found.")
# except json.JSONDecodeError:
#     print("Error: Could not decode knowledgebase.json.")