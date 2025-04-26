def build_dish_chunks(restaurants):
    """
    Turn each restaurant's menu items into individual chunks
    with metadata for hybrid filtering and RAG retrieval.
    """
    chunks = []
    for resto in restaurants:
        resto_id = resto['name']
        for item in resto.get('menu_items', []):
            dish_name = item['name']
            desc      = item.get('description', '')
            price     = item.get('price', None)
            attrs     = item.get('attributes', {})
            # build text blob
            txt = f"{dish_name}: {desc}."
            if price is not None:
                txt += f" Price ₹{price}."
            # tags from attributes
            tag_parts = []
            for k, v in attrs.items():
                if k == 'spice_level':
                    tag_parts.append(f"spice level {v}")
                elif isinstance(v, bool) and v is True:
                    tag_parts.append(k.replace('_',' '))
                elif isinstance(v, str):
                    tag_parts.append(f"{k.replace('_',' ')}: {v}")
            if tag_parts:
                txt += " Attributes: " + ", ".join(tag_parts) + "."
            chunks.append({
                'resto_id': resto_id,
                'dish_name': dish_name,
                'text': txt,
                'price': price,
                'attributes': attrs,
                'address': resto.get('address', ''),
                'phone_number': resto.get('phone_number', ''),
                'rating': resto.get('rating', ''),
                'rating_count': resto.get('rating_count', 0),
                'cuisine': resto.get('cuisine', '')
            })
    return chunks