def build_dish_chunks(restaurants):
    """
    Turn each restaurant's menu items into individual chunks
    with metadata for hybrid filtering and RAG retrieval.
    """
    chunks = []
    for resto in restaurants:
        resto_id = resto['name']
        for item in resto.get('menu', []):
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
                elif v is True:
                    tag_parts.append(k.replace('_',' '))
            if tag_parts:
                txt += " Attributes: " + ", ".join(tag_parts) + "."
            chunks.append({
                'resto_id': resto_id,
                'dish_name': dish_name,
                'text': txt,
                'price': price,
                'attributes': attrs
            })
    return chunks
