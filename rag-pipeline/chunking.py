import json

def build_dish_chunks(restaurants):
    """
    Turn each restaurant's menu items into individual chunks
    with metadata based on the provided knowledgebase.json structure.
    Includes available restaurant-level info in each chunk.
    """
    chunks = []
    for resto in restaurants:
        resto_id = resto.get('name', 'Unknown Restaurant')
        resto_price_range = resto.get('price_range', 'N/A')
        resto_dietary_options = resto.get('dietary_options', [])
        resto_address = resto.get('address', '')
        resto_phone = resto.get('phone_number', '')

        for item in resto.get('menu_items', []):
            dish_name = item.get('name', 'Unknown Dish')
            desc = item.get('description', '')
            price = item.get('price', None)
            attrs = item.get('attributes', {})

            txt = f"{dish_name}: {desc}."
            if price is not None:
                try:
                    txt += f" Price ₹{float(price):.2f}."
                except (ValueError, TypeError):
                    txt += f" Price {price}."

            tag_parts = []
            veg_status = attrs.get('veg_nonveg')
            if veg_status:
                tag_parts.append(f"Type: {veg_status}")
            spice = attrs.get('spice_level')
            if spice is not None:
                try:
                    tag_parts.append(f"Spice Level: {int(spice)}")
                except (ValueError, TypeError):
                    tag_parts.append(f"Spice Level: {spice}")
            category = attrs.get('category')
            if category:
                tag_parts.append(f"Category: {category}")

            if tag_parts:
                txt += " Attributes: " + ", ".join(tag_parts) + "."

            txt += f" Restaurant: {resto_id}."
            if resto_price_range != 'N/A':
                txt += f" General Price Range: {resto_price_range}."

            chunks.append({
                'resto_id': resto_id,
                'dish_name': dish_name,
                'text': txt,
                'price': price,
                'attributes': attrs,
                'resto_address': resto_address,
                'resto_phone_number': resto_phone,
                'resto_price_range': resto_price_range,
                'resto_dietary_options': resto_dietary_options,
                'dish_description': desc
            })
    return chunks