import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Feature Extraction
def extract_features(description, tags):
    """Extract features like spice counter, sweet counter, gluten-free, dietary tags, allergens, etc."""
    description = description.lower()

    keywords = {
        "spicy": {"spicy": 1, "hot": 1, "chili": 1, "masala": 1, "pepper": 1, "jalapeno": 2, "curry": 1, "szechuan": 2, "peri-peri": 2, "wasabi": 3, "fiery": 2, "zesty": 1, "pungent": 2},
        "sweet": {"sweet": 1, "dessert": 2, "sugar": 1, "honey": 1, "chocolate": 2, "caramel": 2, "vanilla": 1, "candy": 1, "pudding": 1, "cake": 2, "brownie": 2, "syrupy": 2, "custard": 1},
        "gluten_free": ["gluten-free", "gluten free", "gf", "celiac-friendly", "no gluten", "without gluten", "wheat-free"],
        "preparation": ["grilled", "fried", "baked", "steamed", "raw", "smoked", "poached", "sautéed", "seared"],
        "dish_characteristics": ["savory", "dessert", "appetizer", "main course", "beverage", "snack", "healthy", "organic", "comfort food", "street food", "brunch"],
        "cuisine": ["indian", "chinese", "italian", "mexican", "thai", "japanese", "continental", "mediterranean", "american", "fast food", "korean", "french", "vietnamese", "greek"],
        "dietary": ["vegan", "vegetarian", "keto", "paleo", "low-carb", "low fat", "high protein"],
        "allergens": ["peanut", "nut", "soy", "dairy", "milk", "egg", "shellfish", "wheat", "sesame"]
    }

    # Weighted counters
    spice_counter = sum(description.count(word) * weight for word, weight in keywords["spicy"].items())
    sweet_counter = sum(description.count(word) * weight for word, weight in keywords["sweet"].items())

    # Match preparation, dish characteristics, cuisine, dietary, allergens
    preparation_tags = [prep for prep in keywords["preparation"] if prep in description]
    dish_tags = [char for char in keywords["dish_characteristics"] if char in description]
    cuisine_tags = [cuisine for cuisine in keywords["cuisine"] if cuisine in description]
    dietary_tags = [diet for diet in keywords["dietary"] if diet in description]
    allergens = [allergen for allergen in keywords["allergens"] if allergen in description]

    # Gluten-free detection
    gluten_free = any(phrase in description for phrase in keywords["gluten_free"])

    features = {
        "spice_counter": spice_counter,
        "sweet_counter": sweet_counter,
        "gluten_free": gluten_free,
        "preparation_tags": preparation_tags,
        "dish_tags": dish_tags,
        "cuisine_tags": cuisine_tags,
        "dietary_tags": dietary_tags,
        "allergens": allergens
    }

    # Dynamic tag updates
    if spice_counter > 0 and "spicy" not in tags:
        tags.append("spicy")
    if sweet_counter > 0 and "sweet" not in tags:
        tags.append("sweet")
    if gluten_free and "gluten-free" not in tags:
        tags.append("gluten-free")
    tags.extend(preparation_tags + dish_tags + cuisine_tags + dietary_tags)

    return features

# Feedback Tags
def determine_customer_feedback_tags(rating, count_of_rating):
    if rating is None or count_of_rating is None:
        return []
    tags = []
    if rating > 4.5 and count_of_rating > 50:
        tags.append("highly rated")
    if rating > 4.0 and count_of_rating > 20:
        tags.append("popular")
    if rating < 3.5 and count_of_rating > 50:
        tags.append("value for money")
    return tags

# Affordability
def determine_affordability_tag(price):
    if price is None:
        return None
    if price < 100:
        return "budget"
    if 100 <= price < 300:
        return "mid-range"
    if 300 <= price < 500:
        return "expensive"
    if price >= 500:
        return "luxury"
    return None

# Drop nulls
def drop_null_columns(data):
    return {key: value for key, value in data.items() if value is not None}

# Restaurant type
def determine_restaurant_type(menu):
    has_veg = any(item["type"] == "veg" for item in menu)
    has_non_veg = any(item["type"] == "non-veg" for item in menu)
    if has_veg and has_non_veg:
        return "veg and non-veg"
    if has_veg:
        return "pure veg"
    if has_non_veg:
        return "non-veg"
    return "unknown"

# Restaurant features
def determine_restaurant_features(menu):
    features = []
    has_veg = any(item["type"] == "veg" for item in menu)
    has_non_veg = any(item["type"] == "non-veg" for item in menu)
    has_gluten_free = any(item.get("gluten_free", False) for item in menu)
    has_customizable = any(item.get("is_customizable", False) for item in menu)
    highly_rated_count = sum(1 for item in menu if "highly rated" in item.get("feedback_tags", []))

    if menu:
        features.append("delivery available")
    if has_veg:
        features.append("vegetarian options")
    if not has_non_veg and has_veg:
        features.append("100% vegetarian")
    if has_gluten_free:
        features.append("gluten-free options")
    if has_customizable:
        features.append("customizable dishes")
    if highly_rated_count > 5:
        features.append("highly rated")
    return features

# Main Preprocessing
def preprocess_data(raw_data_path, output_path):
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    knowledge_base = []
    for restaurant in raw_data:
        structured_restaurant = {
            "restaurant_name": restaurant.get('restaurant_name', 'Unknown'),
            "location": restaurant.get('location', 'Unknown'),
            "available_time": restaurant.get('available_time', 'Unknown'),
            "contact": restaurant.get('contact', 'Unknown'),
            "menu": []
        }
        for item in restaurant.get('menu_items', []):
            description = (item.get('small_description') or '') + " " + (item.get('big_description') or '')
            tags = item.get('tags', [])

            features = extract_features(description, tags)

            feedback_tags = determine_customer_feedback_tags(item.get("rating"), item.get("count_of_rating"))
            affordability_tag = determine_affordability_tag(item.get("price"))
            if affordability_tag and affordability_tag not in tags:
                tags.append(affordability_tag)

            is_veg = item.get('is_veg', None)
            dish_type = "veg" if is_veg == 1 else "non-veg"

            if dish_type == "veg" and "veg" not in tags:
                tags.append("veg")
            if dish_type == "non-veg" and "non-veg" not in tags:
                tags.append("non-veg")

            popularity_score = None
            if item.get("rating") and item.get("count_of_rating"):
                popularity_score = round(item["rating"] * item["count_of_rating"], 2)

            structured_item = {
                "item_name": item.get('product_name', 'Unknown'),
                "price": item.get('price', 'Unknown'),
                "tags": tags,
                "spice_level": item.get('spice_level', 'Unknown'),
                "spice_counter": features["spice_counter"],
                "sweet_counter": features["sweet_counter"],
                "gluten_free": features["gluten_free"],
                "type": dish_type,
                "short_description": item.get('small_description', 'Unknown'),
                "long_description": item.get('big_description', 'Unknown'),
                "preparation_tags": features["preparation_tags"],
                "dish_tags": features["dish_tags"],
                "cuisine_tags": features["cuisine_tags"],
                "dietary_tags": features["dietary_tags"],
                "allergens": features["allergens"],
                "feedback_tags": feedback_tags,
                "affordability_tag": affordability_tag,
                "is_customizable": item.get("is_customizable", False),
                "popularity_score": popularity_score
            }

            structured_item = drop_null_columns(structured_item)
            structured_restaurant["menu"].append(structured_item)

        if not structured_restaurant["menu"]:
            continue

        structured_restaurant["type"] = determine_restaurant_type(structured_restaurant["menu"])
        structured_restaurant["features"] = determine_restaurant_features(structured_restaurant["menu"])
        structured_restaurant = drop_null_columns(structured_restaurant)
        knowledge_base.append(structured_restaurant)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=4, ensure_ascii=False)
    print(f" Knowledge base saved to {output_path}")

def preprocess_and_index(knowledge_base_path, index_path, metadata_path):
    """Preprocess the knowledge base and create a FAISS index."""
    # Load the knowledge base
    with open(knowledge_base_path, 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)

    # Load SentenceTransformer model
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Flatten important fields
    documents = []
    metadata = []

    for restaurant in knowledge_base:
        for item in restaurant["menu"]:
            text = f"{restaurant['restaurant_name']} | {restaurant['location']} | {item['item_name']} | {item.get('short_description', '')} | {item.get('long_description', '')} | Tags: {', '.join(item.get('tags', []))}"
            documents.append(text)
            metadata.append({
                "restaurant_name": restaurant['restaurant_name'],
                "item_name": item['item_name'],
                "location": restaurant['location']
            })
            

    # Embed all documents
    print(documents)
    embeddings = embedder.encode(documents, convert_to_numpy=True, normalize_embeddings=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    # Save FAISS index and metadata
    faiss.write_index(index, index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)

    print("✅ FAISS index and metadata saved!")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(current_dir, 'output', 'raw_extracted_data.json')
    knowledge_base_path = os.path.join(current_dir, 'output', 'knowledge_base.json')
    index_path = os.path.join(current_dir, 'output', 'faiss_index.bin')
    metadata_path = os.path.join(current_dir, 'output', 'metadata.pkl')

    # preprocess_data(raw_data_path, knowledge_base_path)
    preprocess_and_index(knowledge_base_path, index_path, metadata_path)