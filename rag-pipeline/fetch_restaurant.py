import json
import os

def load_restaurants(json_path=r'D:\rag-restaurant-assistant\data\processed\knowledgebase.json'):
    '''
    Load the list of restaurants from a JSON file.
    If json_path is None, defaults to data/restaurants.json.
    '''
    
    if json_path is None:
        base = os.path.dirname(__file__)
        json_path = os.path.join(base, '..', 'data', 'restaurants.json')
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
if __name__ == '__main__':
    print(load_restaurants())
