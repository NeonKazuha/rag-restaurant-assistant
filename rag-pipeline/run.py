import os
from fetch_restaurant import load_restaurants
from chunking import build_dish_chunks
from index_faiss import create_faiss_index
from chatbot import answer

def main():
    #Load from JSON
    base = os.path.dirname(__file__)
    restaurants = load_restaurants()

    #Build chunks & index
    chunks = build_dish_chunks(restaurants)
    embed_model, index = create_faiss_index(chunks)

    #Chat loop
    print("Zomato RAG Chatbot ready! Type 'exit' to quit.")
    while True:
        q = input("\nYou: ")
        if q.lower() in ('exit', 'quit'):
            break
        resp = answer(q, embed_model, index, chunks)
        print("Bot:", resp)

if __name__ == "__main__":
    main()
