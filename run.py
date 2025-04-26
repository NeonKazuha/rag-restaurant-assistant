import os
from rag_pipeline.load_restaurants import load_restaurants
from rag_pipeline.build_chunks import build_dish_chunks
from rag_pipeline.index_faiss import create_faiss_index
from rag_pipeline.rag_chatbot import answer_dynamic

def main():
    # 1) Load from JSON
    base = os.path.dirname(__file__)
    data_path = os.path.join(base, '..', 'data', 'restaurants.json')
    restaurants = load_restaurants(data_path)

    # 2) Build chunks & index
    chunks = build_dish_chunks(restaurants)
    embed_model, index = create_faiss_index(chunks)

    # 3) Chat loop
    print("Zomato RAG Chatbot ready! Type 'exit' to quit.")
    while True:
        q = input("\nYou: ")
        if q.lower() in ('exit', 'quit'):
            break
        resp = answer_dynamic(q, embed_model, index, chunks)
        print("Bot:", resp)

if __name__ == "__main__":
    main()
