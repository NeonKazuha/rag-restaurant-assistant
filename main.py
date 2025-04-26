from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
import json
from pathlib import Path
import re

# 1. Utility: normalize text
def normalize(text: str) -> str:
    t = text.lower()
    t = re.sub(r'[^a-z0-9]+', ' ', t)
    return t.strip()

# 2. Load JSON data into Document list per menu item
def load_json_docs(json_path: Path):
    data = json.loads(json_path.read_text())
    docs = []
    for restaurant in data["restaurants"]:
        rest_name = restaurant["name"]
        for item in restaurant.get("menu", []):
            content = (
                f"Restaurant: {rest_name}\n"
                f"Item: {item['item']}\n"
                f"Price: ${item['price']}\n"
                f"Dietary: {', '.join(item['dietary_restrictions'])}\n"
            )
            metadata = {
                "restaurant": rest_name,
                "item": item['item'],
                "price": item['price'],
                "dietary_restrictions": item['dietary_restrictions'],
            }
            docs.append(Document(page_content=content, metadata=metadata))
    return docs

# 3. Build vectorstore
def build_vectorstore(docs, persist_path="./vectordb"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embeddings)
    vectordb.save_local(persist_path)
    return vectordb

# 4a. Search items by exact dietary restriction
def search_menu_by_dietary_restriction(docs, restriction):
    norm_req = normalize(restriction)
    results = []
    for doc in docs:
        for lab in doc.metadata.get("dietary_restrictions", []):
            if normalize(lab) == norm_req:
                results.append(doc)
                break
    return results

# 4b. Search items containing a keyword in dietary labels
def search_menu_by_keyword(docs, keyword):
    norm_key = normalize(keyword)
    results = []
    for doc in docs:
        for lab in doc.metadata.get("dietary_restrictions", []):
            if norm_key in normalize(lab):
                results.append(doc)
                break
    return results

# 5. Main
if _name_ == "_main_":
    BASE = Path(_file_).parent.parent
    JSON_PATH = BASE / "data" / "restaurants.json"

    # Load and index docs
    docs = load_json_docs(JSON_PATH)
    vectordb = build_vectorstore(docs)
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})

    # Local LLM pipeline
    local_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_length=150
    )
    llm = HuggingFacePipeline(pipeline=local_pipeline)

    # Prompt for RAG
    prompt_template = '''
Use the following items to answer the question succinctly.
If you don't know the answer, say "I am not sure." 

Items:
{context}

Question: {question}
Answer:'''  # noqa: E501
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )

    # Chat loop
    print("RAG Chatbot (open-source) ready. Type 'exit' to quit.")
    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        qnorm = normalize(query)
        # Structured dietary filters
        if any(x in qnorm for x in ["gluten free", "vegan", "vegetarian"]):
            if "gluten free" in qnorm:
                key = "gluten free"
            elif "vegan" in qnorm:
                key = "vegan"
            else:
                key = "vegetarian"
            items = search_menu_by_dietary_restriction(docs, key)
            if items:
                print(f"Bot: Here are the {key} options:")
                for doc in items:
                    print(f"- {doc.metadata['item']} (${doc.metadata['price']}) at {doc.metadata['restaurant']}")
            else:
                print(f"Bot: No {key} options found.")

        # Generic contains keyword extraction for any label
        elif m := re.search(r"contain(?:s)? ([a-z ]+)", qnorm):
            key = m.group(1)
            items = search_menu_by_keyword(docs, key)
            if items:
                print(f"Bot: Here are items that contain {key}:")
                for doc in items:
                    print(f"- {doc.metadata['item']} (${doc.metadata['price']}) at {doc.metadata['restaurant']}")
            else:
                print(f"Bot: No items containing {key} found.")

        # Non-veg detection maps to 'seafood' label
        elif "non veg" in qnorm or "non-veg" in qnorm:
            items = search_menu_by_keyword(docs, "seafood")
            if items:
                print("Bot: Here are non-veg options (seafood):")
                for doc in items:
                    print(f"- {doc.metadata['item']} (${doc.metadata['price']}) at {doc.metadata['restaurant']}")
            else:
                print("Bot: No non-veg options found.")

        else:
            # fallback to semantic RAG
            resp = chain.invoke({"query": query})
            print(f"Bot: {resp['result']}")x