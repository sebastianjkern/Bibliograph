import chromadb
import fitz  # PyMuPDF
import nltk
from sentence_transformers import SentenceTransformer

nltk.download('punkt')

def extract_sentences(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page in doc:
        text += page.get_text()

    return nltk.tokenize.sent_tokenize(text)


def setup_chroma():
    client = chromadb.Client(
        settings=chromadb.config.Settings(persist_directory="./chroma_db")
    )
    
    collection = client.create_collection("pdf_sentences")
    return collection


def store_sentences(collection, sentences, model):
    embeddings = model.encode(sentences).tolist()

    collection.add(
        documents=sentences,
        embeddings=embeddings,
        ids=[str(i) for i in range(len(sentences))]
    )


def search(collection, model, query, k=5):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    return results["documents"][0]


if __name__ == "__main__":
    pdf_file = "test.pdf"

    # Load model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Extract
    sentences = extract_sentences(pdf_file)

    # Init DB
    collection = setup_chroma()

    # Store
    store_sentences(collection, sentences, model)

    # Query loop
    while True:
        query = input("\nSearch query (or 'exit'): ")
        if query.lower() == "exit":
            break

        results = search(collection, model, query)

        print("\nTop results:")
        for r in results:
            print("-", r)