import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_openai_client(base_url: str = "http://localhost:1234/v1") -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set in environment.")
    return OpenAI(base_url=base_url, api_key=api_key)


def create_embeddings(
    texts,
    model: str = "text-embedding-nomic-embed-text-v1.5",
    base_url: str = "http://localhost:1234/v1"
):
    """
    Generate embeddings for a list of texts.

    Parameters
    ----------
    texts : list[str]
        Input texts to embed.
    model : str
        Embedding model name.
    base_url : str
        API base URL.

    Returns
    -------
    list[list[float]]
        List of embedding vectors.
    """
    client = get_openai_client(base_url)
    response = client.embeddings.create(
        model=model,
        input=texts
    )
    return [item.embedding for item in response.data]


# Example usage
if __name__ == "__main__":
    texts = ["This is a test sentence"] * 50
    embeddings = create_embeddings(texts)
    print(embeddings[0])