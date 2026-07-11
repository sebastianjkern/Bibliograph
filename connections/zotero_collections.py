import os

import pandas as pd
from dotenv import load_dotenv
from pyzotero import Zotero


def fetch_zotero_collections(output_file: str = "test.csv"):
    """
    Fetches collections from a Zotero library and saves them to a CSV file.
    Parameters:
        library_id (str): The ID of the Zotero library.
        library_type (str): The type of the Zotero library (e.g., 'public', 'private').
        api_key (str): Your Zotero API key.
        output_file (str): The path to save the CSV file (default: "test.csv").
    Returns:
        pd.DataFrame: DataFrame containing the collections data.
    """
    # Load environment variables from .env file
    load_dotenv()  # Automatically loads variables from .env into os.environ

    # Retrieve configuration values
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE")
    api_key = os.getenv("ZOTERO_API_KEY")

    # Initialize Zotero client with the retrieved credentials
    zotero = Zotero(library_id, library_type=library_type, api_key=api_key)

    # Fetch collections and store them in a pandas DataFrame for database storage
    collections = zotero.collections()

    collections_data = [
    c.get("data")
    for c in collections
    ]

    df = pd.DataFrame(collections_data)
    df.to_csv(output_file, index=False)

    print(df)
    return df

# Entry point for standalone execution.
# Example: set environment variables in .env file first, then run the script directly.
try:
    df = fetch_zotero_collections()
except Exception as e:
    print(f"An error occurred: {e}")