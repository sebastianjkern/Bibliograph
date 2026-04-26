from pyzotero import Zotero
import os
from dotenv import load_dotenv
load_dotenv()  # Automatically loads variables from .env into os.environ

import pandas as pd

def fetch_zotero_items() -> pd.DataFrame:
    """
    Fetches items from Zotero and returns them as a pandas DataFrame.

    Args:
        limit (int): Number of top items to fetch. Defaults to 5.
        output_df (bool): If True, prints the DataFrame. Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing Zotero item details.
    """
    # Retrieve configuration values
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE")
    api_key = os.getenv("ZOTERO_API_KEY")

    # Initialize Zotero client with the retrieved credentials
    zotero = Zotero(library_id, library_type=library_type, api_key=api_key)

    # Retrieve items sorted by relevance
    # items = zotero.items()
    items = zotero.everything(zotero.items())

    # Collect detailed information for each item
    item_list = []
    for item in items:
        item_data = item['data']
        record = {
            'key': item_data.get('key', 'N/A'),
            'title': item_data.get('title', 'N/A'),
            'type': item_data.get('itemType', 'Unknown'),
            # 'date_added': item_data.get('dateAdded', 'N/A'),
            # 'creators': ', '.join([f'{creator["firstName"]} {creator["lastName"]}' for creator in item_data.get('creators', [])]),
            'abstract': item_data.get('abstractNote', 'No abstract available'),
            'tags': ', '.join([tag['tag'] for tag in item_data.get('tags', [])]),
            # 'repository': item_data.get('repository', 'N/A'),
            # 'series': item_data.get('series', 'N/A'),
            'DOI': item_data.get('DOI', 'N/A')
        }
        item_list.append(record)

    # Convert list of records into a pandas DataFrame
    df = pd.DataFrame(item_list)

    df.to_csv("test.csv")

    print(df)
    return df

try:
    df = fetch_zotero_items()
except Exception as e:
    print(f"An error occurred: {e}")