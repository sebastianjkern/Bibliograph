import os

import pandas as pd
from dotenv import load_dotenv
from pyzotero import Zotero

load_dotenv()

def download_pdfs_from_df(df: pd.DataFrame, output_dir: str = "pdfs"):
    # Config
    library_id = os.getenv("ZOTERO_LIBRARY_ID")
    library_type = os.getenv("ZOTERO_LIBRARY_TYPE")
    api_key = os.getenv("ZOTERO_API_KEY")

    zot = Zotero(library_id, library_type, api_key)

    os.makedirs(output_dir, exist_ok=True)

    # Nur Attachments berücksichtigen
    attachments = df[df["type"] == "attachment"]

    for _, row in attachments.iterrows():
        key = row["key"]

        try:
            # Metadaten holen, um MIME-Type zu prüfen
            item = zot.item(key)
            content_type = item["data"].get("contentType", "")

            if content_type != "application/pdf":
                continue  # Skip alles was kein PDF ist

            pdf_bytes = zot.file(key)

            # Sauberer Dateiname
            filename = f"{key}.pdf"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "wb") as f:
                f.write(pdf_bytes)

            print(f"Saved: {filename}")

        except Exception as e:
            print(f"Failed {key}: {e}")

if __name__ == "__main__":
    import pandas as pd

    try:
        # Option 1: CSV laden (empfohlen)
        # df = pd.read_csv("zotero_items.csv")

        # Option 2: Testdaten (dein Beispiel)
        data = [
            {"key": "7248WM5Y", "title": "PDF", "type": "attachment"},
            {"key": "2G4UI6UB", "title": "Snapshot 1", "type": "attachment"},
            {"key": "V56F23QX", "title": "WorldCover Viewer", "type": "webpage"},
            {"key": "J8ZKZGW9", "title": "World Countries", "type": "attachment"},
        ]
        df = pd.DataFrame(data)

        print(f"Loaded {len(df)} records")

        download_pdfs_from_df(df, output_dir="pdfs")

        print("Download finished.")

    except Exception as e:
        print(f"Error during execution: {e}")
