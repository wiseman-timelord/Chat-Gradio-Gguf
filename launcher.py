# Script: `.\scripts\launcher.py`

from pathlib import Path
from scripts import temporary

def main():
    try:
        utility.load_config()  # Updated import
        from scripts.utility import load_and_chunk_documents, create_vectorstore
        docs = load_and_chunk_documents(Path("files"))
        if docs:
            create_vectorstore(docs)
        from scripts.models import initialize_model
        initialize_model(None)
        from scripts.interface import launch_interface
        launch_interface()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()