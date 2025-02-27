# Script: `.\scripts\launcher.py`

# Imports...
print("Starting `launcher` Imports.")
from pathlib import Path
from scripts import temporary  # Assuming ALLOWED_EXTENSIONS is defined here
from scripts.utility import load_config, load_and_chunk_documents, create_vectorstore
from scripts.models import initialize_model
from scripts.interface import launch_interface
print("`launcher` Imports Complete.")

# Functions...
def main():
    try:
        print("Starting `launcher.main`.")
        load_config()
        
        # Load documents from the "files" directory
        files_dir = Path("files")
        if files_dir.exists() and files_dir.is_dir():
            file_paths = [
                str(file) for file in files_dir.iterdir()
                if file.is_file() and file.suffix[1:].lower() in temporary.ALLOWED_EXTENSIONS
            ]
            if file_paths:
                docs = load_and_chunk_documents(file_paths)
                if docs:
                    print("Building knowledge base...")
                    create_vectorstore(docs)
            else:
                print("No files in `.files`.")
        else:
            print("'files' directory does not exist.")
        
        # Proceed with model initialization and interface launch
        initialize_model(None)
        launch_interface()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()