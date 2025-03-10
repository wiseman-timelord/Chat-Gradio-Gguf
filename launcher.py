# Script: `.\scripts\launcher.py`

print("Starting `launcher` Imports.")
from pathlib import Path
import os
from scripts import temporary
from scripts.utility import load_config
from scripts.interface import launch_interface
print("`launcher` Imports Complete.")

def main():
    try:
        print("Starting `launcher.main`.")
        project_root = Path(__file__).parent.parent
        os.chdir(project_root)
        print(f"Set working directory to: {project_root}")
        load_config()  # Updates temporary.MODEL_FOLDER
        launch_interface()  # Calls get_available_models()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()