# Script: `.\launcher.py`

print("Starting `launcher` Imports.")
from pathlib import Path
import os
from scripts import temporary
from scripts.utility import load_config, save_config
from scripts.interface import launch_interface
print("`launcher` Imports Complete.")

def main():
    try:
        print("Starting `launcher.main`.")
        # Get the absolute path of the directory containing launcher.py
        script_dir = Path(__file__).parent.resolve()
        # Set project_root to the script's directory (Text-Gradio-Gguf-main)
        project_root = script_dir
        os.chdir(project_root)
        print(f"Working directory set to: {project_root}")
        
        # Set the data directory as an absolute path
        temporary.DATA_DIR = str(project_root / "data")
        print(f"Data directory set to: {temporary.DATA_DIR}")
        
        # Ensure the data directory exists
        Path(temporary.DATA_DIR).mkdir(parents=True, exist_ok=True)
        
        print("Loading configuration from persistent.json...")
        load_config()  # Uses temporary.DATA_DIR implicitly via working directory
        print("Saving configuration to persistent.json...")
        save_config()  # Uses temporary.DATA_DIR implicitly via working directory
        print("Launching Gradio Interface.")
        launch_interface()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()