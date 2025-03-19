# Script: `.\launcher.py`

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
        script_dir = Path(__file__).parent.resolve()
        os.chdir(script_dir)
        print(f"Working directory: {script_dir}")
        temporary.DATA_DIR = str(script_dir / "data")
        print(f"Data directory: {temporary.DATA_DIR}")
        Path(temporary.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(temporary.HISTORY_DIR).mkdir(parents=True, exist_ok=True)  # Added
        Path(temporary.VECTORSTORE_DIR).mkdir(parents=True, exist_ok=True)  # Added
        
        print("Loading persistent config...")
        load_config()
        print("Launching Gradio Interface...")
        try:
            launch_interface()
        except Exception as e:
            print(f"Error launching interface: {str(e)}")
            raise
    except Exception as e:
        print(f"Error in launcher: {str(e)}")
        raise

if __name__ == "__main__":
    main()