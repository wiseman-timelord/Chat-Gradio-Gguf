# Script: `.\scripts\launcher.py`

# Imports...
print("Starting `launcher` Imports.")
from pathlib import Path
from scripts import temporary
from scripts.utility import load_config
from scripts.interface import launch_interface
print("`launcher` Imports Complete.")

# Functions...
def main():
    try:
        print("Starting `launcher.main`.")
        load_config()
        
        # Launch the interface
        launch_interface()
    except Exception as e:
        print(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()