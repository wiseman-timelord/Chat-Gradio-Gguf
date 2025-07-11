# launcher.py

# Imports
print("Starting `launcher` Imports.")
import sys, argparse
from pathlib import Path
import os
from scripts import temporary
from scripts.settings import load_config
from scripts.interface import launch_interface
print("`launcher` Imports Complete.")

# Functions
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('platform', choices=['windows', 'linux'], help='Target platform')
    return parser.parse_args()

def initialize_platform_settings():
    """Set platform-specific paths and configurations"""
    from scripts.settings import load_config
    load_config()  # Ensure config is loaded
    
    if temporary.PLATFORM == "windows":
        if "vulkan" in temporary.BACKEND_TYPE.lower():
            temporary.LLAMA_CLI_PATH = "data/llama-vulkan-bin/llama-cli.exe"
        elif "cuda" in temporary.BACKEND_TYPE.lower():
            temporary.LLAMA_CLI_PATH = "data/llama-cuda-bin/llama-cli.exe"
        elif "hip" in temporary.BACKEND_TYPE.lower():
            temporary.LLAMA_CLI_PATH = "data/llama-hip-radeon-bin/llama-cli.exe"
    elif temporary.PLATFORM == "linux":
        if "vulkan" in temporary.BACKEND_TYPE.lower():
            temporary.LLAMA_CLI_PATH = "data/llama-vulkan-bin/llama-cli"
        elif "cuda" in temporary.BACKEND_TYPE.lower():
            temporary.LLAMA_CLI_PATH = "data/llama-cuda-bin/llama-cli"
    else:
        raise ValueError(f"Unsupported platform: {temporary.PLATFORM}")
    
    print(f"Initialized {temporary.PLATFORM} with backend: {temporary.BACKEND_TYPE}")

def shutdown_program(llm_state, models_loaded_state, session_log, attached_files):
    """Gracefully shutdown the program, saving current session if active."""
    from scripts.utility import save_session_history
    from scripts.models import unload_models
    import time
    
    print("\nInitiating shutdown sequence...")
    
    # Save current session if active and has content
    if temporary.SESSION_ACTIVE and session_log:
        print("Saving current session before exit...")
        try:
            save_session_history(session_log, attached_files)
            print(f"Session saved to history: {temporary.session_label}")
        except Exception as e:
            print(f"Error saving session: {str(e)}")
    
    # Unload models if loaded
    if models_loaded_state and llm_state:
        print("Unloading model...")
        try:
            status, _, _ = unload_models(llm_state, models_loaded_state)
            print(status)
        except Exception as e:
            print(f"Error unloading model: {str(e)}")
    
    # Graceful shutdown sequence
    for i in range(3, 0, -1):
        print(f"Closing in {i}...")
        time.sleep(1)
    
    print("Shutdown complete. Goodbye!")
    shutdown_platform()  # Call platform-specific cleanup
    sys.exit(0)

def shutdown_platform():
    """Platform-specific shutdown procedures"""
    if temporary.PLATFORM == "windows":
        # Windows-specific cleanup
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except:
            pass
    elif temporary.PLATFORM == "linux":
        # Linux-specific cleanup
        try:
            if hasattr(temporary, 'tts_engine'):
                temporary.tts_engine.stop()
                del temporary.tts_engine
        except:
            pass
    print(f"Cleaned up {temporary.PLATFORM} resources")

# Main Function
def main():
    """Main entry point for the application."""
    try:
        # Parse command-line arguments
        args = parse_args()  # Add this line
        
        # Initialize platform
        temporary.PLATFORM = args.platform
        initialize_platform(temporary.PLATFORM)
        print(f"Starting Chat-Gradio-Gguf for {temporary.PLATFORM} platform")
        
        # Set up directories and paths
        script_dir = Path(__file__).parent.resolve()
        os.chdir(script_dir)
        print(f"Working directory: {script_dir}")
        
        temporary.DATA_DIR = str(script_dir / "data")
        temporary.HISTORY_DIR = str(script_dir / "data/history")
        temporary.TEMP_DIR = str(script_dir / "data/temp")
        
        # Create required directories
        Path(temporary.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(temporary.HISTORY_DIR).mkdir(parents=True, exist_ok=True)
        Path(temporary.TEMP_DIR).mkdir(parents=True, exist_ok=True)
        
        print(f"Data directory: {temporary.DATA_DIR}")
        print(f"History directory: {temporary.HISTORY_DIR}")
        print(f"Temp directory: {temporary.TEMP_DIR}")
        
        # Set platform-specific defaults
        temporary.BACKEND_TYPE = "Vulkan"  # Default for both platforms
        
        # Load configuration
        print("Loading persistent config...")
        load_config()
        
        # Launch interface
        print("Launching Gradio Interface...")
        try:
            launch_interface()
        except Exception as e:
            print(f"Error launching interface: {str(e)}")
            raise
        
    except Exception as e:
        print(f"Fatal error in launcher: {str(e)}")
        shutdown_platform()
        sys.exit(1)

if __name__ == "__main__":
    main()