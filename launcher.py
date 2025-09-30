# launcher.py

# Disable Logging
import os, logging, logging.config
# 1.  replace dictConfig by a no-op
logging.config.dictConfig = lambda *_, **__: None
# 2.  replace fileConfig by a no-op
logging.config.fileConfig = lambda *_, **__: None
# 3.  make basicConfig a no-op as well
logging.basicConfig = lambda *_, **__: None

# Early Imports
import sys, argparse
from pathlib import Path
import os
from scripts.utility import short_path
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('platform', choices=['windows', 'linux'], help='Target platform')
    return parser.parse_args()
args = parse_args()
import scripts.temporary as temporary
temporary.PLATFORM = args.platform
temporary.BACKEND_TYPE = "Vulkan"
from scripts.settings import load_config
from scripts.interface import launch_interface
from scripts.utility import detect_cpu_config



# Functions
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
    
    print(f"Script mode `{temporary.PLATFORM}` with backend `{temporary.BACKEND_TYPE}`")

def shutdown_program(llm_state, models_loaded_state, session_log, attached_files):
    """Gracefully shutdown the program, saving current session if active."""
    from scripts.utility import save_session_history
    from scripts.models import unload_models
    import time
    
    from scripts.temporary import set_status
    set_status("Shutting down...")
    
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
    for i in range(3, -1, -1):        # include 0
        print(f"Closing in...{i}", end="\r")
        time.sleep(1)
    print()                           # newline after countdown

    print("Shutdown complete. Goodbye!")
    shutdown_platform()
    if temporary.demo is not None:
        temporary.demo.close()        # stops the Gradio server
    os._exit(0)  

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
            from scripts import utility
            if hasattr(utility, 'tts_engine'):
                utility.tts_engine.stop()
                del utility.tts_engine
        except:
            pass
    print(f"Cleaned up {temporary.PLATFORM} resources")

# Main Function
def main():
    """Main entry point for the application."""
    try:
        print("`main` Function Started.")

        # Parse command-line arguments
        args = parse_args()
        
        # Initialize platform
        temporary.PLATFORM = args.platform
        initialize_platform_settings()
        
        # Set up directories and paths
        script_dir = Path(__file__).parent.resolve()
        os.chdir(script_dir)
        print(f"Working directory: {short_path(script_dir)}")
        
        temporary.DATA_DIR = str(script_dir / "data")
        temporary.HISTORY_DIR = str(script_dir / "data/history")
        temporary.TEMP_DIR = str(script_dir / "data/temp")
        
        # Create required directories
        Path(temporary.DATA_DIR).mkdir(parents=True, exist_ok=True)
        Path(temporary.HISTORY_DIR).mkdir(parents=True, exist_ok=True)
        Path(temporary.TEMP_DIR).mkdir(parents=True, exist_ok=True)
        
        print(f"Data Directory: {short_path(temporary.DATA_DIR)}")
        print(f"Session History: {short_path(temporary.HISTORY_DIR)}")
        print(f"Temp Directory: {short_path(temporary.TEMP_DIR)}")
        
        # Initialize CPU configuration
        from scripts.utility import detect_cpu_config
        detect_cpu_config()
        print(f"CPU Configuration: {temporary.CPU_PHYSICAL_CORES} physical cores, "
              f"{temporary.CPU_LOGICAL_CORES} logical cores")
        
        # Set platform-specific defaults
        temporary.BACKEND_TYPE = "Vulkan"  # Default for both platforms
        
        # Print final configuration
        from scripts.temporary import set_status
        set_status("Config loaded")
        print("\nConfiguration:")
        print(f"  Backend: {temporary.BACKEND_TYPE}")
        print(f"  Model: {temporary.MODEL_NAME or 'None'}")
        print(f"  Context Size: {temporary.CONTEXT_SIZE}")
        print(f"  VRAM Allocation: {temporary.VRAM_SIZE} MB")
        print(f"  CPU Threads: {temporary.CPU_THREADS}")
        print(f"  GPU Layers: {temporary.GPU_LAYERS if hasattr(temporary, 'GPU_LAYERS') else 'Auto'}")
        
        # Launch interface
        print("\nLaunching Gradio Interface...")
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