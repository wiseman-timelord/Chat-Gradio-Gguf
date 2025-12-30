# launcher.py

# Set FastEmbed env vars FIRST before ANY imports
import os
from pathlib import Path
cache_dir = Path(__file__).parent / "data" / "fastembed_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir.absolute())
os.environ["FASTEMBED_OFFLINE"] = "1"

# Disable Logging
import logging, logging.config
logging.config.dictConfig = lambda *_, **__: None
logging.config.fileConfig = lambda *_, **__: None
logging.basicConfig = lambda *_, **__: None

# Imports
import sys, argparse, time
from pathlib import Path
from scripts.utility import short_path
import scripts.temporary as temporary
from scripts.settings import load_config
from scripts.interface import launch_interface
from scripts.utility import detect_cpu_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('platform', choices=['windows', 'linux'], help='Target platform')
    return parser.parse_args()

def initialize_platform_settings():
    """Initialize platform-specific settings with validation."""
    valid_backends = ["CPU_CPU", "VULKAN_CPU", "VULKAN_VULKAN"]
    if temporary.BACKEND_TYPE not in valid_backends:
        print(f"Warning: Invalid backend_type '{temporary.BACKEND_TYPE}', defaulting to CPU_CPU")
        temporary.BACKEND_TYPE = "CPU_CPU"
        temporary.VULKAN_AVAILABLE = False

    # REMOVED: Don't force SRAM_ONLY here - let config.json control it
    # The layer allocation mode should be set by load_config() from persistent.json

    # ------------------------------------------------------------------
    # Vulkan VRAM optimisation – Polaris / AMD+NVIDIA universal
    # ------------------------------------------------------------------
    if temporary.BACKEND_TYPE in ("VULKAN_CPU", "VULKAN_VULKAN"):
        if temporary.PLATFORM == "windows":
            # Windows: persistent for the whole console session
            os.environ["GGML_CUDA_NO_PINNED"]   = "1"
            os.environ["GGML_VK_NO_PIPELINE_CACHE"] = "0"
            print("[Vulkan] GGML_CUDA_NO_PINNED=1   (frees ~300 MB VRAM)")
            print("[Vulkan] GGML_VK_NO_PIPELINE_CACHE=0  (cached SPIR-V pipelines)")
        else:
            # Linux: export for child processes
            os.environ["GGML_CUDA_NO_PINNED"]   = "1"
            os.environ["GGML_VK_NO_PIPELINE_CACHE"] = "0"
            print("[Vulkan] Exported GGML_CUDA_NO_PINNED=1")
            print("[Vulkan] Exported GGML_VK_NO_PIPELINE_CACHE=0")

    # Set platform-specific paths
    if temporary.PLATFORM == "windows":
        if "VULKAN" in temporary.BACKEND_TYPE:
            temporary.LLAMA_CLI_PATH = "data/llama-vulkan-bin/llama-cli.exe"
    elif temporary.PLATFORM == "linux":
        if "VULKAN" in temporary.BACKEND_TYPE:
            temporary.LLAMA_CLI_PATH = "data/llama-vulkan-bin/llama-cli"
    else:
        raise ValueError(f"Unsupported platform: {temporary.PLATFORM}")

    print(f"Script mode `{temporary.PLATFORM}` with backend `{temporary.BACKEND_TYPE}`")

def shutdown_program(llm_state, models_loaded_state, session_log, attached_files):
    """Gracefully shutdown the program, saving current session if active."""
    from scripts.utility import save_session_history
    from scripts.models import unload_models
    
    temporary.set_status("Shutting down...")
    
    # Save current session if active and has content
    if temporary.SESSION_ACTIVE and session_log:
        try:
            save_session_history(session_log, attached_files)
            print(f"Session saved to history: {temporary.session_label}")
        except Exception as e:
            print(f"Error saving session: {str(e)}")
    
    # Unload models if loaded
    if models_loaded_state and llm_state:
        try:
            status, _, _ = unload_models(llm_state, models_loaded_state)
            print(status)
        except Exception as e:
            print(f"Error unloading model: {str(e)}")
    
    # Graceful shutdown sequence
    for i in range(3, -1, -1):
        print(f"Closing in...{i}", end="\r")
        time.sleep(1)
    print()
    
    print("Shutdown complete. Goodbye!")
    shutdown_platform()
    if temporary.demo is not None:
        temporary.demo.close()
    os._exit(0)

def shutdown_platform():
    """Platform-specific cleanup procedures."""
    if temporary.PLATFORM == "windows":
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except:
            pass
    elif temporary.PLATFORM == "linux":
        try:
            from scripts import utility
            if hasattr(utility, 'tts_engine'):
                utility.tts_engine.stop()
                del utility.tts_engine
        except:
            pass
    print(f"Cleaned up {temporary.PLATFORM} resources")

def setup_directories():
    """Setup and create required directories."""
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    temporary.DATA_DIR = str(script_dir / "data")
    temporary.HISTORY_DIR = str(script_dir / "data/history")
    temporary.TEMP_DIR = str(script_dir / "data/temp")
    
    # Create required directories
    for dir_path in [temporary.DATA_DIR, temporary.HISTORY_DIR, temporary.TEMP_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return script_dir

def print_configuration():
    """Print current configuration settings."""
    print("\nConfiguration:")
    print(f"  Backend: {temporary.BACKEND_TYPE}")
    print(f"  Model: {temporary.MODEL_NAME or 'None'}")
    print(f"  Context Size: {temporary.CONTEXT_SIZE}")
    print(f"  VRAM Allocation: {temporary.VRAM_SIZE} MB")
    print(f"  CPU Threads: {temporary.CPU_THREADS}")
    print(f"  GPU Layers: {getattr(temporary, 'GPU_LAYERS', 'Auto')}")

def preload_auxiliary_models():
    """Pre-load spaCy and FastEmbed before main model to avoid memory conflicts."""
    
    # 1. Pre-load spaCy (pip package, no special path needed)
    try:
        from scripts.utility import get_nlp_model
        nlp = get_nlp_model()
        if nlp:
            print("[INIT] ✓ spaCy model pre-loaded")
        else:
            print("[INIT] ⚠ spaCy model not available (will use fallback)")
    except Exception as e:
        print(f"[INIT] ⚠ spaCy pre-load failed: {e}")
    
    # 2. Pre-load FastEmbed with correct cache path
    try:
        import os
        from pathlib import Path
        
        # Set cache path to match installer location BEFORE importing
        cache_dir = Path(__file__).parent / "data" / "fastembed_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir.absolute())
        os.environ["FASTEMBED_OFFLINE"] = "1"  # Prefer local cache
        
        # Now trigger the lazy load
        temporary.context_injector._ensure_embedding_model()
        
        if temporary.context_injector.embedding:
            print("[INIT] ✓ FastEmbed model pre-loaded from cache")
        else:
            print("[INIT] ⚠ FastEmbed model not available (RAG disabled)")
    except Exception as e:
        print(f"[INIT] ⚠ FastEmbed pre-load failed: {e}")
        print("[INIT]   RAG features will be unavailable")

def main():
    """Main entry point for the application."""
    try:
        print("`main` Function Started.")
        
        # Parse command-line arguments and initialize platform
        args = parse_args()
        temporary.PLATFORM = args.platform
        
        # Load config FIRST to get all saved settings including layer allocation
        load_config()
        
        # Then initialize platform settings (paths, validation)
        initialize_platform_settings()
        
        # Setup directories and paths
        script_dir = setup_directories()
        print(f"Working directory: {short_path(script_dir)}")
        print(f"Data Directory: {short_path(temporary.DATA_DIR)}")
        print(f"Session History: {short_path(temporary.HISTORY_DIR)}")
        print(f"Temp Directory: {short_path(temporary.TEMP_DIR)}")
        
        # Initialize CPU configuration
        detect_cpu_config()
        print(f"CPU Configuration: {temporary.CPU_PHYSICAL_CORES} physical cores, "
              f"{temporary.CPU_LOGICAL_CORES} logical cores")
        
        # Print final configuration
        temporary.set_status("Config loaded")
        print_configuration()
        
        # NEW: Pre-load auxiliary models to avoid memory conflicts
        print("\n[INIT] Pre-loading auxiliary models...")
        preload_auxiliary_models()
        
        # Launch interface
        print("\nLaunching Gradio Interface...")
        launch_interface()
        
    except Exception as e:
        print(f"Fatal error in launcher: {str(e)}")
        shutdown_platform()
        sys.exit(1)

if __name__ == "__main__":
    main()