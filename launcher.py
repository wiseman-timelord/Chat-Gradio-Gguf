# launcher.py - The entry point of the scripts of the main program.
# Set embedding cache env vars FIRST before ANY imports
import os
from pathlib import Path
cache_dir = Path(__file__).parent / "data" / "embedding_cache"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir.absolute())
os.environ["HF_HOME"] = str(cache_dir.parent.absolute())
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir.absolute())
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode
# CRITICAL: Force fully offline mode - prevents hanging when no network
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Disable Logging
import logging, logging.config
logging.config.dictConfig = lambda *_, **__: None
logging.config.fileConfig = lambda *_, **__: None
logging.basicConfig = lambda *_, **__: None

# Imports
import sys, argparse, time
from pathlib import Path
from scripts.utility import short_path
import scripts.configuration as cfg
from scripts.configuration import load_config
from scripts.display import launch_display
from scripts.utility import detect_cpu_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('platform', choices=['windows', 'linux'], help='Target platform')
    return parser.parse_args()

def initialize_platform_settings():
    """Initialize platform-specific settings with validation."""
    valid_backends = ["CPU_CPU", "VULKAN_CPU", "VULKAN_VULKAN"]
    if cfg.BACKEND_TYPE not in valid_backends:
        print(f"Warning: Invalid backend_type '{cfg.BACKEND_TYPE}', defaulting to CPU_CPU")
        cfg.BACKEND_TYPE = "CPU_CPU"
        cfg.VULKAN_AVAILABLE = False

    # Vulkan VRAM optimisation
    if cfg.BACKEND_TYPE in ("VULKAN_CPU", "VULKAN_VULKAN"):
        if cfg.PLATFORM == "windows":
            os.environ["GGML_CUDA_NO_PINNED"] = "1"
            os.environ["GGML_VK_NO_PIPELINE_CACHE"] = "0"
            print("[Vulkan] GGML_CUDA_NO_PINNED=1   (frees ~300 MB VRAM)")
            print("[Vulkan] GGML_VK_NO_PIPELINE_CACHE=0  (cached SPIR-V pipelines)")
        else:
            os.environ["GGML_CUDA_NO_PINNED"] = "1"
            os.environ["GGML_VK_NO_PIPELINE_CACHE"] = "0"
            print("[Vulkan] Exported GGML_CUDA_NO_PINNED=1")
            print("[Vulkan] Exported GGML_VK_NO_PIPELINE_CACHE=0")

    # Set platform-specific paths
    if cfg.PLATFORM == "windows":
        if "VULKAN" in cfg.BACKEND_TYPE:
            cfg.LLAMA_CLI_PATH = "data/llama-vulkan-bin/llama-cli.exe"
    elif cfg.PLATFORM == "linux":
        if "VULKAN" in cfg.BACKEND_TYPE:
            cfg.LLAMA_CLI_PATH = "data/llama-vulkan-bin/llama-cli"
    else:
        raise ValueError(f"Unsupported platform: {cfg.PLATFORM}")

    print(f"Script mode `{cfg.PLATFORM}` with backend `{cfg.BACKEND_TYPE}`")

def shutdown_program(llm_state, models_loaded_state, session_log, attached_files):
    """Gracefully shutdown the program, saving current session if active."""
    from scripts.utility import save_session_history
    from scripts.inference import unload_models
    
    cfg.set_status("Shutting down...")
    
    # Save current session if active and has content
    if cfg.SESSION_ACTIVE and session_log:
        try:
            save_session_history(session_log, attached_files)
            print(f"Session saved to history: {cfg.session_label}")
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
    print(f"Closing Program...")
    shutdown_platform()
    
    # Force terminate in a separate thread to ensure it happens
    def force_exit():
        time.sleep(1)
        print("[SHUTDOWN] Force terminating...")
        os._exit(0)
    
    import threading
    exit_thread = threading.Thread(target=force_exit, daemon=True)
    exit_thread.start()
    
    # Also try to close browser (may or may not work)
    try:
        from scripts.browser import close_browser
        close_browser()
    except:
        pass
    
    # If we get here, force exit anyway
    time.sleep(1)
    os._exit(0)

def shutdown_platform():
    """Platform-specific cleanup procedures."""
    if cfg.PLATFORM == "windows":
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except:
            pass
    print(f"Cleaned up {cfg.PLATFORM} resources")

def setup_directories():
    """Setup and create required directories."""
    script_dir = Path(__file__).parent.resolve()
    os.chdir(script_dir)
    
    cfg.DATA_DIR = str(script_dir / "data")
    cfg.HISTORY_DIR = str(script_dir / "data/history")
    cfg.TEMP_DIR = str(script_dir / "data/temp")
    
    # Create required directories
    for dir_path in [cfg.DATA_DIR, cfg.HISTORY_DIR, cfg.TEMP_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    return script_dir

def print_configuration():
    """Print current configuration cfg."""
    print("\nConfiguration:")
    print(f"  Backend: {cfg.BACKEND_TYPE}")
    print(f"  Model: {cfg.MODEL_NAME or 'None'}")
    print(f"  Context Size: {cfg.CONTEXT_SIZE}")
    print(f"  VRAM Allocation: {cfg.VRAM_SIZE} MB")
    print(f"  CPU Threads: {cfg.CPU_THREADS}")
    print(f"  GPU Layers: {getattr(cfg, 'GPU_LAYERS', 'Auto')}")

def preload_auxiliary_models():
    """Pre-load spaCy and sentence-transformers before main model to avoid memory conflicts."""
    
    # 1. Pre-load spaCy (pip package, no special path needed)
    try:
        from scripts.utility import get_nlp_model
        nlp = get_nlp_model()
        if nlp:
            print("[INIT] OK spaCy model pre-loaded")
        else:
            print("[INIT] WARN spaCy model not available (will use fallback)")
    except Exception as e:
        print(f"[INIT] WARN spaCy pre-load failed: {e}")
    
    # 2. Pre-load sentence-transformers embedding model
    try:
        import os
        from pathlib import Path
        
        # Set cache path to match installer location
        cache_dir = Path(__file__).parent / "data" / "embedding_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL: Set offline mode BEFORE any huggingface imports
        # This prevents hanging when offline
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir.absolute())
        os.environ["HF_HOME"] = str(cache_dir.parent.absolute())
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir.absolute())
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode
        
        # Now trigger the lazy load
        cfg.context_injector._ensure_embedding_model()
        
        if cfg.context_injector.embedding:
            print("[INIT] OK Embedding model pre-loaded from cache")
        else:
            print("[INIT] WARN Embedding model not available (RAG disabled)")
    except Exception as e:
        print(f"[INIT] WARN Embedding pre-load failed: {e}")
        print("[INIT]   RAG features will be unavailable")

def main():
    """Main entry point for the application."""
    try:
        print("`main` Function Started.")
        
        # Load system constants from INI FIRST (installer-generated)
        from scripts.configuration import load_system_ini
        load_system_ini()
        
        # Parse command-line arguments and initialize platform
        args = parse_args()
        cfg.PLATFORM = args.platform

        # Load user settings from JSON (model, context, etc.)
        load_config()
        
        # Initialize TTS AFTER load_config so saved voice selection is respected
        from scripts.tools import initialize_tts
        initialize_tts()

        # Then initialize platform settings (paths, validation)
        initialize_platform_settings()
        
        # Setup directories and paths
        script_dir = setup_directories()
        print(f"Working directory: {short_path(script_dir)}")
        print(f"Data Directory: {short_path(cfg.DATA_DIR)}")
        print(f"Session History: {short_path(cfg.HISTORY_DIR)}")
        print(f"Temp Directory: {short_path(cfg.TEMP_DIR)}")
        
        # Initialize CPU configuration
        detect_cpu_config()
        print(f"CPU Configuration: {cfg.CPU_PHYSICAL_CORES} physical cores, "
              f"{cfg.CPU_LOGICAL_CORES} logical cores")
        
        # Print final configuration
        cfg.set_status("Config loaded")
        print_configuration()
        
        # Pre-load auxiliary models to avoid memory conflicts
        print("\n[INIT] Pre-loading auxiliary inference...")
        preload_auxiliary_models()
        
        # Launch display
        print("\nLaunching Gradio display...")
        launch_display()
        
    except Exception as e:
        print(f"Fatal error in launcher: {str(e)}")
        shutdown_platform()
        sys.exit(1)

if __name__ == "__main__":
    main()