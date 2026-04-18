# launcher.py - The entry point of the scripts of the main program.

# =============================================================================
# CRITICAL: Pydantic v1/v2 compatibility shim MUST be before ANY other imports
# — with one exception: libraries that REQUIRE pydantic v2 must be pre-loaded
#   here first. Once a module is cached in sys.modules, later "import X" calls
#   return the cached copy and are unaffected by the shim that follows.
# =============================================================================
import sys as _sys
import os as _os

# Set embedding cache env vars FIRST before ANY imports
_cache_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "data", "embedding_cache")
_os.makedirs(_cache_dir, exist_ok=True)
_os.environ["TRANSFORMERS_CACHE"] = _cache_dir
_os.environ["HF_HOME"] = _os.path.dirname(_cache_dir)
_os.environ["SENTENCE_TRANSFORMERS_HOME"] = _cache_dir
_os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode
_os.environ["HF_HUB_OFFLINE"] = "1"  # CRITICAL: Force fully offline mode
_os.environ["TRANSFORMERS_OFFLINE"] = "1"

# -----------------------------------------------------------------------------
# PRE-IMPORT pydantic-v2-dependent libraries BEFORE the shim replaces pydantic.
#
# Why: modern spaCy (3.5+) and sentence_transformers import pydantic v2 symbols
# (field_validator, model_validator, etc.) at module load time.  The shim below
# remaps sys.modules["pydantic"] to pydantic.v1 for Gradio 3.x compatibility,
# which breaks those imports if they happen after the shim is applied.
#
# Fix: load them now while the real pydantic v2 is still present.  Python caches
# the result in sys.modules, so any later "import spacy" / "import
# sentence_transformers" in display.py or utility.py just returns the cached
# module — the shim has no effect on already-resolved references.
# -----------------------------------------------------------------------------
try:
    import spacy as _spacy_preload
    print(f"[COMPAT] spaCy {_spacy_preload.__version__} pre-loaded before pydantic shim")
except Exception as _e:
    print(f"[COMPAT] spaCy pre-load skipped (will retry later): {_e}")

try:
    import sentence_transformers as _st_preload
    print(f"[COMPAT] sentence_transformers pre-loaded before pydantic shim")
except Exception as _e:
    print(f"[COMPAT] sentence_transformers pre-load skipped (RAG may be unavailable): {_e}")

# -----------------------------------------------------------------------------
# Pydantic v1/v2 compatibility shim for Gradio 3.x
# Gradio 3.50.2 requires Pydantic v1 APIs.
# Applied AFTER the pydantic-v2-dependent pre-imports above.
# -----------------------------------------------------------------------------
try:
    import pydantic as _pydantic
    _pydantic_major = int(_pydantic.VERSION.split(".")[0])
    if _pydantic_major >= 2:
        try:
            import pydantic.v1 as _pydantic_v1
            # Remap both the top-level module and common sub-modules
            _sys.modules["pydantic"] = _pydantic_v1
            _sys.modules["pydantic.fields"] = _pydantic_v1.fields
            _sys.modules["pydantic.main"] = _pydantic_v1.main
            _sys.modules["pydantic.validators"] = _pydantic_v1.validators
            _sys.modules["pydantic.types"] = _pydantic_v1.types
            _sys.modules["pydantic.errors"] = _pydantic_v1.errors
            _sys.modules["pydantic.networks"] = _pydantic_v1.networks
            print(f"[COMPAT] Pydantic v{_pydantic_major} detected — applied v1 shim for Gradio 3.x")
        except ImportError:
            print(f"[COMPAT] WARNING: Pydantic v{_pydantic_major} detected but v1 compat layer missing.")
            print("[COMPAT]   Gradio 3.x requires pydantic <2. Run: pip install 'pydantic <2'")
except Exception as _e:
    print(f"[COMPAT] Pydantic check skipped: {_e}")

# Disable Logging (before any library imports)
# CRITICAL: Must import logging.config explicitly as it is a submodule
import logging as _logging
import logging.config as _logging_config
_logging_config.dictConfig = lambda **__: None
_logging_config.fileConfig = lambda **__: None
_logging.basicConfig = lambda **__: None

# =============================================================================
# NOW safe to import other modules
# =============================================================================
import sys
import os                   # FIX: was only imported as _os above; functions below use `os` unaliased
import argparse
import time
from pathlib import Path
from scripts.utility import short_path
import scripts.configure as cfg
from scripts.configure import load_config
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
        from scripts.display import close_browser
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
    # 1. spaCy — already loaded before the pydantic shim (see top of file).
    #    get_nlp_model() just retrieves the cached module; no re-import needed.
    try:
        from scripts.utility import get_nlp_model
        nlp = get_nlp_model()
        if nlp:
            print("[INIT] OK spaCy model pre-loaded")
        else:
            print("[INIT] WARN spaCy model not available (will use fallback)")
    except Exception as e:
        print(f"[INIT] WARN spaCy pre-load failed: {e}")

    # 2. Pre-load sentence-transformers embedding model.
    #    sentence_transformers itself was pre-loaded before the pydantic shim
    #    (see top of file), so _ensure_embedding_model() is safe to call here.
    try:
        # Now trigger the lazy load of the actual embedding model weights
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
        from scripts.configure import load_system_ini
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