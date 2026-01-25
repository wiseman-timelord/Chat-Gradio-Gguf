# scripts/settings.py

# Imports
import json
from pathlib import Path
import scripts.temporary as temporary
from scripts.models import get_available_models

CONFIG_PATH = Path("data/persistent.json")

# Default settings
DEFAULTS = {
    "MODEL_FOLDER": "path/to/your/models",
    "CONTEXT_SIZE": 32768,
    "VRAM_SIZE": 8192,
    "BATCH_SIZE": 1024,
    "TEMPERATURE": 0.66,
    "REPEAT_PENALTY": 1.0,
    "DYNAMIC_GPU_LAYERS": True,
    "MMAP": True,
    "MLOCK": True,
    "MAX_HISTORY_SLOTS": 10,
    "MAX_ATTACH_SLOTS": 8,
    "SESSION_LOG_HEIGHT": 650,
    "INPUT_LINES": 27,
    "PRINT_RAW_OUTPUT": False,
    "SHOW_THINK_PHASE": False,
    "CPU_ONLY_MODE": True,
    "VRAM_OPTIONS": [0, 756, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 49152, 65536],
    "CTX_OPTIONS": [1024, 2048, 4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072],
    "BATCH_OPTIONS": [128, 256, 512, 1024, 2048, 4096, 8192],
    "TEMP_OPTIONS": [0.0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0],
    "REPEAT_OPTIONS": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    "HISTORY_SLOT_OPTIONS": [4, 8, 10, 12, 16],
    "ATTACH_SLOT_OPTIONS": [2, 4, 6, 8, 10],
    "SESSION_LOG_HEIGHT_OPTIONS": [250, 450, 550, 600, 625, 650, 700, 800, 1000, 1400],
}

# Default configuration template
DEFAULT_CONFIG = {
    "model_settings": {
        "layer_allocation_mode": "SRAM_ONLY",
        "model_dir": "path/to/your/models",
        "model_name": "Select_a_model...",
        "context_size": 32768,
        "vram_size": 8192,
        "temperature": 0.66,
        "repeat_penalty": 1.0,
        "llama_cli_path": None,
        "llama_bin_path": None,
        "selected_gpu": None,
        "selected_cpu": None,
        "mmap": True,
        "mlock": True,
        "n_batch": 1024,
        "dynamic_gpu_layers": True,
        "max_history_slots": 12,
        "max_attach_slots": 6,
        "session_log_height": 650,
        "show_think_phase": False,
        "print_raw_output": False,
        "cpu_threads": None,
        "bleep_on_events": False,
        "use_python_bindings": True,
    }
}

# Functions...
def load_system_ini():
    """Load system constants from constants.ini (created by installer)."""
    import configparser
    
    ini_path = Path("data/constants.ini")
    if not ini_path.exists():
        raise RuntimeError(
            f"System configuration file not found: {ini_path}\n"
            "Re-run the installer to generate constants.ini."
        )
    
    try:
        config = configparser.ConfigParser()
        config.read(ini_path, encoding='utf-8')
        
        if 'system' not in config:
            raise RuntimeError("constants.ini missing [system] section")
        
        system = config['system']
        
        temporary.PLATFORM = system.get('platform')
        temporary.BACKEND_TYPE = system.get('backend_type', 'CPU_CPU')
        temporary.VULKAN_AVAILABLE = system.getboolean('vulkan_available', False)
        temporary.EMBEDDING_MODEL_NAME = system.get('embedding_model', 'BAAI/bge-small-en-v1.5')
        temporary.EMBEDDING_BACKEND = system.get('embedding_backend', 'sentence_transformers')
        temporary.GRADIO_VERSION = system.get('gradio_version', '3.50.2')
        temporary.LLAMA_CLI_PATH = system.get('llama_cli_path', None)
        temporary.LLAMA_BIN_PATH = system.get('llama_bin_path', None)
        
        print(f"[INI] Platform: {temporary.PLATFORM}")
        print(f"[INI] Backend: {temporary.BACKEND_TYPE}")
        print(f"[INI] Vulkan: {temporary.VULKAN_AVAILABLE}")
        print(f"[INI] Embedding Model: {temporary.EMBEDDING_MODEL_NAME}")
        print(f"[INI] Gradio Version: {temporary.GRADIO_VERSION}")
        
        temporary.OS_VERSION = system.get('os_version', 'unknown')
        print(f"[INI] OS Version: {temporary.OS_VERSION}")

        if temporary.PLATFORM == "windows":
            temporary.WINDOWS_VERSION = system.get('windows_version', temporary.OS_VERSION)
            print(f"[INI] Windows Version: {temporary.WINDOWS_VERSION}")
        else:
            temporary.WINDOWS_VERSION = None
        
        return True
        
    except Exception as e:
        raise RuntimeError(f"Cannot read constants.ini: {e}") from e

def load_config():
    """Load configuration with strict validation - no defaults, error on missing keys."""
    if not CONFIG_PATH.exists():
        raise RuntimeError(
            f"Configuration file not found: {CONFIG_PATH}\n"
            "Re-run the installer to generate persistent.json."
        )

    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Cannot read configuration file {CONFIG_PATH}: {e}") from e

    # Must have model_settings section
    if "model_settings" not in config:
        raise RuntimeError(
            f"Configuration file missing 'model_settings' section.\n"
            "Re-run the installer to regenerate persistent.json."
        )
    
    model_settings = config["model_settings"]
    
    # Required keys - error if missing
    required_keys = [
        "model_dir", "model_name", "context_size", "vram_size", "temperature",
        "repeat_penalty", "selected_gpu", "selected_cpu", "mmap", "mlock",
        "n_batch", "dynamic_gpu_layers", "max_history_slots", "max_attach_slots",
        "session_log_height", "show_think_phase", "print_raw_output", "cpu_threads",
        "bleep_on_events", "use_python_bindings", "layer_allocation_mode"
    ]
    
    missing_keys = [k for k in required_keys if k not in model_settings]
    if missing_keys:
        raise RuntimeError(
            f"Configuration file missing required keys: {', '.join(missing_keys)}\n"
            "Re-run the installer to regenerate persistent.json."
        )

    # Load MODEL_FOLDER first (critical for model discovery)
    temporary.MODEL_FOLDER = model_settings["model_dir"]
    print(f"[CONFIG] Model folder: {temporary.MODEL_FOLDER}")
    
    # Load hardware settings
    temporary.CONTEXT_SIZE = model_settings["context_size"]
    temporary.VRAM_SIZE = model_settings["vram_size"]
    temporary.TEMPERATURE = model_settings["temperature"]
    temporary.REPEAT_PENALTY = model_settings["repeat_penalty"]
    temporary.CPU_THREADS = model_settings["cpu_threads"]
    print(f"[CONFIG] Context: {temporary.CONTEXT_SIZE}, VRAM: {temporary.VRAM_SIZE}MB, Temp: {temporary.TEMPERATURE}")
    
    # Layer allocation
    temporary.LAYER_ALLOCATION_MODE = model_settings["layer_allocation_mode"]
    
    # Only force SRAM_ONLY if backend is CPU_CPU (from INI)
    if temporary.BACKEND_TYPE == "CPU_CPU":
        if temporary.LAYER_ALLOCATION_MODE != "SRAM_ONLY":
            print(f"[CONFIG] CPU_CPU backend requires SRAM_ONLY (was {temporary.LAYER_ALLOCATION_MODE})")
            temporary.LAYER_ALLOCATION_MODE = "SRAM_ONLY"
            model_settings["layer_allocation_mode"] = "SRAM_ONLY"
    
    print(f"[CONFIG] Layer allocation: {temporary.LAYER_ALLOCATION_MODE}")
    
    # Load UI settings
    temporary.MAX_HISTORY_SLOTS = model_settings["max_history_slots"]
    temporary.MAX_ATTACH_SLOTS = model_settings["max_attach_slots"]
    temporary.SESSION_LOG_HEIGHT = model_settings["session_log_height"]
    temporary.SHOW_THINK_PHASE = model_settings["show_think_phase"]
    temporary.PRINT_RAW_OUTPUT = model_settings["print_raw_output"]
    temporary.BLEEP_ON_EVENTS = model_settings["bleep_on_events"]
    temporary.USE_PYTHON_BINDINGS = model_settings["use_python_bindings"]
    print(f"[CONFIG] UI: History={temporary.MAX_HISTORY_SLOTS}, Attach={temporary.MAX_ATTACH_SLOTS}, Height={temporary.SESSION_LOG_HEIGHT}")

    # Load hardware selection settings
    temporary.SELECTED_GPU = model_settings["selected_gpu"]
    temporary.SELECTED_CPU = model_settings["selected_cpu"]
    temporary.MMAP = model_settings["mmap"]
    temporary.MLOCK = model_settings["mlock"]
    temporary.BATCH_SIZE = model_settings["n_batch"]
    temporary.DYNAMIC_GPU_LAYERS = model_settings["dynamic_gpu_layers"]
    
    # Ensure selected_cpu is always a label string
    if isinstance(temporary.SELECTED_CPU, (int, float)):
        print(f"[CONFIG] SELECTED_CPU is numeric ({temporary.SELECTED_CPU}) - setting to 'Auto-Select'")
        temporary.SELECTED_CPU = "Auto-Select"
    elif not isinstance(temporary.SELECTED_CPU, str):
        temporary.SELECTED_CPU = "Auto-Select"
    
    print(f"[CONFIG] SELECTED_GPU: {temporary.SELECTED_GPU}")
    print(f"[CONFIG] SELECTED_CPU: {temporary.SELECTED_CPU}")
    print(f"[CONFIG] MMAP: {temporary.MMAP}")
    print(f"[CONFIG] MLOCK: {temporary.MLOCK}")
    print(f"[CONFIG] BATCH_SIZE: {temporary.BATCH_SIZE}")
    print(f"[CONFIG] DYNAMIC_GPU_LAYERS: {temporary.DYNAMIC_GPU_LAYERS}")
    
    # Load model list from the configured folder
    temporary.AVAILABLE_MODELS = get_available_models()
    print(f"[CONFIG] Found {len(temporary.AVAILABLE_MODELS)} models in {temporary.MODEL_FOLDER}")
    
    # Load saved model name
    saved_model = model_settings["model_name"]
    print(f"[CONFIG] Saved model name: {saved_model}")
    
    if saved_model and saved_model in temporary.AVAILABLE_MODELS:
        temporary.MODEL_NAME = saved_model
        print(f"[CONFIG] Model '{saved_model}' found and selected")
    elif temporary.AVAILABLE_MODELS and len(temporary.AVAILABLE_MODELS) > 0:
        real_models = [m for m in temporary.AVAILABLE_MODELS if m != "Select_a_model..."]
        if real_models:
            temporary.MODEL_NAME = real_models[0]
            print(f"[CONFIG] Saved model not found, defaulting to '{temporary.MODEL_NAME}'")
        else:
            temporary.MODEL_NAME = "Select_a_model..."
            print(f"[CONFIG] No models found in folder")
    else:
        temporary.MODEL_NAME = "Select_a_model..."
        print(f"[CONFIG] No models available")

    if temporary.SELECTED_CPU and isinstance(temporary.SELECTED_CPU, str):
        import scripts.utility as utility
        cpu_info = utility.get_cpu_info()
        cpu_labels = [c["label"] for c in cpu_info]
        
        if len(cpu_info) == 1 and temporary.SELECTED_CPU == "Auto-Select":
            temporary.SELECTED_CPU = cpu_labels[0] if cpu_labels else "Default CPU"
            print(f"[CONFIG] Adjusted SELECTED_CPU to: {temporary.SELECTED_CPU}")
    
    temporary.set_status("Configuration loaded", console=True)
    print(f"[CONFIG] ==================== Load Complete ====================")
    return "Configuration loaded."

def save_config():
    """Save current configuration to persistent storage."""
    # Guarantee that SELECTED_CPU is always a string label
    cpu_label = getattr(temporary, 'SELECTED_CPU', None)
    if isinstance(cpu_label, (int, float)):
        cpu_label = "Auto-Select"

    config = {
        "model_settings": {
            "model_dir": temporary.MODEL_FOLDER,
            "model_name": temporary.MODEL_NAME,
            "context_size": temporary.CONTEXT_SIZE,
            "temperature": temporary.TEMPERATURE,
            "repeat_penalty": temporary.REPEAT_PENALTY,
            "vram_size": temporary.VRAM_SIZE,
            "selected_gpu": temporary.SELECTED_GPU,
            "selected_cpu": cpu_label,
            "mmap": temporary.MMAP,
            "mlock": temporary.MLOCK,
            "n_batch": temporary.BATCH_SIZE,
            "dynamic_gpu_layers": temporary.DYNAMIC_GPU_LAYERS,
            "max_history_slots": temporary.MAX_HISTORY_SLOTS,
            "max_attach_slots": temporary.MAX_ATTACH_SLOTS,
            "session_log_height": temporary.SESSION_LOG_HEIGHT,
            "show_think_phase": temporary.SHOW_THINK_PHASE,
            "print_raw_output": temporary.PRINT_RAW_OUTPUT,
            "cpu_threads": temporary.CPU_THREADS,
            "bleep_on_events": temporary.BLEEP_ON_EVENTS,
            "use_python_bindings": temporary.USE_PYTHON_BINDINGS,
            "layer_allocation_mode": getattr(temporary, 'LAYER_ALLOCATION_MODE', 'SRAM_ONLY'),
            "vulkan_enabled": getattr(temporary, 'VULKAN_AVAILABLE', False),
        }
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"[SAVE] Saving configuration:")
    print(f"[SAVE]   Model folder: {config['model_settings']['model_dir']}")
    print(f"[SAVE]   Model name: {config['model_settings']['model_name']}")
    print(f"[SAVE]   Selected CPU: {config['model_settings']['selected_cpu']}")
    
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    temporary.set_status("Settings saved")
    return "Settings saved"

def update_setting(key, value):
    """Update a single setting with optional model reload."""
    reload_required = False
    reload_keys = {"context_size", "n_gpu_layers", "vram_size", "model_folder", "model_name"}
    
    try:
        # Convert value to appropriate type
        if key in {"context_size", "vram_size", "n_gpu_layers", "n_batch", "max_history_slots", "max_attach_slots", "session_log_height", "cpu_threads"}:
            value = int(value)
        elif key in {"temperature", "repeat_penalty"}:
            value = float(value)
        elif key in {"mlock", "dynamic_gpu_layers"}:
            value = bool(value)
        
        # Set the attribute
        attr_name = key.upper() if hasattr(temporary, key.upper()) else key
        setattr(temporary, attr_name, value)
        
        reload_required = key in reload_keys
        
        if reload_required:
            from scripts.models import change_model
            reload_result = change_model(temporary.MODEL_NAME.split('/')[-1])
            message = f"Setting '{key}' updated to '{value}', model reload triggered."
            return message, *reload_result
        else:
            message = f"Setting '{key}' updated to '{value}'."
            return message, None, None
            
    except Exception as e:
        message = f"Error updating setting '{key}': {str(e)}"
        return message, None, None