# scripts/settings.py

# Imports
import json
from pathlib import Path
import scripts.temporary as temporary
from scripts.models import get_available_models  # Added import for model validation

CONFIG_PATH = Path("data/persistent.json")

# Default settings
DEFAULTS = {
    "MODEL_FOLDER": "path/to/your/models",
    "CONTEXT_SIZE": 8192,
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
    "VRAM_OPTIONS": [2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 49152, 65536],
    "CTX_OPTIONS": [8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072],
    "BATCH_OPTIONS": [128, 256, 512, 1024, 2048, 4096],
    "TEMP_OPTIONS": [0.0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0],
    "REPEAT_OPTIONS": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    "HISTORY_SLOT_OPTIONS": [4, 8, 10, 12, 16],
    "ATTACH_SLOT_OPTIONS": [2, 4, 6, 8, 10],
    "SESSION_LOG_HEIGHT_OPTIONS": [450, 475, 500, 550, 650, 800, 1050, 1300],
}

def load_config():
    """
    Load configuration from persistent.json and set in temporary.py.
    Always scan for available models and cache the result.
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        # Load model_settings
        model_settings = config.get("model_settings", {})
        # Handle DEFAULTS keys with potential mismatches
        if "model_dir" in model_settings:
            temporary.MODEL_FOLDER = model_settings["model_dir"]
        else:
            temporary.MODEL_FOLDER = DEFAULTS["MODEL_FOLDER"]
        
        # Load other settings
        if "context_size" in model_settings:
            temporary.CONTEXT_SIZE = model_settings["context_size"]
        if "temperature" in model_settings:
            temporary.TEMPERATURE = model_settings["temperature"]
        if "repeat_penalty" in model_settings:
            temporary.REPEAT_PENALTY = model_settings["repeat_penalty"]
        if "llama_cli_path" in model_settings:
            temporary.LLAMA_CLI_PATH = model_settings["llama_cli_path"]
        if "vram_size" in model_settings:
            temporary.VRAM_SIZE = model_settings["vram_size"]
        if "selected_gpu" in model_settings:
            temporary.SELECTED_GPU = model_settings["selected_gpu"]
        if "selected_cpu" in model_settings:
            temporary.SELECTED_CPU = model_settings["selected_cpu"]
        if "mmap" in model_settings:
            temporary.MMAP = model_settings["mmap"]
        if "mlock" in model_settings:
            temporary.MLOCK = model_settings["mlock"]
        if "n_batch" in model_settings:
            temporary.BATCH_SIZE = model_settings["n_batch"]
        if "dynamic_gpu_layers" in model_settings:
            temporary.DYNAMIC_GPU_LAYERS = model_settings["dynamic_gpu_layers"]
        if "max_history_slots" in model_settings:
            temporary.MAX_HISTORY_SLOTS = model_settings["max_history_slots"]
        if "max_attach_slots" in model_settings:
            temporary.MAX_ATTACH_SLOTS = model_settings["max_attach_slots"]
        if "session_log_height" in model_settings:
            temporary.SESSION_LOG_HEIGHT = model_settings["session_log_height"]
        
        # Load backend_config
        backend_config = config.get("backend_config", {})
        if "backend_type" in backend_config:
            temporary.BACKEND_TYPE = backend_config["backend_type"]
        if "llama_bin_path" in backend_config:
            temporary.LLAMA_CLI_PATH = backend_config["llama_bin_path"]
    else:
        # Set defaults if JSON doesn’t exist
        for key, value in DEFAULTS.items():
            setattr(temporary, key, value)
        temporary.BACKEND_TYPE = "Not Configured"
        temporary.LLAMA_CLI_PATH = "data/llama-vulkan-bin/llama-cli.exe"
        temporary.SELECTED_GPU = None
        temporary.SELECTED_CPU = None
        temporary.MODEL_NAME = "Browse_for_model_folder..."
    
    # Scan for available models and cache the result
    available_models = get_available_models()
    temporary.AVAILABLE_MODELS = available_models
    
    # Validate model_name against available models
    if CONFIG_PATH.exists():
        model_settings = config.get("model_settings", {})
        if "model_name" in model_settings and model_settings["model_name"] in available_models:
            temporary.MODEL_NAME = model_settings["model_name"]
        else:
            # Set to first available model if possible, otherwise "Browse_for_model_folder..."
            temporary.MODEL_NAME = available_models[0] if available_models else "Browse_for_model_folder..."
    else:
        temporary.MODEL_NAME = "Browse_for_model_folder..."
    
    return "Configuration loaded."

def save_config():
    """
    Save current settings from temporary.py to persistent.json.
    """
    config = {
        "model_settings": {
            "model_dir": temporary.MODEL_FOLDER,
            "model_name": temporary.MODEL_NAME,
            "context_size": temporary.CONTEXT_SIZE,
            "temperature": temporary.TEMPERATURE,
            "repeat_penalty": temporary.REPEAT_PENALTY,
            "llama_cli_path": temporary.LLAMA_CLI_PATH,
            "vram_size": temporary.VRAM_SIZE,
            "selected_gpu": temporary.SELECTED_GPU,
            "selected_cpu": temporary.SELECTED_CPU,
            "mmap": temporary.MMAP,
            "mlock": temporary.MLOCK,
            "n_batch": temporary.BATCH_SIZE,
            "dynamic_gpu_layers": temporary.DYNAMIC_GPU_LAYERS,
            "max_history_slots": temporary.MAX_HISTORY_SLOTS,
            "max_attach_slots": temporary.MAX_ATTACH_SLOTS,
            "session_log_height": temporary.SESSION_LOG_HEIGHT,
        },
        "backend_config": {
            "backend_type": temporary.BACKEND_TYPE,
            "llama_bin_path": temporary.LLAMA_CLI_PATH,
        }
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)
    return "Settings saved."