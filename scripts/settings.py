# scripts/settings.py

# Imports
import json
from pathlib import Path
import scripts.temporary as temporary
from scripts.models import get_available_models, change_model  # Added import for model validation

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
    "PRINT_RAW_OUTPUT": False,
    "CPU_ONLY_MODE": True,
    "VRAM_OPTIONS": [2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 49152, 65536],
    "CTX_OPTIONS": [8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072],
    "BATCH_OPTIONS": [128, 256, 512, 1024, 2048, 4096],
    "TEMP_OPTIONS": [0.0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0],
    "REPEAT_OPTIONS": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    "HISTORY_SLOT_OPTIONS": [4, 8, 10, 12, 16],
    "ATTACH_SLOT_OPTIONS": [2, 4, 6, 8, 10],
    "SESSION_LOG_HEIGHT_OPTIONS": [450, 475, 500, 550, 650, 800, 1050, 1300],
}

# scripts/settings.py

def load_config():
    """
    Load configuration from persistent.json and set in temporary.py.
    Always scan for available models and cache the result.
    """
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        temporary.set_status("Config loaded", console=True)

        # Load model_settings
        model_settings = config.get("model_settings", {})
        if "model_dir" in model_settings:
            temporary.MODEL_FOLDER = model_settings["model_dir"]
        else:
            temporary.MODEL_FOLDER = DEFAULTS["MODEL_FOLDER"]

        # Load other settings
        for key, attr in {
            "context_size": "CONTEXT_SIZE",
            "temperature": "TEMPERATURE",
            "repeat_penalty": "REPEAT_PENALTY",
            "llama_cli_path": "LLAMA_CLI_PATH",
            "vram_size": "VRAM_SIZE",
            "selected_gpu": "SELECTED_GPU",
            "mmap": "MMAP",
            "mlock": "MLOCK",
            "n_batch": "BATCH_SIZE",
            "dynamic_gpu_layers": "DYNAMIC_GPU_LAYERS",
            "max_history_slots": "MAX_HISTORY_SLOTS",
            "max_attach_slots": "MAX_ATTACH_SLOTS",
            "session_log_height": "SESSION_LOG_HEIGHT",
            "print_raw_output": "PRINT_RAW_OUTPUT",
            "cpu_threads": "CPU_THREADS",
            "vulkan_available": "VULKAN_AVAILABLE",
            "llama_bin_path": "LLAMA_CLI_PATH"
        }.items():
            if key in model_settings:
                setattr(temporary, attr, model_settings[key])
               
    else:
        print(f"⚠️  Config missing/corrupted, re-run installer.")

    # Scan for available models and cache the result
    available_models = get_available_models()
    temporary.AVAILABLE_MODELS = available_models

    # Validate model_name against available models
    if CONFIG_PATH.exists():
        model_settings = config.get("model_settings", {})
        temporary.MODEL_NAME = (
            model_settings.get("model_name")
            if model_settings.get("model_name") in available_models
            else (available_models[0] if available_models else "Select_a_model...")
        )
    else:
        temporary.MODEL_NAME = "Select_a_model..."

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
            "mmap": temporary.MMAP,
            "mlock": temporary.MLOCK,
            "n_batch": temporary.BATCH_SIZE,
            "dynamic_gpu_layers": temporary.DYNAMIC_GPU_LAYERS,
            "max_history_slots": temporary.MAX_HISTORY_SLOTS,
            "max_attach_slots": temporary.MAX_ATTACH_SLOTS,
            "session_log_height": temporary.SESSION_LOG_HEIGHT,
            "print_raw_output": temporary.PRINT_RAW_OUTPUT,
            "cpu_threads": temporary.CPU_THREADS,
            "vulkan_available": getattr(temporary, "VULKAN_AVAILABLE", False),
            "llama_bin_path": temporary.LLAMA_CLI_PATH,
        }
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)

    temporary.set_status("Settings saved")
    return "Settings saved"

def update_setting(key, value):
    """Update a setting and return components requiring reload if necessary, with a confirmation message."""
    reload_required = False
    try:
        if key == "temperature":
            temporary.TEMPERATURE = float(value)
        elif key == "context_size":
            temporary.CONTEXT_SIZE = int(value)
            reload_required = True
        elif key == "n_gpu_layers":
            temporary.GPU_LAYERS = int(value)
            reload_required = True
        elif key == "vram_size":
            temporary.VRAM_SIZE = int(value)
            reload_required = True
        elif key == "selected_gpu":
            temporary.SELECTED_GPU = value
        elif key == "selected_cpu":
            temporary.SELECTED_CPU = value
        elif key == "repeat_penalty":
            temporary.REPEAT_PENALTY = float(value)
        elif key == "mlock":
            temporary.MLOCK = bool(value)
        elif key == "n_batch":
            temporary.BATCH_SIZE = int(value)
        elif key == "model_folder":
            temporary.MODEL_FOLDER = value
            reload_required = True
        elif key == "model_name":
            temporary.MODEL_NAME = value
            reload_required = True       
        elif key == "max_history_slots":
            temporary.MAX_HISTORY_SLOTS = int(value)
        elif key == "max_attach_slots":
            temporary.MAX_ATTACH_SLOTS = int(value)
        elif key == "session_log_height":
            temporary.SESSION_LOG_HEIGHT = int(value)

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