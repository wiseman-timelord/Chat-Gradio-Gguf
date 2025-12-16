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
    "SESSION_LOG_HEIGHT_OPTIONS": [450, 475, 500, 550, 650, 800, 1050, 1300],
}

# Functions...
def load_config():
    """
    Load configuration from persistent.json and set in temporary.py.
    If the file is missing or unreadable we abort instead of falling back.
    """
    if not CONFIG_PATH.exists():
        raise RuntimeError(
            f"Configuration file not found: {CONFIG_PATH}\n"
            "Re-run the installer or restore the file from backup."
        )

    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(
            f"Cannot read or decode configuration file {CONFIG_PATH}: {e}"
        ) from e

    model_settings = config.get("model_settings")
    if not isinstance(model_settings, dict):
        raise RuntimeError(
            "Invalid configuration: 'model_settings' object is missing or corrupted."
        )

    # --- mandatory keys -------------------------------------------------
    required = {
        "backend_type",
        "vulkan_available",
        "model_dir",
        "context_size",
        "vram_size",
        "temperature",
        "repeat_penalty",
        "max_history_slots",
        "max_attach_slots",
        "session_log_height",
        "show_think_phase",
        "print_raw_output",
        "cpu_threads",
        "bleep_on_events",
        "use_python_bindings",
    }
    missing = required - model_settings.keys()
    if missing:
        raise RuntimeError(
            f"Configuration corrupted: missing key(s) {', '.join(missing)}"
        )

    # --- load without any fallback --------------------------------------
    temporary.BACKEND_TYPE      = model_settings["backend_type"]
    temporary.VULKAN_AVAILABLE  = model_settings["vulkan_available"]
    temporary.LAYER_ALLOCATION_MODE = model_settings.get("layer_allocation_mode")
    temporary.MODEL_FOLDER      = model_settings["model_dir"]
    temporary.CONTEXT_SIZE      = model_settings["context_size"]
    temporary.VRAM_SIZE         = model_settings["vram_size"]
    temporary.TEMPERATURE       = model_settings["temperature"]
    temporary.REPEAT_PENALTY    = model_settings["repeat_penalty"]
    temporary.MAX_HISTORY_SLOTS = model_settings["max_history_slots"]
    temporary.MAX_ATTACH_SLOTS  = model_settings["max_attach_slots"]
    temporary.SESSION_LOG_HEIGHT= model_settings["session_log_height"]
    temporary.SHOW_THINK_PHASE  = model_settings["show_think_phase"]
    temporary.PRINT_RAW_OUTPUT  = model_settings["print_raw_output"]
    temporary.CPU_THREADS       = model_settings["cpu_threads"]
    temporary.BLEEP_ON_EVENTS   = model_settings["bleep_on_events"]
    temporary.USE_PYTHON_BINDINGS=model_settings["use_python_bindings"]

    # --- optional keys ---------------------------------------------------
    optional_map = {
        "llama_cli_path": "LLAMA_CLI_PATH",
        "llama_bin_path": "LLAMA_BIN_PATH", 
        "selected_gpu": "SELECTED_GPU",
        "selected_cpu": "SELECTED_CPU",  # ADD THIS
        "mmap": "MMAP",
        "mlock": "MLOCK",
        "n_batch": "BATCH_SIZE",
        "dynamic_gpu_layers": "DYNAMIC_GPU_LAYERS",
    }
    for json_key, tmp_attr in optional_map.items():
        if json_key in model_settings:
            setattr(temporary, tmp_attr, model_settings[json_key])

    # --- model list ------------------------------------------------------
    temporary.AVAILABLE_MODELS = get_available_models()
    temporary.MODEL_NAME = (
        model_settings.get("model_name")
        if model_settings.get("model_name") in temporary.AVAILABLE_MODELS
        else (temporary.AVAILABLE_MODELS[0] if temporary.AVAILABLE_MODELS else "Select_a_model...")
    )

    temporary.set_status("Configuration loaded", console=True)
    return "Configuration loaded."

def save_config():
    config = {
        "model_settings": {
            "model_dir": temporary.MODEL_FOLDER,
            "model_name": temporary.MODEL_NAME,
            "context_size": temporary.CONTEXT_SIZE,
            "temperature": temporary.TEMPERATURE,
            "repeat_penalty": temporary.REPEAT_PENALTY,
            "llama_cli_path": temporary.LLAMA_CLI_PATH,
            "llama_bin_path": temporary.LLAMA_BIN_PATH,
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
            "show_think_phase": temporary.SHOW_THINK_PHASE,
            "print_raw_output": temporary.PRINT_RAW_OUTPUT,
            "cpu_threads": temporary.CPU_THREADS,
            "vulkan_available": temporary.VULKAN_AVAILABLE,
            "backend_type": temporary.BACKEND_TYPE,
            "bleep_on_events": temporary.BLEEP_ON_EVENTS,
            "use_python_bindings": temporary.USE_PYTHON_BINDINGS,
            "layer_allocation_mode": temporary.LAYER_ALLOCATION_MODE,
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