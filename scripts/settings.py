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
    "SESSION_LOG_HEIGHT_OPTIONS": [450, 475, 500, 550, 650, 800, 1050, 1300],
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
        "selected_cpu": None,  # Should be None or string label, NEVER a number
        "mmap": True,
        "mlock": True,
        "n_batch": 1024,
        "dynamic_gpu_layers": True,
        "max_history_slots": 12,
        "max_attach_slots": 6,
        "session_log_height": 500,
        "show_think_phase": False,
        "print_raw_output": False,
        "cpu_threads": None,
        "bleep_on_events": False,
        "use_python_bindings": True,
    }
}

# Functions...
def load_system_ini():
    """Load system constants from system.ini (created by installer)."""
    import configparser
    
    ini_path = Path("data/system.ini")
    if not ini_path.exists():
        raise RuntimeError(
            f"System configuration file not found: {ini_path}\n"
            "Re-run the installer to generate system.ini."
        )
    
    try:
        config = configparser.ConfigParser()
        config.read(ini_path, encoding='utf-8')
        
        if 'system' not in config:
            raise RuntimeError("system.ini missing [system] section")
        
        system = config['system']
        
        # Load INI-only constants
        temporary.PLATFORM = system.get('platform')
        temporary.BACKEND_TYPE = system.get('backend_type', 'CPU_CPU')
        temporary.VULKAN_AVAILABLE = system.getboolean('vulkan_available', False)
        temporary.EMBEDDING_MODEL_NAME = system.get('embedding_model', 'BAAI/bge-small-en-v1.5')
        
        print(f"[INI] Platform: {temporary.PLATFORM}")
        print(f"[INI] Backend: {temporary.BACKEND_TYPE}")
        print(f"[INI] Vulkan: {temporary.VULKAN_AVAILABLE}")
        print(f"[INI] Embedding: {temporary.EMBEDDING_MODEL_NAME}")
        
        # Windows-specific
        # OS version (generic)
        temporary.OS_VERSION = system.get('os_version', 'unknown')
        print(f"[INI] OS Version: {temporary.OS_VERSION}")

        # Windows-specific (legacy compatibility)
        if temporary.PLATFORM == "windows":
            temporary.WINDOWS_VERSION = system.get('windows_version', temporary.OS_VERSION)
            print(f"[INI] Windows Version: {temporary.WINDOWS_VERSION}")
        
        return True
        
    except Exception as e:
        raise RuntimeError(f"Cannot read system.ini: {e}") from e

def load_config():
    """Load configuration with validation and error handling."""
    if not CONFIG_PATH.exists():
        raise RuntimeError(
            f"Configuration file not found: {CONFIG_PATH}\n"
            "Re-run the installer or restore the file from backup."
        )

    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        raise RuntimeError(f"Cannot read configuration file {CONFIG_PATH}: {e}") from e

    model_settings = config.get("model_settings", {})

    # Validate required keys
    required_keys = {
        "model_dir", "context_size",
        "vram_size", "temperature", "repeat_penalty", "max_history_slots",
        "max_attach_slots", "session_log_height", "show_think_phase",
        "print_raw_output", "cpu_threads", "bleep_on_events", "use_python_bindings",
        "layer_allocation_mode"
    }
    
    missing = required_keys - model_settings.keys()
    if missing:
        raise RuntimeError(f"Configuration corrupted: missing keys {', '.join(missing)}")

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
    temporary.LAYER_ALLOCATION_MODE = model_settings.get("layer_allocation_mode", "SRAM_ONLY")
    
    # Only force SRAM_ONLY if backend is CPU_CPU (from INI)
    if temporary.BACKEND_TYPE == "CPU_CPU":
        if temporary.LAYER_ALLOCATION_MODE != "SRAM_ONLY":
            print(f"[CONFIG] CPU_CPU backend requires SRAM_ONLY (was {temporary.LAYER_ALLOCATION_MODE})")
            temporary.LAYER_ALLOCATION_MODE = "SRAM_ONLY"
            # Save corrected value back to JSON
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

    # Load optional settings with fallback
    optional_map = {
        "llama_cli_path": "LLAMA_CLI_PATH",
        "llama_bin_path": "LLAMA_BIN_PATH", 
        "selected_gpu": "SELECTED_GPU",
        "selected_cpu": "SELECTED_CPU",
        "mmap": "MMAP",
        "mlock": "MLOCK",
        "n_batch": "BATCH_SIZE",
        "dynamic_gpu_layers": "DYNAMIC_GPU_LAYERS",
    }

    for json_key, tmp_attr in optional_map.items():
        if json_key in model_settings:
            value = model_settings[json_key]
            # --- MIGRATION: ensure selected_cpu is always a label string ---
            if json_key == "selected_cpu":
                if isinstance(value, (int, float)):
                    print(f"[CONFIG] ⚠ SELECTED_CPU is numeric ({value}) - migrating to 'Auto-Select'")
                    value = "Auto-Select"
                elif not isinstance(value, str):
                    value = "Auto-Select"
            setattr(temporary, tmp_attr, value)
            print(f"[CONFIG] {tmp_attr}: {getattr(temporary, tmp_attr)}")
    
    # CRITICAL: Load model list from the configured folder
    temporary.AVAILABLE_MODELS = get_available_models()
    print(f"[CONFIG] Found {len(temporary.AVAILABLE_MODELS)} models in {temporary.MODEL_FOLDER}")
    
    # Load saved model name
    saved_model = model_settings.get("model_name")
    print(f"[CONFIG] Saved model name: {saved_model}")
    
    if saved_model and saved_model in temporary.AVAILABLE_MODELS:
        temporary.MODEL_NAME = saved_model
        print(f"[CONFIG] ✓ Model '{saved_model}' found and selected")
    elif temporary.AVAILABLE_MODELS and len(temporary.AVAILABLE_MODELS) > 0:
        # Filter out placeholder
        real_models = [m for m in temporary.AVAILABLE_MODELS if m != "Select_a_model..."]
        if real_models:
            temporary.MODEL_NAME = real_models[0]
            print(f"[CONFIG] ⚠ Saved model not found, defaulting to '{temporary.MODEL_NAME}'")
        else:
            temporary.MODEL_NAME = "Select_a_model..."
            print(f"[CONFIG] ⚠ No models found in folder")
    else:
        temporary.MODEL_NAME = "Select_a_model..."
        print(f"[CONFIG] ⚠ No models available")

    if temporary.SELECTED_CPU and isinstance(temporary.SELECTED_CPU, str):
        # Check if the saved CPU label is actually valid for current system
        import scripts.utility as utility
        cpu_info = utility.get_cpu_info()
        cpu_labels = [c["label"] for c in cpu_info]
        
        # If we only have one CPU and saved value is "Auto-Select", use the actual CPU label
        if len(cpu_info) == 1 and temporary.SELECTED_CPU == "Auto-Select":
            temporary.SELECTED_CPU = cpu_labels[0] if cpu_labels else "Default CPU"
            print(f"[CONFIG] Adjusted SELECTED_CPU to: {temporary.SELECTED_CPU}")
    
    # ← UNINDENT THESE (outside the if block)
    temporary.set_status("Configuration loaded", console=True)
    print(f"[CONFIG] ==================== Load Complete ====================")
    return "Configuration loaded."

def save_config():
    """Save current configuration to persistent storage."""
    # Guarantee that SELECTED_CPU is always a string label (never a number)
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
            "llama_cli_path": getattr(temporary, 'LLAMA_CLI_PATH', None),
            "llama_bin_path": getattr(temporary, 'LLAMA_BIN_PATH', None),
            "vram_size": temporary.VRAM_SIZE,
            "selected_gpu": temporary.SELECTED_GPU,
            "selected_cpu": cpu_label,  # <-- always a string
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
        }
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Add validation logging
    print(f"[SAVE] Saving configuration:")
    print(f"[SAVE]   Model folder: {config['model_settings']['model_dir']}")
    print(f"[SAVE]   Model name: {config['model_settings']['model_name']}")
    print(f"[SAVE]   Selected CPU: {config['model_settings']['selected_cpu']}")
    print(f"[SAVE]   CPU threads: {config['model_settings']['cpu_threads']}")
    
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