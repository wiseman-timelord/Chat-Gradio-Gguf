# Script: `.\scripts\models.py`

# Imports...
import time, re, traceback, json
from pathlib import Path
import gradio as gr
from scripts.prompts import get_system_message, get_reasoning_instruction
import scripts.temporary as temporary
from scripts.prompts import prompt_templates, get_system_message
from scripts.temporary import (
    CONTEXT_SIZE, GPU_LAYERS, BATCH_SIZE, LLAMA_CLI_PATH, BACKEND_TYPE, VRAM_SIZE,
    DYNAMIC_GPU_LAYERS, MMAP, current_model_settings, handling_keywords, llm,
    MODEL_NAME, REPEAT_PENALTY, TEMPERATURE, MODELS_LOADED, CHAT_FORMAT_MAP
)
from llama_cpp import Llama

# Functions...
def get_model_metadata(model_path: str) -> tuple[dict, int]:
    print(f"Getting metadata for: {model_path}")
    metadata = {}
    layer_count = 0

    try:
        # First try low-level metadata extraction for Qwen models
        from llama_cpp.llama_cpp import llama_model_loader
        with open(model_path, 'rb') as f:
            loader = llama_model_loader(f)
            for i in range(loader.n_kv()):
                key = loader.kv_key_at(i).decode('utf-8')
                value = loader.kv_value_at(i)
                if isinstance(value, (int, float, str, bool)):
                    metadata[key] = value
                elif isinstance(value, list):
                    metadata[key] = [v.decode('utf-8') if isinstance(v, bytes) else v for v in value]
            loader.close()

        architecture = metadata.get('general.architecture', 'unknown').lower()
        print(f"Debug: Architecture detected via low-level loader: '{architecture}'")

        # Qwen-specific layer detection
        layer_keys = [
            f"{architecture}.block_count",  # For qwen3/qwen2
            "qwen.block_count",             # Fallback for older versions
            "num_hidden_layers",
            "n_layers",
            "block_count"
        ]

        for key in layer_keys:
            if key in metadata:
                try:
                    layer_count = int(metadata[key])
                    print(f"Debug: Found layer count ({layer_count}) using key '{key}'")
                    break
                except (ValueError, TypeError):
                    continue
        else:
            raise ValueError(f"Could not determine layer count for '{model_path}'")

    except Exception as low_level_error:
        print(f"Low-level extraction failed, trying normal load: {low_level_error}")
        try:
            # Fallback to normal model load
            model = Llama(model_path, n_ctx=4096, n_batch=1, n_gpu_layers=0, verbose=True)
            metadata = model.metadata
            architecture = metadata.get('general.architecture', 'unknown').lower()
            print(f"Debug: Architecture detected via normal load: '{architecture}'")

            layer_keys = [
                f"{architecture}.block_count",
                f"{architecture}.num_hidden_layers",
                "num_hidden_layers",
                "n_layers",
                "block_count"
            ]
            
            for key in layer_keys:
                if key in metadata:
                    try:
                        layer_count = int(metadata[key])
                        print(f"Debug: Found layer count ({layer_count}) using key '{key}'")
                        break
                    except (ValueError, TypeError):
                        continue
            else:
                layer_count = int(metadata.get("llama.block_count", 0))

            del model

        except Exception as e:
            error_msg = f"Failed both metadata methods for '{model_path}': {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            raise ValueError(error_msg)

    return metadata, layer_count

def get_chat_format(metadata):
    """Determine chat format using CHAT_FORMAT_MAP from temporary"""
    architecture = metadata.get('general.architecture', 'unknown').lower()
    
    # Find longest matching key prefix
    match = None
    for key in sorted(temporary.CHAT_FORMAT_MAP.keys(), key=len, reverse=True):
        if architecture.startswith(key):
            match = key
            break
            
    return temporary.CHAT_FORMAT_MAP.get(match, 'llama-2')  # Default fallback

import traceback
from llama_cpp import Llama

def get_model_layers(model_path: str) -> int:
    """
    Get the number of layers for a GGUF model.
    """
    _, layers = get_model_metadata(model_path)
    return layers

def get_model_size(model_path: str) -> float:
    return Path(model_path).stat().st_size / (1024 * 1024)

def clean_content(role, content):
    """Remove prefixes from session_log content for model input."""
    if role == 'user':
        return content.replace("User:\n", "", 1).strip()
    return content.strip()

def set_cpu_affinity():
    from scripts import utility
    cpu_only_backends = ["CPU Only - AVX2", "CPU Only - AVX512", "CPU Only - NoAVX", "CPU Only - OpenBLAS"]
    if temporary.BACKEND_TYPE in cpu_only_backends and temporary.SELECTED_CPU:
        cpus = utility.get_cpu_info()
        selected_cpu = next((cpu for cpu in cpus if cpu["label"] == temporary.SELECTED_CPU), None)
        if selected_cpu:
            try:
                p = psutil.Process()
                p.cpu_affinity(selected_cpu["core_range"])
                print(f"Set CPU affinity to {selected_cpu['label']}")
            except Exception as e:
                print(f"Failed to set CPU affinity: {e}")

def get_available_models():
    model_dir = Path(temporary.MODEL_FOLDER)
    print(f"Scanning directory: {model_dir}")
    files = list(model_dir.glob("*.gguf"))
    models = [f.name for f in files if f.is_file()]
    if models:
        choices = models
    else:
        choices = ["Select_a_model..."]
    print(f"Models Found: {choices}")
    return choices

def get_model_settings(model_name):
    model_name_lower = model_name.lower()
    is_uncensored = any(keyword in model_name_lower for keyword in handling_keywords["uncensored"])
    is_reasoning = any(keyword in model_name_lower for keyword in handling_keywords["reasoning"])
    is_nsfw = any(keyword in model_name_lower for keyword in handling_keywords["nsfw"])
    is_code = any(keyword in model_name_lower for keyword in handling_keywords["code"])
    is_roleplay = any(keyword in model_name_lower for keyword in handling_keywords["roleplay"])
    return {
        "category": "chat",
        "is_uncensored": is_uncensored,
        "is_reasoning": is_reasoning,
        "is_nsfw": is_nsfw,
        "is_code": is_code,
        "is_roleplay": is_roleplay,
        "detected_keywords": [kw for kw in handling_keywords if any(k in model_name_lower for k in handling_keywords[kw])]
    }

def calculate_gpu_layers(models, available_vram):
    from math import floor
    if not models or available_vram <= 0:
        return {model: 0 for model in models}
    total_size = sum(get_model_size(Path(temporary.MODEL_FOLDER) / model) for model in models if model != "Select_a_model...")
    if total_size == 0:
        return {model: 0 for model in models}
    vram_allocations = {
        model: (get_model_size(Path(temporary.MODEL_FOLDER) / model) / total_size) * available_vram
        for model in models if model != "Select_a_model..."
    }
    gpu_layers = {}
    for model in models:
        if model == "Select_a_model...":
            gpu_layers[model] = 0
            continue
        model_path = Path(temporary.MODEL_FOLDER) / model
        num_layers = get_model_layers(str(model_path))
        if num_layers == 0:
            gpu_layers[model] = 0
            continue
        model_file_size = get_model_size(str(model_path))
        adjusted_model_size = model_file_size * 1.125
        layer_size = adjusted_model_size / num_layers if num_layers > 0 else 0
        max_layers = floor(vram_allocations[model] / layer_size) if layer_size > 0 else 0
        gpu_layers[model] = min(max_layers, num_layers) if DYNAMIC_GPU_LAYERS else num_layers
    return gpu_layers

def inspect_model(model_dir, model_name, vram_size):
    from scripts.settings import save_config
    if model_name == "Select_a_model...":
        return "Select a model to inspect."
    model_path = Path(model_dir) / model_name
    if not model_path.exists():
        return f"Model file '{model_path}' not found."
    save_config()
    try:
        metadata, layers = get_model_metadata(str(model_path))
        architecture = metadata.get('general.architecture', 'unknown')
        params_str = metadata.get('general.size_label', 'Unknown')
        max_ctx = metadata.get(f'{architecture}.context_length', 'Unknown')
        embed = metadata.get(f'{architecture}.embedding_length', 'Unknown')
        model_size_mb = get_model_size(str(model_path))
        model_size_gb = model_size_mb / 1024
        if layers > 0:
            fit_layers = calculate_single_model_gpu_layers_with_layers(
                str(model_path), vram_size, layers, DYNAMIC_GPU_LAYERS
            )
        else:
            fit_layers = "Unknown"
        author = metadata.get('general.organization', 'Unknown')
        return (
            f"Results: Params = {params_str}, "
            f"Fit/Layers = {fit_layers}/{layers}, "
            f"Size = {model_size_gb:.2f} GB, "
            f"Max Ctx = {max_ctx}, "
            f"Embed = {embed}, "
            f"Author = {author}"
        )
    except Exception as e:
        return f"Error inspecting model: {str(e)}"

def load_models(model_folder, model, vram_size, llm_state, models_loaded_state):
    from scripts.settings import save_config
    """
    Prepare the model environment for inference using the llama.cpp binary.
    
    Args:
        model_folder (str): Directory containing the model files.
        model (str): Name of the model file to load.
        vram_size (int): Available VRAM size in MB.
        llm_state: Current state of the LLM (not used with binary).
        models_loaded_state (bool): Whether models are currently loaded.
    
    Returns:
        tuple: (status message, success boolean, llm_state, models_loaded_state)
    """
    print(f"Initiating load for model: {model} from {model_folder}...")
    from scripts.temporary import DYNAMIC_GPU_LAYERS

    save_config()

    if model in ["Select_a_model...", "No models found"]:
        return "Select a model to load.", False, llm_state, models_loaded_state

    model_path = Path(model_folder) / model
    if not model_path.exists():
        return f"Error: Model file '{model_path}' not found.", False, llm_state, models_loaded_state

    try:
        metadata, num_layers = get_model_metadata(str(model_path))
        if num_layers <= 0:
            return f"Error: Could not determine layer count for model '{model}'.", False, llm_state, models_loaded_state
        chat_format = get_chat_format(metadata)

        # Calculate GPU layers only if not CPU-only backend
        cpu_only_backends = ["CPU Only - AVX2", "CPU Only - AVX512", "CPU Only - NoAVX", "CPU Only - OpenBLAS"]
        if BACKEND_TYPE not in cpu_only_backends:
            GPU_LAYERS = calculate_single_model_gpu_layers_with_layers(
                str(model_path), vram_size, num_layers, DYNAMIC_GPU_LAYERS
            )
        else:
            GPU_LAYERS = 0
            print(f"Debug: CPU-only backend detected ({BACKEND_TYPE}). Setting GPU_LAYERS to 0.")

        # Store model details for inference
        MODEL_NAME = model
        status = f"Model '{model}' prepared for loading with chat_format '{chat_format}'. GPU layers: {GPU_LAYERS}/{num_layers}"
        return status, True, None, True  # No llm_state needed with binary
    except Exception as e:
        error_msg = f"Error preparing model: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, False, llm_state, models_loaded_state

def calculate_single_model_gpu_layers_with_layers(model_path: str, available_vram: int, num_layers: int, dynamic_gpu_layers: bool = True) -> int:
    from math import floor
    if num_layers <= 0 or available_vram <= 0:
        return 0
    model_file_size = get_model_size(model_path)
    metadata, _ = get_model_metadata(model_path)
    factor = 1.2 if metadata.get('general.architecture') == 'llama' else 1.1
    adjusted_model_size = model_file_size * factor
    layer_size = adjusted_model_size / num_layers if num_layers > 0 else 0
    max_layers = floor(available_vram / layer_size) if layer_size > 0 else 0
    result = min(max_layers, num_layers) if dynamic_gpu_layers else num_layers
    return result

def unload_models(llm_state, models_loaded_state):
    import gc
    if models_loaded_state:
        del llm_state
        gc.collect()
        print(f"Model {temporary.MODEL_NAME} unloaded.")
        return "Model unloaded successfully.", None, False
    print("Warning: No model was loaded to unload.")
    return "No model loaded to unload.", llm_state, models_loaded_state

def get_response_stream(session_log, settings, web_search_enabled=False, search_results=None, cancel_event=None, llm_state=None, models_loaded_state=False):
    """
    Generate a response stream by calling the llama.cpp binary.
    
    Args:
        session_log (list): List of conversation messages.
        settings (dict): Model settings (temperature, repeat_penalty, etc.).
        web_search_enabled (bool): Whether web search is enabled.
        search_results (str): Web search results to include.
        cancel_event (Event): Event to cancel the stream.
        llm_state: Not used (kept for compatibility).
        models_loaded_state (bool): Whether a model is loaded.
    
    Yields:
        str: Chunks of the generated response.
    """
    if not models_loaded_state:
        yield "Error: No model loaded. Please load a model first."
        return

    model_path = Path(MODEL_FOLDER) / MODEL_NAME
    if not model_path.exists():
        yield f"Error: Model file '{model_path}' not found."
        return

    # Prepare system message and user query
    system_message = get_system_message(
        is_uncensored=settings.get("is_uncensored", False),
        is_nsfw=settings.get("is_nsfw", False),
        web_search_enabled=web_search_enabled,
        is_reasoning=settings.get("is_reasoning", False),
        is_roleplay=settings.get("is_roleplay", False)
    ) + "\nRespond directly without prefixes like 'AI-Chat:'."

    if web_search_enabled and search_results:
        system_message += "\n\nSearch Results:\n" + re.sub(r'https?://(www\.)?([^/]+).*', r'\2', str(search_results))

    if not session_log or len(session_log) < 2 or session_log[-2]['role'] != 'user':
        yield "Error: No user input to process."
        return

    user_query = clean_content('user', session_log[-2]['content'])
    prompt = f"{system_message}\n\nUser: {user_query}\nAssistant:"

    # Construct the command for llama.cpp binary
    command = [
        LLAMA_CLI_PATH,
        '-m', str(model_path),
        '-p', prompt,
        '-n', str(BATCH_SIZE),
        '--temp', str(settings.get("temperature", TEMPERATURE)),
        '--repeat_penalty', str(settings.get("repeat_penalty", REPEAT_PENALTY)),
        '-c', str(CONTEXT_SIZE),
        '-ngl', str(GPU_LAYERS),
    ]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        for line in process.stdout:
            if cancel_event and cancel_event.is_set():
                process.terminate()
                yield "<CANCELLED>"
                return
            content = line.strip()
            if content:
                content = re.sub(r'^AI-Chat:[\s\n]*', '', content, flags=re.IGNORECASE)
                content = re.sub(r'\n{2,}', '\n', content)
                yield content
        process.wait()
        if process.returncode != 0:
            error = process.stderr.read()
            yield f"Error from binary: {error}"
    except Exception as e:
        yield f"Error generating response: {str(e)}"