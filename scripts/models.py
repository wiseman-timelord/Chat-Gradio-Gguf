# Script: `.\scripts\models.py`

# Imports...
import time, re
from pathlib import Path
import gradio as gr
from scripts.prompts import get_system_message, get_reasoning_instruction
import scripts.temporary as temporary
from scripts.prompts import prompt_templates
from scripts.temporary import (
    CONTEXT_SIZE, GPU_LAYERS, BATCH_SIZE, LLAMA_CLI_PATH, BACKEND_TYPE, VRAM_SIZE,
    DYNAMIC_GPU_LAYERS, MMAP, current_model_settings, handling_keywords, llm,
    MODEL_NAME, REPEAT_PENALTY, TEMPERATURE, MODELS_LOADED, CHAT_FORMAT_MAP
)

# Functions...
def get_chat_format(metadata):
    """
    Determine the chat format based on the model's architecture.
    """
    architecture = metadata.get('general.architecture', 'unknown')
    return CHAT_FORMAT_MAP.get(architecture, 'llama2')

def get_model_metadata(model_path: str) -> dict:
    """
    Retrieve metadata from a GGUF model, including the number of layers.
    """
    try:
        from llama_cpp import Llama
        chat_format = 'chatml' if 'qwen' in model_path.lower() else None
        model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_batch=1,
            n_gpu_layers=0,
            verbose=True,
            chat_format=chat_format
        )
        metadata = model.metadata
        print(f"Debug: Metadata keys for '{model_path}': {list(metadata.keys())}")
        
        architecture = metadata.get('general.architecture', 'unknown')
        layers = metadata.get(f'{architecture}.block_count', 0)
        
        if layers == 0:
            possible_keys = ['block_count', 'layer_count', 'num_hidden_layers', 'num_layers']
            for key in metadata:
                if any(pk in key for pk in possible_keys):
                    layers = metadata[key]
                    print(f"Debug: Found layers ({layers}) in key '{key}'")
                    break
            else:
                print(f"Warning: Could not find layer count for '{model_path}' in metadata.")
                layers = 0
        
        metadata['layers'] = layers
        del model
        return metadata
    except Exception as e:
        import traceback
        print(f"Error reading model metadata for '{model_path}': {str(e)}\n{traceback.format_exc()}")
        return {}

def get_model_layers(model_path: str) -> int:
    """
    Get the number of layers for a GGUF model.
    """
    metadata = get_model_metadata(model_path)
    layers = metadata.get('layers', 0)
    return int(layers)

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
        metadata = get_model_metadata(str(model_path))
        architecture = metadata.get('general.architecture', 'unknown')
        params_str = metadata.get('general.size_label', 'Unknown')
        layers = metadata.get(f'{architecture}.block_count', 'Unknown')
        max_ctx = metadata.get(f'{architecture}.context_length', 'Unknown')
        embed = metadata.get(f'{architecture}.embedding_length', 'Unknown')
        model_size_mb = get_model_size(str(model_path))
        model_size_gb = model_size_mb / 1024
        if isinstance(layers, int) and layers > 0:
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
    from scripts.temporary import CONTEXT_SIZE, BATCH_SIZE, MMAP, DYNAMIC_GPU_LAYERS
    from scripts.settings import save_config
    from pathlib import Path
    import traceback

    save_config()

    if model in ["Select_a_model...", "No models found"]:
        return "Select a model to load.", False, llm_state, models_loaded_state

    model_path = Path(model_folder) / model
    if not model_path.exists():
        return f"Error: Model file '{model_path}' not found.", False, llm_state, models_loaded_state

    metadata = get_model_metadata(str(model_path))
    chat_format = get_chat_format(metadata)

    num_layers = get_model_layers(str(model_path))
    if num_layers <= 0:
        return f"Error: Could not determine layer count for model '{model}'.", False, llm_state, models_loaded_state

    temporary.GPU_LAYERS = calculate_single_model_gpu_layers_with_layers(
        str(model_path), vram_size, num_layers, DYNAMIC_GPU_LAYERS
    )

    try:
        from llama_cpp import Llama
    except ImportError:
        return "Error: llama-cpp-python not installed. Python bindings are required.", False, llm_state, models_loaded_state

    try:
        if models_loaded_state:
            unload_models(llm_state, models_loaded_state)

        print(f"Debug: Loading model '{model}' from '{model_folder}' with Python bindings and chat_format '{chat_format}'")
        new_llm = Llama(
            model_path=str(model_path),
            n_ctx=temporary.CONTEXT_SIZE,
            n_gpu_layers=temporary.GPU_LAYERS,
            n_batch=temporary.BATCH_SIZE,
            mmap=temporary.MMAP,
            mlock=temporary.MLOCK,
            verbose=True,
            chat_format=chat_format
        )

        test_output = new_llm.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=temporary.BATCH_SIZE,
            stream=False
        )
        print(f"Debug: Test inference successful: {test_output}")

        temporary.MODEL_NAME = model
        status = f"Model '{model}' loaded successfully with chat_format '{chat_format}'. GPU layers: {temporary.GPU_LAYERS}/{num_layers}"
        return status, True, new_llm, True
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, False, None, False

def calculate_single_model_gpu_layers_with_layers(model_path: str, available_vram: int, num_layers: int, dynamic_gpu_layers: bool = True) -> int:
    from math import floor
    if num_layers <= 0 or available_vram <= 0:
        print("Debug: Invalid input (layers or VRAM), returning 0 layers")
        return 0
    model_file_size = get_model_size(model_path)
    metadata = get_model_metadata(model_path)
    factor = 1.2 if metadata.get('general.architecture') == 'llama' else 1.1
    adjusted_model_size = model_file_size * factor
    layer_size = adjusted_model_size / num_layers if num_layers > 0 else 0
    max_layers = floor(available_vram / layer_size) if layer_size > 0 else 0
    result = min(max_layers, num_layers) if dynamic_gpu_layers else num_layers
    print(f"Debug: Model size = {model_file_size:.2f} MB, Layers = {num_layers}, VRAM = {available_vram} MB")
    print(f"Debug: Adjusted size = {adjusted_model_size:.2f} MB, Layer size = {layer_size:.2f} MB")
    print(f"Debug: Max layers with VRAM = {max_layers}, Final result = {result}")
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
    import re
    import traceback
    from scripts import temporary
    from scripts.utility import clean_content
    from scripts.prompts import get_system_message

    def should_stream(input_text, settings):
        """Determine if response should be streamed based on input characteristics"""
        stream_keywords = ["write", "generate", "story", "report", "essay", "explain", "describe"]
        input_length = len(input_text.strip())
        is_long_input = input_length > 100
        is_creative_task = any(keyword in input_text.lower() for keyword in stream_keywords)
        is_interactive_mode = settings.get("is_reasoning", False)
        return is_creative_task or is_long_input or (is_interactive_mode and input_length > 50)

    if not models_loaded_state or llm_state is None:
        yield "Error: No model loaded. Please load a model first."
        return

    n_ctx = temporary.CONTEXT_SIZE

    system_message = get_system_message(
        is_uncensored=settings.get("is_uncensored", False),
        is_nsfw=settings.get("is_nsfw", False),
        web_search_enabled=web_search_enabled,
        is_reasoning=settings.get("is_reasoning", False),
        is_roleplay=settings.get("is_roleplay", False)
    ) + "\nRespond directly without prefixes like 'AI-Chat:'."
    
    if web_search_enabled and search_results:
        system_message += "\n\nSearch Results:\n" + re.sub(r'https?://(www\.)?([^/]+).*', r'\2', str(search_results))

    try:
        system_tokens = len(llm_state.tokenize(system_message.encode('utf-8')))
    except Exception as e:
        yield f"Error tokenizing system message: {str(e)}"
        return

    if not session_log or len(session_log) < 2 or session_log[-2]['role'] != 'user':
        yield "Error: No user input to process."
        return

    user_query = clean_content('user', session_log[-2]['content'])
    try:
        user_tokens = len(llm_state.tokenize(user_query.encode('utf-8')))
    except Exception as e:
        yield f"Error tokenizing user query: {str(e)}"
        return

    available_tokens = n_ctx - system_tokens - user_tokens - (temporary.BATCH_SIZE // 8)
    if available_tokens < 0:
        yield "Error: Context size exceeded."
        return

    messages = [{"role": "system", "content": system_message}]
    current_tokens = 0
    
    for msg in reversed(session_log[:-2]):
        content = clean_content(msg['role'], msg['content'])
        try:
            msg_tokens = len(llm_state.tokenize(content.encode('utf-8')))
            if current_tokens + msg_tokens > available_tokens:
                break
            messages.insert(1, {"role": msg['role'], "content": content})
            current_tokens += msg_tokens
        except Exception:
            continue

    messages.append({"role": "user", "content": user_query})

    try:
        if should_stream(user_query, settings):
            for chunk in llm_state.create_chat_completion(
                messages=messages,
                max_tokens=temporary.BATCH_SIZE,
                temperature=float(settings.get("temperature", temporary.TEMPERATURE)),
                repeat_penalty=float(settings.get("repeat_penalty", temporary.REPEAT_PENALTY)),
                stream=True
            ):
                if cancel_event and cancel_event.is_set():
                    yield "<CANCELLED>"
                    return
                if chunk.get('choices'):
                    content = chunk['choices'][0].get('delta', {}).get('content', '')
                    if content:
                        content = re.sub(r'^AI-Chat:[\s\n]*', '', content, flags=re.IGNORECASE)
                        content = re.sub(r'\n{2,}', '\n', content)
                        yield content
        else:
            response = llm_state.create_chat_completion(
                messages=messages,
                max_tokens=temporary.BATCH_SIZE,
                stream=False
            )
            content = response['choices'][0]['message']['content']
            content = re.sub(r'^AI-Chat:[\s\n]*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'\n{2,}', '\n', content)
            yield content
    except Exception as e:
        yield f"Error generating response: {str(e)}"