# Script: `.\scripts\models.py`

# Imports...
import time, re, json
from pathlib import Path
import gradio as gr
from scripts.prompts import get_system_message, get_reasoning_instruction
import scripts.temporary as temporary
from scripts.prompts import prompt_templates
from scripts.temporary import (
    CONTEXT_SIZE, GPU_LAYERS, BATCH_SIZE, BACKEND_TYPE, VRAM_SIZE,
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

def get_model_layers(model_path: str) -> int:
    """Return layer count for the model (GGUF)."""
    meta = get_model_metadata(model_path)
    arch = meta.get("general.architecture", "unknown")
    keys_to_try = (
        f"{arch}.block_count",
        "llama.block_count",
        "layers"            # very old fallback
    )

    for k in keys_to_try:
        if k in meta:
            try:
                layers = int(meta[k])
                print(f"[LAYERS] Using key '{k}' → {layers}")
                return layers
            except (ValueError, TypeError):
                print(f"[LAYERS] Bad value for key '{k}': {meta[k]}")
    print(f"[LAYERS] No valid key found in {Path(model_path).name}")
    return 0

def get_model_size(model_path: str) -> float:
    return Path(model_path).stat().st_size / (1024 * 1024)

def clean_content(role, content):
    """Remove prefixes from session_log content for model input."""
    if role == 'user':
        return content.replace("User:\n", "", 1).strip()
    return content.strip()

def get_available_models():
    from scripts.utility import short_path    
    model_dir = Path(temporary.MODEL_FOLDER)
    print(f"Finding Models: {short_path(model_dir)}")
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

# Get Model MetaData
def get_model_metadata(model_path: str) -> dict:
    """Extract metadata from a GGUF model file – read-only, no context."""
    try:
        from llama_cpp import Llama
        print(f"[META] Opening {Path(model_path).name} for metadata…")
        llm = Llama(model_path=model_path, n_ctx=0, verbose=False)
        meta = llm.metadata
        llm.close()
        print(f"[META] Keys found: {list(meta.keys())}")
        return meta
    except Exception as e:
        print(f"[META] Error reading metadata: {e}")
        return {}

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
        return f"{params_str}|{fit_layers}/{layers}|{model_size_gb:.1f}GB|{max_ctx}"
    except Exception as e:
        return f"Error inspecting model: {str(e)}"

def load_models(model_folder, model, vram_size, llm_state, models_loaded_state):
    from scripts.temporary import (
        CONTEXT_SIZE, BATCH_SIZE, MMAP, MLOCK, DYNAMIC_GPU_LAYERS,
        BACKEND_TYPE, CPU_THREADS
    )
    from scripts.settings import save_config
    from pathlib import Path
    import traceback

    save_config()

    if model in ["Select_a_model...", "No models found"]:
        return "Select a model to load.", False, llm_state, models_loaded_state

    model_path = Path(model_folder) / model
    if not model_path.exists():
        # Handle Linux paths
        model_path = Path(model_folder) / model.replace('\\', '/')
        if not model_path.exists():
            return f"Error: Model file '{model_path}' not found.", False, llm_state, models_loaded_state

    metadata = get_model_metadata(str(model_path))
    chat_format = get_chat_format(metadata)

    num_layers = get_model_layers(str(model_path))
    if num_layers <= 0:
        return f"Error: Could not determine layer count for model '{model}'.", False, llm_state, models_loaded_state

    # Calculate GPU layers and determine CPU-thread usage
    gpu_layers = calculate_single_model_gpu_layers_with_layers(
        str(model_path), vram_size, num_layers, DYNAMIC_GPU_LAYERS
    )
    temporary.GPU_LAYERS = gpu_layers

    # NEW: Always enable CPU threads for Vulkan; skip only on full-GPU non-Vulkan
    use_cpu_threads = True
    if "vulkan" in BACKEND_TYPE.lower():
        use_cpu_threads = True
        if CPU_THREADS is None or CPU_THREADS < 2:
            temporary.CPU_THREADS = 2  # min for Vulkan
    elif gpu_layers >= num_layers:
        use_cpu_threads = False  # full GPU offload on non-Vulkan
    else:
        use_cpu_threads = True

    try:
        from llama_cpp import Llama
    except ImportError:
        return "Error: llama-cpp-python not installed. Python bindings are required.", False, llm_state, models_loaded_state

    try:
        if models_loaded_state:
            unload_models(llm_state, models_loaded_state)

        temporary.set_status(f"Load {gpu_layers}/{num_layers}", console=True)

        kwargs = {
            "model_path": str(model_path),
            "n_ctx": CONTEXT_SIZE,
            "n_ctx_per_seq": CONTEXT_SIZE,      # ← NEW
            "n_gpu_layers": gpu_layers,
            "n_batch": BATCH_SIZE,
            "mmap": MMAP,
            "mlock": MLOCK,
            "verbose": True,
            "chat_format": chat_format
        }

        if use_cpu_threads and CPU_THREADS is not None:
            kwargs["n_threads"] = CPU_THREADS

        new_llm = Llama(**kwargs)

        # Quick smoke test
        test_out = new_llm.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=BATCH_SIZE,
            stream=False
        )
        temporary.set_status("Model ready", console=True)
        return "Model ready", True, new_llm, True

    except Exception as e:
        tb = traceback.format_exc()
        err = (f"Error loading model: {e}\n"
               f"GPU Layers: {gpu_layers}/{num_layers}\n"
               f"CPU Threads: {temporary.CPU_THREADS if use_cpu_threads else 'none'}\n"
               f"{tb}")
        print(err)
        return err, False, None, False

def calculate_single_model_gpu_layers_with_layers(
    model_path: str,
    available_vram: int,
    num_layers: int,
    dynamic_gpu_layers: bool = True
) -> int:
    """Compute how many layers fit on GPU (b5587)."""
    from math import floor

    if num_layers <= 0 or available_vram <= 0:
        print(f"[GPU-LAYERS] Invalid input: layers={num_layers}, VRAM={available_vram} MB")
        return 0

    model_mb = get_model_size(model_path)
    meta = get_model_metadata(model_path)
    factor = 1.2 if meta.get("general.architecture") == "llama" else 1.1
    adjusted_mb = model_mb * factor
    layer_mb = adjusted_mb / num_layers
    max_layers = floor(available_vram / layer_mb)

    if not dynamic_gpu_layers:
        gpu_layers = num_layers
        print(f"[GPU-LAYERS] Dynamic off-load disabled → {gpu_layers} layers")
    else:
        gpu_layers = min(max_layers, num_layers)
        if "vulkan" in str(temporary.BACKEND_TYPE).lower():
            gpu_layers = max(1, gpu_layers - 2)
        cpu_fallback = num_layers - gpu_layers
        print(f"[GPU-LAYERS] Model {adjusted_mb:.0f} MB, layer {layer_mb:.1f} MB")
        print(f"[GPU-LAYERS] VRAM {available_vram} MB → GPU {gpu_layers}/{num_layers} (CPU {cpu_fallback})")

    return gpu_layers

def unload_models(llm_state, models_loaded_state):
    import gc
    if models_loaded_state:
        del llm_state
        gc.collect()
        temporary.set_status("Unloaded", console=True)
        return "Model unloaded successfully.", None, False
    temporary.set_status("Model off", console=True)
    return "Model off", llm_state, models_loaded_state

def get_response_stream(session_log, settings, web_search_enabled=False, search_results=None,
                        cancel_event=None, llm_state=None, models_loaded_state=False):
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
        system_message += "\n\nSearch Results:\n" + re.sub(
            r'https?://(www\.)?([^/]+).*', r'\2', str(search_results)
        )

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

    # Buffer to accumulate the entire response
    full_response_parts = []

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
                        # Clean & forward to Gradio
                        content = re.sub(r'^AI-Chat:[\s\n]*', '', content, flags=re.IGNORECASE)
                        content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)
                        content = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', content)
                        content = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', content)
                        content = re.sub(r'\n{3,}', '\n\n', content)
                        full_response_parts.append(content)
                        yield content

            # All chunks received – one-shot raw print if enabled
            complete_raw = ''.join(full_response_parts)
            if temporary.PRINT_RAW_MODEL_OUTPUT and complete_raw:
                print("\n***RAW_OUTPUT_FROM_MODEL_START***")
                print(complete_raw, flush=True)
                print("***RAW_OUTPUT_FROM_MODEL_END***\n")

            # Final yield so caller can replace the bubble with fully-formatted text
            yield complete_raw

        else:
            response = llm_state.create_chat_completion(
                messages=messages,
                max_tokens=temporary.BATCH_SIZE,
                stream=False
            )
            content = response['choices'][0]['message']['content']
            content = re.sub(r'^AI-Chat:[\s\n]*', '', content, flags=re.IGNORECASE)
            content = re.sub(r'([a-z])([A-Z])', r'\1 \2', content)
            content = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', content)
            content = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', content)
            content = re.sub(r'\n{3,}', '\n\n', content)

            if temporary.PRINT_RAW_MODEL_OUTPUT and content:
                print("\n***RAW_OUTPUT_FROM_MODEL_START***")
                print(content, flush=True)
                print("***RAW_OUTPUT_FROM_MODEL_END***\n")

            yield content

    except Exception as e:
        yield f"Error generating response: {str(e)}"

# Helper function to reload model with new settings
def change_model(model_name):
    try:
        from scripts.temporary import MODEL_FOLDER, VRAM_SIZE, MODELS_LOADED, llm
        status, models_loaded, llm_state, _ = load_models(
            MODEL_FOLDER,
            model_name,
            VRAM_SIZE,
            llm,
            MODELS_LOADED
        )
        return status, models_loaded, llm_state
    except Exception as e:
        return f"Error changing model: {str(e)}", False, None