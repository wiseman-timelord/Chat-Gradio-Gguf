# Script: `.\scripts\models.py`

# Imports...
import time, re, json, traceback
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

def calculate_single_model_gpu_layers_with_layers(
    model_path: str,
    available_vram: int,
    num_layers: int,
    dynamic_gpu_layers: bool = True
) -> int:
    """Compute how many layers fit on GPU with safety checks."""
    from math import floor

    if temporary.CPU_ONLY_MODE:          # NEW – pure CPU
        return 0

    if num_layers <= 0:
        print(f"[GPU-LAYERS] Invalid layer count {num_layers}, using fallback 32")
        num_layers = 32
    if available_vram <= 0:
        print(f"[GPU-LAYERS] Invalid VRAM {available_vram} MB")
        return 0

    model_mb = get_model_size(model_path)
    meta = get_model_metadata(model_path)
    arch = meta.get("general.architecture", "unknown")
    if arch in ["qwen2", "qwen2.5", "qwen"]:
        factor = 1.15
    elif arch == "llama":
        factor = 1.2
    else:
        factor = 1.25

    adjusted_mb = model_mb * factor
    layer_mb = adjusted_mb / num_layers
    max_layers = floor(available_vram / layer_mb)

    if not dynamic_gpu_layers:
        gpu_layers = num_layers
        print(f"[GPU-LAYERS] Dynamic off-load disabled → {gpu_layers} layers")
    else:
        gpu_layers = min(max_layers, num_layers)
        if temporary.VULKAN_AVAILABLE:
            gpu_layers = max(1, gpu_layers - 2)
        cpu_fallback = num_layers - gpu_layers
        print(f"[GPU-LAYERS] Model {adjusted_mb:.0f} MB, layer {layer_mb:.1f} MB")
        print(f"[GPU-LAYERS] VRAM {available_vram} MB → GPU {gpu_layers}/{num_layers} (CPU {cpu_fallback})")

    return gpu_layers

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
    """Extract metadata from a GGUF model file with multiple fallback methods."""
    from pathlib import Path
    
    # Method 1: Try with llama-cpp-python
    try:
        from llama_cpp import Llama
        print(f"[META] Opening {Path(model_path).name} for metadata…")
        
        # Try with minimal context
        try:
            llm = Llama(model_path=model_path, n_ctx=0, verbose=False, n_gpu_layers=0)
            meta = llm.metadata
            llm.close()
            print(f"[META] Keys found: {list(meta.keys())}")
            return meta
        except Exception as e1:
            print(f"[META] First attempt failed: {e1}")
            
            # Try with small context and no GPU
            try:
                llm = Llama(model_path=model_path, n_ctx=512, verbose=False, n_gpu_layers=0, n_threads=1)
                meta = llm.metadata
                llm.close()
                print(f"[META] Keys found (fallback): {list(meta.keys())}")
                return meta
            except Exception as e2:
                print(f"[META] Second attempt failed: {e2}")
    except ImportError:
        print("[META] llama-cpp-python not available")
    
    # Method 2: Parse GGUF header directly (basic implementation)
    try:
        import struct
        
        print(f"[META] Attempting direct GGUF header parsing...")
        with open(model_path, 'rb') as f:
            # Read GGUF magic and version
            magic = f.read(4)
            if magic != b'GGUF':
                print("[META] Not a valid GGUF file")
                return {}
            
            version = struct.unpack('<I', f.read(4))[0]
            print(f"[META] GGUF version: {version}")
            
            # For Qwen 2.5 models, we can make educated guesses
            model_name = Path(model_path).name.lower()
            
            # Detect architecture from filename
            if 'qwen2.5' in model_name or 'qwen-2.5' in model_name:
                arch = 'qwen2'  # Use qwen2 as base
                if '14b' in model_name:
                    layers = 40  # Qwen2.5-14B typically has 40 layers
                elif '7b' in model_name:
                    layers = 32  # Qwen2.5-7B typically has 32 layers
                elif '32b' in model_name:
                    layers = 64  # Qwen2.5-32B typically has 64 layers
                elif '72b' in model_name:
                    layers = 80  # Qwen2.5-72B typically has 80 layers
                else:
                    layers = 40  # Default for unknown Qwen2.5 variants
                    
                return {
                    'general.architecture': arch,
                    f'{arch}.block_count': layers,
                    f'{arch}.context_length': 131072,  # Qwen2.5 default
                    'general.name': 'qwen2.5',
                    '_fallback': True  # Mark as fallback data
                }
            elif 'qwen' in model_name:
                arch = 'qwen2'
                layers = 32  # Default
                return {
                    'general.architecture': arch,
                    f'{arch}.block_count': layers,
                    '_fallback': True
                }
            
    except Exception as e:
        print(f"[META] Direct parsing failed: {e}")
    
    # Method 3: Return minimal defaults based on filename patterns
    print("[META] Using filename-based defaults")
    model_name = Path(model_path).name.lower()
    
    # Common patterns
    if 'llama' in model_name:
        return {'general.architecture': 'llama', 'llama.block_count': 32}
    elif 'mistral' in model_name:
        return {'general.architecture': 'llama', 'llama.block_count': 32}
    elif 'qwen' in model_name:
        return {'general.architecture': 'qwen2', 'qwen2.block_count': 40}
    
    return {}

def get_model_layers(model_path: str) -> int:
    """Return layer count for the model with enhanced fallbacks."""
    from pathlib import Path
    
    meta = get_model_metadata(model_path)
    
    # Check if we used fallback data
    if meta.get('_fallback'):
        print(f"[LAYERS] Using fallback layer count")
    
    arch = meta.get("general.architecture", "unknown")
    
    # Extended list of keys to try
    keys_to_try = [
        f"{arch}.block_count",
        "llama.block_count",
        "qwen2.block_count",
        "qwen.block_count",
        "layers",
        "n_layers",
        "num_hidden_layers",
    ]
    
    for k in keys_to_try:
        if k in meta:
            try:
                layers = int(meta[k])
                print(f"[LAYERS] Using key '{k}' → {layers}")
                return layers
            except (ValueError, TypeError):
                print(f"[LAYERS] Bad value for key '{k}': {meta[k]}")
    
    # Final fallback based on model size (filename heuristics)
    model_name = Path(model_path).name.lower()
    print(f"[LAYERS] Using heuristic fallback for {model_name}")
    
    # Common model size to layer mappings
    size_patterns = {
        '3b': 26, '7b': 32, '8b': 32, '13b': 40, '14b': 40,
        '20b': 48, '30b': 60, '32b': 64, '34b': 60, '40b': 60,
        '70b': 80, '72b': 80, '120b': 120, '180b': 180
    }
    
    for pattern, layer_count in size_patterns.items():
        if pattern in model_name:
            print(f"[LAYERS] Heuristic match '{pattern}' → {layer_count} layers")
            return layer_count
    
    # Absolute fallback
    print("[LAYERS] No pattern matched, using default 32 layers")
    return 32  # Better than 0 for attempting load


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
    """Compute how many layers fit on GPU with safety checks."""
    from math import floor
    import scripts.temporary as tmp

    # ===== CPU-Only fast-path =====
    if tmp.BACKEND_TYPE == "CPU-Only":
        print("[GPU-LAYERS] CPU-Only mode – forcing 0 GPU layers")
        return 0
    # ==============================

    # Safety check for invalid layer count
    if num_layers <= 0:
        print(f"[GPU-LAYERS] Invalid layer count {num_layers}, using fallback 32")
        num_layers = 32  # Use reasonable default instead of failing
    
    if available_vram <= 0:
        print(f"[GPU-LAYERS] Invalid VRAM {available_vram} MB")
        return 0
    
    model_mb = get_model_size(model_path)
    meta = get_model_metadata(model_path)
    
    # Adjust factor based on architecture
    arch = meta.get("general.architecture", "unknown")
    if arch in ["qwen2", "qwen2.5", "qwen"]:
        factor = 1.15  # Qwen models typically need less overhead
    elif arch == "llama":
        factor = 1.2
    else:
        factor = 1.25  # Conservative for unknown architectures
    
    adjusted_mb = model_mb * factor
    layer_mb = adjusted_mb / num_layers
    max_layers = floor(available_vram / layer_mb)
    
    if not dynamic_gpu_layers:
        gpu_layers = num_layers
        print(f"[GPU-LAYERS] Dynamic off-load disabled → {gpu_layers} layers")
    else:
        gpu_layers = min(max_layers, num_layers)
        if "vulkan" in str(tmp.BACKEND_TYPE).lower():
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
    from scripts import temporary
    from scripts.utility import clean_content

    if not models_loaded_state or llm_state is None:
        yield "Error: No model loaded. Please load a model first."
        return

    system_message = get_system_message(
        is_uncensored=settings.get("is_uncensored", False),
        is_nsfw=settings.get("is_nsfw", False),
        web_search_enabled=web_search_enabled,
        is_reasoning=settings.get("is_reasoning", False),
        is_roleplay=settings.get("is_roleplay", False)
    ) + "\nRespond directly without prefixes like 'AI-Chat:'."

    if web_search_enabled and search_results:
        system_message += f"\n\nWeb search results:\n{search_results}"

    # Build message list
    messages = [{"role": "system", "content": system_message}]
    for msg in reversed(session_log[:-2]):
        messages.insert(1, {"role": msg["role"], "content": clean_content(msg["role"], msg["content"])})
    messages.append({"role": "user", "content": clean_content("user", session_log[-2]["content"])})

    yield "AI-Chat:\n"

    # State tracking for <think> phase
    in_think_block = False
    buffer = ""  # Accumulate tokens to detect tags
    seen_real_text = False
    thinking_started = False
    raw_output = ""  # Track complete raw output for printing
    
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

        token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if not token:
            continue

        # Drop pure-whitespace tokens until real text arrives
        if not seen_real_text:
            if token.strip() == "":
                continue
            seen_real_text = True

        # [INST] guard: stop if model tries to speak as user
        if token.strip().startswith("[INST]"):
            return

        # Accumulate tokens in buffer for tag detection
        buffer += token
        
        # Check for <think> opening tag
        if not in_think_block and "<think>" in buffer:
            in_think_block = True
            thinking_started = True
            
            # Split buffer at <think>
            before_think = buffer.split("<think>")[0]
            after_think = buffer.split("<think>", 1)[1]
            
            # Yield content before <think> if any
            if before_think:
                yield before_think
            
            # Start "Thinking" line with dots for existing content
            buffer = after_think
            space_count = buffer.count(" ")
            yield "Thinking" + ("." * space_count)
            continue
        
        # If inside <think> block
        if in_think_block:
            # Check for </think> closing tag
            if "</think>" in buffer:
                # Extract content after </think>
                after_close = buffer.split("</think>", 1)[1]
                
                # End thinking phase
                in_think_block = False
                buffer = after_close
                
                # Yield newline to separate thinking from answer
                yield "\n"
                
                # Yield content after </think>
                if after_close:
                    yield after_close
                    buffer = ""  # Clear buffer after yielding
            else:
                # Count spaces in new token and yield dots
                space_count = token.count(" ")
                if space_count > 0:
                    yield "." * space_count
        else:
            # Normal streaming: yield the token and clear buffer
            if not thinking_started or buffer == token:
                # Direct yield if no thinking or buffer is just current token
                yield token
                buffer = ""
            else:
                # We have accumulated buffer after </think>, yield it
                yield buffer
                buffer = ""

    # Yield any remaining buffer content
    if buffer and not in_think_block:
        yield buffer

    if temporary.PRINT_RAW_OUTPUT:
        print("\n***RAW_OUTPUT_FROM_MODEL_START***")
        print("(streaming output shown above)", flush=True)
        print("***RAW_OUTPUT_FROM_MODEL_END***\n")

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