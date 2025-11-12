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
    MODEL_NAME, REPEAT_PENALTY, TEMPERATURE, MODELS_LOADED, CHAT_FORMAT_MAP,
    THINK_MIN_CHARS_BEFORE_CLOSE  # NEW
)


# Functions...
def get_chat_format(metadata):
    """
    Determine the chat format based on the model's architecture.
    """
    architecture = metadata.get('general.architecture', 'unknown')
    fmt = CHAT_FORMAT_MAP.get(architecture, 'llama-2')
    # quick fix for the stale import problem
    if fmt == 'llama2':
        fmt = 'llama-2'
    return fmt

def get_model_size(model_path: str) -> float:
    return Path(model_path).stat().st_size / (1024 * 1024)

def clean_content(role, content):
    """Remove prefixes from session_log content for model input."""
    if role == 'user':
        # Handle both "User:\n" prefix and structured "User Query:\n" format
        content = content.replace("User:\n", "", 1)
        # Don't strip "User Query:" header as it's part of structured context
        return content.strip()
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
    is_harmony = any(keyword in model_name_lower for keyword in handling_keywords["harmony"])
    return {
        "category": "chat",
        "is_uncensored": is_uncensored,
        "is_reasoning": is_reasoning,
        "is_nsfw": is_nsfw,
        "is_code": is_code,
        "is_roleplay": is_roleplay,
        "is_harmony": is_harmony,
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
    """
    Extract GGUF metadata without creating a llama_context.
    Works with Gemma and other models that require large n_ctx_train.
    Always returns a usable dict.
    """
    from pathlib import Path
    import struct

    path = Path(model_path)

    # ----------------------------------------------------------
    # 1.  Zero-cost route:  llama-cpp-python  ≥ 0.2.0
    # ----------------------------------------------------------
    try:
        from llama_cpp import LlamaMetadata
        meta = LlamaMetadata(model_path=str(path))
        print(f"[META] LlamaMetadata used – keys: {list(meta.keys())}")
        return meta
    except Exception as e:
        # LlamaMetadata not available or file unreadable → continue
        print(f"[META] LlamaMetadata failed: {e}")

    # ----------------------------------------------------------
    # 2.  Direct header parse (same as before)
    # ----------------------------------------------------------
    try:
        with path.open("rb") as f:
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError("Invalid GGUF magic")
            version = struct.unpack('<I', f.read(4))[0]
            print(f"[META] GGUF version: {version}")

        name_lower = path.name.lower()
        if 'qwen2.5' in name_lower or 'qwen-2.5' in name_lower:
            arch, layers = 'qwen2', 40
            if '14b' in name_lower: layers = 40
            elif '7b' in name_lower: layers = 32
            elif '32b' in name_lower: layers = 64
            elif '72b' in name_lower: layers = 80
            return {
                'general.architecture': arch,
                f'{arch}.block_count': layers,
                f'{arch}.context_length': 131072,
                'general.name': 'qwen2.5',
                '_fallback': True
            }
        elif 'qwen' in name_lower:
            return {'general.architecture': 'qwen2', 'qwen2.block_count': 32, '_fallback': True}
    except Exception as e:
        print(f"[META] Direct parsing failed: {e}")

    # ----------------------------------------------------------
    # 3.  Filename-only fallback
    # ----------------------------------------------------------
    print("[META] Using filename-based defaults")
    name_lower = path.name.lower()
    if 'llama' in name_lower:
        return {'general.architecture': 'llama', 'llama.block_count': 32}
    elif 'mistral' in name_lower:
        return {'general.architecture': 'llama', 'llama.block_count': 32}
    elif 'qwen' in name_lower:
        return {'general.architecture': 'qwen2', 'qwen2.block_count': 40}

    # Ultimate safety
    return {
        'general.architecture': 'unknown',
        'general.name': path.stem,
        '_fallback': True,
        '_error': 'All metadata extraction methods failed'
    }

def get_model_layers(model_path: str) -> int:
    """
    Return layer count for the model with enhanced fallbacks.
    Always returns a positive integer.
    """
    from pathlib import Path

    meta = get_model_metadata(model_path)
    if meta.get('_fallback'):
        print(f"[LAYERS] Using fallback layer count for {Path(model_path).name}")

    arch = meta.get("general.architecture", "unknown")

    keys = (
        f"{arch}.block_count",
        "llama.block_count",
        "qwen2.block_count",
        "qwen.block_count",
        "layers",
        "n_layers",
        "num_hidden_layers",
    )
    for k in keys:
        if k in meta:
            try:
                layers = int(meta[k])
                if layers > 0:
                    print(f"[LAYERS] Using key '{k}' → {layers}")
                    return layers
            except (ValueError, TypeError):
                print(f"[LAYERS] Bad value for key '{k}': {meta[k]}")

    # heuristic fallback from filename
    name_lower = Path(model_path).name.lower()
    size_map = {
        '3b': 26, '7b': 32, '8b': 32, '13b': 40, '14b': 40,
        '20b': 48, '30b': 60, '32b': 64, '34b': 60, '40b': 60,
        '70b': 80, '72b': 80, '120b': 120, '180b': 180
    }
    for pattern, count in size_map.items():
        if pattern in name_lower:
            print(f"[LAYERS] Heuristic match '{pattern}' → {count}")
            return count

    print("[LAYERS] No pattern matched, using default 32")
    return 32

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
    """
    Load a GGUF model with llama-cpp-python.
    Returns: (status: str, success: bool, llm_obj, models_loaded_flag)
    """
    from scripts.temporary import (
        CONTEXT_SIZE, BATCH_SIZE, MMAP, MLOCK, DYNAMIC_GPU_LAYERS,
        BACKEND_TYPE, CPU_THREADS, set_status   # <-- CPU_ONLY_MODE removed
    )
    from scripts.settings import save_config
    from pathlib import Path
    import traceback

    save_config()  # persist any pending config changes

    if model in {"Select_a_model...", "No models found"}:
        return "Select a model to load.", False, llm_state, models_loaded_state

    model_path = Path(model_folder) / model
    if not model_path.exists():
        model_path = Path(model_folder) / model.replace('\\', '/')
        if not model_path.exists():
            return f"Error: Model file '{model_path}' not found.", False, llm_state, models_loaded_state

    metadata = get_model_metadata(str(model_path))
    chat_format = get_chat_format(metadata)

    num_layers = get_model_layers(str(model_path))
    if num_layers <= 0:
        return f"Error: Could not determine layer count for model '{model}'.", False, llm_state, models_loaded_state

    gpu_layers = calculate_single_model_gpu_layers_with_layers(
        str(model_path), vram_size, num_layers, DYNAMIC_GPU_LAYERS
    )

    # CPU-thread logic
    use_cpu_threads = True
    if "vulkan" in BACKEND_TYPE.lower():
        use_cpu_threads = True
        if CPU_THREADS is None or CPU_THREADS < 2:
            CPU_THREADS = 2
    elif gpu_layers >= num_layers:   # full GPU offload
        use_cpu_threads = False
    else:
        use_cpu_threads = True

    try:
        from llama_cpp import Llama
    except ImportError:
        return "Error: llama-cpp-python not installed. Python bindings are required.", False, llm_state, models_loaded_state

    if models_loaded_state:
        unload_models(llm_state, models_loaded_state)

    set_status(f"Load {gpu_layers}/{num_layers}", console=True)

    kwargs = dict(
        model_path=str(model_path),
        n_ctx=CONTEXT_SIZE,
        n_ctx_per_seq=CONTEXT_SIZE,
        n_gpu_layers=gpu_layers,
        n_batch=BATCH_SIZE,
        mmap=MMAP,
        mlock=MLOCK,
        verbose=True,
        chat_format=chat_format
    )
    if use_cpu_threads and CPU_THREADS is not None:
        kwargs["n_threads"] = CPU_THREADS

    try:
        new_llm = Llama(**kwargs)
        # smoke test
        new_llm.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=BATCH_SIZE,
            stream=False
        )

        # commit state only on success
        import scripts.temporary as tmp
        tmp.GPU_LAYERS = gpu_layers
        tmp.MODEL_NAME = model
        set_status("Model ready", console=True)
        return "Model ready", True, new_llm, True

    except Exception as e:
        # rollback on failure
        import scripts.temporary as tmp
        tmp.GPU_LAYERS = 0
        tb = traceback.format_exc()
        err = (f"Error loading model: {e}\n"
               f"GPU Layers: {gpu_layers}/{num_layers}\n"
               f"CPU Threads: {CPU_THREADS if use_cpu_threads else 'none'}\n"
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

def update_thinking_phase_constants():
    """
    Call this during initialization to set up thinking phase detection.
    This demonstrates the pattern but you may not need all these tags.
    """
    import scripts.temporary as temporary
    
    # Standard thinking tags (like Qwen3)
    standard_opening = ["<think>"]
    standard_closing = ["</think>"]
    
    # gpt-oss Harmony format tags
    # Opening: assistant starts analysis channel
    gpt_oss_opening = [
        "<|start|>assistant<|channel|>analysis<|message|>",
        "<|channel|>analysis",  # Simpler pattern
    ]
    
    # Closing: various patterns that end thinking
    gpt_oss_closing = [
        "<|end|><|start|>assistant<|channel|>final<|message|>",
        "<|end|><|start|>assistant<|end|><|start|>assistant<|message|>",  # Your example
    ]
    
    # Partial patterns to watch for (simplified detection)
    gpt_oss_partial = [
        "<|end|>",  # Generic end marker that starts close sequence
        "<|channel|>final",  # Switch to final channel
    ]
    
    # Update temporary module (if using the list-based approach)
    # Note: The new implementation doesn't rely heavily on these lists,
    # but keeping them for compatibility
    temporary.THINK_OPENING_TAGS = standard_opening + gpt_oss_opening
    temporary.THINK_CLOSING_TAGS = standard_closing + gpt_oss_closing
    temporary.THINK_CLOSING_PARTIAL_PATTERNS = gpt_oss_partial
    
    print(f"[THINKING] Configured for {len(temporary.THINK_OPENING_TAGS)} opening patterns")
    print(f"[THINKING] Configured for {len(temporary.THINK_CLOSING_TAGS)} closing patterns")

def get_response_stream(session_log, settings, web_search_enabled=False, search_results=None,
                        cancel_event=None, llm_state=None, models_loaded_state=False):
    """
    Enhanced streaming handler for both standard and gpt-oss models.
    
    Handles:
    - Standard <think></think> tags (Qwen, DeepSeek)
    - gpt-oss proper: <|channel|>analysis ... <|channel|>final
    - gpt-oss malformed (HuiHui): <<USR>> ... >> OR ```js ... ```
    """
    import re
    from scripts import temporary
    from scripts.utility import clean_content

    if not models_loaded_state or llm_state is None:
        yield "Error: No model loaded. Please load a model first."
        return

    # Build system message
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

    # State tracking
    in_thinking_phase = False
    output_buffer = ""
    seen_real_text = False
    char_count_since_think_start = 0
    raw_output = "AI-Chat:\n"
    
    # Track if we've seen code fences (for HuiHui malformed detection)
    code_fence_count = 0
    found_final_answer = False

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
        
        raw_output += token
        output_buffer += token

        # Skip initial whitespace
        if not seen_real_text:
            if token.strip() == "":
                continue
            seen_real_text = True

        # === HALLUCINATION GUARDS ===
        # HuiHui model sometimes outputs code examples with repeating patterns
        # Stop if we detect infinite loops or user role markers
        if "[INST]" in output_buffer or "<<USER>>" in output_buffer:
            if temporary.PRINT_RAW_OUTPUT:
                print("\n***RAW_OUTPUT_FROM_MODEL_START***")
                print(raw_output, flush=True)
                print("***RAW_OUTPUT_FROM_MODEL_END***\n")
            return
        
        # === THINKING PHASE DETECTION ===
        
        # Standard <think> opening
        if not in_thinking_phase and "<think>" in output_buffer:
            in_thinking_phase = True
            char_count_since_think_start = 0
            
            before_think = output_buffer.split("<think>", 1)[0]
            if before_think.strip():
                yield before_think
            
            output_buffer = output_buffer.split("<think>", 1)[1]
            
            if temporary.SHOW_THINK_PHASE:
                yield "<think>"
            else:
                yield "Thinking"
            continue

        # gpt-oss proper format: <|channel|>analysis
        if not in_thinking_phase and "<|channel|>analysis" in output_buffer:
            in_thinking_phase = True
            char_count_since_think_start = 0
            
            parts = output_buffer.split("<|channel|>analysis", 1)
            before_analysis = parts[0]
            
            # Clean Harmony tags
            before_analysis = re.sub(r'<\|start\|>assistant', '', before_analysis)
            before_analysis = re.sub(r'<\|message\|>', '', before_analysis)
            
            if before_analysis.strip():
                yield before_analysis
            
            output_buffer = parts[1] if len(parts) > 1 else ""
            
            if temporary.SHOW_THINK_PHASE:
                yield "<|channel|>analysis"
            else:
                yield "Thinking"
            continue

        # HuiHui malformed format: <<USR>> marker
        if not in_thinking_phase and "<<USR>>" in output_buffer:
            in_thinking_phase = True
            char_count_since_think_start = 0
            code_fence_count = 0
            
            parts = output_buffer.split("<<USR>>", 1)
            before_think = parts[0]
            
            if before_think.strip():
                yield before_think
            
            output_buffer = parts[1] if len(parts) > 1 else ""
            
            if temporary.SHOW_THINK_PHASE:
                yield "<<USR>>"
            else:
                yield "Thinking"
            continue

        # === INSIDE THINKING PHASE ===
        if in_thinking_phase:
            char_count_since_think_start += len(token)
            
            # Standard </think> closing
            if "</think>" in output_buffer:
                in_thinking_phase = False
                
                parts = output_buffer.split("</think>", 1)
                think_content = parts[0]
                after_think = parts[1] if len(parts) > 1 else ""
                
                if temporary.SHOW_THINK_PHASE:
                    yield think_content
                    yield "</think>\n"
                else:
                    spaces = think_content.count(" ")
                    if spaces > 0:
                        yield "." * min(spaces, 50)
                    yield "\n"
                
                output_buffer = after_think
                if after_think.strip():
                    yield after_think
                    output_buffer = ""
                continue
            
            # gpt-oss proper: <|channel|>final
            if char_count_since_think_start >= temporary.THINK_MIN_CHARS_BEFORE_CLOSE:
                end_pattern = r'<\|end\|>.*?<\|channel\|>final'
                match = re.search(end_pattern, output_buffer, re.DOTALL)
                
                if not match:
                    alt_pattern = r'<\|end\|><\|start\|>assistant<\|end\|><\|start\|>assistant<\|message\|>'
                    match = re.search(alt_pattern, output_buffer, re.DOTALL)
                
                if match:
                    in_thinking_phase = False
                    
                    think_end_pos = match.start()
                    think_content = output_buffer[:think_end_pos]
                    after_think = output_buffer[match.end():]
                    
                    if temporary.SHOW_THINK_PHASE:
                        yield think_content
                        yield match.group(0)
                        yield "\n"
                    else:
                        spaces = think_content.count(" ")
                        if spaces > 0:
                            yield "." * min(spaces, 50)
                        yield "\n"
                    
                    output_buffer = after_think
                    if after_think.strip():
                        yield after_think
                        output_buffer = ""
                    continue
            
            # HuiHui malformed: Look for >> OR first ```js after thinking
            if char_count_since_think_start >= temporary.THINK_MIN_CHARS_BEFORE_CLOSE:
                
                # Pattern 1: <|end|><|start|>assistant>>
                if "<|start|>assistant>>" in output_buffer:
                    in_thinking_phase = False
                    found_final_answer = True
                    
                    parts = output_buffer.split("<|start|>assistant>>", 1)
                    think_content = parts[0]
                    after_think = parts[1] if len(parts) > 1 else ""
                    
                    # Clean <<USR>> tag from thinking if present
                    think_content = think_content.replace("<<USR>>", "")
                    
                    if temporary.SHOW_THINK_PHASE:
                        yield think_content
                        yield "\n"
                    else:
                        spaces = think_content.count(" ")
                        if spaces > 0:
                            yield "." * min(spaces, 50)
                        yield "\n"
                    
                    output_buffer = after_think
                    continue
                
                # Pattern 2: Track code fences (```js signals end of thinking in HuiHui)
                if "```" in token:
                    code_fence_count += 1
                
                # After seeing first code fence and 100+ chars, look for second fence
                # OR look for import/console.log patterns that signal hallucination
                if code_fence_count >= 1 and char_count_since_think_start >= 150:
                    # Check if we're seeing hallucinated code (like import statements)
                    if "import" in output_buffer or "console.log" in output_buffer or "openai.chat.completions" in output_buffer:
                        # This is hallucinated code, extract actual answer before it
                        
                        # Find the first ``` after initial thinking
                        first_fence = output_buffer.find("```")
                        if first_fence != -1:
                            in_thinking_phase = False
                            found_final_answer = True
                            
                            think_content = output_buffer[:first_fence]
                            
                            # The actual answer might be BEFORE the thinking in HuiHui's case
                            # Look for patterns like ">>" followed by actual content
                            if ">>" in think_content:
                                parts = think_content.split(">>", 1)
                                actual_think = parts[0]
                                actual_answer = parts[1]
                                
                                if temporary.SHOW_THINK_PHASE:
                                    yield actual_think
                                    yield "\n"
                                else:
                                    spaces = actual_think.count(" ")
                                    if spaces > 0:
                                        yield "." * min(spaces, 50)
                                    yield "\n"
                                
                                # Yield answer and stop (don't continue with hallucinated code)
                                if actual_answer.strip():
                                    yield actual_answer.strip()
                                
                                # Stop generation here
                                if temporary.PRINT_RAW_OUTPUT:
                                    print("\n***RAW_OUTPUT_FROM_MODEL_START***")
                                    print(raw_output, flush=True)
                                    print("***RAW_OUTPUT_FROM_MODEL_END***\n")
                                return
            
            # Still thinking - display according to mode
            if temporary.SHOW_THINK_PHASE:
                yield token
                output_buffer = ""
            else:
                spaces = token.count(" ")
                if spaces > 0:
                    yield "." * min(spaces, 10)
                # Keep last 100 chars for pattern detection
                if len(output_buffer) > 200:
                    output_buffer = output_buffer[-100:]
            continue

        # === NORMAL OUTPUT (not thinking) ===
        
        # If we've found the final answer and see repeating patterns, stop
        if found_final_answer:
            # Check for infinite loops (same content repeating)
            if len(output_buffer) > 500:
                # Look for repeating patterns of 100+ characters
                mid_point = len(output_buffer) // 2
                first_half = output_buffer[:mid_point]
                second_half = output_buffer[mid_point:mid_point*2]
                
                # If content is repeating, stop generation
                if first_half == second_half:
                    if temporary.PRINT_RAW_OUTPUT:
                        print("\n***RAW_OUTPUT_FROM_MODEL_START***")
                        print(raw_output, flush=True)
                        print("***RAW_OUTPUT_FROM_MODEL_END***\n")
                    return
        
        # Yield buffer periodically
        if len(output_buffer) > 20:
            yield output_buffer
            output_buffer = ""

    # Final flush
    if output_buffer and not in_thinking_phase:
        yield output_buffer

    # Debug output
    if temporary.PRINT_RAW_OUTPUT:
        print("\n***RAW_OUTPUT_FROM_MODEL_START***")
        print(raw_output, flush=True)
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