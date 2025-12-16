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
    is_moe = any(keyword in model_name_lower for keyword in handling_keywords["moe"])
    is_vision = any(keyword in model_name_lower for keyword in handling_keywords["vision"])  # NEW
    
    return {
        "category": "chat",
        "is_uncensored": is_uncensored,
        "is_reasoning": is_reasoning,
        "is_nsfw": is_nsfw,
        "is_code": is_code,
        "is_roleplay": is_roleplay,
        "is_moe": is_moe,
        "is_vision": is_vision,  # NEW
        "detected_keywords": [kw for kw in handling_keywords if any(k in model_name_lower for k in handling_keywords[kw])]
    }

def find_mmproj_file(model_path):
    """
    Find corresponding mmproj file for vision models using flexible search.
    Searches for ANY .gguf file containing 'mmproj' in the model directory.
    Returns: path to mmproj file or None
    """
    from pathlib import Path
    import scripts.temporary
    
    model_dir = Path(model_path).parent
    
    # Find ANY file with mmproj in the name (case-insensitive)
    mmproj_files = []
    for file in model_dir.glob("*.gguf"):
        if "mmproj" in file.name.lower():
            mmproj_files.append(file)
    
    if not mmproj_files:
        print("[VISION] No mmproj file found in directory")
        return None
    
    # Prefer F16 version if multiple found
    for mmproj in mmproj_files:
        if "f16" in mmproj.name.lower():
            print(f"[VISION] Found mmproj (F16): {mmproj.name}")
            return str(mmproj)
    
    # Otherwise use first found
    print(f"[VISION] Found mmproj: {mmproj_files[0].name}")
    return str(mmproj_files[0])

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


def get_mmproj_context_llama(mmproj_path):
    """Use llama_cpp to get mmproj context"""
    try:
        from llama_cpp import LlamaMetadata
        metadata = LlamaMetadata(model_path=str(mmproj_path))
        
        # Try various context keys used in vision models
        for key in ['clip.context_length', 'context_length', 'n_ctx']:
            if key in metadata:
                return int(metadata[key])
        
        # Fallback to common values
        return 4096
    except:
        return 4096

def load_models(model_folder, model, vram_size, llm_state, models_loaded_state):
    """
    Load a GGUF model with llama-cpp-python, including vision models.
    CRITICAL: Vision chat handlers must be created BEFORE Llama() initialization.
    FIXED: GPU selection now properly passed to Vulkan backend.
    Returns: (status: str, success: bool, llm_obj, models_loaded_flag)
    """
    from scripts.temporary import (
        CONTEXT_SIZE, BATCH_SIZE, MMAP, MLOCK, DYNAMIC_GPU_LAYERS,
        BACKEND_TYPE, CPU_THREADS, set_status
    )
    from scripts.settings import save_config
    from pathlib import Path
    import traceback
    import os

    save_config()

    if model in {"Select_a_model...", "No models found"}:
        return "Select a model to load.", False, llm_state, models_loaded_state

    model_path = Path(model_folder) / model
    if not model_path.exists():
        model_path = Path(model_folder) / model.replace('\\', '/')
        if not model_path.exists():
            return f"Error: Model file '{model_path}' not found.", False, llm_state, models_loaded_state

    # Validate model file is valid GGUF
    try:
        with open(model_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                return f"Not a valid GGUF file: {model_path}", False, llm_state, models_loaded_state
    except Exception as e:
        return f"Cannot read model file: {e}", False, llm_state, models_loaded_state

    metadata = get_model_metadata(str(model_path))
    chat_format = get_chat_format(metadata)
    
    # Check if this is a vision model
    model_settings = get_model_settings(model)
    is_vision = model_settings.get("is_vision", False)
    is_reasoning = model_settings.get("is_reasoning", False)

    num_layers = get_model_layers(str(model_path))
    if num_layers <= 0:
        return f"Error: Could not determine layer count for model '{model}'.", False, llm_state, models_loaded_state

    gpu_layers = calculate_single_model_gpu_layers_with_layers(
        str(model_path), vram_size, num_layers, DYNAMIC_GPU_LAYERS
    )

    use_cpu_threads = True
    if "vulkan" in BACKEND_TYPE.lower():
        use_cpu_threads = True
        if CPU_THREADS is None or CPU_THREADS < 2:
            CPU_THREADS = 2
    elif gpu_layers >= num_layers:
        use_cpu_threads = False
    else:
        use_cpu_threads = True

    try:
        from llama_cpp import Llama
    except ImportError:
        return "Error: llama-cpp-python not installed. Python bindings are required.", False, llm_state, models_loaded_state

    if models_loaded_state:
        unload_models(llm_state, models_loaded_state)

    if BACKEND_TYPE.lower() == "cpu-only":
        MAX_CTX = 32_768
    elif "vulkan" in BACKEND_TYPE.lower():
        MAX_CTX = 48_768 if vram_size >= 12_288 else 32_768
    else:
        MAX_CTX = CONTEXT_SIZE
    effective_ctx = min(CONTEXT_SIZE, MAX_CTX)

    set_status(f"Loading model {gpu_layers}/{num_layers} layers...", console=True, priority=True)

    # CRITICAL FIX: Set Vulkan device selection BEFORE creating Llama object
    if temporary.BACKEND_TYPE in ["VULKAN_VULKAN", "VULKAN_CPU"]:
        if temporary.SELECTED_GPU and temporary.SELECTED_GPU != "Auto-Select":
            from scripts.utility import get_available_gpus
            gpu_list = get_available_gpus()
            
            print("\n[VULKAN] GPU Selection:")
            print(f"  User selected: {temporary.SELECTED_GPU}")
            print(f"  Available GPUs:")
            for idx, gpu_name in enumerate(gpu_list):
                marker = " <- SELECTED" if gpu_name == temporary.SELECTED_GPU else ""
                print(f"    Vulkan{idx}: {gpu_name}{marker}")
            
            try:
                gpu_index = gpu_list.index(temporary.SELECTED_GPU)
                # Set environment variable for Vulkan device selection
                os.environ["GGML_VULKAN_DEVICE"] = str(gpu_index)
                print(f"[VULKAN] Set GGML_VULKAN_DEVICE={gpu_index}")
                print(f"[VULKAN] Model will use: {temporary.SELECTED_GPU}\n")
            except (ValueError, IndexError):
                print(f"[VULKAN] Warning: '{temporary.SELECTED_GPU}' not found in GPU list")
                print(f"[VULKAN] Defaulting to Vulkan0 (first device)\n")
        else:
            print(f"[VULKAN] Auto-select mode - will use Vulkan0 (first device)\n")

    # Base kwargs
    kwargs = dict(
        model_path=str(model_path),
        n_ctx=effective_ctx,
        n_ctx_per_seq=effective_ctx,
        n_batch=BATCH_SIZE,
        mmap=MMAP,
        mlock=MLOCK,
        verbose=True,
        chat_format=chat_format
    )

    # ---------- GPU-layer decision based on backend type ----------
    if temporary.BACKEND_TYPE == "VULKAN_VULKAN":
        # Full Vulkan support - wheel compiled with Vulkan
        kwargs["n_gpu_layers"] = gpu_layers
        
        # Add main_gpu parameter for multi-GPU systems
        if temporary.SELECTED_GPU and temporary.SELECTED_GPU != "Auto-Select":
            from scripts.utility import get_available_gpus
            gpu_list = get_available_gpus()
            try:
                gpu_index = gpu_list.index(temporary.SELECTED_GPU)
                kwargs["main_gpu"] = gpu_index
                print(f"[LOAD] Vulkan wheel main_gpu={gpu_index}")
            except (ValueError, IndexError):
                pass  # Already warned above, will use default
        
        print(f"[LOAD] Vulkan wheel – off-loading {gpu_layers} layers")
        
    elif temporary.BACKEND_TYPE == "VULKAN_CPU":
        # Vulkan binary available, but CPU wheel
        # GPU offloading handled by Vulkan binary via environment variable
        kwargs["n_gpu_layers"] = 0  # CPU wheel doesn't handle GPU
        print(f"[LOAD] Vulkan binary will handle {gpu_layers} layers (CPU wheel)")
        
    elif temporary.BACKEND_TYPE == "CPU_CPU":
        # Pure CPU mode
        kwargs["n_gpu_layers"] = 0
        print("[LOAD] CPU_CPU mode - no GPU offloading")
        
    else:
        print(f"[LOAD] Unknown backend {temporary.BACKEND_TYPE} - defaulting to CPU")
        kwargs["n_gpu_layers"] = 0

    if use_cpu_threads and CPU_THREADS is not None:
        kwargs["n_threads"] = CPU_THREADS

    # CRITICAL: Vision handler setup BEFORE Llama() initialization
    if is_vision:
        model_dir = model_path.parent
        mmproj_path_str = find_mmproj_file(str(model_path))
        
        if not mmproj_path_str:
            return f"Vision model requires mmproj file in {model_dir}", False, llm_state, models_loaded_state
        
        mmproj_path = Path(mmproj_path_str)
        
        try:
            model_lower = model.lower()
            
            # Qwen3-VL detection (NEW - highest priority)
            if "qwen3-vl" in model_lower or "qwen3vl" in model_lower:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                chat_handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path))
                kwargs["chat_handler"] = chat_handler
                kwargs["n_batch"] = 512  # Smaller batch for vision
                set_status(f"Qwen3-VL mode with {mmproj_path.name}", console=True)
                
            # Qwen2.5-VL detection  
            elif "qwen2.5-vl" in model_lower or "qwen2.5vl" in model_lower:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                chat_handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path))
                kwargs["chat_handler"] = chat_handler
                kwargs["n_batch"] = 512
                set_status(f"Qwen2.5-VL mode with {mmproj_path.name}", console=True)
                
            # Apriel detection (uses LLaVA architecture)
            elif "apriel" in model_lower:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
                kwargs["chat_handler"] = chat_handler
                set_status(f"Apriel (LLaVA) mode with {mmproj_path.name}", console=True)
                
            # MiniCPM detection
            elif "minicpm" in model_lower:
                from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
                chat_handler = MiniCPMv26ChatHandler(clip_model_path=str(mmproj_path))
                kwargs["chat_handler"] = chat_handler
                set_status(f"MiniCPM mode with {mmproj_path.name}", console=True)
                
            # Moondream detection
            elif "moondream" in model_lower:
                from llama_cpp.llama_chat_format import MoondreamChatHandler
                chat_handler = MoondreamChatHandler(clip_model_path=str(mmproj_path))
                kwargs["chat_handler"] = chat_handler
                set_status(f"Moondream mode with {mmproj_path.name}", console=True)
                
            # LLaVA and QVQ detection
            elif "llava" in model_lower or "qvq" in model_lower:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
                kwargs["chat_handler"] = chat_handler
                set_status(f"LLaVA mode with {mmproj_path.name}", console=True)
                
            else:
                # Default to LLaVA handler for unknown vision models
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
                kwargs["chat_handler"] = chat_handler
                set_status(f"Default vision mode with {mmproj_path.name}", console=True)
            
        except Exception as e:
            return f"Vision handler failed: {e}\nMake sure llama-cpp-python ≥0.3.2 is installed", False, llm_state, models_loaded_state

    # Now create Llama object with vision handler already configured
    try:
        # --- enforce hard limits to keep Vulkan graph allocatable ---
        n_ctx_train = metadata.get("llama.context_length", 32768)
        n_vocab     = metadata.get("tokenizer.ggml.model_count", 32000)
        n_embd      = metadata.get("llama.embedding_length", 4096)

        # 1. cap context to what the model was trained on
        effective_ctx = min(CONTEXT_SIZE, n_ctx_train)

        # 2. start from the user-selected u-batch (or default)
        n_ubatch = kwargs.get("n_ubatch", BATCH_SIZE)

        # 3. compute worst-case graph memory with *that* u-batch
        worst_mb = 3 * (n_ubatch * n_vocab * n_embd * 4) / 1024 / 1024
        if worst_mb > vram_size * 0.75:                # keep 25 % head-room
            n_ubatch = int((vram_size * 0.75 * 1024 * 1024 /
                            (3 * n_vocab * n_embd * 4)))
            n_ubatch = max(64, (n_ubatch // 64) * 64)
            print(f"[VULKAN] Vocab {n_vocab} huge – graph {worst_mb:.0f} MB – "
                  f"u-batch capped to {n_ubatch}")

        kwargs.update({
            "n_ctx": effective_ctx,
            "n_ctx_per_seq": effective_ctx,
            "n_batch": n_ubatch,
            "n_ubatch": n_ubatch,
        })
        # ---------------------------------------------------------

        new_llm = Llama(**kwargs)
        
        # Test inference
        test_msg = [{"role": "user", "content": "Hello"}]
        new_llm.create_chat_completion(
            messages=test_msg,
            max_tokens=BATCH_SIZE,
            stream=False
        )

        import scripts.temporary as tmp
        temporary.GPU_LAYERS = gpu_layers
        temporary.MODEL_NAME = model
        
        # Enhanced status for vision+thinking models
        status_parts = ["Model ready"]
        if is_vision:
            status_parts.append("vision")
        if is_reasoning:
            status_parts.append("thinking")
        status_msg = " + ".join(status_parts)
        
        set_status(status_msg, console=True, priority=True)
        from scripts.utility import beep
        beep()   
        return status_msg, True, new_llm, True

    except Exception as e:
        import scripts.temporary as tmp
        temporary.GPU_LAYERS = 0
        tb = traceback.format_exc()
        err = (f"Error loading model: {e}\n"
               f"GPU Layers: {gpu_layers}/{num_layers}\n"
               f"CPU Threads: {CPU_THREADS if use_cpu_threads else 'none'}\n"
               f"{tb}")
        print(err)
        set_status(f"Model load error", console=True, priority=True)
        return err, False, None, False
        
def calculate_single_model_gpu_layers_with_layers(
    model_path: str,
    available_vram: int,
    num_layers: int,
    dynamic_gpu_layers: bool = True
) -> int:
    """
    Fit as many whole layers as possible into the configured VRAM
    while keeping memory free for Vulkan graph buffers that scale with context.
    """
    from math import floor
    import scripts.temporary as tmp

    if tmp.BACKEND_TYPE == "CPU_CPU":
        return 0
    if tmp.BACKEND_TYPE not in ("VULKAN_VULKAN", "VULKAN_CPU"):
        return 0
    if available_vram <= 0 or num_layers <= 0:
        return 0

    model_mb = get_model_size(model_path)
    meta = get_model_metadata(model_path)
    arch = meta.get("general.architecture", "unknown")

    # model-specific overhead factor
    factor = 1.15 if arch in ("qwen2", "qwen2.5", "qwen") else 1.20 if arch == "llama" else 1.25
    adjusted_mb = model_mb * factor
    layer_mb = adjusted_mb / num_layers

    # context-dependent graph buffer reserve
    embedding_dim = {"llama": 4096, "qwen2": 5120, "qwen": 5120}.get(arch, 4096)
    graph_mb = 3 * (tmp.CONTEXT_SIZE / 1024) ** 2 * embedding_dim * 4 / 1024 / 1024
    reserve_mb = max(64, int(graph_mb))          # at least 64 MB
    usable_vram = max(0, available_vram - reserve_mb)

    max_layers = floor(usable_vram / layer_mb)
    gpu_layers = min(max_layers, num_layers) if dynamic_gpu_layers else num_layers
    gpu_layers = max(0, gpu_layers)

    print(f"[GPU-LAYERS] Model {adjusted_mb:.0f} MB  layer {layer_mb:.1f} MB  "
          f"VRAM {available_vram} MB  reserve {reserve_mb} MB  usable {usable_vram} MB → GPU {gpu_layers}/{num_layers}")
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

def build_messages_with_context_management(session_log, system_message, context_size):
    """
    Build the message list for the LLM.
    The system message is injected **once**, at the very first turn, UNLESS:
    - Model is Harmony/MOE (is_harmony=True) - these models don't use system prompts
    - Model is code-focused (is_code=True) - these use instruct format only
    System message is never shown in the Gradio chat log.
    """
    max_tokens   = context_size
    system_tokens= len(system_message) // 4

    available_for_history = int((max_tokens - system_tokens) * 0.30)
    available_for_input   = int((max_tokens - system_tokens) * 0.60)

    messages = []

    # ---------------  CONDITIONAL SYSTEM PROMPT  ---------------
    # Only add system message if:
    # 1. First turn (empty session_log)
    # 2. System message is not empty (Harmony/Code models return "")
    if not session_log and system_message:
        messages.append({"role": "system", "content": system_message})

    # ---------------  HISTORY  ---------------
    history_chars = 0
    for msg in reversed(session_log[:-2]):          # skip last two (current turn)
        content = clean_content(msg["role"], msg["content"])
        if history_chars + len(content) > available_for_history * 4:
            break
        messages.insert(1 if messages else 0, {     # insert after system if present
            "role": msg["role"],
            "content": content
        })
        history_chars += len(content)

    # ---------------  CURRENT INPUT  ---------------
    current_input = clean_content("user", session_log[-2]["content"])
    if len(current_input) > available_for_input * 4:
        current_input = temporary.context_injector.get_relevant_context(
            query=current_input[:1000], k=6, include_temp=True
        ) or current_input[:available_for_input * 4]
    messages.append({"role": "user", "content": current_input})

    return messages

def get_response_stream(session_log, settings, web_search_enabled=False, search_results=None,
                        cancel_event=None, llm_state=None, models_loaded_state=False):
    """
    Enhanced streaming response handler with robust thinking phase detection.
    Supports:
    - Standard <think></think> tags
    - gpt-oss Harmony format with channels
    - Hybrid models that may or may not use thinking
    - Filters out repeating AI-Chat: name tags
    - Vision models with image content (NEW)
    """
    import re
    from scripts import temporary
    from scripts.utility import clean_content, read_file_content
    from pathlib import Path

    if not models_loaded_state or llm_state is None:
        yield "Error: No model loaded. Please load a model first."
        return

    # Build system message
    system_message = get_system_message(
        is_uncensored=settings.get("is_uncensored", False),
        is_nsfw=settings.get("is_nsfw", False),
        web_search_enabled=web_search_enabled,
        is_reasoning=settings.get("is_reasoning", False),
        is_roleplay=settings.get("is_roleplay", False),
        is_code=settings.get("is_code", False),
        is_moe=settings.get("is_moe", False),
        is_vision=settings.get("is_vision", False)  # NEW
    ) + "\nRespond directly without prefixes like 'AI-Chat:'."

    if web_search_enabled and search_results:
        system_message += f"\n\nWeb search results:\n{search_results}"

    # Build message list
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    for msg in reversed(session_log[:-2]):
        messages.insert(1 if messages else 0, {
            "role": msg["role"], 
            "content": clean_content(msg["role"], msg["content"])
        })
    
    # NEW: Format current user message with images if vision model
    current_content = clean_content("user", session_log[-2]["content"])
    
    if settings.get("is_vision", False) and temporary.session_attached_files:
        # Check for images in attached files
        image_files = [f for f in temporary.session_attached_files 
                      if Path(f).suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}]
        
        if image_files:
            # Format multimodal message
            from scripts.utility import read_file_content
            
            multimodal_content = []
            
            # Add images first
            for img_path in image_files:
                data_uri, file_type, success, error = read_file_content(img_path)
                if success and file_type == "image":
                    multimodal_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    })
                    print(f"[VISION] Added image: {Path(img_path).name}")
            
            # Add text query
            multimodal_content.append({
                "type": "text",
                "text": current_content
            })
            
            messages.append({"role": "user", "content": multimodal_content})
            print(f"[VISION] Multimodal message: {len(image_files)} images + text")
        else:
            # No images, standard text
            messages.append({"role": "user", "content": current_content})
    else:
        # Non-vision model or no images
        messages.append({"role": "user", "content": current_content})

    # State tracking
    in_thinking_phase = False
    thinking_content_buffer = ""
    output_buffer = ""
    seen_real_text = False
    char_count_since_think_start = 0
    raw_output = ""

    # Detect if this is a gpt-oss model
    is_gpt_oss = 'gpt-oss' in temporary.MODEL_NAME.lower() if temporary.MODEL_NAME else False
    
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

        # ===== Strip repeating AI-Chat: tags =====
        output_buffer = re.sub(r'(^|\n)\s*AI-Chat:\s*(\n|$)', r'\1', output_buffer)
        output_buffer = re.sub(r'\bAI-Chat:\s*\n\s*', '', output_buffer)
        # ==============================================

        # Skip initial whitespace until we see real content
        if not seen_real_text:
            if token.strip() == "":
                continue
            seen_real_text = True

        # Guard against model trying to speak as user
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
            if before_think:
                yield before_think
            
            output_buffer = output_buffer.split("<think>", 1)[1]
            thinking_content_buffer = ""
            
            if temporary.SHOW_THINK_PHASE:
                yield "<think>"
            else:
                yield "Thinking"
            continue

        # gpt-oss Harmony format detection: <|start|>assistant<|channel|>analysis
        if not in_thinking_phase and "<|channel|>analysis" in output_buffer:
            in_thinking_phase = True
            char_count_since_think_start = 0
            
            parts = output_buffer.split("<|channel|>analysis", 1)
            before_analysis = parts[0]
            
            before_analysis = re.sub(r'<\|start\|>assistant', '', before_analysis)
            if before_analysis.strip():
                yield before_analysis
            
            output_buffer = parts[1] if len(parts) > 1 else ""
            thinking_content_buffer = ""
            
            if temporary.SHOW_THINK_PHASE:
                yield "<|channel|>analysis"
            else:
                yield "Thinking"
            continue

        # === INSIDE THINKING PHASE ===
        if in_thinking_phase:
            thinking_content_buffer += token
            char_count_since_think_start += len(token)
            
            # Standard </think> closing
            if "</think>" in output_buffer:
                in_thinking_phase = False
                
                parts = output_buffer.split("</think>", 1)
                think_content = parts[0]
                after_think = parts[1] if len(parts) > 1 else ""
                
                if temporary.SHOW_THINK_PHASE:
                    yield think_content + "</think>\n"
                else:
                    spaces = think_content.count(" ")
                    if spaces > 0:
                        yield "." * min(spaces, 50)
                    yield "\n"
                
                output_buffer = after_think
                thinking_content_buffer = ""
                if after_think:
                    yield after_think
                    output_buffer = ""
                continue
            
            # gpt-oss: Check for channel switch to 'final'
            if char_count_since_think_start >= temporary.THINK_MIN_CHARS_BEFORE_CLOSE:
                end_pattern = r'<\|end\|>.*?<\|channel\|>final'
                match = re.search(end_pattern, output_buffer, re.DOTALL)
                
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
                    thinking_content_buffer = ""
                    if after_think:
                        yield after_think
                        output_buffer = ""
                    continue
            
            # Still thinking - display according to mode
            if temporary.SHOW_THINK_PHASE:
                yield token
                output_buffer = ""
            else:
                spaces = token.count(" ")
                if spaces > 0:
                    yield "." * min(spaces, 10)
                output_buffer = ""
            continue

        # === NORMAL OUTPUT (not thinking) ===
        if len(output_buffer) > 20:
            yield output_buffer
            output_buffer = ""

    # Final flush
    if output_buffer and not in_thinking_phase:
        yield output_buffer

    # Print raw output if debugging enabled
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