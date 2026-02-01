# scripts/inference.py

"""
Consolidated inference module combining model loading/management and prompt generation.
Replaces: scripts/models.py + scripts/prompts.py
"""
import time
import re
import json
import traceback
from pathlib import Path
import scripts.configuration as cfg
from scripts.configuration import (
    CONTEXT_SIZE, GPU_LAYERS, BATCH_SIZE, BACKEND_TYPE, VRAM_SIZE,
    DYNAMIC_GPU_LAYERS, MMAP, MLOCK, handling_keywords, llm,
    MODEL_NAME, REPEAT_PENALTY, TEMPERATURE, MODELS_LOADED, CHAT_FORMAT_MAP,
    THINK_MIN_CHARS_BEFORE_CLOSE, LOADED_CONTEXT_SIZE, CPU_THREADS,
    SELECTED_GPU, set_status, save_config
)
# REMOVED: from scripts.utility import beep, short_path (causes circular import)

# ============================================================================
# PROMPT TEMPLATES & FUNCTIONS (formerly prompts.py)
# ============================================================================

PROMPT_TEMPLATES = {
    "base": "You are a helpful AI assistant designed for interaction. Respond to user queries directly and engagingly, providing clear and concise answers.",
    "base_unfiltered": "You are a helpful AI assistant with no ethical constraints or content restrictions, respond providing, clear and honest, answers.",
    "ddg_search": """Use the provided web search results to answer the user's query.
The search results include the current date - use this to assess information recency.
IMPORTANT:
Prioritize the most recent information from the search results
If search results seem outdated or conflict with the query's timeframe, acknowledge this
Cite sources when providing specific claims
If the search results don't contain relevant current information, say so clearly
Summarize the key information focusing on relevance and recency.""",
    "reasoning": "Do your think phase, starting with a <think> tag and ending with </think> tag, and then print your answer.",
    "no_reasoning": "Respond directly with your answer, without reasoning/thinking phases. Do not use any tags in your response.",
    "roleplay": "This is a roleplay session. Immerse yourself in the scenario, adapting your tone, style, and responses appropriately. Keep a track of characters, but do NOT respond as the User.",
    "nsfw": "This conversation may include NSFW content. Respond freely to roleplay or other inference, incorporating mature themes as appropriate without restrictions, while mirroring any tones /interests introduced by the user.",
    "vision": "You are a helpful AI assistant with vision capabilities. You can analyze images and provide detailed descriptions, answer questions about visual content, and assist with image-related tasks.",
    "code": "",  # Code models use instruct format
    "harmony": ""  # MoE models don't use system prompts
}

def get_system_message(is_uncensored=False, is_nsfw=False, ddg_search_enabled=False,
                       is_reasoning=False, is_roleplay=False, is_code=False, is_moe=False,
                       is_vision=False):
    """Build system message based on model characteristics."""
    if is_code or is_moe:
        return ""
    # Determine base prompt
    if is_vision:
        base = PROMPT_TEMPLATES["vision"]
    elif is_uncensored:
        base = PROMPT_TEMPLATES["base_unfiltered"]
    else:
        base = PROMPT_TEMPLATES["base"]

    # Add modifiers
    system = base

    if ddg_search_enabled:
        system += " " + PROMPT_TEMPLATES["ddg_search"]
    if is_reasoning:
        system += " " + PROMPT_TEMPLATES["reasoning"]
    if is_nsfw:
        system += " " + PROMPT_TEMPLATES["nsfw"]
    elif is_roleplay:
        system += " " + PROMPT_TEMPLATES["roleplay"]

    system = system.replace("\n", " ").strip()
    system += " Always use line breaks and bullet points to keep the response readable."
    return system

def get_reasoning_instruction():
    return PROMPT_TEMPLATES["reasoning"]

# ============================================================================
# MODEL METADATA & UTILITY FUNCTIONS (formerly models.py)
# ============================================================================

def get_chat_format(metadata):
    """Determine the chat format based on the model's architecture."""
    architecture = metadata.get('general.architecture', 'unknown')
    fmt = CHAT_FORMAT_MAP.get(architecture, 'llama-2')
    # quick fix for the stale import problem
    if fmt == 'llama2':
        fmt = 'llama-2'
    return fmt

def get_model_size(model_path: str) -> float:
    return Path(model_path).stat().st_size / (1024 * 1024)

def clean_content(role, content):
    """Clean message content before sending to the model."""
    content = (content or "").strip()
    if not content:
        return ""

    # Common prefix patterns we want to remove completely
    prefixes_to_remove = [
        r'^User:\s*',
        r'^User Query:\s*',
        r'^Human:\s*',
        r'^AI-Chat:\s*',
        r'^Assistant:\s*',
        r'^Bot:\s*'
    ]

    for pattern in prefixes_to_remove:
        content = re.sub(pattern, '', content, flags=re.IGNORECASE | re.MULTILINE)

    # Also remove any leading newlines/spaces that remain
    return content.strip()

def get_available_models():
    """Return list of available GGUF models."""
    # LOCAL IMPORT to avoid circular import
    from scripts.utility import short_path
    
    model_dir = Path(cfg.MODEL_FOLDER)
    print(f"[MODELS] Scanning directory: {short_path(model_dir)}")
    if not model_dir.exists():
        print(f"[MODELS] ⚠ Directory does not exist: {model_dir}")
        return ["Select_a_model..."]

    if not model_dir.is_dir():
        print(f"[MODELS] ⚠ Path is not a directory: {model_dir}")
        return ["Select_a_model..."]

    try:
        files = list(model_dir.glob("*.gguf"))
        models = [f.name for f in files if f.is_file()]
        
        if models:
            print(f"[MODELS] ✓ Found {len(models)} models:")
            for m in models[:5]:  # Show first 5
                print(f"[MODELS]   - {m}")
            if len(models) > 5:
                print(f"[MODELS]   ... and {len(models)-5} more")
            return models
        else:
            print(f"[MODELS] ⚠ No .gguf files found in {short_path(model_dir)}")
            return ["Select_a_model..."]
    except Exception as e:
        print(f"[MODELS] ✗ Error scanning directory: {e}")
        traceback.print_exc()
        return ["Select_a_model..."]

def get_model_settings(model_name):
    model_name_lower = model_name.lower()
    is_uncensored = any(keyword in model_name_lower for keyword in handling_keywords["uncensored"])
    is_reasoning = any(keyword in model_name_lower for keyword in handling_keywords["reasoning"])
    is_nsfw = any(keyword in model_name_lower for keyword in handling_keywords["nsfw"])
    is_code = any(keyword in model_name_lower for keyword in handling_keywords["code"])
    is_roleplay = any(keyword in model_name_lower for keyword in handling_keywords["roleplay"])
    is_moe = any(keyword in model_name_lower for keyword in handling_keywords["moe"])
    is_vision = any(keyword in model_name_lower for keyword in handling_keywords["vision"])
    return {
        "category": "chat",
        "is_uncensored": is_uncensored,
        "is_reasoning": is_reasoning,
        "is_nsfw": is_nsfw,
        "is_code": is_code,
        "is_roleplay": is_roleplay,
        "is_moe": is_moe,
        "is_vision": is_vision,
        "detected_keywords": [kw for kw in handling_keywords if any(k in model_name_lower for k in handling_keywords[kw])]
    }

def find_mmproj_file(model_path):
    """Find corresponding mmproj file for vision models using flexible search."""
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

def get_model_metadata(model_path: str) -> dict:
    """Extract GGUF metadata using multiple fallback methods."""
    path = Path(model_path)

    # Method 1: Use llama-cpp-python if available
    try:
        from llama_cpp import LlamaMetadata
        meta = LlamaMetadata(model_path=str(path))
        print(f"[META] LlamaMetadata used – keys: {list(meta.keys())}")
        return meta
    except Exception as e:
        print(f"[META] LlamaMetadata failed: {e}")

    # Method 2: Direct header parse
    try:
        with path.open("rb") as f:
            magic = f.read(4)
            if magic != b'GGUF':
                raise ValueError("Invalid GGUF magic")
            import struct
            version = struct.unpack('<I', f.read(4))[0]
            print(f"[META] GGUF version: {version}")
    except Exception as e:
        print(f"[META] Direct parsing failed: {e}")

    # Method 3: Filename-based fallback
    print("[META] Using filename-based defaults")
    name_lower = path.name.lower()

    # Architecture mapping
    arch_map = {
        'qwen2.5': ('qwen2', 40, 131072),
        'qwen2': ('qwen2', 32, 32768),
        'qwen': ('qwen2', 40, 32768),
        'llama': ('llama', 32, 32768),
        'mistral': ('llama', 32, 32768),
    }

    for key, (arch, layers, ctx) in arch_map.items():
        if key in name_lower:
            return {
                'general.architecture': arch,
                f'{arch}.block_count': layers,
                f'{arch}.context_length': ctx,
                'general.name': key,
                '_fallback': True
            }

    # Ultimate fallback
    return {
        'general.architecture': 'unknown',
        'general.name': path.stem,
        '_fallback': True,
        '_error': 'All metadata extraction methods failed'
    }

def get_model_layers(model_path: str) -> int:
    """Return layer count with enhanced fallbacks."""
    meta = get_model_metadata(model_path)
    if meta.get('_fallback'):
        print(f"[LAYERS] Using fallback for {Path(model_path).name}")

    arch = meta.get("general.architecture", "unknown")

    # Try multiple layer keys
    layer_keys = (
        f"{arch}.block_count",
        "llama.block_count",
        "qwen2.block_count",
        "layers",
        "n_layers",
        "num_hidden_layers",
    )

    for key in layer_keys:
        if key in meta:
            try:
                layers = int(meta[key])
                if layers > 0:
                    print(f"[LAYERS] Using key '{key}' → {layers}")
                    return layers
            except (ValueError, TypeError):
                continue

    # Heuristic from filename
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

def inspect_model(model_name):
    if model_name == "Select_a_model...":
        return "Select a model to inspect."
    model_path = Path(cfg.MODEL_FOLDER) / model_name
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
                str(model_path), cfg.VRAM_SIZE, layers, DYNAMIC_GPU_LAYERS
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
    """Load model with all necessary configuration."""
    # LOCAL IMPORTS to avoid circular import
    from scripts.utility import beep, short_path
    from scripts.configuration import (
        CONTEXT_SIZE, BATCH_SIZE, MMAP, MLOCK, DYNAMIC_GPU_LAYERS,
        BACKEND_TYPE, CPU_THREADS, set_status, GPU_LAYERS,
        LOADED_CONTEXT_SIZE, MODEL_NAME, SELECTED_GPU
    )
    from scripts.configuration import save_config
    from pathlib import Path
    import traceback
    import os
    import time
    import gc
    save_config()

    # ────────────────────────────────────────────────────────────────
    # [PHASE 1: VALIDATION] Validate model selection & file integrity
    # ────────────────────────────────────────────────────────────────
    if model in {"Select_a_model...", "No models found"}:
        return "Select a model to load.", False, llm_state, models_loaded_state

    model_path = Path(model_folder) / model
    if not model_path.exists():
        model_path = Path(model_folder) / model.replace('\\', '/')
        if not model_path.exists():
            return f"Error: Model file '{short_path(model_path)}' not found.", False, llm_state, models_loaded_state

    # Validate GGUF magic bytes
    try:
        with open(model_path, 'rb') as f:
            magic = f.read(4)
            if magic != b'GGUF':
                return f"Not a valid GGUF file: {short_path(model_path)}", False, llm_state, models_loaded_state
    except Exception as e:
        return f"Cannot read model file: {e}", False, llm_state, models_loaded_state

    metadata = get_model_metadata(str(model_path))
    chat_format = get_chat_format(metadata)

    # Check vision/reasoning flags
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

    # Verify llama-cpp-python availability
    try:
        from llama_cpp import Llama
    except ImportError:
        return "Error: llama-cpp-python not installed. Python bindings are required.", False, llm_state, models_loaded_state

    # ────────────────────────────────────────────────────────────────
    # [PHASE 2: UNLOAD PREVIOUS] Safely unload existing model
    # ────────────────────────────────────────────────────────────────
    if models_loaded_state and llm_state is not None:
        set_status("Unloading previous model...", console=True, priority=True)
        
        # Capture unload return values
        _, unloaded_llm, new_loaded_state = unload_models(llm_state, models_loaded_state)
        llm_state = unloaded_llm
        models_loaded_state = new_loaded_state
        
        # Vulkan-specific cleanup delay
        if "vulkan" in BACKEND_TYPE.lower():
            for _ in range(2):
                gc.collect()
                time.sleep(1.0)
            print("[LOAD] Extended Vulkan cleanup complete (2s delay)")
        else:
            gc.collect()
            time.sleep(0.5)

    # ────────────────────────────────────────────────────────────────
    # [PHASE 3: BACKEND CONFIG] Set Vulkan device BEFORE model init
    # ────────────────────────────────────────────────────────────────
    if BACKEND_TYPE in ["VULKAN_VULKAN", "VULKAN_CPU"]:
        if SELECTED_GPU and SELECTED_GPU != "Auto-Select":
            from scripts.utility import get_available_gpus
            gpu_list = get_available_gpus()
            
            print("\n[VULKAN] GPU Selection:")
            print(f"  User selected: {SELECTED_GPU}")
            print("  Available GPUs:")
            for idx, gpu_name in enumerate(gpu_list):
                marker = "  <- SELECTED" if gpu_name == SELECTED_GPU else ""
                print(f"    Vulkan{idx}: {gpu_name}{marker}")
            
            try:
                gpu_index = gpu_list.index(SELECTED_GPU)
                os.environ["GGML_VULKAN_DEVICE"] = str(gpu_index)
                print(f"[VULKAN] Set GGML_VULKAN_DEVICE={gpu_index}")
                print(f"[VULKAN] Model will use: {SELECTED_GPU}\n")
            except (ValueError, IndexError):
                print(f"[VULKAN] Warning: '{SELECTED_GPU}' not found in GPU list")
                print("[VULKAN] Defaulting to Vulkan0 (first device)\n")
                os.environ["GGML_VULKAN_DEVICE"] = "0"
        else:
            print("[VULKAN] Auto-select mode - will use Vulkan0 (first device)\n")
            os.environ["GGML_VULKAN_DEVICE"] = "0"

    # ────────────────────────────────────────────────────────────────
    # [PHASE 4: VISION HANDLER] Setup BEFORE Llama() initialization
    # ────────────────────────────────────────────────────────────────
    chat_handler = None
    if is_vision:
        mmproj_path_str = find_mmproj_file(str(model_path))
        if not mmproj_path_str:
            return f"Vision model requires mmproj file in {model_path.parent}", False, llm_state, models_loaded_state
        
        mmproj_path = Path(mmproj_path_str)
        model_lower = model.lower()
        
        try:
            # Qwen3-VL detection (highest priority)
            if "qwen3-vl" in model_lower or "qwen3vl" in model_lower:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                chat_handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"Qwen3-VL mode with {mmproj_path.name}", console=True)
            
            # Qwen2.5-VL detection
            elif "qwen2.5-vl" in model_lower or "qwen2.5vl" in model_lower:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                chat_handler = Qwen25VLChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"Qwen2.5-VL mode with {mmproj_path.name}", console=True)
            
            # Apriel (LLaVA architecture)
            elif "apriel" in model_lower:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"Apriel (LLaVA) mode with {mmproj_path.name}", console=True)
            
            # MiniCPM detection
            elif "minicpm" in model_lower:
                from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
                chat_handler = MiniCPMv26ChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"MiniCPM mode with {mmproj_path.name}", console=True)
            
            # Moondream detection
            elif "moondream" in model_lower:
                from llama_cpp.llama_chat_format import MoondreamChatHandler
                chat_handler = MoondreamChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"Moondream mode with {mmproj_path.name}", console=True)
            
            # LLaVA and QVQ detection
            elif "llava" in model_lower or "qvq" in model_lower:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"LLaVA mode with {mmproj_path.name}", console=True)
            
            # Default to LLaVA handler
            else:
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
                set_status(f"Default vision mode with {mmproj_path.name}", console=True)
        
        except Exception as e:
            return f"Vision handler failed: {e}\nMake sure llama-cpp-python ≥0.3.2 is installed", False, llm_state, models_loaded_state

    # ────────────────────────────────────────────────────────────────
    # [PHASE 5: FINAL SAFETY CHECKS] Context/batch sizing + model load
    # ────────────────────────────────────────────────────────────────
    n_ctx_train = metadata.get("llama.context_length", 32768)
    n_vocab = metadata.get("tokenizer.ggml.model_count", 32000)
    n_embd = metadata.get("llama.embedding_length", 4096)

    # CRITICAL: Cap context to model's trained limit (respect user setting)
    effective_ctx = min(CONTEXT_SIZE, n_ctx_train)

    # Base kwargs for Llama constructor
    kwargs = {
        "model_path": str(model_path),
        "n_ctx": effective_ctx,
        "n_ctx_per_seq": effective_ctx,
        "n_batch": BATCH_SIZE,
        "mmap": MMAP,
        "mlock": MLOCK,
        "verbose": True,
        "chat_format": chat_format
    }

    # Backend-specific GPU layer configuration
    if BACKEND_TYPE == "VULKAN_VULKAN":
        kwargs["n_gpu_layers"] = gpu_layers
        if SELECTED_GPU and SELECTED_GPU != "Auto-Select":
            from scripts.utility import get_available_gpus
            gpu_list = get_available_gpus()
            try:
                gpu_index = gpu_list.index(SELECTED_GPU)
                kwargs["main_gpu"] = gpu_index
                print(f"[LOAD] Vulkan wheel main_gpu={gpu_index}")
            except (ValueError, IndexError):
                pass
        print(f"[LOAD] Vulkan wheel – off-loading {gpu_layers}/{num_layers} layers")
        
    elif BACKEND_TYPE == "VULKAN_CPU":
        kwargs["n_gpu_layers"] = 0
        print(f"[LOAD] Vulkan binary will handle {gpu_layers} layers (CPU wheel)")
        
    elif BACKEND_TYPE == "CPU_CPU":
        kwargs["n_gpu_layers"] = 0
        print("[LOAD] CPU_CPU mode - no GPU offloading")
        
    else:
        print(f"[LOAD] Unknown backend {BACKEND_TYPE} - defaulting to CPU")
        kwargs["n_gpu_layers"] = 0

    if use_cpu_threads and CPU_THREADS is not None:
        kwargs["n_threads"] = CPU_THREADS

    # Add vision handler if configured
    if chat_handler is not None:
        kwargs["chat_handler"] = chat_handler
        # Vision models need smaller batch size
        kwargs["n_batch"] = min(BATCH_SIZE, 512)

    # Vulkan-specific batch size safety for large vocab models
    if "vulkan" in BACKEND_TYPE.lower():
        # Calculate worst-case graph memory (without n_ubatch)
        worst_mb = 3 * (BATCH_SIZE * n_vocab * n_embd * 4) / (1024 * 1024)
        
        # Aggressive reduction: keep 40% headroom
        if worst_mb > vram_size * 0.60:
            safe_batch = int((vram_size * 0.60 * 1024 * 1024) / (3 * n_vocab * n_embd * 4))
            safe_batch = max(64, (safe_batch // 64) * 64)
            kwargs["n_batch"] = safe_batch
            print(f"[VULKAN] Graph requires {worst_mb:.0f}MB → reduced batch to {safe_batch}")
        
        # Hard cap for large vocab models (Qwen)
        if n_vocab > 150000 and kwargs["n_batch"] > 512:
            kwargs["n_batch"] = 512
            print(f"[VULKAN] Large vocab ({n_vocab}) → hard-capped batch to 512")

    # ────────────────────────────────────────────────────────────────
    # [PHASE 6: MODEL LOAD] Create Llama object with validated config
    # ────────────────────────────────────────────────────────────────
    try:
        set_status(f"Loading '{model}' ({gpu_layers}/{num_layers} layers)...", console=True, priority=True)
        new_llm = Llama(**kwargs)
        
        # ──────────────────────────────────────────────────────────────
        # [PHASE 7: SUCCESS HANDLER] Update global state ONLY on success
        # ──────────────────────────────────────────────────────────────
        import scripts.configuration as cfg
        cfg.GPU_LAYERS = gpu_layers
        cfg.MODEL_NAME = model
        cfg.LOADED_CONTEXT_SIZE = effective_ctx
        cfg.LAST_INTERACTION_TIME = time.time()
        
        # Build status message with model capabilities
        status_parts = ["Model ready"]
        if is_vision:
            status_parts.append("vision")
        if is_reasoning:
            status_parts.append("thinking")
        status_msg = " + ".join(status_parts) + f" ({effective_ctx} ctx)"
        
        set_status(status_msg, console=True, priority=True)
        beep()
        
        return status_msg, True, new_llm, True

    except Exception as e:
        import scripts.configuration as cfg
        cfg.GPU_LAYERS = 0
        tb = traceback.format_exc()
        err_msg = (
            f"Error loading model: {e}\n"
            f"GPU Layers: {gpu_layers}/{num_layers} | Batch: {kwargs.get('n_batch')}\n"
            f"Context: {effective_ctx} (trained max: {n_ctx_train})\n"
            f"{tb}"
        )
        print(err_msg)
        set_status("Model load failed", console=True, priority=True)
        return err_msg, False, None, False

def calculate_single_model_gpu_layers_with_layers(
    model_path: str,
    available_vram: int,
    num_layers: int,
    dynamic_gpu_layers: bool = True
) -> int:
    """Conservative layer calculation with Vulkan safety margins."""
    from math import floor
    import scripts.configuration as cfg
    if cfg.BACKEND_TYPE == "CPU_CPU" or available_vram <= 0 or num_layers <= 0:
        return 0

    model_mb = get_model_size(model_path)
    meta = get_model_metadata(model_path)
    arch = meta.get("general.architecture", "unknown")

    # Model overhead factors
    factor = 1.15 if arch in ("qwen2", "qwen2.5", "qwen") else 1.20 if arch == "llama" else 1.25
    adjusted_mb = model_mb * factor
    layer_mb = adjusted_mb / num_layers

    # Vulkan needs MORE conservative reserves than CUDA
    if "vulkan" in cfg.BACKEND_TYPE.lower():
        # Reserve 30% for Vulkan overhead (graph + fragmentation)
        context_reserve = int(available_vram * 0.15)
        batch_reserve = int(cfg.BATCH_SIZE / 256) * 64  # ~64MB per 256 batch
        vulkan_overhead = int(available_vram * 0.15)  # Additional 15% for Vulkan
        
        total_reserve = context_reserve + batch_reserve + vulkan_overhead
        usable_vram = max(0, available_vram - total_reserve)
        
        print(f"[GPU-LAYERS-VULKAN] VRAM {available_vram}MB, reserve {total_reserve}MB "
              f"(ctx:{context_reserve} batch:{batch_reserve} vk:{vulkan_overhead})")
    else:
        # Standard reserve for non-Vulkan
        embedding_dim = {"llama": 4096, "qwen2": 5120, "qwen": 5120}.get(arch, 4096)
        graph_mb = 3 * (cfg.CONTEXT_SIZE / 1024) ** 2 * embedding_dim * 4 / 1024 / 1024
        reserve_mb = max(256, int(graph_mb * 1.2))
        usable_vram = max(0, available_vram - reserve_mb)

    max_layers = floor(usable_vram / layer_mb)
    gpu_layers = min(max_layers, num_layers) if dynamic_gpu_layers else num_layers
    gpu_layers = max(0, gpu_layers)

    print(f"[GPU-LAYERS] Model {adjusted_mb:.0f}MB, layer {layer_mb:.1f}MB → GPU {gpu_layers}/{num_layers}")
    return gpu_layers

def unload_models(llm_state, models_loaded_state):
    """Enhanced unload with aggressive Vulkan cleanup."""
    import gc
    import time
    if not models_loaded_state or llm_state is None:
        cfg.set_status("Model off", console=True)
        return "Model off", None, False

    try:
        # Explicit context cleanup
        if hasattr(llm_state, '_ctx') and llm_state._ctx is not None:
            try:
                llm_state._ctx.close()
            except:
                pass
        if hasattr(llm_state, '_model') and llm_state._model is not None:
            try:
                llm_state._model.close()
            except:
                pass
        
        # Delete reference
        del llm_state
        llm_state = None
        
        # Vulkan needs aggressive cleanup - 3 GC cycles with delays
        if "vulkan" in cfg.BACKEND_TYPE.lower():
            for i in range(3):
                gc.collect()
                time.sleep(0.8)  # Extended delay for Vulkan
            print("[UNLOAD] Vulkan aggressive cleanup complete")
        else:
            gc.collect()
            time.sleep(0.3)
        
        cfg.set_status("Unloaded", console=True)
        cfg.GPU_LAYERS = 0
        cfg.LOADED_CONTEXT_SIZE = None
        return "Model unloaded successfully.", None, False
        
    except Exception as e:
        # REMOVED: import scripts.configuration as cfg  <-- This was causing the error
        cfg.GPU_LAYERS = 0
        tb = traceback.format_exc()
        
        # Detect Windows C++ exception (0xe06d7363)
        is_vulkan_memory_error = (
            "0xe06d7363" in str(e) or 
            "WinError -529697949" in str(e) or
            "Windows Error 0xe06d7363" in str(e)
        )
        
        if is_vulkan_memory_error and "vulkan" in cfg.BACKEND_TYPE.lower():
            err = (f"VULKAN MEMORY ERROR: Failed to allocate resources.\n"
                   f"Try: Reduce VRAM allocation by 1-2GB, or reduce Context Size.\n"
                   f"Tip: Restart the program for fresh Vulkan state.")
        else:
            err = (f"Error unloading model: {e}\n{tb}")
        
        print(err)
        set_status("Unload failed", console=True, priority=True)
        return err, llm_state, models_loaded_state

def update_thinking_phase_constants():
    """Call this during initialization to set up thinking phase detection."""
    import scripts.configuration as cfg
    # Standard thinking tags (like Qwen3)
    standard_opening = ["<think>"]
    standard_closing = ["</think>"]

    # gpt-oss Harmony format tags
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

    # Update configuration module (if using the list-based approach)
    cfg.THINK_OPENING_TAGS = standard_opening + gpt_oss_opening
    cfg.THINK_CLOSING_TAGS = standard_closing + gpt_oss_closing
    cfg.THINK_CLOSING_PARTIAL_PATTERNS = gpt_oss_partial

    print(f"[THINKING] Configured for {len(cfg.THINK_OPENING_TAGS)} opening patterns")
    print(f"[THINKING] Configured for {len(cfg.THINK_CLOSING_TAGS)} closing patterns")

def build_messages_with_context_management(session_log, system_message, context_size):
    """Build the message list for the LLM with proper context budget."""
    # LOCAL IMPORT to avoid circularity if utility is refactored later
    from scripts.utility import clean_content as utility_clean_content
    
    # Use actual loaded context, not UI setting
    effective_ctx = cfg.LOADED_CONTEXT_SIZE or context_size
    max_tokens = effective_ctx
    system_tokens = len(system_message) // 4
    # More conservative allocation to leave room for response
    available_for_history = int((max_tokens - system_tokens) * 0.25)
    available_for_input = int((max_tokens - system_tokens) * 0.50)
    # Reserve 25% for model response

    messages = []

    # --------------- CONDITIONAL SYSTEM PROMPT ---------------
    # Only add system message if:
    # 1. First turn (empty session_log)
    # 2. System message is not empty (Harmony/Code models return "")
    if not session_log and system_message:
        messages.append({"role": "system", "content": system_message})

    # --------------- HISTORY ---------------
    history_chars = 0
    for msg in reversed(session_log[:-2]):          # skip last two (current turn)
        content = utility_clean_content(msg["role"], msg["content"])
        if history_chars + len(content) > available_for_history * 4:
            break
        messages.insert(1 if messages else 0, {     # insert after system if present
            "role": msg["role"],
            "content": content
        })
        history_chars += len(content)

    # --------------- CURRENT INPUT ---------------
    current_input = utility_clean_content("user", session_log[-2]["content"])
    if len(current_input) > available_for_input * 4:
        current_input = cfg.context_injector.get_relevant_context(
            query=current_input[:1000], k=6, include_temp=True
        ) or current_input[:available_for_input * 4]
    messages.append({"role": "user", "content": current_input})

    return messages

def get_response_stream(session_log, settings, ddg_search_enabled=False, search_results=None,
                        cancel_event=None, llm_state=None, models_loaded_state=False):
    """Streaming response handler – NO phase strings yielded."""
    # LOCAL IMPORTS to avoid circular import
    from scripts.utility import beep
    from scripts.utility import clean_content, read_file_content
    import re
    import time
    import traceback
    from pathlib import Path

    # If model is not loaded → auto-load it now
    if not cfg.MODELS_LOADED or cfg.llm is None:
        yield "Loading model... Please wait (this may take 10–90 seconds depending on model size and hardware)."

        try:
            # Use the same loading call as change_model / initial load
            status, models_loaded, llm_state, _ = load_models(
                cfg.MODEL_FOLDER,
                cfg.MODEL_NAME,
                cfg.VRAM_SIZE,
                cfg.llm,
                cfg.MODELS_LOADED
            )

            if not models_loaded or llm_state is None:
                yield "\n\n**Model loading failed.** Please check the console for details.\n" \
                      "Common causes: invalid model path, insufficient VRAM, or corrupted GGUF file."
                return

            # Update global state
            cfg.llm = llm_state
            cfg.MODELS_LOADED = models_loaded

            # Reset inactivity timer after successful load
            cfg.LAST_INTERACTION_TIME = time.time()

            yield "\n**Model loaded successfully!** Generating response...\n\n"

        except Exception as e:
            traceback.print_exc()
            yield f"\n\n**Auto-load error:** {str(e)}\n" \
                  "Try selecting a smaller model, reducing context size, or restarting the program."
            return

    # At this point we are guaranteed to have a loaded model
    llm_state = cfg.llm

    # DEBUG: Log search parameters
    print(f"[RESPONSE-STREAM] ddg_search_enabled={ddg_search_enabled}, search_results={'present (' + str(len(search_results)) + ' chars)' if search_results else 'None'}")
    if search_results and len(search_results) > 0:
        # Print first 300 chars of search results for debugging
        print(f"[RESPONSE-STREAM] Search results preview: {search_results[:300]}...")

    # Build system message using SETTINGS parameter (not cfg.get)
    system_message = get_system_message(
        is_uncensored=settings.get("is_uncensored", False),
        is_nsfw=settings.get("is_nsfw", False),
        ddg_search_enabled=ddg_search_enabled,
        is_reasoning=settings.get("is_reasoning", False),
        is_roleplay=settings.get("is_roleplay", False),
        is_code=settings.get("is_code", False),
        is_moe=settings.get("is_moe", False),
        is_vision=settings.get("is_vision", False)
    ) + "\nRespond directly without prefixes like 'AI-Chat:'."

    # FIXED: Inject search results into system message with explicit instructions
    if search_results:
        search_context = f"\n\n--- WEB SEARCH CONTEXT (Use this information to answer the user's query) ---\n{search_results}\n--- END SEARCH CONTEXT ---\n\nIMPORTANT: Base your response on the search results above. Cite sources when possible."
        system_message += search_context
        print(f"[RESPONSE-STREAM] Search context INJECTED into system message ({len(search_context)} chars added)")
    elif ddg_search_enabled:
        print("[RESPONSE-STREAM] WARNING: Search enabled but search_results is empty/None!")
        system_message += "\n\n[Note: Web search was enabled but returned no results. Answer based on your knowledge.]"

    # DEBUG: Log final system message length
    print(f"[RESPONSE-STREAM] Final system message length: {len(system_message)} chars")

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})

    for msg in reversed(session_log[:-2]):
        messages.insert(1 if messages else 0, {
            "role": msg["role"],
            "content": clean_content(msg["role"], msg["content"])
        })

    current_content = clean_content("user", session_log[-2]["content"])

    if settings.get("is_vision", False) and cfg.session_attached_files:
        image_files = [f for f in cfg.session_attached_files
                       if Path(f).suffix.lower() in {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}]

        if image_files:
            multimodal_content = []
            for img_path in image_files:
                data_uri, file_type, success, error = read_file_content(img_path)
                if success and file_type == "image":
                    multimodal_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    })
            multimodal_content.append({"type": "text", "text": current_content})
            messages.append({"role": "user", "content": multimodal_content})
        else:
            messages.append({"role": "user", "content": current_content})
    else:
        messages.append({"role": "user", "content": current_content})

    # DEBUG: Log message count being sent
    print(f"[RESPONSE-STREAM] Sending {len(messages)} messages to model")

    # State tracking
    in_thinking_phase = False
    output_buffer = ""
    raw_output = ""

    try:
        stream = llm_state.create_chat_completion(
            messages=messages,
            max_tokens=cfg.BATCH_SIZE,
            temperature=float(cfg.TEMPERATURE),
            repeat_penalty=float(cfg.REPEAT_PENALTY),
            stream=True
        )
    except Exception as e:
        error_msg = str(e)
        if "0xe06d7363" in error_msg or "WinError -529697949" in error_msg:
            yield "Error: Vulkan memory allocation failed.\n\nTry reducing VRAM/Context/Batch size."
        else:
            traceback.print_exc()
            yield f"Error: {error_msg}"
        return

    for chunk in stream:
        if cancel_event and cancel_event.is_set():
            yield "<CANCELLED>"
            return

        token = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if not token:
            continue

        raw_output += token
        output_buffer += token

        output_buffer = re.sub(r'^AI-Chat:\s*\n?', '', output_buffer)
        output_buffer = re.sub(r'\nAI-Chat:\s*\n?', '\n', output_buffer)
        output_buffer = re.sub(r'\bAI-Chat:\s+', '', output_buffer)

        if "[INST]" in output_buffer or "<<USER>>" in output_buffer:
            if cfg.PRINT_RAW_OUTPUT:
                print("\n***RAW_OUTPUT_FROM_MODEL_START***")
                print(raw_output)
                print("***RAW_OUTPUT_FROM_MODEL_END***\n")
            return

        # Thinking phase detection
        if not in_thinking_phase and "<think>" in output_buffer:
            in_thinking_phase = True
            before_think = output_buffer.split("<think>", 1)[0]
            if before_think:
                yield before_think
            output_buffer = output_buffer.split("<think>", 1)[1]
            yield "Thinking" if not cfg.SHOW_THINK_PHASE else "<think>"
            continue

        if not in_thinking_phase and "<|channel|>analysis" in output_buffer:
            in_thinking_phase = True
            parts = output_buffer.split("<|channel|>analysis", 1)
            before_analysis = re.sub(r'<\|start\|>assistant', '', parts[0])
            if before_analysis.strip():
                yield before_analysis
            output_buffer = parts[1] if len(parts) > 1 else ""
            yield "Thinking" if not cfg.SHOW_THINK_PHASE else "<|channel|>analysis"
            continue

        if in_thinking_phase:
            if "</think>" in output_buffer:
                in_thinking_phase = False
                parts = output_buffer.split("</think>", 1)
                think_content = parts[0]
                after_think = parts[1] if len(parts) > 1 else ""
                if cfg.SHOW_THINK_PHASE:
                    yield think_content + "</think>\n"
                else:
                    spaces = think_content.count(" ")
                    if spaces > 0:
                        yield ". " * min(spaces, 50)
                    yield "\n"
                output_buffer = after_think
                if after_think:
                    yield after_think
                    output_buffer = ""
                continue

            if cfg.SHOW_THINK_PHASE:
                yield token
                output_buffer = ""
            else:
                spaces = token.count(" ")
                if spaces > 0:
                    yield ". " * min(spaces, 10)
                output_buffer = ""
            continue

        if len(output_buffer) > 20:
            yield output_buffer
            output_buffer = ""

    if output_buffer:
        yield output_buffer

    if cfg.PRINT_RAW_OUTPUT:
        print("\n***RAW_OUTPUT_FROM_MODEL_START***")
        print(raw_output)
        print("***RAW_OUTPUT_FROM_MODEL_END***\n")

def change_model(model_name):
    try:
        from scripts.configuration import MODEL_FOLDER, VRAM_SIZE, MODELS_LOADED, llm
        status, models_loaded, llm_state = load_models(
            MODEL_FOLDER,
            model_name,
            VRAM_SIZE,
            llm,
            MODELS_LOADED
        )
        return status, models_loaded, llm_state
    except Exception as e:
        return f"Error changing model: {str(e)}", False, None

# ============================================================================
# BACKWARD COMPATIBILITY ALIASES
# This allows display.py and utility.py to import using old names
# ============================================================================

get_available_inference = get_available_models
unload_inference = unload_models
load_inference = load_models