# Script: `.\scripts\models.py`

# Imports...
import subprocess
from llama_cpp import Llama
from pathlib import Path
from scripts.temporary import (
    N_CTX, N_GPU_LAYERS, USE_PYTHON_BINDINGS,
    LLAMA_CLI_PATH, MODEL_LOADED, BACKEND_TYPE, VRAM_SIZE,
    DYNAMIC_GPU_LAYERS, MMAP, MLOCK, current_model_settings, time,
    quality_llm, fast_llm, prompt_templates
)

# Classes...
class ContextInjector:
    def __init__(self):
        self.vectorstores = {}
        self.current_vectorstore = None
        self.current_mode = None
        self._load_default_vectorstores()

    def _load_default_vectorstores(self):
        """Load all mode-specific vectorstores that exist."""
        modes = ["code", "rpg", "uncensored", "general"]
        for mode in modes:
            vs_path = Path("data/vectors") / mode / "knowledge"
            if vs_path.exists():
                from langchain_community.vectorstores import FAISS
                from langchain_huggingface import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                self.vectorstores[mode] = FAISS.load_local(
                    str(vs_path),
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded {mode} vectorstore.")

    def set_mode(self, mode: str):
        """Set the current mode and select the corresponding vectorstore."""
        valid_modes = ["code", "rpg", "uncensored", "general"]
        if mode in valid_modes:
            if mode not in self.vectorstores:
                self.load_vectorstore(mode)
            if mode in self.vectorstores:
                self.current_vectorstore = self.vectorstores[mode]
                self.current_mode = mode
            else:
                self.current_vectorstore = None
                self.current_mode = None
                print(f"No vectorstore found for mode: {mode}")
        else:
            print(f"Invalid mode: {mode}. Expected one of {valid_modes}")

    def load_vectorstore(self, mode: str):
        """Load a specific vectorstore by mode."""
        vs_path = Path("data/vectors") / mode / "knowledge"
        if vs_path.exists():
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vectorstores[mode] = FAISS.load_local(
                str(vs_path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Loaded {mode} vectorstore.")
        else:
            print(f"Vectorstore not found for mode: {mode}")

    def inject_context(self, prompt: str) -> str:
        """Inject context from the current mode's vectorstore into the prompt."""
        if self.current_vectorstore is None:
            return prompt
        from sentence_transformers import SentenceTransformer
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = embedding_model.encode([prompt])[0]
        docs = self.current_vectorstore.similarity_search_by_vector(query_embedding, k=4)
        context = "\n".join([doc.page_content for doc in docs])
        return f"Relevant information:\n{context}\n\nQuery: {prompt}"

context_injector = ContextInjector()

# Functions...
def reload_vectorstore(name: str):
    context_injector.load_vectorstore(name)
    return "Knowledge updated"

def get_model_size(model_path: str) -> float:
    """Get model file size in MB."""
    return Path(model_path).stat().st_size / (1024 * 1024)

def get_model_layers(model_path: str) -> int:
    """Get number of layers from GGUF model metadata."""
    try:
        llm_temp = Llama(model_path=model_path, verbose=False)
        metadata = llm_temp.metadata
        num_layers = metadata.get('llama.block_count', 0)
        del llm_temp
        return num_layers
    except Exception:
        return 0

def calculate_gpu_layers(model_path: str, vram_size: float) -> tuple:
    """Calculate GPU layers and return (gpu_layers, total_layers)."""
    if not Path(model_path).exists():
        raise ValueError(f"Model file not found at {model_path}!")
    if vram_size <= 0:
        raise ValueError(f"Invalid VRAM size: {vram_size} MB!")
    
    model_size = get_model_size(model_path)  # Model size in MB
    num_layers = get_model_layers(model_path)
    if num_layers == 0:
        raise RuntimeError(f"Failed to retrieve layer count from {model_path}!")
    
    layer_size = (model_size * 1.25) / num_layers  # Per readme.md
    available_vram = (vram_size / 1024) * 0.9  # Convert MB to GB, use 90% of VRAM
    max_layers = int(available_vram / layer_size)
    final_layers = min(max_layers, num_layers) if max_layers > 0 else 0
    
    print(f"Model: {Path(model_path).name}")
    print(f"Model Size: {model_size:.2f} MB")
    print(f"Total Layers: {num_layers}")
    print(f"VRAM Available: {vram_size} MB ({available_vram:.2f} GB usable)")
    print(f"Calculated GPU Layers: {final_layers}")
    
    return final_layers, num_layers

def get_llm(model_type: str):
    from scripts.temporary import (
        quality_llm, fast_llm, QUALITY_MODEL_NAME, FAST_MODEL_NAME, 
        MODEL_FOLDER, N_CTX, N_GPU_LAYERS, N_BATCH, MMAP, MLOCK
    )
    from llama_cpp import Llama
    from pathlib import Path

    if model_type == "quality":
        if quality_llm is None and QUALITY_MODEL_NAME != "Select_a_model...":
            model_path = Path(MODEL_FOLDER) / QUALITY_MODEL_NAME
            if not model_path.exists():
                raise ValueError(f"Quality model file not found: {model_path}")
            quality_llm = Llama(
                model_path=str(model_path),
                n_ctx=N_CTX,
                n_gpu_layers=N_GPU_LAYERS,
                n_batch=N_BATCH,
                mmap=MMAP,
                mlock=MLOCK,
                verbose=False
            )
        return quality_llm
    elif model_type == "fast":
        if fast_llm is None and FAST_MODEL_NAME != "Select_a_model...":
            model_path = Path(MODEL_FOLDER) / FAST_MODEL_NAME
            if not model_path.exists():
                raise ValueError(f"Fast model file not found: {model_path}")
            fast_llm = Llama(
                model_path=str(model_path),
                n_ctx=N_CTX,
                n_gpu_layers=N_GPU_LAYERS,
                n_batch=N_BATCH,
                mmap=MMAP,
                mlock=MLOCK,
                verbose=False
            )
        return fast_llm
    elif model_type == "both":
        # Ensure both models are loaded for dual-model tasks
        quality = get_llm("quality")
        fast = get_llm("fast")
        if quality is None or fast is None:
            raise ValueError("Both models must be loaded for 'both' mode")
        return {"quality": quality, "fast": fast}
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

def update_generate_mode(quality_model, fast_model):
    quality_selected = quality_model != "Select_a_model..."
    fast_selected = fast_model != "Select_a_model..."
    if quality_selected and not fast_selected:
        options = ["Quality"]
        settings = get_model_settings(quality_model)
        if settings["category"] in ["uncensored", "general"] and "reasoning" in settings.get("enhancements", []):
            options.append("TOT")
        return gr.update(choices=options, value="Quality", interactive=len(options) > 1)
    elif fast_selected and not quality_selected:
        options = ["Fast"]
        settings = get_model_settings(fast_model)
        if settings["category"] in ["uncensored", "general"] and "reasoning" in settings.get("enhancements", []):
            options.append("TOT")
        return gr.update(choices=options, value="Fast", interactive=len(options) > 1)
    else:
        options = ["Both", "Quality", "Fast"]
        settings = get_model_settings(fast_model if fast_selected else quality_model)
        if fast_selected and settings["category"] in ["uncensored", "general"] and "reasoning" in settings.get("enhancements", []):
            options.append("TOT")
        return gr.update(choices=options, value="Both", interactive=True)

def get_model_type_for_task(task: str, mode: str):
    from scripts.temporary import QUALITY_MODEL_NAME, FAST_MODEL_NAME
    if QUALITY_MODEL_NAME != "Select_a_model..." and FAST_MODEL_NAME == "Select_a_model...":
        return "quality"
    elif QUALITY_MODEL_NAME == "Select_a_model..." and FAST_MODEL_NAME != "Select_a_model...":
        return "fast"
    elif QUALITY_MODEL_NAME != "Select_a_model..." and FAST_MODEL_NAME != "Select_a_model...":
        if task == "label":
            return "fast" if mode == "Both Models" else ("quality" if mode == "Quality ONLY" else "fast")
        elif task == "response":
            return "fast" if mode == "Fast ONLY" else "quality"
    else:
        raise ValueError("No model selected")

def get_model_settings(model_name):
    model_name_lower = model_name.lower()
    category = "general"  # Default to general
    enhancements = []

    # Detect primary category
    for cat, keywords in category_keywords.items():
        if any(keyword in model_name_lower for keyword in keywords):
            category = cat
            break

    # Detect reasoning enhancement
    if any(keyword in model_name_lower for keyword in reasoning_keywords):
        enhancements.append("reasoning")

    # Compile settings
    settings = {
        "category": category,
        "enhancements": enhancements,
        "temperature": temperature_defaults[category],
        "system_prompt": model_prompts[category],
        "prompt_template": prompt_templates[category]
    }
    return settings

def get_available_models():
    from .temporary import MODEL_FOLDER  # Changed from MODEL_DIR
    model_dir = Path(MODEL_FOLDER)
    return [f.name for f in model_dir.glob("*.gguf") if f.is_file()]

def unload_models():
    global quality_llm, fast_llm
    if quality_llm is not None:
        del quality_llm
        quality_llm = None
    if fast_llm is not None:
        del fast_llm
        fast_llm = None
    print("Models unloaded successfully.")

def get_streaming_response(prompt: str, model_type: str):
    from scripts.temporary import USE_PYTHON_BINDINGS, REPEAT_PENALTY, N_CTX, N_BATCH, MMAP, MLOCK, BACKEND_TYPE, LLAMA_CLI_PATH
    enhanced_prompt = context_injector.inject_context(prompt)
    settings = get_model_settings(QUALITY_MODEL_NAME if model_type == "quality" else FAST_MODEL_NAME)
    formatted_prompt = settings["prompt_template"].format(
        system_prompt=settings["system_prompt"],
        user_input=enhanced_prompt  # Changed prompt to user_input to match template
    )
    llm = get_llm(model_type)
    if USE_PYTHON_BINDINGS:
        if "reasoning" in settings["enhancements"]:
            yield "Thinking:"
            start_time = time.time()
            for i in range(5):
                time.sleep(0.5)
                yield f"Thinking:\n{'█' * (i + 1)}"
            elapsed_time = time.time() - start_time
            yield f"Thought for {elapsed_time:.1f}s."
        stream = llm.create_completion(
            prompt=formatted_prompt,
            temperature=settings["temperature"],  # Use category-specific temperature
            repeat_penalty=REPEAT_PENALTY,
            max_tokens=2048,
            stream=True
        )
        full_response = ""
        for output in stream:
            full_response += output["choices"][0]["text"]
            yield full_response
    else:
        cmd = [
            LLAMA_CLI_PATH,
            "-m", f"{MODEL_FOLDER}/{QUALITY_MODEL_NAME if model_type == 'quality' else FAST_MODEL_NAME}",
            "-p", formatted_prompt,
            "--temp", str(settings["temperature"]),  # Use category-specific temperature
            "--repeat-penalty", str(REPEAT_PENALTY),
            "--ctx-size", str(N_CTX),
            "--batch-size", str(N_BATCH),
            "--n-predict", "2048",
            "--log-disable"
        ]
        if MMAP:
            cmd += ["--mmap"]
        if MLOCK:
            cmd += ["--mlock"]
        if "vulkan" in BACKEND_TYPE.lower():
            cmd += ["--vulkan", "--gpu-layers", str(N_GPU_LAYERS)]
        elif "kompute" in BACKEND_TYPE.lower():
            cmd += ["--kompute", "--gpu-layers", str(N_GPU_LAYERS)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
        buffer = ""
        if current_model_settings["category"] == "reasoning":
            yield "Thinking:"
            start_time = time.time()
            for i in range(5):
                time.sleep(0.5)
                yield f"Thinking:\n{'█' * (i + 1)}"
            elapsed_time = time.time() - start_time
            yield f"Thought for {elapsed_time:.1f}s."
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            buffer += line
            yield buffer

def get_response(prompt: str, model_type: str, disable_think: bool = False, rp_settings: dict = None) -> str:
    from scripts.temporary import (
        USE_PYTHON_BINDINGS, REPEAT_PENALTY, N_CTX, N_BATCH, MMAP, MLOCK, 
        BACKEND_TYPE, LLAMA_CLI_PATH, MODEL_FOLDER, QUALITY_MODEL_NAME, 
        FAST_MODEL_NAME, prompt_templates, time, current_model_settings
    )
    from pathlib import Path

    # Enhance prompt with context
    enhanced_prompt = context_injector.inject_context(prompt)
    settings = get_model_settings(QUALITY_MODEL_NAME if model_type == "quality" else FAST_MODEL_NAME)
    mode = settings["category"]

    # Determine formatted prompt based on mode
    if mode == "nsfw" and rp_settings:  # RP/NSFW mode uses nsfw template with roleplay settings
        used_npcs = [npc for npc in [rp_settings.get("ai_npc1", ""), rp_settings.get("ai_npc2", ""), rp_settings.get("ai_npc3", "")] if npc and npc != "Unused"]
        formatted_prompt = prompt_templates["nsfw"].format(
            system_prompt=settings["system_prompt"],
            rp_location=rp_settings.get("rp_location", "Public"),
            user_name=rp_settings.get("user_name", "Human"),
            user_role=rp_settings.get("user_role", "Lead Roleplayer"),
            ai_npc1=used_npcs[0] if used_npcs else "Randomer",
            ai_npc2=used_npcs[1] if len(used_npcs) > 1 else "",
            ai_npc3=used_npcs[2] if len(used_npcs) > 2 else "",
            user_input=enhanced_prompt
        )
    else:
        # Use the mode-specific template or fallback to general
        formatted_prompt = prompt_templates.get(mode, prompt_templates["general"]).format(
            system_prompt=settings["system_prompt"],
            user_input=enhanced_prompt
        )

    llm = get_llm(model_type)
    if USE_PYTHON_BINDINGS:
        thinking_output = ""
        if "reasoning" in settings["enhancements"] and not disable_think:
            thinking_output = "Thinking:\n"
            start_time = time.time()
            for i in range(5):
                time.sleep(0.5)
                thinking_output += "█"
            elapsed_time = time.time() - start_time
            thinking_output += f"\nThought for {elapsed_time:.1f}s.\n"
        
        output = llm.create_completion(
            prompt=formatted_prompt,
            temperature=settings["temperature"],
            repeat_penalty=REPEAT_PENALTY,
            stop=["</s>", "USER:", "ASSISTANT:"],
            max_tokens=2048
        )
        response_text = output["choices"][0]["text"]
        return f"{thinking_output}{response_text}"
    else:
        cmd = [
            LLAMA_CLI_PATH,
            "-m", f"{MODEL_FOLDER}/{QUALITY_MODEL_NAME if model_type == 'quality' else FAST_MODEL_NAME}",
            "-p", formatted_prompt,
            "--temp", str(settings["temperature"]),
            "--repeat-penalty", str(REPEAT_PENALTY),
            "--ctx-size", str(N_CTX),
            "--batch-size", str(N_BATCH),
            "--n-predict", "2048",
            "--log-disable"
        ]
        if MMAP:
            cmd += ["--mmap"]
        if MLOCK:
            cmd += ["--mlock"]
        if "vulkan" in BACKEND_TYPE.lower():
            cmd += ["--vulkan", "--gpu-layers", str(N_GPU_LAYERS)]
        elif "kompute" in BACKEND_TYPE.lower():
            cmd += ["--kompute", "--gpu-layers", str(N_GPU_LAYERS)]
        
        proc = subprocess.run(cmd, capture_output=True, text=True)
        thinking_output = ""
        if "reasoning" in current_model_settings["category"] and not disable_think:
            thinking_output = "Thinking:\n"
            start_time = time.time()
            for i in range(5):
                time.sleep(0.5)
                thinking_output += "█"
            elapsed_time = time.time() - start_time
            thinking_output += f"\nThought for {elapsed_time:.1f}s.\n"
        return f"{thinking_output}{proc.stdout}"