# Script: `.\scripts\models.py`

# Imports...
import time
from llama_cpp import Llama
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from scripts.temporary import (
    N_CTX, N_GPU_LAYERS, USE_PYTHON_BINDINGS,
    LLAMA_CLI_PATH, BACKEND_TYPE, VRAM_SIZE,
    DYNAMIC_GPU_LAYERS, MMAP, MLOCK, current_model_settings,
    prompt_templates, llm, MODEL_NAME, MODEL_FOLDER
)

# Classes...
class ContextInjector:
    def __init__(self):
        self.vectorstores = {}
        self.current_vectorstore = None
        self.current_mode = None
        self.session_vectorstore = None  # Added for session-specific RAG
        self._load_default_vectorstores()

    def _load_default_vectorstores(self):
        modes = ["code", "rpg", "chat"]
        for mode in modes:
            vs_path = Path("data/vectors") / mode / "knowledge"
            if vs_path.exists():
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                self.vectorstores[mode] = FAISS.load_local(
                    str(vs_path),
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded {mode} vectorstore.")

    def set_mode(self, mode: str):
        valid_modes = ["code", "rpg", "chat"]
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
        vs_path = Path("data/vectors") / mode / "knowledge"
        if vs_path.exists():
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vectorstores[mode] = FAISS.load_local(
                str(vs_path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Loaded {mode} vectorstore.")
        else:
            print(f"Vectorstore not found for mode: {mode}")

    def set_session_vectorstore(self, vectorstore):
        self.session_vectorstore = vectorstore

    def inject_context(self, prompt: str) -> str:
        if self.current_vectorstore is None and self.session_vectorstore is None:
            return prompt
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = embedding_model.encode([prompt])[0]
        docs = []
        if self.current_vectorstore:
            docs.extend(self.current_vectorstore.similarity_search_by_vector(query_embedding, k=2))
        if self.session_vectorstore:
            docs.extend(self.session_vectorstore.similarity_search_by_vector(query_embedding, k=2))
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

def calculate_gpu_layers(models, available_vram):
    from math import floor
    total_size = sum(get_model_size(Path(MODEL_FOLDER) / model) for model in models if model != "Select_a_model...")
    if total_size == 0:
        return {model: 0 for model in models}
    vram_allocations = {model: (get_model_size(Path(MODEL_FOLDER) / model) / total_size) * available_vram for model in models if model != "Select_a_model..."}
    gpu_layers = {}
    for model in models:
        if model == "Select_a_model...":
            gpu_layers[model] = 0
            continue
        model_path = Path(MODEL_FOLDER) / model
        num_layers = get_model_layers(str(model_path))
        safe_size = get_model_size(str(model_path)) * 1.1875
        layer_size = safe_size / num_layers if num_layers > 0 else 0
        max_layers = floor(vram_allocations[model] / layer_size) if layer_size > 0 else 0
        gpu_layers[model] = min(max_layers, num_layers)
    return gpu_layers

def get_llm():
    from scripts.temporary import llm, MODEL_NAME, MODEL_FOLDER, N_CTX, N_GPU_LAYERS, N_BATCH, MMAP, MLOCK
    if MODEL_NAME != "Select_a_model...":
        if llm is None:
            model_path = Path(MODEL_FOLDER) / MODEL_NAME
            llm = Llama(
                model_path=str(model_path),
                n_ctx=N_CTX,
                n_gpu_layers=N_GPU_LAYERS,
                n_batch=N_BATCH,
                mmap=MMAP,
                mlock=MLOCK,
                verbose=False
            )
        return llm
    return None

def get_model_settings(model_name):
    from .temporary import category_keywords, handling_keywords, temperature_defaults
    model_name_lower = model_name.lower()
    category = "chat"  # Default to chat
    is_uncensored = any(keyword in model_name_lower for keyword in handling_keywords["uncensored"])
    is_reasoning = any(keyword in model_name_lower for keyword in handling_keywords["reasoning"])
    for cat, keywords in category_keywords.items():
        if any(keyword in model_name_lower for keyword in keywords):
            category = cat
            break
    settings = {
        "category": category,
        "is_uncensored": is_uncensored,
        "is_reasoning": is_reasoning,
        "temperature": temperature_defaults[category]
    }
    return settings

def determine_operation_mode(quality_model):
    if quality_model == "Select_a_model...":
        return "Select models to enable mode detection."
    settings = get_model_settings(quality_model)
    return settings["category"].capitalize()

def get_available_models():
    from .temporary import MODEL_FOLDER
    model_dir = Path(MODEL_FOLDER)
    return [f.name for f in model_dir.glob("*.gguf") if f.is_file()]

def inspect_model(model_name):
    from .temporary import MODEL_FOLDER
    from pathlib import Path
    
    if model_name == "Select_a_model...":
        return "Select a model to inspect."
    
    model_path = Path(MODEL_FOLDER) / model_name
    if not model_path.exists():
        return f"Model file not found: {model_path}"
    
    try:
        model_size_mb = get_model_size(str(model_path))
        num_layers = get_model_layers(str(model_path))
        settings = get_model_settings(model_name)
        model_type = settings["category"].capitalize()
        if num_layers > 0:
            model_size_gb = model_size_mb / 1024
            memory_per_layer_gb = (model_size_gb * 1.1875) / num_layers
            memory_per_layer_str = f"{memory_per_layer_gb:.3f} GB"
        else:
            memory_per_layer_str = "N/A"
        return f"Model: {model_name} | Type: {model_type} | Size: {model_size_mb:.2f} MB | Layers: {num_layers} | Memory/Layer: {memory_per_layer_str}"
    except Exception as e:
        return f"Error inspecting model: {str(e)}"

def unload_models():
    from scripts.temporary import llm
    if llm is not None:
        del llm
        llm = None
    print("Models unloaded successfully.")

def get_response(prompt: str, disable_think: bool = False, rp_settings: dict = None, session_history: str = "") -> str:
    from scripts.temporary import (
        USE_PYTHON_BINDINGS, REPEAT_PENALTY, N_CTX, N_BATCH, MMAP, MLOCK, 
        BACKEND_TYPE, LLAMA_CLI_PATH, MODEL_FOLDER, MODEL_NAME, prompt_templates, time
    )
    import subprocess

    enhanced_prompt = context_injector.inject_context(prompt)
    settings = get_model_settings(MODEL_NAME)
    mode = settings["category"]
    if mode == "rpg" and rp_settings:
        used_npcs = [npc for npc in [rp_settings.get("ai_npc1", ""), rp_settings.get("ai_npc2", ""), rp_settings.get("ai_npc3", "")] if npc and npc != "Unused"]
        num_npcs = max(1, len(used_npcs))
        template_key = f"rpg_{num_npcs}"
        formatted_prompt = prompt_templates[template_key].format(
            agent_name_1=used_npcs[0] if used_npcs else "Randomer",
            agent_name_2=used_npcs[1] if len(used_npcs) > 1 else "",
            agent_name_3=used_npcs[2] if len(used_npcs) > 2 else "",
            location_name=rp_settings.get("rp_location", "Public"),
            human_name=rp_settings.get("user_name", "Human"),
            human_role=rp_settings.get("user_role", "Lead Roleplayer"),
            session_history=session_history,
            human_input=enhanced_prompt
        )
    elif mode == "chat":
        template_key = "uncensored" if settings["is_uncensored"] else "chat"
        formatted_prompt = prompt_templates[template_key].format(user_input=enhanced_prompt)
    else:
        formatted_prompt = prompt_templates.get(mode, prompt_templates["chat"]).format(user_input=enhanced_prompt)
    llm = get_llm()
    if not llm:
        raise ValueError("No model loaded.")
    if USE_PYTHON_BINDINGS:
        thinking_output = ""
        if settings["is_reasoning"] and not disable_think:
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
            "-m", f"{MODEL_FOLDER}/{MODEL_NAME}",
            "-p", formatted_prompt,
            "--temp", str(settings["temperature"]),
            "--repeat-penalty", str(REPEAT_PENALTY),
            "--ctx-size", str(N_CTX),
            "--batch-size", str(N_BATCH),
            "--n-predict", "2048",
        ]
        if N_GPU_LAYERS > 0:
            cmd += ["--n-gpu-layers", str(N_GPU_LAYERS)]
        if MMAP:
            cmd += ["--mmap"]
        if MLOCK:
            cmd += ["--mlock"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        thinking_output = ""
        if settings["is_reasoning"] and not disable_think:
            thinking_output = "Thinking:\n"
            start_time = time.time()
            for i in range(5):
                time.sleep(0.5)
                thinking_output += "█"
            elapsed_time = time.time() - start_time
            thinking_output += f"\nThought for {elapsed_time:.1f}s.\n"
        return f"{thinking_output}{proc.stdout}"