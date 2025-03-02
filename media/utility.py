# Script: `.\scripts\utility.py`

# Imports...
import re, subprocess, json, time, random
from pathlib import Path
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from .models import context_injector
from .temporary import (
    TEMP_DIR, HISTORY_DIR, VECTORSTORE_DIR, SESSION_FILE_FORMAT,
    ALLOWED_EXTENSIONS, current_session_id, session_label, RAG_CHUNK_SIZE_DEVIDER,
    RAG_CHUNK_OVERLAP_DEVIDER, N_CTX
)

# Functions...
def get_available_gpus():
    """Detect available GPUs on Windows for Vulkan/Kompute compatibility."""
    try:
        output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
        gpus = [line.strip() for line in output.split('\n') if line.strip() and 'Name' not in line]
        return gpus if gpus else ["CPU Only"]
    except Exception:
        return ["CPU Only"]

def process_file(file_path: Path) -> dict:
    """Process a file and return its metadata."""
    if file_path.suffix[1:].lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    return {
        "name": file_path.name,
        "content": file_path.read_text(encoding="utf-8"),
        "type": file_path.suffix[1:].lower()
    }

def strip_html(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r'<[^>]+>', '', text)

def save_session_history(history: list) -> str:
    """Save chat history using current_session_id, overwriting the existing file."""
    history_dir = Path(HISTORY_DIR)
    history_dir.mkdir(exist_ok=True)
    if current_session_id is None:
        current_session_id = datetime.now().strftime(SESSION_FILE_FORMAT)
    file_path = history_dir / f"session_{current_session_id}.json"
    label = session_label if session_label else "Untitled"
    session_data = {
        "label": label,
        "history": history
    }
    with open(file_path, "w") as f:
        json.dump(session_data, f)
    manage_session_history()
    return f"Session saved to {file_path}"

def manage_session_history():
    history_dir = Path(HISTORY_DIR)
    session_files = sorted(history_dir.glob("session_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    while len(session_files) > MAX_SESSIONS:
        oldest_file = session_files.pop()
        oldest_file.unlink()

def load_session_history(file_path: str) -> tuple:
    with open(file_path, "r") as f:
        session_data = json.load(f)
    return session_data["label"], session_data["history"]

def web_search(query: str, num_results: int = 3) -> str:
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    import time
    import random
    
    wrapper = DuckDuckGoSearchAPIWrapper()
    try:
        results = wrapper.results(query, max_results=num_results * 2)  # Fetch extra to filter
        snippets = []
        progress = "Web Search Progress:\n"
        
        for i, result in enumerate(results[:num_results]):
            url = result.get('link', 'unknown')
            snippet = result.get('snippet', 'No snippet available.')
            delay = random.uniform(2, 5)  # Random delay between 2-5 seconds
            progress += f"Connecting to {url} in {delay:.1f}s\n"
            time.sleep(delay)
            progress += "█\n"
            snippets.append(f"{url}:\n{snippet}")
        
        result_text = "\n\n".join(snippets)
        return f"{progress}Results:\n{result_text}" if snippets else f"{progress}No results found."
    except Exception as e:
        return f"Error during web search: {str(e)}"

def load_and_chunk_documents(file_paths: list) -> list:
    """Load and chunk documents from a list of file paths for RAG."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    from .temporary import N_CTX, RAG_CHUNK_SIZE_DEVIDER, RAG_CHUNK_OVERLAP_DEVIDER
    documents = []
    try:
        chunk_size = N_CTX // (RAG_CHUNK_SIZE_DEVIDER if RAG_CHUNK_SIZE_DEVIDER != 0 else 4)
        chunk_overlap = N_CTX // (RAG_CHUNK_OVERLAP_DEVIDER if RAG_CHUNK_OVERLAP_DEVIDER != 0 else 32)
        for file_path in file_paths:
            if Path(file_path).suffix[1:].lower() in ALLOWED_EXTENSIONS:
                loader = TextLoader(file_path)
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                chunks = splitter.split_documents(docs)
                documents.extend(chunks)
    except Exception as e:
        print(f"Error loading documents: {e}")
    return documents

def create_vectorstore(documents: list, mode: str) -> None:
    """Create and save a FAISS vector store from documents for a specific mode."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embeddings)
        save_dir = Path("data/vectors") / mode / "knowledge"
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_dir))
        print(f"Saved {mode} vectorstore to {save_dir}")
    except Exception as e:
        print(f"Error creating vectorstore for {mode}: {e}")

def create_session_vectorstore(loaded_files):
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    if not loaded_files:
        return None
    docs = load_and_chunk_documents(loaded_files)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error creating session vectorstore: {e}")
        return None

def delete_vectorstore(mode: str) -> str:
    """Delete the vectorstore directory for a specific mode."""
    vs_dir = Path("data/vectors") / mode
    if vs_dir.exists():
        import shutil
        shutil.rmtree(vs_dir)
        print(f"Deleted vectorstore directory: {vs_dir}")
        if mode in context_injector.vectorstores:  # Assuming context_injector is global
            del context_injector.vectorstores[mode]
            if context_injector.current_mode == mode:
                context_injector.current_vectorstore = None
                context_injector.current_mode = None
        return f"Deleted {mode} knowledge base."
    else:
        return f"No {mode} knowledge base found."


def delete_all_vectorstores() -> str:
    """Delete all mode-specific vectorstore directories."""
    modes = ["code", "rpg", "chat"]  # Updated to new modes
    deleted = []
    for mode in modes:
        vs_dir = Path("data/vectors") / mode
        if vs_dir.exists():
            import shutil
            shutil.rmtree(vs_dir)
            deleted.append(mode)
            if mode in context_injector.vectorstores:
                del context_injector.vectorstores[mode]
    if deleted:
        context_injector.current_vectorstore = None
        context_injector.current_mode = None
        return f"Deleted knowledge bases: {', '.join(deleted)}."
    else:
        return "No knowledge bases found to delete."

def get_saved_sessions():
    """Get list of saved session files sorted by modification time."""
    history_dir = Path(HISTORY_DIR)
    session_files = sorted(history_dir.glob("session_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return [f.name for f in session_files]

def trim_session_history(max_sessions):
    history_dir = Path(HISTORY_DIR)
    session_files = sorted(history_dir.glob("session_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    while len(session_files) > max_sessions:
        oldest_file = session_files.pop()
        oldest_file.unlink()

def load_models(quality_model, fast_model, vram_size):
    from scripts import temporary, models
    global quality_llm, fast_llm
    if quality_model == "Select_a_model...":
        return "Select a primary model to load.", gr.update(visible=False)
    models_to_load = [quality_model]
    if fast_model != "Select_a_model...":
        models_to_load.append(fast_model)
    gpu_layers = models.calculate_gpu_layers(models_to_load, vram_size)
    temporary.N_GPU_LAYERS_QUALITY = gpu_layers.get(quality_model, 0)
    temporary.N_GPU_LAYERS_FAST = gpu_layers.get(fast_model, 0)
    if quality_model != "Select_a_model...":
        model_path = Path(temporary.MODEL_FOLDER) / quality_model
        temporary.quality_llm = Llama(
            model_path=str(model_path),
            n_ctx=temporary.N_CTX,
            n_gpu_layers=temporary.N_GPU_LAYERS_QUALITY,
            n_batch=temporary.N_BATCH,
            mmap=temporary.MMAP,
            mlock=temporary.MLOCK,
            verbose=False
        )
    if fast_model != "Select_a_model...":
        model_path = Path(temporary.MODEL_FOLDER) / fast_model
        temporary.fast_llm = Llama(
            model_path=str(model_path),
            n_ctx=temporary.N_CTX,
            n_gpu_layers=temporary.N_GPU_LAYERS_FAST,
            n_batch=temporary.N_BATCH,
            mmap=temporary.MMAP,
            mlock=temporary.MLOCK,
            verbose=False
        )
    temporary.MODELS_LOADED = True
    status = f"Model(s) loaded, layer distribution: Primary VRAM={temporary.N_GPU_LAYERS_QUALITY}, Fast VRAM={temporary.N_GPU_LAYERS_FAST}"
    return status, gr.update(visible=True)

def save_config():
    config_path = Path("data/persistent.json")
    from scripts import temporary
    config = {
        "model_settings": {
            "model_dir": temporary.MODEL_FOLDER,
            "quality_model": temporary.QUALITY_MODEL_NAME,
            "fast_model": temporary.FAST_MODEL_NAME,
            "n_ctx": temporary.N_CTX,
            "repeat_penalty": temporary.REPEAT_PENALTY,
            "use_python_bindings": temporary.USE_PYTHON_BINDINGS,
            "llama_cli_path": temporary.LLAMA_CLI_PATH,
            "vram_size": temporary.VRAM_SIZE,
            "selected_gpu": temporary.SELECTED_GPU,
            "mmap": temporary.MMAP,
            "mlock": temporary.MLOCK,
            "n_batch": temporary.N_BATCH,
            "dynamic_gpu_layers": temporary.DYNAMIC_GPU_LAYERS
        },
        "backend_config": {
            "type": temporary.BACKEND_TYPE,
            "llama_bin_path": temporary.LLAMA_BIN_PATH
        },
        "rp_settings": {
            "rp_location": temporary.RP_LOCATION if hasattr(temporary, 'RP_LOCATION') else "Public",
            "user_name": temporary.USER_NAME if hasattr(temporary, 'USER_NAME') else "Human",
            "user_role": temporary.USER_ROLE if hasattr(temporary, 'USER_ROLE') else "Lead Roleplayer",
            "ai_npc1": temporary.AI_NPC1 if hasattr(temporary, 'AI_NPC1') else "Randomer",
            "ai_npc2": temporary.AI_NPC2 if hasattr(temporary, 'AI_NPC2') else "Unused",
            "ai_npc3": temporary.AI_NPC3 if hasattr(temporary, 'AI_NPC3') else "Unused"
        }
    }
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return "Settings saved to persistent.json"

def update_setting(key, value):
    """Update a setting and return components requiring reload if necessary."""
    from scripts import temporary
    from .interface import change_model

    reload_required = False
    # Model settings
    if key == "temperature":
        temporary.TEMPERATURE = float(value)
    elif key == "n_ctx":
        temporary.N_CTX = int(value)
        reload_required = True
    elif key == "n_gpu_layers":
        temporary.N_GPU_LAYERS = int(value)
        reload_required = True
    elif key == "vram_size":
        temporary.VRAM_SIZE = int(value)
        reload_required = True
    elif key == "selected_gpu":
        temporary.SELECTED_GPU = value
    elif key == "repeat_penalty":
        temporary.REPEAT_PENALTY = float(value)
    elif key == "mlock":
        temporary.MLOCK = bool(value)
    elif key == "n_batch":
        temporary.N_BATCH = int(value)
    elif key == "model_folder":
        temporary.MODEL_FOLDER = value
        reload_required = True
    # RPG settings
    elif key == "rp_location":
        temporary.RP_LOCATION = str(value)
    elif key == "user_name":
        temporary.USER_NAME = str(value)
    elif key == "user_role":
        temporary.USER_ROLE = str(value)
    elif key == "ai_npc1":
        temporary.AI_NPC1 = str(value)
    elif key == "ai_npc2":
        temporary.AI_NPC2 = str(value)
    elif key == "ai_npc3":
        temporary.AI_NPC3 = str(value)

    if reload_required:
        return change_model(temporary.MODEL_PATH.split('/')[-1])
    return None, None
    
def load_config():
    config_path = Path("data/persistent.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            from scripts import temporary
            # Model settings
            temporary.MODEL_FOLDER = config["model_settings"].get("model_dir", "models")
            temporary.QUALITY_MODEL_NAME = config["model_settings"].get("quality_model", "Select_a_model...")
            temporary.FAST_MODEL_NAME = config["model_settings"].get("fast_model", "Select_a_model...")
            temporary.N_CTX = int(config["model_settings"].get("n_ctx", 8192))
            temporary.REPEAT_PENALTY = float(config["model_settings"].get("repeat_penalty", 1.0))
            temporary.USE_PYTHON_BINDINGS = bool(config["model_settings"].get("use_python_bindings", False))
            temporary.LLAMA_CLI_PATH = config["model_settings"].get("llama_cli_path", "")
            temporary.VRAM_SIZE = int(config["model_settings"].get("vram_size", 8192))
            temporary.SELECTED_GPU = config["model_settings"].get("selected_gpu", None)
            temporary.MMAP = bool(config["model_settings"].get("mmap", True))
            temporary.MLOCK = bool(config["model_settings"].get("mlock", False))
            temporary.N_BATCH = int(config["model_settings"].get("n_batch", 1024))
            temporary.DYNAMIC_GPU_LAYERS = bool(config["model_settings"].get("dynamic_gpu_layers", True))
            # Backend config
            temporary.BACKEND_TYPE = config["backend_config"].get("type", "GPU/CPU - Vulkan")
            temporary.LLAMA_BIN_PATH = config["backend_config"].get("llama_bin_path", "")
            # RP settings
            temporary.RP_LOCATION = config["rp_settings"].get("rp_location", "Public")
            temporary.USER_NAME = config["rp_settings"].get("user_name", "Human")
            temporary.USER_ROLE = config["rp_settings"].get("user_role", "Lead Roleplayer")
            temporary.AI_NPC1 = config["rp_settings"].get("ai_npc1", "Randomer")
            temporary.AI_NPC2 = config["rp_settings"].get("ai_npc2", "Unused")
            temporary.AI_NPC3 = config["rp_settings"].get("ai_npc3", "Unused")    
            
            
