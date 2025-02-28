# Script: `.\scripts\utility.py`

# Imports...
import re, subprocess, json
from pathlib import Path
from datetime import datetime
from .temporary import (
    TEMP_DIR, HISTORY_DIR, VECTORSTORE_DIR, SESSION_FILE_FORMAT,
    ALLOWED_EXTENSIONS, current_session_id, session_label, RAG_CHUNK_SIZE_DEVIDER,
    RAG_CHUNK_OVERLAP_DEVIDER
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
    """Perform a web search using DuckDuckGo API."""
    try:
        import requests
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_redirect": 1}
        response = requests.get(url, params=params)
        data = response.json()
        results = data.get("RelatedTopics", [])[:num_results]
        result_text = "\n".join([r["Text"] for r in results if "Text" in r]) if results else "No results found."
        return result_text
    except Exception as e:
        return f"Error: {str(e)}"

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

def create_vectorstore(documents: list) -> None:
    """Create and save a FAISS vector store from documents."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embeddings)
        save_dir = Path(VECTORSTORE_DIR) / "general_knowledge"
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_dir))
    except Exception as e:
        print(f"Error creating vectorstore: {e}")

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

def save_config():
    config_path = Path("data/persistent.json")
    from scripts import temporary
    config = {
        "model_settings": {
            "model_dir": temporary.MODEL_FOLDER,
            "n_ctx": temporary.N_CTX,
            "temperature": temporary.TEMPERATURE,
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

    if reload_required:
        return change_model(temporary.MODEL_PATH.split('/')[-1])
    return None, None

def load_config():
    config_path = Path("data/persistent.json")
    with open(config_path) as f:
        config = json.load(f)
        # Model settings
        from scripts import temporary
        temporary.MODEL_FOLDER = config["model_settings"].get("model_dir", "models")
        temporary.N_CTX = int(config["model_settings"].get("n_ctx", 8192))
        temporary.TEMPERATURE = float(config["model_settings"].get("temperature", 0.75))
        temporary.REPEAT_PENALTY = float(config["model_settings"].get("repeat_penalty", 1.0))
        temporary.USE_PYTHON_BINDINGS = bool(config["model_settings"].get("use_python_bindings", False))
        temporary.LLAMA_CLI_PATH = config["model_settings"].get("llama_cli_path", "data/llama-vulkan-bin/llama-cli.exe")
        temporary.VRAM_SIZE = int(config["model_settings"].get("vram_size", 8192))
        temporary.SELECTED_GPU = config["model_settings"].get("selected_gpu", None)
        temporary.MMAP = bool(config["model_settings"].get("mmap", True))
        temporary.MLOCK = bool(config["model_settings"].get("mlock", False))
        temporary.N_BATCH = int(config["model_settings"].get("n_batch", 1024))
        temporary.DYNAMIC_GPU_LAYERS = bool(config["model_settings"].get("dynamic_gpu_layers", True))
        # Interface settings with backward compatibility
        if "interface_settings" in config:
            temporary.RAG_MAX_DOCS = int(config["interface_settings"].get("max_docs", 6))
            temporary.MAX_SESSIONS = int(config["interface_settings"].get("max_sessions", 7))
        else:
        # Backend config
            temporary.BACKEND_TYPE = config["backend_config"].get("type", "GPU/CPU - Vulkan")
            temporary.LLAMA_BIN_PATH = config["backend_config"].get("llama_bin_path", "data/llama-vulkan-bin")
