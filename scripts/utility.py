# Script: `.\scripts\utility.py`

import re
import subprocess
import json
from pathlib import Path
from datetime import datetime
from temporary import (
    TEMP_DIR, HISTORY_DIR, VECTORSTORE_DIR, SESSION_FILE_FORMAT,
    ALLOWED_EXTENSIONS, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, MAX_SESSIONS
)

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
    """Save chat history with a session label."""
    history_dir = Path(HISTORY_DIR)
    history_dir.mkdir(exist_ok=True)
    from temporary import session_label
    label = session_label if session_label else "Untitled"
    session_id = datetime.now().strftime(SESSION_FILE_FORMAT) + (f"_{label.replace(' ', '_')}" if label != "Untitled" else "")
    file_path = history_dir / f"session_{session_id}.json"
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

def load_and_chunk_documents(directory: Path) -> list:
    """Load and chunk documents for RAG."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.document_loaders import TextLoader
    documents = []
    try:
        for file in directory.iterdir():
            if file.suffix[1:].lower() in ALLOWED_EXTENSIONS:
                loader = TextLoader(str(file))
                docs = loader.load()
                splitter = RecursiveCharacterTextSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)
                chunks = splitter.split_documents(docs)
                documents.extend(chunks)
    except Exception as e:
        print(f"Error loading documents: {e}")
    return documents

def create_vectorstore(documents: list) -> None:
    """Create and save a FAISS vector store from documents."""
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.vectorstores import FAISS
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(documents, embeddings)
        vectorstore.save_local(str(VECTORSTORE_DIR / "general_knowledge"))
    except Exception as e:
        print(f"Error creating vectorstore: {e}")

# Moved Functions from interface.py and launcher.py
def save_config():
    """Save current settings to config.json, preserving existing keys."""
    from temporary import (
        MODEL_PATH, N_GPU_LAYERS, N_CTX, TEMPERATURE, USE_PYTHON_BINDINGS,
        LLAMA_CLI_PATH, VRAM_SIZE, SELECTED_GPU, DYNAMIC_GPU_LAYERS, MMAP,
        MLOCK, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_MAX_DOCS, MAX_SESSIONS,
        BACKEND_TYPE, LLAMA_BIN_PATH
    )
    from pathlib import Path
    import json

    config_path = Path("data/config.json")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = {}

    # Update model_settings
    config["model_settings"] = {
        "model_path": MODEL_PATH,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_ctx": N_CTX,
        "temperature": TEMPERATURE,
        "use_python_bindings": USE_PYTHON_BINDINGS,
        "llama_cli_path": LLAMA_CLI_PATH,
        "vram_size": VRAM_SIZE,
        "selected_gpu": SELECTED_GPU,
        "dynamic_gpu_layers": DYNAMIC_GPU_LAYERS,
        "mmap": MMAP,
        "mlock": MLOCK
    }

    # Update rag_settings
    config["rag_settings"] = {
        "chunk_size": RAG_CHUNK_SIZE,
        "chunk_overlap": RAG_CHUNK_OVERLAP,
        "max_docs": RAG_MAX_DOCS
    }

    # Update history_settings
    config["history_settings"] = {
        "max_sessions": MAX_SESSIONS
    }

    # Update backend_config (typically set by installer, but preserved here)
    config["backend_config"] = {
        "type": BACKEND_TYPE,
        "llama_bin_path": LLAMA_BIN_PATH
    }

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

def update_setting(key, value):
    """Update a setting and return components requiring reload if necessary."""
    from temporary import (
        MODEL_PATH, N_GPU_LAYERS, N_CTX, TEMPERATURE, VRAM_SIZE, SELECTED_GPU,
        MAX_SESSIONS, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_MAX_DOCS
    )
    from interface import change_model  # Avoid circular import

    reload_required = False
    if key == "temperature":
        globals()["TEMPERATURE"] = float(value)
    elif key == "n_ctx":
        globals()["N_CTX"] = int(value)
        reload_required = True
    elif key == "n_gpu_layers":
        globals()["N_GPU_LAYERS"] = int(value)
        reload_required = True
    elif key == "vram_size":
        globals()["VRAM_SIZE"] = int(value)
        reload_required = True
    elif key == "selected_gpu":
        globals()["SELECTED_GPU"] = value
    elif key == "max_sessions":
        globals()["MAX_SESSIONS"] = int(value)
    elif key == "rag_chunk_size":
        globals()["RAG_CHUNK_SIZE"] = int(value)
    elif key == "rag_chunk_overlap":
        globals()["RAG_CHUNK_OVERLAP"] = int(value)
    elif key == "rag_max_docs":
        globals()["RAG_MAX_DOCS"] = int(value)

    if reload_required:
        return change_model(MODEL_PATH.split('/')[-1])
    return None, None

def load_config():
    """Load configuration from config.json into temporary globals."""
    from temporary import (
        MODEL_PATH, N_GPU_LAYERS, N_CTX, TEMPERATURE, USE_PYTHON_BINDINGS,
        BACKEND_TYPE, LLAMA_CLI_PATH, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP,
        RAG_MAX_DOCS, VRAM_SIZE, SELECTED_GPU, DYNAMIC_GPU_LAYERS, MMAP,
        MLOCK, MAX_SESSIONS, LLAMA_BIN_PATH
    )
    from pathlib import Path
    import json

    config_path = Path(__file__).parent.parent / "data" / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            # Model Settings
            globals()["MODEL_PATH"] = config["model_settings"]["model_path"]
            globals()["N_GPU_LAYERS"] = config["model_settings"]["n_gpu_layers"]
            globals()["N_CTX"] = config["model_settings"]["n_ctx"]
            globals()["TEMPERATURE"] = config["model_settings"]["temperature"]
            globals()["USE_PYTHON_BINDINGS"] = config["model_settings"]["use_python_bindings"]
            globals()["LLAMA_CLI_PATH"] = config["model_settings"]["llama_cli_path"]
            globals()["VRAM_SIZE"] = config["model_settings"]["vram_size"]
            globals()["SELECTED_GPU"] = config["model_settings"]["selected_gpu"]
            globals()["DYNAMIC_GPU_LAYERS"] = config["model_settings"]["dynamic_gpu_layers"]
            globals()["MMAP"] = config["model_settings"].get("mmap", True)
            globals()["MLOCK"] = config["model_settings"].get("mlock", False)
            # RAG Settings
            globals()["RAG_CHUNK_SIZE"] = config["rag_settings"]["chunk_size"]
            globals()["RAG_CHUNK_OVERLAP"] = config["rag_settings"]["chunk_overlap"]
            globals()["RAG_MAX_DOCS"] = config["rag_settings"]["max_docs"]
            # History Settings
            globals()["MAX_SESSIONS"] = config.get("history_settings", {}).get("max_sessions", 10)
            # Backend Config
            globals()["BACKEND_TYPE"] = config["backend_config"]["type"]
            globals()["LLAMA_BIN_PATH"] = config["backend_config"]["llama_bin_path"]
    else:
        raise FileNotFoundError("Configuration file not found at data/config.json")