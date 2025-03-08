# Script: `.\scripts\utility.py`

# Imports...
import re, subprocess, json, time, random, psutil, shutil  # Ensure shutil is included
from pathlib import Path
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from .models import context_injector, load_models
from .temporary import (
    TEMP_DIR, HISTORY_DIR, VECTORSTORE_DIR, SESSION_FILE_FORMAT,
    ALLOWED_EXTENSIONS, current_session_id, session_label, RAG_CHUNK_SIZE_DEVIDER,
    RAG_CHUNK_OVERLAP_DEVIDER, N_CTX, last_save_time
)
from . import temporary

# Functions...
def get_cpu_info():
    """
    Retrieve information about available CPUs and their cores.
    Returns a list of dictionaries with CPU labels and core ranges.
    """
    try:
        # Use wmic to get CPU names
        output = subprocess.check_output("wmic cpu get name", shell=True).decode()
        cpu_names = [line.strip() for line in output.split('\n') if line.strip() and 'Name' not in line]
        cpus = []
        for i, name in enumerate(cpu_names):
            cpus.append({
                "label": f"CPU {i}: {name}",
                "core_range": list(range(psutil.cpu_count(logical=True)))  # All logical cores
            })
        return cpus
    except Exception as e:
        print(f"Error getting CPU info: {e}")
        # Fallback to a default CPU entry
        return [{"label": "CPU 0", "core_range": list(range(psutil.cpu_count(logical=True)))}]
    
def get_available_gpus():
    """Detect available GPUs with fallback using dxdiag."""
    try:
        output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
        gpus = [line.strip() for line in output.split('\n') if line.strip() and 'Name' not in line]
        return gpus if gpus else ["CPU Only"]
    except Exception:
        try:
            # Fallback to dxdiag
            temp_file = Path(TEMP_DIR) / "dxdiag.txt"
            subprocess.run(f"dxdiag /t {temp_file}", shell=True, check=True)
            time.sleep(2)  # Wait for dxdiag to write
            with open(temp_file, 'r') as f:
                content = f.read()
                gpu = re.search(r"Card name: (.+)", content)
                return [gpu.group(1).strip()] if gpu else ["CPU Only"]
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

def save_session_history(history: list, loaded_files: list, force_save: bool = False) -> str:
    """Save chat history and attached files with time-based logic."""
    global last_save_time, current_session_id
    current_time = time.time()
    history_dir = Path(HISTORY_DIR)
    history_dir.mkdir(exist_ok=True)

    if not force_save and (current_time - last_save_time < 60):
        print("Session not saved yet (waiting for 60s interval).")
        return "Session not saved yet (waiting for interval)."

    if current_session_id is None:
        current_session_id = datetime.now().strftime(SESSION_FILE_FORMAT)
        print(f"Generated new session_id: {current_session_id}")
    file_path = history_dir / f"session_{current_session_id}.json"
    label = session_label if session_label else "Untitled"
    session_data = {
        "session_id": current_session_id,
        "label": label,
        "history": history,
        "attached_files": loaded_files
    }
    try:
        with open(file_path, "w") as f:
            json.dump(session_data, f, indent=2)
        manage_session_history()
        last_save_time = current_time
        print(f"Session saved to {file_path} with {len(loaded_files)} attached files.")
        return f"Session saved to {file_path}"
    except Exception as e:
        print(f"Error saving session: {str(e)}")
        return f"Error saving session: {str(e)}"
        
def load_session_history(file_path: str) -> tuple:
    with open(file_path, "r") as f:
        session_data = json.load(f)
    session_id = session_data.get("session_id", None)
    label = session_data.get("label", "Untitled")
    history = session_data.get("history", [])
    attached_files = session_data.get("attached_files", [])
    
    # Convert old tuple format to new dictionary format if necessary
    if history and isinstance(history[0], list) and len(history[0]) == 2:
        new_history = []
        for user_msg, ai_msg in history:
            new_history.append({'role': 'user', 'content': user_msg})
            new_history.append({'role': 'assistant', 'content': ai_msg})
        history = new_history
    
    print(f"Loaded session {session_id} from {file_path} with label '{label}' and {len(attached_files)} files.")
    return session_id, label, history, attached_files

def manage_session_history():
    """Limit saved sessions to MAX_HISTORY_SLOTS."""
    history_dir = Path(HISTORY_DIR)
    session_files = sorted(history_dir.glob("session_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    while len(session_files) > temporary.MAX_HISTORY_SLOTS:
        oldest_file = session_files.pop()
        oldest_file.unlink()
        print(f"Deleted oldest session: {oldest_file}")
        
def load_session_history(session_file):
    with open(session_file, 'r') as f:
        data = json.load(f)
    session_id = data.get('session_id', session_file.stem.replace('session_', ''))
    label = data.get('label', 'Untitled')
    history = data.get('history', [])
    attached_files = data.get('attached_files', [])
    return session_id, label, history, attached_files

def web_search(query: str, num_results: int = 3) -> str:
    """Perform a web search using DuckDuckGo without artificial delays."""
    wrapper = DuckDuckGoSearchAPIWrapper()
    try:
        results = wrapper.results(query, max_results=num_results)
        snippets = [f"{result.get('link', 'unknown')}:\n{result.get('snippet', 'No snippet available.')}" for result in results]
        return f"Results:\n{'nn'.join(snippets)}" if snippets else "No results found."
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

def create_session_vectorstore(loaded_files):
    """Create and save a session-specific FAISS vector store."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    if not loaded_files:
        print("No files provided for session vectorstore creation.")
        return None
    docs = load_and_chunk_documents(loaded_files)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        save_dir = Path("data/vectors/session") / f"session_{current_session_id}"
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_dir))
        print(f"Saved session vectorstore to {save_dir} with {len(docs)} documents.")
        return vectorstore
    except Exception as e:
        print(f"Error creating session vectorstore: {e}")
        return None

def delete_all_session_vectorstores() -> str:
    """Delete all session-specific vectorstore directories."""
    session_vs_dir = Path("data/vectors/session")
    if session_vs_dir.exists():
        import shutil
        for vs_dir in session_vs_dir.iterdir():
            if vs_dir.is_dir() and vs_dir.name.startswith("session_"):
                shutil.rmtree(vs_dir)
                print(f"Deleted session vectorstore: {vs_dir}")
        return "All session vectorstores deleted."
    else:
        return "No session vectorstores found to delete."

def delete_all_history_and_vectors():
    """
    Delete all history JSON files in HISTORY_DIR and all vectorstores in VECTORSTORE_DIR.
    Returns a status message indicating the result.
    """
    history_dir = Path(HISTORY_DIR)
    vectorstore_dir = Path(VECTORSTORE_DIR)
    
    # Delete all history JSON files
    for file in history_dir.glob('*.json'):
        try:
            file.unlink()
            print(f"Deleted history file: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    # Delete the entire vectorstore directory
    if vectorstore_dir.exists():
        try:
            shutil.rmtree(vectorstore_dir)
            print(f"Deleted vectorstore directory: {vectorstore_dir}")
        except Exception as e:
            print(f"Error deleting {vectorstore_dir}: {e}")
    
    return "All history and vectorstores deleted."

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
    config_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the 'data' directory exists
    config = {
        "model_settings": {
            "model_dir": temporary.MODEL_FOLDER,
            "model_name": temporary.MODEL_NAME,
            "n_ctx": temporary.N_CTX,
            "temperature": temporary.TEMPERATURE,
            "repeat_penalty": temporary.REPEAT_PENALTY,
            "use_python_bindings": temporary.USE_PYTHON_BINDINGS,
            "llama_cli_path": temporary.LLAMA_CLI_PATH,
            "vram_size": temporary.VRAM_SIZE,
            "selected_gpu": temporary.SELECTED_GPU,
            "selected_cpu": temporary.SELECTED_CPU,
            "mmap": temporary.MMAP,
            "mlock": temporary.MLOCK,
            "n_batch": temporary.N_BATCH,
            "dynamic_gpu_layers": temporary.DYNAMIC_GPU_LAYERS,
            "afterthought_time": temporary.AFTERTHOUGHT_TIME,
            "max_history_slots": temporary.MAX_HISTORY_SLOTS,
            "max_attach_slots": temporary.MAX_ATTACH_SLOTS,
            "session_log_height": temporary.SESSION_LOG_HEIGHT,
            "input_lines": temporary.INPUT_LINES
        },
        "backend_config": {
            "type": temporary.BACKEND_TYPE,
            "llama_bin_path": temporary.LLAMA_BIN_PATH
        },
        "rp_settings": {
            "rp_location": temporary.RP_LOCATION,
            "user_name": temporary.USER_PC_NAME,
            "user_role": temporary.USER_PC_ROLE,
            "ai_npc": temporary.AI_NPC_NAME,
            "ai_npc_role": temporary.AI_NPC_ROLE
        }
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    return "Settings saved to persistent.json"

def update_setting(key, value):
    """Update a setting and return components requiring reload if necessary."""
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
    elif key == "selected_cpu":
        temporary.SELECTED_CPU = value
    elif key == "repeat_penalty":
        temporary.REPEAT_PENALTY = float(value)
    elif key == "mlock":
        temporary.MLOCK = bool(value)
    elif key == "afterthought_time":
        temporary.AFTERTHOUGHT_TIME = bool(value)
    elif key == "n_batch":
        temporary.N_BATCH = int(value)
    elif key == "model_folder":
        temporary.MODEL_FOLDER = value
        reload_required = True
    elif key == "model_name":
        temporary.MODEL_NAME = value
        reload_required = True
    elif key == "max_history_slots":
        temporary.MAX_HISTORY_SLOTS = int(value)
    elif key == "max_attach_slots":
        temporary.MAX_ATTACH_SLOTS = int(value)
    elif key == "session_log_height":
        temporary.SESSION_LOG_HEIGHT = int(value)
    elif key == "input_lines":
        temporary.INPUT_LINES = int(value)
    # RPG settings
    elif key == "rp_location":
        temporary.RP_LOCATION = str(value)
    elif key == "user_name":
        temporary.USER_PC_NAME = str(value)
    elif key == "user_role":
        temporary.USER_PC_ROLE = str(value)
    elif key == "ai_npc":
        temporary.AI_NPC_NAME = str(value)

    if reload_required:
        return change_model(temporary.MODEL_NAME.split('/')[-1])  # Reload model if necessary
    return None, None
    
def load_config():
    config_path = Path("data/persistent.json")
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            # Model settings
            temporary.MODEL_FOLDER = config["model_settings"].get("model_dir", ".\models")
            temporary.MODEL_NAME = config["model_settings"].get("model_name", "Select_a_model...")
            temporary.N_CTX = int(config["model_settings"].get("n_ctx", 8192))
            temporary.TEMPERATURE = float(config["model_settings"].get("temperature", 0.5))
            temporary.REPEAT_PENALTY = float(config["model_settings"].get("repeat_penalty", 1.0))
            temporary.USE_PYTHON_BINDINGS = bool(config["model_settings"].get("use_python_bindings", True))
            temporary.LLAMA_CLI_PATH = config["model_settings"].get("llama_cli_path", "")
            temporary.VRAM_SIZE = int(config["model_settings"].get("vram_size", 8192))
            temporary.SELECTED_GPU = config["model_settings"].get("selected_gpu", None)
            temporary.SELECTED_CPU = config["model_settings"].get("selected_cpu", None)
            temporary.MMAP = bool(config["model_settings"].get("mmap", True))
            temporary.MLOCK = bool(config["model_settings"].get("mlock", True))
            temporary.AFTERTHOUGHT_TIME = bool(config["model_settings"].get("afterthought_time", True))
            temporary.N_BATCH = int(config["model_settings"].get("n_batch", 1024))
            temporary.DYNAMIC_GPU_LAYERS = bool(config["model_settings"].get("dynamic_gpu_layers", True))
            temporary.MAX_HISTORY_SLOTS = int(config["model_settings"].get("max_history_slots", 10))
            temporary.MAX_ATTACH_SLOTS = int(config["model_settings"].get("max_attach_slots", 6))
            temporary.SESSION_LOG_HEIGHT = int(config["model_settings"].get("session_log_height", 450))
            temporary.INPUT_LINES = int(config["model_settings"].get("input_lines", 5))
            # Backend config
            temporary.BACKEND_TYPE = config["backend_config"].get("type", "")
            temporary.LLAMA_BIN_PATH = config["backend_config"].get("llama_bin_path", "")
            # RP settings
            temporary.RP_LOCATION = config["rp_settings"].get("rp_location", "Public")
            temporary.USER_PC_NAME = config["rp_settings"].get("user_name", "Human")
            temporary.USER_PC_ROLE = config["rp_settings"].get("user_role", "Lead Roleplayer")
            temporary.AI_NPC_NAME = config["rp_settings"].get("ai_npc", "Robot")
            temporary.AI_NPC_ROLE = config["rp_settings"].get("ai_npc_role", "Randomers")
    else:
        # Default to ".\models" if no config exists
        temporary.MODEL_FOLDER = ".\models"
    
    # Resolve MODEL_FOLDER to an absolute path
    temporary.MODEL_FOLDER = str(Path(temporary.MODEL_FOLDER).resolve())      
