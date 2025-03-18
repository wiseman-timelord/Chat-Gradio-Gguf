# Script: `.\scripts\utility.py`

# Imports...
import re, subprocess, json, time, random, psutil, shutil, os, zipfile, yake
import win32com.client
import pythoncom  # Added for COM initialization
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
    ALLOWED_EXTENSIONS, current_session_id, session_label, RAG_CHUNK_SIZE_DEVIDER, BATCH_SIZE,
    RAG_CHUNK_OVERLAP_DEVIDER, CONTEXT_SIZE, last_save_time
)
from . import temporary
from scripts.models import clean_content, get_available_models

# Functions...
def filter_operational_content(text):
    """Remove operational tags and metadata from the text."""
    # Updated to handle <think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<answer>.*?</answer>', '', text, flags=re.DOTALL)
    return text.strip()

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
            
def generate_session_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def speak_text(text):
    """Read text aloud using PyWin32's text-to-speech functionality."""
    try:
        pythoncom.CoInitialize()  # Initialize COM for this thread
        speaker = win32com.client.Dispatch("SAPI.SpVoice")
        print(f"DEBUG: Attempting to speak: {text[:50]}..." if len(text) > 50 else f"DEBUG: Attempting to speak: {text}")
        speaker.Speak(text)
        print(f"DEBUG: Successfully spoke: {text[:50]}..." if len(text) > 50 else f"DEBUG: Successfully spoke: {text}")
    except Exception as e:
        print(f"Error speaking text: {str(e)}")
        raise  # Re-raise to catch in chat_interface
    finally:
        pythoncom.CoUninitialize()  # Clean up COM initialization
    
def save_session_history(session_log, attached_files, vector_files):
    if not temporary.current_session_id:
        temporary.current_session_id = generate_session_id()
    if not temporary.session_label and session_log:
        temporary.session_label = create_session_label(session_log[0]["content"] if session_log[0]["role"] == "user" else "")
    os.makedirs(HISTORY_DIR, exist_ok=True)
    session_file = Path(HISTORY_DIR) / f"session_{temporary.current_session_id}.json"
    session_data = {
        "session_id": temporary.current_session_id,
        "label": temporary.session_label,
        "history": session_log,
        "attached_files": attached_files,  # Store original file paths
        "vector_files": vector_files if vector_files else []
    }
    with open(session_file, "w") as f:
        json.dump(session_data, f)
    manage_session_history()
        

def load_session_history(session_file):
    try:
        with open(session_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading session file {session_file}: {e}")
        return None, "Error", [], [], []

    session_id = data.get("session_id", session_file.stem.replace('session_', ''))
    label = data.get("label", "Untitled")
    history = data.get("history", [])
    attached_files = data.get("attached_files", [])
    vector_files = data.get("vector_files", [])

    # Check and filter attached files
    attached_files = [file for file in attached_files if Path(file).exists()]
    if len(attached_files) != len(data.get("attached_files", [])):
        print(f"Removed missing attached files from session {session_id}")

    try:
        unzip_session_files(session_id)
        attach_dir = Path(TEMP_DIR) / f"session_{session_id}" / "attach"
        vector_dir = Path(TEMP_DIR) / f"session_{session_id}" / "vector"
        if attach_dir.exists():
            attached_files = [str(f) for f in attach_dir.glob("*") if f.is_file()]
        if vector_dir.exists():
            vector_files = [str(f) for f in vector_dir.glob("*") if f.is_file()]
        else:
            vector_files = []
            print("No VectorStore found for Session History slot.")  # Print message when no vectorstore exists
            # If no vector files but needed, a temporary vectorstore could be created here if required
            # For now, we leave it empty as per requirement to handle absence gracefully
    except Exception as e:
        print(f"Error unzipping session files for {session_id}: {e}")

    temporary.session_attached_files = attached_files
    temporary.session_vector_files = vector_files

    return session_id, label, history, attached_files, vector_files

def zip_session_files(session_id, attached_files, vector_files):
    temp_dir = Path(TEMP_DIR) / f"session_{session_id}"
    vector_dir = Path(VECTORSTORE_DIR)  # Change to VECTORSTORE_DIR
    os.makedirs(vector_dir, exist_ok=True)
    if temp_dir.exists():
        attach_zip = Path(TEMP_DIR) / f"session_{session_id}_attach.zip"
        vector_zip = vector_dir / f"session_{session_id}_vector.zip"  # Updated path
        if attached_files:
            with zipfile.ZipFile(attach_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in attached_files:
                    zf.write(file, Path(file).name)
        if vector_files:
            with zipfile.ZipFile(vector_zip, "w", zipfile.ZIP_DEFLATED) as zf:
                for file in vector_files:
                    zf.write(file, Path(file).name)
        shutil.rmtree(temp_dir, ignore_errors=True)

def unzip_session_files(session_id):
    temp_dir = Path(TEMP_DIR) / f"session_{session_id}"
    attach_zip = Path(TEMP_DIR) / f"session_{session_id}_attach.zip"
    vector_zip = Path(VECTORSTORE_DIR) / f"session_{session_id}_vector.zip"  # Updated path
    if attach_zip.exists():
        with zipfile.ZipFile(attach_zip, "r") as zf:
            zf.extractall(temp_dir / "attach")
    if vector_zip.exists():
        with zipfile.ZipFile(vector_zip, "r") as zf:
            zf.extractall(temp_dir / "vector")

def manage_session_history():
    """Limit saved sessions to MAX_HISTORY_SLOTS."""
    history_dir = Path(HISTORY_DIR)
    session_files = sorted(history_dir.glob("session_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    while len(session_files) > temporary.MAX_HISTORY_SLOTS:
        oldest_file = session_files.pop()
        oldest_file.unlink()
        print(f"Deleted oldest session: {oldest_file}")
        
def load_session_history(session_file):
    """
    Load session history from a JSON file, returning five values with defaults for missing keys.

    Args:
        session_file (Path): Path to the session JSON file.

    Returns:
        tuple: (session_id, label, history, attached_files, vector_files)
    """
    try:
        with open(session_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading session file {session_file}: {e}")
        return None, "Error", [], [], []  # Always return 5 values

    session_id = data.get("session_id", session_file.stem.replace('session_', ''))
    label = data.get("label", "Untitled")
    history = data.get("history", [])
    attached_files = data.get("attached_files", [])
    vector_files = data.get("vector_files", [])

    try:
        unzip_session_files(session_id)
        attach_dir = Path(TEMP_DIR) / f"session_{session_id}" / "attach"
        vector_dir = Path(TEMP_DIR) / f"session_{session_id}" / "vector"
        if attach_dir.exists():
            attached_files = [str(f) for f in attach_dir.glob("*") if f.is_file()]
        if vector_dir.exists():
            vector_files = [str(f) for f in vector_dir.glob("*") if f.is_file()]
        else:
            vector_files = []  # Empty list if no vector directory
    except Exception as e:
        print(f"Error unzipping session files for {session_id}: {e}")

    temporary.session_attached_files = attached_files
    temporary.session_vector_files = vector_files

    # Debugging print to confirm 5 values
    print(f"load_session_history returning: {session_id}, {label}, {len(history)}, {len(attached_files)}, {len(vector_files)}")
    return session_id, label, history, attached_files, vector_files

def process_uploaded_files(files):
    """Process uploaded files (if needed beyond copying)."""
    return [file.name for file in files] if files else []

def web_search(query: str, num_results: int = 3) -> str:
    """Perform a web search using DuckDuckGo and return formatted results.

    Args:
        query (str): The search query.
        num_results (int): Number of results to return. Defaults to 3.

    Returns:
        str: Formatted search results or an error message.
    """
    wrapper = DuckDuckGoSearchAPIWrapper()
    try:
        results = wrapper.results(query, max_results=num_results)
        if not results:
            return "No results found."
        snippets = [f"{r.get('link', 'unknown')}:\n{r.get('snippet', 'No snippet available.')}" 
                    for r in results]
        return f"Results:\n{'nn'.join(snippets)}"
    except Exception as e:
        return f"Error during web search: {str(e)}"

def summarize_document(file_path):
    """Summarize the contents of a document using YAKE, up to 100 characters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        kw_extractor = yake.KeywordExtractor(lan="en", n=4, dedupLim=0.9, top=1)
        keywords = kw_extractor.extract_keywords(content)
        summary = keywords[0][0] if keywords else "No summary available"
        return summary[:100]  # Truncate to 100 characters
    except Exception as e:
        print(f"Error summarizing document {file_path}: {e}")
        return "Error generating summary"

def get_attached_files_summary(attached_files):
    """Generate a list of summaries for attached files."""
    if not attached_files:
        return "No attached files to summarize."
    summary_list = []
    for file in attached_files:
        summary = summarize_document(file)
        summary_list.append(f"**{Path(file).name}** - {summary}")
    return "\n".join(summary_list)

def load_and_chunk_documents(file_paths: list) -> list:
    """Load and chunk documents from a list of file paths for RAG."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    from .temporary import CONTEXT_SIZE, RAG_CHUNK_SIZE_DEVIDER, RAG_CHUNK_OVERLAP_DEVIDER
    documents = []
    try:
        chunk_size = CONTEXT_SIZE // (RAG_CHUNK_SIZE_DEVIDER if RAG_CHUNK_SIZE_DEVIDER != 0 else 4)
        chunk_overlap = CONTEXT_SIZE // (RAG_CHUNK_OVERLAP_DEVIDER if RAG_CHUNK_OVERLAP_DEVIDER != 0 else 32)
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

def create_session_vectorstore(file_paths, session_id):
    if not file_paths:
        return None
    docs = load_and_chunk_documents(file_paths)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vectorstore = FAISS.from_documents(docs, embeddings)
        save_dir = Path(VECTORSTORE_DIR) / f"session_{session_id}"
        save_dir.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(save_dir))
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {e}")
        return None

def delete_all_session_vectorstores() -> str:
    """Delete all session-specific vectorstore directories."""
    session_vs_dir = Path("data/vectors")  # Updated path
    if session_vs_dir.exists():
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

def extract_links_with_descriptions(text):
    """Extract URLs from text and generate concise descriptions."""
    import re
    links = re.findall(r'(https?://\S+)', text)
    if not links:
        return ""
    descriptions = []
    for link in links:
        desc_prompt = f"Provide a one-sentence description for the following link: {link}"
        try:
            response = temporary.llm.create_chat_completion(
                messages=[{"role": "user", "content": desc_prompt}],
                max_tokens=50,
                temperature=0.5,
                stream=False
            )
            description = response['choices'][0]['message']['content'].strip()
            descriptions.append(f"{link}: {description}")
        except Exception as e:
            descriptions.append(f"{link}: Unable to generate description due to {str(e)}")
    return "\n".join(descriptions)

def get_saved_sessions():
    """Get list of saved session files sorted by modification time."""
    history_dir = Path(HISTORY_DIR)  # Updated to use HISTORY_DIR
    session_files = sorted(history_dir.glob("session_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return [f.name for f in session_files]

def update_setting(key, value):
    """Update a setting and return components requiring reload if necessary, with a confirmation message."""
    reload_required = False
    try:
        # Model settings
        if key == "temperature":
            temporary.TEMPERATURE = float(value)
        elif key == "context_size":
            temporary.CONTEXT_SIZE = int(value)
            reload_required = True
        elif key == "n_gpu_layers":
            temporary.GPU_LAYERS = int(value)
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
        elif key == "n_batch":
            temporary.BATCH_SIZE = int(value)
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

        if reload_required:
            reload_result = change_model(temporary.MODEL_NAME.split('/')[-1])
            message = f"Setting '{key}' updated to '{value}', model reload triggered."
            return message, *reload_result  # Unpack reload_result assuming it returns two values
        else:
            message = f"Setting '{key}' updated to '{value}'."
            return message, None, None
    except Exception as e:
        message = f"Error updating setting '{key}': {str(e)}"
        return message, None, None
    
def load_config():
    config_path = Path("data/persistent.json")
    try:
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                
                if "backend_config" not in config:
                    config["backend_config"] = {
                        "backend_type": "Not Configured",
                        "llama_bin_path": ""
                    }

                temporary.MODEL_FOLDER = config["model_settings"].get("model_dir", ".\models")
                temporary.AVAILABLE_MODELS = get_available_models()
                
                if "model_name" in config["model_settings"]:
                    temporary.MODEL_NAME = config["model_settings"]["model_name"]
                if "context_size" in config["model_settings"]:
                    temporary.CONTEXT_SIZE = int(config["model_settings"]["context_size"])
                if "temperature" in config["model_settings"]:
                    temporary.TEMPERATURE = float(config["model_settings"]["temperature"])
                if "repeat_penalty" in config["model_settings"]:
                    temporary.REPEAT_PENALTY = float(config["model_settings"]["repeat_penalty"])
                if "llama_cli_path" in config["model_settings"]:
                    temporary.LLAMA_CLI_PATH = config["model_settings"]["llama_cli_path"]
                if "vram_size" in config["model_settings"]:
                    temporary.VRAM_SIZE = int(config["model_settings"]["vram_size"])
                if "selected_gpu" in config["model_settings"]:
                    temporary.SELECTED_GPU = config["model_settings"]["selected_gpu"]
                if "selected_cpu" in config["model_settings"]:
                    temporary.SELECTED_CPU = config["model_settings"]["selected_cpu"]
                if "mmap" in config["model_settings"]:
                    temporary.MMAP = bool(config["model_settings"]["mmap"])
                if "mlock" in config["model_settings"]:
                    temporary.MLOCK = bool(config["model_settings"]["mlock"])
                if "n_batch" in config["model_settings"]:
                    temporary.BATCH_SIZE = int(config["model_settings"]["n_batch"])
                if "dynamic_gpu_layers" in config["model_settings"]:
                    temporary.DYNAMIC_GPU_LAYERS = bool(config["model_settings"]["dynamic_gpu_layers"])
                if "max_history_slots" in config["model_settings"]:
                    temporary.MAX_HISTORY_SLOTS = int(config["model_settings"]["max_history_slots"])
                if "max_attach_slots" in config["model_settings"]:
                    temporary.MAX_ATTACH_SLOTS = int(config["model_settings"]["max_attach_slots"])
                if "session_log_height" in config["model_settings"]:
                    temporary.SESSION_LOG_HEIGHT = int(config["model_settings"]["session_log_height"])
                if "input_lines" in config["model_settings"]:
                    temporary.INPUT_LINES = int(config["model_settings"]["input_lines"])
                
                if "backend_type" in config["backend_config"]:
                    temporary.BACKEND_TYPE = config["backend_config"]["backend_type"]
                if "llama_bin_path" in config["backend_config"]:
                    temporary.LLAMA_BIN_PATH = config["backend_config"]["llama_bin_path"]
                
                # Ensure loaded values are in allowed options
                if temporary.MAX_ATTACH_SLOTS not in temporary.ATTACH_SLOT_OPTIONS:
                    temporary.MAX_ATTACH_SLOTS = temporary.ATTACH_SLOT_OPTIONS[0]
                if temporary.INPUT_LINES not in temporary.INPUT_LINES_OPTIONS:
                    temporary.INPUT_LINES = temporary.INPUT_LINES_OPTIONS[0]
                if temporary.MAX_HISTORY_SLOTS not in temporary.HISTORY_SLOT_OPTIONS:
                    temporary.MAX_HISTORY_SLOTS = temporary.HISTORY_SLOT_OPTIONS[0]
                if temporary.SESSION_LOG_HEIGHT not in temporary.SESSION_LOG_HEIGHT_OPTIONS:
                    temporary.SESSION_LOG_HEIGHT = temporary.SESSION_LOG_HEIGHT_OPTIONS[0]
                
                if temporary.MODEL_NAME not in temporary.AVAILABLE_MODELS:
                    temporary.MODEL_NAME = "Browse_for_model_folder..." if not temporary.AVAILABLE_MODELS else temporary.AVAILABLE_MODELS[0]
                
                temporary.MODEL_FOLDER = str(Path(temporary.MODEL_FOLDER).resolve())
                return "Configuration loaded successfully."
        else:
            temporary.MODEL_FOLDER = str(Path(".\models").resolve())
            temporary.AVAILABLE_MODELS = get_available_models()
            return "Config file not found, using default settings from temporary.py."
    except Exception as e:
        temporary.MODEL_FOLDER = str(Path(temporary.MODEL_FOLDER if 'temporary.MODEL_FOLDER' in globals() else ".\models").resolve())
        temporary.AVAILABLE_MODELS = get_available_models()
        return f"Error loading configuration: {str(e)}"
    
def save_config():
    config_path = Path("data/persistent.json")
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "model_settings": {
                "model_dir": str(Path(temporary.MODEL_FOLDER).resolve()),
                "model_name": temporary.MODEL_NAME,
                "context_size": temporary.CONTEXT_SIZE,
                "temperature": temporary.TEMPERATURE,
                "repeat_penalty": temporary.REPEAT_PENALTY,
                "llama_cli_path": temporary.LLAMA_CLI_PATH,
                "vram_size": temporary.VRAM_SIZE,
                "selected_gpu": temporary.SELECTED_GPU,
                "selected_cpu": temporary.SELECTED_CPU,
                "mmap": temporary.MMAP,
                "mlock": temporary.MLOCK,
                "n_batch": temporary.BATCH_SIZE,
                "dynamic_gpu_layers": temporary.DYNAMIC_GPU_LAYERS,
                "max_history_slots": temporary.MAX_HISTORY_SLOTS,
                "max_attach_slots": temporary.MAX_ATTACH_SLOTS,
                "session_log_height": temporary.SESSION_LOG_HEIGHT,
                "input_lines": temporary.INPUT_LINES
            },
            "backend_config": {
                "backend_type": temporary.BACKEND_TYPE,
                "llama_bin_path": temporary.LLAMA_BIN_PATH
            }
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Saved MODEL_FOLDER: {temporary.MODEL_FOLDER}")
        print(f"Saved MODEL_NAME: {temporary.MODEL_NAME}")
        print(f"Saved CONTEXT_SIZE: {temporary.CONTEXT_SIZE}")
        print(f"Saved TEMPERATURE: {temporary.TEMPERATURE}")
        print(f"Saved REPEAT_PENALTY: {temporary.REPEAT_PENALTY}")
        print(f"Saved LLAMA_CLI_PATH: {temporary.LLAMA_CLI_PATH}")
        print(f"Saved VRAM_SIZE: {temporary.VRAM_SIZE}")
        print(f"Saved SELECTED_GPU: {temporary.SELECTED_GPU}")
        print(f"Saved SELECTED_CPU: {temporary.SELECTED_CPU}")
        print(f"Saved MMAP: {temporary.MMAP}")
        print(f"Saved MLOCK: {temporary.MLOCK}")
        print(f"Saved BATCH_SIZE: {temporary.BATCH_SIZE}")
        print(f"Saved DYNAMIC_GPU_LAYERS: {temporary.DYNAMIC_GPU_LAYERS}")
        print(f"Saved MAX_HISTORY_SLOTS: {temporary.MAX_HISTORY_SLOTS}")
        print(f"Saved MAX_ATTACH_SLOTS: {temporary.MAX_ATTACH_SLOTS}")
        print(f"Saved SESSION_LOG_HEIGHT: {temporary.SESSION_LOG_HEIGHT}")
        print(f"Saved INPUT_LINES: {temporary.INPUT_LINES}")
        print(f"Saved BACKEND_TYPE: {temporary.BACKEND_TYPE}")
        print(f"Saved LLAMA_BIN_PATH: {temporary.LLAMA_BIN_PATH}")
        print("Settings saved successfully to persistent.json")
        return "Settings saved successfully."
    except Exception as e:
        print(f"Error saving config: {str(e)}")
        return f"Error saving configuration: {str(e)}"