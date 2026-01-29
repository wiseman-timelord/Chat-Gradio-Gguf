# Script: `.\scripts\utility.py`

# Imports
import tempfile
import re, subprocess, json, time, random, psutil, shutil, os, zipfile, spacy, sys, PyPDF2
from pathlib import Path
from datetime import datetime
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation
import gradio as gr
from scripts.inference import load_inference, clean_content
import scripts.configuration as cfg
from scripts.configuration import (
    TEMP_DIR, HISTORY_DIR, SESSION_FILE_FORMAT, ALLOWED_EXTENSIONS, 
    current_session_id, session_label
)

# Import search functions from tools module
# NOTE: hybrid_search is DDG-based, web_search is comprehensive multi-source
from scripts.tools import (
    hybrid_search, format_search_status_for_chat,
    web_search, format_web_search_status_for_chat
)

# Variables
_nlp_model = None

# NOTE: Platform-specific imports (win32com, pythoncom, pyttsx3) are imported
# lazily inside the functions that need them. This is because cfg.PLATFORM
# is not set until after the launcher parses command-line arguments.


# =============================================================================
# BEEP FUNCTION (simple utility, doesn't belong in tools.py)
# =============================================================================

def beep() -> None:
    """Play a notification beep if enabled."""
    # Fixed: was referencing 'temporary', changed to cfg
    if not getattr(cfg, "BLEEP_ON_EVENTS", False):
        return
    if cfg.PLATFORM == "windows":
        _beep_windows()
    elif cfg.PLATFORM == "linux":
        _beep_linux()


def _beep_windows() -> None:
    try:
        import winsound
        winsound.Beep(1000, 150)
    except:
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_OK)
        except:
            pass


def _beep_linux() -> None:
    methods = [
        lambda: subprocess.run(['beep', '-f', '1000', '-l', '150'], timeout=2, check=True, 
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL),
        lambda: subprocess.run(['paplay', '/usr/share/sounds/freedesktop/stereo/complete.oga'], 
                              timeout=2, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL) 
                if os.path.exists('/usr/share/sounds/freedesktop/stereo/complete.oga') else (_ for _ in ()).throw(Exception()),
        lambda: subprocess.run(['play', '-n', 'synth', '0.15', 'sin', '1000'], timeout=2, check=True,
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL),
        lambda: print("\a", end="", flush=True),
    ]
    for method in methods:
        try:
            method()
            return
        except:
            continue


# =============================================================================
# GENERAL UTILITY FUNCTIONS
# =============================================================================

def has_vulkan_binary():
    """Check if Vulkan binary is available (VULKAN_CPU or VULKAN_VULKAN modes)"""
    return cfg.BACKEND_TYPE in ["VULKAN_CPU", "VULKAN_VULKAN"]

def has_vulkan_wheel():
    """Check if Python wheel has Vulkan support (VULKAN_VULKAN mode only)"""
    return cfg.BACKEND_TYPE == "VULKAN_VULKAN"

def is_cpu_only():
    """Check if running in pure CPU mode (CPU_CPU mode)"""
    return cfg.BACKEND_TYPE == "CPU_CPU"

def short_path(path_str, max_len=44):
    """Truncate path to last max_len chars with ... prefix"""
    path = str(path_str)
    if len(path) <= max_len:
        return path
    return "..." + path[-max_len:]

def filter_operational_content(text):
    """Remove operational tags, metadata, and AI prefixes from the text."""
    # Remove AI-Chat: prefix patterns
    text = re.sub(r'^AI-Chat:\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\nAI-Chat:\s*\n?', '\n', text)
    
    # Remove thinking/answer tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<answer>.*?</answer>', '', text, flags=re.DOTALL)
    
    # Remove llama.cpp operational output patterns
    patterns = [
        r"ggml_vulkan:.*",
        r"load_tensors:.*",
        r"main:.*",
        r"Error executing CLI:.*",
        r"CLI Error:.*",
        r"build:.*",
        r"llama_model_load.*",
        r"print_info:.*",
        r"load:.*",
        r"llama_init_from_model:.*",
        r"llama_kv_cache_init:.*",
        r"sampler.*",
        r"eval:.*",
        r"embd_inp.size.*",
        r"waiting for user input",
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    
    return text.strip()


# =============================================================================
# CPU/GPU DETECTION
# =============================================================================

def detect_cpu_config():
    """Detect CPU configuration and set thread options."""
    try:
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False) or 1,
            "logical_cores": psutil.cpu_count(logical=True) or 1
        }
        
        cfg.CPU_PHYSICAL_CORES = cpu_info["physical_cores"]
        cfg.CPU_LOGICAL_CORES = cpu_info["logical_cores"]
        
        max_threads = cfg.CPU_LOGICAL_CORES
        cfg.CPU_THREAD_OPTIONS = list(range(1, max_threads + 1))
        
        if cfg.CPU_THREADS is None or cfg.CPU_THREADS > max_threads:
            cfg.CPU_THREADS = max(1, max_threads // 2)
        
        if "vulkan" in cfg.BACKEND_TYPE.lower():
            cfg.CPU_THREADS = max(2, cfg.CPU_THREADS)
        
        print(f"[CPU] Detected: {cfg.CPU_PHYSICAL_CORES} cores, "
              f"{cfg.CPU_LOGICAL_CORES} threads")
        print(f"[CPU] Current: {cfg.CPU_THREADS}")
        
    except Exception as e:
        cfg.set_status("CPU fallback", console=True)
        print(f"[CPU] Detection error: {e}")
        cfg.CPU_PHYSICAL_CORES = 4
        cfg.CPU_LOGICAL_CORES = 8
        cfg.CPU_THREAD_OPTIONS = list(range(1, 9))
        cfg.CPU_THREADS = 4

def get_available_gpus_windows():
    """Retrieve available GPUs on Windows using multiple methods."""
    try:
        output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
        gpus = [line.strip() for line in output.split('\n') if line.strip() and 'Name' not in line]
        if gpus:
            return gpus
    except:
        pass
    
    try:
        temp_file = Path(cfg.TEMP_DIR) / "dxdiag.txt"
        subprocess.run(f"dxdiag /t {temp_file}", shell=True, check=True)
        time.sleep(2)
        with open(temp_file, 'r') as f:
            content = f.read()
            gpu = re.search(r"Card name: (.+)", content)
            if gpu:
                return [gpu.group(1).strip()]
    except:
        pass
    
    return ["CPU Only"]

def calculate_optimal_gpu_layers(model_path, vram_mb, context_size):
    """Calculate optimal number of GPU layers based on VRAM and model size."""
    try:
        from pathlib import Path
        
        if not Path(model_path).exists():
            print(f"[GPU-CALC] Model not found: {model_path}")
            return 0
        
        model_size_bytes = Path(model_path).stat().st_size
        model_size_mb = model_size_bytes / (1024 * 1024)
        context_mb = context_size / 1024
        context_overhead = context_mb * 2
        available_vram = max(0, vram_mb - context_overhead - 200)
        
        quant_match = re.search(r'[qQ](\d+)', Path(model_path).name)
        if quant_match:
            quant = int(quant_match.group(1))
            bits_per_weight = quant
        else:
            bits_per_weight = 4
        
        mb_per_layer = model_size_mb / 80
        estimated_layers = int(available_vram / mb_per_layer) if mb_per_layer > 0 else 0
        
        return max(0, min(estimated_layers, 100))
        
    except Exception as e:
        print(f"[GPU-CALC] Error: {e}")
        return 0

def get_available_gpus():
    """Get list of available GPUs based on platform."""
    if cfg.PLATFORM == "windows":
        return get_available_gpus_windows()
    else:
        return get_available_gpus_linux()

def get_available_gpus_linux():
    """Retrieve available GPUs on Linux."""
    gpus = []
    
    try:
        output = subprocess.check_output("lspci | grep -i vga", shell=True).decode()
        for line in output.strip().split('\n'):
            if line:
                parts = line.split(': ')
                if len(parts) > 1:
                    gpus.append(parts[1].strip()[:50])
    except:
        pass
    
    if not gpus:
        try:
            output = subprocess.check_output("vulkaninfo --summary 2>/dev/null | grep deviceName", shell=True).decode()
            for line in output.strip().split('\n'):
                if 'deviceName' in line:
                    name = line.split('=')[1].strip() if '=' in line else line.split(':')[1].strip()
                    gpus.append(name[:50])
        except:
            pass
    
    return gpus if gpus else ["CPU Only"]

def get_cpu_info():
    """Get CPU information for display."""
    try:
        cpu_info = []
        
        if cfg.PLATFORM == "windows":
            try:
                output = subprocess.check_output("wmic cpu get name", shell=True).decode()
                for line in output.split('\n'):
                    line = line.strip()
                    if line and 'Name' not in line:
                        cpu_info.append({
                            "label": line[:50],
                            "cores": cfg.CPU_PHYSICAL_CORES,
                            "threads": cfg.CPU_LOGICAL_CORES
                        })
            except:
                pass
        else:
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    content = f.read()
                model_match = re.search(r'model name\s*:\s*(.+)', content)
                if model_match:
                    cpu_info.append({
                        "label": model_match.group(1).strip()[:50],
                        "cores": cfg.CPU_PHYSICAL_CORES,
                        "threads": cfg.CPU_LOGICAL_CORES
                    })
            except:
                pass
        
        if not cpu_info:
            cpu_info.append({
                "label": f"CPU ({cfg.CPU_PHYSICAL_CORES}c/{cfg.CPU_LOGICAL_CORES}t)",
                "cores": cfg.CPU_PHYSICAL_CORES,
                "threads": cfg.CPU_LOGICAL_CORES
            })
        
        return cpu_info
        
    except Exception as e:
        print(f"[CPU-INFO] Error: {e}")
        return [{"label": "Default CPU", "cores": 4, "threads": 8}]


# =============================================================================
# SESSION MANAGEMENT
# =============================================================================

def get_nlp_model():
    """Get or initialize the spaCy NLP model."""
    global _nlp_model
    if _nlp_model is None:
        if spacy is None:
            return None  # Graceful fallback if spacy not installed
        try:
            _nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            try:
                subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=True)
                _nlp_model = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"[NLP] Failed to load spaCy model: {e}")
                return None
    return _nlp_model

def summarize_session(messages):
    """Generate a short label for a session based on initial messages."""
    if not messages:
        return "Empty Session"
    
    first_user_msg = None
    for msg in messages:
        if msg.get('role') == 'user':
            first_user_msg = msg.get('content', '')
            break
    
    if not first_user_msg:
        return "Session " + datetime.now().strftime("%Y%m%d_%H%M")
    
    text = first_user_msg[:500]
    
    nlp = get_nlp_model()
    if nlp:
        try:
            doc = nlp(text)
            candidates = []
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'EVENT', 'WORK_OF_ART']:
                    candidates.append(ent.text)
            for chunk in doc.noun_chunks:
                if chunk.root.pos_ in ['NOUN', 'PROPN']:
                    candidates.append(chunk.text)
            
            if candidates:
                label = candidates[0][:40]
                return label.strip()
        except:
            pass
    
    words = text.split()[:6]
    return ' '.join(words)[:40] if words else "Session"

def get_saved_sessions():
    """Get list of saved session files, sorted by date (newest first)."""
    history_path = Path(HISTORY_DIR)
    if not history_path.exists():
        return []
    
    sessions = []
    for file in history_path.glob("*.json"):
        try:
            mtime = file.stat().st_mtime
            sessions.append((mtime, file.name))
        except:
            continue
    
    sessions.sort(reverse=True)
    return [s[1] for s in sessions[:cfg.MAX_HISTORY_SLOTS]]

def save_session_history(messages, attached_files=None):
    """Save current session to history."""
    if not messages:
        return
    
    session_id = cfg.current_session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    label = cfg.session_label or summarize_session(messages)
    
    cfg.current_session_id = session_id
    cfg.session_label = label
    
    filename = f"{session_id}.json"
    filepath = Path(HISTORY_DIR) / filename
    
    session_data = {
        "session_id": session_id,
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "messages": messages,
        "attached_files": attached_files or []
    }
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        print(f"[SESSION] Saved: {label} ({filename})")
    except Exception as e:
        print(f"[SESSION] Save error: {e}")

def load_session_history(filename):
    """Load a session from history file."""
    filepath = Path(HISTORY_DIR) / filename
    
    if not filepath.exists():
        return None, None, [], []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        session_id = data.get("session_id", "")
        label = data.get("label", "Unnamed Session")
        messages = data.get("messages", [])
        attached_files = data.get("attached_files", [])
        
        return session_id, label, messages, attached_files
        
    except Exception as e:
        print(f"[SESSION] Load error: {e}")
        return None, None, [], []


# =============================================================================
# FILE HANDLING
# =============================================================================

def read_file_content(file_path):
    """Read content from various file types."""
    try:
        path = Path(file_path)
        suffix = path.suffix.lower()
        
        if suffix in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.xml', '.csv']:
            encodings_to_try = ['utf-8', 'cp1252', 'iso-8859-1', 'latin-1']
            content = None
            for encoding in encodings_to_try:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is not None:
                return content, "text", True, None
            else:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read(), "text", True, None
        
        elif suffix == '.pdf':
            reader = PyPDF2.PdfReader(str(path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text, "text", True, None
        
        elif suffix == '.docx':
            doc = Document(str(path))
            text = "\n".join(para.text for para in doc.paragraphs)
            return text, "text", True, None
        
        elif suffix == '.xlsx':
            wb = load_workbook(str(path), data_only=True)
            text_parts = []
            for sheet in wb.worksheets:
                for row in sheet.iter_rows(values_only=True):
                    row_text = "\t".join(str(cell) if cell is not None else "" for cell in row)
                    if row_text.strip():
                        text_parts.append(row_text)
            return "\n".join(text_parts), "text", True, None
        
        elif suffix == '.pptx':
            prs = Presentation(str(path))
            text_parts = []
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text_parts.append(shape.text)
            return "\n".join(text_parts), "text", True, None
        
        elif suffix in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']:
            import base64
            with open(path, 'rb') as f:
                data = base64.b64encode(f.read()).decode('utf-8')
            mime_type = {
                '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                '.gif': 'image/gif', '.bmp': 'image/bmp', '.webp': 'image/webp'
            }.get(suffix, 'image/png')
            return f"data:{mime_type};base64,{data}", "image", True, None
        
        else:
            return None, None, False, f"Unsupported file type: {suffix}"
            
    except Exception as e:
        return None, None, False, str(e)

def summarize_document(file_path):
    """Summarize the contents of a document using spaCy, up to 100 characters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        nlp = get_nlp_model()
        if nlp is None:
            words = content.split()[:10]
            return ' '.join(words)[:100] if words else "No summary available"
        
        doc = nlp(content[:2000])
        
        candidates = []
        for ent in doc.ents:
            candidates.append(ent.text)
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in ['NOUN', 'PROPN']:
                candidates.append(chunk.text)
        
        if candidates:
            summary = candidates[0]
        else:
            sentences = [sent.text.strip() for sent in doc.sents]
            summary = sentences[0] if sentences else "No summary available"
        
        return summary[:100]
        
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

def eject_file(file_list, slot_index, is_attach=True):
    """Eject a file from the specified slot."""
    if 0 <= slot_index < len(file_list):
        removed_file = file_list.pop(slot_index)
        # Update global state
        if is_attach:
            cfg.session_attached_files = file_list
        status_msg = f"Ejected {Path(removed_file).name}"
        print(f"[FILES] {status_msg}")
    else:
        status_msg = "No file to eject"
    
    # Return updated file list and status, then UI updates happen via .then()
    return file_list, status_msg
    
def update_file_slot_ui(file_list, is_attach=True):
    """Update file slot UI components."""
    import gradio as gr
    import scripts.configuration as cfg
    from pathlib import Path
    
    max_slots = cfg.MAX_POSSIBLE_ATTACH_SLOTS
    current_max = cfg.MAX_ATTACH_SLOTS
    button_updates = []
    
    for i in range(max_slots):
        # Hide slots beyond current MAX_ATTACH_SLOTS setting
        if i >= current_max:
            button_updates.append(gr.update(value="", visible=False, variant="primary"))
        elif i < len(file_list):
            filename = Path(file_list[i]).name
            short_name = filename[:36] + ".." if len(filename) > 38 else filename
            button_updates.append(gr.update(value=short_name, visible=True, variant="primary"))
        else:
            button_updates.append(gr.update(value="", visible=False, variant="primary"))
    
    show_upload = len(file_list) < current_max if is_attach else True
    button_updates.append(gr.update(visible=show_upload))
    
    return button_updates

def load_and_chunk_documents(file_paths: list) -> list:
    """Load and chunk documents from a list of file paths for RAG."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    from scripts.configuration import CONTEXT_SIZE, RAG_CHUNK_SIZE_DIVIDER, RAG_CHUNK_OVERLAP_DIVIDER
    documents = []
    try:
        chunk_size = CONTEXT_SIZE // (RAG_CHUNK_SIZE_DIVIDER if RAG_CHUNK_SIZE_DIVIDER != 0 else 4)
        chunk_overlap = CONTEXT_SIZE // (RAG_CHUNK_OVERLAP_DIVIDER if RAG_CHUNK_OVERLAP_DIVIDER != 0 else 32)
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

def delete_all_session_histories():
    """Delete all history JSON files in HISTORY_DIR."""
    history_dir = Path(HISTORY_DIR)
    for file in history_dir.glob('*.json'):
        try:
            file.unlink()
            print(f"Deleted history file: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    return "All session histories deleted."

def create_session_vectorstore(file_paths):
    """Thin wrapper so display.py can call the injector transparently."""
    from scripts.configuration import context_injector
    context_injector.set_session_vectorstore(file_paths)

def process_attach_files(files, attached_files):
    """Process uploaded files for attachment."""
    if not files:
        return "No files selected.", attached_files
    status, updated_files = process_files(files, attached_files, cfg.MAX_ATTACH_SLOTS, is_attach=True)
    # Update global state
    cfg.session_attached_files = updated_files
    return status, updated_files

def process_files(files, existing_files, max_files, is_attach=True):
    """Process uploaded files for attach or vector, ensuring no duplicates and respecting max file limits."""
    import gradio as gr
    
    print(f"[FILES] Raw input type: {type(files)}, value: {files}")
    
    if not files:
        return "No files uploaded.", existing_files

    # Handle Gradio file upload format - files can be list of file objects or dicts
    normalized_files = []
    if isinstance(files, (list, tuple)):
        for f in files:
            if f is None:
                continue
            # Handle Gradio file object (has .name attribute) or dict
            if hasattr(f, 'name'):
                file_path = f.name
            elif isinstance(f, (str, Path)):
                file_path = str(f)
            elif isinstance(f, dict):
                # Gradio sometimes returns dict with 'name' or 'path' key
                file_path = f.get('name') or f.get('path') or f.get('data')
            else:
                print(f"[FILES] Skipping unknown file format: {type(f)} - {f}")
                continue
            
            if file_path and os.path.isfile(file_path):
                normalized_files.append(str(file_path))
                print(f"[FILES] Added file: {file_path}")
            else:
                print(f"[FILES] File not found or invalid: {file_path}")
    elif isinstance(files, str):
        # Single file path as string
        if os.path.isfile(files):
            normalized_files.append(files)
    elif hasattr(files, 'name'):
        # Single file object
        normalized_files.append(files.name)
    
    print(f"[FILES] Normalized {len(normalized_files)} files")

    if not normalized_files:
        return "No valid files to add.", existing_files

    # Remove duplicates by filename (keep existing, replace with new)
    new_files = [f for f in normalized_files if f not in existing_files]
    if not new_files:
        return "No new files to add.", existing_files

    # Remove any existing files with same filename as new ones
    for f in new_files:
        file_name = Path(f).name
        existing_files = [ef for ef in existing_files if Path(ef).name != file_name]

    available_slots = max_files - len(existing_files)
    processed_files = new_files[:available_slots]
    updated_files = processed_files + existing_files

    # Update global state
    if is_attach:
        cfg.session_attached_files = updated_files
    else:
        # Vector files use different storage if needed
        cfg.session_attached_files = updated_files  # Use same for now

    status = f"Attached {len(processed_files)} file(s)."
    print(f"[FILES] Status: {status}, Total files: {len(updated_files)}")
    return status, updated_files


# =============================================================================
# RESEARCH CAPABILITY CHECK (for UI visibility)
# =============================================================================

def is_research_available() -> bool:
    """Check if hybrid search capabilities are available (newspaper library)."""
    try:
        from newspaper import Article
        return True
    except ImportError:
        return False

def get_research_capabilities() -> dict:
    """Get information about available research capabilities."""
    return {
        "hybrid_search": is_research_available(),  # DDG + newspaper for deep fetch
        "web_search": is_research_available(),     # Comprehensive web search
        "ddg_available": True  # DDG is always available
    }