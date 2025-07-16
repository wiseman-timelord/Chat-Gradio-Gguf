# Script: `.\scripts\utility.py`

# Imports...
import tempfile
import re, subprocess, json, time, random, psutil, shutil, os, zipfile, yake, sys
from pathlib import Path
from datetime import datetime
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from newspaper import Article
from .models import load_models, clean_content
import scripts.settings as settings
from .temporary import (
    TEMP_DIR, HISTORY_DIR, SESSION_FILE_FORMAT, ALLOWED_EXTENSIONS, 
    current_session_id, session_label
)
from . import temporary

# Conditional imports based on platform
if temporary.PLATFORM == "windows":
    import win32com.client
    import pythoncom
elif temporary.PLATFORM == "linux":
    try:
        import pyttsx3
    except ImportError:
        print("Warning: pyttsx3 not installed. Text-to-speech will be unavailable on Linux.")

# Functions...
def short_path(path_str, max_len=44):
    """Truncate path to last max_len chars with ... prefix"""
    path = str(path_str)
    if len(path) <= max_len:
        return path
    return "..." + path[-max_len:]

def filter_operational_content(text):
    """Remove operational tags and metadata from the text."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<answer>.*?</answer>', '', text, flags=re.DOTALL)
    return text.strip()

def detect_cpu_config():
    """Detect CPU configuration and set thread options"""
    try:
        import psutil
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False) or 1,
            "logical_cores": psutil.cpu_count(logical=True) or 1
        }
        
        temporary.CPU_PHYSICAL_CORES = cpu_info["physical_cores"]
        temporary.CPU_LOGICAL_CORES = cpu_info["logical_cores"]
        
        # Generate thread options (1 to logical_cores-1)
        max_threads = max(1, temporary.CPU_LOGICAL_CORES - 1)
        temporary.CPU_THREAD_OPTIONS = list(range(1, max_threads + 1))
        temporary.CPU_THREADS = min(4, max_threads)  # Default to 4 or max available
        
        # Vulkan-specific: Even with all layers on GPU, still uses some CPU threads
        if "vulkan" in temporary.BACKEND_TYPE.lower():
            temporary.CPU_THREADS = max(2, temporary.CPU_THREADS)  # Ensure at least 2 threads
            
    except Exception as e:
        temporary.set_status("CPU fallback", console=True)
        # Fallback values
        temporary.CPU_PHYSICAL_CORES = 4
        temporary.CPU_LOGICAL_CORES = 8
        temporary.CPU_THREAD_OPTIONS = [1, 2, 3, 4, 5, 6, 7]
        temporary.CPU_THREADS = 4

def get_available_gpus_windows():
    """
    Retrieve available GPUs on Windows using wmic and dxdiag.
    """
    try:
        output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
        gpus = [line.strip() for line in output.split('\n') if line.strip() and 'Name' not in line]
        return gpus if gpus else ["CPU Only"]
    except Exception:
        try:
            temp_file = Path(temporary.TEMP_DIR) / "dxdiag.txt"
            subprocess.run(f"dxdiag /t {temp_file}", shell=True, check=True)
            time.sleep(2)
            with open(temp_file, 'r') as f:
                content = f.read()
                gpu = re.search(r"Card name: (.+)", content)
                return [gpu.group(1).strip()] if gpu else ["CPU Only"]
        except Exception:
            return ["CPU Only"]

def get_available_gpus_linux():
    """Get available GPUs on Linux systems with improved detection"""
    gpus = []

    try:
        # 1) NVIDIA GPUs (via nvidia-smi)
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
                text=True
            )
            gpus.extend(f"NVIDIA {line.strip()}" for line in output.splitlines() if line.strip())
        except Exception:
            pass

        # 2) AMD GPUs (via rocminfo – ROCm stack)
        try:
            output = subprocess.check_output(
                ["rocminfo"],
                stderr=subprocess.DEVNULL,
                text=True
            )
            for line in output.splitlines():
                if "Marketing Name" in line:
                    name = line.split(":", 1)[-1].strip()
                    if name:
                        gpus.append(f"AMD {name}")
        except Exception:
            pass

        # 3) Generic PCI bus scan (catches AMD, Intel, and others)
        try:
            output = subprocess.check_output(
                ["lspci", "-nn"],
                stderr=subprocess.DEVNULL,
                text=True
            )
            for line in output.splitlines():
                lower = line.lower()
                if "vga" in lower or "display" in lower or "3d" in lower:
                    name = line.split(":", 2)[-1].strip()
                    if "nvidia" in lower:
                        gpus.append(f"NVIDIA {name}")
                    elif "amd" in lower or "ati" in lower:
                        gpus.append(f"AMD {name}")
                    elif "intel" in lower:
                        gpus.append(f"Intel {name}")
                    else:
                        gpus.append(name)
        except Exception:
            pass

    except Exception as e:
        print(f"GPU detection error: {str(e)}")

    # Remove duplicates while preserving order
    seen = set()
    unique_gpus = [g for g in gpus if not (g in seen or seen.add(g))]

    if not unique_gpus:
        temporary.set_status("No GPU", console=True)
        sys.exit(1)          # critical exit → back to launcher caller

    return unique_gpus

def get_cpu_info():
    """
    Get CPU information for configuration purposes.
    Returns a list of dictionaries with CPU information.
    """
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False) or 1
        logical_count = psutil.cpu_count(logical=True) or 1
        
        # Try to get CPU brand/model info
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
            model_match = re.search(r'model name\s*:\s*(.+)', cpuinfo)
            if model_match:
                model = model_match.group(1).strip()
            else:
                model = "Unknown CPU"
        except:
            model = "Generic CPU"
        
        return [{
            "label": f"{model} ({cpu_count} cores, {logical_count} threads)",
            "physical_cores": cpu_count,
            "logical_cores": logical_count
        }]
    except ImportError:
        # Fallback if psutil is not available
        return [{
            "label": "Generic CPU",
            "physical_cores": 4,
            "logical_cores": 8
        }]
    except Exception as e:
        print(f"Error getting CPU info: {e}")
        return [{
            "label": "Default CPU",
            "physical_cores": 4,
            "logical_cores": 8
        }]

def get_available_gpus():
    """Returns list of GPUs with Intel marked appropriately"""
    gpus = []
    if temporary.PLATFORM == "windows":
        # Windows detection logic that identifies Intel GPUs
        gpus = get_available_gpus_windows()  # Should include Intel GPUs
    elif temporary.PLATFORM == "linux":
        # Linux detection that identifies Intel GPUs
        gpus = get_available_gpus_linux()    # Should include Intel GPUs
    return gpus
            
def generate_session_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def speak_text(text):
    """Speak the given text using platform-specific text-to-speech."""
    if not text or not isinstance(text, str):
        return
        
    # Check if we just spoke this text
    if hasattr(temporary, 'LAST_SPOKEN') and temporary.LAST_SPOKEN == text:
        return
        
    if temporary.PLATFORM == "windows":
        try:
            import pythoncom
            import win32com.client
            
            if not pythoncom.CoInitialized():
                pythoncom.CoInitialize()
                
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
            temporary.LAST_SPOKEN = text
            
        except Exception as e:
            print(f"Windows TTS error: {str(e)}")
            try:
                import winsound
                winsound.Beep(1000, 200)
            except:
                pass
                
        finally:
            if pythoncom.CoInitialized():
                pythoncom.CoUninitialize()
                
    elif temporary.PLATFORM == "linux":
        try:
            # Initialize engine if not exists
            if not hasattr(temporary, 'tts_engine'):
                import pyttsx3
                try:
                    temporary.tts_engine = pyttsx3.init(driverName='espeak')
                    temporary.tts_engine.setProperty('rate', 150)
                    temporary.tts_engine.setProperty('volume', 0.9)
                except:
                    temporary.tts_engine = pyttsx3.init()
                    
            temporary.tts_engine.say(text)
            temporary.tts_engine.runAndWait()
            temporary.LAST_SPOKEN = text
            
        except Exception as e:
            print(f"Linux TTS error (pyttsx3): {str(e)}")
            try:
                # Fallback to direct espeak
                subprocess.run(
                    ['espeak', '-v', 'en-us', '-s', '150', text],
                    check=False
                )
            except FileNotFoundError:
                try:
                    # Final fallback to spd-say
                    subprocess.run(
                        ['spd-say', '--wait', '-r', '-50', '-t', 'female3', text],
                        check=False
                    )
                except FileNotFoundError:
                    print("All Linux TTS methods failed - no speech available")
                except Exception as e:
                    print(f"spd-say error: {str(e)}")
            except Exception as e:
                print(f"espeak error: {str(e)}")
    else:
        raise ValueError(f"Unsupported platform: {temporary.PLATFORM}")

def generate_session_label(session_log):
    """Generate a session label using YAKE on the entire session log, up to 25 characters."""
    if not session_log:
        return "Untitled"
    text_for_yake = " ".join([clean_content(msg['role'], msg['content']) for msg in session_log])
    text_for_yake = filter_operational_content(text_for_yake)
    kw_extractor = yake.KeywordExtractor(lan="en", n=4, dedupLim=0.9, top=1)
    keywords = kw_extractor.extract_keywords(text_for_yake)
    description = keywords[0][0] if keywords else "No description"
    if len(description) > 25:
        description = description[:25]
    return description

def save_session_history(session_log, attached_files):
    """Save session history with error handling and timestamp."""
    try:
        # Ensure directories exist
        Path(HISTORY_DIR).mkdir(parents=True, exist_ok=True)
        Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)
        
        if not temporary.current_session_id:
            temporary.current_session_id = generate_session_id()
        
        if not temporary.session_label or temporary.session_label == "Untitled":
            temporary.session_label = generate_session_label(session_log)
        
        session_file = Path(HISTORY_DIR) / f"session_{temporary.current_session_id}.json"
        
        session_data = {
            "session_id": temporary.current_session_id,
            "label": temporary.session_label,
            "history": session_log,
            "attached_files": [str(Path(f).resolve()) for f in attached_files] if attached_files else [],
            "last_saved": datetime.now().isoformat()
        }
        
        # Atomic write using temporary file
        temp_file = Path(TEMP_DIR) / f"temp_{temporary.current_session_id}.json"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(session_data, f, ensure_ascii=False, indent=2)
        
        # Atomic move
        temp_file.replace(session_file)
        temporary.set_status("Ready")
        manage_session_history()
        return True
        
    except Exception as e:
        print(f"Error saving session: {str(e)}")
        return False

def load_session_history(session_file):
    """Load session history from a file."""
    try:
        with open(session_file, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading session file {session_file}: {e}")
        return None, "Error", [], []

    session_id = data.get("session_id", session_file.stem.replace('session_', ''))
    label = data.get("label", "Untitled")
    history = data.get("history", [])
    attached_files = [str(Path(file).resolve()) for file in data.get("attached_files", []) if Path(file).exists()]

    if len(attached_files) != len(data.get("attached_files", [])):
        print(f"Removed missing attached files from session {session_id}")

    temporary.session_attached_files = attached_files
    return session_id, label, history, attached_files

def manage_session_history():
    """Limit saved sessions to MAX_HISTORY_SLOTS."""
    history_dir = Path(HISTORY_DIR)
    session_files = sorted(history_dir.glob("session_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    while len(session_files) > temporary.MAX_HISTORY_SLOTS:
        oldest_file = session_files.pop()
        oldest_file.unlink()
        print(f"Deleted oldest session: {oldest_file}")

def process_uploaded_files(files):
    """Process uploaded files (if needed beyond copying)."""
    return [file.name for file in files] if files else []

def chunk_text_for_speech(text, max_chars=500):
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Split into sentences first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_length = len(sentence)
        
        if current_length + sentence_length <= max_chars:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            # Handle long individual sentences
            while sentence_length > max_chars:
                split_pos = sentence[:max_chars].rfind('. ') + 1
                if split_pos <= 0:
                    split_pos = max_chars
                chunks.append(sentence[:split_pos])
                sentence = sentence[split_pos:].lstrip()
                sentence_length = len(sentence)
            if sentence:
                current_chunk.append(sentence)
                current_length += sentence_length
                
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def web_search(query: str, num_results: int = 3) -> str:
    """Perform a web search using DuckDuckGo and extract article content for relevant results."""
    try:
        results = DuckDuckGoSearchAPIWrapper().results(query, num_results)
        if not results:
            return "No results found."

        formatted = []
        links = []
        for result in results:
            link = result.get('link', '').strip()
            if not link:
                continue

            domain = re.sub(r'https?://(www\.)?([^/]+).*', r'\2', link)
            snippet = result.get('snippet', 'No snippet available.').strip()

            try:
                article = Article(link)
                article.download()
                article.parse()
                summary = article.text[:500].strip()
                formatted.append(
                    f"[{domain}]({link}): {snippet}\n"
                    f"{summary}..." if summary else f"[{domain}]({link}): {snippet}"
                )
                links.append(link)
            except Exception:
                formatted.append(f"[{domain}]({link}): {snippet}")
                links.append(link)

        if links:
            formatted.append("\n\nLinks:\n" + "\n".join([f"- {link}" for link in links]))

        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {type(e).__name__} – {str(e)}"

def summarize_document(file_path):
    """Summarize the contents of a document using YAKE, up to 100 characters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        kw_extractor = yake.KeywordExtractor(lan="en", n=4, dedupLim=0.9, top=1)
        keywords = kw_extractor.extract_keywords(content)
        summary = keywords[0][0] if keywords else "No summary available"
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

def delete_all_session_histories():
    """
    Delete all history JSON files in HISTORY_DIR.
    Returns a status message indicating the result.
    """
    history_dir = Path(HISTORY_DIR)
    for file in history_dir.glob('*.json'):
        try:
            file.unlink()
            print(f"Deleted history file: {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    return "All session histories deleted."

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
    history_dir = Path(HISTORY_DIR)
    session_files = sorted(history_dir.glob("session_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return [f.name for f in session_files]

def process_files(files, existing_files, max_files, is_attach=True):
    """
    Process uploaded files for attach or vector, ensuring no duplicates and respecting max file limits.
    """
    if not files:
        return "No files uploaded.", existing_files

    new_files = [f for f in files if os.path.isfile(f) and f not in existing_files]
    if not new_files:
        return "No new files to add.", existing_files

    for f in new_files:
        file_name = Path(f).name
        existing_files = [ef for ef in existing_files if Path(ef).name != file_name]

    available_slots = max_files - len(existing_files)
    processed_files = new_files[:available_slots]
    updated_files = processed_files + existing_files

    if is_attach:
        temporary.session_attached_files = updated_files
    else:
        temporary.session_vector_files = updated_files

    status = f"Processed {len(processed_files)} new {'attach' if is_attach else 'vector'} files."
    return status, updated_files

