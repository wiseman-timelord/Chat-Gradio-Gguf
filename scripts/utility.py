# Script: `.\scripts\utility.py`

# Imports...
import tempfile
import re, subprocess, json, time, random, psutil, shutil, os, zipfile, yake
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
def filter_operational_content(text):
    """Remove operational tags and metadata from the text."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<answer>.*?</answer>', '', text, flags=re.DOTALL)
    return text.strip()

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
    try:
        # NVIDIA detection
        try:
            output = subprocess.check_output(
                "nvidia-smi --query-gpu=name --format=csv,noheader",
                shell=True,
                stderr=subprocess.DEVNULL
            ).decode()
            return [f"NVIDIA {line.strip()}" for line in output.split('\n') if line.strip()]
        except:
            pass
        
        # AMD detection
        try:
            output = subprocess.check_output(
                "rocminfo | grep 'Marketing Name' | awk -F: '{print $2}'",
                shell=True,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return [f"AMD {line.strip()}" for line in output.split('\n') if line.strip()]
        except:
            pass
        
        # Intel detection
        try:
            output = subprocess.check_output(
                "lspci | grep VGA | grep Intel | cut -d' ' -f5-",
                shell=True,
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return [f"Intel {output}"] if output else []
        except:
            pass
        
        return ["CPU Only"]
    except Exception as e:
        print(f"GPU detection error: {str(e)}")
        return ["CPU Only"]

def get_available_gpus():
    """Returns list of GPUs with Intel marked appropriately"""
    gpus = []
    if PLATFORM == "windows":
        # Windows detection logic that identifies Intel GPUs
        gpus = get_available_gpus_windows()  # Should include Intel GPUs
    elif PLATFORM == "linux":
        # Linux detection that identifies Intel GPUs
        gpus = get_available_gpus_linux()    # Should include Intel GPUs
    return gpus
            
def generate_session_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def speak_text(text):
    """Speak the given text using platform-specific text-to-speech.
    
    Args:
        text (str): The text to be spoken
        
    Handles:
        - Windows: Uses SAPI via win32com
        - Linux: Attempts pyttsx3, falls back to espeak, then spd-say
        - Thread safety on Windows
        - Avoids repeating the same text
        - Graceful degradation if TTS unavailable
    """
    if not text or not isinstance(text, str):
        return
        
    # Check if we just spoke this text
    if hasattr(temporary, 'LAST_SPOKEN') and temporary.LAST_SPOKEN == text:
        return
        
    if temporary.PLATFORM == "windows":
        try:
            import pythoncom
            import win32com.client
            
            # Initialize COM in new thread if needed
            if not pythoncom.CoInitialized():
                pythoncom.CoInitialize()
                
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
            temporary.LAST_SPOKEN = text
            
        except Exception as e:
            print(f"Windows TTS error: {str(e)}")
            try:
                # Fallback to simple beep if TTS fails
                import winsound
                winsound.Beep(1000, 200)
            except:
                pass
                
        finally:
            # Clean up COM if we initialized it
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
                    # Fallback to default driver
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
                    check=True
                )
            except FileNotFoundError:
                try:
                    # Final fallback to spd-say
                    subprocess.run(
                        ['spd-say', '--wait', '-r', '-50', '-t', 'female3', text],
                        check=True
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
        print(f"Session saved: {session_file}")
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
        links = []  # Store links separately
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
                links.append(link)  # Collect links
            except Exception:
                formatted.append(f"[{domain}]({link}): {snippet}")
                links.append(link)

        # Add links section if we have any
        if links:
            formatted.append("\n\nLinks:\n" + "\n".join([f"- {link}" for link in links]))
            
        return "\n\n".join(formatted)
    except Exception as e:
        return f"Search error: {str(e)}"

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

def update_setting(key, value):
    """Update a setting and return components requiring reload if necessary, with a confirmation message."""
    reload_required = False
    try:
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

        if reload_required:
            reload_result = change_model(temporary.MODEL_NAME.split('/')[-1])
            message = f"Setting '{key}' updated to '{value}', model reload triggered."
            return message, *reload_result
        else:
            message = f"Setting '{key}' updated to '{value}'."
            return message, None, None
    except Exception as e:
        message = f"Error updating setting '{key}': {str(e)}"
        return message, None, None