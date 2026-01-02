# Script: `.\scripts\utility.py`

# Imports
import tempfile
import re, subprocess, json, time, random, psutil, shutil, os, zipfile, spacy, sys, PyPDF2
from pathlib import Path
from datetime import datetime
from newspaper import Article
from ddgs import DDGS
from ddgs.exceptions import DDGSException
import requests.exceptions  # For HTTPError and Timeout
from docx import Document
from openpyxl import load_workbook
from pptx import Presentation
from .models import load_models, clean_content
import scripts.settings as settings
from .temporary import (
    TEMP_DIR, HISTORY_DIR, SESSION_FILE_FORMAT, ALLOWED_EXTENSIONS, 
    current_session_id, session_label
)
from . import temporary


# Variables
_nlp_model = None

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
def has_vulkan_binary():
    """Check if Vulkan binary is available (VULKAN_CPU or VULKAN_VULKAN modes)"""
    return temporary.BACKEND_TYPE in ["VULKAN_CPU", "VULKAN_VULKAN"]

def has_vulkan_wheel():
    """Check if Python wheel has Vulkan support (VULKAN_VULKAN mode only)"""
    return temporary.BACKEND_TYPE == "VULKAN_VULKAN"

def is_cpu_only():
    """Check if running in pure CPU mode (CPU_CPU mode)"""
    return temporary.BACKEND_TYPE == "CPU_CPU"

def short_path(path_str, max_len=44):
    """Truncate path to last max_len chars with ... prefix"""
    path = str(path_str)
    if len(path) <= max_len:
        return path
    return "..." + path[-max_len:]

def filter_operational_content(text):
    """Remove operational tags and metadata from the text."""
    text = re.sub(r'^AI-Chat:\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<answer>.*?</answer>', '', text, flags=re.DOTALL)
    return text.strip()

def beep():
    """
    PC speaker beep if enabled. Falls back to system sound if PC speaker unavailable.
    Uses platform-specific methods for both Windows and Linux.
    """
    if not getattr(temporary, "BLEEP_ON_EVENTS", False):
        return
    
    if temporary.PLATFORM == "windows":
        try:
            # Try PC speaker first (frequency, duration in ms)
            import winsound
            winsound.Beep(1000, 120)
            return
        except RuntimeError:
            # PC speaker not available, try system sound
            try:
                winsound.MessageBeep(winsound.MB_OK)
            except Exception as e:
                print(f"[BEEP] Sound unavailable: {e}")
        except Exception as e:
            print(f"[BEEP] Windows beep failed: {e}")
    
    elif temporary.PLATFORM == "linux":
        try:
            # Try PC speaker via /dev/console
            import subprocess
            subprocess.run(
                ['beep', '-f', '1000', '-l', '120'],
                timeout=1,
                check=False,
                stderr=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL
            )
            return
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # beep command not available, try terminal bell
            try:
                print("\a", end="", flush=True)
            except Exception as e:
                print(f"[BEEP] Linux beep failed: {e}")
        except Exception as e:
            print(f"[BEEP] Linux beep error: {e}")

def detect_cpu_config():
    """Detect CPU configuration and set thread options."""
    try:
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False) or 1,
            "logical_cores": psutil.cpu_count(logical=True) or 1
        }
        
        temporary.CPU_PHYSICAL_CORES = cpu_info["physical_cores"]
        temporary.CPU_LOGICAL_CORES = cpu_info["logical_cores"]
        
        # Generate thread options
        max_threads = temporary.CPU_LOGICAL_CORES
        temporary.CPU_THREAD_OPTIONS = list(range(1, max_threads + 1))
        
        # Set default threads
        if temporary.CPU_THREADS is None or temporary.CPU_THREADS > max_threads:
            temporary.CPU_THREADS = max(1, max_threads // 2)
        
        # Vulkan-specific: ensure minimum threads
        if "vulkan" in temporary.BACKEND_TYPE.lower():
            temporary.CPU_THREADS = max(2, temporary.CPU_THREADS)
        
        print(f"[CPU] Detected: {temporary.CPU_PHYSICAL_CORES} cores, "
              f"{temporary.CPU_LOGICAL_CORES} threads")
        print(f"[CPU] Current: {temporary.CPU_THREADS}")
        
    except Exception as e:
        temporary.set_status("CPU fallback", console=True)
        print(f"[CPU] Detection error: {e}")
        # Fallback values
        temporary.CPU_PHYSICAL_CORES = 4
        temporary.CPU_LOGICAL_CORES = 8
        temporary.CPU_THREAD_OPTIONS = list(range(1, 9))
        temporary.CPU_THREADS = 4

def get_available_gpus_windows():
    """Retrieve available GPUs on Windows using multiple methods."""
    try:
        # Try WMIC first
        output = subprocess.check_output("wmic path win32_VideoController get name", shell=True).decode()
        gpus = [line.strip() for line in output.split('\n') if line.strip() and 'Name' not in line]
        if gpus:
            return gpus
    except:
        pass
    
    # Fallback to dxdiag
    try:
        temp_file = Path(temporary.TEMP_DIR) / "dxdiag.txt"
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
    """
    Calculate optimal number of GPU layers based on VRAM and model size.
    Returns 0 for SRAM_ONLY mode, or calculated layers for VRAM_SRAM.
    """
    try:
        from pathlib import Path
        import struct
        
        if not Path(model_path).exists():
            print(f"[GPU-CALC] Model not found: {model_path}")
            return 0
        
        # Estimate model memory usage
        model_size_bytes = Path(model_path).stat().st_size
        model_size_mb = model_size_bytes / (1024 * 1024)
        
        # Context cache estimation (rough: 1MB per 1K context)
        context_mb = context_size / 1024
        
        # Reserve 20% VRAM for overhead
        usable_vram = vram_mb * 0.8
        
        # Estimate layers (typical model: ~40-80 layers, ~50-200MB each)
        # This is a rough heuristic - adjust based on testing
        estimated_layer_size = model_size_mb / 40  # Assume ~40 layers
        available_for_layers = usable_vram - context_mb
        
        if available_for_layers <= 0:
            return 0
        
        optimal_layers = int(available_for_layers / estimated_layer_size)
        
        # Cap at reasonable maximum (most models have 32-80 layers)
        optimal_layers = min(optimal_layers, 128)
        
        print(f"[GPU-CALC] Model: {model_size_mb:.0f}MB, Context: {context_mb:.0f}MB, "
              f"VRAM: {vram_mb}MB → {optimal_layers} layers")
        
        return max(0, optimal_layers)
        
    except Exception as e:
        print(f"[GPU-CALC] Error: {e}")
        return 0

def get_available_gpus_linux():
    """Get available GPUs on Linux systems with proper Intel detection."""
    gpus = []
    
    # NVIDIA GPUs
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True
        )
        gpus.extend(line.strip() for line in output.splitlines() if line.strip())
    except:
        pass
    
    # AMD GPUs via ROCm
    try:
        output = subprocess.check_output(["rocminfo"], stderr=subprocess.DEVNULL, text=True)
        for line in output.splitlines():
            if "Marketing Name" in line:
                name = line.split(":", 1)[-1].strip()
                if name:
                    gpus.append(name)
    except:
        pass
    
    # Generic PCI bus scan for all GPU types including Intel
    try:
        output = subprocess.check_output(["lspci", "-nn"], stderr=subprocess.DEVNULL, text=True)
        for line in output.splitlines():
            lower = line.lower()
            if any(keyword in lower for keyword in ["vga", "display", "3d"]):
                name = line.split(":", 2)[-1].strip()
                name = re.sub(r'\[[\da-f:]+\]', '', name).strip()
                if name and name not in gpus:
                    gpus.append(name)
    except:
        pass
    
    # Remove duplicates while preserving order
    seen = set()
    unique_gpus = [g for g in gpus if not (g in seen or seen.add(g))]
    
    if not unique_gpus:
        temporary.set_status("No GPU", console=True)
        sys.exit(1)
    
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
        
        # Return actual CPU info
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
    """Unified GPU detection for both Windows and Linux with Intel support."""
    if temporary.PLATFORM == "windows":
        return get_available_gpus_windows()
    elif temporary.PLATFORM == "linux":
        return get_available_gpus_linux()
    return ["CPU Only"]
            
def generate_session_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def speak_text(text):
    """Speak text with robust error handling for Windows."""
    if not text or not text.strip():
        return
    
    try:
        if temporary.PLATFORM == "windows":
            _speak_windows_safe(text)
        elif temporary.PLATFORM == "linux":
            _speak_linux(text)
    except Exception as e:
        print(f"[TTS] Speech error (contained): {e}")
        # Don't propagate - speech is non-critical

def _speak_windows_safe(text):
    """Windows TTS with isolated COM handling."""
    import threading
    import queue
    
    result_queue = queue.Queue()
    
    def _speak_isolated():
        import pythoncom
        import win32com.client
        
        try:
            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
            result_queue.put(("success", None))
        except Exception as e:
            result_queue.put(("error", str(e)))
        finally:
            try:
                pythoncom.CoUninitialize()
            except:
                pass
    
    thread = threading.Thread(target=_speak_isolated, daemon=True)
    thread.start()
    thread.join(timeout=30)
    
    if thread.is_alive():
        print("[TTS-WIN] Speech timed out (ignored)")
        return
    
    try:
        status, error = result_queue.get_nowait()
        if status == "error":
            print(f"[TTS-WIN] Error (ignored): {error}")
    except queue.Empty:
        pass

def _speak_linux(text):
    """Linux TTS implementation with multiple fallbacks."""
    try:
        # Initialize engine if needed
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
        print(f"[TTS-LINUX] Spoke: {text[:50]}...")
        
    except Exception as e:
        print(f"[TTS-LINUX] pyttsx3 error: {str(e)}")
        _speak_linux_fallback(text)

def cleanup_tts_resources():
    """Clean up TTS resources when toggling off."""
    try:
        if temporary.PLATFORM == "windows":
            # Force COM cleanup on main thread
            import pythoncom
            try:
                pythoncom.CoUninitialize()
            except:
                pass
        elif temporary.PLATFORM == "linux":
            if hasattr(temporary, 'tts_engine'):
                try:
                    temporary.tts_engine.stop()
                    del temporary.tts_engine
                except:
                    pass
    except Exception as e:
        print(f"[TTS] Cleanup error (ignored): {e}")

def get_nlp_model():
    """Lazy load spaCy model on first use."""
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
            print("[SPACY] Language model loaded")
        except OSError:
            print("[SPACY] WARNING: Model not found. Run: python -m spacy download en_core_web_sm")
            _nlp_model = False  # Mark as failed to avoid repeated attempts
    return _nlp_model if _nlp_model is not False else None

def read_file_content(file_path, max_chars=50000):
    """
    Read file content with support for multiple formats INCLUDING IMAGES.
    Returns tuple: (content: str, file_type: str, success: bool, error: str)
    
    Supported formats:
    - Text: .txt, .py, .json, .yaml, .md, .xml, .html, .css, .js, .sh, .bat, .ps1
    - PDF: .pdf
    - Word: .docx
    - Excel: .xlsx, .xls
    - PowerPoint: .pptx
    - Images: .png, .jpg, .jpeg, .gif, .bmp, .webp (returns base64 data URI)
    """
    from pathlib import Path
    import base64
    
    file_path = Path(file_path)
    if not file_path.exists():
        return "", "unknown", False, f"File not found: {file_path}"
    
    extension = file_path.suffix.lower()
    
    try:
        # IMAGE FILES
        image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
        
        if extension in image_extensions:
            try:
                with open(file_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    # Return data URI format for vision models
                    mime_type = f"image/{extension[1:]}" if extension != '.jpg' else "image/jpeg"
                    data_uri = f"data:{mime_type};base64,{image_data}"
                    return data_uri, "image", True, ""
            except Exception as e:
                return "", "image", False, f"Image read error: {str(e)}"
        
        # Text-based files
        text_extensions = {'.txt', '.py', '.json', '.yaml', '.yml', '.md', 
                          '.xml', '.html', '.css', '.js', '.sh', '.bat', 
                          '.ps1', '.psd1', '.xaml', '.csv', '.log'}
        
        if extension in text_extensions:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_chars)
                return content, "text", True, ""
        
        # PDF files
        elif extension == '.pdf':
            try:
                import PyPDF2
                content = []
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    max_pages = min(10, len(reader.pages))
                    for page_num in range(max_pages):
                        page = reader.pages[page_num]
                        content.append(page.extract_text())
                text = '\n'.join(content)[:max_chars]
                return text, "pdf", True, ""
            except ImportError:
                return "", "pdf", False, "PyPDF2 not installed. Run: pip install PyPDF2"
            except Exception as e:
                return "", "pdf", False, f"PDF error: {str(e)}"
        
        # Word documents
        elif extension == '.docx':
            try:
                from docx import Document
                doc = Document(file_path)
                content = '\n'.join([para.text for para in doc.paragraphs])[:max_chars]
                return content, "docx", True, ""
            except ImportError:
                return "", "docx", False, "python-docx not installed. Run: pip install python-docx"
            except Exception as e:
                return "", "docx", False, f"DOCX error: {str(e)}"
        
        # Excel files
        elif extension in {'.xlsx', '.xls'}:
            try:
                from openpyxl import load_workbook
                wb = load_workbook(file_path, read_only=True, data_only=True)
                content = []
                for sheet_name in wb.sheetnames[:3]:
                    sheet = wb[sheet_name]
                    content.append(f"=== Sheet: {sheet_name} ===")
                    for row in list(sheet.rows)[:50]:
                        row_data = [str(cell.value) if cell.value is not None else "" 
                                   for cell in row]
                        content.append(" | ".join(row_data))
                text = '\n'.join(content)[:max_chars]
                return text, "excel", True, ""
            except ImportError:
                return "", "excel", False, "openpyxl not installed. Run: pip install openpyxl"
            except Exception as e:
                return "", "excel", False, f"Excel error: {str(e)}"
        
        # PowerPoint files
        elif extension == '.pptx':
            try:
                from pptx import Presentation
                prs = Presentation(file_path)
                content = []
                for i, slide in enumerate(prs.slides[:10], 1):
                    content.append(f"=== Slide {i} ===")
                    for shape in slide.shapes:
                        if hasattr(shape, "text"):
                            content.append(shape.text)
                text = '\n'.join(content)[:max_chars]
                return text, "pptx", True, ""
            except ImportError:
                return "", "pptx", False, "python-pptx not installed. Run: pip install python-pptx"
            except Exception as e:
                return "", "pptx", False, f"PowerPoint error: {str(e)}"
        
        else:
            return "", "unsupported", False, f"Unsupported file type: {extension}"
    
    except Exception as e:
        return "", "error", False, f"Unexpected error: {str(e)}"

def prepare_user_input_for_model(user_input, context_size, threshold=0.5):
    """
    If user input exceeds threshold, chunk it and use RAG.
    Otherwise, return as-is.
    
    Args:
        user_input: Raw user text
        context_size: Model's n_ctx
        threshold: Fraction of context to trigger chunking (0.5 = 50%)
    
    Returns:
        (processed_input, was_chunked)
    """
    max_chars = int(context_size * 3)  # ~4 chars per token
    threshold_chars = int(max_chars * threshold)
    
    if len(user_input) < threshold_chars:
        return user_input, False
    
    # Create temporary chunks for this input
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=context_size // 6,
        chunk_overlap=context_size // 24
    )
    chunks = splitter.split_text(user_input)
    
    # Store in a temporary in-memory vector store
    return chunks, True

def get_attached_files_context(attached_files, query=None, max_total_chars=8000, context_size=None):
    """
    Generate context string from attached files with smart chunking for large content.
    
    Args:
        attached_files: List of file paths
        query: Optional user query for RAG-based filtering
        max_total_chars: Maximum total characters to include (soft limit)
        context_size: Total context size to calculate appropriate limits
    
    Returns:
        str: Formatted context string with file contents
    """
    if not attached_files:
        return ""
    
    from pathlib import Path
    import scripts.temporary as temporary
    
    # Use provided context_size or fall back to temporary.CONTEXT_SIZE
    if context_size is None:
        context_size = temporary.CONTEXT_SIZE
    
    # Calculate dynamic limits based on context size and query complexity
    base_context_reserve = min(context_size // 3, 8000)  # Reserve space for conversation
    available_for_files = context_size - base_context_reserve
    
    # If we have a query, prioritize RAG retrieval
    rag_context = None
    if query and hasattr(temporary.context_injector, 'get_relevant_context'):
        try:
            # Get relevant chunks using RAG
            rag_context = temporary.context_injector.get_relevant_context(query, k=6)
            if rag_context:
                # RAG found relevant content, use it as primary source
                file_list = [Path(f).name for f in attached_files]
                context_parts = [
                    f"📎 Attached Files ({len(file_list)}): {', '.join(file_list)}",
                    "",
                    "📖 Relevant Content from Files:",
                    rag_context[:available_for_files],
                    ""
                ]
                return "\n".join(context_parts)
        except Exception as e:
            print(f"[ATTACH] RAG retrieval error: {e}")
    
    # No RAG or RAG failed, fall back to direct content inclusion with smart chunking
    context_parts = []
    total_chars = 0
    
    # Build list of files with their names
    file_list = [Path(f).name for f in attached_files]
    context_parts.append(f"📎 Attached Files ({len(file_list)}): {', '.join(file_list)}")
    context_parts.append("")
    
    # Process each file with intelligent chunking
    remaining_chars = available_for_files
    files_processed = 0
    
    for file_path in attached_files:
        if total_chars >= available_for_files:
            # Add indicator for remaining files
            remaining_count = len(attached_files) - files_processed
            if remaining_count > 0:
                context_parts.append(f"... ({remaining_count} more files not shown due to context limits)")
            break
        
        file_name = Path(file_path).name
        
        try:
            # Read file content with dynamic chunk size
            chunk_size = min(remaining_chars // (len(attached_files) - files_processed), 10000)
            content, file_type, success, error = read_file_content(file_path, max_chars=chunk_size)
            
            if success and content.strip():
                # Smart truncation for large files
                if len(content) > chunk_size * 0.9:  # File was truncated
                    # Try to find a good break point (end of sentence/paragraph)
                    truncated_content = content[:chunk_size]
                    last_period = truncated_content.rfind('.')
                    last_newline = truncated_content.rfind('\n')
                    break_point = max(last_period, last_newline)
                    
                    if break_point > len(truncated_content) * 0.7:  # Found good break point
                        content = truncated_content[:break_point + 1]
                        content += f"\n...[Content truncated - {len(content)} chars shown]"
                    else:
                        content = truncated_content
                        content += f"\n...[File truncated due to context limits]"
                
                context_parts.append(f"--- {file_name} ({file_type}) ---")
                context_parts.append(content)
                context_parts.append("")
                
                total_chars += len(content)
                remaining_chars -= len(content)
                files_processed += 1
                
            elif not success:
                context_parts.append(f"⚠️ {file_name}: {error}")
                files_processed += 1
                
        except Exception as e:
            context_parts.append(f"⚠️ {file_name}: Error reading file - {str(e)}")
            files_processed += 1
    
    return "\n".join(context_parts)

def sanitize_label(label: str) -> str:
    """
    Make a session-label JSON-safe and Windows-cp1252-safe.
    Returns 'Untitled' if nothing usable remains.
    """
    if not label:
        return "Untitled"

    # 1. Remove C0 / C1 control codes (0x00-0x1F, 0x7F-0x9F) incl. 0x9d
    label = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', label)

    # 2. Replace common Unicode punctuation with ASCII
    label = (label
             .replace('\u2018', "'").replace('\u2019', "'")   # ‘ ’
             .replace('\u201C', '"').replace('\u201D', '"')   # “ ”
             .replace('\u2013', '-').replace('\u2014', '-')   # – —
             .replace('\u2026', '...')                        # …
             .replace('\u00A0', ' '))                         # non-breaking space

    # 3. Drop any remaining non-printable / non-basic-plane code-points
    label = ''.join(c for c in label if c.isprintable() and ord(c) < 0x10000)

    # 4. Collapse multiple spaces / leading-trailing junk
    label = re.sub(r'\s+', ' ', label).strip()

    # 5. Final guard
    if not label:
        return "Untitled"

    # 6. Length cap
    return label[:50].strip()

# Replace generate_session_label function:
def generate_session_label(session_log):
    """
    Generate a session label using spaCy keyword extraction, up to 50 characters.
    Output is sanitised for JSON safety.
    """
    if not session_log:
        return "Untitled"

    nlp = get_nlp_model()
    if nlp is None:
        # fallback: first few words of first user message
        for msg in session_log:
            if msg['role'] == 'user':
                content = clean_content(msg['role'], msg['content'])
                words = content.split()[:3]
                label = ' '.join(words) if words else "Untitled"
                return sanitize_label(label)
        return "Untitled"

    text_for_analysis = " ".join(
        clean_content(msg['role'], msg['content']) for msg in session_log
    )
    text_for_analysis = filter_operational_content(text_for_analysis)
    if not text_for_analysis.strip():
        return "Untitled"

    try:
        doc = nlp(text_for_analysis[:1000])   # speed guard
        candidates = []

        for ent in doc.ents:
            if ent.label_ in {'PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT'}:
                candidates.append((ent.text, len(ent.text.split())))

        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in {'NOUN', 'PROPN'}:
                candidates.append((chunk.text, len(chunk.text.split())))

        for token in doc:
            if token.pos_ in {'NOUN', 'PROPN'} and not token.is_stop:
                candidates.append((token.text, 1))

        if candidates:
            candidates.sort(key=lambda x: (-x[1], text_for_analysis.find(x[0])))
            description = candidates[0][0]
        else:
            words = [t.text for t in doc if not t.is_stop and t.is_alpha][:3]
            description = ' '.join(words) if words else "No description"

        return sanitize_label(description)          # ← critical
    except Exception as e:
        print(f"[SPACY] label error: {e}")
        words = text_for_analysis.split()[:3]
        label = ' '.join(words) if words else "Untitled"
        return sanitize_label(label)

def save_session_history(session_log, attached_files):
    """
    Save session history with UTF-8 encoding.
    Preserves the current label if already present (prevents re-editing corruption).
    """
    try:
        Path(HISTORY_DIR).mkdir(parents=True, exist_ok=True)
        Path(TEMP_DIR).mkdir(parents=True, exist_ok=True)

        if not temporary.current_session_id:
            temporary.current_session_id = generate_session_id()

        # ← do NOT regenerate label if one exists
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

        temp_file = Path(TEMP_DIR) / f"temp_{temporary.current_session_id}.json"
        with open(temp_file, 'w', encoding='utf-8') as f:   # ← UTF-8 enforced
            json.dump(session_data, f, ensure_ascii=False, indent=2)

        temp_file.replace(session_file)
        temporary.set_status("Ready")
        manage_session_history()
        return True
    except Exception as e:
        print(f"Error saving session: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_session_history(session_file):
    try:
        with open(session_file, 'r', encoding='utf-8', errors='replace') as f:
            data = json.load(f)
    except UnicodeDecodeError as e:
        print(f"Encoding error in {session_file}: {e}")
        try:
            with open(session_file, 'r', encoding='utf-8', errors='ignore') as f:
                data = json.load(f)
            print("Recovered session with ignored errors")
        except Exception as e2:
            print(f"Failed to recover {session_file}: {e2}")
            return None, "Corrupted", [], []
    except Exception as e:
        print(f"Error loading session file {session_file}: {e}")
        return None, "Error", [], []

    session_id = data.get("session_id", session_file.stem.replace('session_', ''))
    label = sanitize_label(data.get("label", "Untitled"))
    history = data.get("history", [])
    attached_files = [str(Path(f).resolve()) for f in data.get("attached_files", []) if Path(f).exists()]

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

def web_search(query: str, num_results, max_hits: int = 6) -> str:
    """
    DuckDuckGo text search that also dumps raw results to terminal
    when temporary.PRINT_RAW_OUTPUT is True.
    """
    import scripts.temporary as tmp           # local import avoids cycles

    if not query.strip():
        return "Empty query."

    try:
        hits = DDGS().text(query, max_results=max_hits)
    except DDGSException as e:
        raw = f"DuckDuckGo error: {e}"
        if temporary.PRINT_RAW_OUTPUT:
            print("=== RAW DDG ===\n", raw, "\n=== END ===", flush=True)
        return raw

    if not hits:
        raw = "DuckDuckGo returned zero results."
        if temporary.PRINT_RAW_OUTPUT:
            print("=== RAW DDG ===\n", raw, "\n=== END ===", flush=True)
        return raw

    raw = "\n\n".join(
        f"[{i}] **{h.get('title','')}**\n{h.get('body','')}\n*Source:* <{h.get('href','')}>"
        for i, h in enumerate(hits, 1)
    )

    if temporary.PRINT_RAW_OUTPUT:
        print("=== RAW DDG ===\n", raw, "\n=== END ===", flush=True)

    return raw

def summarize_document(file_path):
    """Summarize the contents of a document using spaCy, up to 100 characters."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        nlp = get_nlp_model()
        if nlp is None:
            # Fallback: first few words
            words = content.split()[:10]
            return ' '.join(words)[:100] if words else "No summary available"
        
        # Process with spaCy
        doc = nlp(content[:2000])  # Limit to first 2000 chars
        
        # Extract key phrases
        candidates = []
        
        # Get entities
        for ent in doc.ents:
            candidates.append(ent.text)
        
        # Get noun chunks
        for chunk in doc.noun_chunks:
            if chunk.root.pos_ in ['NOUN', 'PROPN']:
                candidates.append(chunk.text)
        
        if candidates:
            summary = candidates[0]
        else:
            # Fallback to first sentence
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

def get_saved_sessions():
    """Get list of saved session files sorted by modification time."""
    history_dir = Path(HISTORY_DIR)
    session_files = sorted(history_dir.glob("session_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return [f.name for f in session_files]

def create_session_vectorstore(file_paths):
    """
    Thin wrapper so interface.py can call the injector transparently.
    Returns nothing; side-effect is that context_injector now has data.
    """
    from scripts.temporary import context_injector
    context_injector.set_session_vectorstore(file_paths)

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

