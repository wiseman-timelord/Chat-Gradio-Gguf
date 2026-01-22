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
from scripts.sounds import speak_text, beep, cleanup_tts_resources

# Variables
_nlp_model = None

# NOTE: Platform-specific imports (win32com, pythoncom, pyttsx3) are imported
# lazily inside the functions that need them. This is because temporary.PLATFORM
# is not set until after the launcher parses command-line arguments.

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

def detect_cpu_config():
    """Detect CPU configuration and set thread options."""
    try:
        cpu_info = {
            "physical_cores": psutil.cpu_count(logical=False) or 1,
            "logical_cores": psutil.cpu_count(logical=True) or 1
        }
        
        temporary.CPU_PHYSICAL_CORES = cpu_info["physical_cores"]
        temporary.CPU_LOGICAL_CORES = cpu_info["logical_cores"]
        
        max_threads = temporary.CPU_LOGICAL_CORES
        temporary.CPU_THREAD_OPTIONS = list(range(1, max_threads + 1))
        
        if temporary.CPU_THREADS is None or temporary.CPU_THREADS > max_threads:
            temporary.CPU_THREADS = max(1, max_threads // 2)
        
        if "vulkan" in temporary.BACKEND_TYPE.lower():
            temporary.CPU_THREADS = max(2, temporary.CPU_THREADS)
        
        print(f"[CPU] Detected: {temporary.CPU_PHYSICAL_CORES} cores, "
              f"{temporary.CPU_LOGICAL_CORES} threads")
        print(f"[CPU] Current: {temporary.CPU_THREADS}")
        
    except Exception as e:
        temporary.set_status("CPU fallback", console=True)
        print(f"[CPU] Detection error: {e}")
        temporary.CPU_PHYSICAL_CORES = 4
        temporary.CPU_LOGICAL_CORES = 8
        temporary.CPU_THREAD_OPTIONS = list(range(1, 9))
        temporary.CPU_THREADS = 4

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
    """Calculate optimal number of GPU layers based on VRAM and model size."""
    try:
        from pathlib import Path
        
        if not Path(model_path).exists():
            print(f"[GPU-CALC] Model not found: {model_path}")
            return 0
        
        model_size_bytes = Path(model_path).stat().st_size
        model_size_mb = model_size_bytes / (1024 * 1024)
        context_mb = context_size / 1024
        usable_vram = vram_mb * 0.8
        estimated_layer_size = model_size_mb / 40
        available_for_layers = usable_vram - context_mb
        
        if available_for_layers <= 0:
            return 0
        
        optimal_layers = int(available_for_layers / estimated_layer_size)
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
    
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True
        )
        gpus.extend(line.strip() for line in output.splitlines() if line.strip())
    except:
        pass
    
    try:
        output = subprocess.check_output(["rocminfo"], stderr=subprocess.DEVNULL, text=True)
        for line in output.splitlines():
            if "Marketing Name" in line:
                name = line.split(":", 1)[-1].strip()
                if name:
                    gpus.append(name)
    except:
        pass
    
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
    
    seen = set()
    unique_gpus = [g for g in gpus if not (g in seen or seen.add(g))]
    
    if not unique_gpus:
        # Don't exit - return a fallback instead
        print("[GPU] WARNING: No GPUs detected, using CPU-only mode")
        return ["CPU Only"]
    
    return unique_gpus

def get_cpu_info():
    """Get CPU information for configuration purposes (Windows and Linux)."""
    try:
        import psutil
        cpu_count = psutil.cpu_count(logical=False) or 1
        logical_count = psutil.cpu_count(logical=True) or 1
        
        model = "Unknown CPU"
        
        if temporary.PLATFORM == "linux":
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    cpuinfo = f.read()
                model_match = re.search(r'model name\s*:\s*(.+)', cpuinfo)
                if model_match:
                    model = model_match.group(1).strip()
            except:
                model = "Generic CPU"
        
        elif temporary.PLATFORM == "windows":
            try:
                import subprocess
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip() and l.strip() != "Name"]
                    if lines:
                        model = lines[0]
            except:
                try:
                    import platform
                    model = platform.processor() or "Generic CPU"
                except:
                    model = "Generic CPU"
        
        return [{
            "label": f"{model} ({cpu_count} cores, {logical_count} threads)",
            "physical_cores": cpu_count,
            "logical_cores": logical_count
        }]
    except ImportError:
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

def summarize_session(messages):
    """Generate a short label for the session based on the first user message.
    Uses spaCy to extract key noun phrases for summarization, max 60 chars.
    """
    MAX_LABEL_LENGTH = 50
    
    if not messages:
        return "New Session"
    
    # Find the first user message
    first_user_msg = next((m['content'] for m in messages if m.get('role') == 'user'), "")
    if not first_user_msg.strip():
        return "Untitled Session"
    
    # Clean the input - remove User: prefix if present
    clean_msg = re.sub(r'^User:\s*\n?', '', first_user_msg.strip(), flags=re.MULTILINE).strip()
    
    nlp = get_nlp_model()
    if nlp:
        try:
            # Process limited text to avoid slow processing
            doc = nlp(clean_msg[:500])
            
            # Strategy 1: Extract key noun chunks (spaCy's actual summarization approach)
            noun_chunks = list(doc.noun_chunks)
            if noun_chunks:
                # Get unique root nouns to form a summary
                key_phrases = []
                seen_roots = set()
                for chunk in noun_chunks:
                    root_text = chunk.root.text.lower()
                    if root_text not in seen_roots and len(chunk.text) > 2:
                        seen_roots.add(root_text)
                        key_phrases.append(chunk.text)
                        # Stop if we have enough for a good summary
                        if len(' '.join(key_phrases)) >= 40:
                            break
                
                if key_phrases:
                    summary = ' '.join(key_phrases)
                    if len(summary) > MAX_LABEL_LENGTH:
                        return summary[:MAX_LABEL_LENGTH - 3] + "..."
                    return summary
            
            # Strategy 2: Extract named entities as fallback
            entities = [ent.text for ent in doc.ents if len(ent.text) > 2]
            if entities:
                summary = ' '.join(entities[:5])
                if len(summary) > MAX_LABEL_LENGTH:
                    return summary[:MAX_LABEL_LENGTH - 3] + "..."
                return summary
            
            # Strategy 3: Get subject and verb from first sentence
            for sent in doc.sents:
                subjects = [tok.text for tok in sent if tok.dep_ in ('nsubj', 'nsubjpass')]
                verbs = [tok.lemma_ for tok in sent if tok.pos_ == 'VERB']
                if subjects and verbs:
                    summary = f"{subjects[0]} {verbs[0]}"
                    if len(summary) <= MAX_LABEL_LENGTH:
                        return summary
                break  # Only process first sentence
                
        except Exception as e:
            print(f"[SESSION-LABEL] spaCy processing error: {e}")
    else:
        print("[SESSION-LABEL] spaCy model not available, using fallback")
    
    # Fallback: First few words, trimmed to max length
    words = clean_msg.split()[:8]
    fallback = ' '.join(words)
    if len(fallback) > MAX_LABEL_LENGTH:
        return fallback[:MAX_LABEL_LENGTH - 3] + "..."
    return fallback or "Chat Started"

def get_nlp_model():
    """Load spaCy NLP model lazily."""
    global _nlp_model
    if _nlp_model is None:
        try:
            _nlp_model = spacy.load("en_core_web_sm")
        except OSError:
            print("[NLP] spaCy model not found, using basic processing")
            _nlp_model = None
    return _nlp_model

def read_file_content(file_path):
    """Read content from various file types with proper error handling."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    
    try:
        if suffix in ['.txt', '.md', '.py', '.json', '.yaml', '.xml', '.html', '.css', '.js', '.bat', '.ps1']:
            # Try multiple encodings for cross-platform compatibility
            encodings_to_try = ['utf-8', 'cp1252', 'iso-8859-1', 'latin-1']
            content = None
            for encoding in encodings_to_try:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break  # Success, exit loop
                except UnicodeDecodeError:
                    continue  # Try next encoding
            
            if content is not None:
                return content, "text", True, None
            else:
                # Last resort: read with errors='replace' to substitute bad chars
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

def save_session_history(session_messages, attached_files):
    """Save session history to JSON file."""
    import re  # Add if not already imported
    
    if not temporary.session_label:
        temporary.session_label = "Untitled"
    
    # Sanitize label for filename (remove invalid chars, limit length)
    safe_label = re.sub(r'[^a-zA-Z0-9_-]', '_', temporary.session_label)[:50]
    if not safe_label:
        safe_label = "Untitled"
    
    filepath = Path(HISTORY_DIR) / f"session_{temporary.current_session_id}_{safe_label}.json"
    
    data = {
        "session_id": temporary.current_session_id,
        "label": temporary.session_label,
        "history": session_messages,
        "attached_files": attached_files
    }
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Session saved: {filepath.name}")
    return filepath

def load_session_history(filename):
    """Load session history from JSON file."""
    filepath = Path(HISTORY_DIR) / filename
    
    if not filepath.exists():
        return None, None, [], []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get("session_id"), data.get("label"), data.get("history", []), data.get("attached_files", [])

def get_saved_sessions():
    """Get list of saved session files sorted by modification time."""
    history_dir = Path(HISTORY_DIR)
    session_files = sorted(history_dir.glob("session_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
    return [f.name for f in session_files]


# ============================================================================
# WEB SEARCH / RESEARCH FUNCTIONS
# ============================================================================

def web_search(query: str, num_results=5, max_hits: int = 6) -> str:
    """
    Quick DuckDuckGo text search with automatic current date injection.
    This is the fast "Search" mode - returns snippets only.
    """
    if not query.strip():
        return "Empty query."

    current_date = datetime.now().strftime("%B %d, %Y")
    current_year = datetime.now().year
    
    # Expanded time-sensitive keywords including year references
    time_sensitive_keywords = [
        'latest', 'current', 'new', 'recent', 'today', 'now', 
        'this year', 'this month', 'this week',
        '2024', '2025', '2026', '2027',  # Cover near years
        'update', 'breaking', 'news', 'developments', 'happening'
    ]
    
    # Check if query already has a year or is time-sensitive
    query_lower = query.lower()
    has_year = any(str(y) in query_lower for y in range(2020, 2030))
    is_time_sensitive = any(keyword in query_lower for keyword in time_sensitive_keywords)
    
    if is_time_sensitive and not has_year:
        # Add current year to make search more relevant
        enhanced_query = f"{query} {current_year}"
    else:
        enhanced_query = query
    
    # Create informative header for LLM context
    date_header = (
        f"[Web Search Results]\n"
        f"[Current Date: {current_date}]\n"
        f"[Search Query: {enhanced_query}]\n"
        f"[Note: Prioritize recent information. Results may contain outdated content.]\n\n"
    )

    try:
        hits = DDGS().text(enhanced_query, max_results=max_hits)
    except DDGSException as e:
        raw = f"DuckDuckGo error: {e}"
        if temporary.PRINT_RAW_OUTPUT:
            print("=== RAW DDG ===\n", raw, "\n=== END ===", flush=True)
        return date_header + raw

    if not hits:
        raw = "DuckDuckGo returned zero results."
        if temporary.PRINT_RAW_OUTPUT:
            print("=== RAW DDG ===\n", raw, "\n=== END ===", flush=True)
        return date_header + raw

    raw = "\n\n".join(
        f"[{i}] **{h.get('title','')}**\n{h.get('body','')}\n*Source:* <{h.get('href','')}>"
        for i, h in enumerate(hits, 1)
    )

    if temporary.PRINT_RAW_OUTPUT:
        print("=== RAW DDG ===\n", date_header + raw, "\n=== END ===", flush=True)

    return date_header + raw

def web_research(query: str, max_pages: int = 5, use_js: bool = False) -> str:
    """
    Deep web research with full page reading capabilities.
    This is the comprehensive "Research" mode - fetches and analyzes full pages.
    
    Args:
        query: Search query
        max_pages: Maximum pages to fetch and analyze (default 5)
        use_js: Use Qt WebEngine for JS-rendered pages (slower but more accurate)
    
    Returns:
        Formatted research results string for LLM context
    """
    try:
        from scripts.research import web_research as deep_research
        return deep_research(query, mode="deep", max_pages=max_pages, use_js=use_js)
    except ImportError as e:
        print(f"[RESEARCH] Module not available: {e}")
        # Fallback to enhanced DDG search
        return _research_fallback(query, max_pages)
    except Exception as e:
        print(f"[RESEARCH] Error: {e}")
        return _research_fallback(query, max_pages)


def _research_fallback(query: str, max_pages: int = 5) -> str:
    """
    Fallback research using newspaper3k/4k for article extraction.
    Used when research.py module is unavailable.
    """
    current_date = datetime.now().strftime("%B %d, %Y")
    current_year = datetime.now().year
    
    # Enhance query with year if time-sensitive
    query_lower = query.lower()
    time_sensitive = any(kw in query_lower for kw in [
        'latest', 'current', 'new', 'recent', 'today', 'now',
        'update', 'breaking', 'news', 'developments', 'happening'
    ])
    has_year = any(str(y) in query_lower for y in range(2020, 2030))
    
    search_query = f"{query} {current_year}" if time_sensitive and not has_year else query
    
    header = (
        f"[Deep Research Results]\n"
        f"[Current Date: {current_date}]\n"
        f"[Query: {search_query}]\n"
        f"[Note: Verify information recency. Prioritize recent sources.]\n\n"
    )
    
    try:
        hits = DDGS().text(search_query, max_results=max_pages + 2)
    except DDGSException as e:
        return header + f"Search error: {e}"
    
    if not hits:
        return header + "No results found."
    
    results = []
    for i, hit in enumerate(hits[:max_pages], 1):
        url = hit.get('href', '')
        title = hit.get('title', 'Untitled')
        snippet = hit.get('body', '')
        
        # Try to extract full article content
        article_content = ""
        try:
            article = Article(url)
            article.download()
            article.parse()
            if article.text:
                # Limit to first ~2000 chars
                article_content = article.text[:2000]
                if len(article.text) > 2000:
                    article_content += "\n[...truncated...]"
                # Try to get publish date
                if article.publish_date:
                    article_content = f"[Published: {article.publish_date.strftime('%Y-%m-%d')}]\n{article_content}"
        except Exception as e:
            print(f"[RESEARCH-FALLBACK] Article extraction failed for {url}: {e}")
            article_content = snippet
        
        results.append(f"Source {i}: {title}")
        results.append(f"URL: {url}")
        results.append(f"Content:\n{article_content or snippet}")
        results.append("")
    
    return header + "\n".join(results)


def is_research_available() -> bool:
    """Check if deep research capabilities are available."""
    try:
        from scripts.research import is_deep_research_available
        return is_deep_research_available()
    except ImportError:
        # Check if newspaper is available as fallback
        try:
            from newspaper import Article
            return True
        except ImportError:
            return False


def get_research_capabilities() -> dict:
    """Get information about available research capabilities."""
    try:
        from scripts.research import get_research_capabilities
        return get_research_capabilities()
    except ImportError:
        return {
            "quick_search": True,
            "deep_research": is_research_available(),
            "js_rendering": False,
            "async_fetch": False,
            "bs4_parsing": False,
            "qt_version": 0
        }


# ============================================================================
# REMAINING UTILITY FUNCTIONS
# ============================================================================

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

def load_and_chunk_documents(file_paths: list) -> list:
    """Load and chunk documents from a list of file paths for RAG."""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader
    # Fixed typo: DIVIDER not DEVIDER
    from .temporary import CONTEXT_SIZE, RAG_CHUNK_SIZE_DIVIDER, RAG_CHUNK_OVERLAP_DIVIDER
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
    """Thin wrapper so interface.py can call the injector transparently."""
    from scripts.temporary import context_injector
    context_injector.set_session_vectorstore(file_paths)

def process_files(files, existing_files, max_files, is_attach=True):
    """Process uploaded files for attach or vector, ensuring no duplicates and respecting max file limits.
    
    Handles both Gradio 3.x (_TemporaryFileWrapper objects) and Gradio 4.x (string paths).
    """
    if not files:
        return "No files uploaded.", existing_files

    # Normalize files to string paths - handle both Gradio 3.x and 4.x formats
    normalized_files = []
    for f in files:
        if f is None:
            continue
        # Handle _TemporaryFileWrapper or file-like objects (Gradio 3.x)
        if hasattr(f, 'name'):
            file_path = f.name
        # Handle string paths (Gradio 4.x or already normalized)
        elif isinstance(f, (str, Path)):
            file_path = str(f)
        # Handle dict format (some Gradio versions return {"name": path, ...})
        elif isinstance(f, dict) and 'name' in f:
            file_path = f['name']
        else:
            print(f"[FILES] Skipping unknown file format: {type(f)}")
            continue
        
        if file_path and os.path.isfile(file_path):
            normalized_files.append(file_path)
    
    if not normalized_files:
        return "No valid files to add.", existing_files

    # Filter out files already in existing_files
    new_files = [f for f in normalized_files if f not in existing_files]
    if not new_files:
        return "No new files to add.", existing_files

    # Remove existing files with the same name (replace with new version)
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
