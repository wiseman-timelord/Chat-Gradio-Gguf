# scripts/temporary.py

# Imports
import time
import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pathlib import Path

# System constants (platform, backend, etc.) loaded from data/constants.ini
PLATFORM = None
BACKEND_TYPE = "CPU_CPU"
VULKAN_AVAILABLE = False
LAYER_ALLOCATION_MODE = "SRAM_ONLY"
OS_VERSION = None  # Ubuntu version or Windows version string
WINDOWS_VERSION = None  # Windows-specific version (8.1, 10, 11)
EMBEDDING_BACKEND = "sentence_transformers"  # Always sentence_transformers now
GRADIO_VERSION = None
LOADED_CONTEXT_SIZE = None

# Configuration variables with defaults
MODEL_FOLDER = "path/to/your/models"
CONTEXT_SIZE = 32768
VRAM_SIZE = 8192
BATCH_SIZE = 1024
TEMPERATURE = 0.66
REPEAT_PENALTY = 1.1
DYNAMIC_GPU_LAYERS = True
MMAP = True
MLOCK = True
MAX_HISTORY_SLOTS = 12
MAX_ATTACH_SLOTS = 6
SESSION_LOG_HEIGHT = 650
INPUT_LINES = 27
VRAM_OPTIONS = [0, 756, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 49152, 65536]
CTX_OPTIONS = [1024, 2048, 4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072]
BATCH_OPTIONS = [128, 256, 512, 1024, 2048, 4096, 8096]
TEMP_OPTIONS = [0.0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0]
REPEAT_OPTIONS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
HISTORY_SLOT_OPTIONS = [4, 8, 10, 12, 16]
ATTACH_SLOT_OPTIONS = [2, 4, 6, 8, 10]
SESSION_LOG_HEIGHT_OPTIONS = [250, 450, 550, 600, 625, 650, 700, 800, 1000, 1400]

# General Constants/Variables/Lists/Maps/Arrays
TEMP_DIR = "data/temp"
HISTORY_DIR = "data/history"
SESSION_FILE_FORMAT = "%Y%m%d_%H%M%S"
session_label = ""
current_session_id = None
MODELS_LOADED = False
AVAILABLE_MODELS = None
SESSION_ACTIVE = False
MODEL_NAME = "Select_a_model..."
GPU_LAYERS = 0
SELECTED_GPU = None
STREAM_OUTPUT = True
USE_PYTHON_BINDINGS = True
DATA_DIR = None  # Will be set by launcher.py
llm = None
SPEECH_ENABLED = False
LLAMA_CLI_PATH = None  # will be set from constants.ini
LLAMA_BIN_PATH = None  # will be set from constants.ini
global_status = None
_status_lock = None  # Tracks which operation has status priority
_status_lock_message = ""  # Message to restore when lock releases 
PRINT_RAW_OUTPUT = False
SHOW_THINK_PHASE = False 
BLEEP_ON_EVENTS = False
THINK_MIN_CHARS_BEFORE_CLOSE = 100

# CPU Configuration
CPU_THREADS = None  # Will be auto-detected
CPU_THREAD_OPTIONS = []  # Will be populated with available thread counts
CPU_PHYSICAL_CORES = 1
CPU_LOGICAL_CORES = 1
SELECTED_CPU = None

# Arrays
session_attached_files = []

# UI Constants/Variables
USER_COLOR = "#ffffff"
THINK_COLOR = "#c8a2c8"
RESPONSE_COLOR = "#add8e6"
SEPARATOR = "=" * 40
MID_SEPARATOR = "-" * 30
ALLOWED_EXTENSIONS = {"bat", "py", "ps1", "txt", "json", "yaml", "psd1", "xaml", 
                      "png", "jpg", "jpeg", "gif", "bmp", "webp"}
MMPROJ_EXTENSIONS = ["-mmproj-", "mmproj"] 
MAX_POSSIBLE_HISTORY_SLOTS = 16
MAX_POSSIBLE_ATTACH_SLOTS = 10
demo = None
PROGRESS_COLORS = ['#ffffff', '#0000ff', '#ffff00', '#ff0000']
PROGRESS_CYCLE_TIME = 500  # milliseconds

# RAG CONSTANTS
RAG_CHUNK_SIZE_DIVIDER = 6
RAG_CHUNK_OVERLAP_DIVIDER = 24
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
LARGE_INPUT_THRESHOLD = 0.4  # 40% of context triggers RAG
RAG_RETRIEVAL_CHUNKS = 8     # How many chunks to retrieve
CONTEXT_ALLOCATION_RATIOS = {
    "system": 0.1,
    "history": 0.3,
    "current": 0.6
}

# Status text entries  
STATUS_MESSAGES = {
    "model_loading": "Loading model...",
    "model_loaded": "Model loaded successfully",
    "model_unloading": "Unloading model...",
    "model_unloaded": "Model unloaded successfully",
    "vram_calc": "Calculating layers...",
    "rag_process": "Analyzing documents...",
    "session_restore": "Restoring session...",
    "config_saved": "Settings saved",
    "docs_processed": "Documents ready",
    "generating_response": "Generating response...",
    "response_generated": "Response generated",
    "error": "An error occurred"
}

CHAT_FORMAT_MAP = {
    'qwen2'     : 'chatml',
    'llama'     : 'llama-2',
    'qwen3'     : 'chatml',
    'qwen3moe'  : 'chatml',
    'deepseek2' : 'deepseek',
    'stablelm'  : 'chatml',
}

# Handling Keywords for Special Model Behaviors
handling_keywords = {
    "code": ["code", "code", "program", "dev", "copilot", "Python", "Powershell"],
    "uncensored": ["uncensored", "unfiltered", "unbiased", "unlocked"],
    "reasoning": ["reason", "r1", "think", "thinking"],
    "nsfw": ["nsfw", "adult", "mature", "explicit", "lewd"],
    "roleplay": ["rpg", "role", "adventure"],
    "moe": ["moe", "mixtral", "deepseek-v2", "gpt-oss", "harmony", "qwen2moe", "qwen2.5moe", "dbrx", "arctic", "grok"],
    "vision": ["vision", "llava", "moondream", "minicpm", "qvq", "apriel", "qwen3-vl", "qwen3vl", "qwen2.5-vl"]
}

# Classes...
class ContextInjector:
    """
    Universal RAG with support for both file attachments AND large pasted inputs.
    Provides unlimited context through intelligent chunking and retrieval.
    Uses sentence-transformers for embeddings (cross-platform Win 7-11, Ubuntu 22-25).
    """
    def __init__(self):
        self.embedding = None
        self.file_index = None          # FAISS index for attached files
        self.temp_index = None          # FAISS index for large pasted inputs
        self.file_chunks = []           # Chunks from attached files
        self.temp_chunks = []           # Chunks from large pasted inputs
        self._model_load_attempted = False
    
    def _ensure_embedding_model(self):
        """Initialize embedding model on first use with proper cache path (fully offline)."""
        if self._model_load_attempted:
            return
        
        self._model_load_attempted = True
        
        import os
        from pathlib import Path
        
        # Use the value from config (loaded by settings.py)
        model_to_load = EMBEDDING_MODEL_NAME
        
        # Use the cache path matching the installer
        cache_dir = Path(__file__).parent.parent / "data" / "embedding_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL: Set HF_HUB_OFFLINE=1 BEFORE importing sentence_transformers
        # This prevents ANY network requests to HuggingFace
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir.absolute())
        os.environ["HF_HOME"] = str(cache_dir.parent.absolute())
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir.absolute())
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU mode
        
        try:
            import torch
            torch.set_grad_enabled(False)  # Disable gradients for inference
            
            from sentence_transformers import SentenceTransformer
            
            print(f"[RAG] Cache directory: {cache_dir}")
            print(f"[RAG] Attempting to load (offline): {model_to_load}")
            
            # Load the model with CPU-only mode and local_files_only
            # HF_HUB_OFFLINE=1 ensures no network requests at all
            self.embedding = SentenceTransformer(
                model_to_load, 
                device="cpu",
                local_files_only=True,
                cache_folder=str(cache_dir)  # Explicit cache folder
            )
            self.embedding.eval()  # Set to evaluation mode
            
            print(f"[RAG] Successfully loaded: {model_to_load}")
            
        except OSError as e:
            # OSError typically means model files not found in cache
            print(f"[RAG] Model not found in cache: {e}")
            print(f"[RAG] Expected cache location: {cache_dir}")
            print("[RAG] RAG features will be disabled. Re-run installer to download model.")
            self.embedding = None
        except Exception as e:
            print(f"[RAG] Failed to load embedding model: {e}")
            print("[RAG] RAG features will be disabled.")
            self.embedding = None

    def _embed_texts(self, texts):
        """Embed a list of texts using sentence-transformers (CPU-only)."""
        if self.embedding is None:
            return None
        
        try:
            # Use convert_to_tensor=True then convert to numpy to avoid numpy 2.x issues
            embeddings = self.embedding.encode(
                texts, 
                batch_size=32, 
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_tensor=True,
                device="cpu"
            )
            # Convert tensor to numpy array
            return embeddings.cpu().numpy().astype('float32')
        except Exception as e:
            print(f"[RAG] Embedding error: {e}")
            return None

    def set_session_vectorstore(self, file_paths):
            """Create/refresh vector store from list of file paths with improved chunking."""
            if not file_paths:
                self.file_index = None
                self.file_chunks = []
                return
            
            self._ensure_embedding_model()
            
            if self.embedding is None:
                print("[RAG] Cannot create vectorstore - embedding model unavailable")
                return

            all_docs = []
            
            # Use loaded context size for chunk sizing
            effective_ctx = LOADED_CONTEXT_SIZE or CONTEXT_SIZE
            chunk_size = effective_ctx // RAG_CHUNK_SIZE_DIVIDER
            chunk_overlap = effective_ctx // RAG_CHUNK_OVERLAP_DIVIDER
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            for path in file_paths:
                try:
                    if path.suffix.lower() in ['.txt', '.md', '.py', '.json', '.yaml', '.yml']:
                        loader = TextLoader(str(path), encoding='utf-8')
                        documents = loader.load()
                        for doc in documents:
                            chunks = splitter.split_text(doc.page_content)
                            for chunk in chunks:
                                all_docs.append(type('Doc', (), {'page_content': chunk, 'metadata': {'source': str(path)}})())
                except Exception as e:
                    print(f"[RAG] Error loading {path}: {e}")
                    continue
            
            if not all_docs:
                print("[RAG] No documents could be loaded")
                return

            texts = [d.page_content for d in all_docs]
            
            # Create embeddings using sentence-transformers
            embeddings = self._embed_texts(texts)
            
            if embeddings is None:
                print("[RAG] Failed to create embeddings")
                return

            self.file_index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
            self.file_index.add(embeddings)
            self.file_chunks = texts
            print(f"[RAG] Ingested {len(texts)} chunks from {len(file_paths)} files")

    def add_temporary_input(self, large_input_text):
            """
            Chunk and index large pasted user input for RAG retrieval.
            This enables unlimited context for direct text input.
            
            Args:
                large_input_text: The full text that exceeds context limits
            """
            if not large_input_text or not large_input_text.strip():
                return
            
            self._ensure_embedding_model()
            
            if self.embedding is None:
                print("[RAG] Cannot chunk temporary input - embedding model unavailable")
                return
            
            # Use loaded context size for chunk sizing
            effective_ctx = LOADED_CONTEXT_SIZE or CONTEXT_SIZE
            chunk_size = effective_ctx // RAG_CHUNK_SIZE_DIVIDER
            chunk_overlap = effective_ctx // RAG_CHUNK_OVERLAP_DIVIDER
            
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            # Split the large input into chunks
            chunks = splitter.split_text(large_input_text)
            
            if not chunks:
                print("[RAG-TEMP] No chunks created from input")
                return
            
            print(f"[RAG-TEMP] Split large input into {len(chunks)} chunks ({len(large_input_text)} chars)")
            
            # Create embeddings using sentence-transformers
            embeddings = self._embed_texts(chunks)
            
            if embeddings is None:
                print("[RAG-TEMP] Failed to create embeddings")
                return
            
            # Create new temporary index
            self.temp_index = faiss.IndexFlatIP(embeddings.shape[1])
            faiss.normalize_L2(embeddings)
            self.temp_index.add(embeddings)
            self.temp_chunks = chunks
            
            print(f"[RAG-TEMP] Indexed {len(chunks)} chunks for retrieval")

    def clear_temporary_input(self):
        """Clear temporary input chunks (called after response generation)."""
        self.temp_index = None
        self.temp_chunks = []
        print("[RAG-TEMP] Cleared temporary input chunks")

    def get_relevant_context(self, query, k=4, include_temp=True):
        """
        Return top-k most relevant chunks from BOTH file attachments AND temporary input.
        
        Args:
            query: The search query (typically current user input)
            k: Number of chunks to retrieve from each source
            include_temp: Whether to include temporary large input chunks
        
        Returns:
            Concatenated string of relevant chunks, or None if no data available
        """
        if not query.strip():
            return None
        
        self._ensure_embedding_model()
        
        if self.embedding is None:
            return None
        
        # Check if we have any data
        has_files = self.file_index is not None and len(self.file_chunks) > 0
        has_temp = include_temp and self.temp_index is not None and len(self.temp_chunks) > 0
        
        if not has_files and not has_temp:
            return None
        
        # Embed the query using sentence-transformers
        q_vec = self._embed_texts([query])
        if q_vec is None:
            return None
        
        faiss.normalize_L2(q_vec)
        
        all_results = []
        
        # Search file chunks
        if has_files:
            try:
                scores, idxs = self.file_index.search(q_vec, min(k * 2, len(self.file_chunks)))
                
                # Deduplicate file chunks
                seen_chunks = set()
                for i in idxs[0]:
                    if i < len(self.file_chunks) and len(all_results) < k:
                        chunk = self.file_chunks[i]
                        chunk_key = chunk[:50].strip()
                        if chunk_key not in seen_chunks:
                            seen_chunks.add(chunk_key)
                            all_results.append(("FILE", chunk))
                
                print(f"[RAG] Retrieved {len([r for r in all_results if r[0] == 'FILE'])} file chunks")
            except Exception as e:
                print(f"[RAG] Error searching file chunks: {e}")
        
        # Search temporary input chunks
        if has_temp:
            try:
                scores, idxs = self.temp_index.search(q_vec, min(k * 2, len(self.temp_chunks)))
                
                # Deduplicate temp chunks
                seen_chunks = set()
                temp_results = []
                for i in idxs[0]:
                    if i < len(self.temp_chunks) and len(temp_results) < k:
                        chunk = self.temp_chunks[i]
                        chunk_key = chunk[:50].strip()
                        if chunk_key not in seen_chunks:
                            seen_chunks.add(chunk_key)
                            temp_results.append(("TEMP", chunk))
                
                all_results.extend(temp_results)
                print(f"[RAG-TEMP] Retrieved {len(temp_results)} temporary input chunks")
            except Exception as e:
                print(f"[RAG-TEMP] Error searching temp chunks: {e}")
        
        if not all_results:
            return None
        
        # Format results with source indicators
        formatted_chunks = []
        for source, chunk in all_results:
            if source == "FILE":
                formatted_chunks.append(f"[From Attached Files]\n{chunk}")
            else:
                formatted_chunks.append(f"[From Your Input]\n{chunk}")
        
        return "\n\n".join(formatted_chunks)

context_injector = ContextInjector()

# Functions...
def validate_backend_type(backend):
    """Validate and normalize backend type."""
    if backend not in ["CPU_CPU", "VULKAN_CPU", "VULKAN_VULKAN"]:
        print(f"[BACKEND] Invalid backend '{backend}', defaulting to CPU_CPU")
        return "CPU_CPU"
    return backend

def set_status(msg: str, console=False, priority=False):
    """
    Update both UI and/or terminal with priority support.
    
    Args:
        msg: Status message to display
        console: Also print to terminal
        priority: If True, locks status bar (e.g., for model loading)
    """
    global _status_lock, _status_lock_message
    
    # If status is locked and this is not a priority message, store for later
    if _status_lock and not priority:
        if console or len(msg.split()) > 3:
            print(f"[Background] {msg}")
        return
    
    # Priority message - acquire lock
    if priority and "Load" in msg or "loading" in msg.lower():
        _status_lock = "model_loading"
        _status_lock_message = msg
    
    # Release lock on completion
    if _status_lock == "model_loading" and ("ready" in msg.lower() or "error" in msg.lower()):
        _status_lock = None
        _status_lock_message = ""
    
    # Update UI
    if global_status is not None:
        global_status.value = msg
    
    if console or len(msg.split()) > 3:
        print(msg)