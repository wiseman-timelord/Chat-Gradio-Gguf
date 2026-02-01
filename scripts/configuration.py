# scripts/configuration.py

import json
import configparser
import time
import threading
import os
from pathlib import Path

import faiss
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# =============================================================================
# DEFAULT CONFIGURATION DICTIONARIES
# =============================================================================

DEFAULTS = {
    "MODEL_FOLDER": "path/to/your/models",
    "CONTEXT_SIZE": 32768,
    "VRAM_SIZE": 8192,
    "BATCH_SIZE": 1024,
    "TEMPERATURE": 0.66,
    "REPEAT_PENALTY": 1.0,
    "DYNAMIC_GPU_LAYERS": True,
    "MMAP": True,
    "MLOCK": True,
    "MAX_HISTORY_SLOTS": 10,
    "MAX_ATTACH_SLOTS": 8,
    "SESSION_LOG_HEIGHT": 650,
    "INPUT_LINES": 27,
    "PRINT_RAW_OUTPUT": False,
    "SHOW_THINK_PHASE": False,
    "CPU_ONLY_MODE": True,
    "VRAM_OPTIONS": [0, 756, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 49152, 65536],
    "CTX_OPTIONS": [1024, 2048, 4096, 8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072],
    "BATCH_OPTIONS": [128, 256, 512, 1024, 2048, 4096, 8192],
    "TEMP_OPTIONS": [0.0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0],
    "REPEAT_OPTIONS": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    "HISTORY_SLOT_OPTIONS": [4, 8, 10, 12, 16],
    "ATTACH_SLOT_OPTIONS": [2, 4, 6, 8, 10],
    "SESSION_LOG_HEIGHT_OPTIONS": [250, 450, 550, 600, 625, 650, 700, 800, 1000, 1400],
}

DEFAULT_CONFIG = {
    "model_settings": {
        "layer_allocation_mode": "SRAM_ONLY",
        "model_dir": "path/to/your/models",
        "model_name": "Select_a_model...",
        "context_size": 32768,
        "vram_size": 8192,
        "temperature": 0.66,
        "repeat_penalty": 1.0,
        "llama_cli_path": None,
        "llama_bin_path": None,
        "selected_gpu": None,
        "selected_cpu": None,
        "mmap": True,
        "mlock": True,
        "n_batch": 1024,
        "dynamic_gpu_layers": True,
        "max_history_slots": 12,
        "max_attach_slots": 6,
        "session_log_height": 650,
        "show_think_phase": False,
        "print_raw_output": False,
        "cpu_threads": None,
        "bleep_on_events": False,
        "use_python_bindings": True,
        "filter_mode": "gradio3",  # Output filtering mode: gradio3, gradio5, or custom
        # TTS Settings
        "tts_enabled": False,
        "tts_voice": None,
        "tts_output_device": "default",
        "tts_sample_rate": 44100,
        "max_tts_length": 4500,
    }
}

CONFIG_PATH = Path("data/persistent.json")

# =============================================================================
# SYSTEM STATE VARIABLES
# =============================================================================

# System constants (platform, backend, etc.) loaded from data/constants.ini
PLATFORM = None
BACKEND_TYPE = "CPU_CPU"
VULKAN_AVAILABLE = False
LAYER_ALLOCATION_MODE = "SRAM_ONLY"
OS_VERSION = None  # Ubuntu version or Windows version string
WINDOWS_VERSION = None  # Windows-specific version (8.1, 10, 11)
EMBEDDING_BACKEND = "sentence_transformers"  # Always sentence_transformers now
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
GRADIO_VERSION = None
LOADED_CONTEXT_SIZE = None

# Output Filtering Configuration
FILTER_MODE = "gradio3"  # Will be set based on GRADIO_VERSION on startup

FILTER_PRESETS = {
    "gradio3": [
        ("<p>", "\n"),
        ("</p>", "\n"),
        ("\n\n\n\r", "\n\r"),
        ("\n\n\n\t", "\n\t"),
        ("\r\n\n\n", "\r\n"),
        ("\t\n\n\n", "\t\n"),
        ("\n\r\n", "\n"),
        ("\n\n\r", "\r"),
        ("\n\n\t", "\t"),
        ("\r\n\n", "\r"),
        ("\t\n\n", "\t"),
        ("\n\r", "\r"),
        ("\n\t", "\t"),
        ("\r\n", "\r"),
        ("\t\n", "\t"),
    ],
    "gradio5": [
        ("\n\n\n\r", "\n\r"),
        ("\n\n\n\t", "\n\t"),
        ("\r\n\n\n", "\r\n"),
        ("\t\n\n\n", "\t\n"),
        ("\n\n\r", "\n\r"),
        ("\n\n\t", "\n\t"),
        ("\r\n\n", "\r\n"),
        ("\t\n\n", "\t\n"),
        ("\n\n\n", "\n\n"),
    ],
}

ACTIVE_FILTER = []
CUSTOM_FILTER_PATH = "data/custom_filter.txt"

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
TTS_TYPE = "builtin"                     # TTS type: "builtin" or "coqui"
COQUI_VOICE_ID = None                    # Coqui speaker ID (e.g., "p243")
COQUI_VOICE_ACCENT = None                # Coqui voice accent (e.g., "british")
COQUI_MODEL = "tts_models/en/vctk/vits"  # Coqui model name
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
LLAMA_CLI_PATH = None  # will be set from constants.ini
LLAMA_BIN_PATH = None  # will be set from constants.ini
global_status = None
_status_lock = None  # Tracks which operation has status priority
_status_lock_message = ""  # Message to restore when lock releases 
PRINT_RAW_OUTPUT = False
SHOW_THINK_PHASE = False 
BLEEP_ON_EVENTS = False
THINK_MIN_CHARS_BEFORE_CLOSE = 100
USER_INPUT_MAX_LINES = 10  # Recalculated by display.py based on SESSION_LOG_HEIGHT

# Dynamic Model Loading
LAST_INTERACTION_TIME = None
INACTIVITY_TIMEOUT = 3600  # 1 hour in seconds
_inactivity_checker_thread = None
_inactivity_checker_stop = False

# CPU Configuration
CPU_THREADS = None  # Will be auto-detected
CPU_THREAD_OPTIONS = []  # Will be populated with available thread counts
CPU_PHYSICAL_CORES = 1
CPU_LOGICAL_CORES = 1
SELECTED_CPU = "Auto-Select"

# =============================================================================
# Sound Hardware Configuration (shared by Bleep and TTS)
# =============================================================================
SOUND_OUTPUT_DEVICE = "Default Sound Device"        # Selected audio output device
SOUND_SAMPLE_RATE = 44100              # Audio sample rate (44100 or 48000)
SOUND_SAMPLE_RATE_OPTIONS = [44100, 48000]  # Available sample rates

# =============================================================================
# TTS (Text-to-Speech) Configuration
# =============================================================================
TTS_ENABLED = False                    # Master enable/disable for TTS
TTS_ENGINE = "none"                    # Detected engine: "pyttsx3", "espeak-ng", "none"
TTS_AUDIO_BACKEND = "none"             # Audio backend: "windows", "pulseaudio", "pipewire", "none"
TTS_VOICE = None                       # Selected voice ID
TTS_VOICE_NAME = None                  # Selected voice display name
MAX_TTS_LENGTH = 4500
TTS_TYPE = "builtin"                     # TTS type: "builtin" or "coqui"
COQUI_VOICE_ID = None                    # Coqui speaker ID (e.g., "p243")
COQUI_VOICE_ACCENT = None                # Coqui voice accent (e.g., "british")
COQUI_MODEL = "tts_models/en/vctk/vits"  # Coqui model name

# Arrays
session_attached_files = []
session_vector_files = []

# Search Configuration - DDG Search and Web Search are mutually exclusive
DDG_SEARCH_ENABLED = False     # Hybrid DDG search (snippets + deep fetch)
WEB_SEARCH_ENABLED = False     # Comprehensive web search (multi-source + parallel fetch)

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
    "error": "An error occurred",
    "tts_speaking": "Speaking...",
    "tts_stopped": "Speech stopped"
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

# =============================================================================
# CONTEXT INJECTOR CLASS (RAG)
# =============================================================================

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
        self._embedding_dim = None      # Track dimension for validation
        self._model_name = None         # Track which model was loaded
    
    def _ensure_embedding_model(self):
        """Initialize embedding model on first use with proper cache path (fully offline)."""
        if self._model_load_attempted and self.embedding is not None:
            return  # Already loaded successfully
        
        # Reset attempt flag if previous load failed (allows retry with different model)
        if self.embedding is None:
            self._model_load_attempted = False
        
        self._model_load_attempted = True
        
        # Set cache directories to project-local paths
        cache_dir = Path(__file__).parent.parent / "data" / "embedding_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL: Ensure offline mode is set
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir.absolute())
        os.environ["HF_HOME"] = str(cache_dir.parent.absolute())
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir.absolute())
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Try loading from cache first
            model_name = EMBEDDING_MODEL_NAME
            self._model_name = model_name
            
            # Check if model exists in cache
            model_cache_path = cache_dir / model_name.replace("/", "_")
            
            print(f"[RAG] Loading embedding model: {model_name}")
            
            if model_cache_path.exists():
                print(f"[RAG] Loading from cache: {model_cache_path}")
                self.embedding = SentenceTransformer(str(model_cache_path))
            else:
                # Try loading by name (will use HF cache)
                print(f"[RAG] Downloading/loading to: {cache_dir}")
                self.embedding = SentenceTransformer(model_name, cache_folder=str(cache_dir))
            
            # Validate model loaded and get dimension
            if self.embedding is None:
                raise RuntimeError("Model loading returned None")
            
            # Test embedding to validate dimension and catch OOM early
            test_embedding = self.embedding.encode(["test"], convert_to_numpy=True, show_progress_bar=False)
            self._embedding_dim = test_embedding.shape[1]
            
            print(f"[RAG] Embedding model loaded successfully (dim={self._embedding_dim})")
            
        except Exception as e:
            print(f"[RAG] Failed to load embedding model: {e}")
            print(f"[RAG] Tried model: {EMBEDDING_MODEL_NAME}")
            print(f"[RAG] If using 'large' model, ensure you have ~2GB free RAM")
            self.embedding = None
            self._embedding_dim = None
            # Reset flag so we can try again later (e.g., if user fixes memory issue)
            self._model_load_attempted = False
    
    def _embed_texts(self, texts, batch_size=32):
        """Create embeddings for a list of texts using sentence-transformers."""
        if self.embedding is None:
            return None
        
        if not texts:
            return None
        
        try:
            # Process in batches to avoid OOM with large models
            embeddings = self.embedding.encode(
                texts, 
                convert_to_numpy=True, 
                show_progress_bar=False,
                batch_size=batch_size  # Limit batch size for large models
            )
            
            # Validate embeddings aren't empty or NaN
            if embeddings is None or embeddings.size == 0:
                print("[RAG] Warning: Empty embeddings returned")
                return None
            
            # Check for NaN or all-zero vectors (indicates model failure)
            if np.isnan(embeddings).any() or not np.any(embeddings):
                print("[RAG] Warning: Invalid embeddings (NaN or all zeros)")
                return None
            
            return embeddings.astype(np.float32)
        except Exception as e:
            print(f"[RAG] Embedding error: {e}")
            # If OOM, suggest using smaller model
            if "out of memory" in str(e).lower() or "unable to allocate" in str(e).lower():
                print(f"[RAG] Memory error - consider using smaller embedding model (current: {EMBEDDING_MODEL_NAME})")
            return None

    def set_session_vectorstore(self, file_paths):
        """Create FAISS index from attached files for RAG retrieval."""
        self._ensure_embedding_model()
        
        if self.embedding is None:
            print("[RAG] Cannot create vectorstore - embedding model unavailable")
            return
        
        if not file_paths:
            self.file_index = None
            self.file_chunks = []
            return
        
        # Use loaded context size for chunk sizing
        effective_ctx = LOADED_CONTEXT_SIZE or CONTEXT_SIZE
        chunk_size = effective_ctx // RAG_CHUNK_SIZE_DIVIDER
        chunk_overlap = effective_ctx // RAG_CHUNK_OVERLAP_DIVIDER
        
        texts = []
        for path in file_paths:
            try:
                with open(path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len
                    )
                    chunks = splitter.split_text(content)
                    texts.extend(chunks)
            except Exception as e:
                print(f"[RAG] Error loading {path}: {e}")
        
        if not texts:
            self.file_index = None
            self.file_chunks = []
            return
        
        # Create embeddings using sentence-transformers with batching for large models
        embeddings = self._embed_texts(texts, batch_size=16)  # Smaller batch for large models
        
        if embeddings is None:
            print("[RAG] Failed to create embeddings - index not created")
            self.file_index = None
            self.file_chunks = []
            return
        
        # Validate embedding dimension matches expected
        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]
        
        if embeddings.shape[1] != self._embedding_dim:
            print(f"[RAG] Dimension mismatch! Expected {self._embedding_dim}, got {embeddings.shape[1]}")
            self.file_index = None
            return

        self.file_index = faiss.IndexFlatIP(self._embedding_dim)
        faiss.normalize_L2(embeddings)
        self.file_index.add(embeddings)
        self.file_chunks = texts
        print(f"[RAG] Ingested {len(texts)} chunks from {len(file_paths)} files (dim={self._embedding_dim})")

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
        
        # Create embeddings using sentence-transformers with smaller batch for stability
        embeddings = self._embed_texts(chunks, batch_size=16)
        
        if embeddings is None:
            print("[RAG-TEMP] Failed to create embeddings")
            return
        
        # Validate dimension
        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]
        
        if embeddings.shape[1] != self._embedding_dim:
            print(f"[RAG-TEMP] Dimension mismatch! Expected {self._embedding_dim}, got {embeddings.shape[1]}")
            return
        
        # Create new temporary index
        self.temp_index = faiss.IndexFlatIP(self._embedding_dim)
        faiss.normalize_L2(embeddings)
        self.temp_index.add(embeddings)
        self.temp_chunks = chunks
        
        print(f"[RAG-TEMP] Indexed {len(chunks)} chunks for retrieval (dim={self._embedding_dim})")

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
        q_vec = self._embed_texts([query], batch_size=1)  # Single query, batch size 1
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

# =============================================================================
# CONFIGURATION FUNCTIONS (Previously cfg.py)
# =============================================================================

def load_system_ini():
    """Load system constants from constants.ini (created by installer)."""
    ini_path = Path("data/constants.ini")
    if not ini_path.exists():
        raise RuntimeError(
            f"System configuration file not found: {ini_path}\n"
            "Re-run the installer to generate constants.ini."
        )
    
    try:
        config = configparser.ConfigParser()
        config.read(ini_path, encoding='utf-8')
        
        if 'system' not in config:
            raise RuntimeError("constants.ini missing [system] section")
        
        system = config['system']
        
        global PLATFORM, BACKEND_TYPE, VULKAN_AVAILABLE, EMBEDDING_MODEL_NAME
        global EMBEDDING_BACKEND, GRADIO_VERSION, LLAMA_CLI_PATH, LLAMA_BIN_PATH
        global OS_VERSION, WINDOWS_VERSION, TTS_ENGINE, TTS_AUDIO_BACKEND
        global TTS_TYPE, COQUI_VOICE_ID, COQUI_VOICE_ACCENT, COQUI_MODEL
        
        PLATFORM = system.get('platform')
        BACKEND_TYPE = system.get('backend_type', 'CPU_CPU')
        VULKAN_AVAILABLE = system.getboolean('vulkan_available', False)
        EMBEDDING_MODEL_NAME = system.get('embedding_model', 'BAAI/bge-small-en-v1.5')
        EMBEDDING_BACKEND = system.get('embedding_backend', 'sentence_transformers')
        GRADIO_VERSION = system.get('gradio_version', '3.50.2')
        LLAMA_CLI_PATH = system.get('llama_cli_path', None)
        LLAMA_BIN_PATH = system.get('llama_bin_path', None)
        
        print(f"[INI] Platform: {PLATFORM}")
        print(f"[INI] Backend: {BACKEND_TYPE}")
        print(f"[INI] Vulkan: {VULKAN_AVAILABLE}")
        print(f"[INI] Embedding Model: {EMBEDDING_MODEL_NAME}")
        print(f"[INI] Gradio Version: {GRADIO_VERSION}")
        
        OS_VERSION = system.get('os_version', 'unknown')
        print(f"[INI] OS Version: {OS_VERSION}")

        if PLATFORM == "windows":
            WINDOWS_VERSION = system.get('windows_version', OS_VERSION)
            print(f"[INI] Windows Version: {WINDOWS_VERSION}")
        else:
            WINDOWS_VERSION = None
        
        # Load TTS configuration
        if 'tts' in config:
            tts_section = config['tts']
            TTS_TYPE = tts_section.get('tts_type', 'builtin')
            
            if TTS_TYPE == "coqui":
                COQUI_VOICE_ID = tts_section.get('coqui_voice_id', 'p243')
                COQUI_VOICE_ACCENT = tts_section.get('coqui_voice_accent', 'british')
                COQUI_MODEL = tts_section.get('coqui_model', 'tts_models/en/vctk/vits')
                print(f"[INI] TTS Type: Coqui")
                print(f"[INI] Coqui Voice: {COQUI_VOICE_ID} ({COQUI_VOICE_ACCENT})")
                print(f"[INI] Coqui Model: {COQUI_MODEL}")
            else:
                print(f"[INI] TTS Type: Built-in (pyttsx3/espeak-ng)")
            
            # Legacy support for old INI format
            if 'tts_engine' in tts_section:
                TTS_ENGINE = tts_section.get('tts_engine', 'none')
            if 'tts_audio_backend' in tts_section:
                TTS_AUDIO_BACKEND = tts_section.get('tts_audio_backend', 'none')
        else:
            print("[INI] TTS section not found - will detect at runtime")
            TTS_TYPE = "builtin"
        
        return True
        
    except Exception as e:
        raise RuntimeError(f"Cannot read constants.ini: {e}") from e

def load_config():
    """Load configuration with strict validation - no defaults, error on missing keys."""
    global MODEL_FOLDER, MODEL_NAME, CONTEXT_SIZE, VRAM_SIZE, TEMPERATURE
    global REPEAT_PENALTY, MMAP, MLOCK, BATCH_SIZE, DYNAMIC_GPU_LAYERS
    global MAX_HISTORY_SLOTS, MAX_ATTACH_SLOTS, SESSION_LOG_HEIGHT, SHOW_THINK_PHASE
    global PRINT_RAW_OUTPUT, CPU_THREADS, BLEEP_ON_EVENTS, USE_PYTHON_BINDINGS
    global LAYER_ALLOCATION_MODE, SELECTED_GPU, SELECTED_CPU, SOUND_OUTPUT_DEVICE
    global SOUND_SAMPLE_RATE, TTS_ENABLED, TTS_VOICE, TTS_VOICE_NAME, MAX_TTS_LENGTH
    global AVAILABLE_MODELS
    
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"Configuration file not found: {CONFIG_PATH}\nRe-run the installer.")

    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            file_content = f.read().strip()
            if not file_content:
                raise RuntimeError(f"Configuration file {CONFIG_PATH} is empty")
            config = json.loads(file_content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Cannot parse configuration file {CONFIG_PATH}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Cannot read configuration file {CONFIG_PATH}: {e}") from e

    if not isinstance(config, dict) or "model_settings" not in config:
        raise RuntimeError("Configuration file missing 'model_settings' section or invalid format.")

    model_settings = config["model_settings"]

    # Required fields with sane fallback only if missing
    MODEL_FOLDER = model_settings.get("model_dir", "path/to/your/models")
    MODEL_NAME = model_settings.get("model_name", "Select_a_model...")
    CONTEXT_SIZE = model_settings.get("context_size", 32768)
    VRAM_SIZE = model_settings.get("vram_size", 8192)
    TEMPERATURE = model_settings.get("temperature", 0.66)
    REPEAT_PENALTY = model_settings.get("repeat_penalty", 1.0)
    MMAP = model_settings.get("mmap", True)
    MLOCK = model_settings.get("mlock", True)
    BATCH_SIZE = model_settings.get("n_batch", 1024)
    DYNAMIC_GPU_LAYERS = model_settings.get("dynamic_gpu_layers", True)
    MAX_HISTORY_SLOTS = model_settings.get("max_history_slots", 12)
    MAX_ATTACH_SLOTS = model_settings.get("max_attach_slots", 6)
    SESSION_LOG_HEIGHT = model_settings.get("session_log_height", 650)
    SHOW_THINK_PHASE = model_settings.get("show_think_phase", False)
    PRINT_RAW_OUTPUT = model_settings.get("print_raw_output", False)
    CPU_THREADS = model_settings.get("cpu_threads", None)
    BLEEP_ON_EVENTS = model_settings.get("bleep_on_events", False)
    USE_PYTHON_BINDINGS = model_settings.get("use_python_bindings", True)
    LAYER_ALLOCATION_MODE = model_settings.get("layer_allocation_mode", "SRAM_ONLY")

    # Hardware selection (strings)
    SELECTED_GPU = model_settings.get("selected_gpu")
    SELECTED_CPU = model_settings.get("selected_cpu", "Auto-Select")   # default to valid UI choice

    # Sound / TTS
    raw_sound_device = model_settings.get("sound_output_device", "default")
    # Normalize: Windows always uses Default, Linux uses the saved value or default
    if PLATFORM == "windows":
        SOUND_OUTPUT_DEVICE = "Default Sound Device"
    else:
        SOUND_OUTPUT_DEVICE = raw_sound_device if raw_sound_device else "default"
    SOUND_SAMPLE_RATE = model_settings.get("sound_sample_rate", 44100)
    TTS_ENABLED = model_settings.get("tts_enabled", False)
    TTS_VOICE = model_settings.get("tts_voice")
    TTS_VOICE_NAME = model_settings.get("tts_voice_name")
    MAX_TTS_LENGTH = model_settings.get("max_tts_length", 4500)

    # Post-load adjustment: ensure SELECTED_CPU is a valid string label
    from scripts.utility import get_cpu_info
    cpu_info = get_cpu_info()
    cpu_labels = ["Auto-Select"] + [c["label"] for c in cpu_info]
    if SELECTED_CPU not in cpu_labels:
        SELECTED_CPU = "Auto-Select"

    # Model list refresh
    from scripts.inference import get_available_models
    AVAILABLE_MODELS = get_available_models()
    if MODEL_NAME not in AVAILABLE_MODELS:
        real_models = [m for m in AVAILABLE_MODELS if m != "Select_a_model..."]
        MODEL_NAME = real_models[0] if real_models else "Select_a_model..."

    # Post-load: Auto-select GPU if not configured
    if SELECTED_GPU is None or SELECTED_GPU == "Auto":
        try:
            from scripts.utility import get_available_gpus
            available_gpus = get_available_gpus()
            # Filter out "CPU Only" placeholder if GPUs exist
            real_gpus = [g for g in available_gpus if g != "CPU Only"]
            
            if len(real_gpus) == 1:
                SELECTED_GPU = real_gpus[0]
                print(f"[CONFIG] Auto-selected sole GPU: {SELECTED_GPU}")
            elif len(real_gpus) > 1:
                SELECTED_GPU = real_gpus[1]  # Prefer secondary GPU (index 1)
                print(f"[CONFIG] Auto-selected secondary GPU: {SELECTED_GPU}")
            else:
                SELECTED_GPU = "Auto-Select"
        except Exception as e:
            print(f"[CONFIG] GPU auto-selection failed: {e}")
            SELECTED_GPU = "Auto-Select"

    print(f"[CONFIG] Loaded -> Model: {MODEL_NAME} | CPU: {SELECTED_CPU}")
    set_status("Configuration loaded", console=True)
    return "Configuration loaded."

def save_config():
    """Save current configuration to persistent storage."""
    config = {
        "model_settings": {
            "model_dir": MODEL_FOLDER,
            "model_name": MODEL_NAME,
            "context_size": CONTEXT_SIZE,
            "vram_size": VRAM_SIZE,
            "temperature": TEMPERATURE,
            "repeat_penalty": REPEAT_PENALTY,
            "selected_gpu": SELECTED_GPU,
            "selected_cpu": SELECTED_CPU or "Auto-Select",
            "mmap": MMAP,
            "mlock": MLOCK,
            "n_batch": BATCH_SIZE,
            "dynamic_gpu_layers": DYNAMIC_GPU_LAYERS,
            "max_history_slots": MAX_HISTORY_SLOTS,
            "max_attach_slots": MAX_ATTACH_SLOTS,
            "session_log_height": SESSION_LOG_HEIGHT,
            "show_think_phase": SHOW_THINK_PHASE,
            "print_raw_output": PRINT_RAW_OUTPUT,
            "cpu_threads": CPU_THREADS,
            "bleep_on_events": BLEEP_ON_EVENTS,
            "use_python_bindings": USE_PYTHON_BINDINGS,
            "layer_allocation_mode": LAYER_ALLOCATION_MODE,
            "sound_output_device": SOUND_OUTPUT_DEVICE,
            "sound_sample_rate": SOUND_SAMPLE_RATE,
            "tts_enabled": TTS_ENABLED,
            "tts_voice": TTS_VOICE,
            "tts_voice_name": TTS_VOICE_NAME,
            "max_tts_length": MAX_TTS_LENGTH,
        }
    }

    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    set_status("Settings saved")
    return "Settings saved"

def update_setting(key, value):
    """Update a single setting with optional model reload."""
    reload_required = False
    reload_keys = {"context_size", "n_gpu_layers", "vram_size", "model_folder", "model_name"}
    
    try:
        # Convert value to appropriate type
        if key in {
            "context_size", "vram_size", "n_gpu_layers", "n_batch",
            "max_history_slots", "max_attach_slots", "session_log_height",
            "cpu_threads", "tts_sample_rate", "max_tts_length"          
        }:
            value = int(value)
        elif key in {"temperature", "repeat_penalty"}:
            value = float(value)
        elif key in {"mlock", "dynamic_gpu_layers", "tts_enabled"}:
            value = bool(value)
        
        # Map key to global variable name and set
        attr_name = key.upper() if key.upper() in globals() else key
        globals()[attr_name] = value
        
        reload_required = key in reload_keys
        
        if reload_required:
            from scripts.inference import change_model
            reload_result = change_model(MODEL_NAME.split('/')[-1])
            message = f"Setting '{key}' updated to '{value}', model reload triggered."
            return message, *reload_result
        else:
            message = f"Setting '{key}' updated to '{value}'."
            return message, None, None
            
    except Exception as e:
        message = f"Error updating setting '{key}': {str(e)}"
        return message, None, None

# =============================================================================
# UTILITY FUNCTIONS (Previously in cfg.py)
# =============================================================================

def validate_backend_type(backend):
    """Validate and normalize backend type."""
    if backend not in ["CPU_CPU", "VULKAN_CPU", "VULKAN_VULKAN"]:
        print(f"[BACKEND] Invalid backend '{backend}', defaulting to CPU_CPU")
        return "CPU_CPU"
    return backend

def start_inactivity_checker():
    """Start background thread to check for model inactivity and auto-unload."""
    global _inactivity_checker_thread, _inactivity_checker_stop
    
    # Already running?
    if _inactivity_checker_thread is not None and _inactivity_checker_thread.is_alive():
        return
    
    def checker():
        global LAST_INTERACTION_TIME, _inactivity_checker_stop, llm, MODELS_LOADED, GPU_LAYERS, LOADED_CONTEXT_SIZE
        
        while not _inactivity_checker_stop:
            time.sleep(60)  # Check every minute
            
            if _inactivity_checker_stop:
                break
            
            # Only check if model is loaded
            if MODELS_LOADED and llm is not None:
                if LAST_INTERACTION_TIME is not None:
                    elapsed = time.time() - LAST_INTERACTION_TIME
                    if elapsed > INACTIVITY_TIMEOUT:
                        print(f"[INACTIVITY] {elapsed:.0f}s elapsed (> {INACTIVITY_TIMEOUT}s timeout)")
                        set_status("Auto-unloading model due to inactivity...", console=True)
                        
                        # Unload model
                        from scripts.inference import unload_models
                        status, new_llm, new_models_loaded = unload_models(llm, MODELS_LOADED)
                        llm = new_llm
                        MODELS_LOADED = new_models_loaded
                        GPU_LAYERS = 0
                        LOADED_CONTEXT_SIZE = None
                        
                        set_status("Model auto-unloaded (1 hour idle)", console=True)
                        # Beep to notify user
                        try:
                            from scripts.utility import beep
                            beep()
                        except:
                            pass
    
    # Initialize checker state
    _inactivity_checker_stop = False
    
    # Start thread
    _inactivity_checker_thread = threading.Thread(target=checker, daemon=True, name="InactivityChecker")
    _inactivity_checker_thread.start()
    print("[INACTIVITY] Checker started (1 hour timeout)")

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