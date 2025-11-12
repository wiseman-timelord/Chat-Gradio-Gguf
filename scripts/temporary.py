# scripts/temporary.py

# Imports
import time
from scripts.prompts import prompt_templates
from fastembed import TextEmbedding
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pathlib import Path

# Configuration variables with defaults
PLATFORM = None          # will be set by launcher.py
VULKAN_AVAILABLE = False   # set by settings.load_config() - whether Vulkan binary exists
BACKEND_TYPE = "CPU-Only"  # current runtime mode: "CPU-Only" or "Vulkan"
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
SESSION_LOG_HEIGHT = 500
INPUT_LINES = 27
VRAM_OPTIONS = [0, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 49152, 65536]
CTX_OPTIONS = [8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072]
BATCH_OPTIONS = [128, 256, 512, 1024, 2048, 4096, 8096]
TEMP_OPTIONS = [0.0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0]
REPEAT_OPTIONS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
HISTORY_SLOT_OPTIONS = [4, 8, 10, 12, 16]
ATTACH_SLOT_OPTIONS = [2, 4, 6, 8, 10]
SESSION_LOG_HEIGHT_OPTIONS = [450, 475, 500, 550, 650, 800, 1050, 1300]

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
LLAMA_CLI_PATH = None  # will be set by launcher.py
global_status = None 
PRINT_RAW_OUTPUT = False
SHOW_THINK_PHASE = False 
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
ALLOWED_EXTENSIONS = {"bat", "py", "ps1", "txt", "json", "yaml", "psd1", "xaml"}
MAX_POSSIBLE_HISTORY_SLOTS = 16
MAX_POSSIBLE_ATTACH_SLOTS = 10
demo = None

# RAG CONSTANTS
RAG_CHUNK_SIZE_DIVIDER = 6
RAG_CHUNK_OVERLAP_DIVIDER = 24
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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
    "reasoning": ["reason", "r1", "think"],
    "nsfw": ["nsfw", "adult", "mature", "explicit", "lewd"],
    "roleplay": ["rp", "role", "adventure"],
    "harmony": ["gpt-oss"]
}

# prompt template table
current_model_settings = {
    "category": "chat"
}

# RAG Context Injector
class ContextInjector:
    """
    End-to-end RAG with improved chunking for large files
    """
    def __init__(self):
        self.embedding = None
        self.index = None
        self.chunks = []
        self._model_load_attempted = False
    
    def _ensure_embedding_model(self):
        """Initialize embedding model on first use with proper cache path."""
        if self._model_load_attempted:
            return
        
        self._model_load_attempted = True
        
        import os
        from pathlib import Path
        
        # FIX: Use absolute path consistently
        cache_dir = Path(__file__).parent.parent / "data" / "fastembed_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set environment variables BEFORE importing fastembed
        os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir.absolute())
        os.environ["FASTEMBED_OFFLINE"] = "1"  # Prefer local cache
        
        try:
            from fastembed import TextEmbedding
            
            print(f"[RAG] Loading embedding model from: {cache_dir}")
            
            self.embedding = TextEmbedding(
                model_name=EMBEDDING_MODEL_NAME,
                cache_dir=str(cache_dir.absolute()),
                providers=["CPUExecutionProvider"]
            )
            print("[RAG] Embedding model loaded from cache")
        except Exception as e:
            print(f"[RAG] Warning: Could not load embedding model: {e}")
            print("[RAG] RAG features will be disabled. Run installer to download model.")
            self.embedding = None

    def set_session_vectorstore(self, file_paths):
        """Create/refresh vector store from list of file paths with improved chunking."""
        if not file_paths:
            self.index = None
            self.chunks = []
            return
        
        self._ensure_embedding_model()
        
        if self.embedding is None:
            print("[RAG] Cannot create vectorstore - embedding model unavailable")
            return

        all_docs = []
        
        # Dynamic chunk sizing based on context size
        from scripts.temporary import CONTEXT_SIZE
        base_chunk_size = CONTEXT_SIZE // RAG_CHUNK_SIZE_DIVIDER
        base_chunk_overlap = CONTEXT_SIZE // RAG_CHUNK_OVERLAP_DIVIDER
        
        # Adjust chunk size based on file count and size
        total_files = len(file_paths)
        avg_file_size = sum(Path(fp).stat().st_size for fp in file_paths if Path(fp).exists()) / total_files if total_files > 0 else 0
        
        # Smaller chunks for larger files to improve retrieval
        if avg_file_size > 100000:  # Files larger than 100KB
            chunk_size = min(base_chunk_size, 1000)
            chunk_overlap = min(base_chunk_overlap, 100)
        else:
            chunk_size = base_chunk_size
            chunk_overlap = base_chunk_overlap
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        for fp in file_paths:
            if Path(fp).suffix[1:].lower() not in ALLOWED_EXTENSIONS:
                continue
            try:
                docs = TextLoader(fp).load()
                # Split into smaller chunks for better retrieval
                chunks = splitter.split_documents(docs)
                all_docs.extend(chunks)
                print(f"[RAG] Split {fp} into {len(chunks)} chunks")
            except Exception as e:
                print(f"[RAG] Skip {fp}: {e}")

        if not all_docs:
            self.index = None
            self.chunks = []
            return

        texts = [d.page_content for d in all_docs]
        
        # Create embeddings in batches to avoid memory issues
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.embedding.embed(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        embeddings = np.array(all_embeddings).astype('float32')

        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks = texts
        print(f"[RAG] Ingested {len(texts)} chunks from {len(file_paths)} files")

    def get_relevant_context(self, query, k=4):
        """Return top-k most relevant chunks concatenated as string."""
        if self.index is None or not query.strip():
            return None
        
        self._ensure_embedding_model()
            
        if self.embedding is None:
            return None
            
        q_vec = self.embedding.embed([query])
        q_vec = np.array(q_vec).astype('float32')
        faiss.normalize_L2(q_vec)
        
        # Search for more chunks than needed to allow for deduplication
        scores, idxs = self.index.search(q_vec, k * 2)
        
        # Deduplicate and select top k unique chunks
        seen_chunks = set()
        top_chunks = []
        
        for i in idxs[0]:
            if i < len(self.chunks) and len(top_chunks) < k:
                chunk = self.chunks[i]
                # Simple deduplication based on first 50 characters
                chunk_key = chunk[:50].strip()
                if chunk_key not in seen_chunks:
                    seen_chunks.add(chunk_key)
                    top_chunks.append(chunk)
        
        return "\n\n".join(top_chunks)
        
context_injector = ContextInjector()


# Status Updater
def set_status(msg: str, console=False):
    """Update both UI and/or terminal"""
    if global_status is not None:
        global_status.value = msg
    if console or len(msg.split()) > 3:
        print(msg)