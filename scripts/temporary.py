# scripts/temporary.py

# Imports
import time
from scripts.prompts import prompt_templates
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from pathlib import Path

# Configuration variables with defaults
PLATFORM = None          # will be set by launcher.py
MODEL_FOLDER = "path/to/your/models"
CONTEXT_SIZE = 8192
VRAM_SIZE = 8192
BATCH_SIZE = 2048
TEMPERATURE = 0.66
REPEAT_PENALTY = 1.1
DYNAMIC_GPU_LAYERS = True
MMAP = True
MLOCK = True
MAX_HISTORY_SLOTS = 12
MAX_ATTACH_SLOTS = 6
SESSION_LOG_HEIGHT = 500
INPUT_LINES = 27
VRAM_OPTIONS = [2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 49152, 65536]
CTX_OPTIONS = [8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072]
BATCH_OPTIONS = [128, 256, 512, 1024, 2048, 4096, 8096, 16384, 32768]
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
BACKEND_TYPE = "Not Configured"
DATA_DIR = None  # Will be set by launcher.py
llm = None
SPEECH_ENABLED = False
LLAMA_CLI_PATH = None  # will be set by launcher.py
global_status = None 

# CPU COnfiguration
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
PRINT_RAW_OUTPUT = False
demo = None

# Rag CONSTANTS
RAG_CHUNK_SIZE_DIVIDER = 4          # typo fix
RAG_CHUNK_OVERLAP_DIVIDER = 32      # typo fix
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, 22 MB

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
    'qwen2': 'chatml',
    'llama': 'llama-2',
    'qwen3': 'chatml',
    'qwen3moe': 'chatml',
    'deepseek2': 'deepseek',
    'stablelm': 'chatml',
}

# Handling Keywords for Special Model Behaviors
handling_keywords = {
    "code": ["code", "coder", "program", "dev", "copilot", "codex", "Python", "Powershell"],
    "uncensored": ["uncensored", "unfiltered", "unbiased", "unlocked"],
    "reasoning": ["reason", "r1", "think"],
    "nsfw": ["nsfw", "adult", "mature", "explicit", "lewd"],
    "roleplay": ["rp", "role", "adventure"]
}

# prompt template table
current_model_settings = {
    "category": "chat"  # prompt_template removed as not needed
}

# Rag
class ContextInjector:
    """
    End-to-end RAG:
      - ingest files ? chunks ? embeddings ? FAISS
      - retrieve top-k relevant chunks for a query
    """
    def __init__(self):
        self.embedding = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = None          # faiss index
        self.chunks = []           # parallel list to index

    # ingestion
    def set_session_vectorstore(self, file_paths):
        """Create/refresh vector store from list of file paths."""
        if not file_paths:
            self.index = None
            self.chunks = []
            return

        all_docs = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=temporary.CONTEXT_SIZE // RAG_CHUNK_SIZE_DIVIDER,
            chunk_overlap=temporary.CONTEXT_SIZE // RAG_CHUNK_OVERLAP_DIVIDER
        )
        for fp in file_paths:
            if Path(fp).suffix[1:].lower() not in ALLOWED_EXTENSIONS:
                continue
            try:
                docs = TextLoader(fp).load()
                all_docs.extend(splitter.split_documents(docs))
            except Exception as e:
                print(f"[RAG] Skip {fp}: {e}")

        if not all_docs:
            self.index = None
            self.chunks = []
            return

        texts = [d.page_content for d in all_docs]
        embeddings = self.embedding.encode(texts, show_progress_bar=False)
        embeddings = np.array(embeddings).astype('float32')

        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine via inner-product
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.chunks = texts
        print(f"[RAG] Ingested {len(texts)} chunks from {len(file_paths)} files")

    # retrieval
    def get_relevant_context(self, query, k=4):
        """Return top-k most relevant chunks concatenated as string."""
        if self.index is None or not query.strip():
            return None
        q_vec = self.embedding.encode([query], show_progress_bar=False)
        q_vec = np.array(q_vec).astype('float32')
        faiss.normalize_L2(q_vec)
        scores, idxs = self.index.search(q_vec, k)
        top = [self.chunks[i] for i in idxs[0] if i < len(self.chunks)]
        return "\n\n".join(top)

# Status Updater
def set_status(msg: str, console=False):
    """Update both UI and/or terminal"""
    if global_status is not None:          # UI ready?
        global_status.value = msg
    if console or len(msg.split()) > 3:
        print(msg)