# Script: `.\scripts\temporary.py`

# Imports...
import time

# General Constants/Variables/Lists/Maps/Arrays
MODEL_FOLDER = "models"  # Directory for GGUF models
VECTORSTORE_DIR = "data/vectorstores"  # Directory for vector stores
TEMP_DIR = "data/temp"  # Temporary directory aligned with installer
HISTORY_DIR = "data/history"  # Directory for session history
VULKAN_DLL_PATH = "C:\\Windows\\SysWOW64\\vulkan-1.dll"  # Reference path for Vulkan DLL
SESSION_FILE_FORMAT = "%Y%m%d_%H%M%S"  # Format for session file names
MODEL_LOADED = False  # Tracks if model is loaded
rag_documents = []  # Placeholder for RAG documents
session_label = ""  # Current session label
current_session_id = None  # Unique ID for the current session
RAG_CHUNK_SIZE_DEVIDER = 4  # RAG chunk size = n_ctx / RAG_CHUNK_SIZE_DEVIDER
RAG_CHUNK_OVERLAP_DEVIDER = 32  # RAG chunk overlap = n_ctx / RAG_CHUNK_OVERLAP_DEVIDER

# Configurable Settings (Loaded from JSON)
MODEL_PATH = "models/Select_a_model..."  # Default model path
N_CTX = 8192  # Default context window size
TEMPERATURE = 0.75  # Default temperature
VRAM_SIZE = 8192  # Default VRAM size in MB (8GB)
SELECTED_GPU = None  # Selected GPU device
DYNAMIC_GPU_LAYERS = True # Kompute/Vulkan = True, Avx2 = False
MMAP = True  # Use memory mapping
MLOCK = False  # Use memory locking
USE_PYTHON_BINDINGS = True  # Use Python bindings by default
LLAMA_CLI_PATH = ""  # Path to llama-cli.exe, set by config
BACKEND_TYPE = ""  # Backend type (e.g., "GPU/CPU - Vulkan"), set by config
LLAMA_BIN_PATH = ""  # Directory of llama.cpp binaries, set by config
N_GPU_LAYERS = 0  # Number of layers to offload to GPU, calculated at runtime
RAG_MAX_DOCS = 6  # Max RAG documents
MAX_SESSIONS = 10  # Max number of saved sessions, configurable via HISTORY_OPTIONS
RAG_AUTO_LOAD = ["general_knowledge"]  # Default RAG vectorstores to load
REPEAT_PENALTY = 1.0
N_BATCH = 1024  # Default batch size

# UI Constants
USER_COLOR = "#ffffff"  # Color for user messages
THINK_COLOR = "#c8a2c8"  # Color for thinking indicators
RESPONSE_COLOR = "#add8e6"  # Color for assistant responses
SEPARATOR = "=" * 40  # UI separator line
MID_SEPARATOR = "-" * 30  # UI mid-separator line

# Model Constants
DEFAULT_TEMPERATURE = 0.75  # Fallback temperature
DEFAULT_N_CTX = 4096  # Fallback context size
DEFAULT_N_GPU_LAYERS = 35  # Fallback GPU layers

# Options for Dropdowns
ALLOWED_EXTENSIONS = {"bat", "py", "ps1", "txt", "json", "yaml", "psd1", "xaml"}  # Supported file extensions
CTX_OPTIONS = [8192, 16384, 24576, 32768]  # Context window options
TEMP_OPTIONS = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]  # Temperature options per readme
VRAM_OPTIONS = [1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768]  # VRAM options in MB
REPEAT_OPTIONS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  # Repeat penalty options
BATCH_OPTIONS = [128, 256, 512, 1024, 2048, 4096] # n_batch (output length)

# Program options
HISTORY_OPTIONS = [5, 10, 20, 40, 80]  # Options for max sessions in UI
MAX_DOCS_OPTIONS = [2, 4, 6, 8]  # Added

# Global LLM instance
llm = None  # Placeholder for Llama instance

# Status text entries
STATUS_TEXTS = {
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

# Model Categories and Keywords
category_keywords = {
    "code": ["code", "cq"],
    "nsfw": ["nsfw", "rp"],
    "reasoning": ["reasoner", "deepseek-r1"],
    "uncensored": ["uncensored"],
    "general_chat": []
}

# System Prompts per Category
model_prompts = {
    "code": "You're an AI coding assistant who helps in solving coding questions. Please provide accurate and efficient code solutions.",
    "nsfw": "You're an AI designed for roleplay and NSFW content. Please engage in explicit and adult-themed conversations.",
    "reasoning": "You're an AI assistant that thinks step-by-step before answering. Please provide detailed and reasoned responses.",
    "uncensored": "You're an AI assistant that can handle any topic without censorship.",
    "general_chat": "You're a helpful assistant. Please answer questions and provide information in a friendly and informative manner."
}

# Temperature Defaults per Category
temperature_defaults = {
    "code": 0.5,
    "nsfw": 0.75,
    "reasoning": 0.5,
    "uncensored": 0.75,
    "general_chat": 0.75
}

# Prompt Templates per Category
prompt_templates = {
    "code": "{system_prompt}\n\nUSER: {prompt}\nASSISTANT:",
    "nsfw": "{system_prompt}\n\nUSER: {prompt}\nASSISTANT:",
    "reasoning": "{system_prompt}\n\nUSER: {prompt}\nASSISTANT: Let's think step by step.",
    "uncensored": "{system_prompt}\n\nUSER: {prompt}\nASSISTANT:",
    "general_chat": "{system_prompt}\n\nUSER: {prompt}\nASSISTANT:"
}

# Current Model Settings
current_model_settings = {
    "category": "general_chat",
    "system_prompt": model_prompts["general_chat"],
    "temperature": temperature_defaults["general_chat"],
    "prompt_template": prompt_templates["general_chat"]
}