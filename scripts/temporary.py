# Script: `.\scripts\temporary.py`

# General Constants/Variables/Lists/Maps/Arrays
MODEL_DIR = "models"  # Updated per readme.md
VECTORSTORE_DIR = "data/vectorstores"
TEMP_DIR = "temp"
HISTORY_DIR = "data/history"
SESSION_FILE_FORMAT = "%Y%m%d_%H%M%S"
MODEL_LOADED = False
ACTIVE_SESSION = False
message_history = []
rag_documents = []
session_label = ""

# Configurable Settings
MODEL_PATH = "models/Lamarckvergence-14B-GGUF"  # Updated per readme.md
N_CTX = 8192
TEMPERATURE = 0.7
DYNAMIC_GPU_LAYERS = True
VRAM_SIZE = 8192  # Default 8GB in MB per readme.md
SELECTED_GPU = None
MMAP = True
MLOCK = False
BACKEND_TYPE = "vulkan"
LLAMA_CLI_PATH = ""
RAG_AUTO_LOAD = ["general_knowledge"]
RAG_CHUNK_SIZE = 2048
RAG_CHUNK_OVERLAP = 256
RAG_MAX_DOCS = 5
MAX_SESSIONS = 10

# UI Constants
USER_COLOR = "#ffffff"
THINK_COLOR = "#c8a2c8"
RESPONSE_COLOR = "#add8e6"
SEPARATOR = "=" * 40
MID_SEPARATOR = "-" * 30

# Model Constants
DEFAULT_TEMPERATURE = 0.7
DEFAULT_N_CTX = 4096
DEFAULT_N_GPU_LAYERS = 35

# Session Settings
MAX_SESSIONS = 10  # Default max num sessions history
HISTORY_OPTIONS = [5, 10, 20, 40, 80]  # Max num sessions options 

# Options for Dropdowns
ALLOWED_EXTENSIONS = {"bat", "py", "ps1", "txt", "json", "yaml", "psd1", "xaml"}
CTX_OPTIONS = [8192, 16384, 24576, 32768]
TEMP_OPTIONS = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]  # Per readme.md
VRAM_OPTIONS = [1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768]


# Global LLM instance
llm = None