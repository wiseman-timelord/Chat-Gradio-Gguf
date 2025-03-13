# Script: `.\scripts\temporary.py`

# Imports...
import time
from scripts.prompts import prompt_templates 

# General Constants/Variables/Lists/Maps/Arrays
MODEL_FOLDER = "path/to/your/models"
VECTORSTORE_DIR = "data/vectors"
TEMP_DIR = "data/temp"
HISTORY_DIR = "data/history"
SESSION_FILE_FORMAT = "%Y%m%d_%H%M%S"
session_label = ""
current_session_id = None
RAG_CHUNK_SIZE_DEVIDER = 4
RAG_CHUNK_OVERLAP_DEVIDER = 32
MODELS_LOADED = False
SESSION_ACTIVE = False
MAX_HISTORY_SLOTS = 10
yake_history_detail = [None] * MAX_HISTORY_SLOTS
MAX_ATTACH_SLOTS = 8
last_save_time = 0 
AFTERTHOUGHT_TIME = True
MODEL_NAME = "Select_a_model..."
CONTEXT_SIZE = 8192
VRAM_SIZE = 8192
GPU_LAYERS = 0
BATCH_SIZE = 1024
SELECTED_GPU = None
SELECTED_CPU = None
DYNAMIC_GPU_LAYERS = True
MMAP = True
MLOCK = True  # Always True, option removed
STREAM_OUTPUT = True
USE_PYTHON_BINDINGS = True  # Always True, option removed
LLAMA_CLI_PATH = "data/llama-vulkan-bin/llama-cli.exe"
BACKEND_TYPE = "Not Configured"
LLAMA_BIN_PATH = ""
RAG_AUTO_LOAD = ["general_knowledge"]
REPEAT_PENALTY = 1.0
TEMPERATURE = 0.66
SESSION_LOG_HEIGHT = 650
INPUT_LINES = 27
DATA_DIR = None  # Will be set by launcher.py

# Global LLM instance
llm = None

# UI Constants
USER_COLOR = "#ffffff"
THINK_COLOR = "#c8a2c8"
RESPONSE_COLOR = "#add8e6"
SEPARATOR = "=" * 40
MID_SEPARATOR = "-" * 30

# Model Constants
DEFAULT_CONTEXT_SIZE = 4096
DEFAULT_GPU_LAYERS = 35

# Options for Dropdowns
ALLOWED_EXTENSIONS = {"bat", "py", "ps1", "txt", "json", "yaml", "psd1", "xaml"}
CTX_OPTIONS = [8192, 16384, 24576, 32768, 49152, 65536, 98304, 131072]
VRAM_OPTIONS = [2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768, 49152, 65536]
REPEAT_OPTIONS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
BATCH_OPTIONS = [128, 256, 512, 1024, 2048, 4096]
TEMP_OPTIONS = [0.0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 1.0] 
MAX_POSSIBLE_HISTORY_SLOTS = 16
HISTORY_SLOT_OPTIONS = [4, 8, 10, 12, 16]
MAX_POSSIBLE_ATTACH_SLOTS = 10
ATTACH_SLOT_OPTIONS = [2, 4, 6, 8, 10]
SESSION_LOG_HEIGHT_OPTIONS = [400, 550, 650, 700, 750, 850, 1000, 1200]
INPUT_LINES_OPTIONS = [15, 21, 25, 27, 29, 33, 39, 47]

# RPG Settings
RP_LOCATION = "Public"
USER_PC_NAME = "Human"
USER_PC_ROLE = "Lead Roleplayer"
AI_NPC_NAME = "Robot"
AI_NPC_ROLE = "Randomers"

# TOT Settings
TOT_VARIATIONS = [
    "Please provide a detailed answer.",
    "Be concise.",
    "Think step by step."
]

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
    "code": ["code", "coder", "program", "dev", "copilot", "codex", "Python", "Powershell"],
    "rpg": ["nsfw", "adult", "mature", "explicit", "rp", "roleplay"],
    "chat": []
}

# Handling Keywords for Special Model Behaviors
handling_keywords = {
    "uncensored": ["uncensored", "unfiltered", "unbiased", "unlocked"],
    "reasoning": ["reason", "r1", "think"]
}

# Reasoning enhancement keywords
reasoning_keywords = ["reasoner", "r1", "reasoning", "reason"]

current_model_settings = {
    "category": "chat",
    "prompt_template": prompt_templates["chat"]
}