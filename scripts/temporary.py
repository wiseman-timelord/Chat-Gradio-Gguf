# scripts/temporary.py

# Imports...
import time
from scripts.prompts import prompt_templates 

# Configuration variables with defaults
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
SELECTED_CPU = None
STREAM_OUTPUT = True
USE_PYTHON_BINDINGS = True
LLAMA_CLI_PATH = "data/llama-vulkan-bin/llama-cli.exe"
BACKEND_TYPE = "Not Configured"
DATA_DIR = None  # Will be set by launcher.py
llm = None
SPEECH_ENABLED = False
AUTO_SUMMARY = False
CURRENT_SUMMARY = ""

# Arrays
session_attached_files = []

# UI Constants
USER_COLOR = "#ffffff"
THINK_COLOR = "#c8a2c8"
RESPONSE_COLOR = "#add8e6"
SEPARATOR = "=" * 40
MID_SEPARATOR = "-" * 30

# Options for Dropdowns
ALLOWED_EXTENSIONS = {"bat", "py", "ps1", "txt", "json", "yaml", "psd1", "xaml"}
MAX_POSSIBLE_HISTORY_SLOTS = 16
MAX_POSSIBLE_ATTACH_SLOTS = 10

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

CHAT_FORMAT_MAP = {
    'qwen2': 'chatml',   # Qwen2 uses a ChatML-like format
    'llama': 'llama-2',  # Llama models use llama-2 format
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
    "category": "chat"  # Simplified; prompt_template removed as it’s not needed
}