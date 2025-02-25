# Script: `.\scripts\temporary.py`

# Imports...
import time

# General Constants/Variables/Lists/Maps/Arrays
MODEL_DIR = "models"  # Updated per readme.md
VECTORSTORE_DIR = "data/vectorstores"
TEMP_DIR = "temp"
HISTORY_DIR = "data/history"
SESSION_FILE_FORMAT = "%Y%m%d_%H%M%S"
MODEL_LOADED = False
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
    "code": 0.6,
    "nsfw": 0.8,
    "reasoning": 0.5,
    "uncensored": 0.7,
    "general_chat": 0.7
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