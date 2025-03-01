# Script: `.\scripts\temporary.py`

# Imports...
import time

# General Constants/Variables/Lists/Maps/Arrays
MODEL_FOLDER = "models"
VECTORSTORE_DIR = "data/vectorstores"
TEMP_DIR = "data/temp"
HISTORY_DIR = "data/history"
VULKAN_DLL_PATH = "C:\\Windows\\SysWOW64\\vulkan-1.dll"
SESSION_FILE_FORMAT = "%Y%m%d_%H%M%S"
MODEL_LOADED = False
rag_documents = []
session_label = ""
current_session_id = None
RAG_CHUNK_SIZE_DEVIDER = 4
RAG_CHUNK_OVERLAP_DEVIDER = 32

# Configurable Settings (Loaded from JSON)
QUALITY_MODEL_NAME = "Select_a_model..."
FAST_MODEL_NAME = "Select_a_model..."
N_CTX = 8192
VRAM_SIZE = 8192
SELECTED_GPU = None
DYNAMIC_GPU_LAYERS = True
MMAP = True
MLOCK = False
USE_PYTHON_BINDINGS = True
LLAMA_CLI_PATH = ""
BACKEND_TYPE = ""
LLAMA_BIN_PATH = ""
N_GPU_LAYERS = 0
RAG_AUTO_LOAD = ["general_knowledge"]
REPEAT_PENALTY = 1.0
N_BATCH = 1024

# TOT Settings
TOT_VARIATIONS = [
    "Please provide a detailed answer.",
    "Be concise.",
    "Think step by step."
]

# UI Constants
USER_COLOR = "#ffffff"
THINK_COLOR = "#c8a2c8"
RESPONSE_COLOR = "#add8e6"
SEPARATOR = "=" * 40
MID_SEPARATOR = "-" * 30

# Model Constants
DEFAULT_N_CTX = 4096
DEFAULT_N_GPU_LAYERS = 35

# Options for Dropdowns
ALLOWED_EXTENSIONS = {"bat", "py", "ps1", "txt", "json", "yaml", "psd1", "xaml"}
CTX_OPTIONS = [8192, 16384, 24576, 32768]
VRAM_OPTIONS = [1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768]
REPEAT_OPTIONS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
BATCH_OPTIONS = [128, 256, 512, 1024, 2048, 4096]

# Global LLM instances
quality_llm = None
fast_llm = None

# RPG Settings
RP_LOCATION = "Public"
USER_NAME = "Human"
USER_ROLE = "Lead Roleplayer"
AI_NPC1 = "Randomer"
AI_NPC2 = "Unused"
AI_NPC3 = "Unused"

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
    "uncensored": ["uncensored", "unfiltered", "unbiased", "unlocked"],
    "general": []  # Fallback category for models without specific keywords
}

# Reasoning enhancement keywords
reasoning_keywords = ["reasoner", "r1", "reasoning", "reason"]

# System Prompts per Category
model_prompts = {
    "code": "You are a coding assistant...",
    "rpg": "You are an NSFW conversational AI...",
    "uncensored": "You are an uncensored AI...",
    "general": "You are a general-purpose AI..."
}

# Temperature defaults per category
temperature_defaults = {
    "code": 0.25,
    "rpg": 0.75,
    "uncensored": 0.75,
    "general": 0.5  # Middle value for general models
}

# Prompt Templates per Category
prompt_templates = {
    "code": "You are a coding assistant. Provide code solutions and explanations.\nUser: {user_input}\nAI: ",
    "rpg": "You are a role-playing AI. Respond in, character and context to the scenario, to the user.\nUser: {user_input}\nAI: ",
    "uncensored": "You are an uncensored AI. Provide unfiltered responses.\nUser: {user_input}\nAI: ",
    "general": "You are a general-purpose AI assistant.\nUser: {user_input}\nAI: ",
    "rp": "You are roleplaying in {rp_location}. Characters: {user_name} ({user_role}), {ai_npc1}, {ai_npc2}, {ai_npc3}.\nUser: {user_input}\nAI: "
}

# Current Model Settings
current_model_settings = {
    "category": "general",  # Changed from "general_chat"
    "system_prompt": model_prompts["general"],
    "temperature": temperature_defaults["general"],
    "prompt_template": prompt_templates["general"]
}