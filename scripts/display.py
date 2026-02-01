# Script: `.\scripts\display.py`
# Compatible with Gradio 3.50.2 for Qt5 WebEngine (Windows 7-8.1) support


# Imports - display.py


# Standard library
import asyncio
import json
import os
import random
import re
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from queue import Queue

# Third-party
import gradio as gr
import pyperclip
import spacy
import tkinter as tk
from tkinter import filedialog
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

# Project imports - consolidated configuration
import scripts.configuration as cfg
from scripts.configuration import save_config

# Commonly used constants (kept explicit for clarity & auto-complete)
from scripts.configuration import (
    MODEL_NAME, MODEL_FOLDER, SESSION_ACTIVE,
    MAX_HISTORY_SLOTS, MAX_ATTACH_SLOTS, SESSION_LOG_HEIGHT,
    CONTEXT_SIZE, BATCH_SIZE, TEMPERATURE, REPEAT_PENALTY,
    VRAM_SIZE, SELECTED_GPU, SELECTED_CPU, MLOCK, BACKEND_TYPE,
    ALLOWED_EXTENSIONS, VRAM_OPTIONS, CTX_OPTIONS, BATCH_OPTIONS,
    TEMP_OPTIONS, REPEAT_OPTIONS, HISTORY_SLOT_OPTIONS,
    SESSION_LOG_HEIGHT_OPTIONS, ATTACH_SLOT_OPTIONS, HISTORY_DIR,
    USER_COLOR, THINK_COLOR, RESPONSE_COLOR, PRINT_RAW_OUTPUT,
    SHOW_THINK_PHASE, BLEEP_ON_EVENTS, TTS_ENABLED, TTS_VOICE_NAME,
    MAX_TTS_LENGTH, context_injector, STATUS_MESSAGES
)

# Utility & helper modules
from scripts import utility
from scripts.utility import (
    hybrid_search, is_research_available, get_research_capabilities, short_path,
    get_saved_sessions, get_cpu_info, load_session_history, save_session_history,
    get_available_gpus, filter_operational_content, process_files, eject_file, 
    summarize_session, beep, update_file_slot_ui 
)

# Model handling
from scripts.inference import (
    get_response_stream, get_available_models, unload_inference, get_model_settings,
    inspect_model, load_inference, change_model, load_models
)

# Tools (search, TTS, etc.)
from scripts.tools import (
    format_search_status_for_chat, web_search, format_web_search_status_for_chat,
    get_voice_choices, get_output_device_choices, get_sample_rate_choices,
    speak_last_response, stop_speaking, get_tts_status, initialize_tts,
    get_voice_id_by_name, speak_text,
    synthesize_last_response, play_tts_audio
)


# GRADIO 3.x COMPATIBILITY LAYER

# Gradio 3.x Chatbot uses list of tuples: [(user_msg, bot_msg), ...]
# Gradio 4.x Chatbot uses list of dicts: [{"role": "user", "content": msg}, ...]
# 
# We maintain internal state as message dicts for compatibility with inference.py
# but convert to/from tuple format for the Chatbot component.


def messages_to_tuples(messages):
    """Convert internal message dicts → Gradio tuple format with labels."""
    if not messages:
        return []
    
    tuples = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        # Clean content to remove any existing prefixes before adding our own
        content = re.sub(r'^AI-Chat:\s*\n?', '', msg.get('content', '').strip(), flags=re.MULTILINE)
        content = re.sub(r'^User:\s*\n?', '', content, flags=re.MULTILINE)
        content = content.strip()
        
        if msg.get('role') == 'user':
            display_content = f"User:\n{content}" if content else "User:"
            
            if i + 1 < len(messages) and messages[i + 1].get('role') == 'assistant':
                bot_msg = messages[i + 1].get('content', '').strip()
                # Clean AI-Chat: prefix from bot message too
                bot_msg = re.sub(r'^AI-Chat:\s*\n?', '', bot_msg, flags=re.MULTILINE)
                bot_msg = bot_msg.strip()
                bot_display = f"AI-Chat:\n{bot_msg}" if bot_msg else "AI-Chat:"
                tuples.append((display_content, bot_display))
                i += 1  # skip next (assistant)
            else:
                tuples.append((display_content, None))
                
        elif msg.get('role') == 'assistant' and (i == 0 or messages[i-1].get('role') != 'user'):
            # Orphan assistant message - clean and format
            content = re.sub(r'^AI-Chat:\s*\n?', '', content, flags=re.MULTILINE).strip()
            display_content = f"AI-Chat:\n{content}" if content else "AI-Chat:"
            tuples.append((None, display_content))
            
        i += 1
    
    return tuples


def tuples_to_messages(tuples):
    """Convert Gradio tuple format back to message dict list."""
    if not tuples:
        return []
    
    messages = []
    for user_msg, bot_msg in tuples:
        if user_msg is not None:
            # Strip the "User:\n" prefix if present
            clean_user = re.sub(r'^User:\s*\n?', '', user_msg, flags=re.MULTILINE).strip()
            messages.append({'role': 'user', 'content': clean_user})
        if bot_msg is not None:
            # Strip the "AI-Chat:\n" prefix if present
            clean_bot = re.sub(r'^AI-Chat:\s*\n?', '', bot_msg, flags=re.MULTILINE).strip()
            messages.append({'role': 'assistant', 'content': clean_bot})
    
    return messages


def get_chatbot_output(session_tuples, session_messages):
    """
    Return correct format for Chatbot based on Gradio version.
    Ensures labels are visible in all versions.
    """
    if cfg.GRADIO_VERSION.startswith('3.'):
        # Gradio 3.x: Use tuple format directly
        return session_tuples
    else:
        # Gradio 4.x/5.x: Convert tuples (which have labels) to messages format
        # This preserves the "User:" and "AI-Chat:" labels in the content
        messages_with_labels = []
        for user_tuple, bot_tuple in session_tuples:
            if user_tuple:
                # Keep the "User:\n" prefix in the content for display
                clean_user = re.sub(r'^User:\s*\n?', '', user_tuple, flags=re.MULTILINE).strip()
                display_user = f"User:\n{clean_user}" if clean_user else "User:"
                messages_with_labels.append({'role': 'user', 'content': display_user})
            if bot_tuple:
                # Keep the "AI-Chat:\n" prefix in the content for display
                clean_bot = re.sub(r'^AI-Chat:\s*\n?', '', bot_tuple, flags=re.MULTILINE).strip()
                display_bot = f"AI-Chat:\n{clean_bot}" if clean_bot else "AI-Chat:"
                messages_with_labels.append({'role': 'assistant', 'content': display_bot})
        return messages_with_labels

# Functions...
def update_cpu_select():
    from scripts.utility import get_cpu_info
    cpu_info = get_cpu_info()
    choices = ["Auto-Select"] + [c["label"] for c in cpu_info]
    value = cfg.SELECTED_CPU
    if value not in choices:
        value = "Auto-Select"
    return gr.update(choices=choices, value=value)

def get_panel_choices(model_settings):
    """Determine available panel choices based on model settings - safe version"""
    choices = ["History", "Attachments"]
    
    # If model_settings is not a dict (e.g. string, None, or missing), show both panels
    if not isinstance(model_settings, dict):
        print("[DEBUG] model_settings is not a dict → showing all panels")
        return choices
    
    # Only hide Attachments for known NSFW/roleplay models
    if model_settings.get("is_nsfw", False) or model_settings.get("is_roleplay", False):
        if "Attachments" in choices:
            choices.remove("Attachments")
    
    return choices

def ensure_model_loaded():
    """Lazy load model if not already loaded. Returns (success, status_message)"""
    if cfg.MODELS_LOADED and cfg.llm is not None:
        return True, "Model ready"

    model_name = cfg.MODEL_NAME
    if model_name in ["Select_a_model...", "No models found", "", None]:
        return False, "No valid model selected"

    cfg.set_status("Loading model on first use...", priority=True, console=True)
    
    try:
        from scripts.inference import load_models
        status, loaded, llm_new, _ = load_models(
            cfg.MODEL_FOLDER,
            model_name,
            cfg.VRAM_SIZE,
            cfg.llm,
            cfg.MODELS_LOADED
        )
        cfg.llm = llm_new
        cfg.MODELS_LOADED = loaded
        cfg.LAST_INTERACTION_TIME = time.time()   # reset inactivity timer
        
        if loaded:
            cfg.set_status("Model loaded", console=True)
            from scripts.utility import beep
            beep()
            return True, "Model loaded successfully"
        else:
            return False, f"Load failed: {status}"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Load error: {str(e)}"

def update_panel_choices(model_settings, current_panel):
    choices = get_panel_choices(model_settings)
    if current_panel not in choices:
        current_panel = choices[0] if choices else "History"
    return gr.update(choices=choices, value=current_panel), current_panel

def update_panel_on_mode_change(current_panel):
    """Update panel visibility based on the selected panel."""
    choices = ["History", "Attachments"]
    new_panel = current_panel if current_panel in choices else choices[0]
    attach_visible = new_panel == "Attachments"
    history_visible = new_panel == "History"
    return (
        gr.update(choices=choices, value=new_panel),
        gr.update(visible=attach_visible),
        gr.update(visible=history_visible),
        new_panel
    )

def process_attach_files(files, attached_files):
    """Process uploaded files for attachment."""
    if not files:
        return "No files selected.", attached_files
    return process_files(files, attached_files, cfg.MAX_ATTACH_SLOTS, is_attach=True)

def process_vector_files(files, vector_files, models_loaded):
    if not models_loaded:
        return "Error: Load model first.", vector_files
    return process_files(files, vector_files, cfg.MAX_ATTACH_SLOTS, is_attach=False)

def update_config_settings(ctx, batch, temp, repeat, vram, gpu, cpu, model, print_raw):
    cfg.CONTEXT_SIZE = int(ctx)
    cfg.BATCH_SIZE = int(batch)
    cfg.TEMPERATURE = float(temp)
    cfg.REPEAT_PENALTY = float(repeat)
    cfg.VRAM_SIZE = int(vram)
    cfg.SELECTED_GPU = gpu
    cfg.SELECTED_CPU = cpu
    cfg.MODEL_NAME = model
    cfg.PRINT_RAW_OUTPUT = bool(print_raw)
    status_message = (
        f"Updated settings: Context Size={ctx}, Batch Size={batch}, "
        f"Temperature={temp}, Repeat Penalty={repeat}, VRAM Size={vram}, "
        f"Selected GPU={gpu}, CPU={cpu}, Model={model}"
    )
    return status_message

def get_model_loaded_display(is_loaded):
    """Return a gr.update for the model loaded indicator textbox."""
    if is_loaded:
        return gr.update(value="🟢 SO LOADED")
    else:
        return gr.update(value="🔴 NOT LOADED")

def get_ini_display_text():
    """Build display string of INI constants NOT shown on the Hardware/Models Config tab."""
    lines = []
    lines.append(f"Platform:  {getattr(cfg, 'PLATFORM', 'N/A')}")
    lines.append(f"OS Version:  {getattr(cfg, 'OS_VERSION', 'N/A')}")
    lines.append(f"Gradio Version:  {getattr(cfg, 'GRADIO_VERSION', 'N/A')}")
    lines.append(f"Embedding Model:  {getattr(cfg, 'EMBEDDING_MODEL_NAME', 'N/A')}")
    lines.append(f"Embedding Backend:  {getattr(cfg, 'EMBEDDING_BACKEND', 'N/A')}")
    lines.append(f"Llama Bin Path:  {getattr(cfg, 'LLAMA_BIN_PATH', 'N/A')}")
    lines.append(f"TTS Engine:  {getattr(cfg, 'TTS_ENGINE', 'N/A')}")
    if getattr(cfg, 'TTS_TYPE', '') == "coqui":
        lines.append(f"Coqui Voice ID:  {getattr(cfg, 'COQUI_VOICE_ID', 'N/A')}")
        lines.append(f"Coqui Voice Accent:  {getattr(cfg, 'COQUI_VOICE_ACCENT', 'N/A')}")
        lines.append(f"Coqui Model:  {getattr(cfg, 'COQUI_MODEL', 'N/A')}")
    return "\n".join(lines)

def get_debug_globals_text():
    """Build display string of critical runtime globals NOT visible on Config/Settings tabs."""
    lines = []
    lines.append(f"LOADED_CONTEXT_SIZE:  {getattr(cfg, 'LOADED_CONTEXT_SIZE', 'N/A')}")
    lines.append(f"GPU_LAYERS:  {getattr(cfg, 'GPU_LAYERS', 0)}")
    lines.append(f"SESSION_ACTIVE:  {cfg.SESSION_ACTIVE}")
    embedding_status = "Loaded" if (cfg.context_injector and cfg.context_injector.embedding is not None) else "Not loaded"
    lines.append(f"EMBEDDING_MODEL:  {embedding_status}")
    lines.append(f"USE_PYTHON_BINDINGS:  {getattr(cfg, 'USE_PYTHON_BINDINGS', 'N/A')}")
    lines.append(f"MMAP:  {getattr(cfg, 'MMAP', 'N/A')}")
    lines.append(f"MLOCK:  {getattr(cfg, 'MLOCK', 'N/A')}")
    lines.append(f"CPU_PHYSICAL_CORES:  {getattr(cfg, 'CPU_PHYSICAL_CORES', 'N/A')}")
    lines.append(f"CPU_LOGICAL_CORES:  {getattr(cfg, 'CPU_LOGICAL_CORES', 'N/A')}")
    lines.append(f"INACTIVITY_TIMEOUT:  {getattr(cfg, 'INACTIVITY_TIMEOUT', 'N/A')}s")
    lines.append(f"FILTER_MODE:  {getattr(cfg, 'FILTER_MODE', 'N/A')}")
    return "\n".join(lines)

def save_all_settings():
    """Save all configuration settings and return a status message."""
    cfg.save_config()
    return cfg.STATUS_MESSAGES["config_saved"]

def set_session_log_base_height(new_height):
    """Set the base session log height from the Configuration page dropdown."""
    cfg.SESSION_LOG_HEIGHT = int(new_height)
    return gr.update(height=cfg.SESSION_LOG_HEIGHT)

def estimate_lines(text, chars_per_line=80):
    """Estimate the number of lines in the textbox based on content."""
    if not text:
        return 0
    segments = text.split('\n')
    total_lines = 0
    for segment in segments:
        total_lines += max(1, (len(segment) + chars_per_line - 1) // chars_per_line)
    return total_lines

def update_session_log_height(text):
    """Adjust the Session Log height based on the number of lines in User Input."""
    lines = estimate_lines(text)
    initial_lines = 3
    max_lines = cfg.USER_INPUT_MAX_LINES
    if lines <= initial_lines:
        adjustment = 0
    else:
        effective_extra_lines = min(lines - initial_lines, max_lines - initial_lines)
        adjustment = effective_extra_lines * 20
    new_height = max(100, cfg.SESSION_LOG_HEIGHT - adjustment)
    return gr.update(height=new_height)

def format_response(output: str) -> str:
    """Format response with thinking phase detection and code highlighting."""
    import re
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import HtmlFormatter
    from scripts.configuration import THINK_COLOR, GRADIO_VERSION
    
    formatted = []
    
    # 1. Extract thinking blocks for separate display
    think_patterns = [
        (r'<think>(.*?)</think>', '[Thinking] '),
        (r'<\|channel\|>analysis(.*?)<\|end\|>.*?<\|channel\|>final', '[Thinking] ')
    ]
    
    for pattern, prefix in think_patterns:
        thinks = re.findall(pattern, output, re.DOTALL)
        for thought in thinks:
            if thought.strip():
                clean_thought = re.sub(r'<\|[^>]+\|>', '', thought)
                formatted.append(f'<span style="color: {THINK_COLOR}">{prefix}{clean_thought.strip()}</span>')
    
    # 2. Remove thinking content from main output
    clean_output = output
    clean_output = re.sub(r'<think>.*?</think>', '', clean_output, flags=re.DOTALL)
    clean_output = re.sub(
        r'<\|channel\|>analysis.*?(?:<\|end\|>.*?<\|channel\|>final|<\|end\|><\|start\|>assistant<\|end\|><\|start\|>assistant<\|message\|>)',
        '', clean_output, flags=re.DOTALL
    )
    clean_output = re.sub(r'<\|[^>]+\|>', '', clean_output)
    
    # 3. Remove annoying "Thinking...." filler lines
    lines = clean_output.split('\n')
    filtered_lines = [
        line for line in lines
        if not (line.strip().startswith("Thinking") and all(c in '.… ' for c in line.strip()[8:]))
    ]
    clean_output = '\n'.join(filtered_lines)
    
    # 4. Highlight code blocks
    code_blocks = re.findall(r'```(\w+)?\n(.*?)```', clean_output, re.DOTALL)
    for lang, code in code_blocks:
        if lang:
            try:
                lexer = get_lexer_by_name(lang, stripall=True)
                formatted_code = highlight(code, lexer, HtmlFormatter())
                clean_output = clean_output.replace(f'```{lang}\n{code}```', formatted_code)
            except:
                pass  # silently skip invalid languages
    
    # 5. Common basic normalization
    clean_output = re.sub(r' {2,}', ' ', clean_output)  # reduce multiple spaces to 1
    
    # 6. Apply the configurable output filter (replaces hardcoded Gradio version checks)
    clean_output = apply_output_filter(clean_output)
        
    clean_output = clean_output.strip()
    
    # 6. Combine thinking + final output
    if formatted:
        return '\n'.join(formatted) + '\n\n' + clean_output
    
    return clean_output

# OUTPUT FILTERING FUNCTIONS

def get_filter_text_for_display():
    """Get the current filter as displayable/editable text."""
    if cfg.FILTER_MODE == "custom":
        # Load from custom file
        custom_path = Path(cfg.CUSTOM_FILTER_PATH)
        if custom_path.exists():
            try:
                return custom_path.read_text(encoding='utf-8')
            except Exception as e:
                print(f"[FILTER] Error reading custom filter: {e}")
        # Fallback to active filter
        return filter_list_to_text(cfg.ACTIVE_FILTER)
    else:
        # Return the preset filter
        preset = cfg.FILTER_PRESETS.get(cfg.FILTER_MODE, [])
        return filter_list_to_text(preset)

def filter_list_to_text(filter_list):
    """Convert filter list to editable text format."""
    lines = []
    for find, replace in filter_list:
        # Escape special characters for display
        find_escaped = repr(find)[1:-1]  # Remove outer quotes from repr
        replace_escaped = repr(replace)[1:-1]
        lines.append(f"{find_escaped} -> {replace_escaped}")
    return "\n".join(lines)

def text_to_filter_list(text):
    """Parse editable text format back to filter list."""
    filter_list = []
    for line in text.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ' -> ' in line:
            parts = line.split(' -> ', 1)
            if len(parts) == 2:
                # Unescape the strings
                try:
                    find = parts[0].encode().decode('unicode_escape')
                    replace = parts[1].encode().decode('unicode_escape')
                    filter_list.append((find, replace))
                except Exception as e:
                    print(f"[FILTER] Error parsing line '{line}': {e}")
    return filter_list

def load_filter_preset(preset_name):
    """Load a filter preset and return the text for display."""
    if preset_name == "User":
        # Load custom filter
        cfg.FILTER_MODE = "custom"
        custom_path = Path(cfg.CUSTOM_FILTER_PATH)
        if custom_path.exists():
            try:
                text = custom_path.read_text(encoding='utf-8')
                cfg.ACTIVE_FILTER = text_to_filter_list(text)
                return text, f"Loaded custom filter ({len(cfg.ACTIVE_FILTER)} rules)"
            except Exception as e:
                return "", f"Error loading custom filter: {e}"
        else:
            return "", "No custom filter saved yet. Edit and click Save."
    
    elif preset_name == "Light":
        cfg.FILTER_MODE = "gradio5"
        preset = cfg.FILTER_PRESETS.get("gradio5", [])
        cfg.ACTIVE_FILTER = preset.copy()
        text = filter_list_to_text(preset)
        return text, f"Loaded Light filter (Gradio 5 style, {len(preset)} rules)"
    
    elif preset_name == "Full":
        cfg.FILTER_MODE = "gradio3"
        preset = cfg.FILTER_PRESETS.get("gradio3", [])
        cfg.ACTIVE_FILTER = preset.copy()
        text = filter_list_to_text(preset)
        return text, f"Loaded Full filter (Gradio 3 style, {len(preset)} rules)"
    
    return "", "Unknown preset"

def save_custom_filter(filter_text):
    """Save the current filter text as a custom filter."""
    try:
        # Parse and validate
        filter_list = text_to_filter_list(filter_text)
        
        # Save to file
        custom_path = Path(cfg.CUSTOM_FILTER_PATH)
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        custom_path.write_text(filter_text, encoding='utf-8')
        
        # Update runtime state
        cfg.FILTER_MODE = "custom"
        cfg.ACTIVE_FILTER = filter_list
        
        # Save mode to persistent.json
        cfg.save_config()
        
        return f"Custom filter saved ({len(filter_list)} rules)"
    except Exception as e:
        return f"Error saving filter: {e}"

def initialize_filter_from_config():
    """Initialize the filter based on saved config or Gradio version."""
    filter_mode = getattr(cfg, 'FILTER_MODE', None)
    
    if filter_mode == "custom":
        # Load custom filter
        custom_path = Path(cfg.CUSTOM_FILTER_PATH)
        if custom_path.exists():
            try:
                text = custom_path.read_text(encoding='utf-8')
                cfg.ACTIVE_FILTER = text_to_filter_list(text)
                print(f"[FILTER] Loaded custom filter ({len(cfg.ACTIVE_FILTER)} rules)")
                return
            except Exception as e:
                print(f"[FILTER] Error loading custom filter: {e}")
    
    # Default based on Gradio version
    if cfg.GRADIO_VERSION and cfg.GRADIO_VERSION.startswith('3.'):
        cfg.FILTER_MODE = "gradio3"
        cfg.ACTIVE_FILTER = cfg.FILTER_PRESETS.get("gradio3", []).copy()
    else:
        cfg.FILTER_MODE = "gradio5"
        cfg.ACTIVE_FILTER = cfg.FILTER_PRESETS.get("gradio5", []).copy()
    
    print(f"[FILTER] Using {cfg.FILTER_MODE} filter ({len(cfg.ACTIVE_FILTER)} rules)")

def apply_output_filter(text):
    """Apply the active filter to output text."""
    if not cfg.ACTIVE_FILTER:
        return text
    
    for find, replace in cfg.ACTIVE_FILTER:
        text = text.replace(find, replace)
    
    return text

def get_initial_model_value():
    """Get initial model selection with proper fallback - safe against bad model metadata"""
    available_models = cfg.AVAILABLE_MODELS or get_available_models()
    base_choices = ["Select_a_model..."]
    
    if available_models and available_models != base_choices:
        available_models = [m for m in available_models if m not in base_choices]
        available_models = base_choices + available_models
    else:
        available_models = base_choices
    
    # Prefer saved model if valid
    if cfg.MODEL_NAME in available_models and cfg.MODEL_NAME not in base_choices:
        default_model = cfg.MODEL_NAME
    elif len(available_models) > 1:
        default_model = available_models[1]  # first real model if exists
    else:
        default_model = base_choices[0]
    
    # Safe check for is_reasoning
    is_reasoning = False
    if default_model not in base_choices:
        try:
            model_meta = get_model_settings(default_model)
            if isinstance(model_meta, dict):
                is_reasoning = model_meta.get("is_reasoning", False)
            else:
                print(f"[DEBUG] get_model_settings({default_model}) returned non-dict: {type(model_meta)}")
        except Exception as e:
            print(f"[DEBUG] Error getting model settings for {default_model}: {e}")
    
    return default_model, is_reasoning

def update_model_list(model_folder):
    """Update model dropdown choices and set correct initial value from loaded config."""
    
    # Update folder if changed (safe even if same)
    if model_folder and model_folder.strip() and model_folder != cfg.MODEL_FOLDER:
        cfg.MODEL_FOLDER = model_folder
        print(f"[MODEL] Folder updated to: {cfg.MODEL_FOLDER}")
    
    # Get current available models (uses cfg.MODEL_FOLDER)
    available = get_available_models()
    
    choices = ["Select_a_model..."]
    if available:
        choices += available
    else:
        choices = ["No models found"]
    
    # Prefer loaded MODEL_NAME if still valid, otherwise fallback
    selected_value = cfg.MODEL_NAME
    if selected_value not in choices:
        real_models = [m for m in available if m != "Select_a_model..."]
        selected_value = real_models[0] if real_models else "Select_a_model..."
        cfg.MODEL_NAME = selected_value  # keep globals in sync
        print(f"[MODEL] Loaded model '{cfg.MODEL_NAME}' no longer exists → reset to '{selected_value}'")
    
    print(f"[MODEL] Dropdown updated | choices={len(choices)-1} models | selected={selected_value}")
    
    return gr.update(
        choices=choices,
        value=selected_value,
        interactive=(selected_value != "Select_a_model..." and selected_value != "No models found")
    )

def handle_load_model(model_name, model_folder, vram_size, ctx_size, gpu, cpu, cpu_threads, llm_state, models_loaded_state):
    """Explicitly load the currently selected model with current cfg."""
    import scripts.configuration as cfg
    from scripts.inference import load_inference, get_model_settings
    from scripts.utility import beep
    if not model_name or model_name in ["Select_a_model...", "No models found", " ", None]:
        return (
            llm_state,
            models_loaded_state,
            "❌ Error: No valid model selected. Choose a model first.",
            "❌ Error: No valid model selected. Choose a model first.",
            "❌ Error: No valid model selected. Choose a model first.",
            gr.update(interactive=False),
            get_model_loaded_display(False)
        )

    # Update temporary globals with current UI values
    cfg.MODEL_NAME = model_name
    cfg.MODEL_FOLDER = model_folder
    cfg.VRAM_SIZE = int(vram_size)
    cfg.CONTEXT_SIZE = int(ctx_size)
    cfg.SELECTED_GPU = gpu
    cfg.SELECTED_CPU = cpu
    cfg.CPU_THREADS = int(cpu_threads) if cpu_threads else None

    try:
        # Load the model
        status, loaded, new_llm, _ = load_models(
            model_folder,
            model_name,
            int(vram_size),
            llm_state,
            models_loaded_state
        )
        
        if loaded:
            # Update global state
            cfg.MODELS_LOADED = True
            cfg.llm = new_llm
            cfg.LOADED_CONTEXT_SIZE = int(ctx_size)
            cfg.LAST_INTERACTION_TIME = time.time()
            
            # Success
            beep()
            status_msg = f"✅ Model loaded: {model_name} ({cfg.LOADED_CONTEXT_SIZE} ctx)"
            input_interactive = True
        else:
            status_msg = f"❌ Load failed: {status[:150]}"
            input_interactive = False
            new_llm = llm_state
            loaded = False
            
        return (
            new_llm,
            loaded,
            status_msg,  # interaction_global_status
            status_msg,  # config_status
            status_msg,  # filter_status
            gr.update(interactive=input_interactive),
            get_model_loaded_display(loaded)
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        status_msg = f"❌ Load error: {str(e)[:150]}"
        return (
            llm_state,
            models_loaded_state,
            status_msg,
            status_msg,
            status_msg,
            gr.update(interactive=False),
            get_model_loaded_display(False)
        )


def handle_unload_model(llm_state, models_loaded_state):
    """Explicitly unload the currently loaded model."""
    import scripts.configuration as cfg
    from scripts.inference import unload_models
    from scripts.utility import beep
    if not models_loaded_state or llm_state is None:
        return (
            llm_state,
            False,
            "ℹ️ No model currently loaded.",
            "ℹ️ No model currently loaded.",
            "ℹ️ No model currently loaded.",
            gr.update(interactive=False),
            get_model_loaded_display(False)
        )

    try:
        # Unload the model
        status, new_llm, new_models_loaded = unload_models(llm_state, models_loaded_state)
        
        # Update global state
        cfg.MODELS_LOADED = new_models_loaded
        cfg.llm = new_llm
        cfg.GPU_LAYERS = 0
        cfg.LOADED_CONTEXT_SIZE = None
        
        # Success
        beep()
        status_msg = "✅ Model unloaded successfully."
        input_interactive = False
        
        return (
            new_llm,
            new_models_loaded,
            status_msg,  # interaction_global_status
            status_msg,  # config_status
            status_msg,  # filter_status
            gr.update(interactive=input_interactive),
            get_model_loaded_display(False)
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        status_msg = f"❌ Unload error: {str(e)[:150]}"
        return (
            llm_state,
            models_loaded_state,
            status_msg,
            status_msg,
            status_msg,
            gr.update(interactive=True),  # Assume still loaded on error
            get_model_loaded_display(True)
        )

def handle_model_selection(model_name, model_folder_state):
    if not model_name:
        return model_folder_state, model_name, "No model selected."
    return model_folder_state, model_name, f"Selected model: {model_name}"

def browse_on_click(current_path):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        root.update_idletasks()
        folder_selected = filedialog.askdirectory(initialdir=current_path or os.path.expanduser("~"))
        root.attributes('-topmost', False)
        root.destroy()
        return folder_selected if folder_selected else current_path
    except ImportError:
        print("[BROWSE] tkinter not available - install python3-tk")
        return current_path
    except Exception as e:
        print(f"[BROWSE] Error: {e}")
        return current_path

def save_rp_settings(rp_location, user_name, user_role, ai_npc, ai_npc_role):
    from scripts import configuration
    cfg.RP_LOCATION = rp_location
    cfg.USER_PC_NAME = user_name
    cfg.USER_PC_ROLE = user_role
    cfg.AI_NPC_NAME = ai_npc
    cfg.AI_NPC_ROLE = ai_npc_role
    cfg.save_config()
    return (
        rp_location, user_name, user_role, ai_npc, ai_npc_role,
        rp_location, user_name, user_role, ai_npc, ai_npc_role
    )

def process_uploaded_files(files, loaded_files, models_loaded):
    from scripts.utility import create_session_vectorstore
    import scripts.configuration as cfg
    import os
    print("Uploaded files:", files)
    if not models_loaded:
        return "Error: Load a model first.", loaded_files
    
    max_files = cfg.MAX_ATTACH_SLOTS
    if len(loaded_files) >= max_files:
        return f"Max files ({max_files}) reached.", loaded_files
    
    new_files = [f for f in files if os.path.isfile(f) and f not in loaded_files]
    print("New files to add:", new_files)
    available_slots = max_files - len(loaded_files)
    for file in reversed(new_files[:available_slots]):
        loaded_files.insert(0, file)
    
    session_vectorstore = create_session_vectorstore(loaded_files)
    cfg.context_injector.set_session_vectorstore(session_vectorstore)
    
    print("Updated loaded_files:", loaded_files)
    return f"Processed {min(len(new_files), available_slots)} new files.", loaded_files

def start_new_session(session_messages, attached_files, llm_state, models_loaded_state):
    """
    Start a fresh session. Preserves model loaded state to prevent unnecessary reloads.
    """
    import scripts.configuration as cfg
    from scripts.utility import save_session_history

    # 1. Save current session if it was active and has messages
    if cfg.SESSION_ACTIVE and session_messages:
        try:
            save_session_history(session_messages, attached_files)
            print("[SESSION] Previous session auto-saved before starting new one")
        except Exception as e:
            print(f"[SESSION] Failed to save previous session: {e}")

    # 2. Reset session state (but NOT model state)
    cfg.SESSION_ACTIVE = False
    cfg.current_session_id = None
    cfg.session_label = ""
    cfg.session_attached_files = []

    # 3. Return values - use get_chatbot_output for empty lists
    # IMPORTANT: Return llm_state and models_loaded_state to preserve them
    chatbot_output = get_chatbot_output([], [])
    return (
        chatbot_output,                     # session_log
        [],                                 # session_messages → empty
        [],                                 # attached_files → empty
        "New session started.",             # status
        False,                              # has_ai_response
        *update_action_buttons("waiting_for_input", False),
        llm_state,                          # Preserve llm state
        models_loaded_state                 # Preserve models_loaded state
    )

def _get_cpu_default():
    """Helper function to get CPU default value."""
    import scripts.utility as utility
    cpu_info = utility.get_cpu_info()
    if len(cpu_info) > 1:
        return "Auto-Select"
    else:
        cpu_labs = [c["label"] for c in cpu_info]
        return cpu_labs[0] if cpu_labs else "Default CPU"

def load_session_by_index(idx):
    saved_sessions = utility.get_saved_sessions()
    if idx >= len(saved_sessions):
        chatbot_output = get_chatbot_output([], [])
        return (chatbot_output, [], [], "No session found.", False) + tuple(update_action_buttons("waiting_for_input", False))
    
    filename = saved_sessions[idx]
    session_id, label, history, attached_files = utility.load_session_history(filename)
    
    if not session_id:
        chatbot_output = get_chatbot_output([], [])
        return (chatbot_output, [], [], "Error loading session.", False) + tuple(update_action_buttons("waiting_for_input", False))
    
    cfg.current_session_id = session_id
    cfg.session_label = label
    cfg.session_attached_files = attached_files
    cfg.SESSION_ACTIVE = True
    
    # Check if session has AI response
    has_ai = any(msg.get('role') == 'assistant' for msg in history)
    
    status = f"Loaded session: {label}"
    # Return correct format for Gradio version
    chatbot_output = get_chatbot_output(messages_to_tuples(history), history)
    
    # Return action button updates as well (5 additional outputs)
    button_updates = update_action_buttons("waiting_for_input", has_ai)
    
    return (chatbot_output, history, attached_files, status, has_ai) + tuple(button_updates)

def edit_previous_prompt(session_tuples, session_messages):
    """Remove the last exchange and put the user's message back in the input box for editing."""
    if not session_messages or len(session_messages) < 2:
        chatbot_output = get_chatbot_output(session_tuples, session_messages)
        return chatbot_output, session_messages, "", "No previous message to edit.", False
    
    # Get the last user message
    last_user_msg = ""
    
    # Find and remove the last user-assistant pair
    if session_messages[-1].get('role') == 'assistant':
        session_messages = session_messages[:-1]
    if session_messages and session_messages[-1].get('role') == 'user':
        last_user_msg = session_messages[-1].get('content', '')
        session_messages = session_messages[:-1]
    
    # Update tuples from messages
    session_tuples = messages_to_tuples(session_messages)
    
    has_ai_response = len([m for m in session_messages if m.get('role') == 'assistant']) > 0
    
    # Return correct format for Gradio version
    chatbot_output = get_chatbot_output(session_tuples, session_messages)
    return chatbot_output, session_messages, last_user_msg, "Editing previous message.", has_ai_response

def copy_last_response(session_messages):
    """Copy last AI response to clipboard, excluding thinking phase"""
    if session_messages and session_messages[-1].get('role') == 'assistant':
        response = session_messages[-1]['content']
        
        clean_response = re.sub(r'<[^>]+>', '', response)
        
        lines = clean_response.split('\n')
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Thinking") and all(c in '.… ' for c in stripped[8:]):
                continue
            filtered_lines.append(line)
        
        clean_response = '\n'.join(filtered_lines).strip()
        
        pyperclip.copy(clean_response)
        return "AI Response copied to clipboard (thinking phase excluded)."
    return "No response available to copy."

def update_session_buttons():
    """Update session history buttons."""
    sessions = get_saved_sessions()[:cfg.MAX_HISTORY_SLOTS]
    button_updates = []
    
    for i in range(cfg.MAX_POSSIBLE_HISTORY_SLOTS):
        # Hide buttons beyond current MAX_HISTORY_SLOTS setting
        if i >= cfg.MAX_HISTORY_SLOTS:
            button_updates.append(gr.update(value="", visible=False))
        elif i < len(sessions):
            session_path = Path(cfg.HISTORY_DIR) / sessions[i]
            try:
                stat = session_path.stat()
                update_time = stat.st_mtime if stat.st_mtime else stat.st_ctime
                formatted_time = datetime.fromtimestamp(update_time).strftime("%Y-%m-%d %H:%M")
                
                session_id, label, history, attached_files = load_session_history(session_path)
                btn_label = f"{formatted_time} - {label}"
            except Exception as e:
                print(f"Error loading session {session_path}: {e}")
                btn_label = f"Session {i+1}"
            button_updates.append(gr.update(value=btn_label, visible=True))
        else:
            button_updates.append(gr.update(value="", visible=False))
    
    return button_updates

def format_session_id(session_id):
    """Format session ID into a readable date-time string."""
    try:
        dt = datetime.strptime(session_id, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return session_id

def update_action_buttons(phase, has_ai_response=False):
    """Update action buttons based on interaction phase and history state."""
    # Button visibility logic based on phase AND history state
    if phase == "waiting_for_input":
        if has_ai_response:
            # Has history: show all action buttons
            action_visible, edit_visible, copy_visible, wait_visible = True, True, True, False
        else:
            # Fresh session: only Send Input visible
            action_visible, edit_visible, copy_visible, wait_visible = True, False, False, False
    elif phase in ("input_submitted", "generating_response", "speaking"):
        # During generation: hide action buttons, show wait indicator
        action_visible, edit_visible, copy_visible, wait_visible = False, False, False, True
    else:
        action_visible, edit_visible, copy_visible, wait_visible = True, False, False, False

    # Configure Send Input button appearance
    action_value = "Send Input"
    action_variant = "secondary" if phase == "waiting_for_input" else "primary"
    action_classes = ["send-button-green"] if phase == "waiting_for_input" else []
    action_interactive = (phase == "waiting_for_input")

    # Configure wait indicator
    wait_value = "..Wait For Response.."
    if phase == "speaking" and getattr(cfg, 'TTS_ENABLED', False):
        wait_value = "🔊 Speaking Response..."

    return [
        gr.update(value=action_value, variant=action_variant, elem_classes=action_classes, interactive=action_interactive, visible=action_visible),
        gr.update(visible=edit_visible),
        gr.update(visible=copy_visible),
        gr.update(visible=False),  # cancel_input (placeholder)
        gr.update(value=wait_value, variant="primary", interactive=False, visible=wait_visible)
    ]

def handle_model_inspect(model_name):
    """Handle model inspection."""
    from scripts.inference import inspect_model
    
    if not model_name or model_name == "Select_a_model...":
        return "Select a model to inspect."
    
    return inspect_model(model_name)

def delete_all_sessions():
    """Delete all session history files."""
    import shutil
    history_path = Path(cfg.HISTORY_DIR)
    
    try:
        for file in history_path.glob("*.json"):
            file.unlink()
        return "All session history deleted."
    except Exception as e:
        return f"Error deleting history: {str(e)}"

def handle_customization_save(
    max_hist, height, max_att, show_think, print_raw, bleep,
    ctx, batch, temp, repeat, vram, gpu, cpu, cpu_threads, model, model_path,
    layer_allocation_mode=None
):
    """Save ALL configuration settings to both temporary globals and persistent.json."""
    try:
        # Check if context size changed while model is loaded
        new_ctx = int(ctx) if ctx is not None else cfg.CONTEXT_SIZE
        ctx_changed = (
            cfg.LOADED_CONTEXT_SIZE is not None and 
            new_ctx != cfg.LOADED_CONTEXT_SIZE
        )
        
        # Program settings (customization) - with None checks and defaults
        cfg.MAX_HISTORY_SLOTS = int(max_hist) if max_hist is not None else cfg.MAX_HISTORY_SLOTS
        cfg.SESSION_LOG_HEIGHT = int(height) if height is not None else cfg.SESSION_LOG_HEIGHT
        cfg.MAX_ATTACH_SLOTS = int(max_att) if max_att is not None else cfg.MAX_ATTACH_SLOTS
        cfg.SHOW_THINK_PHASE = bool(show_think) if show_think is not None else cfg.SHOW_THINK_PHASE
        cfg.PRINT_RAW_OUTPUT = bool(print_raw) if print_raw is not None else cfg.PRINT_RAW_OUTPUT
        cfg.BLEEP_ON_EVENTS = bool(bleep) if bleep is not None else cfg.BLEEP_ON_EVENTS
        
        # Model/Hardware settings - with None checks
        cfg.CONTEXT_SIZE = int(ctx) if ctx is not None else cfg.CONTEXT_SIZE
        cfg.BATCH_SIZE = int(batch) if batch is not None else cfg.BATCH_SIZE
        cfg.TEMPERATURE = float(temp) if temp is not None else cfg.TEMPERATURE
        cfg.REPEAT_PENALTY = float(repeat) if repeat is not None else cfg.REPEAT_PENALTY
        cfg.VRAM_SIZE = int(vram) if vram is not None else cfg.VRAM_SIZE
        cfg.SELECTED_GPU = gpu if gpu is not None else cfg.SELECTED_GPU
        cfg.SELECTED_CPU = cpu if cpu is not None else cfg.SELECTED_CPU
        cfg.CPU_THREADS = int(cpu_threads) if cpu_threads is not None else cfg.CPU_THREADS
        cfg.MODEL_NAME = model if model is not None else cfg.MODEL_NAME
        cfg.MODEL_FOLDER = model_path if model_path is not None else cfg.MODEL_FOLDER
        
        # Layer allocation mode (only if Vulkan available)
        if layer_allocation_mode is not None and cfg.VULKAN_AVAILABLE:
            cfg.LAYER_ALLOCATION_MODE = layer_allocation_mode
        
        # Save to persistent.json
        from scripts.configuration import save_config
        save_config()
        
        print(f"[SAVE] Settings saved: CTX={cfg.CONTEXT_SIZE}, Batch={cfg.BATCH_SIZE}, "
              f"Temp={cfg.TEMPERATURE}, VRAM={cfg.VRAM_SIZE}, GPU={cfg.SELECTED_GPU}, "
              f"CPU={cfg.SELECTED_CPU}, Threads={cfg.CPU_THREADS}")
        
        # Return appropriate message
        if ctx_changed:
            return f"Settings saved. NOTE: Context size changed to {new_ctx} but model loaded at {cfg.LOADED_CONTEXT_SIZE}. Reload model to apply new context size."
        
        return "Settings saved successfully."
    except Exception as e:
        print(f"[SAVE] Error: {e}")
        return f"Error saving settings: {str(e)}"

def update_backend_ui():
    """Update UI components based on backend type and saved layer allocation mode."""
    backend_type = cfg.BACKEND_TYPE
    vulkan_available = cfg.VULKAN_AVAILABLE
    layer_mode = cfg.LAYER_ALLOCATION_MODE
    
    allocation_visible = vulkan_available
    gpu_visible = backend_type in ["VULKAN_VULKAN", "VULKAN_CPU"]
    vram_visible = gpu_visible and layer_mode == "VRAM_SRAM"
    gpu_row_visible = gpu_visible or vram_visible
    
    print(f"[UI-INIT] Backend: {backend_type}, Layer mode: {layer_mode}, Vulkan: {vulkan_available}")
    print(f"[UI-INIT] Allocation visible: {allocation_visible}, VRAM visible: {vram_visible}")
    
    return [
        gr.update(value=backend_type),
        gr.update(visible=allocation_visible, value=layer_mode),
        gr.update(visible=gpu_row_visible),
        gr.update(visible=gpu_visible),
        gr.update(visible=vram_visible),
        gr.update(visible=True),
        gr.update()
    ]

def build_progress_html(step: int, ddg_search_enabled: bool = False, 
                        web_search_enabled: bool = False,
                        tts_enabled: bool = False):
    """
    Build dynamic progress indicator HTML based on enabled features.
    
    Cases:
    1. Vanilla (no search, no TTS): 9 steps (0-8)
    2. DDG hybrid search: 14 steps (adds 5 search phases + Inject Context)
    3. Web search: 14 steps (adds 5 search phases + Inject Context)  
    4. TTS enabled: +2 steps (Generating TTS, Playing TTS)
    
    Note: DDG and Web search are mutually exclusive.
    """
    # Base phases (always present) - indices 0-5
    base_phases = [
        "Handle Input",      # 0
        "Build Prompt",      # 1
        "Inject RAG",        # 2
        "Add System",        # 3
        "Assemble History",  # 4
        "Check Model",       # 5
    ]
    
    phases = base_phases.copy()
    
    # Add search phases if either search mode is enabled (they're mutually exclusive)
    if ddg_search_enabled:
        phases.append("DDG Pre-Search")    # 6
        phases.append("Analyze Results")   # 7
        phases.append("Deep Fetch")        # 8
        phases.append("Merge Results")     # 9
        phases.append("Inject Context")    # 10
    elif web_search_enabled:
        phases.append("Search Discovery")  # 6
        phases.append("Rank & Select")     # 7
        phases.append("Parallel Fetch")    # 8
        phases.append("Process & Merge")   # 9
        phases.append("Inject Context")    # 10
    
    # Add generation phases
    phases.extend([
        "Generate Stream",   # 6 or 11
        "Split Thinking",    # 7 or 12
        "Format Response"    # 8 or 13
    ])
    
    # Add TTS phases if enabled
    if tts_enabled:
        phases.append("Generating TTS")  # 9 or 14
        phases.append("Playing TTS")     # 10 or 15
    
    # Build HTML segments
    segments = []
    for i, phase in enumerate(phases):
        if i < step:
            color = "#00ff00"  # Completed - green
        elif i == step:
            color = "#4488ff"  # Current - blue
        else:
            color = "#666666"  # Pending - gray
        segments.append(f'<span style="color:{color}; font-weight:bold;">{phase}</span>')
    
    return " → ".join(segments)

def handle_cpu_threads_change(new_threads):
    """Handle CPU threads slider changes"""
    cfg.CPU_THREADS = int(new_threads)
    return f"CPU threads set to {new_threads}"

def extract_search_query(user_input: str) -> str:
    """
    Extract a clean, focused search query from natural language user input.
    Prioritizes substantive content over meta/discussion text.
    """
    import re
    
    original = user_input.strip()
    
    # Step 1: Split on fallback instructions (keep only the first part)
    fallback_patterns = [
        r'\s*if\s+(?:you\s+)?(?:cannot|can\'t|could\s+not).*$',  # "if you cannot..."
        r'\s*if\s+that\s+fails.*$',  # "if that fails..."
        r'\s*otherwise.*$',  # "otherwise..."
        r'\s*alternatively.*$',  # "alternatively..."
        r'\s+in\s+which\s+case.*$',  # "in which case..."
        r'\s+if\s+not.*$',  # "if not..."
    ]
    
    working_text = original
    for pattern in fallback_patterns:
        working_text = re.sub(pattern, '', working_text, flags=re.IGNORECASE | re.DOTALL)
    
    # Step 2: Remove meta/development/testing preambles (CRITICAL FIX)
    # These patterns remove "I am developing...", "Here is a test...", etc.
    meta_patterns = [
        r'^.*?(?:i\'?m?|i\s+am)\s+(?:developing|building|creating|working\s+on|testing)\s+(?:the|my|this)?\s*(?:internet\s+based\s+tools?|tools?|chatbot|ai|program|script|features?).*?(?:so\s+here|here)\s+(?:is|are|goes)\s*(?:a\s+)?(?:test|example).*?\n+',
        r'^.*?(?::\s*\n+|\.\.\.\n+|,\s*so\s+here\s+is\s+a\s+test[:\s]*\n+)',
        r'^(?:test|testing)[:\s-]+',
        r'^.*?(?:just|simply)?\s*(?:trying\s+to|want\s+to|need\s+to)\s+test\s+.*?(?:please|find|search|look)',
    ]
    
    query = working_text
    for pattern in meta_patterns:
        query = re.sub(pattern, '', query, flags=re.IGNORECASE)
    
    # Step 3: If query is still mostly meta-text, look for the actual topic sentence
    # The real query usually contains specific entities (Iran, 2026, etc.)
    sentences = re.split(r'(?<=[.!?])\s+', query)
    
    # Score each sentence - prefer sentences with:
    # - Proper nouns (capitalized words not at start)
    # - Years/dates
    # - Location names
    # - Action words (find, search, what, tell me about)
    best_sentence = ""
    best_score = -1
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue
            
        score = 0
        
        # Prefer sentences with capitalized words in the middle (proper nouns)
        words = sentence.split()
        for i, word in enumerate(words):
            if i > 0 and word[0].isupper() and len(word) > 2:
                score += 3  # Proper noun likely
        
        # Prefer sentences with years
        if re.search(r'\b20\d\d\b', sentence):
            score += 5
        
        # Prefer sentences with location indicators
        location_words = ['iran', 'iraq', 'syria', 'ukraine', 'russia', 'china', 'protest', 'uprising', 'war', 'election', 'news']
        for loc in location_words:
            if loc in sentence.lower():
                score += 4
        
        # Demerit for meta words
        meta_words = ['chatbot', 'test', 'testing', 'developing', 'tool', 'feature', 'internet', 'based', 'i am', "i'm"]
        for meta in meta_words:
            if meta in sentence.lower():
                score -= 2
        
        if score > best_score:
            best_score = score
            best_sentence = sentence
    
    # If we found a good substantive sentence, use it
    if best_score > 0:
        query = best_sentence
    else:
        # Fallback: if no good sentence found, use what we have but clean it
        pass
    
    # Step 4: Remove common request wrappers
    wrappers = [
        r'^(?:can\s+you|could\s+you|would\s+you|will\s+you)?\s*(?:please\s+)?(?:search|find|look\s+up|research|google|check|investigate|tell\s+me)\s+(?:for\s+|about\s+|on\s+|into\s+)?',
        r'^(?:what\s+(?:is|are)\s+(?:the\s+)?(?:latest|current|recent|new)\s+(?:news|information|updates|developments)\s+(?:on|about|regarding)\s+)',
        r'^(?:find\s+(?:out\s+)?(?:what\s+you\s+can\s+)?(?:about\s+)?)',
        r'^(?:i\s+(?:want|need|would\s+like)\s+(?:to\s+know\s+)?(?:about\s+)?)',
    ]
    
    for pattern in wrappers:
        query = re.sub(pattern, '', query, flags=re.IGNORECASE)
    
    # Step 5: Clean up
    query = query.strip()
    query = re.sub(r'^["\']+', '', query)  # Remove leading quotes
    query = re.sub(r'["\']+$', '', query)  # Remove trailing quotes
    query = re.sub(r'\s+', ' ', query)  # Normalize whitespace
    query = re.sub(r'[.!?]+$', '', query)  # Remove trailing punctuation
    
    # If query is empty or too short, use a keyword extraction fallback
    if len(query) < 5:
        # Extract all capitalized words and years from original
        keywords = []
        # Find capitalized phrases (proper nouns)
        for match in re.finditer(r'\b[A-Z][a-zA-Z]{2,}\b', original):
            word = match.group()
            if word.lower() not in ['i', 'the', 'a', 'an', 'and', 'or', 'but', 'so', 'here', 'test']:
                keywords.append(word)
        # Find years
        for match in re.finditer(r'\b20\d\d\b', original):
            keywords.append(match.group())
        
        query = ' '.join(keywords[:6]) if keywords else original[:80]
    
    # Step 6: Truncate if too long
    if len(query) > 80:
        query = query[:80].rsplit(' ', 1)[0]
    
    print(f"[SEARCH-QUERY] Original: '{original[:60]}...' → Extracted: '{query}'")
    return query

def toggle_tts_sound(current_state):
    """Toggle TTS Sound enabled/disabled."""
    import gradio as gr
    import scripts.configuration as cfg
    new_state = not current_state
    cfg.TTS_ENABLED = new_state

    variant = "primary" if new_state else "secondary"
    label = "🔊 TTS Sound ON" if new_state else "🔊 TTS Sound"

    # Get proper status text
    status = get_tts_status()

    return (
        new_state,
        gr.update(variant=variant, value=label),
        gr.update(variant=variant),
        status
    )


def speak_response_handler(session_messages):
    """Handle speaking the last AI response."""
    result = speak_last_response(session_messages)
    return result, result


def stop_tts_handler():
    """Stop any ongoing TTS playback."""
    stop_speaking()
    return "Speech stopped", "Speech stopped"


def update_tts_voice(voice_name):
    """Update the selected TTS voice."""
    import scripts.configuration as cfg
    
    # Get the voice ID from the name
    voice_id = get_voice_id_by_name(voice_name)
    cfg.TTS_VOICE = voice_id
    cfg.TTS_VOICE_NAME = voice_name
    
    # Update configuration
    from scripts.configuration import save_config
    save_config()
    
    return f"Voice set to: {voice_name}"


def update_sound_output_device(device_name):
    """Deprecated - device selection now uses system default only."""
    cfg.SOUND_OUTPUT_DEVICE = "Default Sound Device"
    return "Audio output: Default Sound Device"


def update_sound_sample_rate(sample_rate):
    """Update the audio sample rate (shared by Bleep and TTS)."""
    import scripts.configuration as cfg
    cfg.SOUND_SAMPLE_RATE = int(sample_rate)
    return f"Sample rate set to: {sample_rate}"

# Global cancel event
import threading
_cancel_event = threading.Event()

def conversation_display(
    user_input, session_tuples, session_messages, loaded_files,
    is_reasoning_model, cancel_flag, ddg_search_enabled, web_search_enabled,
    interaction_phase, llm_state, models_loaded_state,
    has_ai_response_state, tts_enabled
):
    """
    Main conversation handler - Gradio 3.x compatible.
    Uses tuple format for Chatbot display, message dicts internally.
    """
    import gradio as gr
    from scripts import configuration, utility
    from scripts.inference import get_model_settings, get_response_stream, load_models
    from scripts.configuration import context_injector
    from scripts.utility import read_file_content, filter_operational_content
    from pathlib import Path
    import time
    import re
    from datetime import datetime

    # CRITICAL FIX: Recover from Gradio state reset (e.g., after New Session)
    # If Gradio states were reset but global config still holds the model, restore them
    if llm_state is None and cfg.llm is not None:
        llm_state = cfg.llm
        models_loaded_state = cfg.MODELS_LOADED
        print("[CONVERSATION] Restored model state from global config after session reset")
    
    # Update last interaction time (used for auto-unload)
    cfg.LAST_INTERACTION_TIME = time.time()

    
    # AUTO-LOAD MODEL IF NOT LOADED YET
    
    if not models_loaded_state or llm_state is None:
        if cfg.MODEL_NAME in ["Select_a_model...", "No models found", "", None]:
            yield (
                get_chatbot_output(session_tuples, session_messages),
                session_messages,
                "❌ No valid model selected. Please choose one in Configuration tab.",
                gr.update(visible=True), gr.update(visible=False),
                *update_action_buttons("waiting_for_input", has_ai_response_state),
                cancel_flag, loaded_files, "waiting_for_input",
                llm_state, models_loaded_state, tts_enabled
            )
            return

        model_path = Path(cfg.MODEL_FOLDER) / cfg.MODEL_NAME
        if not model_path.is_file():
            yield (
                get_chatbot_output(session_tuples, session_messages),
                session_messages,
                f"❌ Model file missing: {cfg.MODEL_NAME}\nCheck folder path.",
                gr.update(visible=True), gr.update(visible=False),
                *update_action_buttons("waiting_for_input", has_ai_response_state),
                cancel_flag, loaded_files, "waiting_for_input",
                llm_state, models_loaded_state, tts_enabled
            )
            return

        # Disable input during load
        yield (
            get_chatbot_output(session_tuples, session_messages),
            session_messages,
            "⏳ Loading model... (this may take 10–120 seconds) ",
            gr.update(visible=False),
            gr.update(visible=True, value="Auto-loading model — please wait... "),
            *update_action_buttons("input_submitted", has_ai_response_state),
            cancel_flag, loaded_files, "input_submitted",
            llm_state, models_loaded_state, tts_enabled
        )

        try:
            status, loaded, new_llm, _ = load_models(
                cfg.MODEL_FOLDER,
                cfg.MODEL_NAME,
                cfg.VRAM_SIZE,
                llm_state,
                models_loaded_state
            )

            if loaded:
                cfg.MODELS_LOADED = True
                cfg.llm = new_llm
                cfg.LOADED_CONTEXT_SIZE = cfg.CONTEXT_SIZE
                cfg.LAST_INTERACTION_TIME = time.time()

                yield (
                    get_chatbot_output(session_tuples, session_messages),
                    session_messages,
                    f"✅ Model loaded: {cfg.MODEL_NAME} ",
                    gr.update(visible=False),
                    gr.update(visible=True, value="Model ready — processing your message... "),
                    *update_action_buttons("input_submitted", has_ai_response_state),
                    cancel_flag, loaded_files, "input_submitted",
                    new_llm, True, tts_enabled  # CRITICAL: Return raw values, not gr.update()
                )

                # Update local variables so rest of function uses loaded model
                llm_state = new_llm
                models_loaded_state = True

            else:
                yield (
                    get_chatbot_output(session_tuples, session_messages),
                    session_messages,
                    f"❌ Model load failed: {status}",
                    gr.update(visible=True),
                    gr.update(visible=False),
                    *update_action_buttons("waiting_for_input", has_ai_response_state),
                    cancel_flag, loaded_files, "waiting_for_input",
                    llm_state, False, tts_enabled  # Return raw False, not gr.update()
                )
                return

        except Exception as e:
            yield (
                get_chatbot_output(session_tuples, session_messages),
                session_messages,
                f"❌ Load error: {str(e)[:120]}",
                gr.update(visible=True),
                gr.update(visible=False),
                *update_action_buttons("waiting_for_input", has_ai_response_state),
                cancel_flag, loaded_files, "waiting_for_input",
                llm_state, False, tts_enabled  # Return raw False
            )
            return
    
    # ── Proceed to input validation AFTER auto-load completes ────────────────
    if not user_input.strip():
        yield (
            get_chatbot_output(session_tuples, session_messages),
            session_messages,
            " ",
            gr.update(visible=True),
            gr.update(visible=False),
            *update_action_buttons("waiting_for_input", has_ai_response_state),
            cancel_flag,
            loaded_files,
            "waiting_for_input",
            llm_state,
            models_loaded_state,  # Return state value directly
            tts_enabled
        )
        return

    # Determine if there's existing AI response history (before we add new message)
    has_ai_response = len([m for m in session_messages if m.get('role') == 'assistant']) > 0
    interaction_phase = "input_submitted"
    
    # Initialize variables that will be built in phases
    processed_input = user_input
    file_contents_section = ""
    complete_user_message = ""
    search_results = None

    
    # PHASE 0: Handle Input - Validate and prepare input
    
    yield (
        get_chatbot_output(session_tuples, session_messages), 
        session_messages, 
        " ",
        gr.update(visible=False, value=""),
        gr.update(visible=True, value=build_progress_html(4, ddg_search_enabled, web_search_enabled, tts_enabled)),
        *update_action_buttons("input_submitted", has_ai_response),
        cancel_flag, 
        loaded_files, 
        "input_submitted",
        llm_state, 
        models_loaded_state,  # Return state value directly
        tts_enabled
    )

    
    # PHASE 1: Build Prompt - Process user input and attached files
    
    processed_input = user_input
    file_contents_section = ""
    
    if loaded_files:
        file_parts = []
        for file_path in loaded_files:
            try:
                content, file_type, success, error = read_file_content(file_path)
                filename = Path(file_path).name
                
                if success and file_type == "text":
                    display_content = content[:8000] + "\n[...truncated...]" if len(content) > 8000 else content
                    file_parts.append(f"\n\n--- Attached File: {filename} ---\n{display_content}")
                elif success and file_type == "image":
                    file_parts.append(f"\n\n--- Attached Image: {filename} ---\n[Image attached for vision processing]")
                else:
                    file_parts.append(f"\n\n--- Attached File: {filename} ---\n[Error reading file: {error}]")
            except Exception as e:
                filename = Path(file_path).name if file_path else "Unknown"
                file_parts.append(f"\n\n--- Attached File: {filename} ---\n[Error: {str(e)}]")
        
        if file_parts:
            file_contents_section = "\n".join(file_parts)
    
    yield (
        get_chatbot_output(session_tuples, session_messages), session_messages, "",
        gr.update(visible=False), gr.update(visible=True, value=build_progress_html(1, ddg_search_enabled, web_search_enabled, tts_enabled)),
        *update_action_buttons("input_submitted", has_ai_response),
        cancel_flag, loaded_files, "input_submitted",
        llm_state, models_loaded_state, tts_enabled  # Return state values
    )

    
    # PHASE 2: Inject RAG - Process large inputs through RAG system
    
    # Use LOADED context, not UI setting (model may have been loaded at different size)
    effective_context = cfg.LOADED_CONTEXT_SIZE or cfg.CONTEXT_SIZE
    context_threshold = cfg.LARGE_INPUT_THRESHOLD
    max_input_chars = int(effective_context * 3 * context_threshold)  # ~chars, not tokens
    
    # Calculate total input size (user input + attached file contents)
    total_input_size = len(user_input) + len(file_contents_section)
    
    if total_input_size > max_input_chars:
        try:
            # RAG the user input if it's the large part
            if len(user_input) > max_input_chars // 2:
                context_injector.add_temporary_input(user_input)
                processed_input = user_input[:1000] + "\n\n[Large input processed with RAG]"
            
            # Also RAG attached files if they're large
            if len(file_contents_section) > max_input_chars // 2:
                context_injector.add_temporary_input(file_contents_section)
                file_contents_section = file_contents_section[:2000] + "\n\n[Large attachments indexed for retrieval]"
                
        except Exception as e:
            print(f"[RAG-TEMP] Error: {e}")
    
    yield (
        get_chatbot_output(session_tuples, session_messages), session_messages, "",
        gr.update(visible=False), gr.update(visible=True, value=build_progress_html(2, ddg_search_enabled, web_search_enabled, tts_enabled)),
        *update_action_buttons("input_submitted", has_ai_response),
        cancel_flag, loaded_files, "input_submitted",
        llm_state, models_loaded_state, tts_enabled  # Return state values
    )
    
    # PHASE 3: Add System - Build complete user message with file contents
    
    complete_user_message = processed_input
    if file_contents_section:
        complete_user_message = processed_input + file_contents_section
    
    yield (
        get_chatbot_output(session_tuples, session_messages), session_messages, "",
        gr.update(visible=False), gr.update(visible=True, value=build_progress_html(3, ddg_search_enabled, web_search_enabled, tts_enabled)),
        *update_action_buttons("input_submitted", has_ai_response),
        cancel_flag, loaded_files, "input_submitted",
        llm_state, models_loaded_state, tts_enabled  # Return state values
    )
    
    # PHASE 4: Assemble History - Add message to session and update display
    session_messages = list(session_messages) if session_messages else []
    session_messages.append({'role': 'user', 'content': complete_user_message})

    session_tuples = list(session_tuples) if session_tuples else []
    user_display = f"User:\n{complete_user_message}"
    session_tuples.append((user_display, None))

    has_ai_response = len([m for m in session_messages[:-1] if m.get('role') == 'assistant']) > 0
    interaction_phase = "input_submitted"

    # CRITICAL FIX: Clear input box IMMEDIATELY and hide it
    yield (
        get_chatbot_output(session_tuples, session_messages), 
        session_messages, 
        " ",
        gr.update(visible=False, value=""),
        gr.update(visible=True, value=build_progress_html(4, ddg_search_enabled, web_search_enabled, tts_enabled)),
        *update_action_buttons("input_submitted", has_ai_response),
        cancel_flag, 
        loaded_files, 
        "input_submitted",
        llm_state, 
        models_loaded_state,  # Return state value directly
        tts_enabled
    )
    
    # PHASE 5: Check Model - Get model settings and prepare for generation
    
    model_settings = get_model_settings(cfg.MODEL_NAME)
    
    
    # SEARCH LOGIC: Handle DDG Search and Web Search (mutually exclusive)
    
    search_results = None
    search_metadata = None
    search_status_text = ""
    search_warning = ""
    
    # Determine which search mode is active (only one can be active at a time)
    search_active = ddg_search_enabled or web_search_enabled
    
    if search_active and user_input.strip():
        try:
            # Calculate context-scaled search parameters
            # Base values are calibrated for 32768 context (multiplier = 1.0)
            effective_ctx = cfg.LOADED_CONTEXT_SIZE or cfg.CONTEXT_SIZE
            context_multiplier = effective_ctx / 32768.0
            
            # Warn if context is below recommended minimum for search
            if effective_ctx < 16384:
                search_warning = f"⚠️ Low context ({effective_ctx}) may limit search quality. Recommend 32768+."
                print(f"[SEARCH-WARNING] {search_warning}")
            
            # Scale parameters based on context (min values ensure functionality)
            # DDG Search: base 8 results, 4 deep fetch at 32k context
            # Web Search: base 12 results, 6 deep fetch at 32k context
            if ddg_search_enabled:
                scaled_ddg_results = max(4, round(8 * context_multiplier))
                scaled_deep_fetch = max(2, round(4 * context_multiplier))
            else:
                scaled_max_results = max(6, round(12 * context_multiplier))
                scaled_deep_fetch = max(3, round(6 * context_multiplier))
            
            # Use proper query extraction
            search_query = extract_search_query(user_input)
            
            # DEBUG: Log the extracted query
            print(f"[SEARCH-DEBUG] Original input: {user_input[:100]}...")
            print(f"[SEARCH-DEBUG] Extracted query: '{search_query}'")
            
            if ddg_search_enabled:
                # DDG Hybrid Search with scaled parameters
                print(f"[HYBRID-SEARCH] Query: {search_query} (ctx={effective_ctx}, results={scaled_ddg_results}, deep={scaled_deep_fetch})")
                result = utility.hybrid_search(search_query, ddg_results=scaled_ddg_results, deep_fetch=scaled_deep_fetch)
            else:
                # Comprehensive Web Search with scaled parameters
                print(f"[WEB-SEARCH] Query: {search_query} (ctx={effective_ctx}, results={scaled_max_results}, deep={scaled_deep_fetch})")
                result = utility.web_search(search_query, max_results=scaled_max_results, deep_fetch=scaled_deep_fetch)
            
            # Extract content and metadata from dict return format
            if isinstance(result, dict):
                search_results = result.get('content', '')
                search_metadata = result.get('metadata', {})
            else:
                search_results = result
                search_type = 'hybrid' if ddg_search_enabled else 'web_search'
                search_metadata = {'type': search_type, 'query': search_query, 'sources': [], 'error': None}
            
            # Add context scaling info to metadata
            if search_metadata:
                search_metadata['context_size'] = effective_ctx
                search_metadata['context_multiplier'] = round(context_multiplier, 2)
            
            # Format search status for chat display
            if search_metadata:
                search_status_text = utility.format_search_status_for_chat(search_metadata)
                if search_warning:
                    search_status_text = f"{search_warning}\n{search_status_text}" if search_status_text else search_warning
                if search_status_text:
                    print(f"[SEARCH-STATUS]\n{search_status_text}")
                
        except Exception as e:
            print(f"[SEARCH] Error: {e}")
            import traceback
            traceback.print_exc()
            search_results = f"Search error: {str(e)}"
            search_type = 'hybrid' if ddg_search_enabled else 'web_search'
            search_metadata = {'type': search_type, 'query': user_input[:100], 'error': str(e), 'sources': []}
            search_status_text = f"⚠️ Search Error: {str(e)}"
    
    yield (
        get_chatbot_output(session_tuples, session_messages), session_messages, "",
        gr.update(visible=False), gr.update(visible=True, value=build_progress_html(5, ddg_search_enabled, web_search_enabled, tts_enabled)),
        *update_action_buttons("input_submitted", has_ai_response),
        cancel_flag, loaded_files, "input_submitted",
        llm_state, models_loaded_state, tts_enabled  # Return state values
    )

    # Search progress phases (if any search is enabled)
    if ddg_search_enabled or web_search_enabled:
        for phase_idx in [6, 7, 8, 9, 10]:
            yield (
                get_chatbot_output(session_tuples, session_messages), session_messages, "",
                gr.update(visible=False), gr.update(visible=True, value=build_progress_html(phase_idx, ddg_search_enabled, web_search_enabled, tts_enabled)),
                *update_action_buttons("input_submitted", has_ai_response),
                cancel_flag, loaded_files, "input_submitted",
                llm_state, models_loaded_state, tts_enabled  # Return state values
            )
            time.sleep(0.15)

    # Calculate Generate Stream phase index
    if ddg_search_enabled or web_search_enabled:
        generate_phase = 11
    else:
        generate_phase = 6

    # Phase: Generating response
    interaction_phase = "generating_response"
    _cancel_event.clear()

    # Add placeholder for assistant
    session_messages.append({'role': 'assistant', 'content': ""})
    
    accumulated_response = ""
    
    try:
        yield (
            get_chatbot_output(session_tuples, session_messages), session_messages, "",
            gr.update(visible=False), gr.update(visible=True, value=build_progress_html(generate_phase, ddg_search_enabled, web_search_enabled, tts_enabled)),
            *update_action_buttons("generating_response", has_ai_response),
            cancel_flag, loaded_files, "generating_response",
            llm_state, models_loaded_state, tts_enabled  # Return state values
        )
        
        # Stream the response
        for chunk in get_response_stream(
            session_log=session_messages,
            settings=model_settings,
            ddg_search_enabled=ddg_search_enabled,
            search_results=search_results,
            cancel_event=_cancel_event,
            llm_state=llm_state,
            models_loaded_state=models_loaded_state
        ):
            if _cancel_event.is_set():
                accumulated_response += "\n\n[Response cancelled]"
                break
            
            accumulated_response += chunk
            
            # Filter out any AI-Chat: prefix the model might generate
            clean_response = re.sub(r'^AI-Chat:\s*\n?', '', accumulated_response, flags=re.MULTILINE)
            clean_response = re.sub(r'\nAI-Chat:\s*\n?', '\n', clean_response)
            
            # Update assistant message (store clean version)
            session_messages[-1]['content'] = clean_response
            
            # Update tuples - add "AI-Chat:" label for display
            bot_display = f"AI-Chat:\n{clean_response}"
            session_tuples[-1] = (session_tuples[-1][0], bot_display)
            
            # For Gradio 4/5, we also need to update session_messages with label
            if not cfg.GRADIO_VERSION.startswith('3.'):
                session_messages[-1]['content'] = bot_display
            
            yield (
                get_chatbot_output(session_tuples, session_messages), session_messages, "",
                gr.update(visible=False), gr.update(visible=True, value=build_progress_html(generate_phase, ddg_search_enabled, web_search_enabled, tts_enabled)),
                *update_action_buttons("generating_response", has_ai_response),
                cancel_flag, loaded_files, "generating_response",
                llm_state, models_loaded_state, tts_enabled  # Return state values
            )
            
    except Exception as e:
        accumulated_response = f"Error: {str(e)}"
        session_messages[-1]['content'] = accumulated_response
        session_tuples[-1] = (session_tuples[-1][0], accumulated_response)

    # Final formatting
    formatted_response = format_response(accumulated_response)
    # Clean any AI-Chat: prefix before storing
    formatted_response = re.sub(r'^AI-Chat:\s*\n?', '', formatted_response, flags=re.MULTILINE)
    formatted_response = re.sub(r'\nAI-Chat:\s*\n?', '\n', formatted_response)
    session_messages[-1]['content'] = formatted_response
    # Add label for display
    session_tuples[-1] = (session_tuples[-1][0], f"AI-Chat:\n{formatted_response}")

    #  NEW SESSION CREATION + AUTO-SAVE 
    if accumulated_response.strip():
        if not cfg.SESSION_ACTIVE:
            cfg.SESSION_ACTIVE = True
            cfg.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            cfg.session_label = utility.summarize_session(session_messages)
            print(f"[SESSION] New session created: {cfg.session_label}")

        try:
            utility.save_session_history(session_messages, loaded_files)
            print("[SESSION] Auto-saved after complete response")
        except Exception as e:
            print(f"[SESSION] Save error: {e}")
    

    # Clear temporary RAG
    try:
        context_injector.clear_temporary_input()
    except:
        pass

    
    # FIXED: Clear attached files after successful response
    
    cleared_files = []
    cfg.session_attached_files = []
    
    # Auto-speak response if TTS is enabled
    if tts_enabled:
        # Calculate TTS phase indices based on whether search was enabled
        if ddg_search_enabled or web_search_enabled:
            tts_gen_phase = 14
            tts_play_phase = 15
        else:
            tts_gen_phase = 9
            tts_play_phase = 10
        
        # Show Generating TTS phase while synthesis runs
        print("[TTS] Generating speech from response...")
        yield (
            get_chatbot_output(session_tuples, session_messages),
            session_messages,
            "Generating speech...",
            gr.update(visible=False),
            gr.update(visible=True, value=build_progress_html(tts_gen_phase, ddg_search_enabled, web_search_enabled, tts_enabled)),
            *update_action_buttons("speaking", True),
            False,
            [],
            "speaking",
            llm_state,
            models_loaded_state,
            tts_enabled
        )
        
        # BLOCKING: Synthesize audio (this takes time - progress stays on "Generating TTS")
        wav_path = synthesize_last_response(session_messages)
        
        # Switch to Playing TTS phase
        print("[TTS] Playing speech audio...")
        yield (
            get_chatbot_output(session_tuples, session_messages),
            session_messages,
            "Playing speech...",
            gr.update(visible=False),
            gr.update(visible=True, value=build_progress_html(tts_play_phase, ddg_search_enabled, web_search_enabled, tts_enabled)),
            *update_action_buttons("speaking", True),
            False,
            [],
            "speaking",
            llm_state,
            models_loaded_state,
            tts_enabled
        )
        
        # BLOCKING: Play the audio (progress stays on "Playing TTS")
        if wav_path:
            play_tts_audio(wav_path)
        print("[TTS] Speech playback complete")
    
    # Beep AFTER TTS completes (or immediately if TTS disabled)
    beep()
    
    # Final yield - with state values returned directly
    yield (
        get_chatbot_output(session_tuples, session_messages),
        session_messages,
        "Ready — response complete",
        gr.update(visible=True),
        gr.update(visible=False),
        *update_action_buttons("waiting_for_input", True),
        False,                          # cancel_flag reset
        [],                             # cleared attached_files
        "waiting_for_input",
        llm_state,                      # Return actual llm object
        models_loaded_state,            # Return actual boolean
        tts_enabled                     # Return tts state
    )

def toggle_left_expanded_state(current_state):
    return not current_state

def toggle_right_expanded_state(current_state):
    return not current_state

def toggle_ddg_search(current_ddg_state, current_web_state):
    """Toggle DDG Search (hybrid mode). Disables Web Search if enabling DDG."""
    import gradio as gr
    
    new_ddg_state = not current_ddg_state
    
    # If enabling DDG, disable Web Search (mutual exclusivity)
    new_web_state = False if new_ddg_state else current_web_state
    
    ddg_variant = "primary" if new_ddg_state else "secondary"
    ddg_label = "🔍 DDG Search ON" if new_ddg_state else "🔍 DDG Search"
    
    web_variant = "primary" if new_web_state else "secondary"
    web_label = "🌐 Web Search ON" if new_web_state else "🌐 Web Search"
    
    return (
        new_ddg_state,                                          # ddg_search_enabled state
        new_web_state,                                          # web_search_enabled state
        gr.update(variant=ddg_variant, value=ddg_label),        # ddg_search button
        gr.update(variant=ddg_variant),                         # ddg_search_collapsed button
        gr.update(variant=web_variant, value=web_label),        # web_search button
        gr.update(variant=web_variant)                          # web_search_collapsed button
    )


def toggle_web_search(current_web_state, current_ddg_state):
    """Toggle Web Search (comprehensive mode). Disables DDG Search if enabling Web Search."""
    import gradio as gr
    
    new_web_state = not current_web_state
    
    # If enabling Web Search, disable DDG Search (mutual exclusivity)
    new_ddg_state = False if new_web_state else current_ddg_state
    
    web_variant = "primary" if new_web_state else "secondary"
    web_label = "🌐 Web Search ON" if new_web_state else "🌐 Web Search"
    
    ddg_variant = "primary" if new_ddg_state else "secondary"
    ddg_label = "🔍 DDG Search ON" if new_ddg_state else "🔍 DDG Search"
    
    return (
        new_web_state,                                          # web_search_enabled state
        new_ddg_state,                                          # ddg_search_enabled state
        gr.update(variant=web_variant, value=web_label),        # web_search button
        gr.update(variant=web_variant),                         # web_search_collapsed button
        gr.update(variant=ddg_variant, value=ddg_label),        # ddg_search button
        gr.update(variant=ddg_variant)                          # ddg_search_collapsed button
    )


# MAIN display LAUNCH


def launch_display():
    """Launch the Gradio display – supports Gradio 3.50.2 (Qt5 WebEngine) and newer versions."""
    global demo
    import tkinter as tk
    from tkinter import filedialog
    import os
    import gradio as gr
    from pathlib import Path
    from launcher import shutdown_program
    from scripts import configuration, utility, inference
    from scripts.configuration import (
        MODEL_NAME, SESSION_ACTIVE,
        MAX_HISTORY_SLOTS, MAX_ATTACH_SLOTS, SESSION_LOG_HEIGHT,
        MODEL_FOLDER, CONTEXT_SIZE, BATCH_SIZE, TEMPERATURE, REPEAT_PENALTY,
        VRAM_SIZE, SELECTED_GPU, SELECTED_CPU, MLOCK, BACKEND_TYPE,
        ALLOWED_EXTENSIONS, VRAM_OPTIONS, CTX_OPTIONS, BATCH_OPTIONS, TEMP_OPTIONS,
        REPEAT_OPTIONS, HISTORY_SLOT_OPTIONS, SESSION_LOG_HEIGHT_OPTIONS,
        ATTACH_SLOT_OPTIONS, HISTORY_DIR, USER_COLOR, THINK_COLOR, RESPONSE_COLOR,
        context_injector
    )
    # ── Determine Gradio major version ──────────────────────────────────────
    is_gradio_3 = cfg.GRADIO_VERSION.startswith('3.')

    # Output filtering
    initialize_filter_from_config()

    # Output filtering
    initialize_filter_from_config()

    # Determine Gradio major version for CSS parameter placement
    try:
        gradio_ver_parts = gr.__version__.split('.')
        is_gradio_6_plus = int(gradio_ver_parts[0]) >= 6
    except:
        is_gradio_6_plus = False  # Safe fallback for older versions or unusual version strings	
	
    # Common CSS rules (used by all versions)
    css_common = """
    .scrollable { overflow-y: auto }
    .half-width { width: 80px !important }
    .double-height { height: 80px !important }
    .clean-elements { gap: 4px !important; margin-bottom: 4px !important }
    .clean-elements-normbot { gap: 4px !important; margin-bottom: 20px !important }
    .send-button-green { background-color: green !important; color: white !important }
    .send-button-orange { background-color: orange !important; color: white !important }
    .send-button-red { background-color: red !important; color: white !important }
    .scrollable .message { white-space: pre-wrap; word-break: break-word; }
    .hide-label { display:none !important; }
    .message { line-height: 1.4 !important; }
    .progress-indicator { 
        font-family: monospace; 
        font-size: 14px; 
        padding: 16px 10px;
        background: #1a1a1a; 
        border-radius: 4px;
        min-height: 100px;
        line-height: 1.6;
    }
    .model-folder-row { gap: 8px !important; }
    
    /* NEW: Info box that matches textbox background color */
    .info-textbox-match {
        background-color: var(--input-background-fill) !important;
        border: 1px solid var(--border-color-primary) !important;
        border-radius: var(--radius-lg) !important;
        padding: 12px 16px !important;
        color: var(--body-text-color) !important;
        font-family: sans-serif !important;
        line-height: 1.6 !important;
    }
    .info-textbox-match a {
        color: var(--link-text-color, #4aa8ff) !important;
        text-decoration: none !important;
        font-weight: bold;
    }
    .info-textbox-match a:hover {
        text-decoration: underline !important;
    }
    """

    # Aggressive spacing fixes — mostly needed for Gradio 3 + old Qt WebEngine
    css_gradio3_fixes = """
    /* Reduce extra blank lines and tight spacing issues in Qt WebEngine */
    .message p { margin-top: 0.3em !important; margin-bottom: 0.3em !important; }
    .message br + br { display: none !important; }
    """

    # Final CSS: common + version-specific fixes
    final_css = css_common
    if is_gradio_3:
        final_css += css_gradio3_fixes

    # Prepare Blocks kwargs - CSS stays in Blocks for Gradio 5.x, moves to launch() for 6+
    blocks_kwargs = {
        "title": "Chat-Gradio-Gguf"
    }
    if not is_gradio_6_plus:
        blocks_kwargs["css"] = final_css.strip()

    # ── Main display ──────────────────────────────────────────────────────
    with gr.Blocks(**blocks_kwargs) as demo:
        cfg.demo = demo
        model_folder_state = gr.State(cfg.MODEL_FOLDER)

        # States - note session_messages for internal dict format
        states = dict(
            attached_files=gr.State([]),
            session_messages=gr.State([]),
            models_loaded=gr.State(False),
            llm=gr.State(None),
            cancel_flag=gr.State(False),
            interaction_phase=gr.State("waiting_for_input"),
            is_reasoning_model=gr.State(False),
            selected_panel=gr.State("History"),
            left_expanded_state=gr.State(True),
            right_expanded_state=gr.State(True),
            model_settings=gr.State({}),
            ddg_search_enabled=gr.State(False),
            web_search_enabled=gr.State(False),
            has_ai_response=gr.State(False),  # ← MUST BE INITIALIZED HERE
            tts_enabled=gr.State(False)
        )
        
        config_components = {}
        custom_components = {}
        conversation_components = {}
        buttons = {}
        action_buttons = {}

        def handle_backend_mode_change(mode):
            if mode == "SRAM_ONLY":
                return gr.update(visible=False)
            else:
                return gr.update(visible=True)

        with gr.Tabs():
            with gr.Tab("Main Conversations/Interactions"):
                with gr.Row():
                    # LEFT PANEL
                    with gr.Column(visible=True, min_width=300, elem_classes=["clean-elements"]) as left_column_expanded:
                        toggle_button_left_expanded = gr.Button(">-------<", variant="secondary")
                        gr.Markdown("**Dynamic Panel**")
                        panel_toggle = gr.Radio(choices=["History", "Attachments"], label="", value="History")
                        with gr.Group(visible=False) as attach_group:
                            attach_files = gr.UploadButton(
                                "Add Attach Files..", 
                                file_types=[f".{ext}" for ext in cfg.ALLOWED_EXTENSIONS], 
                                file_count="multiple", 
                                variant="secondary"
                            )
                            attach_slots = [gr.Button("Attach Slot Free", variant="huggingface", visible=False) 
                                          for _ in range(cfg.MAX_POSSIBLE_ATTACH_SLOTS)]
                        with gr.Group(visible=True) as history_slots_group:
                            start_new_session_btn = gr.Button("Start New Session..", variant="secondary")
                            buttons["session"] = [gr.Button(f"History Slot {i+1}", variant="huggingface", visible=False) 
                                               for i in range(cfg.MAX_POSSIBLE_HISTORY_SLOTS)]

                    with gr.Column(visible=False, min_width=60, elem_classes=["clean-elements"]) as left_column_collapsed:
                        toggle_button_left_collapsed = gr.Button("<->", variant="secondary")
                        new_session_btn_collapsed = gr.Button("New", variant="secondary")
                        add_attach_files_collapsed = gr.UploadButton(
                            "Add",
                            file_types=[f".{ext}" for ext in cfg.ALLOWED_EXTENSIONS],
                            file_count="multiple",
                            variant="secondary"
                        )

                    # CENTER - Main chat area
                    with gr.Column(scale=30, elem_classes=["clean-elements"]):
                        # Chatbot - same definition for all Gradio versions
                        conversation_components["session_log"] = gr.Chatbot(
                            label="Session Log", 
                            height=cfg.SESSION_LOG_HEIGHT,
                            elem_classes=["scrollable"]
                        )
                        
                        initial_max_lines = max(3, int(((cfg.SESSION_LOG_HEIGHT - 100) / 10) / 2.5) - 6)
                        cfg.USER_INPUT_MAX_LINES = initial_max_lines
                        
                        # User input initially disabled until valid model selected
                        conversation_components["user_input"] = gr.Textbox(
                            label="User Input", 
                            lines=3, 
                            max_lines=initial_max_lines, 
                            interactive=True,                           # ← always allow typing
                            placeholder="Type your message here... (model auto-loads on first send)"
                        )
                        
                        conversation_components["progress_indicator"] = gr.Markdown(
                            value="",
                            visible=False,
                            elem_classes=["progress-indicator"]
                        )

                        with gr.Row(elem_classes=["clean-elements"]):
                            action_buttons["action"] = gr.Button("Send Input", variant="secondary", elem_classes=["send-button-green"], scale=1)
                            action_buttons["edit_previous"] = gr.Button("Edit Previous", variant="secondary", scale=1, visible=False)
                            action_buttons["copy_response"] = gr.Button("Copy Output", variant="secondary", scale=1, visible=False)
                            action_buttons["cancel_input"] = gr.Button("", variant="primary", scale=1, visible=False)  # Hidden placeholder for output compatibility
                            action_buttons["cancel_response"] = gr.Button("..Wait For Response..", variant="primary", scale=1, visible=False)

                    # RIGHT PANEL
                    with gr.Column(visible=True, min_width=300, elem_classes=["clean-elements"]) as right_column_expanded:
                        toggle_button_right_expanded = gr.Button(">-------<", variant="secondary")
                        gr.Markdown("**Tools / Options**")
                        
                        with gr.Row(elem_classes=["clean-elements"]):
                            # Then the rest of your tool buttons
                            action_buttons["ddg_search"] = gr.Button("🔍 DDG Search", variant="secondary", scale=1)
                            action_buttons["web_search"] = gr.Button("🌐 Web Search", variant="secondary", scale=1)
                            action_buttons["tts_sound"] = gr.Button("🔊 TTS Sound", variant="secondary", scale=1)

                    with gr.Column(visible=False, min_width=60, elem_classes=["clean-elements"]) as right_column_collapsed:
                        toggle_button_right_collapsed = gr.Button("<->", variant="secondary")
                        action_buttons["ddg_search_collapsed"] = gr.Button("🔍", variant="secondary")
                        action_buttons["web_search_collapsed"] = gr.Button("🌐", variant="secondary")
                        action_buttons["tts_sound_collapsed"] = gr.Button("🔊", variant="secondary")

                with gr.Row():
                    interaction_global_status = gr.Textbox(
                        value="Ready",
                        label="Status",
                        interactive=False,
                        max_lines=1,
                        scale=20
                    )
                    exit_interaction = gr.Button("Exit Program", variant="stop", elem_classes=["double-height"], scale=1)

            with gr.Tab("Hardware/TTS/Model Configs"):
                # ── Hardware Configuration ──────────────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Hardware Configuration")
                    
                    # Row 1: Backend display + Layer Allocation
                    with gr.Row():
                        backend_display = gr.Textbox(
                            label="Backend Type",
                            value=cfg.BACKEND_TYPE,
                            interactive=False
                        )
                        layer_allocation_radio = gr.Radio(
                            choices=["SRAM_ONLY"] if not cfg.VULKAN_AVAILABLE else ["SRAM_ONLY", "VRAM_SRAM"],
                            label="Layer Allocation Mode",
                            value=cfg.LAYER_ALLOCATION_MODE,
                            interactive=cfg.VULKAN_AVAILABLE
                        )
                    
                    # Row 2: CPU Selection + Threads
                    with gr.Row():
                        cpu_select = gr.Dropdown(
                            choices=["Auto-Select"] + [c["label"] for c in get_cpu_info()],
                            label="CPU",
                            value=cfg.SELECTED_CPU,
                            interactive=True,
                            allow_custom_value=True
                        )
                        cpu_threads = gr.Slider(
                            minimum=1,
                            maximum=cfg.CPU_LOGICAL_CORES,
                            step=1,
                            value=cfg.CPU_THREADS or max(1, cfg.CPU_LOGICAL_CORES // 2),
                            label="CPU Threads",
                            interactive=True
                        )
                    
                    # Row 3: GPU + VRAM (only visible for Vulkan backends)
                    gpu_vram_row = gr.Row(visible=cfg.BACKEND_TYPE in ["VULKAN_CPU", "VULKAN_VULKAN"])
                    with gpu_vram_row:
                        gpu_select = gr.Dropdown(
                            choices=get_available_gpus(),
                            label="GPU",
                            value=cfg.SELECTED_GPU,
                            interactive=True,
                            allow_custom_value=True
                        )
                        vram_size = gr.Dropdown(
                            choices=cfg.VRAM_OPTIONS,
                            label="VRAM Allocation (MB)",
                            value=cfg.VRAM_SIZE,
                            interactive=True
                        )
                    
                    # Row 4: Sound Hardware - Unified default for both platforms
                    with gr.Row():
                        if cfg.PLATFORM == "windows":
                            system_label = "Windows Audio"
                        else:
                            # Show detected backend for Linux
                            backend = getattr(cfg, 'TTS_AUDIO_BACKEND', 'unknown')
                            system_label = f"Linux Audio ({backend})"

                        sound_output_display = gr.Textbox(
                            label=f"Audio Output ({system_label})",
                            value="Default Sound Device",
                            interactive=False,
                            max_lines=1
                        )

                        sound_sample_rate = gr.Dropdown(
                            choices=[str(r) for r in cfg.SOUND_SAMPLE_RATE_OPTIONS],
                            label="Sample Rate (Hz)",
                            value=str(cfg.SOUND_SAMPLE_RATE),
                            interactive=True
                        )


                # ── TTS Configuration ───────────────────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Text-to-Speech (TTS)")
                    with gr.Row():
                        tts_voice = gr.Dropdown(
                            choices=get_voice_choices(),
                            label="TTS Voice",
                            value=cfg.TTS_VOICE_NAME or "Default",
                            interactive=True,
                            allow_custom_value=True
                        )
                        tts_max_len = gr.Slider(
                            label="Max TTS Length (chars)",
                            minimum=500,
                            maximum=4500,
                            step=500,
                            value=cfg.MAX_TTS_LENGTH,
                            interactive=True
                        )
                
                # ── Model Configuration ─────────────────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Model Configuration")
                    
                    # Row 1: Model Folder + Model Selection + Loaded Indicator
                    with gr.Row(elem_classes=["model-folder-row"]):
                        model_folder_textbox = gr.Textbox(
                            label="Model Folder",
                            value=cfg.MODEL_FOLDER,
                            interactive=False,
                            placeholder="Click Browse Folder button to select...",
                            scale=5
                        )
                        
                        model_dropdown = gr.Dropdown(
                            choices=get_available_models(),
                            label="Model Selected",
                            value=cfg.MODEL_NAME,
                            interactive=True,
                            elem_classes="model-select",
                            scale=4
                        )
                        
                        model_loaded_indicator = gr.Textbox(
                            label="Model Loaded",
                            value="🟢 SO LOADED" if cfg.MODELS_LOADED else "🔴 NOT LOADED",
                            interactive=False,
                            max_lines=1,
                            scale=3
                        )
                    
                    # Row 3: Context + Batch
                    with gr.Row():
                        ctx_size = gr.Dropdown(
                            choices=cfg.CTX_OPTIONS,
                            label="Context Size",
                            value=cfg.CONTEXT_SIZE,
                            interactive=True
                        )
                        batch_size = gr.Dropdown(
                            choices=cfg.BATCH_OPTIONS,
                            label="Batch Size",
                            value=cfg.BATCH_SIZE,
                            interactive=True
                        )
                    
                        temperature = gr.Dropdown(
                            choices=cfg.TEMP_OPTIONS,
                            label="Temperature",
                            value=cfg.TEMPERATURE,
                            interactive=True
                        )
                        repeat_penalty = gr.Dropdown(
                            choices=cfg.REPEAT_OPTIONS,
                            label="Repeat Penalty",
                            value=cfg.REPEAT_PENALTY,
                            interactive=True
                        )

                    # Row 1a: Explicit Browse/1Load/Unload buttons
                    with gr.Row():
                        browse_folder_btn = gr.Button("📁 Browse Folder", scale=1, size="sm")
                        load_model_btn = gr.Button("📥 Load Model", variant="primary", scale=1)
                        unload_model_btn = gr.Button("📤 Unload Model", variant="stop", scale=1)
                

                # Save button + status + exit
                gr.Markdown("---")
                with gr.Row():
                    save_config_btn = gr.Button("Save All Configuration", variant="primary", size="lg")
                gr.Markdown("---")
                with gr.Row():
                    config_status = gr.Textbox(
                        value="Configuration loaded",
                        label="Status",
                        interactive=False,
                        max_lines=1,
                        scale=20
                    )
                    exit_config = gr.Button("Exit Program", variant="stop", elem_classes=["double-height"], scale=1)

            with gr.Tab("Program/Filter Settings"):
                        
                # ── Program Options ─────────────────────────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Program Options")
                    # Gradio 3 notice: Session Log Height requires restart
                    if cfg.GRADIO_VERSION and cfg.GRADIO_VERSION.startswith('3.'):
                        gr.Markdown("*\\*\\* Requires restart after configuring UI components on, Gradio v3.x.x and Qt5-Web, installs*")
                    # Row 1: Attach slots and Log height
                    with gr.Row():
                        session_log_height = gr.Dropdown(
                            choices=cfg.SESSION_LOG_HEIGHT_OPTIONS,
                            label="Session Log Height (px)",
                            value=cfg.SESSION_LOG_HEIGHT,
                            interactive=True
                        )
                        max_attach_slots = gr.Dropdown(
                            choices=cfg.ATTACH_SLOT_OPTIONS,
                            label="Max Attachment Slots",
                            value=cfg.MAX_ATTACH_SLOTS,
                            interactive=True
                        )
                        max_history_slots = gr.Dropdown(
                            choices=cfg.HISTORY_SLOT_OPTIONS,
                            label="Max History Slots",
                            value=cfg.MAX_HISTORY_SLOTS,
                            interactive=True
                        )
                        delete_history_btn = gr.Button("Delete All History", variant="stop", size="double")

                    gr.Markdown("### Output Options")
                    # Row 3: Checkboxes horizontal
                    with gr.Row():
                        show_think = gr.Checkbox(
                            label="Show Thinking Phase",
                            value=cfg.SHOW_THINK_PHASE,
                            interactive=True
                        )
                        bleep_events = gr.Checkbox(
                            label="Beep on Events",
                            value=cfg.BLEEP_ON_EVENTS,
                            interactive=True
                        )
                        print_raw = gr.Checkbox(
                            label="Print Raw Model Output (debug)",
                            value=cfg.PRINT_RAW_OUTPUT,
                            interactive=True
                        )
                
                # ── Filter Settings ─────────────────────────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Filter Settings")
                    with gr.Row():
                        filter_user_btn = gr.Button("User Preset")
                        filter_light_btn = gr.Button("Light Preset")
                        filter_full_btn = gr.Button("Full Preset")
                    filter_text = gr.Textbox(
                        label="Custom Filter Rules (find→replace pairs)",
                        value=get_filter_text_for_display(),
                        lines=15,
                        interactive=True,
                        placeholder="One rule per line: find_string → replace_string"
                    )
                
                # ── Save Button ─────────────────────────────────────────────────────────────
                gr.Markdown("---")

                with gr.Row():
                    save_all_btn = gr.Button("Save All Settings", variant="primary", size="lg")

                gr.Markdown("---")
                
                # ── FINAL ROW: Status + Exit (IDENTICAL TO CONVERSATION TAB) ────────────────
                with gr.Row():
                    filter_status = gr.Textbox(
                        value="Filter settings loaded",
                        label="Status",
                        interactive=False,
                        max_lines=1,
                        scale=20
                    )
                    exit_filtering = gr.Button("Exit Program", variant="stop", elem_classes=["double-height"], scale=1)

            with gr.Tab("About/Debug Info"):
                
                # ── Section A: Project Info ──────────────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Chat-Gradio-Gguf")
                    gr.HTML("""
                        <p>A Windows/Linux Chatbot using, Gradio and llama.cpp, by <a href="mailto:wiseman-timelord@mail.com">WiseMan-Time-Lord</a> at <a href="http://wisetime.rf.gd/">WiseTime.Rf.Gd</a></p>
                        
                        <p><strong>Where you may find, this and my other, programming projects on </strong> <a href="https://github.com/wiseman-timelord">GitHub</a></p>
                        
                        <p><strong>Support/Donate to assist in the continuation of my projects at, </strong> <a href="https://patreon.com/WiseManTimeLord">Patreon</a>, <a href="https://ko-fi.com/WiseManTimeLord">Ko-Fi</a></p>
                    """, elem_classes=["info-textbox-match"])
               
                # ── Section B: INI Constants ──────────────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### System Constants (from constants.ini)")
                    ini_display = gr.Textbox(
                        label="INI Values (read-only, set by installer)",
                        value=get_ini_display_text(),
                        lines=10,
                        interactive=False
                    )
                
                # ── Section C: Debug Globals ──────────────────────────────────────────────
                with gr.Group():
                    gr.Markdown("### Runtime Globals (Debug)")
                    debug_display = gr.Textbox(
                        label="Critical Globals (click Refresh to update)",
                        value=get_debug_globals_text(),
                        lines=10,
                        interactive=False
                    )
                    refresh_debug_btn = gr.Button("🔄 Refresh Debug Info", variant="secondary")
                
                gr.Markdown("---")
                
                # ── FINAL ROW: Status + Exit ─────────────────────────────────────────────
                with gr.Row():
                    info_status = gr.Textbox(
                        value="Info tab loaded",
                        label="Status",
                        interactive=False,
                        max_lines=1,
                        scale=20
                    )
                    exit_info = gr.Button("Exit Program", variant="stop", elem_classes=["double-height"], scale=1)

        
        # EVENT HANDLERS
        

        # Fix layer allocation → VRAM visibility
        layer_allocation_radio.change(
            fn=lambda mode: gr.update(visible=(mode == "VRAM_SRAM" and cfg.BACKEND_TYPE in ["VULKAN_CPU", "VULKAN_VULKAN"])),
            inputs=[layer_allocation_radio],
            outputs=[vram_size]
        )

        # Backend type change → GPU/VRAM row visibility
        def update_gpu_vram_visibility(backend_type):
            visible = backend_type in ["VULKAN_CPU", "VULKAN_VULKAN"]
            return gr.update(visible=visible), gr.update(visible=visible)

        # ── Browse button handler (works on ALL Gradio versions) ────────────────────
        def browse_and_update_folder(current_path):
            """Open folder dialog, update path, refresh model list"""
            import tkinter as tk
            from tkinter import filedialog
            from pathlib import Path
            
            root = tk.Tk()
            root.withdraw()
            
            initial_dir = current_path if Path(current_path).is_dir() else str(Path.home())
            
            folder_path = filedialog.askdirectory(
                title="Select Folder Containing GGUF Models",
                initialdir=initial_dir,
                mustexist=True
            )
            
            root.destroy()
            
            if not folder_path or folder_path == current_path:
                return (
                    current_path,
                    gr.update(),
                    gr.update(),
                    "Folder selection unchanged.",
                    "Folder selection unchanged.",
                    "Folder selection unchanged."
                )
            
            cfg.MODEL_FOLDER = str(Path(folder_path).resolve())
            
            available_models = get_available_models()
            
            # Build choices list consistent with update_model_list pattern
            choices = ["Select_a_model..."]
            if available_models and available_models != ["Select_a_model..."]:
                choices.extend(available_models)
                # Auto-select first real model found
                new_value = available_models[0]
                cfg.MODEL_NAME = new_value
            else:
                choices = ["No models found"]
                new_value = "No models found"
            
            status = f"Folder updated: {short_path(folder_path)}\nFound {len(available_models)} models."
            
            return (
                cfg.MODEL_FOLDER,
                gr.update(choices=choices, value=new_value),
                gr.update(value=new_value),
                status,
                status,
                status
            )

        # Browse folder button handler (reuse existing function)
        browse_folder_btn.click(
            fn=browse_and_update_folder,
            inputs=[model_folder_textbox],
            outputs=[
                model_folder_textbox,
                model_dropdown,
                model_dropdown,
                interaction_global_status,
                config_status,
                filter_status
            ]
        )

        # Load model button handler
        load_model_btn.click(
            fn=handle_load_model,
            inputs=[
                model_dropdown,
                model_folder_textbox,
                vram_size,
                ctx_size,
                gpu_select,
                cpu_select,
                cpu_threads,
                states["llm"],
                states["models_loaded"]
            ],
            outputs=[
                states["llm"],
                states["models_loaded"],
                interaction_global_status,
                config_status,
                filter_status,
                conversation_components["user_input"],
                model_loaded_indicator
            ]
        )

        # Unload model button handler
        unload_model_btn.click(
            fn=handle_unload_model,
            inputs=[
                states["llm"],
                states["models_loaded"]
            ],
            outputs=[
                states["llm"],
                states["models_loaded"],
                interaction_global_status,
                config_status,
                filter_status,
                conversation_components["user_input"],
                model_loaded_indicator
            ]
        )
        
        # MAIN CONVERSATION HANDLER - Critical fix for Send Input button
        action_buttons["action"].click(
            fn=conversation_display,
            inputs=[
                conversation_components["user_input"],      # user_input text
                conversation_components["session_log"],     # session_tuples (chatbot display state)
                states["session_messages"],                 # session_messages internal state
                states["attached_files"],                   # loaded_files
                states["is_reasoning_model"],               # is_reasoning_model flag
                states["cancel_flag"],                      # cancel_flag state
                states["ddg_search_enabled"],               # ddg_search_enabled state
                states["web_search_enabled"],               # web_search_enabled state
                states["interaction_phase"],                # interaction_phase state
                states["llm"],                              # llm state object
                states["models_loaded"],                    # models_loaded state flag
                states["has_ai_response"],                   # has_ai_response state flag
                states["tts_enabled"]       # Added TTS state
            ],
            outputs=[
                conversation_components["session_log"],     # Updated chat display
                states["session_messages"],                 # Updated internal messages state
                interaction_global_status,                  # Status text updates
                conversation_components["user_input"],      # Hide/show user input box
                conversation_components["progress_indicator"], # Progress flow diagram
                action_buttons["action"],                   # Main action button state
                action_buttons["edit_previous"],            # Edit button visibility
                action_buttons["copy_response"],            # Copy button visibility
                action_buttons["cancel_input"],             # Cancel input placeholder
                action_buttons["cancel_response"],          # Wait indicator button
                states["cancel_flag"],                      # Updated cancel flag
                states["attached_files"],                   # Updated attached files list
                states["interaction_phase"],                # Updated phase state
                states["llm"],                              # Updated LLM state
                states["models_loaded"],                    # Updated models_loaded state
                states["tts_enabled"]                       # TTS state (matches final gr.update yield)
            ]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        ).then(
            fn=lambda files: update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        )
        
        # ── Model dropdown change: unload old → countdown → load new ──────────────
        def handle_model_change(new_model, current_model, llm_state, models_loaded_state, model_folder):
            """Handle model selection change: unload → 3s countdown → load new model"""
            import time
            from scripts.inference import unload_inference, load_models
            from scripts.utility import beep, short_path

            # No change or invalid → skip
            if new_model == current_model or new_model in ["Select_a_model...", "No models found", "", None]:
                placeholder = (
                    "Enter text here... (model auto-loads on first send)"
                    if new_model not in ["Select_a_model...", "No models found", "", None]
                    else "Select a valid model first..."
                )
                return (
                    llm_state,
                    models_loaded_state,
                    "Model selection unchanged or invalid.",
                    gr.update(
                        interactive=bool(new_model and new_model not in ["Select_a_model...", "No models found"]),
                        placeholder=placeholder
                    )
                )

            status_msg = f"Changing model to: {new_model}"

            # Step 1: Unload if loaded
            if models_loaded_state and llm_state is not None:
                yield (
                    llm_state,
                    models_loaded_state,
                    status_msg + "\nUnloading current model...",
                    gr.update(interactive=False)
                )
                unload_status, new_llm, new_loaded = unload_models(llm_state, models_loaded_state)
                llm_state = new_llm
                models_loaded_state = new_loaded
                cfg.MODELS_LOADED = False
                cfg.llm = None
                status_msg += f"\n{unload_status}"

            # Step 2: 3-second countdown
            for i in range(3, 0, -1):
                yield (
                    llm_state,
                    models_loaded_state,
                    f"{status_msg}\nLoading new model in {i}...",
                    gr.update(interactive=False)
                )
                time.sleep(1.0)

            # Step 3: Load new model
            yield (
                llm_state,
                models_loaded_state,
                status_msg + "\nLoading model...",
                gr.update(interactive=False)
            )

            try:
                load_status, loaded, new_llm, _ = load_models(
                    model_folder,
                    new_model,
                    cfg.VRAM_SIZE,
                    llm_state,
                    models_loaded_state
                )

                if loaded:
                    cfg.MODELS_LOADED = True
                    cfg.llm = new_llm
                    cfg.LOADED_CONTEXT_SIZE = cfg.CONTEXT_SIZE
                    cfg.LAST_INTERACTION_TIME = time.time()

                    # Yield success status but CONTINUE processing input (critical fix)
                    yield (
                        get_chatbot_output(session_tuples, session_messages),
                        session_messages,
                        f"✅ Model loaded: {cfg.MODEL_NAME}",
                        gr.update(visible=False),
                        gr.update(visible=True, value="Model ready — processing your message..."),
                        gr.update(interactive=True),  # re-enable send button
                        gr.update(visible=False),     # hide wait indicator
                        cancel_flag, 
                        loaded_files, 
                        "input_submitted",            # FIXED: Continue normal flow phases
                        new_llm, 
                        gr.update(value=True), 
                        gr.update()
                    )

                    # Update local state to use the newly loaded model
                    llm_state = new_llm
                    models_loaded_state = True
                    # FALL THROUGH to continue processing the user input through normal phases
                    
                else:
                    # Load failed - yield error and exit generator
                    yield (
                        get_chatbot_output(session_tuples, session_messages),
                        session_messages,
                        f"❌ Model load failed: {status}",
                        gr.update(visible=True),
                        gr.update(visible=False),
                        *update_action_buttons("waiting_for_input", has_ai_response_state),
                        cancel_flag, loaded_files, "waiting_for_input",
                        llm_state, gr.update(value=False), gr.update()  # ✅ 13th output added
                    )
                    return  # Exit generator on failure

            except Exception as e:
                status_msg += f"\nError: {str(e)[:120]}"
                yield (
                    llm_state,
                    False,
                    status_msg,
                    gr.update(interactive=False, placeholder="Select a valid model first...")
                )

        # Connect model dropdown change (ONLY ONE connection)
        model_dropdown.change(
            fn=handle_model_change,
            inputs=[
                model_dropdown,
                model_dropdown,
                states["llm"],
                states["models_loaded"],
                model_folder_textbox
            ],
            outputs=[
                states["llm"],
                states["models_loaded"],
                interaction_global_status,
                conversation_components["user_input"]
            ]
        )

        # Sound hardware handlers
        sound_sample_rate.change(
            fn=update_sound_sample_rate,
            inputs=[sound_sample_rate],
            outputs=[config_status]
        )

        # TTS voice handler
        tts_voice.change(
            fn=update_tts_voice,
            inputs=[tts_voice],
            outputs=[config_status]
        )

        # Max TTS length handler
        tts_max_len.change(
            fn=lambda val: setattr(configuration, 'MAX_TTS_LENGTH', int(val)) or f"Max TTS length set to {int(val)} chars",
            inputs=[tts_max_len],
            outputs=[config_status]
        )
        show_think.change(
            fn=lambda x: setattr(configuration, 'SHOW_THINK_PHASE', x) or f"Show thinking phase: {'ON' if x else 'OFF'}",
            inputs=[show_think],
            outputs=[interaction_global_status]
        )

        bleep_events.change(
            fn=lambda x: setattr(configuration, 'BLEEP_ON_EVENTS', x) or f"Beep on events: {'ON' if x else 'OFF'}",
            inputs=[bleep_events],
            outputs=[interaction_global_status]
        )

        print_raw.change(
            fn=lambda x: setattr(configuration, 'PRINT_RAW_OUTPUT', x) or f"Print raw output: {'ON' if x else 'OFF'}",
            inputs=[print_raw],
            outputs=[interaction_global_status]
        )

        session_log_height.change(
            fn=lambda val: (
                setattr(configuration, 'SESSION_LOG_HEIGHT', int(val)),
                f"Session log height set to {int(val)}px (save & restart to apply)"
            )[1],
            inputs=[session_log_height],
            outputs=[interaction_global_status]
        )

        max_attach_slots.change(
            fn=lambda val: (
                setattr(configuration, 'MAX_ATTACH_SLOTS', int(val)),
                f"Max attachment slots set to {int(val)} (save & restart to apply)"
            )[1],
            inputs=[max_attach_slots],
            outputs=[interaction_global_status]
        )

        max_history_slots.change(
            fn=lambda val: (
                setattr(configuration, 'MAX_HISTORY_SLOTS', int(val)),
                f"Max history slots set to {int(val)} (save & restart to apply)"
            )[1],
            inputs=[max_history_slots],
            outputs=[interaction_global_status]
        )

        # Exit buttons - ensure models unloaded early in shutdown
        exit_interaction.click(
            fn=shutdown_program,
            inputs=[states["llm"], states["models_loaded"],
                    states["session_messages"], states["attached_files"]],
            outputs=[]
        )
        
        exit_config.click(
            fn=shutdown_program,
            inputs=[states["llm"], states["models_loaded"],
                    states["session_messages"], states["attached_files"]],
            outputs=[]
        )
        
        exit_filtering.click(
            fn=shutdown_program,
            inputs=[states["llm"], states["models_loaded"],
                    states["session_messages"], states["attached_files"]],
            outputs=[]
        )
        
        exit_info.click(
            fn=shutdown_program,
            inputs=[states["llm"], states["models_loaded"],
                    states["session_messages"], states["attached_files"]],
            outputs=[]
        )

        # Refresh debug info button handler
        def refresh_debug_info():
            return get_ini_display_text(), get_debug_globals_text(), "Debug info refreshed"

        refresh_debug_btn.click(
            fn=refresh_debug_info,
            inputs=[],
            outputs=[ini_display, debug_display, info_status]
        )

        # Panel toggles
        panel_toggle.change(
            fn=update_panel_on_mode_change,
            inputs=[panel_toggle],
            outputs=[panel_toggle, attach_group, history_slots_group, states["selected_panel"]]
        )

        # Left panel expand/collapse
        def toggle_left_panel(current_state):
            new_state = not current_state
            return new_state, gr.update(visible=new_state), gr.update(visible=not new_state)
        
        toggle_button_left_expanded.click(
            fn=toggle_left_panel,
            inputs=[states["left_expanded_state"]],
            outputs=[states["left_expanded_state"], left_column_expanded, left_column_collapsed]
        )
        
        toggle_button_left_collapsed.click(
            fn=toggle_left_panel,
            inputs=[states["left_expanded_state"]],
            outputs=[states["left_expanded_state"], left_column_expanded, left_column_collapsed]
        )

        # Right panel expand/collapse
        def toggle_right_panel(current_state):
            new_state = not current_state
            return new_state, gr.update(visible=new_state), gr.update(visible=not new_state)
        
        toggle_button_right_expanded.click(
            fn=toggle_right_panel,
            inputs=[states["right_expanded_state"]],
            outputs=[states["right_expanded_state"], right_column_expanded, right_column_collapsed]
        )
        
        toggle_button_right_collapsed.click(
            fn=toggle_right_panel,
            inputs=[states["right_expanded_state"]],
            outputs=[states["right_expanded_state"], right_column_expanded, right_column_collapsed]
        )

        # DDG Search toggle - mutually exclusive with Web Search
        action_buttons["ddg_search"].click(
            fn=toggle_ddg_search,
            inputs=[states["ddg_search_enabled"], states["web_search_enabled"]],
            outputs=[
                states["ddg_search_enabled"],
                states["web_search_enabled"],
                action_buttons["ddg_search"],
                action_buttons["ddg_search_collapsed"],
                action_buttons["web_search"],
                action_buttons["web_search_collapsed"]
            ]
        )
        
        action_buttons["ddg_search_collapsed"].click(
            fn=toggle_ddg_search,
            inputs=[states["ddg_search_enabled"], states["web_search_enabled"]],
            outputs=[
                states["ddg_search_enabled"],
                states["web_search_enabled"],
                action_buttons["ddg_search"],
                action_buttons["ddg_search_collapsed"],
                action_buttons["web_search"],
                action_buttons["web_search_collapsed"]
            ]
        )

        # Web Search toggle - mutually exclusive with DDG Search
        action_buttons["web_search"].click(
            fn=toggle_web_search,
            inputs=[states["web_search_enabled"], states["ddg_search_enabled"]],
            outputs=[
                states["web_search_enabled"],
                states["ddg_search_enabled"],
                action_buttons["web_search"],
                action_buttons["web_search_collapsed"],
                action_buttons["ddg_search"],
                action_buttons["ddg_search_collapsed"]
            ]
        )
        
        action_buttons["web_search_collapsed"].click(
            fn=toggle_web_search,
            inputs=[states["web_search_enabled"], states["ddg_search_enabled"]],
            outputs=[
                states["web_search_enabled"],
                states["ddg_search_enabled"],
                action_buttons["web_search"],
                action_buttons["web_search_collapsed"],
                action_buttons["ddg_search"],
                action_buttons["ddg_search_collapsed"]
            ]
        )

        # TTS Sound button handlers
        action_buttons["tts_sound"].click(
            fn=toggle_tts_sound,
            inputs=[states["tts_enabled"]],
            outputs=[
                states["tts_enabled"],
                action_buttons["tts_sound"],
                action_buttons["tts_sound_collapsed"],
                interaction_global_status
            ]
        )
        
        action_buttons["tts_sound_collapsed"].click(
            fn=toggle_tts_sound,
            inputs=[states["tts_enabled"]],
            outputs=[
                states["tts_enabled"],
                action_buttons["tts_sound"],
                action_buttons["tts_sound_collapsed"],
                interaction_global_status
            ]
        )

        # Normal (visible) Start New Session button
        start_new_session_btn.click(
            fn=start_new_session,
            inputs=[
                states["session_messages"],
                states["attached_files"],
                states["llm"],
                states["models_loaded"]
            ],
            outputs=[
                conversation_components["session_log"],
                states["session_messages"],
                states["attached_files"],
                interaction_global_status,
                states["has_ai_response"],
                action_buttons["action"],
                action_buttons["edit_previous"],
                action_buttons["copy_response"],
                action_buttons["cancel_input"],
                action_buttons["cancel_response"],
                states["llm"],
                states["models_loaded"]
            ]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        ).then(
            fn=lambda: update_file_slot_ui([], True),
            inputs=[],
            outputs=attach_slots + [attach_files]
        )

        # Collapsed (sidebar) Start New Session button - same logic
        new_session_btn_collapsed.click(
            fn=start_new_session,
            inputs=[
                states["session_messages"],
                states["attached_files"],
                states["llm"],
                states["models_loaded"]
            ],
            outputs=[
                conversation_components["session_log"],
                states["session_messages"],
                states["attached_files"],
                interaction_global_status,
                states["has_ai_response"],
                action_buttons["action"],
                action_buttons["edit_previous"],
                action_buttons["copy_response"],
                action_buttons["cancel_input"],
                action_buttons["cancel_response"],
                states["llm"],
                states["models_loaded"]
            ]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        # ── Edit Previous Message Button Handler ───────────────────────────────────
        action_buttons["edit_previous"].click(
            fn=edit_previous_prompt,
            inputs=[
                conversation_components["session_log"],
                states["session_messages"]
            ],
            outputs=[
                conversation_components["session_log"],
                states["session_messages"],
                conversation_components["user_input"],  # ← Critical: populates input box with previous message
                interaction_global_status,
                states["has_ai_response"]
            ]
        ).then(
            fn=lambda has_ai: update_action_buttons("waiting_for_input", has_ai),
            inputs=[states["has_ai_response"]],
            outputs=[
                action_buttons["action"],
                action_buttons["edit_previous"],
                action_buttons["copy_response"],
                action_buttons["cancel_input"],
                action_buttons["cancel_response"]
            ]
        )

        # ── Copy Last Response Button Handler ───────────────────────────────────────
        action_buttons["copy_response"].click(
            fn=copy_last_response,
            inputs=[states["session_messages"]],
            outputs=[interaction_global_status]
        )

        # Save Settings button
        def unified_save_wrapper(
            # Hardware
            layer_mode, cpu, cpu_threads_val, gpu, vram, sound_device, sample_rate,
            # Model
            model_folder_val, model, ctx, batch, temp, repeat,
            # TTS
            tts_voice_val, tts_max_len_val,
            # Settings tab
            show_think_val, bleep_val, print_raw_val,
            max_hist, max_att, log_height,
            # Filter
            filter_text_val
        ):
            # Hardware - use existing cfg values if parameters are None
            cfg.LAYER_ALLOCATION_MODE = layer_mode if layer_mode is not None else cfg.LAYER_ALLOCATION_MODE
            cfg.SELECTED_CPU = cpu if cpu is not None else cfg.SELECTED_CPU
            cfg.CPU_THREADS = int(cpu_threads_val) if cpu_threads_val is not None else cfg.CPU_THREADS
            cfg.SELECTED_GPU = gpu if gpu is not None else cfg.SELECTED_GPU
            cfg.VRAM_SIZE = int(vram) if vram is not None else cfg.VRAM_SIZE
            cfg.SOUND_OUTPUT_DEVICE = sound_device if sound_device is not None else cfg.SOUND_OUTPUT_DEVICE
            cfg.SOUND_SAMPLE_RATE = int(sample_rate) if sample_rate is not None else cfg.SOUND_SAMPLE_RATE
            
            # Model
            cfg.MODEL_FOLDER = model_folder_val if model_folder_val is not None else cfg.MODEL_FOLDER
            cfg.MODEL_NAME = model if model is not None else cfg.MODEL_NAME
            cfg.CONTEXT_SIZE = int(ctx) if ctx is not None else cfg.CONTEXT_SIZE
            cfg.BATCH_SIZE = int(batch) if batch is not None else cfg.BATCH_SIZE
            cfg.TEMPERATURE = float(temp) if temp is not None else cfg.TEMPERATURE
            cfg.REPEAT_PENALTY = float(repeat) if repeat is not None else cfg.REPEAT_PENALTY
            
            # TTS
            cfg.TTS_VOICE_NAME = tts_voice_val if tts_voice_val is not None else cfg.TTS_VOICE_NAME
            if tts_voice_val and tts_voice_val != "Default":
                cfg.TTS_VOICE = get_voice_id_by_name(tts_voice_val)
            cfg.MAX_TTS_LENGTH = int(tts_max_len_val) if tts_max_len_val is not None else cfg.MAX_TTS_LENGTH
            
            # Settings
            cfg.SHOW_THINK_PHASE = bool(show_think_val) if show_think_val is not None else cfg.SHOW_THINK_PHASE
            cfg.BLEEP_ON_EVENTS = bool(bleep_val) if bleep_val is not None else cfg.BLEEP_ON_EVENTS
            cfg.PRINT_RAW_OUTPUT = bool(print_raw_val) if print_raw_val is not None else cfg.PRINT_RAW_OUTPUT
            cfg.MAX_HISTORY_SLOTS = int(max_hist) if max_hist is not None else cfg.MAX_HISTORY_SLOTS
            cfg.MAX_ATTACH_SLOTS = int(max_att) if max_att is not None else cfg.MAX_ATTACH_SLOTS
            cfg.SESSION_LOG_HEIGHT = int(log_height) if log_height is not None else cfg.SESSION_LOG_HEIGHT
            
            # Save everything
            result = save_all_settings()
            
            # Save custom filter if present
            if filter_text_val and filter_text_val.strip():
                filter_result = save_custom_filter(filter_text_val)
                result += f"\n{filter_result}"
            
            return result, result, result, result

        # Connect both save buttons to unified handler
        save_config_btn.click(
            fn=unified_save_wrapper,
            inputs=[
                layer_allocation_radio, cpu_select, cpu_threads, gpu_select, vram_size,
                sound_output_display, sound_sample_rate,
                model_folder_textbox, model_dropdown,
                ctx_size, batch_size, temperature, repeat_penalty,
                tts_voice, tts_max_len,
                show_think, bleep_events, print_raw,
                max_history_slots, max_attach_slots, session_log_height,
                filter_text
            ],
            outputs=[interaction_global_status, config_status, filter_status, info_status]
        )

        save_all_btn.click(
            fn=unified_save_wrapper,
            inputs=[
                layer_allocation_radio, cpu_select, cpu_threads, gpu_select, vram_size,
                sound_output_display, sound_sample_rate,
                model_folder_textbox, model_dropdown,
                ctx_size, batch_size, temperature, repeat_penalty,
                tts_voice, tts_max_len,
                show_think, bleep_events, print_raw,
                max_history_slots, max_attach_slots, session_log_height,
                filter_text
            ],
            outputs=[interaction_global_status, config_status, filter_status, info_status]
        )

        # Attach files handler (expanded view)
        attach_files.upload(
            fn=utility.process_attach_files,
            inputs=[attach_files, states["attached_files"]],
            outputs=[interaction_global_status, states["attached_files"]]
        ).then(
            fn=lambda files: utility.update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        )

        # Attach files handler (collapsed view) - shares same state
        add_attach_files_collapsed.upload(
            fn=utility.process_attach_files,
            inputs=[add_attach_files_collapsed, states["attached_files"]],
            outputs=[interaction_global_status, states["attached_files"]]
        ).then(
            fn=lambda files: utility.update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        )

        # Eject file handlers for each attach slot
        for i, btn in enumerate(attach_slots):
            btn.click(
                fn=lambda files, idx=i: utility.eject_file(files, idx, True),
                inputs=[states["attached_files"]],
                outputs=[states["attached_files"], interaction_global_status]
            ).then(
                fn=lambda files: utility.update_file_slot_ui(files, True),
                inputs=[states["attached_files"]],
                outputs=attach_slots + [attach_files]
            )

        # Filtering tab event handlers
        def load_user_filter():
            text, status = load_filter_preset("User")
            return text, status

        def load_light_filter():
            text, status = load_filter_preset("Light")
            return text, status

        def load_full_filter():
            text, status = load_filter_preset("Full")
            return text, status

        filter_user_btn.click(
            fn=load_user_filter,
            inputs=[],
            outputs=[filter_text, filter_status]
        )
        
        filter_light_btn.click(
            fn=load_light_filter,
            inputs=[],
            outputs=[filter_text, filter_status]
        )
        
        filter_full_btn.click(
            fn=load_full_filter,
            inputs=[],
            outputs=[filter_text, filter_status]
        )

        # Delete all history button handler
        delete_history_btn.click(
            fn=delete_all_sessions,
            inputs=[],
            outputs=[filter_status]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        # Initial load - set up model list and session buttons
        demo.load(
            fn=lambda: cfg.MODEL_FOLDER,
            inputs=[],
            outputs=[model_folder_state]
        ).then(
            fn=update_model_list,
            inputs=[model_folder_state],
            outputs=[model_dropdown]
        ).then(
            fn=update_cpu_select,               # ← add this
            inputs=[],
            outputs=[cpu_select]                # ← add this (your cpu_select component)
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        ).then(
            fn=lambda model_name: gr.update(
                interactive=(model_name not in ["Select_a_model...", "No models found", None, ""]),
                placeholder="Enter text here..." if model_name not in ["Select_a_model...", "No models found", None, ""] else "Select a valid model first..."
            ),
            inputs=[model_dropdown],
            outputs=[conversation_components["user_input"]]
        )

        # Attach click handlers to session history buttons
        for i, btn in enumerate(buttons["session"]):
            btn.click(
                fn=lambda idx=i: load_session_by_index(idx),
                inputs=[],
                outputs=[
                    conversation_components["session_log"],
                    states["session_messages"],
                    states["attached_files"],
                    interaction_global_status,
                    states["has_ai_response"],
                    action_buttons["action"],
                    action_buttons["edit_previous"],
                    action_buttons["copy_response"],
                    action_buttons["cancel_input"],
                    action_buttons["cancel_response"]
                ]
            ).then(
                fn=lambda files: update_file_slot_ui(files, True),
                inputs=[states["attached_files"]],
                outputs=attach_slots + [attach_files]
            )

    # Launch with browser
    import threading
    from scripts.browser import launch_custom_browser, wait_for_gradio

    print("[BROWSER] Starting Gradio server in background...")

    # Enable queue for generator/streaming support
    demo.queue()

    # Build launch kwargs - CSS moved to launch() for Gradio 6+ compatibility
    launch_kwargs = {
        "server_name": "localhost",
        "server_port": 7860,
        "show_error": True,
        "share": False,
        "inbrowser": False,
        "prevent_thread_lock": True
    }
    
    # Gradio 6.0+ moved css parameter to launch() - add it there for 6+ to suppress warning
    if is_gradio_6_plus:
        launch_kwargs["css"] = final_css.strip()
    
    gradio_thread = threading.Thread(
        target=lambda: demo.launch(**launch_kwargs),
        daemon=True
    )
    gradio_thread.start()
    
    if wait_for_gradio("http://localhost:7860", timeout=30):
        launch_custom_browser(
            gradio_url="http://localhost:7860/?__theme=dark",
            frameless=False,
            width=1400,
            height=900,
            title="Chat-Gradio-Gguf",
            maximized=True
        )
    else:
        print("[ERROR] Gradio server failed to start")
        import sys
        sys.exit(1)

if __name__ == "__main__":
    launch_display()