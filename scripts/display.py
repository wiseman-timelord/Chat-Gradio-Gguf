# Script: `.\scripts\display.py`
# v2: Gradio 5.x / PyQt6 WebEngine / Windows 10-11 / Ubuntu 24-25
# Includes Qt6 WebEngine browser launcher (merged from browser.py)

# Standard library
import asyncio
import json
import os
import random
import re
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path
from queue import Queue

# Third-party
import gradio as gr
import pyperclip
import tkinter as tk
from tkinter import filedialog
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter

# Project imports - consolidated configuration
import scripts.configure as cfg
from scripts.configure import save_config

# Commonly used constants (kept explicit for clarity & auto-complete)
from scripts.configure import (
    MODEL_NAME, MODEL_FOLDER, SESSION_ACTIVE,
    MAX_HISTORY_SLOTS, MAX_ATTACH_SLOTS, SESSION_LOG_HEIGHT,
    CONTEXT_SIZE, BATCH_SIZE, TEMPERATURE, REPEAT_PENALTY,
    VRAM_SIZE, SELECTED_GPU, SELECTED_CPU, MLOCK, BACKEND_TYPE,
    ALLOWED_EXTENSIONS, VRAM_OPTIONS, CTX_OPTIONS, BATCH_OPTIONS,
    TEMP_OPTIONS, REPEAT_OPTIONS, HISTORY_SLOT_OPTIONS,
    SESSION_LOG_HEIGHT_OPTIONS, ATTACH_SLOT_OPTIONS, HISTORY_DIR,
    USER_COLOR, THINK_COLOR, RESPONSE_COLOR, PRINT_RAW_OUTPUT,
    SHOW_THINK_PHASE, BLEEP_ON_EVENTS, TTS_PHASE, TTS_CURRENT_MSG_IDX, TTS_BUSY,
    MAX_TTS_LENGTH, context_injector, STATUS_MESSAGES
)

# Utility & helper modules
from scripts import utility
from scripts.utility import (
    short_path,
    get_saved_sessions, get_cpu_info, load_session_history, save_session_history,
    get_available_gpus, process_files, eject_file,
    summarize_session, beep, update_file_slot_ui
)

# Model handling
from scripts.inference import (
    get_response_stream, get_available_models, unload_models, get_model_settings,
    load_models
)

# Tools (search, TTS, etc.)
from scripts.tools import (
    format_search_status_for_chat, web_search, format_web_search_status_for_chat,
    get_voice_choices, get_sample_rate_choices,
    speak_last_response, stop_speaking, get_tts_status, initialize_tts,
    get_voice_id_by_name, verify_tts_voice, speak_text,
    synthesize_text_to_file, play_tts_audio
)


# =============================================================================
# BROWSER FUNCTIONS (merged from browser.py)
# Custom Qt6 WebEngine browser window (PyQt6) on all supported platforms.
# - Windows 10/11 : Qt6 WebEngine (PyQt6)
# - Ubuntu 24-25  : Qt6 WebEngine (PyQt6)
# Falls back to system default browser if PyQt6 WebEngine is unavailable.
# =============================================================================

_qt_app = None
_qt_browser = None
_signal_handler = None


def close_browser():
    """Close the Qt browser window from any thread (thread-safe)."""
    global _signal_handler
    if _signal_handler is not None:
        try:
            print("[BROWSER] Requesting close via signal...")
            _signal_handler.request_close()
        except Exception as e:
            print(f"[BROWSER] Signal close failed: {e}")
            if _qt_app is not None:
                try:
                    _qt_app.quit()
                except:
                    pass
    else:
        print("[BROWSER] No signal handler - attempting direct close")
        if _qt_app is not None:
            try:
                _qt_app.quit()
            except:
                pass
    print("[BROWSER] Close requested")


def wait_for_gradio(url="http://localhost:7860", timeout=30):
    """Wait for Gradio server to be fully ready."""
    import requests
    import time as _time
    start_time = _time.time()
    print(f"[BROWSER] Waiting for Gradio server at {url}...")
    while _time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                content = response.text.lower()
                if 'gradio' in content or 'svelte' in content or '<script' in content:
                    _time.sleep(1.5)
                    print("[BROWSER] Gradio server is ready")
                    return True
        except requests.exceptions.RequestException:
            pass
        _time.sleep(0.5)
    print("[BROWSER] Timeout waiting for Gradio server")
    return False

def launch_custom_browser(gradio_url="http://localhost:7860",
                          frameless=False, width=1400, height=900,
                          title="Chat-Gradio-Gguf", maximized=False):
    print(f"[BROWSER] Launching at {gradio_url}")
    print(f"[BROWSER] Platform: {cfg.PLATFORM}, Qt6 WebEngine (PyQt6)")
    try:
        _launch_qt6_browser(gradio_url, title, width, height, frameless, maximized)
    except ImportError as e:
        print(f"[BROWSER] Qt6 WebEngine not available: {e}")
        print("[BROWSER] Falling back to system browser")
        if cfg.PLATFORM == "windows":
            import os
            os.startfile(gradio_url)          # preserves the port
        else:
            import webbrowser
            webbrowser.open(gradio_url)
    except Exception as e:
        print(f"[BROWSER] Qt6 WebEngine failed: {e}")
        traceback.print_exc()
        print("[BROWSER] Falling back to system browser")
        if cfg.PLATFORM == "windows":
            import os
            os.startfile(gradio_url)
        else:
            import webbrowser
            webbrowser.open(gradio_url)

def _launch_qt6_browser(url, title, width, height, frameless, maximized):
    """
    Launch browser using PyQt6 + Qt6 WebEngine.
    v2 primary browser on all supported platforms (Windows 10-11, Ubuntu 24-25).
    """
    global _qt_app, _qt_browser, _signal_handler
    import os as _os

    # Windows: disable GPU acceleration when installer detected insufficient D3D support.
    if cfg.PLATFORM == 'windows' and not getattr(cfg, 'GRAPHICS_ACCELERATION', True):
        chromium_flags = (
            "--disable-gpu "
            "--disable-gpu-compositing "
            "--disable-features=GpuProcessSurface,CanvasOopRasterization "
            "--no-first-run "
            "--disable-default-apps "
            "--disable-background-timer-throttling "
            "--disable-renderer-backgrounding"
        )
        _os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = chromium_flags
        _os.environ["QTWEBENGINE_DISABLE_GPU"] = "1"
        print("[BROWSER] Hardware GPU unavailable - software rendering enabled")
        print(f"[BROWSER] Chromium flags: {chromium_flags}")

    # Linux: Set Chromium flags for sandbox issues and GPU rendering.
    if cfg.PLATFORM == 'linux':
        if _os.geteuid() == 0:
            print("[BROWSER] Running as root - disabling Chromium sandbox")
            chromium_flags = "--no-sandbox --disable-gpu-sandbox"
        else:
            chromium_flags = "--disable-gpu-sandbox"
        chromium_flags += (
            " --disable-gpu --disable-software-rasterizer"
            " --disable-features=GpuProcessSurface,CanvasOopRasterization"
            " --no-first-run --disable-default-apps"
            " --disable-background-timer-throttling"
            " --disable-features=Translate,InterestFeedContentSuggestions,MediaRouter,"
            "OptimizationHints,OptimizationGuideModelDownloading,"
            "AutofillServerCommunication,PasswordManager"
            " --disable-sync --disable-component-extensions-with-background-pages"
            " --disable-backgrounding-occluded-windows --disable-renderer-backgrounding"
        )
        _os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = chromium_flags
        _os.environ["QTWEBENGINE_DISABLE_GPU"] = "1"
        _os.environ["QT_QUICK_BACKEND"] = "software"
        _os.environ["QT_QPA_PLATFORM"] = "xcb"
        _os.environ["QT_QPA_PLATFORMTHEME"] = ""
        _os.environ["QT_QPA_DISABLE_SESSION_MANAGER"] = "1"
        _os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false;qt.webengine.*=false"
        print("[BROWSER] GPU acceleration disabled, using software rendering")
        print("[BROWSER] DBus integration disabled to prevent startup delays")
        print(f"[BROWSER] Chromium flags: {chromium_flags}")

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEngineSettings, QWebEnginePage
    from PyQt6.QtCore import QUrl, Qt, QTimer, pyqtSignal, QObject

    class CloseSignalHandler(QObject):
        close_signal = pyqtSignal()
        def __init__(self, app):
            super().__init__()
            self._app = app
            self.close_signal.connect(self._do_close)
        def _do_close(self):
            print("[BROWSER] Close signal received in main thread")
            if self._app:
                self._app.quit()
        def request_close(self):
            self.close_signal.emit()

    _qt_app = QApplication(sys.argv)
    _signal_handler = CloseSignalHandler(_qt_app)
    _qt_browser = QWebEngineView()
    _qt_browser.setWindowTitle(title)
    if frameless:
        _qt_browser.setWindowFlags(Qt.WindowType.FramelessWindowHint)
    settings = _qt_browser.settings()
    settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.LocalStorageEnabled, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
    settings.setAttribute(QWebEngineSettings.WebAttribute.PluginsEnabled, False)
    settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, False)
    settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, False)
    _qt_browser.resize(width, height)
    if maximized:
        _qt_browser.showMaximized()
    else:
        _qt_browser.show()

    def load_url():
        print(f"[BROWSER] Loading URL: {url}")
        _qt_browser.setUrl(QUrl(url))

    QTimer.singleShot(100, load_url)
    print("[BROWSER] Qt6 WebEngine window created")
    try:
        _qt_app.exec()
    except Exception as e:
        # COM RPC_E_SERVERFAULT (0x80010108) occurs on exit when background threads
        # attempt COM calls after the main thread has terminated. Safe to ignore.
        if "0x80010108" in str(e):
            print("[BROWSER] COM RPC error ignored (normal during shutdown)")
        else:
            raise
    print("[BROWSER] Qt6 event loop exited")


# =============================================================================
# GRADIO 5.x COMPAT STUBS
# Gradio 5 Chatbot uses type="messages" — dict lists are passed directly.
# These stubs preserve call-site compatibility without conversion overhead.
# =============================================================================

def messages_to_tuples(messages):
    """v2 no-op: Gradio 5 Chatbot accepts dict list directly."""
    return messages

def tuples_to_messages(tuples):
    """v2 no-op: already in dict list format."""
    return tuples

def get_chatbot_output(session_tuples, session_messages):
    """Return message dicts for Gradio 5 Chatbot (type='messages')."""
    return session_messages


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def update_cpu_select():
    cpu_info = get_cpu_info()
    choices = ["Auto-Select"] + [c["label"] for c in cpu_info]
    value = cfg.SELECTED_CPU
    if value not in choices:
        value = "Auto-Select"
    return gr.update(choices=choices, value=value)


def ensure_model_loaded():
    """Lazy load model if not already loaded. Returns (success, status_message)"""
    if cfg.MODELS_LOADED and cfg.llm is not None:
        return True, "Model ready"

    model_name = cfg.MODEL_NAME
    if model_name in ["Select_a_model...", "No models found", "", None]:
        return False, "No valid model selected"

    cfg.set_status("Loading model on first use...", priority=True, console=True)
    try:
        status, loaded, llm_new, _ = load_models(
            cfg.MODEL_FOLDER, model_name, cfg.VRAM_SIZE,
            cfg.llm, cfg.MODELS_LOADED
        )
        cfg.llm = llm_new
        cfg.MODELS_LOADED = loaded
        if loaded:
            cfg.set_status("Model loaded", console=True)
            beep()
            return True, "Model loaded successfully"
        else:
            return False, f"Load failed: {status}"
    except Exception as e:
        traceback.print_exc()
        return False, f"Load error: {str(e)}"


def get_model_loaded_display(is_loaded):
    """Return a gr.update for the model loaded indicator textbox."""
    if is_loaded:
        return gr.update(value="🟢 SO LOADED")
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
    lines.append(f"Llama Wheel:  {getattr(cfg, 'LLAMA_WHEEL_VERSION', 'N/A')}")
    lines.append(f"TTS Engine:  {getattr(cfg, 'TTS_ENGINE', 'N/A')}")
    if getattr(cfg, 'TTS_ENGINE', '') == "coqui":
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
    lines.append(f"FILTER_MODE:  {getattr(cfg, 'FILTER_MODE', 'N/A')}")
    return "\n".join(lines)


def save_all_settings():
    """Save all configuration settings and return a status message."""
    cfg.save_config()
    return cfg.STATUS_MESSAGES["config_saved"]


def format_response(output: str) -> str:
    """Format response with thinking phase detection and code highlighting."""
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

    # 3. Fix the "Thinking..." line: collapse dot‑space sequences into contiguous dots
    lines = clean_output.splitlines(keepends=False)
    if lines:
        first_line = lines[0]
        # Check if it's a thinking progress line (starts with "Thinking" and contains dots)
        if first_line.strip().startswith("Thinking") and '.' in first_line:
            # Replace any dot followed by whitespace with a single dot, repeatedly
            # Converts "Thinking. . . ." -> "Thinking...."
            first_line = re.sub(r'\.\s+', '.', first_line)
            # Remove any trailing spaces
            first_line = first_line.rstrip()
            # Insert a blank line after the thinking line
            clean_output = first_line + "\n\n" + "\n".join(lines[1:])
        else:
            clean_output = "\n".join(lines)

    # 4. Highlight code blocks
    code_blocks = re.findall(r'```(\w+)?\n(.*?)```', clean_output, re.DOTALL)
    for lang, code in code_blocks:
        if lang:
            try:
                lexer = get_lexer_by_name(lang, stripall=True)
                formatted_code = highlight(code, lexer, HtmlFormatter())
                clean_output = clean_output.replace(f'```{lang}\n{code}```', formatted_code)
            except:
                pass

    # 5. Common basic normalization
    clean_output = re.sub(r'(?<!>)  +(?![^<]*>)', ' ', clean_output)

    # 5b. Strip decorative separator lines echoed from search context
    clean_output = strip_separators(clean_output)

    # 6. Apply the configurable output filter
    clean_output = apply_output_filter(clean_output)

    clean_output = clean_output.strip()

    # 7. Combine thinking + final output
    if formatted:
        return '\n'.join(formatted) + '\n\n' + clean_output

    return clean_output


# =============================================================================
# OUTPUT FILTERING FUNCTIONS
# =============================================================================

def get_filter_text_for_display():
    """Get the current filter as displayable/editable text."""
    if cfg.FILTER_MODE == "custom":
        custom_path = Path(cfg.CUSTOM_FILTER_PATH)
        if custom_path.exists():
            try:
                return custom_path.read_text(encoding='utf-8')
            except Exception as e:
                print(f"[FILTER] Error reading custom filter: {e}")
        return filter_list_to_text(cfg.ACTIVE_FILTER)
    else:
        preset = cfg.FILTER_PRESETS.get(cfg.FILTER_MODE, [])
        return filter_list_to_text(preset)


def filter_list_to_text(filter_list):
    """Convert filter list to editable text format."""
    lines = []
    for find, replace in filter_list:
        find_escaped = repr(find)[1:-1]
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
    return "", "Unknown preset"


def save_custom_filter(filter_text):
    """Save the current filter text as a custom filter."""
    try:
        filter_list = text_to_filter_list(filter_text)
        custom_path = Path(cfg.CUSTOM_FILTER_PATH)
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        custom_path.write_text(filter_text, encoding='utf-8')
        cfg.FILTER_MODE = "custom"
        cfg.ACTIVE_FILTER = filter_list
        cfg.save_config()
        return f"Custom filter saved ({len(filter_list)} rules)"
    except Exception as e:
        return f"Error saving filter: {e}"


def initialize_filter_from_config():
    """Initialize the filter based on saved config or Gradio version."""
    filter_mode = getattr(cfg, 'FILTER_MODE', None)
    if filter_mode == "custom":
        custom_path = Path(cfg.CUSTOM_FILTER_PATH)
        if custom_path.exists():
            try:
                text = custom_path.read_text(encoding='utf-8')
                cfg.ACTIVE_FILTER = text_to_filter_list(text)
                print(f"[FILTER] Loaded custom filter ({len(cfg.ACTIVE_FILTER)} rules)")
                return
            except Exception as e:
                print(f"[FILTER] Error loading custom filter: {e}")
    # v2 default: always use gradio5 filter
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


def update_model_list(model_folder):
    """Update model dropdown choices and set correct initial value from loaded config."""
    print(f"[MODEL-LIST] update_model_list called with folder='{model_folder}'")

    if model_folder and model_folder.strip() and model_folder != cfg.MODEL_FOLDER:
        cfg.MODEL_FOLDER = model_folder
        print(f"[MODEL-LIST] cfg.MODEL_FOLDER updated to: {cfg.MODEL_FOLDER}")

    available = get_available_models()
    PLACEHOLDERS = {"Select_a_model...", "No models found"}
    real_models = [m for m in available if m not in PLACEHOLDERS]

    if real_models:
        choices = ["Select_a_model..."] + real_models
    else:
        choices = ["No models found"]

    saved = cfg.MODEL_NAME
    if saved in real_models:
        selected_value = saved
    elif real_models:
        selected_value = real_models[0]
        cfg.MODEL_NAME = selected_value
        print(f"[MODEL-LIST] Auto-selected first model: '{selected_value}'")
    else:
        selected_value = "No models found"
        cfg.MODEL_NAME = "Select_a_model..."

    interactive_val = len(real_models) > 0
    print(f"[MODEL-LIST] Returning choices={len(choices)} items, value='{selected_value}', interactive={interactive_val}")
    return gr.update(choices=choices, value=selected_value, interactive=interactive_val)


def handle_load_model(model_name, model_folder, vram_size, ctx_size, gpu, cpu, cpu_threads, llm_state, models_loaded_state):
    """Explicitly load the currently selected model with current cfg."""
    if not model_name or model_name in ["Select_a_model...", "No models found", " ", None]:
        return (
            llm_state, models_loaded_state,
            "❌ Error: No valid model selected. Choose a model first.",
            "❌ Error: No valid model selected. Choose a model first.",
            "❌ Error: No valid model selected. Choose a model first.",
            gr.update(interactive=False),
            get_model_loaded_display(False)
        )

    cfg.MODEL_NAME   = model_name
    cfg.MODEL_FOLDER = model_folder
    cfg.VRAM_SIZE    = int(vram_size)
    cfg.CONTEXT_SIZE = int(ctx_size)
    cfg.SELECTED_GPU = gpu
    cfg.SELECTED_CPU = cpu
    cfg.CPU_THREADS  = int(cpu_threads) if cpu_threads else None

    try:
        status, loaded, new_llm, _ = load_models(
            model_folder, model_name, int(vram_size), llm_state, models_loaded_state
        )
        if loaded:
            cfg.MODELS_LOADED = True
            cfg.llm = new_llm
            cfg.LOADED_CONTEXT_SIZE = int(ctx_size)
            beep()
            status_msg = f"✅ Model loaded: {model_name} ({cfg.LOADED_CONTEXT_SIZE} ctx)"
            input_interactive = True
        else:
            status_msg = f"❌ Load failed: {status[:150]}"
            input_interactive = False
            new_llm = llm_state
            loaded = False
        return (
            new_llm, loaded,
            status_msg, status_msg, status_msg,
            gr.update(interactive=input_interactive),
            get_model_loaded_display(loaded)
        )
    except Exception as e:
        traceback.print_exc()
        status_msg = f"❌ Load error: {str(e)[:150]}"
        return (
            llm_state, models_loaded_state,
            status_msg, status_msg, status_msg,
            gr.update(interactive=False),
            get_model_loaded_display(False)
        )


def handle_unload_model(llm_state, models_loaded_state):
    """Explicitly unload the currently loaded model."""
    if not models_loaded_state or llm_state is None:
        return (
            llm_state, False,
            "ℹ️ No model currently loaded.",
            "ℹ️ No model currently loaded.",
            "ℹ️ No model currently loaded.",
            gr.update(interactive=False),
            get_model_loaded_display(False)
        )
    try:
        status, new_llm, new_models_loaded = unload_models(llm_state, models_loaded_state)
        cfg.MODELS_LOADED = new_models_loaded
        cfg.llm = new_llm
        cfg.GPU_LAYERS = 0
        cfg.LOADED_CONTEXT_SIZE = None
        beep()
        status_msg = "✅ Model unloaded successfully."
        return (
            new_llm, new_models_loaded,
            status_msg, status_msg, status_msg,
            gr.update(interactive=False),
            get_model_loaded_display(False)
        )
    except Exception as e:
        traceback.print_exc()
        status_msg = f"❌ Unload error: {str(e)[:150]}"
        return (
            llm_state, models_loaded_state,
            status_msg, status_msg, status_msg,
            gr.update(interactive=True),
            get_model_loaded_display(True)
        )


def start_new_session(session_messages, attached_files, llm_state, models_loaded_state):
    """Start a fresh session. Preserves model loaded state to prevent unnecessary reloads."""
    if cfg.SESSION_ACTIVE and session_messages:
        try:
            save_session_history(session_messages, attached_files)
            print("[SESSION] Previous session auto-saved before starting new one")
        except Exception as e:
            print(f"[SESSION] Failed to save previous session: {e}")

    cfg.SESSION_ACTIVE = False
    cfg.current_session_id = None
    cfg.session_label = ""
    cfg.session_attached_files = []

    chatbot_output = get_chatbot_output([], [])
    return (
        chatbot_output, [], [], "New session started.", False,
        *update_action_buttons("waiting_for_input", False),
        llm_state, models_loaded_state
    )


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

    has_ai = any(msg.get('role') == 'assistant' for msg in history)
    status = f"Loaded session: {label}"
    chatbot_output = get_chatbot_output(messages_to_tuples(history), history)
    button_updates = update_action_buttons("waiting_for_input", has_ai)

    return (chatbot_output, history, attached_files, status, has_ai) + tuple(button_updates)


def handle_inline_copy(action_str, session_messages):
    """Triggered by JS inline copy button via hidden relay textbox."""
    if not action_str or ':' not in action_str:
        return "Nothing to copy.", ""
    try:
        clean_str = action_str.split('|')[0]
        role_part, idx_s = clean_str.rsplit(':', 1)
        nth = int(idx_s)
    except (ValueError, AttributeError):
        return "Copy error: bad format.", ""

    if role_part == 'user':
        candidates = [m for m in session_messages if m.get('role') == 'user']
    else:
        candidates = [m for m in session_messages if m.get('role') == 'assistant']

    if not candidates or nth >= len(candidates):
        return "Message not found.", ""

    content = candidates[nth].get('content', '') or ''
    clean = re.sub(r'<[^>]+>', '', content)
    clean = (clean
             .replace('&amp;',  '&')
             .replace('&lt;',   '<')
             .replace('&gt;',   '>')
             .replace('&quot;', '"')
             .replace('&#39;',  "'"))
    clean = re.sub(r'^(?:AI-Chat|User):\s*\n?', '', clean.strip(), flags=re.MULTILINE).strip()
    try:
        pyperclip.copy(clean)
        label = "AI response" if role_part == 'bot' else "User message"
        return f"{label} copied to clipboard.", ""
    except Exception as e:
        return f"Copy error: {e}", ""


def _tts_worker(msg_index, text):
    """Background worker for TTS: synthesize then play."""
    from scripts.tools import clear_tts_stop
    clear_tts_stop()

    cfg.TTS_BUSY = True
    cfg.TTS_PHASE = "generating"
    cfg.TTS_CURRENT_MSG_IDX = msg_index
    wav_path = None
    try:
        print(f"[TTS] Starting synthesis for msg {msg_index} ({len(text)} chars)...")
        wav_path = synthesize_text_to_file(text)
        if wav_path and os.path.exists(wav_path):
            cfg.TTS_PHASE = "playing"
            play_tts_audio(wav_path)
        elif not wav_path:
            print("[TTS] Synthesis returned no file, skipping playback")
    except Exception as e:
        print(f"[TTS] Worker error: {e}")
        traceback.print_exc()
    finally:
        cfg.TTS_PHASE = "idle"
        cfg.TTS_CURRENT_MSG_IDX = None
        cfg.TTS_BUSY = False
        print(f"[TTS] Worker done for msg {msg_index}")


def handle_inline_tts(action_str, session_messages):
    """Handle per-message TTS button clicks. 4-phase cycle: play->generating->playing->idle."""
    result = {"tts_state": gr.update(value="idle")}

    if not action_str or not action_str.strip():
        return result["tts_state"]

    clean_str = action_str.split('|')[0]
    parts = clean_str.split(':')
    if len(parts) < 2:
        return result["tts_state"]

    action = parts[0].strip()
    msg_idx_str = parts[1].strip()

    try:
        msg_idx = int(msg_idx_str)
    except ValueError:
        return result["tts_state"]

    if action == "stop":
        print(f"[TTS] Stop requested for msg {msg_idx}")
        stop_speaking()
        cfg.TTS_PHASE = "idle"
        cfg.TTS_CURRENT_MSG_IDX = None
        cfg.TTS_BUSY = False
        return gr.update(value="idle")

    if action == "bot":
        if cfg.TTS_BUSY:
            print(f"[TTS] Already busy (msg {cfg.TTS_CURRENT_MSG_IDX}), ignoring duplicate trigger")
            return gr.update(value=f"{cfg.TTS_CURRENT_MSG_IDX}|{cfg.TTS_PHASE}")

        bot_msgs = [m for m in session_messages if m.get("role") == "assistant"]
        if msg_idx < 0 or msg_idx >= len(bot_msgs):
            print(f"[TTS] Invalid message index: {msg_idx}")
            return result["tts_state"]

        text = bot_msgs[msg_idx].get("content", "")
        if not text.strip():
            return result["tts_state"]

        if len(text) > cfg.MAX_TTS_LENGTH:
            text = text[:cfg.MAX_TTS_LENGTH]
            print(f"[TTS] Text truncated to {cfg.MAX_TTS_LENGTH} chars")

        cfg.TTS_BUSY = True
        cfg.TTS_PHASE = "generating"
        cfg.TTS_CURRENT_MSG_IDX = msg_idx

        thread = threading.Thread(
            target=_tts_worker,
            args=(msg_idx, text),
            daemon=True,
            name=f"TTSWorker-{msg_idx}"
        )
        thread.start()
        return gr.update(value=f"{msg_idx}|generating")

    return result["tts_state"]


def tts_heartbeat():
    """Return current TTS state for JS polling. Called periodically by demo.load()."""
    idx = getattr(cfg, 'TTS_CURRENT_MSG_IDX', None)
    phase = getattr(cfg, 'TTS_PHASE', 'idle')
    if idx is not None and phase != 'idle':
        return f"{idx}|{phase}"
    return "idle"


def handle_inline_edit(idx_str, session_messages):
    """Triggered by JS inline edit button via hidden relay textbox."""
    if not idx_str or not idx_str.strip():
        return gr.update(), gr.update(), session_messages, gr.update(), gr.update(), gr.update()
    try:
        clean_str = idx_str.split('|')[0].strip()
        if not clean_str:
            return gr.update(), gr.update(), session_messages, gr.update(), gr.update(), gr.update()
        nth = int(clean_str)
    except ValueError:
        return gr.update(), gr.update(), session_messages, "Edit error: invalid index.", gr.update(), gr.update()

    user_entries = [(i, m) for i, m in enumerate(session_messages) if m.get('role') == 'user']
    if nth >= len(user_entries):
        return gr.update(), gr.update(), session_messages, "Edit target not found.", gr.update(), gr.update()

    original_idx, target_msg = user_entries[nth]
    user_content = target_msg.get('content', '') or ''
    user_content = re.sub(r'^User:\s*\n?', '', user_content, flags=re.MULTILINE).strip()

    new_messages = session_messages[:original_idx]
    chatbot_out  = get_chatbot_output(messages_to_tuples(new_messages), new_messages)
    has_ai       = any(m.get('role') == 'assistant' for m in new_messages)
    status       = f"✏️ Editing from message {nth + 1} — edit the text above then Send Input."
    return user_content, chatbot_out, new_messages, status, has_ai, gr.update()


def handle_inline_retry(action_str, session_messages):
    """Triggered by JS retry button via hidden relay textbox."""
    if not action_str or not action_str.strip():
        return gr.update(), gr.update(), session_messages, gr.update(), gr.update(), gr.update()

    try:
        clean_str = action_str.split('|')[0].strip()
        if not clean_str:
            return gr.update(), gr.update(), session_messages, gr.update(), gr.update(), gr.update()
        parts = clean_str.split(':')
        nth = int(parts[1])
    except (ValueError, IndexError, AttributeError):
        return gr.update(), gr.update(), session_messages, "Retry error: bad format.", gr.update(), gr.update()

    bot_entries = [(i, m) for i, m in enumerate(session_messages) if m.get('role') == 'assistant']
    if nth >= len(bot_entries):
        return gr.update(), gr.update(), session_messages, "Retry target not found.", gr.update(), gr.update()

    bot_abs_idx, _ = bot_entries[nth]
    user_entries_before = [
        (i, m) for i, m in enumerate(session_messages[:bot_abs_idx])
        if m.get('role') == 'user'
    ]
    if not user_entries_before:
        return gr.update(), gr.update(), session_messages, "No preceding user message found.", gr.update(), gr.update()

    user_abs_idx, user_msg = user_entries_before[-1]
    user_content = user_msg.get('content', '') or ''
    user_content = re.sub(r'^User:\s*\n?', '', user_content, flags=re.MULTILINE).strip()

    new_messages = session_messages[:user_abs_idx]
    chatbot_out = get_chatbot_output(messages_to_tuples(new_messages), new_messages)
    has_ai = any(m.get('role') == 'assistant' for m in new_messages)
    status = f"🔄 Retrying response {nth + 1} — regenerating..."

    return user_content, chatbot_out, new_messages, status, has_ai, gr.update()


def update_session_buttons():
    """Update session history buttons."""
    sessions = get_saved_sessions()[:cfg.MAX_HISTORY_SLOTS]
    button_updates = []

    for i in range(cfg.MAX_POSSIBLE_HISTORY_SLOTS):
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
    """Update action buttons based on interaction phase.
    Dynamic bar is simplified to two states only:
      • waiting_for_input  →  'Send Input' visible, 'Wait' hidden
      • generating         →  'Send Input' hidden, 'Wait' visible
    Edit Previous and Copy Output are handled by per-message inline icons.
    """
    if phase == "waiting_for_input":
        action_visible, wait_visible = True, False
    elif phase in ("input_submitted", "generating_response", "speaking"):
        action_visible, wait_visible = False, True
    else:
        action_visible, wait_visible = True, False

    action_value   = "Send Input"
    action_variant = "secondary" if phase == "waiting_for_input" else "primary"
    action_classes = ["send-button-green"] if phase == "waiting_for_input" else []
    action_interactive = (phase == "waiting_for_input")

    wait_value = "..Wait For Response.."
    if phase == "speaking" and getattr(cfg, 'TTS_ENABLED', False):
        wait_value = "🔊 Speaking Response..."

    return [
        gr.update(value=action_value, variant=action_variant, elem_classes=action_classes,
                  interactive=action_interactive, visible=action_visible),
        gr.update(visible=False),   # edit_previous — removed from bar; compat placeholder
        gr.update(visible=False),   # copy_response — removed from bar; compat placeholder
        gr.update(visible=False),   # cancel_input  — hidden placeholder
        gr.update(value=wait_value, variant="primary", interactive=False, visible=wait_visible),
    ]


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


def get_phase_list(web_search_enabled: bool = False, auto_tts_enabled: bool = False) -> list:
    """Return ordered list of progress phases based on enabled features."""
    phases = [
        "Handle Input",
        "Build Prompt",
        "Inject RAG",
        "Add System",
        "Assemble History",
        "Check Model",
    ]
    if web_search_enabled:
        phases.extend([
            "Search Discovery",
            "Rank & Select",
            "Parallel Fetch",
            "Process & Merge",
            "Inject Context",
        ])
    phases.extend([
        "Generate Stream",
        "Format Response",
    ])
    if auto_tts_enabled:
        phases.extend([
            "Synthesizing Speech",
            "Playing Audio",
        ])
    return phases

def build_progress_html(step: int, web_search_enabled: bool = False, auto_tts_enabled: bool = False) -> str:
    """Build dynamic progress indicator HTML based on enabled features."""
    phases = get_phase_list(web_search_enabled, auto_tts_enabled)
    segments = []
    for i, phase in enumerate(phases):
        if i < step:
            color = "#00ff00"      # completed
        elif i == step:
            color = "#4488ff"      # current
        else:
            color = "#666666"      # pending
        segments.append(f'<span style="color:{color}; font-weight:bold;">{phase}</span>')
    return " → ".join(segments)


def extract_search_query(user_input: str) -> str:
    """Extract a clean, focused search query from natural language user input."""
    original = user_input.strip()
    working = original

    leading_patterns = [
        r'^produce\s+(?:a\s+)?web\s+(?:research|search)\b\s*(?:and\s+compile\s+(?:a\s+)?(?:report|timeline|summary)\s*)?,?\s*(?:upon|on|about|for|regarding|into|relating\s+to)?\s*(?:recent\s+events?\s*(?:relating\s+to|regarding|about|on|in|concerning)?\s*)?',
        r'^compile\s+(?:a\s+)?(?:report|timeline|summary|analysis)\s*,?\s*',
        r'^i\s+want\s+a\s+(?:timeline|report|summary|list)\s+of\s+(?:the\s+)?events?\s+(?:from\s+)?(?:the\s+)?(?:most\s+recent|last|past)\s+\d+\s+days?\s*(?:relating\s+to|about|on|in|regarding|concerning)?\s*',
        r'^i\s+want\s+a\s+(?:timeline|report|summary|list)\s+of\s+(?:the\s+)?(?:events?\s+)?(?:from\s+|about\s+|on\s+|regarding\s+)?',
        r'^(?:do|perform|run|conduct)\s+(?:a\s+)?web\s+(?:search|research)\s*(?:on|for|about|regarding|into|upon)?\s*(?:recent\s+events?\s*(?:relating\s+to|regarding|about|on|in)?\s*)?',
        r'^search\s+(?:the\s+web\s+|online\s+)?(?:for|about|on|regarding|into)?\s*(?:recent\s+events?\s*(?:relating\s+to|regarding|about|on|in)?\s*)?',
        r'^(?:find|look\s+up|look\s+for|get|fetch)\s+(?:(?:recent|latest|current|new)\s+)?(?:news|information|info|data|details|updates?|results?)?\s*(?:about|on|for|regarding|into)?\s*',
        r'^research\s+(?:(?:the|recent|latest|current)\s+)?(?:events?\s+(?:about|on|in|regarding)?\s*)?',
        r'^what\s+(?:are|is)\s+(?:the\s+)?(?:latest|recent|current|new)\s+(?:news|information|updates?|events?|developments?)\s+(?:about|on|regarding|in|concerning)\s+',
        r'^(?:can\s+you|could\s+you|would\s+you|will\s+you|please)?\s*(?:please\s+)?(?:search|find|look\s+up|research|google|check|investigate|tell\s+me\s+about)\s+(?:for\s+|about\s+|on\s+|into\s+)?',
        r'^i\s+(?:want|need|would\s+like)\s+(?:(?:a\s+)?(?:report|timeline|summary|analysis)\s+(?:on|about|of|regarding)\s+|(?:to\s+know|information|info)\s+(?:about|on|regarding)\s+)?',
        r'^i\s+want\s+a\s+timeline\s+of\s+(?:the\s+)?(?:events?\s+(?:from|in|about|on|regarding)?\s*)?',
        r'^.*?(?:i\'?m?|i\s+am)\s+(?:developing|building|creating|working\s+on|testing)\s+(?:the|my|this)?\s*(?:internet\s+based\s+tools?|tools?|chatbot|ai|program|script|features?).*?(?:so\s+here|here)\s+(?:is|are|goes)\s*(?:a\s+)?(?:test|example).*?\n+',
        r'^(?:test|testing)[:\s-]+',
    ]

    query = working
    for pattern in leading_patterns:
        candidate = re.sub(pattern, '', query, flags=re.IGNORECASE).strip()
        if len(candidate) > 4:
            query = candidate

    trailing_patterns = [
        r',?\s*(?:and\s+)?(?:then\s+)?(?:compile|create|write|produce|generate|build|format)\s+(?:a\s+)?(?:report|timeline|summary|analysis|list|overview|review)\b.*$',
        r',?\s*(?:and\s+)?(?:then\s+)?(?:present|display|show|output)\s+(?:it|them|the\s+results?)\b.*$',
        r',?\s*(?:for|over)\s+(?:the\s+)?(?:most\s+recent|last|past)\s+\d+\s+days?\b.*$',
        r',?\s*from\s+the\s+current\s+date\b.*$',
        r'\s*if\s+(?:you\s+)?(?:cannot|can\'t|could\s+not)\b.*$',
        r'\s*if\s+that\s+fails\b.*$',
        r'\s*otherwise\b.*$',
        r'\s*alternatively\b.*$',
        r'\s+in\s+which\s+case\b.*$',
    ]
    for pattern in trailing_patterns:
        candidate2 = re.sub(pattern, '', query, flags=re.IGNORECASE | re.DOTALL).strip()
        if len(candidate2) > 4:
            query = candidate2

    if re.search(r'[.!?]', query):
        sentences = re.split(r'(?<=[.!?])\s+', query)
        best_sentence, best_score = '', -1
        topic_words = [
            'iran', 'iraq', 'israel', 'syria', 'ukraine', 'russia', 'china',
            'middle.?east', 'war', 'conflict', 'election', 'crisis', 'attack',
            'ceasefire', 'nuclear', 'climate', 'economy', 'protest', 'uprising',
        ]
        meta_words = ['chatbot', 'test', 'testing', 'developing', 'tool', 'i am', "i'm"]
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            score = 0
            words = sentence.split()
            for i, word in enumerate(words):
                if i > 0 and word[0].isupper() and len(word) > 2:
                    score += 3
            if re.search(r'\b20\d\d\b', sentence):
                score += 5
            for tw in topic_words:
                if re.search(tw, sentence, re.IGNORECASE):
                    score += 4
            for mw in meta_words:
                if mw in sentence.lower():
                    score -= 2
            if score > best_score:
                best_score, best_sentence = score, sentence
        if best_score >= 0 and best_sentence:
            query = best_sentence

    query = query.strip().strip('"\'').strip()
    query = re.sub(r'\s+', ' ', query)
    query = re.sub(r'[.!?]+$', '', query).strip()

    if len(query) < 5:
        skip = {'the', 'a', 'an', 'and', 'or', 'but', 'so', 'here', 'test', 'i'}
        keywords = [m.group() for m in re.finditer(r'\b[A-Z][a-zA-Z]{2,}\b', original)
                    if m.group().lower() not in skip]
        keywords += [m.group() for m in re.finditer(r'\b20\d\d\b', original)]
        query = ' '.join(keywords[:8]) if keywords else original[:100]

    if len(query) > 120:
        query = query[:120].rsplit(' ', 1)[0]

    print(f"[SEARCH-QUERY] Original: '{original[:60]}...' → Extracted: '{query}'")
    return query


def update_tts_voice(voice_name):
    """Update the selected TTS voice and verify the .pt file is installed."""
    voice_id = get_voice_id_by_name(voice_name)
    cfg.TTS_VOICE      = voice_id
    cfg.TTS_VOICE_NAME = voice_name
    save_config()
    ok, msg = verify_tts_voice(voice_id)
    if ok:
        return f"Voice set to: {voice_name}"
    else:
        return f"Voice set to: {voice_name} — WARNING: {msg}"


def update_sound_sample_rate(sample_rate):
    """Update the audio sample rate (shared by Bleep and TTS)."""
    cfg.SOUND_SAMPLE_RATE = int(sample_rate)
    return f"Sample rate set to: {sample_rate}"


# =============================================================================
# MAIN CONVERSATION HANDLER
# =============================================================================

_cancel_event = threading.Event()


def conversation_display(
    user_input, session_tuples, session_messages, loaded_files,
    is_reasoning_model, cancel_flag, web_search_enabled,
    interaction_phase, llm_state, models_loaded_state,
    has_ai_response_state, tts_speak_enabled
):
    """
    Main conversation handler - Gradio 5.x.
    Uses message dict list directly; Chatbot type='messages' renders natively.
    """
    from scripts.inference import get_model_settings, get_response_stream, load_models
    from scripts.configure import context_injector
    from scripts.utility import read_file_content

    # Helper to clean text for TTS
    def _clean_text_for_tts(text: str) -> str:
        """Remove markdown, tags, and thinking indicators for TTS."""
        text = re.sub(r'^AI-Chat:\s*\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'(?m)^Thinking[.\s]+\r?\n?', '', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = re.sub(r'`[^`]+`', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'!\[.*?\]\([^)]+\)', '', text)
        text = re.sub(r'\*\*', '', text)
        text = re.sub(r'\*', '', text)
        text = re.sub(r'~~', '', text)
        text = re.sub(r'(?<!\w)_|_(?!\w)', '', text)
        text = re.sub(r'[#•→⇒★☆]|[-=]{2,}', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # CRITICAL FIX: Recover from Gradio state reset (e.g., after New Session)
    if llm_state is None and cfg.llm is not None:
        llm_state = cfg.llm
        models_loaded_state = cfg.MODELS_LOADED
        print("[CONVERSATION] Restored model state from global config after session reset")

    # AUTO-LOAD MODEL IF NOT LOADED YET
    if not models_loaded_state or llm_state is None:
        if cfg.MODEL_NAME in ["Select_a_model...", "No models found", "", None]:
            yield (
                get_chatbot_output(session_tuples, session_messages),
                session_messages,
                "❌ No valid model selected. Please choose one in Configuration tab.",
                gr.update(visible=True),
                gr.update(visible=False),
                *update_action_buttons("waiting_for_input", has_ai_response_state),
                cancel_flag,
                loaded_files,
                "waiting_for_input",
                llm_state,
                models_loaded_state
            )
            return

        model_path = Path(cfg.MODEL_FOLDER) / cfg.MODEL_NAME
        if not model_path.is_file():
            yield (
                get_chatbot_output(session_tuples, session_messages),
                session_messages,
                f"❌ Model file missing: {cfg.MODEL_NAME}\nCheck folder path.",
                gr.update(visible=True),
                gr.update(visible=False),
                *update_action_buttons("waiting_for_input", has_ai_response_state),
                cancel_flag,
                loaded_files,
                "waiting_for_input",
                llm_state,
                models_loaded_state
            )
            return

        yield (
            get_chatbot_output(session_tuples, session_messages),
            session_messages,
            "⏳ Loading model... (this may take 10–120 seconds) ",
            gr.update(visible=False),
            gr.update(visible=True, value="Auto-loading model — please wait... "),
            *update_action_buttons("input_submitted", has_ai_response_state),
            cancel_flag,
            loaded_files,
            "input_submitted",
            llm_state,
            models_loaded_state
        )

        try:
            status, loaded, new_llm, _ = load_models(
                cfg.MODEL_FOLDER, cfg.MODEL_NAME, cfg.VRAM_SIZE,
                llm_state, models_loaded_state
            )

            if loaded:
                cfg.MODELS_LOADED = True
                cfg.llm = new_llm
                cfg.LOADED_CONTEXT_SIZE = cfg.CONTEXT_SIZE

                yield (
                    get_chatbot_output(session_tuples, session_messages),
                    session_messages,
                    f"✅ Model loaded: {cfg.MODEL_NAME} ",
                    gr.update(visible=False),
                    gr.update(visible=True, value="Model ready — processing your message... "),
                    *update_action_buttons("input_submitted", has_ai_response_state),
                    cancel_flag,
                    loaded_files,
                    "input_submitted",
                    new_llm,
                    True
                )

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
                    cancel_flag,
                    loaded_files,
                    "waiting_for_input",
                    llm_state,
                    False
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
                cancel_flag,
                loaded_files,
                "waiting_for_input",
                llm_state,
                False
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
            models_loaded_state
        )
        return

    has_ai_response = len([m for m in session_messages if m.get('role') == 'assistant']) > 0
    interaction_phase = "input_submitted"

    processed_input = user_input
    file_contents_section = ""
    complete_user_message = ""
    search_results = None

    # Get ordered phase list based on enabled features
    phase_list = get_phase_list(web_search_enabled, tts_speak_enabled)
    
    def yield_progress(phase_name):
        """Yield progress update for a specific phase name."""
        try:
            step = phase_list.index(phase_name)
        except ValueError:
            step = 0
        
        return (
            get_chatbot_output(session_tuples, session_messages),
            session_messages,
            "",
            gr.update(visible=False),
            gr.update(visible=True, value=build_progress_html(step, web_search_enabled, tts_speak_enabled)),
            *update_action_buttons("input_submitted", has_ai_response),
            cancel_flag,
            loaded_files,
            "input_submitted",
            llm_state,
            models_loaded_state
        )

    # PHASE 0: Handle Input
    yield yield_progress("Handle Input")

    # PHASE 1: Build Prompt
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

    yield yield_progress("Build Prompt")

    # PHASE 2: Inject RAG
    effective_context = cfg.LOADED_CONTEXT_SIZE or cfg.CONTEXT_SIZE
    context_threshold = cfg.LARGE_INPUT_THRESHOLD
    max_input_chars = int(effective_context * 3 * context_threshold)
    total_input_size = len(user_input) + len(file_contents_section)

    if total_input_size > max_input_chars:
        try:
            if len(user_input) > max_input_chars // 2:
                context_injector.add_temporary_input(user_input)
                processed_input = user_input[:1000] + "\n\n[Large input processed with RAG]"
            if len(file_contents_section) > max_input_chars // 2:
                context_injector.add_temporary_input(file_contents_section)
                file_contents_section = file_contents_section[:2000] + "\n\n[Large attachments indexed for retrieval]"
        except Exception as e:
            print(f"[RAG-TEMP] Error: {e}")

    yield yield_progress("Inject RAG")

    # PHASE 3: Add System
    complete_user_message = processed_input
    if file_contents_section:
        complete_user_message = processed_input + file_contents_section

    yield yield_progress("Add System")

    # PHASE 4: Assemble History
    session_messages = list(session_messages) if session_messages else []
    session_messages.append({'role': 'user', 'content': 'User:\n' + complete_user_message})
    has_ai_response = len([m for m in session_messages[:-1] if m.get('role') == 'assistant']) > 0
    interaction_phase = "input_submitted"

    yield yield_progress("Assemble History")

    # PHASE 5: Check Model
    model_settings = get_model_settings(cfg.MODEL_NAME)

    # SEARCH LOGIC
    search_metadata = None
    search_status_text = ""
    search_warning = ""

    if web_search_enabled and user_input.strip():
        try:
            effective_ctx = cfg.LOADED_CONTEXT_SIZE or cfg.CONTEXT_SIZE
            context_multiplier = effective_ctx / 32768.0

            if effective_ctx < 16384:
                search_warning = f"⚠️ Low context ({effective_ctx}) may limit search quality. Recommend 32768+."
                print(f"[SEARCH-WARNING] {search_warning}")

            scaled_max_results = max(6, round(12 * context_multiplier))
            scaled_deep_fetch  = max(3, round(6  * context_multiplier))

            search_query = extract_search_query(user_input)
            print(f"[SEARCH-DEBUG] Original input: {user_input[:100]}...")
            print(f"[SEARCH-DEBUG] Extracted query: '{search_query}'")
            print(f"[WEB-SEARCH] Query: {search_query} (ctx={effective_ctx}, results={scaled_max_results}, deep={scaled_deep_fetch})")

            result = utility.web_search(search_query, max_results=scaled_max_results, deep_fetch=scaled_deep_fetch)

            if isinstance(result, dict):
                search_results = result.get('content', '')
                search_metadata = result.get('metadata', {})
            else:
                search_results = result
                search_metadata = {'type': 'web_search', 'query': search_query, 'sources': [], 'error': None}

            if search_metadata:
                search_metadata['context_size'] = effective_ctx
                search_metadata['context_multiplier'] = round(context_multiplier, 2)

            if search_metadata:
                search_status_text = utility.format_search_status_for_chat(search_metadata)
                if search_warning:
                    search_status_text = f"{search_warning}\n{search_status_text}" if search_status_text else search_warning
                if search_status_text:
                    print(f"[SEARCH-STATUS]\n{search_status_text}")

        except Exception as e:
            print(f"[SEARCH] Error: {e}")
            traceback.print_exc()
            search_results = f"Search error: {str(e)}"
            search_metadata = {'type': 'web_search', 'query': user_input[:100], 'error': str(e), 'sources': []}
            search_status_text = f"⚠️ Search Error: {str(e)}"

    yield yield_progress("Check Model")

    # Web search phases (if enabled)
    if web_search_enabled:
        for phase_name in ["Search Discovery", "Rank & Select", "Parallel Fetch", "Process & Merge", "Inject Context"]:
            yield yield_progress(phase_name)
            time.sleep(0.15)

    # Generate Stream phase
    yield yield_progress("Generate Stream")
    _cancel_event.clear()

    session_messages.append({'role': 'assistant', 'content': "AI-Chat:\n"})
    accumulated_response = ""

    try:
        for chunk in get_response_stream(
            session_log=session_messages,
            settings=model_settings,
            web_search_enabled=web_search_enabled,
            search_results=search_results,
            cancel_event=_cancel_event,
            llm_state=llm_state,
            models_loaded_state=models_loaded_state
        ):
            if _cancel_event.is_set():
                accumulated_response += "\n\n[Response cancelled]"
                break

            accumulated_response += chunk

            clean_response = re.sub(r'^AI-Chat:\s*\n?', '', accumulated_response, flags=re.MULTILINE)
            clean_response = re.sub(r'\nAI-Chat:\s*\n?', '\n', clean_response)

            clean_response = re.sub(
                r'^(Thinking)((?:\. )*\.?)',
                lambda m: m.group(1) + '.' * m.group(2).count('.'),
                clean_response,
                flags=re.MULTILINE
            )

            clean_response = strip_separators(clean_response)
            session_messages[-1]['content'] = "AI-Chat:\n" + clean_response

            # During streaming, keep progress on Generate Stream
            step = phase_list.index("Generate Stream")
            yield (
                get_chatbot_output(session_tuples, session_messages),
                session_messages,
                "",
                gr.update(visible=False),
                gr.update(visible=True, value=build_progress_html(step, web_search_enabled, tts_speak_enabled)),
                *update_action_buttons("generating_response", has_ai_response),
                cancel_flag,
                loaded_files,
                "generating_response",
                llm_state,
                models_loaded_state
            )

    except Exception as e:
        accumulated_response = f"Error: {str(e)}"
        session_messages[-1]['content'] = accumulated_response

    # Format Response phase
    yield yield_progress("Format Response")
    formatted_response = format_response(accumulated_response)
    formatted_response = re.sub(r'^AI-Chat:\s*\n?', '', formatted_response, flags=re.MULTILINE)
    formatted_response = re.sub(r'\nAI-Chat:\s*\n?', '\n', formatted_response)
    session_messages[-1]['content'] = "AI-Chat:\n" + formatted_response

    # Session save
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

    # Clear temporary RAG input
    try:
        context_injector.clear_temporary_input()
    except:
        pass

    cfg.session_attached_files = []
    beep()

    # Auto-TTS: synchronous with progress stages
    if tts_speak_enabled and cfg.TTS_ENABLED:
        try:
            bot_msgs = [m for m in session_messages if m.get("role") == "assistant"]
            if bot_msgs:
                msg_idx = len(bot_msgs) - 1
                text = bot_msgs[msg_idx].get("content", "")
                if text.strip():
                    # Clean text for TTS
                    text = _clean_text_for_tts(text)
                    if text:
                        if len(text) > cfg.MAX_TTS_LENGTH:
                            text = text[:cfg.MAX_TTS_LENGTH]
                            print(f"[AUTO-TTS] Text truncated to {cfg.MAX_TTS_LENGTH} chars")
                        
                        # Synthesizing Speech phase
                        yield yield_progress("Synthesizing Speech")
                        
                        # Generate audio file
                        from scripts.tools import synthesize_text_to_file, play_tts_audio
                        wav_path = synthesize_text_to_file(text)
                        
                        if wav_path and os.path.exists(wav_path):
                            # Playing Audio phase
                            yield yield_progress("Playing Audio")
                            play_tts_audio(wav_path)
                            print(f"[AUTO-TTS] Completed for msg {msg_idx}")
                        else:
                            print("[AUTO-TTS] Synthesis failed")
                    else:
                        print("[AUTO-TTS] Text empty after cleaning")
                else:
                    print("[AUTO-TTS] Last response empty, skipping")
        except Exception as e:
            print(f"[AUTO-TTS] Error: {e}")
            traceback.print_exc()

    # Final yield to reset UI
    yield (
        get_chatbot_output(session_tuples, session_messages),
        session_messages,
        "Ready — response complete",
        gr.update(value="", visible=True),   # Clear and show user input
        gr.update(visible=False),            # Hide progress indicator
        *update_action_buttons("waiting_for_input", True),
        False,          # cancel_flag reset
        [],             # cleared attached_files
        "waiting_for_input",
        llm_state,
        models_loaded_state
    )

def strip_separators(text: str) -> str:
    """Remove decorative separator lines echoed from search context into model output."""
    text = re.sub(r'^\s*[═─━=*_-]{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[═─━]+[^═─━\n]+[═─━]+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def toggle_web_search(current_web_state):
    """Toggle Web Search (comprehensive DDG + deep-fetch mode)."""
    new_web_state = not current_web_state
    web_variant = "primary" if new_web_state else "secondary"
    return (
        new_web_state,
        gr.update(variant=web_variant),
    )

def update_tts_dropdown():
    """Refresh TTS voice dropdown choices based on enabled voices."""
    choices = get_voice_choices()
    # Ensure current value is among new choices; fallback to first available
    current = cfg.TTS_VOICE_NAME
    if current not in choices and choices:
        current = choices[0]
    return gr.update(choices=choices, value=current)

def toggle_tts_speak(current_tts_state):
    """Toggle Auto TTS Speak (automatically speak AI responses when enabled)."""
    new_tts_state = not current_tts_state
    tts_variant = "primary" if new_tts_state else "secondary"
    tts_icon = "🔊" if new_tts_state else "🔇"
    return (
        new_tts_state,
        gr.update(variant=tts_variant, value=tts_icon),
    )


# =============================================================================
# MAIN DISPLAY LAUNCH
# =============================================================================

def launch_display():
    """Launch the Gradio display – Gradio 5.x with Qt6 WebEngine on all platforms."""
    global demo
    import os
    import gradio as gr
    from pathlib import Path
    from launcher import shutdown_program
    from scripts import configure, utility, inference
    from scripts.configure import (
        MODEL_NAME, SESSION_ACTIVE,
        MAX_HISTORY_SLOTS, MAX_ATTACH_SLOTS, SESSION_LOG_HEIGHT,
        MODEL_FOLDER, CONTEXT_SIZE, BATCH_SIZE, TEMPERATURE, REPEAT_PENALTY,
        VRAM_SIZE, SELECTED_GPU, SELECTED_CPU, MLOCK, BACKEND_TYPE,
        ALLOWED_EXTENSIONS, VRAM_OPTIONS, CTX_OPTIONS, BATCH_OPTIONS, TEMP_OPTIONS,
        REPEAT_OPTIONS, HISTORY_SLOT_OPTIONS, SESSION_LOG_HEIGHT_OPTIONS,
        ATTACH_SLOT_OPTIONS, HISTORY_DIR, USER_COLOR, THINK_COLOR, RESPONSE_COLOR,
        context_injector, QT_VERSION
    )

    print(f"[DISPLAY] Qt Version: {QT_VERSION} (PyQt6)")
    print(f"[DISPLAY] Gradio Version: {cfg.GRADIO_VERSION}")
    print(f"[DISPLAY] Graphics Acceleration: {cfg.GRAPHICS_ACCELERATION}")

    initialize_filter_from_config()

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
    .message,
    .message * {
        user-select: text !important;
        -webkit-user-select: text !important;
        -moz-user-select: text !important;
        -ms-user-select: text !important;
    }
    .scrollable {
        user-select: text !important;
        -webkit-user-select: text !important;
    }
    .message {
        -webkit-touch-callout: default !important;
    }
    /* ── Inline per-message action buttons ─────────────────────────────── */
    .cguf-relay {
        position: absolute !important;
        width: 1px !important;
        height: 1px !important;
        min-height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
        border: none !important;
        overflow: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
        z-index: -1 !important;
        clip: rect(0,0,0,0) !important;
    }
    .cguf-actions {
        display: flex;
        justify-content: flex-end;
        gap: 4px;
        margin-top: 5px;
        opacity: 0;
        transition: opacity 0.15s ease;
        pointer-events: none;
    }
    .cguf-hover:hover .cguf-actions {
        opacity: 1;
        pointer-events: auto;
    }
    .cguf-btn {
        background: rgba(38,40,58,0.88);
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 4px;
        color: #9ba3b4;
        cursor: pointer;
        font-size: 11px;
        padding: 2px 7px;
        line-height: 1.4;
        transition: background 0.12s, color 0.1s;
        user-select: none;
        -webkit-user-select: none;
    }
    .cguf-btn:hover {
        background: rgba(68,76,112,0.95);
        border-color: rgba(255,255,255,0.26);
        color: #e0e6f4;
    }
    #cguf-tts-btn button {
        font-size: 16px !important;
        padding: 0 8px !important;
    }
    #cguf-tts-btn.primary button {
        background: rgba(68, 136, 204, 0.3) !important;
        border-color: rgba(68, 136, 204, 0.6) !important;
    }
    #cguf-diag {
        font-family: monospace;
        font-size: 10px;
        color: #777;
        background: #0e0e14;
        padding: 2px 8px;
        border-radius: 3px;
        display: none;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
    }
    .message, .message p, .message span, .message li {
        font-family: ui-sans-serif, system-ui, -apple-system,
                     "Segoe UI Emoji", "Apple Color Emoji",
                     "Noto Color Emoji", "Twemoji Mozilla",
                     sans-serif !important;
    }
    """

    final_css = css_common

    blocks_kwargs = {
        "title": "Chat-Gradio-Gguf",
        "css": final_css.strip()
    }

    with gr.Blocks(**blocks_kwargs) as demo:
        cfg.demo = demo
        model_folder_state = gr.State(cfg.MODEL_FOLDER)

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
            web_search_enabled=gr.State(False),
            has_ai_response=gr.State(False),
            tts_state=gr.State(value="idle"),
            tts_speak_enabled=gr.State(False)
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
                        gr.Markdown("**History**")
                        with gr.Group(visible=True) as history_slots_group:
                            start_new_session_btn = gr.Button("Start New Session..", variant="secondary")
                            buttons["session"] = [gr.Button(f"History Slot {i+1}", variant="huggingface", visible=False)
                                                  for i in range(cfg.MAX_POSSIBLE_HISTORY_SLOTS)]

                    with gr.Column(visible=False, min_width=60, elem_classes=["clean-elements"]) as left_column_collapsed:
                        toggle_button_left_collapsed = gr.Button("<->", variant="secondary")
                        new_session_btn_collapsed = gr.Button("New", variant="secondary")

                    # CENTER
                    with gr.Column(scale=30, elem_classes=["clean-elements"]):
                        conversation_components["session_log"] = gr.Chatbot(
                            label="Session Log",
                            height=cfg.SESSION_LOG_HEIGHT,
                            type="messages",
                            elem_classes=["scrollable"]
                        )

                        initial_max_lines = max(3, int(((cfg.SESSION_LOG_HEIGHT - 100) / 10) / 2.5) - 6)
                        cfg.USER_INPUT_MAX_LINES = initial_max_lines

                        with gr.Row(elem_classes=["clean-elements"]):
                            with gr.Column(scale=10):
                                conversation_components["user_input"] = gr.Textbox(
                                    label="User Input",
                                    lines=3,
                                    max_lines=initial_max_lines,
                                    interactive=True,
                                    placeholder="Type your message here... (model auto-loads on first send)"
                                )
                                conversation_components["progress_indicator"] = gr.Markdown(
                                    value="",
                                    visible=False,
                                    elem_classes=["progress-indicator"]
                                )
                            with gr.Column(scale=1, min_width=55, elem_classes=["clean-elements"]):
                                action_buttons["web_search"] = gr.Button(
                                    "🔍", variant="secondary",
                                    elem_id="cguf-web-btn", min_width=50
                                )
                                action_buttons["tts_speak"] = gr.Button(
                                    "🔇", variant="secondary",
                                    elem_id="cguf-tts-btn", min_width=50
                                )

                        copy_action_box = gr.Textbox(
                            value="", visible=True,
                            elem_id="cguf-copy-action", label="",
                            elem_classes=["cguf-relay"]
                        )
                        tts_action_box = gr.Textbox(
                            value="", visible=True,
                            elem_id="cguf-tts-action", label="",
                            elem_classes=["cguf-relay"]
                        )
                        tts_state_box = gr.Textbox(
                            value="idle", visible=True,
                            elem_id="cguf-tts-state", label="",
                            elem_classes=["cguf-relay"]
                        )
                        edit_action_box = gr.Textbox(
                            value="", visible=True,
                            elem_id="cguf-edit-action", label="",
                            elem_classes=["cguf-relay"]
                        )
                        retry_action_box = gr.Textbox(
                            value="", visible=True,
                            elem_id="cguf-retry-action", label="",
                            elem_classes=["cguf-relay"]
                        )

                        with gr.Row(elem_classes=["clean-elements"]):
                            action_buttons["action"] = gr.Button("Send Input", variant="secondary", elem_classes=["send-button-green"], scale=1)
                            action_buttons["cancel_response"] = gr.Button("..Wait For Response..", variant="primary", scale=1, visible=False)
                        # Hidden off-Row components — compat placeholders for output list lengths
                        action_buttons["edit_previous"] = gr.Button("", variant="secondary", visible=False)
                        action_buttons["copy_response"] = gr.Button("", variant="secondary", visible=False)
                        action_buttons["cancel_input"]  = gr.Button("", variant="primary",   visible=False)

                    # RIGHT PANEL
                    with gr.Column(visible=True, min_width=300, elem_classes=["clean-elements"]) as right_column_expanded:
                        toggle_button_right_expanded = gr.Button(">-------<", variant="secondary")
                        gr.Markdown("**Attachments**")
                        with gr.Group(visible=True) as attach_group:
                            attach_files = gr.UploadButton(
                                "Add Attach Files..",
                                file_types=[f".{ext}" for ext in cfg.ALLOWED_EXTENSIONS],
                                file_count="multiple",
                                variant="secondary"
                            )
                            attach_slots = [gr.Button("Attach Slot Free", variant="huggingface", visible=False)
                                            for _ in range(cfg.MAX_POSSIBLE_ATTACH_SLOTS)]

                    with gr.Column(visible=False, min_width=60, elem_classes=["clean-elements"]) as right_column_collapsed:
                        toggle_button_right_collapsed = gr.Button("<->", variant="secondary")
                        add_attach_files_collapsed = gr.UploadButton(
                            "Add",
                            file_types=[f".{ext}" for ext in cfg.ALLOWED_EXTENSIONS],
                            file_count="multiple",
                            variant="secondary"
                        )

                with gr.Row():
                    interaction_global_status = gr.Textbox(
                        value="Ready", label="Status",
                        interactive=False, max_lines=1, scale=20
                    )
                    exit_interaction = gr.Button("Exit Program", variant="stop", elem_classes=["double-height"], scale=1)

            with gr.Tab("Hardware/TTS/Model Configs"):
                with gr.Group():
                    gr.Markdown("### Hardware Configuration")

                    with gr.Row():
                        backend_display = gr.Textbox(
                            label="Backend Type", value=cfg.BACKEND_TYPE, interactive=False
                        )
                        layer_allocation_radio = gr.Radio(
                            choices=["SRAM_ONLY"] if not cfg.VULKAN_AVAILABLE else ["SRAM_ONLY", "VRAM_SRAM"],
                            label="Layer Allocation Mode",
                            value=cfg.LAYER_ALLOCATION_MODE,
                            interactive=cfg.VULKAN_AVAILABLE
                        )

                    with gr.Row():
                        cpu_select = gr.Dropdown(
                            choices=["Auto-Select"] + [c["label"] for c in get_cpu_info()],
                            label="CPU", value=cfg.SELECTED_CPU,
                            interactive=True, allow_custom_value=True
                        )
                        cpu_threads = gr.Slider(
                            minimum=1, maximum=cfg.CPU_LOGICAL_CORES, step=1,
                            value=cfg.CPU_THREADS or max(1, cfg.CPU_LOGICAL_CORES // 2),
                            label="CPU Threads", interactive=True
                        )

                    gpu_vram_row = gr.Row(visible=cfg.BACKEND_TYPE in ["VULKAN_CPU", "VULKAN_VULKAN"])
                    with gpu_vram_row:
                        gpu_select = gr.Dropdown(
                            choices=get_available_gpus(), label="GPU",
                            value=cfg.SELECTED_GPU, interactive=True, allow_custom_value=True
                        )
                        vram_size = gr.Dropdown(
                            choices=cfg.VRAM_OPTIONS, label="VRAM Allocation (MB)",
                            value=cfg.VRAM_SIZE, interactive=True
                        )

                    with gr.Row():
                        if cfg.PLATFORM == "windows":
                            system_label = "Windows Audio"
                        else:
                            backend = getattr(cfg, 'TTS_AUDIO_BACKEND', 'unknown')
                            system_label = f"Linux Audio ({backend})"

                        sound_output_display = gr.Textbox(
                            label=f"Audio Output ({system_label})",
                            value="Default Sound Device",
                            interactive=False, max_lines=1
                        )
                        sound_sample_rate = gr.Dropdown(
                            choices=[str(r) for r in cfg.SOUND_SAMPLE_RATE_OPTIONS],
                            label="Sample Rate (Hz)",
                            value=str(cfg.SOUND_SAMPLE_RATE), interactive=True
                        )

                with gr.Group():
                    gr.Markdown("### Text-to-Speech (TTS)")
                    with gr.Row():
                        tts_voice = gr.Dropdown(
                            choices=get_voice_choices(), label="TTS Voice",
                            value=cfg.TTS_VOICE_NAME or "Default",
                            interactive=True, allow_custom_value=True
                        )
                        tts_max_len = gr.Slider(
                            label="Max TTS Length (chars)",
                            minimum=500, maximum=4500, step=500,
                            value=cfg.MAX_TTS_LENGTH, interactive=True
                        )

                with gr.Group():
                    gr.Markdown("### Model Configuration")

                    with gr.Row(elem_classes=["model-folder-row"]):
                        model_folder_textbox = gr.Textbox(
                            label="Model Folder", value=cfg.MODEL_FOLDER,
                            interactive=False,
                            placeholder="Click Browse Folder button to select...",
                            scale=5
                        )
                        model_dropdown = gr.Dropdown(
                            choices=get_available_models(), label="Model Selected",
                            value=cfg.MODEL_NAME, interactive=True,
                            elem_classes="model-select", scale=4
                        )
                        model_loaded_indicator = gr.Textbox(
                            label="Model Loaded",
                            value="🟢 SO LOADED" if cfg.MODELS_LOADED else "🔴 NOT LOADED",
                            interactive=False, max_lines=1, scale=3
                        )

                    with gr.Row():
                        ctx_size = gr.Dropdown(
                            choices=cfg.CTX_OPTIONS, label="Context Size",
                            value=cfg.CONTEXT_SIZE, interactive=True
                        )
                        batch_size = gr.Dropdown(
                            choices=cfg.BATCH_OPTIONS, label="Batch Size",
                            value=cfg.BATCH_SIZE, interactive=True
                        )
                        temperature = gr.Dropdown(
                            choices=cfg.TEMP_OPTIONS, label="Temperature",
                            value=cfg.TEMPERATURE, interactive=True
                        )
                        repeat_penalty = gr.Dropdown(
                            choices=cfg.REPEAT_OPTIONS, label="Repeat Penalty",
                            value=cfg.REPEAT_PENALTY, interactive=True
                        )

                    with gr.Row():
                        browse_folder_btn = gr.Button("📁 Browse Folder", scale=1, size="sm")
                        load_model_btn = gr.Button("📥 Load Model", variant="primary", scale=1)
                        unload_model_btn = gr.Button("📤 Unload Model", variant="stop", scale=1)

                gr.Markdown("---")
                with gr.Row():
                    save_config_btn = gr.Button("Save All Configuration", variant="primary", size="lg")
                gr.Markdown("---")
                with gr.Row():
                    config_status = gr.Textbox(
                        value="Configuration loaded", label="Status",
                        interactive=False, max_lines=1, scale=20
                    )
                    exit_config = gr.Button("Exit Program", variant="stop", elem_classes=["double-height"], scale=1)

            with gr.Tab("Interface/Filter Settings"):
                with gr.Group():
                    gr.Markdown("### Program Options")
                    with gr.Row():
                        session_log_height = gr.Dropdown(
                            choices=cfg.SESSION_LOG_HEIGHT_OPTIONS, label="Session Log Height (px)",
                            value=cfg.SESSION_LOG_HEIGHT, interactive=True
                        )
                        max_attach_slots = gr.Dropdown(
                            choices=cfg.ATTACH_SLOT_OPTIONS, label="Max Attachment Slots",
                            value=cfg.MAX_ATTACH_SLOTS, interactive=True
                        )
                        max_history_slots = gr.Dropdown(
                            choices=cfg.HISTORY_SLOT_OPTIONS, label="Max History Slots",
                            value=cfg.MAX_HISTORY_SLOTS, interactive=True
                        )
                        delete_history_btn = gr.Button("Delete All History", variant="stop", size="double")

                    gr.Markdown("### Output Options")
                    with gr.Row():
                        show_think = gr.Checkbox(
                            label="Show Thinking Phase", value=cfg.SHOW_THINK_PHASE, interactive=True
                        )
                        bleep_events = gr.Checkbox(
                            label="Beep on Events", value=cfg.BLEEP_ON_EVENTS, interactive=True
                        )
                        print_raw = gr.Checkbox(
                            label="Print Raw Model Output (debug)", value=cfg.PRINT_RAW_OUTPUT, interactive=True
                        )

                with gr.Group():
                    gr.Markdown("### Filter Settings")
                    with gr.Row():
                        filter_user_btn = gr.Button("User Preset")
                        filter_light_btn = gr.Button("Light Preset")
                    filter_text = gr.Textbox(
                        label="Custom Filter Rules (find→replace pairs)",
                        value=get_filter_text_for_display(),
                        lines=15, interactive=True,
                        placeholder="One rule per line: find_string → replace_string"
                    )

                gr.Markdown("---")
                with gr.Row():
                    save_all_btn = gr.Button("Save All Settings", variant="primary", size="lg")
                gr.Markdown("---")

                with gr.Row():
                    filter_status = gr.Textbox(
                        value="Filter settings loaded", label="Status",
                        interactive=False, max_lines=1, scale=20
                    )
                    exit_filtering = gr.Button("Exit Program", variant="stop", elem_classes=["double-height"], scale=1)

            with gr.Tab("About/Debug Info"):
                with gr.Group():
                    gr.Markdown("### Chat-Gradio-Gguf")
                    gr.HTML("""
                        <p>A Windows/Linux Chatbot using, Gradio and llama.cpp, by <a href="mailto:wiseman-timelord@mail.com">WiseMan-Time-Lord</a> at <a href="http://wisetime.rf.gd/">WiseTime.Rf.Gd</a></p>
                        <p><strong>Where you may find, this and my other, programming projects on </strong> <a href="https://github.com/wiseman-timelord">GitHub</a></p>
                        <p><strong>Support/Donate to assist in the continuation of my projects at, </strong> <a href="https://patreon.com/WiseManTimeLord">Patreon</a>, <a href="https://ko-fi.com/WiseManTimeLord">Ko-Fi</a></p>
                    """, elem_classes=["info-textbox-match"])

                with gr.Group():
                    gr.Markdown("### System Constants (from constants.ini)")
                    ini_display = gr.Textbox(
                        label="INI Values (read-only, set by installer)",
                        value=get_ini_display_text(), lines=11, interactive=False
                    )

                with gr.Group():
                    gr.Markdown("### Runtime Globals (Debug)")
                    debug_display = gr.Textbox(
                        label="Critical Globals (click Refresh to update)",
                        value=get_debug_globals_text(), lines=10, interactive=False
                    )
                    refresh_debug_btn = gr.Button("🔄 Refresh Debug Info", variant="secondary")

                gr.Markdown("---")
                with gr.Row():
                    info_status = gr.Textbox(
                        value="Info tab loaded", label="Status",
                        interactive=False, max_lines=1, scale=20
                    )
                    exit_info = gr.Button("Exit Program", variant="stop", elem_classes=["double-height"], scale=1)


        # ── EVENT HANDLERS ────────────────────────────────────────────────────

        # Layer allocation → VRAM row visibility
        layer_allocation_radio.change(
            fn=lambda mode: gr.update(visible=(mode == "VRAM_SRAM" and cfg.BACKEND_TYPE in ["VULKAN_CPU", "VULKAN_VULKAN"])),
            inputs=[layer_allocation_radio],
            outputs=[vram_size]
        )

        # ── Browse folder button ─────────────────────────────────────────────
        def browse_and_update_folder(current_path):
            """Open folder dialog and update cfg.MODEL_FOLDER."""
            import tkinter as tk
            from tkinter import filedialog

            print(f"[BROWSE] Handler called. current_path='{current_path}'")

            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)

            initial_dir = current_path if Path(current_path).is_dir() else str(Path.home())

            folder_path = filedialog.askdirectory(
                title="Select Folder Containing GGUF Models",
                initialdir=initial_dir,
                mustexist=True
            )
            root.destroy()
            print(f"[BROWSE] Dialog returned: '{folder_path}'")

            if not folder_path:
                print("[BROWSE] Cancelled — no folder chosen.")
                status = "Folder selection cancelled."
                return (current_path, current_path, status, status, status)

            resolved = str(Path(folder_path).resolve())
            cfg.MODEL_FOLDER = resolved
            print(f"[BROWSE] cfg.MODEL_FOLDER set to: {cfg.MODEL_FOLDER}")

            try:
                n = len([f for f in Path(resolved).glob("*.gguf") if "mmproj" not in f.name.lower()])
            except Exception:
                n = 0
            status = f"Folder: {short_path(resolved)} — {n} model(s) found"
            print(f"[BROWSE] Status: {status}")
            return (cfg.MODEL_FOLDER, cfg.MODEL_FOLDER, status, status, status)

        browse_folder_btn.click(
            fn=browse_and_update_folder,
            inputs=[model_folder_textbox],
            outputs=[
                model_folder_textbox, model_folder_state,
                interaction_global_status, config_status, filter_status
            ]
        ).then(
            fn=update_model_list,
            inputs=[model_folder_state],
            outputs=[model_dropdown]
        )

        # Load model button
        load_model_btn.click(
            fn=handle_load_model,
            inputs=[
                model_dropdown, model_folder_textbox, vram_size, ctx_size,
                gpu_select, cpu_select, cpu_threads,
                states["llm"], states["models_loaded"]
            ],
            outputs=[
                states["llm"], states["models_loaded"],
                interaction_global_status, config_status, filter_status,
                conversation_components["user_input"],
                model_loaded_indicator
            ]
        )

        # Unload model button
        unload_model_btn.click(
            fn=handle_unload_model,
            inputs=[states["llm"], states["models_loaded"]],
            outputs=[
                states["llm"], states["models_loaded"],
                interaction_global_status, config_status, filter_status,
                conversation_components["user_input"],
                model_loaded_indicator
            ]
        )

        # ── Main conversation handler ────────────────────────────────────────
        action_buttons["action"].click(
            fn=conversation_display,
            inputs=[
                conversation_components["user_input"],
                conversation_components["session_log"],
                states["session_messages"],
                states["attached_files"],
                states["is_reasoning_model"],
                states["cancel_flag"],
                states["web_search_enabled"],
                states["interaction_phase"],
                states["llm"],
                states["models_loaded"],
                states["has_ai_response"],
                states["tts_speak_enabled"],
            ],
            outputs=[
                conversation_components["session_log"],
                states["session_messages"],
                interaction_global_status,
                conversation_components["user_input"],
                conversation_components["progress_indicator"],
                action_buttons["action"],
                action_buttons["edit_previous"],
                action_buttons["copy_response"],
                action_buttons["cancel_input"],
                action_buttons["cancel_response"],
                states["cancel_flag"],
                states["attached_files"],
                states["interaction_phase"],
                states["llm"],
                states["models_loaded"],
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

        # ── Model dropdown change: unload old → countdown → load new ─────────
        def handle_model_change(new_model, current_model, llm_state, models_loaded_state, model_folder):
            """Handle model selection change: unload → 3s countdown → load new model.
            Outputs: [llm_state, models_loaded, status_text, user_input_update] (4 outputs).
            """
            import time

            if new_model == current_model or new_model in ["Select_a_model...", "No models found", "", None]:
                placeholder = (
                    "Enter text here... (model auto-loads on first send)"
                    if new_model not in ["Select_a_model...", "No models found", "", None]
                    else "Select a valid model first..."
                )
                return (
                    llm_state, models_loaded_state,
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
                    llm_state, models_loaded_state,
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
                    llm_state, models_loaded_state,
                    f"{status_msg}\nLoading new model in {i}...",
                    gr.update(interactive=False)
                )
                time.sleep(1.0)

            # Step 3: Load new model
            yield (
                llm_state, models_loaded_state,
                status_msg + "\nLoading model...",
                gr.update(interactive=False)
            )

            try:
                load_status, loaded, new_llm, _ = load_models(
                    model_folder, new_model, cfg.VRAM_SIZE,
                    llm_state, models_loaded_state
                )

                if loaded:
                    cfg.MODELS_LOADED = True
                    cfg.llm = new_llm
                    cfg.LOADED_CONTEXT_SIZE = cfg.CONTEXT_SIZE

                    yield (
                        new_llm, True,
                        f"✅ Model loaded: {new_model}",
                        gr.update(interactive=True,
                                  placeholder="Enter text here... (model auto-loads on first send)")
                    )
                else:
                    yield (
                        llm_state, False,
                        f"❌ Model load failed: {load_status[:150]}",
                        gr.update(interactive=False, placeholder="Load failed — try again.")
                    )

            except Exception as e:
                status_msg += f"\nError: {str(e)[:120]}"
                yield (
                    llm_state, False,
                    status_msg,
                    gr.update(interactive=False, placeholder="Select a valid model first...")
                )

        model_dropdown.change(
            fn=handle_model_change,
            inputs=[
                model_dropdown, model_dropdown,
                states["llm"], states["models_loaded"],
                model_folder_textbox
            ],
            outputs=[
                states["llm"], states["models_loaded"],
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

        # TTS handlers
        tts_voice.change(
            fn=update_tts_voice,
            inputs=[tts_voice],
            outputs=[config_status]
        )

        tts_max_len.change(
            fn=lambda val: setattr(cfg, 'MAX_TTS_LENGTH', int(val)) or f"Max TTS length set to {int(val)} chars",
            inputs=[tts_max_len],
            outputs=[config_status]
        )

        # Settings checkboxes
        show_think.change(
            fn=lambda x: setattr(cfg, 'SHOW_THINK_PHASE', x) or f"Show thinking phase: {'ON' if x else 'OFF'}",
            inputs=[show_think],
            outputs=[interaction_global_status]
        )
        bleep_events.change(
            fn=lambda x: setattr(cfg, 'BLEEP_ON_EVENTS', x) or f"Beep on events: {'ON' if x else 'OFF'}",
            inputs=[bleep_events],
            outputs=[interaction_global_status]
        )
        print_raw.change(
            fn=lambda x: setattr(cfg, 'PRINT_RAW_OUTPUT', x) or f"Print raw output: {'ON' if x else 'OFF'}",
            inputs=[print_raw],
            outputs=[interaction_global_status]
        )
        session_log_height.change(
            fn=lambda val: (
                setattr(cfg, 'SESSION_LOG_HEIGHT', int(val)),
                f"Session log height set to {int(val)}px (save & restart to apply)"
            )[1],
            inputs=[session_log_height],
            outputs=[interaction_global_status]
        )
        max_attach_slots.change(
            fn=lambda val: (
                setattr(cfg, 'MAX_ATTACH_SLOTS', int(val)),
                f"Max attachment slots set to {int(val)} (save & restart to apply)"
            )[1],
            inputs=[max_attach_slots],
            outputs=[interaction_global_status]
        )
        max_history_slots.change(
            fn=lambda val: (
                setattr(cfg, 'MAX_HISTORY_SLOTS', int(val)),
                f"Max history slots set to {int(val)} (save & restart to apply)"
            )[1],
            inputs=[max_history_slots],
            outputs=[interaction_global_status]
        )

        # CPU threads — inline lambda (no separate function needed)
        cpu_threads.change(
            fn=lambda val: (setattr(cfg, 'CPU_THREADS', int(val)), f"CPU threads set to {int(val)}")[1],
            inputs=[cpu_threads],
            outputs=[config_status]
        )

        # Exit buttons
        for _exit_btn in (exit_interaction, exit_config, exit_filtering, exit_info):
            _exit_btn.click(
                fn=shutdown_program,
                inputs=[states["llm"], states["models_loaded"],
                        states["session_messages"], states["attached_files"]],
                outputs=[]
            )

        # Refresh debug info
        def refresh_debug_info():
            return get_ini_display_text(), get_debug_globals_text(), "Debug info refreshed"

        refresh_debug_btn.click(
            fn=refresh_debug_info,
            inputs=[],
            outputs=[ini_display, debug_display, info_status]
        )

        # Panel expand/collapse
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

        # Web Search toggle
        action_buttons["web_search"].click(
            fn=toggle_web_search,
            inputs=[states["web_search_enabled"]],
            outputs=[states["web_search_enabled"], action_buttons["web_search"]]
        )

        # TTS Speak toggle button
        action_buttons["tts_speak"].click(
            fn=toggle_tts_speak,
            inputs=[states["tts_speak_enabled"]],
            outputs=[states["tts_speak_enabled"], action_buttons["tts_speak"]]
        )

        # Inline copy
        copy_action_box.change(
            fn=handle_inline_copy,
            inputs=[copy_action_box, states["session_messages"]],
            outputs=[interaction_global_status, copy_action_box]
        )

        # Inline TTS
        tts_action_box.change(
            fn=handle_inline_tts,
            inputs=[tts_action_box, states["session_messages"]],
            outputs=[tts_state_box]
        )

        # Inline edit
        edit_action_box.change(
            fn=handle_inline_edit,
            inputs=[edit_action_box, states["session_messages"]],
            outputs=[
                conversation_components["user_input"],
                conversation_components["session_log"],
                states["session_messages"],
                interaction_global_status,
                states["has_ai_response"],
                edit_action_box,
            ]
        ).then(
            fn=lambda has_ai: update_action_buttons("waiting_for_input", has_ai),
            inputs=[states["has_ai_response"]],
            outputs=[
                action_buttons["action"],
                action_buttons["edit_previous"],
                action_buttons["copy_response"],
                action_buttons["cancel_input"],
                action_buttons["cancel_response"],
            ]
        )

        # Inline retry
        retry_action_box.change(
            fn=handle_inline_retry,
            inputs=[retry_action_box, states["session_messages"]],
            outputs=[
                conversation_components["user_input"],
                conversation_components["session_log"],
                states["session_messages"],
                interaction_global_status,
                states["has_ai_response"],
                retry_action_box,
            ]
        ).then(
            fn=conversation_display,
            inputs=[
                conversation_components["user_input"],
                conversation_components["session_log"],
                states["session_messages"],
                states["attached_files"],
                states["is_reasoning_model"],
                states["cancel_flag"],
                states["web_search_enabled"],
                states["interaction_phase"],
                states["llm"],
                states["models_loaded"],
                states["has_ai_response"],
                states["tts_speak_enabled"],
            ],
            outputs=[
                conversation_components["session_log"],
                states["session_messages"],
                interaction_global_status,
                conversation_components["user_input"],
                conversation_components["progress_indicator"],
                action_buttons["action"],
                action_buttons["edit_previous"],
                action_buttons["copy_response"],
                action_buttons["cancel_input"],
                action_buttons["cancel_response"],
                states["cancel_flag"],
                states["attached_files"],
                states["interaction_phase"],
                states["llm"],
                states["models_loaded"],
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

        # ── Unified save handler ─────────────────────────────────────────────
        def unified_save_wrapper(
            layer_mode, cpu, cpu_threads_val, gpu, vram, sound_device, sample_rate,
            model_folder_val, model, ctx, batch, temp, repeat,
            tts_voice_val, tts_max_len_val,
            show_think_val, bleep_val, print_raw_val,
            max_hist, max_att, log_height,
            filter_text_val
        ):
            cfg.LAYER_ALLOCATION_MODE = layer_mode       if layer_mode       is not None else cfg.LAYER_ALLOCATION_MODE
            cfg.SELECTED_CPU          = cpu              if cpu              is not None else cfg.SELECTED_CPU
            cfg.CPU_THREADS           = int(cpu_threads_val) if cpu_threads_val is not None else cfg.CPU_THREADS
            cfg.SELECTED_GPU          = gpu              if gpu              is not None else cfg.SELECTED_GPU
            cfg.VRAM_SIZE             = int(vram)        if vram             is not None else cfg.VRAM_SIZE
            cfg.SOUND_OUTPUT_DEVICE   = sound_device     if sound_device     is not None else cfg.SOUND_OUTPUT_DEVICE
            cfg.SOUND_SAMPLE_RATE     = int(sample_rate) if sample_rate      is not None else cfg.SOUND_SAMPLE_RATE
            cfg.MODEL_FOLDER          = model_folder_val if model_folder_val is not None else cfg.MODEL_FOLDER
            cfg.MODEL_NAME            = model            if model            is not None else cfg.MODEL_NAME
            cfg.CONTEXT_SIZE          = int(ctx)         if ctx              is not None else cfg.CONTEXT_SIZE
            cfg.BATCH_SIZE            = int(batch)       if batch            is not None else cfg.BATCH_SIZE
            cfg.TEMPERATURE           = float(temp)      if temp             is not None else cfg.TEMPERATURE
            cfg.REPEAT_PENALTY        = float(repeat)    if repeat           is not None else cfg.REPEAT_PENALTY
            cfg.TTS_VOICE_NAME        = tts_voice_val    if tts_voice_val    is not None else cfg.TTS_VOICE_NAME
            if tts_voice_val and tts_voice_val != "Default":
                cfg.TTS_VOICE = get_voice_id_by_name(tts_voice_val)
                _tts_ok, _tts_warn = verify_tts_voice(cfg.TTS_VOICE)
                if not _tts_ok:
                    print(f"[TTS] WARNING: {_tts_warn}")
            cfg.MAX_TTS_LENGTH        = int(tts_max_len_val) if tts_max_len_val is not None else cfg.MAX_TTS_LENGTH
            cfg.SHOW_THINK_PHASE      = bool(show_think_val) if show_think_val is not None else cfg.SHOW_THINK_PHASE
            cfg.BLEEP_ON_EVENTS       = bool(bleep_val)  if bleep_val        is not None else cfg.BLEEP_ON_EVENTS
            cfg.PRINT_RAW_OUTPUT      = bool(print_raw_val) if print_raw_val is not None else cfg.PRINT_RAW_OUTPUT
            cfg.MAX_HISTORY_SLOTS     = int(max_hist)    if max_hist         is not None else cfg.MAX_HISTORY_SLOTS
            cfg.MAX_ATTACH_SLOTS      = int(max_att)     if max_att          is not None else cfg.MAX_ATTACH_SLOTS
            cfg.SESSION_LOG_HEIGHT    = int(log_height)  if log_height       is not None else cfg.SESSION_LOG_HEIGHT

            result = save_all_settings()
            if filter_text_val and filter_text_val.strip():
                filter_result = save_custom_filter(filter_text_val)
                result += f"\n{filter_result}"
            return result, result, result, result

        _save_inputs = [
            layer_allocation_radio, cpu_select, cpu_threads, gpu_select, vram_size,
            sound_output_display, sound_sample_rate,
            model_folder_textbox, model_dropdown,
            ctx_size, batch_size, temperature, repeat_penalty,
            tts_voice, tts_max_len,
            show_think, bleep_events, print_raw,
            max_history_slots, max_attach_slots, session_log_height,
            filter_text
        ]
        _save_outputs = [interaction_global_status, config_status, filter_status, info_status]

        save_config_btn.click(fn=unified_save_wrapper, inputs=_save_inputs, outputs=_save_outputs)
        save_all_btn.click(fn=unified_save_wrapper, inputs=_save_inputs, outputs=_save_outputs)

        # Attach files handlers
        attach_files.upload(
            fn=utility.process_attach_files,
            inputs=[attach_files, states["attached_files"]],
            outputs=[interaction_global_status, states["attached_files"]]
        ).then(
            fn=lambda files: utility.update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        )

        add_attach_files_collapsed.upload(
            fn=utility.process_attach_files,
            inputs=[add_attach_files_collapsed, states["attached_files"]],
            outputs=[interaction_global_status, states["attached_files"]]
        ).then(
            fn=lambda files: utility.update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        )

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

        # Filter tab handlers
        def load_user_filter():
            text, status = load_filter_preset("User")
            return text, status

        def load_light_filter():
            text, status = load_filter_preset("Light")
            return text, status

        filter_user_btn.click(fn=load_user_filter, inputs=[], outputs=[filter_text, filter_status])
        filter_light_btn.click(fn=load_light_filter, inputs=[], outputs=[filter_text, filter_status])

        delete_history_btn.click(
            fn=delete_all_sessions,
            inputs=[],
            outputs=[filter_status]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        # Initial load chain
        demo.load(
            fn=lambda: cfg.MODEL_FOLDER,
            inputs=[],
            outputs=[model_folder_state]
        ).then(
            fn=update_model_list,
            inputs=[model_folder_state],
            outputs=[model_dropdown]
        ).then(
            fn=update_cpu_select,
            inputs=[],
            outputs=[cpu_select]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        ).then(
            fn=lambda model_name: gr.update(
                interactive=(model_name not in ["Select_a_model...", "No models found", None, ""]),
                placeholder="Enter text here..." if model_name not in ["Select_a_model...", "No models found", None, ""]
                            else "Select a valid model first..."
            ),
            inputs=[model_dropdown],
            outputs=[conversation_components["user_input"]]
        ).then(
            fn=update_tts_dropdown,
            inputs=[],
            outputs=[tts_voice]
        )

        # JS injection
        _CGUF_JS = """
() => {
    var CGUF_DIAG = false;

    function log(msg) {
        if (!CGUF_DIAG) return;
        var el = document.getElementById('cguf-diag');
        if (el) { el.style.display = 'block'; el.textContent = '[CGUF] ' + msg; }
        console.log('[CGUF]', msg);
    }

    window.cgufFire = function(elemId, value) {
        var wrap = document.getElementById(elemId);
        if (!wrap) { log('ERROR: #' + elemId + ' not in DOM'); return; }
        var inp = wrap.querySelector('textarea') ||
                  wrap.querySelector('input[type="text"]') ||
                  wrap.querySelector('input');
        if (!inp) { log('ERROR: no input inside #' + elemId); return; }
        try {
            var unique = value + '|' + Date.now();
            var proto = inp.tagName === 'TEXTAREA'
                ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
            var setter = Object.getOwnPropertyDescriptor(proto, 'value').set;
            setter.call(inp, unique);
            inp.dispatchEvent(new Event('input',  { bubbles: true }));
            inp.dispatchEvent(new Event('change', { bubbles: true }));
            inp.dispatchEvent(new Event('blur',   { bubbles: true }));
            log('OK ' + elemId + '=' + unique);
        } catch(e) {
            log('ERROR setter: ' + e.message);
        }
    };

    function makeBtn(emoji, title, elemId, payload) {
        var b = document.createElement('button');
        b.className = 'cguf-btn';
        b.textContent = emoji;
        b.title = title;
        b.type = 'button';
        b.addEventListener('click', function(e) {
            e.stopPropagation();
            e.preventDefault();
            window.cgufFire(elemId, payload);
        });
        return b;
    }

    var SEL_PAIRS = [
        ['[data-testid="user"]',                     '[data-testid="bot"]'],
        ['.message-wrap [data-testid="user"]',        '.message-wrap [data-testid="bot"]'],
        ['.message.user',                            '.message.bot'],
        ['div[class*="user"][class*="message"]',     'div[class*="bot"][class*="message"]'],
    ];

    function findMessages() {
        for (var i = 0; i < SEL_PAIRS.length; i++) {
            var us = document.querySelectorAll(SEL_PAIRS[i][0]);
            var bs = document.querySelectorAll(SEL_PAIRS[i][1]);
            if (us.length + bs.length > 0) {
                log('sel[' + i + '] u=' + us.length + ' b=' + bs.length +
                    ' [' + SEL_PAIRS[i][0] + ']');
                return { users: Array.from(us), bots: Array.from(bs) };
            }
        }
        var root = document.querySelector('.message-wrap') ||
                   document.querySelector('.scrollable') ||
                   document.querySelector('.wrap');
        if (root) {
            var dump = Array.from(root.querySelectorAll('*')).slice(0, 25)
                .map(function(el) {
                    return el.tagName.toLowerCase() +
                           (el.className ? ('.' + String(el.className)
                               .split(' ').join('.').slice(0, 60)) : '');
                }).join(' | ');
            log('No msgs. DOM dump: ' + dump.slice(0, 280));
        } else {
            log('.message-wrap/.scrollable/.wrap not found in DOM');
        }
        return { users: [], bots: [] };
    }

    function injectButtons() {
        var res = findMessages();

        res.users.forEach(function(msg, i) {
            if (msg.dataset.cguf) return;
            msg.dataset.cguf = '1';
            msg.classList.add('cguf-hover');
            msg.style.position = 'relative';
            var row = document.createElement('div');
            row.className = 'cguf-actions';
            row.appendChild(makeBtn('📋', 'Copy user message',
                                    'cguf-copy-action', 'user:' + i));
            row.appendChild(makeBtn('✏', 'Edit from here (removes later messages)',
                                    'cguf-edit-action', String(i)));
            msg.appendChild(row);
        });

        res.bots.forEach(function(msg, i) {
            if (msg.dataset.cguf) return;
            msg.dataset.cguf = '1';
            msg.classList.add('cguf-hover');
            msg.style.position = 'relative';
            var row = document.createElement('div');
            row.className = 'cguf-actions';
            row.appendChild(makeBtn('📋', 'Copy AI response',
                                    'cguf-copy-action', 'bot:' + i));
            row.appendChild(makeBtn('↻', 'Re-generate this response',
                                    'cguf-retry-action', 'bot:' + i));
            var ttsBtn = document.createElement('button');
            ttsBtn.className = 'cguf-btn cguf-tts-btn';
            ttsBtn.textContent = '▶';
            ttsBtn.title = 'Play Text-to-Speech';
            ttsBtn.type = 'button';
            ttsBtn.id = 'cguf-tts-btn-' + i;
            ttsBtn.dataset.ttsPhase = 'play';
            ttsBtn.addEventListener('click', function(e) {
                e.stopPropagation();
                e.preventDefault();
                var phase = ttsBtn.dataset.ttsPhase;
                if (phase === 'play') {
                    window.cgufFire('cguf-tts-action', 'bot:' + i);
                    ttsBtn.textContent = '⏳';
                    ttsBtn.title = 'Generating audio...';
                    ttsBtn.dataset.ttsPhase = 'busy';
                } else if (phase === 'stop') {
                    window.cgufFire('cguf-tts-action', 'stop:' + i);
                }
            });
            row.appendChild(ttsBtn);
            msg.appendChild(row);
        });
    }

    var _deb = null;
    var obs = new MutationObserver(function() {
        clearTimeout(_deb);
        _deb = setTimeout(injectButtons, 250);
    });
    obs.observe(document.body, { childList: true, subtree: true });

    (function pollTtsState() {
        var wrap = document.getElementById('cguf-tts-state');
        if (wrap) {
            var inp = wrap.querySelector('textarea') || wrap.querySelector('input');
            if (inp && inp.value) {
                var val = inp.value.trim();
                if (val === 'idle') {
                    document.querySelectorAll('.cguf-tts-btn').forEach(function(btn) {
                        btn.textContent = '▶';
                        btn.title = 'Play Text-to-Speech';
                        btn.dataset.ttsPhase = 'play';
                    });
                } else {
                    var parts = val.split('|');
                    if (parts.length === 2) {
                        var idx = parts[0];
                        var phase = parts[1];
                        var target = document.getElementById('cguf-tts-btn-' + idx);
                        if (target) {
                            if (phase === 'generating') {
                                target.textContent = '⏳';
                                target.title = 'Generating audio...';
                                target.dataset.ttsPhase = 'busy';
                            } else if (phase === 'playing') {
                                target.textContent = '⏹';
                                target.title = 'Stop playback';
                                target.dataset.ttsPhase = 'stop';
                            } else if (phase === 'idle') {
                                target.textContent = '▶';
                                target.title = 'Play Text-to-Speech';
                                target.dataset.ttsPhase = 'play';
                            }
                        }
                    }
                }
            }
        }
        setTimeout(pollTtsState, 500);
    })();

    setTimeout(injectButtons, 800);
    setTimeout(injectButtons, 2000);
    setTimeout(injectButtons, 4000);

    (function setSearchTooltips() {
        function applyTooltips() {
            var web = document.querySelector('#cguf-web-btn button');
            if (web) web.title = 'Web Search';
            var tts = document.querySelector('#cguf-tts-btn button');
            if (tts) tts.title = 'Auto TTS Speak (toggle ON to automatically speak AI responses)';
            if (!web || !tts) setTimeout(applyTooltips, 500);
        }
        applyTooltips();
    })();

    log('injector ready — waiting for messages');
    return [];
}
"""
        demo.load(fn=None, inputs=[], outputs=[], js=_CGUF_JS)

        # TTS heartbeat
        tts_timer = gr.Timer(value=1.0, active=True)
        tts_timer.tick(fn=tts_heartbeat, inputs=[], outputs=[tts_state_box])

        # Start New Session buttons (expanded + collapsed)
        _new_session_outputs = [
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
            states["models_loaded"],
        ]
        _new_session_inputs = [
            states["session_messages"],
            states["attached_files"],
            states["llm"],
            states["models_loaded"],
        ]

        for _btn in (start_new_session_btn, new_session_btn_collapsed):
            _btn.click(
                fn=start_new_session,
                inputs=_new_session_inputs,
                outputs=_new_session_outputs,
            ).then(
                fn=update_session_buttons,
                inputs=[],
                outputs=buttons["session"],
            ).then(
                fn=lambda: update_file_slot_ui([], True),
                inputs=[],
                outputs=attach_slots + [attach_files],
            )

        # Session history buttons
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
    print("[BROWSER] Starting Gradio server in background...")
    demo.queue()

    launch_kwargs = {
        "server_name": "localhost",
        "server_port": 7860,
        "show_error": True,
        "share": False,
        "inbrowser": False,
    }

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