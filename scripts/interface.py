# Script: `.\scripts\interface.py`
# Compatible with Gradio 3.50.2 for Qt5 WebEngine (Windows 7-8.1) support

# Imports...
import gradio as gr
import os, re, json, pyperclip, random, asyncio, queue, threading, spacy, time
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from queue import Queue
import scripts.temporary as temporary
import scripts.settings as settings
from scripts.settings import save_config
from scripts.temporary import (
    MODEL_NAME, SESSION_ACTIVE,
    MAX_HISTORY_SLOTS, MAX_ATTACH_SLOTS, SESSION_LOG_HEIGHT, 
    MODEL_FOLDER, CONTEXT_SIZE, BATCH_SIZE, TEMPERATURE, REPEAT_PENALTY,
    VRAM_SIZE, SELECTED_GPU, SELECTED_CPU, MLOCK, BACKEND_TYPE,
    ALLOWED_EXTENSIONS, VRAM_OPTIONS, CTX_OPTIONS, BATCH_OPTIONS, TEMP_OPTIONS,
    REPEAT_OPTIONS, HISTORY_SLOT_OPTIONS, SESSION_LOG_HEIGHT_OPTIONS,
    ATTACH_SLOT_OPTIONS, HISTORY_DIR, USER_COLOR, THINK_COLOR, RESPONSE_COLOR
)
from scripts import utility
from scripts.utility import (
    web_search, get_saved_sessions, get_cpu_info,
    load_session_history, save_session_history,
    get_available_gpus, filter_operational_content, speak_text, process_files,
    summarize_session, beep
)
from scripts.models import (
    get_response_stream, get_available_models, unload_models, get_model_settings, inspect_model, load_models
)

# ============================================================================
# GRADIO 3.x COMPATIBILITY LAYER
# ============================================================================
# Gradio 3.x Chatbot uses list of tuples: [(user_msg, bot_msg), ...]
# Gradio 4.x Chatbot uses list of dicts: [{"role": "user", "content": msg}, ...]
# 
# We maintain internal state as message dicts for compatibility with models.py
# but convert to/from tuple format for the Chatbot component.
# ============================================================================

def messages_to_tuples(messages):
    """Convert internal message dicts → Gradio tuple format with nice labels"""
    if not messages:
        return []
    
    tuples = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        # Clean content to remove any existing AI-Chat: prefixes before adding our own
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
            # Rare orphan assistant message - clean and format
            content = re.sub(r'^AI-Chat:\s*\n?', '', content, flags=re.MULTILINE).strip()
            display_content = f"AI-Chat:\n{content}" if content else "AI-Chat:"
            tuples.append((None, display_content))
            
        i += 1
    
    return tuples

def tuples_to_messages(tuples):
    """Convert Gradio 3.x tuple format back to message dict list."""
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

# Functions...
def get_panel_choices(model_settings):
    """Determine available panel choices based on model settings."""
    choices = ["History", "Attachments"]
    if model_settings.get("is_nsfw", False) or model_settings.get("is_roleplay", False):
        if "Attachments" in choices:
            choices.remove("Attachments")
    return choices

def update_panel_choices(model_settings, current_panel):
    """Update panel_toggle choices and ensure a valid selection."""
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

def process_attach_files(files, attached_files, models_loaded):
    if not models_loaded:
        return "Error: Load model first.", attached_files
    return process_files(files, attached_files, temporary.MAX_ATTACH_SLOTS, is_attach=True)

def process_vector_files(files, vector_files, models_loaded):
    if not models_loaded:
        return "Error: Load model first.", vector_files
    return process_files(files, vector_files, temporary.MAX_ATTACH_SLOTS, is_attach=False)

def update_config_settings(ctx, batch, temp, repeat, vram, gpu, cpu, model, print_raw):
    temporary.CONTEXT_SIZE = int(ctx)
    temporary.BATCH_SIZE = int(batch)
    temporary.TEMPERATURE = float(temp)
    temporary.REPEAT_PENALTY = float(repeat)
    temporary.VRAM_SIZE = int(vram)
    temporary.SELECTED_GPU = gpu
    temporary.SELECTED_CPU = cpu
    temporary.MODEL_NAME = model
    temporary.PRINT_RAW_OUTPUT = bool(print_raw)
    status_message = (
        f"Updated settings: Context Size={ctx}, Batch Size={batch}, "
        f"Temperature={temp}, Repeat Penalty={repeat}, VRAM Size={vram}, "
        f"Selected GPU={gpu}, CPU={cpu}, Model={model}"
    )
    return status_message

def update_stream_output(stream_output_value):
    temporary.STREAM_OUTPUT = stream_output_value
    status_message = "Stream output enabled." if stream_output_value else "Stream output disabled."
    return status_message

def save_all_settings():
    """Save all configuration settings and return a status message."""
    settings.save_config()
    return temporary.STATUS_MESSAGES["config_saved"]

def set_session_log_base_height(new_height):
    """Set the base session log height from the Configuration page dropdown."""
    temporary.SESSION_LOG_HEIGHT = int(new_height)
    return gr.update(height=temporary.SESSION_LOG_HEIGHT)

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
    max_lines = temporary.USER_INPUT_MAX_LINES
    if lines <= initial_lines:
        adjustment = 0
    else:
        effective_extra_lines = min(lines - initial_lines, max_lines - initial_lines)
        adjustment = effective_extra_lines * 20
    new_height = max(100, temporary.SESSION_LOG_HEIGHT - adjustment)
    return gr.update(height=new_height)

def format_response(output: str) -> str:
    """Format response with thinking phase detection and code highlighting."""
    import re
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import HtmlFormatter
    from scripts.temporary import THINK_COLOR
    
    formatted = []
    
    # Extract think blocks
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
    
    # Remove thinking content
    clean_output = output
    clean_output = re.sub(r'<think>.*?</think>', '', clean_output, flags=re.DOTALL)
    clean_output = re.sub(r'<\|channel\|>analysis.*?(?:<\|end\|>.*?<\|channel\|>final|<\|end\|><\|start\|>assistant<\|end\|><\|start\|>assistant<\|message\|>)',
                         '', clean_output, flags=re.DOTALL)
    clean_output = re.sub(r'<\|[^>]+\|>', '', clean_output)
    
    # Remove "Thinking...." lines
    lines = clean_output.split('\n')
    filtered_lines = [line for line in lines 
                     if not (line.strip().startswith("Thinking") and 
                            all(c in '.… ' for c in line.strip()[8:]))]
    clean_output = '\n'.join(filtered_lines)
    
    # Process code blocks
    code_blocks = re.findall(r'```(\w+)?\n(.*?)```', clean_output, re.DOTALL)
    for lang, code in code_blocks:
        if lang:
            try:
                lexer = get_lexer_by_name(lang, stripall=True)
                formatted_code = highlight(code, lexer, HtmlFormatter())
                clean_output = clean_output.replace(f'```{lang}\n{code}```', formatted_code)
            except:
                pass
    
    # Clean up whitespace - AGGRESSIVE fix for Qt WebEngine double-spacing
    clean_output = clean_output.replace('\r\n', '\n')
    clean_output = clean_output.replace('\r', '\n')
    # Collapse any run of 2+ blank lines to exactly 1 blank line
    clean_output = re.sub(r'\n\s*\n', '\n\n', clean_output)  # Normalize blank lines first
    clean_output = re.sub(r'\n{2,}', '\n\n', clean_output)   # Then collapse to max 2 newlines
    clean_output = clean_output.strip()
    
    if formatted:
        return '\n'.join(formatted) + '\n\n' + clean_output
    return clean_output

def get_initial_model_value():
    """Get initial model selection with proper fallback."""
    available_models = temporary.AVAILABLE_MODELS or get_available_models()
    base_choices = ["Select_a_model..."]
    
    if available_models and available_models != base_choices:
        available_models = [m for m in available_models if m not in base_choices]
        available_models = base_choices + available_models
    else:
        available_models = base_choices
    
    if temporary.MODEL_NAME in available_models and temporary.MODEL_NAME not in base_choices:
        default_model = temporary.MODEL_NAME
    elif len(available_models) > 2:
        default_model = available_models[2]
    else:
        default_model = base_choices[0]
    
    is_reasoning = (
        get_model_settings(default_model)["is_reasoning"] 
        if default_model not in base_choices else False
    )
    
    return default_model, is_reasoning

def update_model_list(new_dir):
    print(f"Updating model list with new_dir: {new_dir}")
    temporary.MODEL_FOLDER = new_dir
    choices = get_available_models()
    if choices and choices[0] != "Select_a_model...":
        value = choices[0]
    else:
        choices = ["Select_a_model..."]
        value = "Select_a_model..."
    print(f"Choices returned: {choices}, Setting value to: {value}")
    return gr.update(choices=choices, value=value)

def handle_model_selection(model_name, model_folder_state):
    if not model_name:
        return model_folder_state, model_name, "No model selected."
    return model_folder_state, model_name, f"Selected model: {model_name}"

def browse_on_click(current_path):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.update_idletasks()
    folder_selected = filedialog.askdirectory(initialdir=current_path or os.path.expanduser("~"))
    root.attributes('-topmost', False)
    root.destroy()
    return folder_selected if folder_selected else current_path

def web_search_trigger(query):
    try:
        result = utility.web_search(query)
        return result if result else "No results found"
    except Exception as e:
        return f"Error: {str(e)}"

def save_rp_settings(rp_location, user_name, user_role, ai_npc, ai_npc_role):
    from scripts import temporary
    temporary.RP_LOCATION = rp_location
    temporary.USER_PC_NAME = user_name
    temporary.USER_PC_ROLE = user_role
    temporary.AI_NPC_NAME = ai_npc
    temporary.AI_NPC_ROLE = ai_npc_role
    settings.save_config()
    return (
        rp_location, user_name, user_role, ai_npc, ai_npc_role,
        rp_location, user_name, user_role, ai_npc, ai_npc_role
    )

def process_uploaded_files(files, loaded_files, models_loaded):
    from scripts.utility import create_session_vectorstore
    import scripts.temporary as temporary
    import os
    print("Uploaded files:", files)
    if not models_loaded:
        return "Error: Load a model first.", loaded_files
    
    max_files = temporary.MAX_ATTACH_SLOTS
    if len(loaded_files) >= max_files:
        return f"Max files ({max_files}) reached.", loaded_files
    
    new_files = [f for f in files if os.path.isfile(f) and f not in loaded_files]
    print("New files to add:", new_files)
    available_slots = max_files - len(loaded_files)
    for file in reversed(new_files[:available_slots]):
        loaded_files.insert(0, file)
    
    session_vectorstore = create_session_vectorstore(loaded_files)
    context_injector.set_session_vectorstore(session_vectorstore)
    
    print("Updated loaded_files:", loaded_files)
    return f"Processed {min(len(new_files), available_slots)} new files.", loaded_files

def eject_file(file_list, slot_index, is_attach=True):
    if 0 <= slot_index < len(file_list):
        removed_file = file_list.pop(slot_index)
        if is_attach:
            temporary.session_attached_files = file_list
        else:
            temporary.session_vector_files = file_list
            session_vectorstore = utility.create_session_vectorstore(file_list, temporary.current_session_id)
            context_injector.set_session_vectorstore(session_vectorstore)
        status_msg = f"Ejected {Path(removed_file).name}"
    else:
        status_msg = "No file to eject"
    updates = update_file_slot_ui(file_list, is_attach)
    return [file_list, status_msg] + updates

def start_new_session(session_messages, attached_files):
    """
    Start a fresh session.
    - Saves current session if it exists and has content
    - Resets all session state
    - Returns updated UI components
    """
    import scripts.temporary as temporary
    from scripts.utility import save_session_history

    # 1. Save current session if it was active and has messages
    if temporary.SESSION_ACTIVE and session_messages:
        try:
            save_session_history(session_messages, attached_files)
            print("[SESSION] Previous session auto-saved before starting new one")
        except Exception as e:
            print(f"[SESSION] Failed to save previous session: {e}")

    # 2. Reset session state
    temporary.SESSION_ACTIVE = False
    temporary.current_session_id = None
    temporary.session_label = ""
    temporary.session_attached_files = []  # Clear attachments

    # 3. Return values matching your button's outputs
    return (
        [],                                 # session_log (tuples) → empty
        [],                                 # session_messages → empty list
        [],                                 # attached_files → empty
        "New session started.",             # interaction_global_status
        "New session started.",             # config_global_status (if you use separate status)
        "",                                 # user_input → cleared
        False,                              # web_search_enabled → default off (adjust if needed)
        False,                              # has_ai_response → false for new session
        *update_action_buttons("waiting_for_input", False)  # action buttons updates
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
        return [], [], [], "No session found.", False
    
    filename = saved_sessions[idx]
    session_id, label, history, attached_files = utility.load_session_history(filename)  # Updated to receive attached_files
    
    if not session_id:
        return [], [], [], "Error loading session.", False
    
    temporary.current_session_id = session_id
    temporary.session_label = label
    temporary.session_attached_files = attached_files  # Set loaded attachments
    temporary.SESSION_ACTIVE = True
    
    # Check if session has AI response
    has_ai = any(msg.get('role') == 'assistant' for msg in history)
    
    status = f"Loaded session: {label}"
    return messages_to_tuples(history), history, attached_files, status, has_ai

def edit_previous_prompt(session_tuples, session_messages):
    """Remove the last exchange and put the user's message back in the input box for editing."""
    if not session_messages or len(session_messages) < 2:
        return session_tuples, session_messages, "", "No previous message to edit.", False
    
    # Get the last user message
    last_user_msg = ""
    
    # Find and remove the last user-assistant pair
    # session_messages is [..., {user}, {assistant}]
    if session_messages[-1].get('role') == 'assistant':
        session_messages = session_messages[:-1]  # Remove assistant
    if session_messages and session_messages[-1].get('role') == 'user':
        last_user_msg = session_messages[-1].get('content', '')
        session_messages = session_messages[:-1]  # Remove user
    
    # Update tuples - remove last tuple
    if session_tuples:
        session_tuples = session_tuples[:-1]
    
    has_ai_response = len([m for m in session_messages if m.get('role') == 'assistant']) > 0
    
    return session_tuples, session_messages, last_user_msg, "Editing previous message.", has_ai_response

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

def update_file_slot_ui(file_list, is_attach=True):
    """Update file slot UI components."""
    max_slots = temporary.MAX_POSSIBLE_ATTACH_SLOTS
    current_max = temporary.MAX_ATTACH_SLOTS
    button_updates = []
    
    for i in range(max_slots):
        # Hide slots beyond current MAX_ATTACH_SLOTS setting
        if i >= current_max:
            button_updates.append(gr.update(value="", visible=False, variant="primary"))
        elif i < len(file_list):
            filename = Path(file_list[i]).name
            short_name = filename[:36] + ".." if len(filename) > 38 else filename
            button_updates.append(gr.update(value=short_name, visible=True, variant="primary"))
        else:
            button_updates.append(gr.update(value="", visible=False, variant="primary"))
    
    show_upload = len(file_list) < current_max if is_attach else True
    button_updates.append(gr.update(visible=show_upload))
    
    return button_updates

def update_session_buttons():
    """Update session history buttons."""
    sessions = get_saved_sessions()[:temporary.MAX_HISTORY_SLOTS]
    button_updates = []
    
    for i in range(temporary.MAX_POSSIBLE_HISTORY_SLOTS):
        # Hide buttons beyond current MAX_HISTORY_SLOTS setting
        if i >= temporary.MAX_HISTORY_SLOTS:
            button_updates.append(gr.update(value="", visible=False))
        elif i < len(sessions):
            session_path = Path(temporary.HISTORY_DIR) / sessions[i]
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
    
    Returns 5 gr.update() objects for:
    - action: Main action button (Send Input when waiting, hidden during generation)
    - edit_previous: Edit previous message button
    - copy_response: Copy last AI response button  
    - cancel_input: Hidden placeholder (never visible)
    - cancel_response: Wait indicator shown during generation
    """
    # Config tuple format: (action_visible, edit_visible, copy_visible, wait_visible)
    if phase == "waiting_for_input":
        if has_ai_response:
            # Has history: show Send, Edit, Copy; hide Wait
            action_visible, edit_visible, copy_visible, wait_visible = True, True, True, False
        else:
            # No history: show Send only; hide Edit, Copy, Wait
            action_visible, edit_visible, copy_visible, wait_visible = True, False, False, False
    elif phase in ("input_submitted", "generating_response", "speaking"):
        # During generation: hide Send, Edit, Copy; show Wait
        action_visible, edit_visible, copy_visible, wait_visible = False, False, False, True
    else:
        # Default fallback
        action_visible, edit_visible, copy_visible, wait_visible = True, False, False, False
    
    # Determine action button appearance
    if phase == "waiting_for_input":
        action_value = "Send Input"
        action_variant = "secondary"
        action_classes = ["send-button-green"]
        action_interactive = True
    else:
        action_value = "Send Input"
        action_variant = "secondary"
        action_classes = ["send-button-green"]
        action_interactive = False
    
    # Determine wait indicator appearance
    wait_variant = "primary"
    wait_classes = []
    
    updates = [
        # 0: action button
        gr.update(
            value=action_value,
            variant=action_variant,
            elem_classes=action_classes,
            interactive=action_interactive,
            visible=action_visible
        ),
        # 1: edit_previous
        gr.update(visible=edit_visible),
        # 2: copy_response
        gr.update(visible=copy_visible),
        # 3: cancel_input - always hidden placeholder
        gr.update(visible=False),
        # 4: cancel_response (wait indicator)
        gr.update(
            value="..Wait For Response..",
            variant=wait_variant,
            elem_classes=wait_classes,
            interactive=False,
            visible=wait_visible
        )
    ]
    
    return updates

def handle_model_load(model_name, ctx, batch, vram, gpu, cpu, threads, llm_state, models_loaded_state):
    """Handle model loading with proper backend support."""
    import scripts.temporary as temporary
    from scripts.models import load_models, get_model_settings
    
    if not model_name or model_name == "Select_a_model...":
        return (
            llm_state, 
            models_loaded_state, 
            "Error: Select a model first.",
            gr.update(),
            gr.update()
        )
    
    temporary.MODEL_NAME = model_name
    temporary.CONTEXT_SIZE = int(ctx)
    temporary.BATCH_SIZE = int(batch)
    temporary.VRAM_SIZE = int(vram)
    temporary.SELECTED_GPU = gpu
    temporary.SELECTED_CPU = cpu
    temporary.CPU_THREADS = int(threads)
    
    try:
        # Fixed: Pass all required arguments to load_models
        status, models_loaded, new_llm, _ = load_models(
            temporary.MODEL_FOLDER,
            model_name,
            int(vram),
            llm_state,
            models_loaded_state
        )
        
        if models_loaded:
            model_settings = get_model_settings(model_name)
            # Beep to notify user model is ready
            beep()
            return (
                new_llm, 
                True, 
                status,
                model_settings,
                gr.update(interactive=True)
            )
        else:
            return (
                llm_state, 
                False, 
                status,
                gr.update(),
                gr.update(interactive=False)
            )
    except Exception as e:
        return (
            llm_state, 
            models_loaded_state, 
            f"Error loading model: {str(e)}",
            gr.update(),
            gr.update(interactive=False)
        )

def handle_model_inspect(model_name):
    """Handle model inspection."""
    from scripts.models import inspect_model
    import scripts.temporary as temporary
    
    if not model_name or model_name == "Select_a_model...":
        return "Select a model to inspect."
    
    # Fixed: Pass all required arguments to inspect_model
    return inspect_model(temporary.MODEL_FOLDER, model_name, temporary.VRAM_SIZE)

def handle_model_inspect(model_name):
    """Handle model inspection."""
    from scripts.models import inspect_model
    
    if not model_name or model_name == "Select_a_model...":
        return "Select a model to inspect."
    
    return inspect_model(model_name)

def delete_all_sessions():
    """Delete all session history files."""
    import shutil
    history_path = Path(temporary.HISTORY_DIR)
    
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
        # Program settings (customization) - with None checks and defaults
        temporary.MAX_HISTORY_SLOTS = int(max_hist) if max_hist is not None else temporary.MAX_HISTORY_SLOTS
        temporary.SESSION_LOG_HEIGHT = int(height) if height is not None else temporary.SESSION_LOG_HEIGHT
        temporary.MAX_ATTACH_SLOTS = int(max_att) if max_att is not None else temporary.MAX_ATTACH_SLOTS
        temporary.SHOW_THINK_PHASE = bool(show_think) if show_think is not None else temporary.SHOW_THINK_PHASE
        temporary.PRINT_RAW_OUTPUT = bool(print_raw) if print_raw is not None else temporary.PRINT_RAW_OUTPUT
        temporary.BLEEP_ON_EVENTS = bool(bleep) if bleep is not None else temporary.BLEEP_ON_EVENTS
        
        # Model/Hardware settings - with None checks
        temporary.CONTEXT_SIZE = int(ctx) if ctx is not None else temporary.CONTEXT_SIZE
        temporary.BATCH_SIZE = int(batch) if batch is not None else temporary.BATCH_SIZE
        temporary.TEMPERATURE = float(temp) if temp is not None else temporary.TEMPERATURE
        temporary.REPEAT_PENALTY = float(repeat) if repeat is not None else temporary.REPEAT_PENALTY
        temporary.VRAM_SIZE = int(vram) if vram is not None else temporary.VRAM_SIZE
        temporary.SELECTED_GPU = gpu if gpu is not None else temporary.SELECTED_GPU
        temporary.SELECTED_CPU = cpu if cpu is not None else temporary.SELECTED_CPU
        temporary.CPU_THREADS = int(cpu_threads) if cpu_threads is not None else temporary.CPU_THREADS
        temporary.MODEL_NAME = model if model is not None else temporary.MODEL_NAME
        temporary.MODEL_FOLDER = model_path if model_path is not None else temporary.MODEL_FOLDER
        
        # Layer allocation mode (only if Vulkan available)
        if layer_allocation_mode is not None and temporary.VULKAN_AVAILABLE:
            temporary.LAYER_ALLOCATION_MODE = layer_allocation_mode
        
        # Save to persistent.json
        from scripts.settings import save_config
        save_config()
        
        print(f"[SAVE] Settings saved: CTX={temporary.CONTEXT_SIZE}, Batch={temporary.BATCH_SIZE}, "
              f"Temp={temporary.TEMPERATURE}, VRAM={temporary.VRAM_SIZE}, GPU={temporary.SELECTED_GPU}, "
              f"CPU={temporary.SELECTED_CPU}, Threads={temporary.CPU_THREADS}")
        
        return "Settings saved successfully."
    except Exception as e:
        print(f"[SAVE] Error: {e}")
        return f"Error saving settings: {str(e)}"

def update_backend_ui():
    """Update UI components based on backend type and saved layer allocation mode."""
    backend_type = temporary.BACKEND_TYPE
    vulkan_available = temporary.VULKAN_AVAILABLE
    layer_mode = temporary.LAYER_ALLOCATION_MODE
    
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

def build_progress_html(step: int, web_search_enabled: bool = False, speech_enabled: bool = False):
    """
    Build dynamic progress indicator HTML based on enabled features.
    
    Cases:
    1. Vanilla (no web search, no speech): 9 steps
    2. Web search only: 11 steps (adds "Producing Research", "Assessing Research")
    3. Speech only: 10 steps (adds "Generating TTS")
    4. Both enabled: 12 steps (all additional steps)
    """
    # Base phases (always present)
    base_phases = [
        "Handle Input",      # 0
        "Build Prompt",      # 1
        "Inject RAG",        # 2
        "Add System",        # 3
        "Assemble History",  # 4
        "Check Model",       # 5
        "Generate Stream",   # 6
        "Split Thinking",    # 7
        "Format Response"    # 8
    ]
    
    # Build dynamic phase list based on enabled features
    phases = base_phases[:6]  # Handle Input through Check Model
    
    # Insert web search phases after "Check Model" (before Generate Stream)
    if web_search_enabled:
        phases.append("Producing Research")
        phases.append("Assessing Research")
    
    # Add generation and processing phases
    phases.append("Generate Stream")
    phases.append("Split Thinking")
    phases.append("Format Response")
    
    # Add TTS phase at the end if speech is enabled
    if speech_enabled:
        phases.append("Generating TTS")
    
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
    temporary.CPU_THREADS = int(new_threads)
    return f"CPU threads set to {new_threads}"

# Global cancel event
import threading
_cancel_event = threading.Event()

def conversation_interface(
    user_input, session_tuples, session_messages, loaded_files,
    is_reasoning_model, cancel_flag, web_search_enabled,
    interaction_phase, llm_state, models_loaded_state,
    speech_enabled, has_ai_response_state
):
    """
    Main conversation handler - Gradio 3.x compatible.
    Uses tuple format for Chatbot display, message dicts internally.
    
    FIXED: 
    - Reads attached file contents and includes them in the prompt
    - Clears attached_files after successful response
    """
    import gradio as gr
    from scripts import temporary, utility
    from scripts.models import get_model_settings, get_response_stream
    from scripts.temporary import context_injector
    from scripts.utility import read_file_content, filter_operational_content, speak_text
    from pathlib import Path
    import time
    import re
    from datetime import datetime

    # Early guards
    if not models_loaded_state or not llm_state:
        yield (
            session_tuples, session_messages, "", 
            gr.update(visible=True), gr.update(visible=False),
            *update_action_buttons("waiting_for_input", False),
            False, loaded_files, "waiting_for_input",
            gr.update(), gr.update(), gr.update()
        )
        return

    if not user_input.strip():
        yield (
            session_tuples, session_messages, "",
            gr.update(visible=True), gr.update(visible=False),
            *update_action_buttons("waiting_for_input", has_ai_response_state),
            False, loaded_files, "waiting_for_input",
            gr.update(), gr.update(), gr.update()
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

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 0: Handle Input - Validate and prepare input
    # ═══════════════════════════════════════════════════════════════════════════
    
    yield (
        session_tuples, session_messages, "",
        gr.update(visible=False), gr.update(visible=True, value=build_progress_html(0, web_search_enabled, speech_enabled)),
        *update_action_buttons("input_submitted", has_ai_response),
        cancel_flag, loaded_files, "input_submitted",
        gr.update(), gr.update(), gr.update()
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
        session_tuples, session_messages, "",
        gr.update(visible=False), gr.update(visible=True, value=build_progress_html(1, web_search_enabled, speech_enabled)),
        *update_action_buttons("input_submitted", has_ai_response),
        cancel_flag, loaded_files, "input_submitted",
        gr.update(), gr.update(), gr.update()
    )

    
    # PHASE 2: Inject RAG - Process large inputs through RAG system
    
    context_threshold = temporary.LARGE_INPUT_THRESHOLD
    max_input_chars = int(temporary.CONTEXT_SIZE * 3 * context_threshold)
    
    if len(user_input) > max_input_chars:
        try:
            context_injector.add_temporary_input(user_input)
            processed_input = user_input[:1000] + "\n\n[Large input processed with RAG]"
        except Exception as e:
            print(f"[RAG-TEMP] Error: {e}")
    
    yield (
        session_tuples, session_messages, "",
        gr.update(visible=False), gr.update(visible=True, value=build_progress_html(2, web_search_enabled, speech_enabled)),
        *update_action_buttons("input_submitted", has_ai_response),
        cancel_flag, loaded_files, "input_submitted",
        gr.update(), gr.update(), gr.update()
    )

    
    # PHASE 3: Add System - Build complete user message with file contents
    
    complete_user_message = processed_input
    if file_contents_section:
        complete_user_message = processed_input + file_contents_section
    
    yield (
        session_tuples, session_messages, "",
        gr.update(visible=False), gr.update(visible=True, value=build_progress_html(3, web_search_enabled, speech_enabled)),
        *update_action_buttons("input_submitted", has_ai_response),
        cancel_flag, loaded_files, "input_submitted",
        gr.update(), gr.update(), gr.update()
    )

    
    # PHASE 4: Assemble History - Add message to session and update display
    
    session_messages = list(session_messages) if session_messages else []
    session_messages.append({'role': 'user', 'content': complete_user_message})
    
    session_tuples = list(session_tuples) if session_tuples else []
    user_display = f"User:\n{complete_user_message}"
    session_tuples.append((user_display, None))
    
    has_ai_response = len([m for m in session_messages[:-1] if m.get('role') == 'assistant']) > 0
    interaction_phase = "input_submitted"
    
    yield (
        session_tuples, session_messages, "",
        gr.update(visible=False), gr.update(visible=True, value=build_progress_html(4, web_search_enabled, speech_enabled)),
        *update_action_buttons("input_submitted", has_ai_response),
        cancel_flag, loaded_files, "input_submitted",
        gr.update(), gr.update(), gr.update()
    )

    
    # PHASE 5: Check Model - Get model settings and prepare for generation
    
    model_settings = get_model_settings(temporary.MODEL_NAME)
    
    # Web search if enabled
    search_results = None
    if web_search_enabled and user_input.strip():
        try:
            clean_query = re.sub(r'^(what|when|where|who|why|how|can you|could you|please)\s+', '', user_input.lower())
            clean_query = re.sub(r'\?+$', '', clean_query).strip()
            words = clean_query.split()[:10]
            search_query = ' '.join(words).strip()
            
            if not search_query or len(search_query) < 3:
                search_query = user_input[:100].strip()
            
            print(f"[WEB-SEARCH] Query: {search_query}")
            search_results = utility.web_search(search_query, num_results=5, max_hits=6)
        except Exception as e:
            print(f"[WEB-SEARCH] Error: {e}")
            search_results = f"Web search error: {str(e)}"
    
    yield (
        session_tuples, session_messages, "",
        gr.update(visible=False), gr.update(visible=True, value=build_progress_html(5, web_search_enabled, speech_enabled)),
        *update_action_buttons("input_submitted", has_ai_response),
        cancel_flag, loaded_files, "input_submitted",
        gr.update(), gr.update(), gr.update()
    )

    # Web search phases (if enabled) - phases 6 and 7
    if web_search_enabled:
        # Phase: Producing Research
        yield (
            session_tuples, session_messages, "",
            gr.update(visible=False), gr.update(visible=True, value=build_progress_html(6, web_search_enabled, speech_enabled)),
            *update_action_buttons("input_submitted", has_ai_response),
            cancel_flag, loaded_files, "input_submitted",
            gr.update(), gr.update(), gr.update()
        )
        time.sleep(0.1)
        
        # Phase: Assessing Research
        yield (
            session_tuples, session_messages, "",
            gr.update(visible=False), gr.update(visible=True, value=build_progress_html(7, web_search_enabled, speech_enabled)),
            *update_action_buttons("input_submitted", has_ai_response),
            cancel_flag, loaded_files, "input_submitted",
            gr.update(), gr.update(), gr.update()
        )
        time.sleep(0.1)

    # Get model settings
    model_settings = get_model_settings(temporary.MODEL_NAME)

    # Calculate Generate Stream phase index (shifts by 2 if web search enabled)
    generate_phase = 8 if web_search_enabled else 6

    # Phase 6 (or 8 with web search): Generating response
    interaction_phase = "generating_response"
    _cancel_event.clear()

    # Add placeholder for assistant
    session_messages.append({'role': 'assistant', 'content': ""})
    
    accumulated_response = ""
    
    try:
        yield (
            session_tuples, session_messages, "",
            gr.update(visible=False), gr.update(visible=True, value=build_progress_html(generate_phase, web_search_enabled, speech_enabled)),
            *update_action_buttons("generating_response", has_ai_response),
            cancel_flag, loaded_files, "generating_response",
            gr.update(), gr.update(), gr.update()
        )
        
        # Stream the response
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
            
            # Filter out any AI-Chat: prefix the model might generate
            clean_response = re.sub(r'^AI-Chat:\s*\n?', '', accumulated_response, flags=re.MULTILINE)
            clean_response = re.sub(r'\nAI-Chat:\s*\n?', '\n', clean_response)
            
            # Update assistant message (store clean version)
            session_messages[-1]['content'] = clean_response
            
            # Update tuples - add "AI-Chat:" label for display
            bot_display = f"AI-Chat:\n{clean_response}"
            session_tuples[-1] = (session_tuples[-1][0], bot_display)
            
            yield (
                session_tuples, session_messages, "",
                gr.update(visible=False), gr.update(visible=True, value=build_progress_html(generate_phase, web_search_enabled, speech_enabled)),
                *update_action_buttons("generating_response", has_ai_response),
                cancel_flag, loaded_files, "generating_response",
                gr.update(), gr.update(), gr.update()
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
    if accumulated_response.strip():  # Only save if we actually got a meaningful response
        if not temporary.SESSION_ACTIVE:
            # This is the very first response → create new session
            temporary.SESSION_ACTIVE = True
            temporary.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            temporary.session_label = utility.summarize_session(session_messages)
            print(f"[SESSION] New session created: {temporary.session_label}")

        # Save (or update) the session history
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

    # TTS: Speak response if enabled
    if speech_enabled and accumulated_response:
        # Calculate TTS phase index: base 9 + 2 if web search enabled
        tts_phase = 11 if web_search_enabled else 9
        
        # Show "Generating TTS" phase in progress
        yield (
            session_tuples, session_messages, "",
            gr.update(visible=False), gr.update(visible=True, value=build_progress_html(tts_phase, web_search_enabled, speech_enabled)),
            *update_action_buttons("speaking", True),
            cancel_flag, loaded_files, "speaking",
            gr.update(), gr.update(), gr.update()
        )
        
        try:
            # Clean response for TTS - remove thinking phases and special chars
            tts_text = filter_operational_content(accumulated_response)
            # Keep only letters, spaces, commas, and periods
            tts_text = re.sub(r'[^a-zA-Z\s,.]', ' ', tts_text)
            # Collapse multiple spaces
            tts_text = re.sub(r'\s+', ' ', tts_text).strip()
            if tts_text:
                print(f"[TTS] Speaking {len(tts_text)} chars...")
                speak_text(tts_text)
        except Exception as e:
            print(f"[TTS] Error: {e}")

    
    # FIXED: Clear attached files after successful response
    
    cleared_files = []  # Empty the attached files list
    temporary.session_attached_files = []  # Also clear the global tracker
    
    # Beep to notify user response is complete (unless TTS is active)
    if not speech_enabled:
        beep()
    
    # Final yield - with cleared attached_files
    yield (
        session_tuples, session_messages, "",
        gr.update(visible=True), gr.update(visible=False),
        *update_action_buttons("waiting_for_input", True),
        False, cleared_files, "waiting_for_input",  # <-- cleared_files instead of loaded_files
        gr.update(), gr.update(), gr.update()
    )

def toggle_left_expanded_state(current_state):
    return not current_state

def toggle_right_expanded_state(current_state):
    return not current_state

def toggle_web_search(current_state):
    new_state = not current_state
    variant = "primary" if new_state else "secondary"
    label = "🌐 Web-Search ON" if new_state else "🌐 Web-Search"
    return new_state, gr.update(variant=variant, value=label), gr.update(variant=variant)

def toggle_speech(current_state):
    new_state = not current_state
    variant = "primary" if new_state else "secondary"
    label = "🔊 Speech ON" if new_state else "🔊 Speech"
    return new_state, gr.update(variant=variant, value=label), gr.update(variant=variant)

# ============================================================================
# MAIN INTERFACE LAUNCH
# ============================================================================

def launch_interface():
    """Launch the Gradio interface - Gradio 3.50.2 compatible."""
    global demo
    import tkinter as tk
    from tkinter import filedialog
    import os
    import gradio as gr
    from pathlib import Path
    from launcher import shutdown_program
    from scripts import temporary, utility, models
    from scripts.temporary import (
        MODEL_NAME, SESSION_ACTIVE,
        MAX_HISTORY_SLOTS, MAX_ATTACH_SLOTS, SESSION_LOG_HEIGHT,
        MODEL_FOLDER, CONTEXT_SIZE, BATCH_SIZE, TEMPERATURE, REPEAT_PENALTY,
        VRAM_SIZE, SELECTED_GPU, SELECTED_CPU, MLOCK, BACKEND_TYPE,
        ALLOWED_EXTENSIONS, VRAM_OPTIONS, CTX_OPTIONS, BATCH_OPTIONS, TEMP_OPTIONS,
        REPEAT_OPTIONS, HISTORY_SLOT_OPTIONS, SESSION_LOG_HEIGHT_OPTIONS,
        ATTACH_SLOT_OPTIONS
    )

    # Gradio 3.x compatible Blocks
    with gr.Blocks(
        title="Conversation-Gradio-Gguf",
        css="""
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
        /* FIX: Reduce extra blank lines in Qt WebEngine rendering */
        .message p { margin-top: 0.3em !important; margin-bottom: 0.3em !important; }
        .message br + br { display: none !important; }
        /* Alternative fix for pre-formatted text appearance */
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
        """
    ) as demo:
        temporary.demo = demo
        model_folder_state = gr.State(temporary.MODEL_FOLDER)

        # States - note session_messages for internal dict format
        states = dict(
            attached_files=gr.State([]),
            session_messages=gr.State([]),  # Internal message dict format
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
            speech_enabled=gr.State(False),
            has_ai_response=gr.State(False)
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
            with gr.Tab("Interaction"):
                with gr.Row():
                    # LEFT PANEL
                    with gr.Column(visible=True, min_width=300, elem_classes=["clean-elements"]) as left_column_expanded:
                        toggle_button_left_expanded = gr.Button(">-------<", variant="secondary")
                        gr.Markdown("**Dynamic Panel**")
                        panel_toggle = gr.Radio(choices=["History", "Attachments"], label="", value="History")
                        with gr.Group(visible=False) as attach_group:
                            attach_files = gr.UploadButton(
                                "Add Attach Files..", 
                                file_types=[f".{ext}" for ext in temporary.ALLOWED_EXTENSIONS], 
                                file_count="multiple", 
                                variant="secondary"
                            )
                            attach_slots = [gr.Button("Attach Slot Free", variant="huggingface", visible=False) 
                                          for _ in range(temporary.MAX_POSSIBLE_ATTACH_SLOTS)]
                        with gr.Group(visible=True) as history_slots_group:
                            start_new_session_btn = gr.Button("Start New Session..", variant="secondary")
                            buttons["session"] = [gr.Button(f"History Slot {i+1}", variant="huggingface", visible=False) 
                                                 for i in range(temporary.MAX_POSSIBLE_HISTORY_SLOTS)]

                    with gr.Column(visible=False, min_width=60, elem_classes=["clean-elements"]) as left_column_collapsed:
                        toggle_button_left_collapsed = gr.Button("<->", variant="secondary")
                        new_session_btn_collapsed = gr.Button("New", variant="secondary")
                        add_attach_files_collapsed = gr.UploadButton(
                            "Add", 
                            file_types=[f".{ext}" for ext in temporary.ALLOWED_EXTENSIONS], 
                            file_count="multiple", 
                            variant="secondary"
                        )

                    # CENTER - Main chat area
                    with gr.Column(scale=30, elem_classes=["clean-elements"]):
                        # Gradio 3.x Chatbot - no type parameter
                        conversation_components["session_log"] = gr.Chatbot(
                            label="Session Log", 
                            height=temporary.SESSION_LOG_HEIGHT,
                            elem_classes=["scrollable"]
                        )
                        
                        initial_max_lines = max(3, int(((temporary.SESSION_LOG_HEIGHT - 100) / 10) / 2.5) - 6)
                        temporary.USER_INPUT_MAX_LINES = initial_max_lines
                        
                        conversation_components["user_input"] = gr.Textbox(
                            label="User Input", 
                            lines=3, 
                            max_lines=initial_max_lines, 
                            interactive=False, 
                            placeholder="Enter text here..."
                        )
                        
                        conversation_components["progress_indicator"] = gr.Markdown(
                            value="",
                            visible=False,
                            elem_classes=["progress-indicator"]
                        )

                        with gr.Row(elem_classes=["clean-elements"]):
                            action_buttons["action"] = gr.Button("Send Input", variant="secondary", elem_classes=["send-button-green"], scale=10)
                            action_buttons["edit_previous"] = gr.Button("Edit Previous", variant="secondary", scale=1, visible=False)
                            action_buttons["copy_response"] = gr.Button("Copy Output", variant="huggingface", scale=1, visible=False)
                            action_buttons["cancel_input"] = gr.Button("", variant="primary", scale=1, visible=False)  # Hidden placeholder for output compatibility
                            action_buttons["cancel_response"] = gr.Button("..Wait For Response..", variant="primary", scale=1, visible=False)

                    # RIGHT PANEL
                    with gr.Column(visible=True, min_width=300, elem_classes=["clean-elements"]) as right_column_expanded:
                        toggle_button_right_expanded = gr.Button(">-------<", variant="secondary")
                        gr.Markdown("**Tools / Options**")
                        with gr.Row(elem_classes=["clean-elements"]):
                            action_buttons["web_search"] = gr.Button("🌐 Web-Search", variant="secondary", scale=1)
                            action_buttons["speech"] = gr.Button("🔊 Speech", variant="secondary", scale=1)

                    with gr.Column(visible=False, min_width=60, elem_classes=["clean-elements"]) as right_column_collapsed:
                        toggle_button_right_collapsed = gr.Button("<->", variant="secondary")
                        action_buttons["web_search_collapsed"] = gr.Button("🌐", variant="secondary")
                        action_buttons["speech_collapsed"] = gr.Button("🔊", variant="secondary")

                with gr.Row():
                    interaction_global_status = gr.Textbox(
                        value="Ready",
                        label="Status",
                        interactive=False,
                        max_lines=1,
                        scale=20
                    )
                    exit_interaction = gr.Button("Exit Program", variant="stop", elem_classes=["double-height"], scale=1)

            with gr.Tab("Configuration"):
                with gr.Column(scale=1, elem_classes=["clean-elements"]):
                    gr.Markdown("**Hardware**")
                    
                    with gr.Row(elem_classes=["clean-elements"]):
                        backend_type_display = gr.Textbox(
                            label="Binaries and Wheel", 
                            value=temporary.BACKEND_TYPE, 
                            interactive=False,
                            scale=5
                        )
                        layer_allocation_radio = gr.Radio(
                            choices=["SRAM_ONLY", "VRAM_SRAM"],
                            label="Model Loaded To",
                            value=temporary.LAYER_ALLOCATION_MODE,
                            visible=temporary.VULKAN_AVAILABLE,
                            scale=5
                        )
                    
                    gpu_row_visible = temporary.BACKEND_TYPE in ["VULKAN_VULKAN", "VULKAN_CPU"]
                    with gr.Row(elem_classes=["clean-elements"], visible=gpu_row_visible) as gpu_row:
                        gpus = utility.get_available_gpus()
                        gpu_choices = ["Auto-Select"] + gpus if len(gpus) > 1 else gpus
                        def_gpu = temporary.SELECTED_GPU if temporary.SELECTED_GPU in gpu_choices else (gpu_choices[0] if gpu_choices else "Auto-Select")
                        
                        config_components["gpu"] = gr.Dropdown(
                            choices=gpu_choices, 
                            label="GPU Selected", 
                            value=def_gpu,
                            interactive=len(gpu_choices) > 1
                        )
                        
                        vram_visible = gpu_row_visible and temporary.LAYER_ALLOCATION_MODE == "VRAM_SRAM"
                        config_components["vram"] = gr.Dropdown(
                            choices=temporary.VRAM_OPTIONS,
                            label="VRAM MB",
                            value=temporary.VRAM_SIZE,
                            visible=vram_visible
                        )
                    
                    with gr.Row(elem_classes=["clean-elements"]):
                        cpu_info = utility.get_cpu_info()
                        cpu_labs = [c["label"] for c in cpu_info]
                        
                        if len(cpu_info) > 1:
                            cpu_opts = ["Auto-Select"] + cpu_labs
                            default_cpu_value = "Auto-Select"
                        else:
                            cpu_opts = cpu_labs
                            default_cpu_value = cpu_labs[0] if cpu_labs else "Default CPU"
                        
                        saved_cpu = temporary.SELECTED_CPU
                        def_cpu = saved_cpu if saved_cpu and saved_cpu in cpu_opts else default_cpu_value
                        
                        config_components["cpu"] = gr.Dropdown(
                            choices=cpu_opts,
                            label="CPU Selected",
                            value=def_cpu,
                            interactive=len(cpu_opts) > 1,
                            scale=5
                        )
                        
                        max_threads = max(temporary.CPU_THREAD_OPTIONS or [8])
                        current_threads = temporary.CPU_THREADS or min(4, max_threads)
                        config_components["cpu_threads"] = gr.Slider(
                            minimum=1,
                            maximum=max_threads,
                            value=current_threads,
                            step=1,
                            label="CPU Threads",
                            interactive=True,
                            scale=5
                        )
                    
                    gr.Markdown("**Model**")
                    with gr.Row(elem_classes=["clean-elements"]):
                        avail = temporary.AVAILABLE_MODELS or get_available_models()
                        mods = ["Select_a_model..."] + [m for m in avail if m != "Select_a_model..."]
                        def_m = temporary.MODEL_NAME if temporary.MODEL_NAME in mods else (mods[1] if len(mods) > 1 else mods[0])
                        
                        config_components["model_path"] = gr.Textbox(
                            label="Folder Location", 
                            value=temporary.MODEL_FOLDER, 
                            interactive=False
                        )
                        config_components["model"] = gr.Dropdown(
                            choices=mods, 
                            label=".gguf File", 
                            value=def_m
                        )
                        keywords_display = gr.Textbox(label="Keywords Detected", interactive=False)
                    
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components["ctx"] = gr.Dropdown(
                            temporary.CTX_OPTIONS, label="Context Length", value=temporary.CONTEXT_SIZE, scale=5
                        )
                        config_components["batch"] = gr.Dropdown(
                            temporary.BATCH_OPTIONS, label="Batch Size", value=temporary.BATCH_SIZE, scale=5
                        )
                        config_components["temp"] = gr.Dropdown(
                            temporary.TEMP_OPTIONS, label="Temperature", value=temporary.TEMPERATURE, scale=5
                        )
                        config_components["repeat"] = gr.Dropdown(
                            temporary.REPEAT_OPTIONS, label="Repeat Penalty", value=temporary.REPEAT_PENALTY, scale=5
                        )
                    
                    with gr.Row(elem_classes=["clean-elements"]):
                        browse = gr.Button("📁 Browse Folders", variant="secondary")
                        config_components["load"] = gr.Button("💾 Load Model", variant="primary")
                        config_components["inspect"] = gr.Button("🔍 Inspect Model")
                        config_components["unload"] = gr.Button("🗑️ Unload Model", variant="stop")
                    
                    gr.Markdown("**Program**")
                    with gr.Row(elem_classes=["clean-elements"]):
                        custom_components["max_hist"] = gr.Dropdown(
                            temporary.HISTORY_SLOT_OPTIONS, label="History Slots", value=temporary.MAX_HISTORY_SLOTS, scale=5
                        )
                        custom_components["height"] = gr.Dropdown(
                            temporary.SESSION_LOG_HEIGHT_OPTIONS, label="Log Height", value=temporary.SESSION_LOG_HEIGHT, scale=5
                        )
                        custom_components["max_att"] = gr.Dropdown(
                            temporary.ATTACH_SLOT_OPTIONS, label="Attach Slots", value=temporary.MAX_ATTACH_SLOTS, scale=5
                        )
                        with gr.Column(scale=5):
                            config_components["show_think_phase"] = gr.Checkbox(
                                label="Show Think Phase Output", value=temporary.SHOW_THINK_PHASE
                            )
                            config_components["print_raw"] = gr.Checkbox(
                                label="Print Debug Model Output", value=temporary.PRINT_RAW_OUTPUT
                            )
                            config_components["bleep_events"] = gr.Checkbox(
                                label="Bleep Upon Major Events", value=temporary.BLEEP_ON_EVENTS
                            )
                    
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components["save_settings"] = gr.Button("💾 Save Settings", variant="primary")
                        config_components["delete_all_history"] = gr.Button("🗑️ Clear All History", variant="stop")
                    
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_global_status = gr.Textbox(
                            value="Ready",
                            label="Status",
                            interactive=False,
                            max_lines=1,
                            scale=20
                        )
                        exit_config = gr.Button("Exit Program", variant="stop", elem_classes=["double-height"], scale=1)

        # ============================================================================
        # EVENT HANDLERS
        # ============================================================================

        # Exit buttons
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

        # Web search toggle
        action_buttons["web_search"].click(
            fn=toggle_web_search,
            inputs=[states["web_search_enabled"]],
            outputs=[states["web_search_enabled"], action_buttons["web_search"], action_buttons["web_search_collapsed"]]
        )
        
        action_buttons["web_search_collapsed"].click(
            fn=toggle_web_search,
            inputs=[states["web_search_enabled"]],
            outputs=[states["web_search_enabled"], action_buttons["web_search"], action_buttons["web_search_collapsed"]]
        )

        # Speech toggle
        action_buttons["speech"].click(
            fn=toggle_speech,
            inputs=[states["speech_enabled"]],
            outputs=[states["speech_enabled"], action_buttons["speech"], action_buttons["speech_collapsed"]]
        )
        
        action_buttons["speech_collapsed"].click(
            fn=toggle_speech,
            inputs=[states["speech_enabled"]],
            outputs=[states["speech_enabled"], action_buttons["speech"], action_buttons["speech_collapsed"]]
        )

        # New session
        def start_new_session_wrapper(models_loaded):
            result = start_new_session(models_loaded)
            # result: (tuples, messages, status, user_input_update, web_search, has_ai)
            tuples, messages, status, user_input_update, web_search, has_ai = result
            
            # Get action button updates - new session has no AI response yet
            btn_updates = update_action_buttons("waiting_for_input", has_ai)
            
            return (
                tuples,           # session_log
                messages,         # session_messages
                status,           # interaction_global_status
                status,           # config_global_status
                user_input_update,# user_input
                web_search,       # web_search_enabled
                has_ai,           # has_ai_response
                *btn_updates      # action, edit_previous, copy_response, cancel_input, cancel_response
            )
        
        # Normal (visible) Start New Session button
        start_new_session_btn.click(
            fn=start_new_session,
            inputs=[
                states["session_messages"],
                states["attached_files"]
            ],
            outputs=[
                conversation_components["session_log"],
                states["session_messages"],
                states["attached_files"],
                interaction_global_status,
                config_global_status,
                conversation_components["user_input"],
                states["web_search_enabled"],
                states["has_ai_response"],
                action_buttons["action"],
                action_buttons["edit_previous"],
                action_buttons["copy_response"],
                action_buttons["cancel_input"],
                action_buttons["cancel_response"]
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
                states["attached_files"]
            ],
            outputs=[
                conversation_components["session_log"],
                states["session_messages"],
                states["attached_files"],
                interaction_global_status,
                config_global_status,
                conversation_components["user_input"],
                states["web_search_enabled"],
                states["has_ai_response"],
                action_buttons["action"],
                action_buttons["edit_previous"],
                action_buttons["copy_response"],
                action_buttons["cancel_input"],
                action_buttons["cancel_response"]
            ]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        # Model folder browsing
        browse.click(
            fn=browse_on_click,
            inputs=[model_folder_state],
            outputs=[model_folder_state]
        ).then(
            fn=lambda f: f,
            inputs=[model_folder_state],
            outputs=[config_components["model_path"]]
        ).then(
            fn=update_model_list,
            inputs=[model_folder_state],
            outputs=[config_components["model"]]
        )

        # Model loading
        def handle_model_load_wrapper(model_name, ctx, batch, vram, gpu, cpu, threads, llm_state, models_loaded_state):
            result = handle_model_load(model_name, ctx, batch, vram, gpu, cpu, threads, llm_state, models_loaded_state)
            # result: (llm, loaded, status, model_settings, user_input)
            return result[0], result[1], result[2], result[2], result[3], result[4]
        
        config_components["load"].click(
            fn=handle_model_load_wrapper,
            inputs=[
                config_components["model"],
                config_components["ctx"],
                config_components["batch"],
                config_components["vram"],
                config_components["gpu"],
                config_components["cpu"],
                config_components["cpu_threads"],
                states["llm"],
                states["models_loaded"]
            ],
            outputs=[
                states["llm"],
                states["models_loaded"],
                interaction_global_status,
                config_global_status,
                states["model_settings"],
                conversation_components["user_input"]
            ]
        )

        # Model unloading
        def handle_model_unload_wrapper(llm_state, models_loaded_state):
            result = handle_model_unload(llm_state, models_loaded_state)
            # result: (llm, loaded, status)
            return result[0], result[1], result[2], result[2]
        
        config_components["unload"].click(
            fn=handle_model_unload_wrapper,
            inputs=[states["llm"], states["models_loaded"]],
            outputs=[states["llm"], states["models_loaded"], interaction_global_status, config_global_status]
        )

        # Model inspection
        def handle_model_inspect_wrapper(model_name):
            result = handle_model_inspect(model_name)
            return result, result
        
        config_components["inspect"].click(
            fn=handle_model_inspect_wrapper,
            inputs=[config_components["model"]],
            outputs=[interaction_global_status, config_global_status]
        )

        # Save settings - with dynamic UI update
        def handle_save_wrapper(
            max_hist, height, max_att, show_think, print_raw, bleep,
            ctx, batch, temp, repeat, vram, gpu, cpu, cpu_threads, model, model_path,
            layer_mode
        ):
            result = handle_customization_save(
                max_hist, height, max_att, show_think, print_raw, bleep,
                ctx, batch, temp, repeat, vram, gpu, cpu, cpu_threads, model, model_path,
                layer_mode
            )
            return result, result
        
        def update_ui_after_save():
            """Update UI components to reflect new settings."""
            # Update session log height
            session_log_update = gr.update(height=temporary.SESSION_LOG_HEIGHT)
            
            # Update history slot visibility
            history_updates = update_session_buttons()
            
            # Update attach slot visibility based on new MAX_ATTACH_SLOTS
            attach_updates = []
            for i in range(temporary.MAX_POSSIBLE_ATTACH_SLOTS):
                # Slots beyond the new max should be hidden
                if i >= temporary.MAX_ATTACH_SLOTS:
                    attach_updates.append(gr.update(visible=False))
                else:
                    attach_updates.append(gr.update())  # Keep current state
            attach_updates.append(gr.update())  # Upload button stays as-is
            
            return [session_log_update] + history_updates + attach_updates
        
        config_components["save_settings"].click(
            fn=handle_save_wrapper,
            inputs=[
                # Program settings
                custom_components["max_hist"],
                custom_components["height"],
                custom_components["max_att"],
                config_components["show_think_phase"],
                config_components["print_raw"],
                config_components["bleep_events"],
                # Model/Hardware settings
                config_components["ctx"],
                config_components["batch"],
                config_components["temp"],
                config_components["repeat"],
                config_components["vram"],
                config_components["gpu"],
                config_components["cpu"],
                config_components["cpu_threads"],
                config_components["model"],
                config_components["model_path"],
                layer_allocation_radio
            ],
            outputs=[interaction_global_status, config_global_status]
        ).then(
            fn=update_ui_after_save,
            inputs=[],
            outputs=[conversation_components["session_log"]] + buttons["session"] + attach_slots + [attach_files]
        )

        # Delete all history
        def delete_all_wrapper():
            result = delete_all_sessions()
            return result, result
        
        config_components["delete_all_history"].click(
            fn=delete_all_wrapper,
            inputs=[],
            outputs=[interaction_global_status, config_global_status]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        # Layer allocation change
        layer_allocation_radio.change(
            fn=handle_backend_mode_change,
            inputs=[layer_allocation_radio],
            outputs=[config_components["vram"]]
        )

        # CPU threads change
        def handle_cpu_threads_wrapper(new_threads):
            result = handle_cpu_threads_change(new_threads)
            return result, result
        
        config_components["cpu_threads"].change(
            fn=handle_cpu_threads_wrapper,
            inputs=[config_components["cpu_threads"]],
            outputs=[interaction_global_status, config_global_status]
        )

        # Main conversation handler
        action_buttons["action"].click(
            fn=conversation_interface,
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
                states["speech_enabled"],
                states["has_ai_response"]
            ],
            outputs=[
                conversation_components["session_log"],
                states["session_messages"],
                conversation_components["user_input"],
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
                states["has_ai_response"],
                gr.State(),
                gr.State()
            ]
        ).then(
            fn=lambda files: update_file_slot_ui(files, True),  # Update attach slot UI after response
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        ).then(
            fn=update_session_buttons,  # Refresh session history after each response
            inputs=[],
            outputs=buttons["session"]
        )

        # Edit previous message
        def edit_previous_wrapper(session_tuples, session_messages):
            tuples, messages, user_input, status, has_ai = edit_previous_prompt(session_tuples, session_messages)
            return (
                tuples, 
                messages, 
                user_input, 
                status, 
                status,
                has_ai,
                *update_action_buttons("waiting_for_input", has_ai)
            )
        
        action_buttons["edit_previous"].click(
            fn=edit_previous_wrapper,
            inputs=[
                conversation_components["session_log"],
                states["session_messages"]
            ],
            outputs=[
                conversation_components["session_log"],
                states["session_messages"],
                conversation_components["user_input"],
                interaction_global_status,
                config_global_status,
                states["has_ai_response"],
                action_buttons["action"],
                action_buttons["edit_previous"],
                action_buttons["copy_response"],
                action_buttons["cancel_input"],
                action_buttons["cancel_response"]
            ]
        )

        # Copy response
        def copy_response_wrapper(session_messages):
            result = copy_last_response(session_messages)
            return result, result
        
        action_buttons["copy_response"].click(
            fn=copy_response_wrapper,
            inputs=[states["session_messages"]],
            outputs=[interaction_global_status, config_global_status]
        )

        # Cancel response
        def cancel_wrapper():
            _cancel_event.set()
            return "Cancelling...", "Cancelling..."
        
        action_buttons["cancel_response"].click(
            fn=cancel_wrapper,
            inputs=[],
            outputs=[interaction_global_status, config_global_status]
        )

        # Session history buttons
        def load_session_wrapper(idx):
            result = load_session_by_index(idx)
            # result: (tuples, messages, files, status, has_ai)
            tuples, messages, files, status, has_ai = result
            
            # Get action button updates based on whether session has AI responses
            btn_updates = update_action_buttons("waiting_for_input", has_ai)
            
            return (
                tuples,           # session_log
                messages,         # session_messages
                files,            # attached_files
                status,           # interaction_global_status
                status,           # config_global_status
                has_ai,           # has_ai_response
                *btn_updates      # action, edit_previous, copy_response, cancel_input, cancel_response
            )
        
        for i, btn in enumerate(buttons["session"]):
            btn.click(
                fn=lambda idx=i: load_session_wrapper(idx),
                inputs=[],
                outputs=[
                    conversation_components["session_log"],
                    states["session_messages"],
                    states["attached_files"],
                    interaction_global_status,
                    config_global_status,
                    states["has_ai_response"],
                    action_buttons["action"],
                    action_buttons["edit_previous"],
                    action_buttons["copy_response"],
                    action_buttons["cancel_input"],
                    action_buttons["cancel_response"]
                ]
            )

        # File attachments
        def process_attach_wrapper(files, attached_files, models_loaded):
            result = process_attach_files(files, attached_files, models_loaded)
            # result: (status, files)
            return result[0], result[0], result[1]
        
        attach_files.upload(
            fn=process_attach_wrapper,
            inputs=[attach_files, states["attached_files"], states["models_loaded"]],
            outputs=[interaction_global_status, config_global_status, states["attached_files"]]
        ).then(
            fn=lambda files: update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        )

        # Attach slot ejection handlers - click to remove file
        def make_eject_handler(slot_index):
            """Factory to create unique handler for each slot index."""
            def handler(attached_files):
                result = eject_file(attached_files, slot_index, is_attach=True)
                # result format: [updated_files, status_msg, *button_updates]
                updated_files = result[0]
                status_msg = result[1]
                button_updates = result[2:]  # gr.update() for each slot + upload button
                return [updated_files, status_msg, status_msg] + button_updates
            return handler
        
        for i, slot_btn in enumerate(attach_slots):
            slot_btn.click(
                fn=make_eject_handler(i),
                inputs=[states["attached_files"]],
                outputs=[
                    states["attached_files"],
                    interaction_global_status,
                    config_global_status
                ] + attach_slots + [attach_files]
            )

        # Initial load
        demo.load(
            fn=lambda: temporary.MODEL_FOLDER,
            inputs=[],
            outputs=[model_folder_state]
        ).then(
            fn=update_model_list,
            inputs=[model_folder_state],
            outputs=[config_components["model"]]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

    # Launch with browser
    import threading
    from scripts.browser import launch_custom_browser, wait_for_gradio
    
    print("[BROWSER] Starting Gradio server in background...")
    
    # Enable queue for generator/streaming support
    demo.queue()
    
    gradio_thread = threading.Thread(
        target=lambda: demo.launch(
            server_name="localhost",
            server_port=7860,
            show_error=True,
            share=False,
            inbrowser=False,
            prevent_thread_lock=True
        ),
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
    launch_interface()