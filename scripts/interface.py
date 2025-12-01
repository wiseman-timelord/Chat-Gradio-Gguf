# Script: `.\scripts\interface.py`

# Imports...
import gradio as gr
from gradio import themes
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
import scripts.settings as settings  # Added import for configuration variables
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
    get_available_gpus, filter_operational_content, speak_text, process_files
)
from scripts.models import (
    get_response_stream, get_available_models, unload_models, get_model_settings, inspect_model, load_models
)

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
    """
    Update panel visibility based on the selected panel, fixed for Conversation mode.

    Args:
        current_panel (str): The currently selected panel.

    Returns:
        tuple: Updates for panel toggle, attach group, history group, and selected panel state.
    """
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
    temporary.SELECTED_CPU = cpu  # Add this line
    temporary.MODEL_NAME = model
    temporary.PRINT_RAW_OUTPUT = bool(print_raw)
    status_message = (
        f"Updated settings: Context Size={ctx}, Batch Size={batch}, "
        f"Temperature={temp}, Repeat Penalty={repeat}, VRAM Size={vram}, "
        f"Selected GPU={gpu}, CPU={cpu}, Model={model}"  # Update message
    )
    return status_message

def update_stream_output(stream_output_value):
    temporary.STREAM_OUTPUT = stream_output_value
    status_message = "Stream output enabled." if stream_output_value else "Stream output disabled."
    return status_message

def save_all_settings():
    """
    Save all configuration settings and return a status message.

    Returns:
        str: Confirmation message.
    """
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
    """Adjust the Session Log height based on the number of lines in User Input, capped at max_lines."""
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
    """
    Format response with thinking phase detection and code highlighting.
    Works with both standard <think> tags and gpt-oss channel format.
    """
    import re
    from pygments import highlight
    from pygments.lexers import get_lexer_by_name
    from pygments.formatters import HtmlFormatter
    from scripts.temporary import THINK_COLOR
    
    formatted = []
    
    # Extract standard <think> blocks
    standard_thinks = re.findall(r'<think>(.*?)</think>', output, re.DOTALL)
    for thought in standard_thinks:
        if thought.strip():
            formatted.append(
                f'<span style="color: {THINK_COLOR}">[Thinking] {thought.strip()}</span>'
            )
    
    # Extract gpt-oss analysis channel blocks
    gpt_oss_thinks = re.findall(
        r'<\|channel\|>analysis(.*?)(?:<\|end\|>|<\|channel\|>final)', 
        output, 
        re.DOTALL
    )
    for thought in gpt_oss_thinks:
        # Clean up any remaining channel tags
        thought = re.sub(r'<\|[^>]+\|>', '', thought)
        if thought.strip():
            formatted.append(
                f'<span style="color: {THINK_COLOR}">[Thinking] {thought.strip()}</span>'
            )
    
    # Remove all thinking content from output
    clean_output = output
    clean_output = re.sub(r'<think>.*?</think>', '', clean_output, flags=re.DOTALL)
    clean_output = re.sub(
        r'<\|channel\|>analysis.*?(?:<\|end\|>.*?<\|channel\|>final|<\|end\|><\|start\|>assistant<\|end\|><\|start\|>assistant<\|message\|>)',
        '',
        clean_output,
        flags=re.DOTALL
    )
    
    # Remove channel tags from output
    clean_output = re.sub(r'<\|[^>]+\|>', '', clean_output)
    
    # Remove "Thinking...." lines (dots mode output)
    lines = clean_output.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that are just "Thinking" followed by dots/spaces
        if stripped.startswith("Thinking") and all(c in '.… ' for c in stripped[8:]):
            continue
        filtered_lines.append(line)
    clean_output = '\n'.join(filtered_lines)
    
    # Process code blocks with syntax highlighting
    code_blocks = re.findall(r'```(\w+)?\n(.*?)```', clean_output, re.DOTALL)
    for lang, code in code_blocks:
        if lang:
            try:
                lexer = get_lexer_by_name(lang, stripall=True)
                formatted_code = highlight(code, lexer, HtmlFormatter())
                clean_output = clean_output.replace(f'```{lang}\n{code}```', formatted_code)
            except:
                pass
    
    # Clean up whitespace
    clean_output = clean_output.replace('\r\n', '\n')
    clean_output = re.sub(r'\n{3,}', '\n\n', clean_output)
    clean_output = clean_output.strip()
    clean_output = clean_output.replace('\n', '<br>')
    
    # Combine thinking blocks with cleaned answer
    if formatted:
        return '<br>'.join(formatted) + '<br><br>' + clean_output
    else:
        return clean_output

def get_initial_model_value():
    available_models = temporary.AVAILABLE_MODELS or get_available_models()
    base_choices = ["Select_a_model..."]
    if available_models and available_models != ["Select_a_model..."]:
        available_models = [m for m in available_models if m not in base_choices]
        available_models = base_choices + available_models
    else:
        available_models = base_choices
    if temporary.MODEL_NAME in available_models and temporary.MODEL_NAME not in base_choices:
        default_model = temporary.MODEL_NAME
    elif len(available_models) > 2:
        default_model = available_models[2]
    else:
        default_model = "Select_a_model..."
    is_reasoning = get_model_settings(default_model)["is_reasoning"] if default_model not in base_choices else False
    return default_model, is_reasoning

def update_model_list(new_dir):
    print(f"Updating model list with new_dir: {new_dir}")
    temporary.MODEL_FOLDER = new_dir
    choices = get_available_models()  # Scan the directory for models
    if choices and choices[0] != "Select_a_model...":  # If models are found
        value = choices[0]  # Default to the first model
    else:  # If no models are found
        choices = ["Select_a_model..."]  # Ensure this is in choices
        value = "Select_a_model..."  # Set value accordingly
    print(f"Choices returned: {choices}, Setting value to: {value}")
    return gr.update(choices=choices, value=value)

def handle_model_selection(model_name, model_folder_state):
    if not model_name:
        return model_folder_state, model_name, "No model selected."
    return model_folder_state, model_name, f"Selected model: {model_name}"

def browse_on_click(current_path):
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)  # Optional enhancement
    root.update_idletasks()            # Optional enhancement
    folder_selected = filedialog.askdirectory(initialdir=current_path or os.path.expanduser("~"))
    root.attributes('-topmost', False) # Optional enhancement
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
    # Insert new files at the beginning (top of the list)
    for file in reversed(new_files[:available_slots]):  # Reverse to maintain upload order
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

def start_new_session(models_loaded):
    import scripts.temporary as temporary
    import gradio as gr

    if not models_loaded:
        return (
            [],
            "Load model first on Configuration page...",
            gr.update(interactive=False),
            gr.update()
        )

    temporary.session_attached_files.clear()
    temporary.current_session_id = None
    temporary.session_label = ""
    temporary.SESSION_ACTIVE = True

    # Use non-priority status
    temporary.set_status("New session ready", console=False, priority=False)

    return (
        [],
        "Type input and click Send to begin...",
        gr.update(interactive=True),
        gr.update()
    )

def load_session_by_index(index):
    """Load a session by index from saved sessions."""
    sessions = utility.get_saved_sessions()
    if index < len(sessions):
        session_file = sessions[index]
        session_id, label, history, attached_files = utility.load_session_history(Path(HISTORY_DIR) / session_file)
        temporary.current_session_id = session_id
        temporary.session_label = label
        temporary.SESSION_ACTIVE = True
        return history, attached_files, f"Loaded session: {label}"
    return [], [], "No session to load"

def copy_last_response(session_log):
    """Copy last AI response to clipboard, excluding thinking phase"""
    if session_log and session_log[-1]['role'] == 'assistant':
        response = session_log[-1]['content']
        
        # Remove HTML tags
        clean_response = re.sub(r'<[^>]+>', '', response)
        
        # Remove "Thinking..." lines
        lines = clean_response.split('\n')
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip lines that are just "Thinking" followed by dots/spaces
            if stripped.startswith("Thinking") and all(c in '.… ' for c in stripped[8:]):
                continue
            filtered_lines.append(line)
        
        clean_response = '\n'.join(filtered_lines).strip()
        
        # Copy to clipboard
        pyperclip.copy(clean_response)
        return "AI Response copied to clipboard (thinking phase excluded)."
    return "No response available to copy."

def update_file_slot_ui(file_list, is_attach=True):
    from pathlib import Path
    button_updates = []
    max_slots = temporary.MAX_POSSIBLE_ATTACH_SLOTS if is_attach else temporary.MAX_POSSIBLE_ATTACH_SLOTS
    for i in range(max_slots):
        if i < len(file_list):
            filename = Path(file_list[i]).name
            short_name = (filename[:36] + ".." if len(filename) > 38 else filename)
            label = f"{short_name}"
            variant = "primary"
            visible = True
        else:
            label = ""
            variant = "primary"
            visible = False
        button_updates.append(gr.update(value=label, visible=visible, variant=variant))
    visible = len(file_list) < temporary.MAX_ATTACH_SLOTS if is_attach else True
    return button_updates + [gr.update(visible=visible)]

def filter_operational_content(text):
    """Remove operational tags and metadata from the text."""
    patterns = [
        r"ggml_vulkan:.*",
        r"load_tensors:.*",
        r"main:.*",
        r"Error executing CLI:.*",
        r"CLI Error:.*",
        r"build:.*",
        r"llama_model_load.*",
        r"print_info:.*",
        r"load:.*",
        r"llama_init_from_model:.*",
        r"llama_kv_cache_init:.*",
        r"sampler.*",
        r"eval:.*",
        r"embd_inp.size.*",
        r"waiting for user input",
        r"<think>.*?</think>",
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    return text.strip()

def update_session_buttons():
    sessions = utility.get_saved_sessions()[:temporary.MAX_HISTORY_SLOTS]
    button_updates = []
    for i in range(temporary.MAX_POSSIBLE_HISTORY_SLOTS):
        if i < len(sessions):
            session_path = Path(HISTORY_DIR) / sessions[i]
            try:
                stat = session_path.stat()
                update_time = stat.st_mtime if stat.st_mtime else stat.st_ctime
                formatted_time = datetime.fromtimestamp(update_time).strftime("%Y-%m-%d %H:%M")
                session_id, label, history, attached_files = utility.load_session_history(session_path)
                btn_label = f"{formatted_time} - {label}"
            except Exception as e:
                print(f"Error loading session {session_path}: {e}")
                btn_label = f"Session {i+1}"
            visible = True
        else:
            btn_label = ""
            visible = False
        button_updates.append(gr.update(value=btn_label, visible=visible))
    return button_updates

def format_session_id(session_id):
    """Format session ID into a readable date-time string."""
    try:
        dt = datetime.strptime(session_id, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return session_id

def update_action_buttons(phase):
    """
    Update all action buttons based on interaction phase.
    Returns tuple of 5 button updates.
    """
    # Check if there's history by looking at the actual session state
    # This will be set by the conversation_interface function
    from scripts import temporary
    
    # Simple check - if we're past initial state, we likely have history
    has_history = phase != "waiting_for_input" or temporary.SESSION_ACTIVE
    
    if phase == "waiting_for_input":
        return (
            gr.update(value="Send Input", variant="secondary", elem_classes=["send-button-green"], interactive=True),
            gr.update(visible=has_history),
            gr.update(visible=has_history),
            gr.update(visible=False),
            gr.update(visible=False)
        )
    elif phase == "afterthought_countdown":
        return (
            gr.update(value="..Please Wait..", variant="secondary", elem_classes=["send-button-orange"], interactive=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True)
        )
    elif phase == "generating_response":
        return (
            gr.update(value="..Wait For Response..", variant="secondary", elem_classes=["send-button-red"], interactive=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True)
        )
    elif phase == "speaking":
        return (
            gr.update(value="..Outputting Speech", variant="secondary", elem_classes=["send-button-orange"], interactive=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=True)
        )
    else:
        return (
            gr.update(value="Send Input", variant="secondary", elem_classes=["send-button-green"], interactive=True),
            gr.update(visible=has_history),
            gr.update(visible=has_history),
            gr.update(visible=False),
            gr.update(visible=False)
        )

def handle_rethink_prompt(session_log, user_input):
    """Cancel and preserve input for editing."""
    if len(session_log) >= 2:
        new_log = session_log[:-2]
    else:
        new_log = session_log
    
    return (
        new_log,
        "Input cancelled - edit and resend",
        user_input,  # Preserve input
        False,
        "waiting_for_input"
    )

def handle_cancel_input(session_log):
    """Cancel and clear input completely."""
    if len(session_log) >= 2:
        new_log = session_log[:-2]
    else:
        new_log = session_log
    
    return (
        new_log,
        "Input cancelled",
        "",  # Clear input
        False,
        "waiting_for_input"
    )

def update_cpu_threads_display():
    """Update the CPU threads slider display based on detected threads"""
    max_threads = max(1, temporary.CPU_LOGICAL_CORES - 1)
    current_threads = min(temporary.CPU_THREADS, max_threads) if temporary.CPU_THREADS else min(4, max_threads)
    return gr.update(maximum=max_threads, value=current_threads)

def save_configuration_settings():
    """Save all configuration settings including CPU threads"""
    try:
        # Update CPU threads from slider
        temporary.CPU_THREADS = temporary.CPU_THREADS or min(4, temporary.CPU_LOGICAL_CORES - 1)
        
        # Save all settings
        from scripts.settings import save_config
        save_config()
        
        return "Settings saved successfully."
    except Exception as e:
        return f"Error saving settings: {str(e)}"

def handle_cpu_threads_change(new_threads):
    """Handle CPU threads slider changes"""
    temporary.CPU_THREADS = int(new_threads)
    return f"CPU threads set to {new_threads}"

# ------------------------------------------------------------------
#  Global cancel object shared by every async call
# ------------------------------------------------------------------
import threading
_cancel_event = threading.Event()          # lives in interface.py
# ------------------------------------------------------------------

async def conversation_interface(
    user_input, session_log, loaded_files,
    is_reasoning_model, cancel_flag, web_search_enabled,
    interaction_phase, llm_state, models_loaded_state,
    speech_enabled
):
    """
    FIXED: All variables defined before use, proper phase management
    """
    # ---------------  embedded imports  ------------------------------
    import gradio as gr
    from scripts import temporary, utility
    from scripts.models import get_model_settings, get_response_stream
    from scripts.temporary import context_injector
    import asyncio, queue, threading, random, re, time
    from pathlib import Path

    # ---------------  early guards  ----------------------------------
    if not models_loaded_state or not llm_state:
        yield (session_log, "Please load a model first.",
               *update_action_buttons("waiting_for_input"),
               False, loaded_files, "waiting_for_input",
               gr.update(), gr.update(), gr.update())
        return

    if not user_input.strip():
        yield (session_log, "No input provided.",
               *update_action_buttons("waiting_for_input"),
               False, loaded_files, "waiting_for_input",
               gr.update(), gr.update(), gr.update())
        return

    # ---------------  initialize all variables  ----------------------
    original_input = user_input
    processed_input = user_input
    input_is_large = False
    error_occurred = False
    visible_resp = ""
    count_secs = 2 if is_reasoning_model else 1
    indicators = ["⏳", "🔄", "⚙️"]

    # ---------------  RAG processing if needed  ----------------------
    context_threshold = temporary.LARGE_INPUT_THRESHOLD
    max_input_chars = int(temporary.CONTEXT_SIZE * 3 * context_threshold)
    
    if len(user_input) > max_input_chars:
        input_is_large = True
        try:
            context_injector.add_temporary_input(user_input)
            processed_input = user_input[:1000] + "\n\n[Large input processed with RAG - full content indexed for retrieval]"
        except Exception as e:
            print(f"[RAG-TEMP] Error processing large input: {e}")
            processed_input = user_input[:max_input_chars]

    # ---------------  append to log  ---------------------------------
    if input_is_large or temporary.session_attached_files:
        session_log.append({'role':'user','content':processed_input})
    else:
        session_log.append({'role':'user','content':f"User:\n{processed_input}"})
    session_log.append({'role':'assistant','content':"AI-Chat:\n"})
    interaction_phase = "afterthought_countdown"

    yield (session_log,
           "Processing..."+(" (RAG)" if input_is_large else ""),
           *update_action_buttons(interaction_phase),
           cancel_flag, loaded_files, interaction_phase,
           gr.update(interactive=False), gr.update(), gr.update())

    # ---------------  countdown  -------------------------------------
    for i in range(count_secs, -1, -1):
        if cancel_flag or _cancel_event.is_set():
            session_log.pop(); session_log.pop()
            context_injector.clear_temporary_input()
            yield (session_log, "Input cancelled.",
                   *update_action_buttons("waiting_for_input"),
                   False, loaded_files, "waiting_for_input",
                   gr.update(interactive=True, value=original_input), gr.update(), gr.update())
            return

        yield (session_log, f"{random.choice(indicators)} Processing... {i}s",
               *update_action_buttons(interaction_phase),
               cancel_flag, loaded_files, interaction_phase,
               gr.update(), gr.update(), gr.update())
        await asyncio.sleep(1)

    # ---------------  web search  ------------------------------------
    search_results = None
    if web_search_enabled:
        yield (session_log, "🔍 Performing web search...",
               *update_action_buttons(interaction_phase),
               cancel_flag, loaded_files, interaction_phase,
               gr.update(), gr.update(), gr.update())
        try:
            search_results = utility.web_search(user_input, num_results=6)
            status = "✓ Web search complete"
        except Exception as e:
            search_results = f"Search error: {str(e)}"
            status = "⚠️ Web search failed"
        
        yield (session_log, status,
               *update_action_buttons(interaction_phase),
               cancel_flag, loaded_files, interaction_phase,
               gr.update(), gr.update(), gr.update())

    # ---------------  get model settings  ----------------------------
    model_settings = get_model_settings(temporary.MODEL_NAME)
    
    # ---------------  streaming  -------------------------------------
    interaction_phase = "generating_response"
    _cancel_event.clear()
    
    response_complete = False
    accumulated_response = "AI-Chat:\n"
    
    try:
        for chunk in get_response_stream(
            session_log=session_log,
            settings=model_settings,
            web_search_enabled=web_search_enabled,
            search_results=search_results,
            cancel_event=_cancel_event,
            llm_state=llm_state,
            models_loaded_state=models_loaded_state
        ):
            if _cancel_event.is_set() or cancel_flag:
                session_log[-1]['content'] = accumulated_response + "\n\n[Generation cancelled]"
                context_injector.clear_temporary_input()
                yield (session_log, "Generation cancelled.",
                       *update_action_buttons("waiting_for_input"),
                       False, loaded_files, "waiting_for_input",
                       gr.update(interactive=True), gr.update(), gr.update())
                return
            
            if chunk == "<CANCELLED>":
                break
                
            accumulated_response += chunk
            session_log[-1]['content'] = accumulated_response
            
            yield (session_log, "Generating response...",
                   *update_action_buttons(interaction_phase),
                   cancel_flag, loaded_files, interaction_phase,
                   gr.update(), gr.update(), gr.update())
        
        response_complete = True
        
    except Exception as e:
        error_occurred = True
        error_msg = f"\n\n[Error: {str(e)}]"
        session_log[-1]['content'] = accumulated_response + error_msg
        print(f"[ERROR] Response generation failed: {e}")

    # ---------------  extract visible response  ----------------------
    visible_resp = accumulated_response.replace("AI-Chat:\n", "", 1).strip()
    
    # Remove thinking tags for speech
    visible_resp = re.sub(r'<think>.*?</think>', '', visible_resp, flags=re.DOTALL)
    visible_resp = re.sub(r'<\|channel\|>analysis.*?(?:<\|end\|>|<\|channel\|>final)', '', visible_resp, flags=re.DOTALL)
    visible_resp = re.sub(r'<\|[^>]+\|>', '', visible_resp)
    
    # Remove "Thinking...." lines
    lines = visible_resp.split('\n')
    filtered_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Thinking") and all(c in '.… ' for c in stripped[8:]):
            continue
        filtered_lines.append(line)
    visible_resp = '\n'.join(filtered_lines).strip()

    # ---------------  speech  ----------------------------------------
    if speech_enabled and visible_resp and response_complete:
        interaction_phase = "speaking"
        yield (session_log, "Speaking response...",
               *update_action_buttons(interaction_phase),
               cancel_flag, loaded_files, interaction_phase,
               gr.update(), gr.update(), gr.update())
        
        try:
            chunks = utility.chunk_text_for_speech(visible_resp, max_chars=500)
            for chunk in chunks:
                if _cancel_event.is_set() or cancel_flag:
                    break
                utility.speak_text(chunk)
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[TTS] Speech error: {e}")

    # ---------------  cleanup  ---------------------------------------
    context_injector.clear_temporary_input()
    
    # ---------------  wrap-up  ---------------------------------------
    interaction_phase = "waiting_for_input"
    cleared_files = []
    utility.beep()
    
    yield (session_log,
           "✅ Response ready" if not error_occurred else "⚠️ Response incomplete",
           *update_action_buttons(interaction_phase),
           False, cleared_files, interaction_phase,
           gr.update(interactive=True, value=""),
           gr.update(value=web_search_enabled),
           gr.update(value=speech_enabled))

# Core Gradio Interface    
# Core Gradio Interface
def launch_interface():
    """Launch the Gradio interface for the Chat-Gradio-Gguf conversationbot with unified status bars."""
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
        """
    ) as demo:
        temporary.demo = demo
        model_folder_state = gr.State(temporary.MODEL_FOLDER)

        # SHARED STATUS STATE - ensures both tabs show same status
        shared_status_state = gr.State("Ready")

        states = dict(
            attached_files=gr.State([]),
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
            speech_enabled=gr.State(False)
        )
        config_components = {}
        custom_components = {}
        conversation_components = {}
        buttons = {}
        action_buttons = {}

        # Status update function for shared state
        def update_shared_status(message):
            return message, message, message

        with gr.Tabs():
            with gr.Tab("Interaction"):
                with gr.Row():
                    # LEFT COLLAPSIBLE PANEL
                    with gr.Column(visible=True, min_width=300, elem_classes=["clean-elements"]) as left_column_expanded:
                        toggle_button_left_expanded = gr.Button("Chat-Gradio-Gguf", variant="secondary")
                        gr.Markdown("**Dynamic Panel**")
                        panel_toggle = gr.Radio(choices=["History", "Attachments"], label="", value="History")
                        with gr.Group(visible=False) as attach_group:
                            attach_files = gr.UploadButton("Add Attach Files..", file_types=[f".{ext}" for ext in temporary.ALLOWED_EXTENSIONS], file_count="multiple", variant="secondary", elem_classes=["clean-elements"])
                            attach_slots = [gr.Button("Attach Slot Free", variant="huggingface", visible=False) for _ in range(temporary.MAX_POSSIBLE_ATTACH_SLOTS)]
                        with gr.Group(visible=True) as history_slots_group:
                            start_new_session_btn = gr.Button("Start New Session..", variant="secondary")
                            buttons = dict(session=[gr.Button(f"History Slot {i+1}", variant="huggingface", visible=False) for i in range(temporary.MAX_POSSIBLE_HISTORY_SLOTS)])

                    with gr.Column(visible=False, min_width=60, elem_classes=["clean-elements"]) as left_column_collapsed:
                        toggle_button_left_collapsed = gr.Button("CGG", variant="secondary", elem_classes=["clean-elements-normbot"])
                        new_session_btn_collapsed = gr.Button("New", variant="secondary", elem_classes=["clean-elements-normbot"])
                        add_attach_files_collapsed = gr.UploadButton("Add", file_types=[f".{ext}" for ext in temporary.ALLOWED_EXTENSIONS], file_count="multiple", variant="secondary", elem_classes=["clean-elements"])

                    # CENTER COLUMN (Session Log & User Input)
                    with gr.Column(scale=30, elem_classes=["clean-elements"]):
                        conversation_components["session_log"] = gr.Chatbot(label="Session Log", height=temporary.SESSION_LOG_HEIGHT, elem_classes=["scrollable"], type="messages")
                        initial_max_lines = max(3, int(((temporary.SESSION_LOG_HEIGHT - 100) / 10) / 2.5) - 6)
                        temporary.USER_INPUT_MAX_LINES = initial_max_lines
                        conversation_components["user_input"] = gr.Textbox(label="User Input", lines=3, max_lines=initial_max_lines, interactive=False, placeholder="Enter text here...")
                        with gr.Row(elem_classes=["clean-elements"]):
                            action_buttons["action"] = gr.Button("Send Input", variant="secondary", elem_classes=["send-button-green"], scale=10)
                            action_buttons["edit_previous"] = gr.Button("Edit Previous", variant="secondary", elem_classes=["send-button-orange"], scale=1, visible=False)
                            action_buttons["copy_response"] = gr.Button("Copy Output", variant="huggingface", scale=1, visible=False)
                            action_buttons["rethink_prompt"] = gr.Button("Rethink Prompt", variant="secondary", elem_classes=["send-button-orange"], scale=1, visible=False)
                            action_buttons["cancel_input"] = gr.Button("Cancel Input", variant="stop", elem_classes=["send-button-red"], scale=1, visible=False)

                    # RIGHT COLLAPSIBLE PANEL
                    with gr.Column(visible=True, min_width=300, elem_classes=["clean-elements"]) as right_column_expanded:
                        with gr.Row(elem_classes=["clean-elements"]):
                            toggle_button_right_expanded = gr.Button("Chat-Gradio-Gguf", variant="secondary", scale=10)
                        gr.Markdown("**Tools / Options**")
                        with gr.Row(elem_classes=["clean-elements"]):
                            action_buttons["web_search"] = gr.Button("🌐 Web-Search", variant="secondary", scale=1)
                            action_buttons["speech"] = gr.Button("🔊 Speech", variant="secondary", scale=1)

                    with gr.Column(visible=False, min_width=60, elem_classes=["clean-elements"]) as right_column_collapsed:
                        toggle_button_right_collapsed = gr.Button("CGG", variant="secondary", elem_classes=["clean-elements-normbot"])
                        action_buttons["web_search_collapsed"] = gr.Button("🌐", variant="secondary", elem_classes=["clean-elements-normbot"])
                        action_buttons["speech_collapsed"] = gr.Button("🔊", variant="secondary", elem_classes=["clean-elements-normbot"])

                with gr.Row():
                    interaction_global_status = gr.Textbox(
                        value="Ready",
                        label="Status",
                        interactive=False,
                        max_lines=1,
                        elem_classes=["clean-elements"],
                        scale=20
                    )
                    exit_interaction = gr.Button(
                         "Exit Program", variant="stop", elem_classes=["double-height"], scale=1
                    )
                    exit_interaction.click(
                        fn=shutdown_program,
                        inputs=[states["llm"], states["models_loaded"],
                                conversation_components["session_log"], states["attached_files"]],
                        outputs=[]
                    ).then(lambda: gr.update(visible=False), outputs=[demo])

            with gr.Tab("Configuration"):
                with gr.Column(scale=1, elem_classes=["clean-elements"]):
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("**Hardware**")
                    with gr.Row(elem_classes=["clean-elements"]) as cpu_row:
                        if temporary.VULKAN_AVAILABLE:
                            backend_radio = gr.Radio(
                                choices=["CPU-Only", "Vulkan"],
                                value=temporary.BACKEND_TYPE,
                                label="Execution Mode",
                                interactive=True,
                            )
                        else:
                            backend_radio = gr.Label(
                                value="CPU-Only",
                                label="Execution Mode",
                            )
                        cpu_labs = [c["label"] for c in utility.get_cpu_info()] or ["Default CPU"]
                        cpu_opts = ["Select..."] + cpu_labs if len(cpu_labs) > 1 else cpu_labs
                        def_cpu = temporary.SELECTED_CPU if temporary.SELECTED_CPU in cpu_opts else cpu_opts[0]
                        config_components["cpu"] = gr.Dropdown(
                            choices=cpu_opts,
                            label="CPU Device",
                            value=def_cpu,
                            scale=4
                        )
                        max_threads = max(temporary.CPU_THREAD_OPTIONS or [8])
                        config_components["cpu_threads"] = gr.Slider(
                            minimum=1,
                            maximum=max(temporary.CPU_THREAD_OPTIONS or [8]),
                            value=temporary.CPU_THREADS or min(4, max(temporary.CPU_THREAD_OPTIONS or [8])),
                            step=1,
                            label="Threads",
                            scale=3,
                            info=f"Available: {max(temporary.CPU_THREAD_OPTIONS or [8])} threads",
                            interactive=True
                        )
                    gpu_row_visible = temporary.VULKAN_AVAILABLE and temporary.BACKEND_TYPE == "Vulkan"
                    with gr.Row(elem_classes=["clean-elements"], visible=gpu_row_visible) as gpu_row:
                        gpus = utility.get_available_gpus()
                        gpu_choices = ["Auto-Select"] + gpus if gpus != ["CPU Only"] else ["CPU Only"]
                        def_gpu = temporary.SELECTED_GPU if temporary.SELECTED_GPU in gpu_choices else gpu_choices[0]
                        config_components["gpu"] = gr.Dropdown(
                            choices=gpu_choices, label="GPU", value=def_gpu, scale=10
                        )
                        config_components["vram"] = gr.Dropdown(
                            choices=temporary.VRAM_OPTIONS,
                            label="VRAM MB",
                            value=temporary.VRAM_SIZE,
                            scale=5,
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("**Model**")
                    with gr.Row(elem_classes=["clean-elements"]):
                        avail = temporary.AVAILABLE_MODELS or get_available_models()
                        mods = ["Select_a_model..."] + [m for m in avail if m != "Select_a_model..."]
                        def_m = temporary.MODEL_NAME if temporary.MODEL_NAME in mods else (mods[1] if len(mods) > 1 else mods[0])
                        config_components["model_path"] = gr.Textbox(
                            label="Folder", value=temporary.MODEL_FOLDER, interactive=False, scale=10
                        )
                        config_components["model"] = gr.Dropdown(
                            choices=mods, label="File", value=def_m, scale=10, info=".gguf"
                        )
                        keywords_display = gr.Textbox(label="Features", interactive=False, scale=10)
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components["ctx"] = gr.Dropdown(
                            temporary.CTX_OPTIONS, label="Context", value=temporary.CONTEXT_SIZE, scale=5
                        )
                        config_components["batch"] = gr.Dropdown(
                            temporary.BATCH_OPTIONS, label="Batch", value=temporary.BATCH_SIZE, scale=5
                        )
                        config_components["temp"] = gr.Dropdown(
                            temporary.TEMP_OPTIONS, label="Temp", value=temporary.TEMPERATURE, scale=5
                        )
                        config_components["repeat"] = gr.Dropdown(
                            temporary.REPEAT_OPTIONS, label="Repeat", value=temporary.REPEAT_PENALTY, scale=5
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        browse = gr.Button("📁 Browse", variant="secondary")
                        config_components["load"] = gr.Button("💾 Load", variant="primary")
                        config_components["inspect"] = gr.Button("🔍 Inspect")
                        config_components["unload"] = gr.Button("🗑️ Unload", variant="stop")
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("**Program**")
                    with gr.Row(elem_classes=["clean-elements"]):
                        custom_components["max_hist"] = gr.Dropdown(
                            temporary.HISTORY_SLOT_OPTIONS, label="Sessions", value=temporary.MAX_HISTORY_SLOTS, scale=5
                        )
                        custom_components["height"] = gr.Dropdown(
                            temporary.SESSION_LOG_HEIGHT_OPTIONS, label="Height", value=temporary.SESSION_LOG_HEIGHT, scale=5
                        )
                        custom_components["max_att"] = gr.Dropdown(
                            temporary.ATTACH_SLOT_OPTIONS, label="Attach", value=temporary.MAX_ATTACH_SLOTS, scale=5
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
                            elem_classes=["clean-elements"],
                            scale=20
                        )
                        exit_config = gr.Button(
                             "Exit Program", variant="stop", elem_classes=["double-height"], scale=1
                        )
                        exit_config.click(
                            fn=shutdown_program,
                            inputs=[states["llm"], states["models_loaded"],
                                    conversation_components["session_log"], states["attached_files"]],
                            outputs=[]
                        ).then(lambda: gr.update(visible=False), outputs=[demo])

        def handle_edit_previous(session_log):
            if len(session_log) < 2:
                return session_log, gr.update(), "No previous input to edit."
            new_log = session_log[:-2]
            last_user_input = session_log[-2]['content'].replace("User:\n", "", 1)
            return new_log, gr.update(value=last_user_input), "Previous input restored. Edit and resend."

        # Wire up shared status state to both status bars
        shared_status_state.change(
            fn=update_shared_status,
            inputs=[shared_status_state],
            outputs=[shared_status_state, interaction_global_status, config_global_status]
        )

        # Update temporary.set_status to use shared state
        def set_status_both(message, console=False):
            """Update both status bars via shared state"""
            return message

        # Store reference to shared status state for temporary module
        temporary.shared_status_state = shared_status_state
        temporary.set_status = lambda msg, console=False, priority=False: set_status_both(msg, console)

        model_folder_state.change(
            fn=lambda f: setattr(temporary, "MODEL_FOLDER", f) or None,
            inputs=[model_folder_state],
            outputs=[]
        ).then(
            fn=update_model_list,
            inputs=[model_folder_state],
            outputs=[config_components["model"]]
        ).then(
            fn=lambda f: f"Model directory updated to: {f}",
            inputs=[model_folder_state],
            outputs=[shared_status_state]
        )

        start_new_session_btn.click(
            fn=start_new_session,
            inputs=[states["models_loaded"]],
            outputs=[conversation_components["session_log"], shared_status_state, conversation_components["user_input"], states["web_search_enabled"]]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        ).then(
            fn=lambda: [],
            inputs=[],
            outputs=[states["attached_files"]]
        )

        new_session_btn_collapsed.click(
            fn=start_new_session,
            inputs=[states["models_loaded"]],
            outputs=[conversation_components["session_log"], shared_status_state, conversation_components["user_input"], states["web_search_enabled"]]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        ).then(
            fn=lambda: [],
            inputs=[],
            outputs=[states["attached_files"]]
        )

        add_attach_files_collapsed.upload(
            fn=process_attach_files,
            inputs=[add_attach_files_collapsed, states["attached_files"], states["models_loaded"]],
            outputs=[shared_status_state, states["attached_files"]]
        ).then(
            fn=lambda files: update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [add_attach_files_collapsed]
        )

        config_components["print_raw"].change(
            fn=lambda v: setattr(temporary, "PRINT_RAW_OUTPUT", bool(v)),
            inputs=[config_components["print_raw"]],
            outputs=[]
        )

        config_components["show_think_phase"].change(
            fn=lambda v: setattr(temporary, "SHOW_THINK_PHASE", bool(v)),
            inputs=[config_components["show_think_phase"]],
            outputs=[]
        )

        browse.click(
            fn=browse_on_click,
            inputs=[model_folder_state],
            outputs=[model_folder_state]
        ).then(
            fn=update_model_list,
            inputs=[model_folder_state],
            outputs=[config_components["model"]]
        ).then(
            fn=lambda f: f,
            inputs=[model_folder_state],
            outputs=[config_components["model_path"]]
        ).then(
            fn=lambda f: f"Model directory updated to: {f}",
            inputs=[model_folder_state],
            outputs=[shared_status_state]
        )

        action_buttons["action"].click(
            fn=lambda phase: True if phase == "generating_response" else False,
            inputs=[states["interaction_phase"]],
            outputs=[states["cancel_flag"]]
        ).then(
            fn=conversation_interface,
            inputs=[
                conversation_components["user_input"],
                conversation_components["session_log"],
                states["attached_files"],
                states["is_reasoning_model"],
                states["cancel_flag"],
                states["web_search_enabled"],
                states["interaction_phase"],
                states["llm"],
                states["models_loaded"],
                states["speech_enabled"]
            ],
            outputs=[
                conversation_components["session_log"],
                shared_status_state,
                action_buttons["action"],
                action_buttons["edit_previous"],
                action_buttons["copy_response"],
                action_buttons["rethink_prompt"],
                action_buttons["cancel_input"],
                states["cancel_flag"],
                states["attached_files"],
                states["interaction_phase"],
                conversation_components["user_input"],
                states["web_search_enabled"],
                states["speech_enabled"]
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

        action_buttons["rethink_prompt"].click(
            fn=handle_rethink_prompt,
            inputs=[conversation_components["session_log"], conversation_components["user_input"]],
            outputs=[
                conversation_components["session_log"],
                shared_status_state,
                conversation_components["user_input"],
                states["cancel_flag"],
                states["interaction_phase"]
            ]
        ).then(
            fn=update_action_buttons,
            inputs=[states["interaction_phase"]],
            outputs=[
                action_buttons["action"],
                action_buttons["edit_previous"],
                action_buttons["copy_response"],
                action_buttons["rethink_prompt"],
                action_buttons["cancel_input"]
            ]
        )

        action_buttons["cancel_input"].click(
            fn=handle_cancel_input,
            inputs=[conversation_components["session_log"]],
            outputs=[
                conversation_components["session_log"],
                shared_status_state,
                conversation_components["user_input"],
                states["cancel_flag"],
                states["interaction_phase"]
            ]
        ).then(
            fn=update_action_buttons,
            inputs=[states["interaction_phase"]],
            outputs=[
                action_buttons["action"],
                action_buttons["edit_previous"],
                action_buttons["copy_response"],
                action_buttons["rethink_prompt"],
                action_buttons["cancel_input"]
            ]
        )

        action_buttons["copy_response"].click(
            fn=copy_last_response,
            inputs=[conversation_components["session_log"]],
            outputs=[shared_status_state]
        )

        def toggle_web_search(enabled):
            new_state = not enabled
            variant = "primary" if new_state else "secondary"
            return new_state, gr.update(variant=variant), gr.update(variant=variant)

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

        def toggle_speech(enabled):
            new_state = not enabled
            variant = "primary" if new_state else "secondary"
            return new_state, gr.update(variant=variant), gr.update(variant=variant)

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

        action_buttons["edit_previous"].click(
            fn=handle_edit_previous,
            inputs=[conversation_components["session_log"]],
            outputs=[conversation_components["session_log"], conversation_components["user_input"], shared_status_state]
        )

        attach_files.upload(
            fn=process_attach_files,
            inputs=[attach_files, states["attached_files"], states["models_loaded"]],
            outputs=[shared_status_state, states["attached_files"]]
        ).then(
            fn=lambda files: update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        )

        for i, btn in enumerate(attach_slots):
            btn.click(
                fn=lambda files, idx=i: eject_file(files, idx, True),
                inputs=[states["attached_files"]],
                outputs=[states["attached_files"], shared_status_state] + attach_slots + [attach_files]
            )

        for i, btn in enumerate(buttons["session"]):
            btn.click(
                fn=load_session_by_index,
                inputs=[gr.State(value=i)],
                outputs=[conversation_components["session_log"], states["attached_files"], shared_status_state]
            ).then(
                fn=update_session_buttons,
                inputs=[],
                outputs=buttons["session"]
            ).then(
                fn=lambda files: update_file_slot_ui(files, True),
                inputs=[states["attached_files"]],
                outputs=attach_slots + [attach_files]
            )

        panel_toggle.change(
            fn=lambda panel: panel,
            inputs=[panel_toggle],
            outputs=[states["selected_panel"]]
        )

        config_components["model"].change(
            fn=handle_model_selection,
            inputs=[config_components["model"], model_folder_state],
            outputs=[model_folder_state, config_components["model"], shared_status_state]
        ).then(
            fn=lambda model_name: models.get_model_settings(model_name)["is_reasoning"],
            inputs=[config_components["model"]],
            outputs=[states["is_reasoning_model"]]
        ).then(
            fn=lambda model_name: models.get_model_settings(model_name),
            inputs=[config_components["model"]],
            outputs=[states["model_settings"]]
        ).then(
            fn=update_panel_choices,
            inputs=[states["model_settings"], states["selected_panel"]],
            outputs=[panel_toggle, states["selected_panel"]]
        ).then(
            fn=lambda model_settings: "none" if not model_settings.get("detected_keywords", []) else ", ".join(model_settings.get("detected_keywords", [])),
            inputs=[states["model_settings"]],
            outputs=[keywords_display]
        )

        config_components["cpu_threads"].change(
            fn=handle_cpu_threads_change,
            inputs=[config_components["cpu_threads"]],
            outputs=[shared_status_state]
        )

        states["selected_panel"].change(
            fn=lambda panel: (
                gr.update(visible=panel == "Attachments"),
                gr.update(visible=panel == "History")
            ),
            inputs=[states["selected_panel"]],
            outputs=[attach_group, history_slots_group]
        )

        for comp in [config_components[k] for k in ["ctx", "batch", "temp", "repeat", "vram", "gpu", "cpu_threads", "model"]]:
            comp.change(
                fn=update_config_settings,
                inputs=[config_components[k] for k in ["ctx", "batch", "temp", "repeat", "vram", "gpu", "cpu_threads", "model"]] + [config_components["print_raw"]],
                outputs=[shared_status_state]
            )

        if temporary.VULKAN_AVAILABLE:
            backend_radio.change(
                fn=lambda mode: [mode, gr.update(visible=(mode == "Vulkan"))],
                inputs=[backend_radio],
                outputs=[backend_radio, gpu_row]
            ).then(
                fn=lambda mode: setattr(temporary, "BACKEND_TYPE", mode) or None,
                inputs=[backend_radio],
                outputs=[]
            )

        config_components["unload"].click(
            fn=unload_models,
            inputs=[states["llm"], states["models_loaded"]],
            outputs=[shared_status_state, states["llm"], states["models_loaded"]]
        ).then(
            fn=lambda: gr.update(interactive=False),
            outputs=[conversation_components["user_input"]]
        )

        config_components["load"].click(
            fn=lambda: temporary.set_status("Loading...", console=True),
            outputs=[shared_status_state]
        ).then(
            fn=load_models,
            inputs=[model_folder_state, config_components["model"], config_components["vram"], states["llm"], states["models_loaded"]],
            outputs=[shared_status_state, states["models_loaded"], states["llm"], states["models_loaded"]]
        ).then(
            fn=lambda status, ml: (status, gr.update(interactive=ml)),
            inputs=[shared_status_state, states["models_loaded"]],
            outputs=[shared_status_state, conversation_components["user_input"]]
        )

        config_components["save_settings"].click(
            fn=lambda: settings.save_config(),
            inputs=[],
            outputs=[shared_status_state]
        )

        config_components["delete_all_history"].click(
            fn=utility.delete_all_session_histories,
            inputs=[],
            outputs=[shared_status_state]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        custom_components["height"].change(
            fn=set_session_log_base_height,
            inputs=[custom_components["height"]],
            outputs=[conversation_components["session_log"]]
        )

        custom_components["max_hist"].change(
            fn=lambda s: setattr(temporary, "MAX_HISTORY_SLOTS", s),
            inputs=[custom_components["max_hist"]],
            outputs=[]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        custom_components["max_att"].change(
            fn=lambda s: setattr(temporary, "MAX_ATTACH_SLOTS", s),
            inputs=[custom_components["max_att"]],
            outputs=[]
        ).then(
            fn=lambda files: update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        )

        # ----------------------------------------------------------
        #  LEFT PANEL TOGGLE
        # ----------------------------------------------------------
        def toggle_left_expanded_state(current_state):
            return not current_state

        toggle_button_left_expanded.click(
            fn=toggle_left_expanded_state,
            inputs=[states["left_expanded_state"]],
            outputs=[states["left_expanded_state"]]
        )

        toggle_button_left_collapsed.click(
            fn=toggle_left_expanded_state,
            inputs=[states["left_expanded_state"]],
            outputs=[states["left_expanded_state"]]
        )

        states["left_expanded_state"].change(
            fn=lambda state: [
                gr.update(visible=state),
                gr.update(visible=not state)
            ],
            inputs=[states["left_expanded_state"]],
            outputs=[left_column_expanded, left_column_collapsed]
        )

        # ----------------------------------------------------------
        #  RIGHT PANEL TOGGLE
        # ----------------------------------------------------------
        def toggle_right_expanded_state(current_state):
            return not current_state

        toggle_button_right_expanded.click(
            fn=toggle_right_expanded_state,
            inputs=[states["right_expanded_state"]],
            outputs=[states["right_expanded_state"]]
        )

        toggle_button_right_collapsed.click(
            fn=toggle_right_expanded_state,
            inputs=[states["right_expanded_state"]],
            outputs=[states["right_expanded_state"]]
        )

        states["right_expanded_state"].change(
            fn=lambda state: [
                gr.update(visible=state),
                gr.update(visible=not state)
            ],
            inputs=[states["right_expanded_state"]],
            outputs=[right_column_expanded, right_column_collapsed]
        )

        # ----------------------------------------------------------
        #  INITIAL LOAD
        # ----------------------------------------------------------
        demo.load(
            fn=get_initial_model_value,
            inputs=[],
            outputs=[config_components["model"], states["is_reasoning_model"]]
        ).then(
            fn=lambda model_name: models.get_model_settings(model_name),
            inputs=[config_components["model"]],
            outputs=[states["model_settings"]]
        ).then(
            fn=update_panel_choices,
            inputs=[states["model_settings"], states["selected_panel"]],
            outputs=[panel_toggle, states["selected_panel"]]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        ).then(
            fn=lambda files: update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        ).then(
            fn=lambda: gr.update(
                maximum=max(temporary.CPU_THREAD_OPTIONS or [8]),
                value=temporary.CPU_THREADS or min(4, max(temporary.CPU_THREAD_OPTIONS or [8]))
            ),
            inputs=[],
            outputs=[config_components["cpu_threads"]]
        ).then(
            fn=lambda: gr.update(value=temporary.PRINT_RAW_OUTPUT),
            inputs=[],
            outputs=[config_components["print_raw"]]
        ).then(
            fn=lambda: gr.update(value=temporary.SHOW_THINK_PHASE),
            inputs=[],
            outputs=[config_components["show_think_phase"]]
        ).then(
            fn=lambda model_settings: "none" if not model_settings.get("detected_keywords", []) else ", ".join(model_settings.get("detected_keywords", [])),
            inputs=[states["model_settings"]],
            outputs=[keywords_display]
        )

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        show_api=False,
        share=False,
    )

if __name__ == "__main__":
    launch_interface()
