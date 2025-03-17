# Script: `.\scripts\interface.py`

# Imports...
import gradio as gr
from gradio import themes
import re, os, json, pyperclip, yake, random, asyncio
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from queue import Queue
import scripts.temporary as temporary
from scripts.temporary import (
    USER_COLOR, THINK_COLOR, RESPONSE_COLOR, SEPARATOR, MID_SEPARATOR,
    ALLOWED_EXTENSIONS, CONTEXT_SIZE, VRAM_SIZE, SELECTED_GPU, SELECTED_CPU,
    current_model_settings, GPU_LAYERS, VRAM_OPTIONS, REPEAT_PENALTY,
    MLOCK, HISTORY_DIR, BATCH_OPTIONS, BATCH_SIZE, MODEL_FOLDER,
    MODEL_NAME, STATUS_TEXTS, CTX_OPTIONS, RP_LOCATION, USER_PC_NAME, USER_PC_ROLE,
    AI_NPC_NAME, AI_NPC_ROLE, SESSION_ACTIVE, TOT_VARIATIONS,
    MAX_HISTORY_SLOTS, MAX_ATTACH_SLOTS, HISTORY_SLOT_OPTIONS, ATTACH_SLOT_OPTIONS,
    BACKEND_TYPE, STREAM_OUTPUT
)
from scripts import utility
from scripts.utility import (
    delete_all_session_vectorstores, create_session_vectorstore, web_search, get_saved_sessions,
    load_session_history, save_session_history, load_and_chunk_documents,
    get_available_gpus, save_config, filter_operational_content
)
from scripts.models import (
    get_response_stream, get_available_models, unload_models, get_model_settings,
    context_injector, inspect_model, load_models
)
from langchain_core.documents import Document

# Functions...
def set_loading_status():
    return "Loading model..."

def update_panel_on_mode_change(mode, current_panel):
    mode = mode.lower()
    choices = ["History"]  # Base choice
    if mode == "chat":
        choices.extend(["Attach", "Vector"])
    elif mode == "coder":
        choices.append("Attach")
    elif mode == "rpg":
        choices.extend(["Vector", "Sheet"])
    
    new_panel = current_panel if current_panel in choices else choices[0]
    attach_visible = new_panel == "Attach"
    vector_visible = new_panel == "Vector"
    rpg_visible = new_panel == "Sheet" and mode == "rpg"
    history_visible = new_panel == "History"
    
    return (
        gr.update(choices=choices, value=new_panel),
        gr.update(visible=attach_visible),
        gr.update(visible=vector_visible),
        gr.update(visible=rpg_visible),
        gr.update(visible=history_visible),
        new_panel
    )

def process_attach_files(files, attached_files, models_loaded):
    if not models_loaded:
        return "Error: Load model first.", attached_files
    max_files = temporary.MAX_ATTACH_SLOTS
    if len(attached_files) >= max_files:
        return f"Max attach files ({max_files}) reached.", attached_files
    
    # Estimate blank prompt token count (using "chat" mode as default)
    blank_prompt = prompt_templates["chat"]
    blank_prompt_chars = len(blank_prompt)
    blank_prompt_tokens = (((blank_prompt_chars / 5) * 4) / 4) * 3
    
    current_total_tokens = blank_prompt_tokens
    context_limit = temporary.CONTEXT_SIZE  # Model's context size in tokens
    
    new_files = []
    rejected_files = []
    
    for f in files:
        if os.path.isfile(f) and f not in attached_files:
            try:
                with open(f, 'r', encoding='utf-8') as test_file:
                    content = test_file.read()
                file_chars = len(content)
                file_tokens = (((file_chars / 5) * 4) / 4) * 3
                
                if current_total_tokens + file_tokens <= context_limit:
                    new_files.append(f)
                    current_total_tokens += file_tokens
                else:
                    rejected_files.append(Path(f).name)
            except Exception:
                continue  # Skip unreadable files
    
    available_slots = max_files - len(attached_files)
    processed_files = new_files[:available_slots]
    
    for file in reversed(processed_files):
        dest = Path(temporary.TEMP_DIR) / f"session_{temporary.current_session_id}" / "attach" / Path(file).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, dest)
        attached_files.insert(0, str(dest))
    
    temporary.session_attached_files = attached_files
    status = f"Processed {len(processed_files)} attach files."
    if rejected_files:
        status += f" Rejected {len(rejected_files)} files due to context limit: {', '.join(rejected_files)}"
    return status, attached_files

def process_vector_files(files, vector_files, models_loaded):
    if not models_loaded:
        return "Error: Load model first.", vector_files
    new_files = [f for f in files if os.path.isfile(f) and f not in vector_files]
    for file in new_files:
        dest = Path(temporary.TEMP_DIR) / f"session_{temporary.current_session_id}" / "vector" / Path(file).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, dest)
        vector_files.append(str(dest))
    
    # Incremental update to vectorstore
    if context_injector.session_vectorstore is None:
        session_vectorstore = utility.create_session_vectorstore(vector_files, temporary.current_session_id)
    else:
        new_docs = utility.load_and_chunk_documents(new_files)
        if new_docs:
            context_injector.session_vectorstore.add_documents(new_docs)
            session_vectorstore = context_injector.session_vectorstore
        else:
            session_vectorstore = context_injector.session_vectorstore
    
    context_injector.set_session_vectorstore(session_vectorstore)
    temporary.session_vector_files = vector_files
    return f"Processed {len(new_files)} vector files.", vector_files

def update_config_settings(ctx, batch, temp, repeat, vram, gpu, cpu, model):
    temporary.CONTEXT_SIZE = int(ctx)
    temporary.BATCH_SIZE = int(batch)
    temporary.TEMPERATURE = float(temp)
    temporary.REPEAT_PENALTY = float(repeat)
    temporary.VRAM_SIZE = int(vram)
    temporary.SELECTED_GPU = gpu
    temporary.SELECTED_CPU = cpu
    temporary.MODEL_NAME = model
    status_message = (
        f"Updated settings: Context Size={ctx}, Batch Size={batch}, "
        f"Temperature={temp}, Repeat Penalty={repeat}, VRAM Size={vram}, "
        f"Selected GPU={gpu}, Selected CPU={cpu}, Model={model}"
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
    utility.save_config()
    return "Settings saved successfully."

def update_session_log_height(h):
    temporary.SESSION_LOG_HEIGHT = int(h)  # Update the variable
    print(f"Updated SESSION_LOG_HEIGHT to {h}")  # Optional: for debugging
    return gr.update(height=h)  # Update the UI

def update_input_lines(l):
    temporary.INPUT_LINES = int(l)
    print(f"Updated INPUT_LINES to {l}")  # Debugging
    return gr.update(lines=l)

def format_response(output: str) -> str:
    formatted = []
    code_blocks = re.findall(r'```(\w+)?\n(.*?)```', output, re.DOTALL)
    for lang, code in code_blocks:
        lexer = get_lexer_by_name(lang, stripall=True)
        formatted_code = highlight(code, lexer, HtmlFormatter())
        output = output.replace(f'```{lang}\n{code}```', formatted_code)
    return f'<span style="color: {RESPONSE_COLOR}">{output}</span>'

def get_initial_model_value():
    # Use cached AVAILABLE_MODELS instead of scanning again
    available_models = temporary.AVAILABLE_MODELS or get_available_models()  # Fallback if None
    if temporary.MODEL_NAME in available_models:
        return temporary.MODEL_NAME, get_model_settings(temporary.MODEL_NAME)["is_reasoning"]
    else:
        if len(available_models) == 1 and available_models[0] != "Browse_for_model_folder...":
            default_model = available_models[0]
            is_reasoning = get_model_settings(default_model)["is_reasoning"]
        else:
            default_model = "Browse_for_model_folder..."
            is_reasoning = False
        return default_model, is_reasoning

def update_model_list(new_dir):
    print(f"Updating model list with new_dir: {new_dir}")
    temporary.MODEL_FOLDER = new_dir
    choices = get_available_models()  # Scan the directory for models
    if choices and choices[0] != "Browse_for_model_folder...":  # If models are found
        value = choices[0]  # Default to the first model
    else:  # If no models are found
        choices = ["Browse_for_model_folder..."]  # Ensure this is in choices
        value = "Browse_for_model_folder..."  # Set value accordingly
    print(f"Choices returned: {choices}, Setting value to: {value}")
    return gr.update(choices=choices, value=value)

def handle_model_selection(model, model_folder_state):
    """Handle model selection with proper validation."""
    if model == "Browse_for_model_folder...":
        new_folder, model_update = browse_for_model_folder(model_folder_state)
        return new_folder, model_update, "Selecting directory..."
    elif model in ["Browse_for_model_folder...", "No models found"]:
        return model_folder_state, gr.update(), "Invalid model selection"
    else:
        temporary.MODEL_NAME = model
        return model_folder_state, gr.update(value=model), f"Selected model: {model}"

def select_directory(current_model_folder):
    """
    Open a directory selection dialog and return the selected path.

    Args:
        current_model_folder (str): The current model folder path.

    Returns:
        str: The selected directory path or the current path if none selected.
    """
    print("Opening directory selection dialog...")
    root = tk.Tk()
    root.withdraw()
    # Force the window to the foreground
    root.attributes('-topmost', True)
    root.update_idletasks()
    initial_dir = current_model_folder if current_model_folder and os.path.exists(current_model_folder) else os.path.expanduser("~")
    path = filedialog.askdirectory(initialdir=initial_dir)
    # Cleanup attributes after selection
    root.attributes('-topmost', False)
    root.destroy()
    if path:
        print(f"Selected path: {path}")
        return path
    else:
        print("No directory selected")
        return current_model_folder

def browse_for_model_folder(model_folder_state):
    new_folder = select_directory(model_folder_state)
    if new_folder:
        model_folder_state = new_folder
        temporary.MODEL_FOLDER = new_folder
        choices = get_available_models()  # Corrected
        if choices and choices[0] != "Browse_for_model_folder...":
            selected_model = choices[0]
        else:
            selected_model = "Browse_for_model_folder..."
        temporary.MODEL_NAME = selected_model
        return model_folder_state, gr.update(choices=choices, value=selected_model)
    else:
        return model_folder_state, gr.update(choices=["Browse_for_model_folder..."], value="Browse_for_model_folder...")

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
    utility.save_config()
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

def toggle_rpg_settings(showing_rpg_right):
    new_showing_rpg_right = not showing_rpg_right
    toggle_label = "Show File Attachments" if new_showing_rpg_right else "Show RPG Settings"
    return (
        gr.update(visible=True, value=toggle_label),
        gr.update(visible=not new_showing_rpg_right),
        gr.update(visible=new_showing_rpg_right),
        new_showing_rpg_right
    )

def create_session_label(text):
    first_line = text.split('\n')[0].strip()
    if len(first_line) > 30:
        last_space = first_line.rfind(' ', 0, 29)
        if last_space != -1:
            first_line = first_line[:last_space]
        else:
            first_line = first_line[:30]
    return first_line

def start_new_session(models_loaded):
    from scripts import temporary
    import gradio as gr
    if not models_loaded:
        return (
            [],                                # chat_components["session_log"]
            "Load model first on Configuration page...",  # status_text
            gr.update(interactive=False),     # chat_components["user_input"]
            gr.update(),                       # switches["web_search"]
            gr.update(),                       # switches["tot"]
            gr.update(),                       # switches["enable_think"]
            gr.update()                        # switches["speak"]
        )
    temporary.current_session_id = None
    temporary.session_label = ""
    temporary.SESSION_ACTIVE = True
    context_injector.set_session_vectorstore(None)  # Clear session vectorstore
    return (
        [],                                # chat_components["session_log"]
        "Type input and click Send to begin...",  # status_text
        gr.update(interactive=True),      # chat_components["user_input"]
        gr.update(),                       # switches["web_search"]
        gr.update(),                       # switches["tot"]
        gr.update(),                       # switches["enable_think"]
        gr.update()                        # switches["speak"]
    )

def load_session_by_index(index):
    sessions = utility.get_saved_sessions()
    if index < len(sessions):
        session_file = sessions[index]
        session_id, label, history, attached_files, vector_files = utility.load_session_history(Path(HISTORY_DIR) / session_file)
        temporary.current_session_id = session_id
        temporary.session_label = label
        temporary.SESSION_ACTIVE = True
        return history, attached_files, vector_files, f"Loaded session: {label}"
    return [], [], [], "No session to load"

def copy_last_response(session_log):
    if session_log and session_log[-1]['role'] == 'assistant':
        response = session_log[-1]['content']
        clean_response = re.sub(r'<[^>]+>', '', response)
        pyperclip.copy(clean_response)
        return "AI Response copied to clipboard."
    return "No response available to copy."

def shutdown_program(models_loaded):
    import time, sys
    if models_loaded:
        print("Shutting Down...")
        print("Unloading model...")
        unload_models()
        print("Model unloaded.")
    print("Closing Gradio server...")
    demo.close()
    print("Gradio server closed.")
    print("\n\nA program by Wiseman-Timelord\n")
    print("GitHub: github.com/wiseman-timelord")
    print("Website: wisetime.rf.gd\n\n")
    for i in range(5, 0, -1):
        print(f"\rExiting program in...{i}s", end='', flush=True)
        time.sleep(1)
    print()
    os._exit(0)

def update_mode_based_options(mode, showing_rpg_right, is_reasoning_model):
    mode = mode.lower()
    if mode == "coder":
        tot_visible = False
        web_visible = True
    elif mode == "rpg":
        tot_visible = False
        web_visible = False
    else:  # chat mode
        tot_visible = True
        web_visible = True
    think_visible = is_reasoning_model
    return [
        gr.update(visible=tot_visible),  # for switches["tot"]
        gr.update(visible=web_visible),  # for switches["web_search"]
        gr.update(visible=think_visible),  # for switches["enable_think"]
    ]

def update_model_based_options(model_name):
    if model_name in ["Browse_for_model_folder...", "No models found"]:
        mode = "Chat"
        think_visible = False
        recommended = "Select a model"
    else:
        settings = get_model_settings(model_name)
        mode = settings["category"].capitalize()
        think_visible = settings["is_reasoning"]
        recommended = mode
    return [
        gr.update(value=mode),
        gr.update(visible=think_visible),
        gr.update(value=recommended),
    ]

def update_file_slot_ui(file_list, is_attach=True):
    from pathlib import Path
    button_updates = []
    max_slots = temporary.MAX_POSSIBLE_ATTACH_SLOTS if is_attach else temporary.MAX_POSSIBLE_ATTACH_SLOTS  # Reuse for vector
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
    visible = len(file_list) < temporary.MAX_ATTACH_SLOTS if is_attach else True  # Vector has no limit UI-wise
    return button_updates + [gr.update(visible=visible)]


def filter_operational_content(text):
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
    ]
    lines = text.split("\n")
    filtered_lines = [line for line in lines if not any(re.search(pattern, line) for pattern in patterns)]
    return "\n".join(filtered_lines)

def update_session_buttons():
    sessions = utility.get_saved_sessions()[:temporary.MAX_HISTORY_SLOTS]  # Limit to MAX_HISTORY_SLOTS
    button_updates = []
    for i in range(temporary.MAX_POSSIBLE_HISTORY_SLOTS):
        if i < len(sessions):  # Only process existing sessions up to MAX_HISTORY_SLOTS
            session_path = Path(HISTORY_DIR) / sessions[i]
            try:
                stat = session_path.stat()
                update_time = stat.st_mtime if stat.st_mtime else stat.st_ctime
                formatted_time = datetime.fromtimestamp(update_time).strftime("%Y-%m-%d %H:%M")
                session_id, label, history, attached_files, vector_files = utility.load_session_history(session_path)
                if history and len(history) >= 2 and history[0]['role'] == 'user' and history[1]['role'] == 'assistant':
                    user_input = history[0]['content']
                    assistant_response = history[1]['content']
                    user_input_clean = utility.clean_content('user', user_input)
                    assistant_response_clean = utility.clean_content('assistant', assistant_response)
                    text_for_yake = user_input_clean + " " + assistant_response_clean
                else:
                    text_for_yake = " ".join([utility.clean_content(msg['role'], msg['content']) for msg in history])
                text_for_yake = filter_operational_content(text_for_yake)
                kw_extractor = yake.KeywordExtractor(lan="en", n=4, dedupLim=0.9, top=1)
                keywords = kw_extractor.extract_keywords(text_for_yake)
                description = keywords[0][0] if keywords else "No description"
                if len(description) > 16:
                    description = description[:16]
                temporary.yake_history_detail[i] = description
                btn_label = f"{formatted_time} - {description}"
            except Exception as e:
                print(f"Error loading session {session_path}: {e}")
                btn_label = f"Session {i+1}"
                temporary.yake_history_detail[i] = None
            visible = True
        else:
            btn_label = ""
            visible = False  # Hide slots beyond the number of sessions
        button_updates.append(gr.update(value=btn_label, visible=visible))
    return button_updates

def format_session_id(session_id):
    """Format session ID into a readable date-time string."""
    try:
        dt = datetime.strptime(session_id, "%Y%m%d_%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return session_id

def update_action_button(phase):
    if phase == "waiting_for_input":
        return gr.update(value="Send Input", variant="secondary")
    elif phase == "afterthought_countdown":
        return gr.update(value="Cancel Input", variant="stop")
    elif phase == "generating_response":
        return gr.update(value="Cancel Response", variant="stop")
    else:
        return gr.update(value="Unknown Phase", variant="primary")

def create_session_label(text):
    """Generate a ~30-character summary from the first line of the input."""
    first_line = text.split('\n')[0].strip()
    if len(first_line) > 30:
        last_space = first_line.rfind(' ', 0, 29)
        if last_space != -1:
            first_line = first_line[:last_space]
        else:
            first_line = first_line[:30]
    return first_line

async def chat_interface(user_input, session_log, tot_enabled, loaded_files, enable_think,
                        is_reasoning_model, rp_location, user_name, user_role, ai_npc,
                        cancel_flag, mode_selection, web_search_enabled, models_loaded,
                        interaction_phase, speak_enabled):
    from scripts import temporary, utility
    from scripts.temporary import STATUS_TEXTS, MODEL_NAME, SESSION_ACTIVE
    from scripts.models import get_model_settings, get_response_stream
    from scripts.utility import speak_text
    import asyncio
    import re
    from pathlib import Path

    if not models_loaded:
        yield session_log, "Please load a model first.", update_action_button("waiting_for_input"), cancel_flag, loaded_files, "waiting_for_input", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    # Start a new session if not active
    if not SESSION_ACTIVE:
        temporary.current_session_id = None
        temporary.session_label = ""
        temporary.SESSION_ACTIVE = True
        session_log = []
        yield session_log, "New session started.", update_action_button("waiting_for_input"), cancel_flag, loaded_files, "waiting_for_input", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        await asyncio.sleep(0.1)

    # Validate input
    if not user_input.strip():
        yield session_log, "No input provided.", update_action_button("waiting_for_input"), cancel_flag, loaded_files, "waiting_for_input", gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        return

    # Append user input and prepare assistant response
    session_log.append({'role': 'user', 'content': f"User:\n{user_input}"})
    session_log.append({'role': 'assistant', 'content': ""})
    interaction_phase = "afterthought_countdown"
    yield session_log, "Processing...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(value=""), gr.update(), gr.update(), gr.update(), gr.update()

    # Afterthought countdown
    if temporary.AFTERTHOUGHT_TIME:
        num_lines = len(user_input.split('\n'))
        countdown_seconds = 6 if num_lines >= 10 else 4 if num_lines >= 5 else 2
    else:
        countdown_seconds = 1
    for i in range(countdown_seconds, -1, -1):
        session_log[-1]['content'] = f"Afterthought countdown... {i}s"
        yield session_log, "Counting down...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        await asyncio.sleep(1)
        if cancel_flag:
            session_log[-1]['content'] = "Input cancelled."
            interaction_phase = "waiting_for_input"
            yield session_log, "Input cancelled.", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            return

    # Determine streaming behavior
    mode = mode_selection.lower()
    use_direct_streaming = (mode == "chat" and not tot_enabled and not web_search_enabled) or mode == "coder"

    # Define prefix based on mode
    prefix = "AI-Chat:" if mode == "chat" else "AI-Coder:" if mode == "coder" else f"{ai_npc}:" if mode == "rpg" else ""

    if not use_direct_streaming:
        if mode == "chat":
            if tot_enabled:
                generation_message = "Aggregating..."
            elif web_search_enabled:
                generation_message = "Researching..."
            else:
                generation_message = "Generating response..."
        elif mode == "rpg":
            generation_message = "Npc Taking Turn..."
        else:
            generation_message = "Generating..."
        session_log[-1]['content'] = generation_message
    else:
        session_log[-1]['content'] = prefix + "\n"  # Add newline after prefix for direct streaming
        yield session_log, "Generating...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    interaction_phase = "generating_response"
    yield session_log, "Generating...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    settings = get_model_settings(MODEL_NAME)

    if mode in ["chat", "rpg"]:
        search_results = None
        if web_search_enabled and mode == "chat":
            session_log[-1]['content'] = "Performing web search..."
            yield session_log, "Performing web search...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            search_results = utility.web_search(user_input)
            if search_results:
                session_log[-1]['content'] = "Web search completed. Generating response..."
            else:
                session_log[-1]['content'] = "No web search results found. Generating response..."
            yield session_log, session_log[-1]['content'], update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    rp_settings = {"rp_location": rp_location, "user_name": user_name, "user_role": user_role, "ai_npc": ai_npc, "ai_npc_role": temporary.AI_NPC_ROLE} if mode == "rpg" else None
    full_response = ""
    progress_count = 0

    async for chunk in get_response_stream(
        session_log,
        mode=mode,
        settings=settings,
        disable_think=not enable_think,
        rp_settings=rp_settings,
        tot_enabled=tot_enabled and mode == "chat",
        web_search_enabled=web_search_enabled and mode == "chat",
        search_results=search_results
    ):
        if cancel_flag:
            break
        full_response += chunk
        if use_direct_streaming:
            session_log[-1]['content'] += chunk  # Append chunk to prefix with newline
            yield session_log, "Generating...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        else:
            new_periods = chunk.count(".")
            if new_periods > 0:
                progress_count += new_periods
                progress_bar = "█" * progress_count
                session_log[-1]['content'] = f"{progress_bar} {generation_message}"
                yield session_log, "Generating...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        if mode == "chat" and tot_enabled and "</answer>" in full_response:
            break

    # Finalize response for non-direct streaming
    if not use_direct_streaming:
        if mode == "rpg":
            session_log[-1]['content'] = f"{prefix}\n{full_response}"
        elif mode == "chat" and tot_enabled:
            if "<answer>" in full_response and "</answer>" in full_response:
                answer_start = full_response.find("<answer>") + len("<answer>")
                answer_end = full_response.find("</answer>")
                final_answer = full_response[answer_start:answer_end].strip()
                progress_bar = "█" * progress_count
                session_log[-1]['content'] = f"{progress_bar} Aggregating...\n\n{prefix}\n{final_answer}"
            else:
                session_log[-1]['content'] = f"{prefix}\n{full_response}"  # Add newline
        else:
            session_log[-1]['content'] = f"{prefix}\n{full_response}"  # Add newline

    # Generate YAKE label after first response
    if len(session_log) >= 2 and session_log[-2]['role'] == 'user' and session_log[-1]['role'] == 'assistant' and not temporary.session_label:
        user_input_clean = utility.clean_content('user', session_log[-2]['content'])
        assistant_response_clean = utility.clean_content('assistant', session_log[-1]['content'])
        text_for_yake = user_input_clean + " " + assistant_response_clean
        text_for_yake = filter_operational_content(text_for_yake)
        kw_extractor = yake.KeywordExtractor(lan="en", n=4, dedupLim=0.9, top=1)
        keywords = kw_extractor.extract_keywords(text_for_yake)
        temporary.session_label = keywords[0][0] if keywords else "No description"

    # Save session with updated label
    utility.save_session_history(session_log, loaded_files, temporary.session_vector_files)

    # Speak functionality
    if speak_enabled:
        try:
            ai_response = session_log[-1]['content']
            if mode == "rpg":
                recent_events_match = re.search(r'<recent_events>(.*?)</recent_events>', ai_response, re.DOTALL)
                speak_content = recent_events_match.group(1).strip() if recent_events_match else ai_response
            elif mode == "chat":
                # Remove the prefix
                if f"{prefix}\n" in ai_response:
                    response_text = ai_response.split(f"{prefix}\n", 1)[1].strip()
                else:
                    response_text = ai_response.strip()
                
                # Check conditions for speaking only first and last paragraphs
                if not tot_enabled and not web_search_enabled and not settings["is_reasoning"] and not settings["is_uncensored"]:
                    # Extract first and last paragraphs
                    paragraphs = [p.strip() for p in response_text.split('\n\n') if p.strip()]
                    if len(paragraphs) > 1:
                        speak_content = paragraphs[0] + "\n\n" + paragraphs[-1]
                    elif paragraphs:
                        speak_content = paragraphs[0]
                    else:
                        speak_content = ""
                else:
                    speak_content = response_text
            else:
                speak_content = None
            if speak_content:
                print(f"DEBUG: Speaking {len(speak_content)} chars: {speak_content}")
                utility.speak_text(speak_content)
                session_log.append({'role': 'system', 'content': f"Speak Summary (AI Output):\n{speak_content}"})
                yield session_log, "Speaking AI output summary...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
        except Exception as e:
            error_msg = f"Error speaking AI output summary: {str(e)}"
            session_log.append({'role': 'system', 'content': error_msg})
            yield session_log, error_msg, update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    # Finalize interaction
    interaction_phase = "waiting_for_input"
    if cancel_flag:
        session_log[-1]['content'] = "Generation cancelled."
    yield session_log, STATUS_TEXTS["response_generated"], update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(value=""), gr.update(value=web_search_enabled), gr.update(value=tot_enabled), gr.update(value=enable_think), gr.update(value=speak_enabled)
    
def launch_interface():
    """Launch the Gradio interface for the Text-Gradio-Gguf chatbot with a split-screen layout."""
    global demo
    import tkinter as tk
    from tkinter import filedialog
    import os
    import gradio as gr
    from pathlib import Path
    from scripts import temporary, utility, models
    from scripts.temporary import (
        STATUS_TEXTS, MODEL_NAME, SESSION_ACTIVE, TOT_VARIATIONS,
        MAX_HISTORY_SLOTS, MAX_ATTACH_SLOTS, SESSION_LOG_HEIGHT, INPUT_LINES,
        AFTERTHOUGHT_TIME, MODEL_FOLDER, CONTEXT_SIZE, BATCH_SIZE, TEMPERATURE, REPEAT_PENALTY,
        VRAM_SIZE, SELECTED_GPU, SELECTED_CPU, MLOCK, BACKEND_TYPE,
        RP_LOCATION, USER_PC_NAME, USER_PC_ROLE, AI_NPC_NAME, AI_NPC_ROLE,
        ALLOWED_EXTENSIONS, VRAM_OPTIONS, CTX_OPTIONS, BATCH_OPTIONS, TEMP_OPTIONS,
        REPEAT_OPTIONS, HISTORY_SLOT_OPTIONS, SESSION_LOG_HEIGHT_OPTIONS,
        INPUT_LINES_OPTIONS, ATTACH_SLOT_OPTIONS
    )

    with gr.Blocks(title="Chat-Gradio-Gguf", css=".scrollable{overflow-y:auto}.send-button{background-color:green!important;color:white!important}.half-width{width:80px!important}.double-height{height:80px!important}.clean-elements{gap:4px!important;margin-bottom:4px!important}.clean-elements-normbot{gap:4px!important;margin-bottom:20px!important}") as demo:
        # Initialize state variables early
        model_folder_state = gr.State(temporary.MODEL_FOLDER)
        
        states = dict(
            attached_files=gr.State([]),
            vector_files=gr.State([]),
            models_loaded=gr.State(False),
            showing_rpg_right=gr.State(False),
            cancel_flag=gr.State(False),
            interaction_phase=gr.State("waiting_for_input"),
            is_reasoning_model=gr.State(False),
            selected_panel=gr.State("History"),
            expanded_state=gr.State(True)  # For collapsible left column
        )

        # Define chat_components once to avoid redefinition
        chat_components = {}

        with gr.Tabs():
            with gr.Tab("Conversation"):
                with gr.Row():
                    # Expanded left column
                    with gr.Column(visible=True, min_width=310, elem_classes=["clean-elements"]) as left_column_expanded:
                        toggle_button_expanded = gr.Button("Text-Gradio-Gguf", variant="secondary")
                        mode_selection = gr.Radio(
                            choices=["Chat", "Coder", "Rpg"],
                            label="Operation Mode",
                            value="Chat"
                        )
                        panel_toggle = gr.Radio(
                            choices=["History"],  # Dynamically updated by mode
                            label="Panel Mode",
                            value="History"
                        )
                        with gr.Group(visible=False) as attach_group:
                            attach_files = gr.UploadButton(
                                "Add Attach Files",
                                file_types=[f".{ext}" for ext in temporary.ALLOWED_EXTENSIONS],
                                file_count="multiple",
                                variant="secondary",
                                elem_classes=["clean-elements"]
                            )
                            attach_slots = [gr.Button(
                                "Attach Slot Free",
                                variant="huggingface",
                                visible=False
                            ) for _ in range(temporary.MAX_POSSIBLE_ATTACH_SLOTS)]
                        with gr.Group(visible=False) as vector_group:
                            vector_files_btn = gr.UploadButton(
                                "Add Vector Files",
                                file_types=[f".{ext}" for ext in temporary.ALLOWED_EXTENSIONS],
                                file_count="multiple",
                                variant="secondary",
                                elem_classes=["clean-elements"]
                            )
                            vector_slots = [gr.Button(
                                "Vector Slot Free",
                                variant="huggingface",
                                visible=False
                            ) for _ in range(temporary.MAX_POSSIBLE_ATTACH_SLOTS)]
                        with gr.Group(visible=True) as history_slots_group:
                            start_new_session_btn = gr.Button("Start New Session...", variant="secondary")
                            buttons = dict(
                                session=[gr.Button(
                                    f"History Slot {i+1}",
                                    variant="huggingface",
                                    visible=False
                                ) for i in range(temporary.MAX_POSSIBLE_HISTORY_SLOTS)]
                            )
                        with gr.Group(visible=False) as rpg_config_group:
                            rpg_fields = dict(
                                rp_location=gr.Textbox(label="RP Location", value=temporary.RP_LOCATION),
                                user_name=gr.Textbox(label="User Name", value=temporary.USER_PC_NAME),
                                user_role=gr.Textbox(label="User Role", value=temporary.USER_PC_ROLE),
                                ai_npc=gr.Textbox(label="AI NPC", value=temporary.AI_NPC_NAME),
                                ai_npc_role=gr.Textbox(label="AI Role", value=temporary.AI_NPC_ROLE),
                                save_rpg=gr.Button("Save RPG Settings", variant="primary")
                            )
                    
                    # Collapsed left column
                    with gr.Column(visible=False, min_width=80, elem_classes=["clean-elements"]) as left_column_collapsed:
                        toggle_button_collapsed = gr.Button("TGG", variant="secondary")
                    
                    # Main interaction column (split-screen effect)
                    with gr.Column(scale=30, elem_classes=["clean-elements"]):
                        with gr.Row(elem_classes=["clean-elements"]):
                            # Left side: Input area and controls
                            with gr.Row(elem_classes=["clean-elements"]):
                                with gr.Column(elem_classes=["clean-elements"]):
                                    with gr.Row(elem_classes=["clean-elements"]):
                                        switches = dict(
                                            web_search=gr.Checkbox(label="Search", value=False, visible=True),
                                            tot=gr.Checkbox(label="T.O.T.", value=False, visible=True),
                                            enable_think=gr.Checkbox(label="THINK", value=False, visible=False),
                                            speak=gr.Checkbox(label="Speak", value=False, visible=True)  # New Speak checkbox
                                        )
                                        # Mutual exclusion logic for web_search, tot, and enable_think (Speak is independent)
                                        switches["web_search"].change(
                                            fn=lambda search_value: [
                                                gr.update(value=False) if search_value else gr.update(),
                                                gr.update(value=False) if search_value else gr.update(),
                                                gr.update()  # Speak remains unaffected
                                            ],
                                            inputs=switches["web_search"],
                                            outputs=[switches["tot"], switches["enable_think"], switches["speak"]]
                                        )
                                        switches["tot"].change(
                                            fn=lambda tot_value: [
                                                gr.update(value=False) if tot_value else gr.update(),
                                                gr.update(value=False) if tot_value else gr.update(),
                                                gr.update()  # Speak remains unaffected
                                            ],
                                            inputs=switches["tot"],
                                            outputs=[switches["web_search"], switches["enable_think"], switches["speak"]]
                                        )
                                        switches["enable_think"].change(
                                            fn=lambda think_value: [
                                                gr.update(value=False) if think_value else gr.update(),
                                                gr.update(value=False) if think_value else gr.update(),
                                                gr.update()  # Speak remains unaffected
                                            ],
                                            inputs=switches["enable_think"],
                                            outputs=[switches["web_search"], switches["tot"], switches["speak"]]
                                        )
                                    with gr.Row(elem_classes=["clean-elements"]):
                                        chat_components["user_input"] = gr.Textbox(
                                            label="User Input",
                                            lines=temporary.INPUT_LINES,
                                            interactive=False,
                                            placeholder="Enter text here..."
                                        )
                                    with gr.Row(elem_classes=["clean-elements"]):
                                        action_buttons = {}
                                        action_buttons["action"] = gr.Button(
                                            "Send Input",
                                            variant="secondary",
                                            elem_classes=["send-button"],
                                            scale=10
                                        )
                                # Right side: Session Log
                                with gr.Column(elem_classes=["clean-elements"]):
                                    with gr.Row(elem_classes=["clean-elements"]):
                                        chat_components["session_log"] = gr.Chatbot(
                                            label="Session Log",
                                            height=temporary.SESSION_LOG_HEIGHT,
                                            elem_classes=["scrollable"],
                                            type="messages"
                                        )
                                    with gr.Row(elem_classes=["clean-elements"]):
                                        action_buttons["edit_previous"] = gr.Button("Edit Previous", variant="huggingface", scale=1)
                                        action_buttons["copy_response"] = gr.Button("Copy Output", variant="huggingface", scale=1)

                # Status bar
                with gr.Row():
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="Select model on Configuration page.",
                        scale=30
                    )
                    gr.Button("Terminate", variant="stop", elem_classes=["double-height"], min_width=110).click(
                        fn=shutdown_program,
                        inputs=[states["models_loaded"]]
                    )

            # Configuration tab
            with gr.Tab("Configuration"):
                with gr.Column(scale=1, elem_classes=["clean-elements"]):
                    is_cpu_only = temporary.BACKEND_TYPE in ["CPU Only - AVX2", "CPU Only - AVX512", "CPU Only - NoAVX", "CPU Only - OpenBLAS"]
                    config_components = {}
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("CPU/GPU Options...")
                    with gr.Row(visible=not is_cpu_only, elem_classes=["clean-elements"]):
                        gpu_choices = utility.get_available_gpus()
                        if len(gpu_choices) == 1:
                            default_gpu = gpu_choices[0]
                        else:
                            gpu_choices = ["Select_processing_device..."] + gpu_choices
                            default_gpu = temporary.SELECTED_GPU if temporary.SELECTED_GPU in gpu_choices else "Select_processing_device..."
                        config_components.update(
                            gpu=gr.Dropdown(choices=gpu_choices, label="Select GPU", value=default_gpu, scale=5),
                            vram=gr.Dropdown(choices=temporary.VRAM_OPTIONS, label="Assign Free VRam", value=temporary.VRAM_SIZE, scale=3),
                        )
                    with gr.Row(visible=is_cpu_only, elem_classes=["clean-elements"]):
                        cpu_choices = [cpu["label"] for cpu in utility.get_cpu_info()] or ["Default CPU"]
                        if len(cpu_choices) == 1:
                            default_cpu = cpu_choices[0]
                        else:
                            cpu_choices = ["Select_processing_device..."] + cpu_choices
                            default_cpu = temporary.SELECTED_CPU if temporary.SELECTED_CPU in cpu_choices else "Select_processing_device..."
                        config_components.update(
                            backend_type=gr.Textbox(label="Backend Type", value=temporary.BACKEND_TYPE, interactive=False, scale=5),
                            cpu=gr.Dropdown(choices=cpu_choices, label="Select CPU", value=default_cpu, scale=5),
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("Model Options...")
                    with gr.Row(elem_classes=["clean-elements"]):
                        available_models = temporary.AVAILABLE_MODELS
                        if available_models is None:
                            available_models = models.get_available_models()
                            print("Warning: AVAILABLE_MODELS was None, scanned models directory as fallback.")
                        if len(available_models) == 1 and available_models[0] != "Browse_for_model_folder...":
                            default_model = available_models[0]
                        elif available_models == ["Browse_for_model_folder..."] or not available_models:
                            available_models = ["Browse_for_model_folder..."]
                            default_model = "Browse_for_model_folder..."
                        else:
                            available_models = ["Select_a_model..."] + [m for m in available_models if m != "Browse_for_model_folder..."]
                            default_model = temporary.MODEL_NAME if temporary.MODEL_NAME in available_models else "Select_a_model..."
                        config_components.update(
                            model=gr.Dropdown(
                                choices=available_models,
                                label="Select Model",
                                value=default_model,
                                allow_custom_value=False,
                                scale=10
                            ),
                            ctx=gr.Dropdown(choices=temporary.CTX_OPTIONS, label="Context Size (Input/Aware)", value=temporary.CONTEXT_SIZE, scale=5),
                            batch=gr.Dropdown(choices=temporary.BATCH_OPTIONS, label="Batch Size (Output)", value=temporary.BATCH_SIZE, scale=5),
                            temp=gr.Dropdown(choices=temporary.TEMP_OPTIONS, label="Temperature (Creativity)", value=temporary.TEMPERATURE, scale=5),
                            repeat=gr.Dropdown(choices=temporary.REPEAT_OPTIONS, label="Repeat Penalty (Restraint)", value=temporary.REPEAT_PENALTY, scale=5),
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            browse=gr.Button("Browse", variant="secondary"), 
                            load_models=gr.Button("Load Model", variant="secondary"),
                            inspect_model=gr.Button("Inspect Model", variant="huggingface"),
                            unload=gr.Button("Unload Model", variant="huggingface"),
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("Program Options...")
                    with gr.Row(elem_classes=["clean-elements"]):
                        custom_components = {}
                        custom_components.update(
                            max_history_slots=gr.Dropdown(choices=temporary.HISTORY_SLOT_OPTIONS, label="Max History Slots", value=temporary.MAX_HISTORY_SLOTS, scale=5),
                            session_log_height=gr.Dropdown(choices=temporary.SESSION_LOG_HEIGHT_OPTIONS, label="Session Log Height", value=temporary.SESSION_LOG_HEIGHT, scale=5),
                            input_lines=gr.Dropdown(choices=temporary.INPUT_LINES_OPTIONS, label="Input Lines", value=temporary.INPUT_LINES, scale=5),
                            max_attach_slots=gr.Dropdown(choices=temporary.ATTACH_SLOT_OPTIONS, label="Max Attach Slots", value=temporary.MAX_ATTACH_SLOTS, scale=5)
                        )
                        with gr.Column(elem_classes=["clean-elements"]):
                            custom_components.update(
                                afterthought_time=gr.Checkbox(label="After-Thought Time", value=temporary.AFTERTHOUGHT_TIME),
                            )
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("Critical Actions...")
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            save_settings=gr.Button("Save Settings", variant="primary")
                        )
                        custom_components.update(
                            delete_all_vectorstores=gr.Button("Delete All History/Vectors", variant="stop")
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        with gr.Column(scale=1, elem_classes=["clean-elements"]):
                            gr.Markdown("About Program...")
                            gr.Markdown("[Text-Gradio-Gguf](https://github.com/wiseman-timelord/Text-Gradio-Gguf) by [Wiseman-Timelord](https://github.com/wiseman-timelord).")
                            gr.Markdown("Donations through, [Patreon](https://patreon.com/WisemanTimelord) or [Ko-fi](https://ko-fi.com/WisemanTimelord).")
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            status_settings=gr.Textbox(label="Status", interactive=False, scale=20),
                            shutdown=gr.Button("Terminate", variant="stop", elem_classes=["double-height"], min_width=110).click(fn=shutdown_program, inputs=[states["models_loaded"]])
                        )

        # Event handlers defined after all components are initialized
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
            outputs=[config_components["status_settings"]]
        )

        start_new_session_btn.click(
            fn=start_new_session,
            inputs=[states["models_loaded"]],
            outputs=[chat_components["session_log"], status_text, chat_components["user_input"], switches["web_search"], switches["tot"], switches["enable_think"], switches["speak"]]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        ).then(
            fn=lambda: ([], []),
            inputs=[],
            outputs=[states["attached_files"], states["vector_files"]]
        )

        action_buttons["action"].click(
            fn=chat_interface,
            inputs=[
                chat_components["user_input"],
                chat_components["session_log"],
                switches["tot"],
                states["attached_files"],
                switches["enable_think"],
                states["is_reasoning_model"],
                rpg_fields["rp_location"],
                rpg_fields["user_name"],
                rpg_fields["user_role"],
                rpg_fields["ai_npc"],
                states["cancel_flag"],
                mode_selection,
                switches["web_search"],
                states["models_loaded"],
                states["interaction_phase"],
                switches["speak"]
            ],
            outputs=[
                chat_components["session_log"],
                status_text,
                action_buttons["action"],
                states["cancel_flag"],
                states["attached_files"],
                states["interaction_phase"],
                chat_components["user_input"],
                switches["web_search"],
                switches["tot"],
                switches["enable_think"],
                switches["speak"]
            ]
        ).then(
            fn=lambda files: update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        action_buttons["copy_response"].click(
            fn=copy_last_response,
            inputs=[chat_components["session_log"]],
            outputs=[status_text]
        )

        attach_files.upload(
            fn=process_attach_files,
            inputs=[attach_files, states["attached_files"], states["models_loaded"]],
            outputs=[status_text, states["attached_files"]]
        ).then(
            fn=lambda files: update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        )

        vector_files_btn.upload(
            fn=process_vector_files,
            inputs=[vector_files_btn, states["vector_files"], states["models_loaded"]],
            outputs=[status_text, states["vector_files"]]
        ).then(
            fn=lambda files: update_file_slot_ui(files, False),
            inputs=[states["vector_files"]],
            outputs=vector_slots + [vector_files_btn]
        )

        for i, btn in enumerate(attach_slots):
            btn.click(
                fn=lambda files, idx=i: eject_file(files, idx, True),
                inputs=[states["attached_files"]],
                outputs=[states["attached_files"], status_text] + attach_slots + [attach_files]
            )

        for i, btn in enumerate(vector_slots):
            btn.click(
                fn=lambda files, idx=i: eject_file(files, idx, False),
                inputs=[states["vector_files"]],
                outputs=[states["vector_files"], status_text] + vector_slots + [vector_files_btn]
            )

        for i, btn in enumerate(buttons["session"]):
            btn.click(
                fn=load_session_by_index,
                inputs=[gr.State(value=i)],
                outputs=[chat_components["session_log"], states["attached_files"], states["vector_files"], status_text]
            ).then(
                fn=lambda: context_injector.load_session_vectorstore(temporary.current_session_id),
                inputs=[],
                outputs=[]
            ).then(
                fn=update_session_buttons,
                inputs=[],
                outputs=buttons["session"]
            ).then(
                fn=lambda files: update_file_slot_ui(files, True),
                inputs=[states["attached_files"]],
                outputs=attach_slots + [attach_files]
            ).then(
                fn=lambda files: update_file_slot_ui(files, False),
                inputs=[states["vector_files"]],
                outputs=vector_slots + [vector_files_btn]
            )

        def update_mode_based_options(mode, showing_rpg_right, is_reasoning_model):
            mode = mode.lower()
            tot_visible = mode == "chat" and not showing_rpg_right
            web_visible = mode == "chat" and not showing_rpg_right
            think_visible = mode == "chat" and is_reasoning_model and not showing_rpg_right
            speak_visible = mode in ["chat", "rpg"]  # Speak visible only in Chat and RPG
            return [
                gr.update(visible=tot_visible),
                gr.update(visible=web_visible),
                gr.update(visible=think_visible),
                gr.update(visible=speak_visible)
            ]

        mode_selection.change(
            fn=lambda mode: (
                gr.update(value=False),
                gr.update(value=False),
                gr.update(value=False) if models.get_model_settings(temporary.MODEL_NAME)["is_reasoning"] else gr.update(),
                gr.update()
            ),
            inputs=[mode_selection],
            outputs=[switches["web_search"], switches["tot"], switches["enable_think"], switches["speak"]]
        ).then(
            fn=update_panel_on_mode_change,
            inputs=[mode_selection, states["selected_panel"]],
            outputs=[panel_toggle, attach_group, vector_group, rpg_config_group, history_slots_group, states["selected_panel"]]
        ).then(
            fn=update_mode_based_options,
            inputs=[mode_selection, states["showing_rpg_right"], states["is_reasoning_model"]],
            outputs=[switches["tot"], switches["web_search"], switches["enable_think"], switches["speak"]]
        )

        panel_toggle.change(
            fn=lambda panel: panel,
            inputs=[panel_toggle],
            outputs=[states["selected_panel"]]
        ).then(
            fn=lambda panel, mode: (
                gr.update(visible=panel == "Attach"),
                gr.update(visible=panel == "Vector"),
                gr.update(visible=panel == "Sheet" and mode.lower() == "rpg"),
                gr.update(visible=panel == "History")
            ),
            inputs=[states["selected_panel"], mode_selection],
            outputs=[attach_group, vector_group, rpg_config_group, history_slots_group]
        )

        rpg_fields["save_rpg"].click(
            fn=save_rp_settings,
            inputs=list(rpg_fields.values())[:-1],
            outputs=list(rpg_fields.values())[:-1]
        )

        config_components["model"].change(
            fn=handle_model_selection,
            inputs=[config_components["model"], model_folder_state],
            outputs=[model_folder_state, config_components["model"], config_components["status_settings"]]
        ).then(
            fn=lambda model_name: models.get_model_settings(model_name)["is_reasoning"],
            inputs=[config_components["model"]],
            outputs=[states["is_reasoning_model"]]
        )

        for comp in [config_components[k] for k in ["ctx", "batch", "temp", "repeat", "vram", "gpu", "cpu", "model"]]:
            comp.change(
                fn=update_config_settings,
                inputs=[config_components[k] for k in ["ctx", "batch", "temp", "repeat", "vram", "gpu", "cpu", "model"]],
                outputs=[config_components["status_settings"]]
            )

        config_components["browse"].click(
            fn=browse_for_model_folder,
            inputs=[model_folder_state],
            outputs=[model_folder_state, config_components["model"]]
        ).then(
            fn=lambda f: f"Model directory updated to: {f}",
            inputs=[model_folder_state],
            outputs=[config_components["status_settings"]]
        )

        config_components["model"].change(
            fn=lambda model: (setattr(temporary, "MODEL_NAME", model), f"Selected model: {model}")[1],
            inputs=[config_components["model"]],
            outputs=[config_components["status_settings"]]
        )

        config_components["unload"].click(
            fn=models.unload_models,
            outputs=[config_components["status_settings"]]
        ).then(
            fn=lambda: False,
            outputs=[states["models_loaded"]]
        ).then(
            fn=lambda: gr.update(interactive=False),
            outputs=[chat_components["user_input"]]
        )

        config_components["load_models"].click(
            fn=set_loading_status,
            outputs=[config_components["status_settings"]]
        ).then(
            fn=load_models,
            inputs=[model_folder_state, config_components["model"], config_components["vram"]],
            outputs=[config_components["status_settings"], states["models_loaded"]]
        ).then(
            fn=lambda status, ml: (status, gr.update(interactive=ml)),
            inputs=[config_components["status_settings"], states["models_loaded"]],
            outputs=[config_components["status_settings"], chat_components["user_input"]]
        )

        config_components["inspect_model"].click(
            fn=inspect_model,
            inputs=[model_folder_state, config_components["model"], config_components["vram"]],
            outputs=[config_components["status_settings"]]
        )

        config_components["save_settings"].click(
            fn=save_all_settings,
            outputs=[config_components["status_settings"]]
        )

        custom_components["delete_all_vectorstores"].click(
            fn=utility.delete_all_history_and_vectors,
            outputs=[config_components["status_settings"]]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        custom_components["session_log_height"].change(
            fn=update_session_log_height,
            inputs=[custom_components["session_log_height"]],
            outputs=[chat_components["session_log"]]
        )

        custom_components["input_lines"].change(
            fn=update_input_lines,
            inputs=[custom_components["input_lines"]],
            outputs=[chat_components["user_input"]]
        )

        custom_components["max_history_slots"].change(
            fn=lambda s: (setattr(temporary, "MAX_HISTORY_SLOTS", s), 
                          setattr(temporary, "yake_history_detail", [None] * s)),
            inputs=[custom_components["max_history_slots"]],
            outputs=[]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        custom_components["max_attach_slots"].change(
            fn=lambda s: setattr(temporary, "MAX_ATTACH_SLOTS", s),
            inputs=[custom_components["max_attach_slots"]],
            outputs=[]
        ).then(
            fn=lambda files: update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        ).then(
            fn=lambda files: update_file_slot_ui(files, False),
            inputs=[states["vector_files"]],
            outputs=vector_slots + [vector_files_btn]
        )

        custom_components["afterthought_time"].change(
            fn=lambda v: setattr(temporary, "AFTERTHOUGHT_TIME", v),
            inputs=[custom_components["afterthought_time"]],
            outputs=[config_components["status_settings"]]
        )

        # Toggle function to switch expanded_state
        def toggle_expanded_state(current_state):
            return not current_state

        # Click events for toggle buttons
        toggle_button_expanded.click(
            fn=toggle_expanded_state,
            inputs=[states["expanded_state"]],
            outputs=[states["expanded_state"]]
        )

        toggle_button_collapsed.click(
            fn=toggle_expanded_state,
            inputs=[states["expanded_state"]],
            outputs=[states["expanded_state"]]
        )

        # Update column visibility when expanded_state changes
        states["expanded_state"].change(
            fn=lambda state: [
                gr.update(visible=state),
                gr.update(visible=not state)
            ],
            inputs=[states["expanded_state"]],
            outputs=[left_column_expanded, left_column_collapsed]
        )

        demo.load(
            fn=get_initial_model_value,
            inputs=[],
            outputs=[config_components["model"], states["is_reasoning_model"]]
        ).then(
            fn=update_panel_on_mode_change,
            inputs=[mode_selection, states["selected_panel"]],
            outputs=[panel_toggle, attach_group, vector_group, rpg_config_group, history_slots_group, states["selected_panel"]]
        ).then(
            fn=update_mode_based_options,
            inputs=[mode_selection, states["showing_rpg_right"], states["is_reasoning_model"]],
            outputs=[switches["tot"], switches["web_search"], switches["enable_think"], switches["speak"]]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        ).then(
            fn=lambda files: update_file_slot_ui(files, True),
            inputs=[states["attached_files"]],
            outputs=attach_slots + [attach_files]
        ).then(
            fn=lambda files: update_file_slot_ui(files, False),
            inputs=[states["vector_files"]],
            outputs=vector_slots + [vector_files_btn]
        )

    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, show_api=False)

if __name__ == "__main__":
    launch_interface()