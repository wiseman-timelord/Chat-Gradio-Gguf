# Script: `.\scripts\interface.py`

# Imports...
import gradio as gr
from gradio import themes
import re, os, json, pyperclip, yake, random, asyncio, queue, threading, time
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
    USER_COLOR, THINK_COLOR, RESPONSE_COLOR, SEPARATOR, MID_SEPARATOR,
    ALLOWED_EXTENSIONS, SELECTED_GPU, SELECTED_CPU,
    current_model_settings, GPU_LAYERS, SESSION_ACTIVE,
    HISTORY_DIR, MODEL_NAME, STATUS_TEXTS, BACKEND_TYPE, STREAM_OUTPUT
)
from scripts import utility
from scripts.utility import (
    web_search, get_saved_sessions,
    load_session_history, save_session_history,
    get_available_gpus, filter_operational_content, speak_text
)
from scripts.models import (
    get_response_stream, get_available_models, unload_models, get_model_settings, inspect_model, load_models
)

# Functions...
def set_loading_status():
    return "Loading model..."

def get_panel_choices(model_settings):
    """Determine available panel choices based on model settings."""
    choices = ["History", "Attach"]
    if model_settings.get("is_nsfw", False) or model_settings.get("is_roleplay", False):
        if "Attach" in choices:
            choices.remove("Attach")
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
    choices = ["History", "Attach"]
    new_panel = current_panel if current_panel in choices else choices[0]
    attach_visible = new_panel == "Attach"
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
    """
    Update the session log height in the UI.
    
    Args:
        h (int): New height value.
    
    Returns:
        gr.update: Updated UI component.
    """
    temporary.SESSION_LOG_HEIGHT = int(h)
    return gr.update(height=h)

def update_input_lines(l):
    temporary.INPUT_LINES = int(l)
    print(f"Updated INPUT_LINES to {l}")  # Debugging
    return gr.update(lines=l)

def format_response(output: str) -> str:
    formatted = []
    # Preserve think blocks during streaming
    think_blocks = re.findall(r'<think>(.*?)</think>', output, re.DOTALL)
    for thought in think_blocks:
        formatted.append(f'<span style="color: {THINK_COLOR}">[Thinking] {thought.strip()}</span>')
    
    # Process remaining content
    clean_output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
    code_blocks = re.findall(r'```(\w+)?\n(.*?)```', clean_output, re.DOTALL)
    for lang, code in code_blocks:
        lexer = get_lexer_by_name(lang, stripall=True)
        formatted_code = highlight(code, lexer, HtmlFormatter())
        output = output.replace(f'```{lang}\n{code}```', formatted_code)
    return '<br>'.join(formatted) + final_output

def get_initial_model_value():
    available_models = temporary.AVAILABLE_MODELS or get_available_models()
    base_choices = ["Select_a_model...", "Browse_for_model_folder..."]
    if available_models and available_models != ["Browse_for_model_folder..."]:
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
    if choices and choices[0] != "Browse_for_model_folder...":  # If models are found
        value = choices[0]  # Default to the first model
    else:  # If no models are found
        choices = ["Browse_for_model_folder..."]  # Ensure this is in choices
        value = "Browse_for_model_folder..."  # Set value accordingly
    print(f"Choices returned: {choices}, Setting value to: {value}")
    return gr.update(choices=choices, value=value)

def handle_model_selection(model_name, model_folder_state):
    if not model_name:
        return model_folder_state, model_name, "No model selected."
    return model_folder_state, model_name, f"Selected model: {model_name}"

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

def start_new_session(models_loaded):
    from scripts import temporary
    import gradio as gr
    if not models_loaded:
        return (
            [],                                # session_log
            "Load model first on Configuration page...",  # status_text
            gr.update(interactive=False),     # user_input
            gr.update(),                       # web_search
            gr.update()                        # speak
        )
    temporary.current_session_id = None
    temporary.session_label = ""
    temporary.SESSION_ACTIVE = True
    return (
        [],                                # session_log
        "Type input and click Send to begin...",  # status_text
        gr.update(interactive=True),      # user_input
        gr.update(),                       # web_search
        gr.update()                        # speak
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
    if session_log and session_log[-1]['role'] == 'assistant':
        response = session_log[-1]['content']
        clean_response = re.sub(r'<[^>]+>', '', response)
        pyperclip.copy(clean_response)
        return "AI Response copied to clipboard."
    return "No response available to copy."

def shutdown_program(llm_state, models_loaded_state):
    import time, sys
    if models_loaded_state:
        print("Shutting Down...")
        print("Unloading model...")
        unload_models(llm_state, models_loaded_state)
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

def update_action_button(phase):
    if phase == "waiting_for_input":
        return gr.update(value="Send Input", variant="secondary", elem_classes=["send-button-green"], interactive=True)
    elif phase == "afterthought_countdown":
        return gr.update(value="Cancel Submission", variant="secondary", elem_classes=["send-button-orange"], interactive=False)
    elif phase == "generating_response":
        return gr.update(value="Wait For Response", variant="secondary", elem_classes=["send-button-red"], interactive=True)  # Interactive for cancellation
    elif phase == "speaking":
        return gr.update(value="Outputting Speak", variant="secondary", elem_classes=["send-button-orange"], interactive=False)
    else:
        return gr.update(value="Unknown Phase", variant="secondary", elem_classes=["send-button-green"], interactive=False)

# Async Converstation Interface
async def conversation_interface(user_input, session_log, loaded_files,
                                 is_reasoning_model, cancel_flag, web_search_enabled,
                                 interaction_phase, speak_enabled, llm_state, models_loaded_state):
    """
    Handle user input and generate AI responses asynchronously for the Chat-Gradio-Gguf interface.
    
    Args:
        user_input (str): The user's input text.
        session_log (list): List of conversation history.
        loaded_files (list): List of attached files.
        is_reasoning_model (bool): Whether the model is a reasoning model.
        cancel_flag (bool): Flag to cancel generation.
        web_search_enabled (bool): Whether web search is enabled.
        interaction_phase (str): Current phase of interaction.
        speak_enabled (bool): Whether text-to-speech is enabled.
        llm_state (Llama): The loaded model instance.
        models_loaded_state (bool): Whether the model is loaded.
    
    Yields:
        tuple: 9 values corresponding to UI updates (reduced from 10 due to fewer switches).
    """
    import gradio as gr
    from scripts import temporary, utility
    from scripts.models import get_model_settings, get_response_stream
    from scripts.utility import filter_operational_content, speak_text
    import asyncio
    import queue
    import threading
    from pathlib import Path
    import random

    # Check if model is loaded
    if not models_loaded_state or llm_state is None:
        yield session_log, "Please load a model first.", update_action_button(interaction_phase), False, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
        return

    # Check for empty input
    if not user_input.strip():
        yield session_log, "No input provided.", update_action_button(interaction_phase), False, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
        return

    # Prepare user input with attached files
    original_input = user_input
    if temporary.session_attached_files:
        for file in temporary.session_attached_files:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                user_input += f"\n\nAttached File Content ({Path(file).name}):\n{file_content}"
            except Exception as e:
                print(f"Error reading attached file {file}: {e}")

    # Append user message and placeholder assistant response
    session_log.append({'role': 'user', 'content': f"User:\n{user_input}"})
    session_log.append({'role': 'assistant', 'content': ""})
    interaction_phase = "afterthought_countdown"
    yield session_log, "Processing...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(interactive=False), gr.update(), gr.update()

    # Afterthought countdown
    input_length = len(original_input.strip())
    countdown_seconds = 1 if input_length <= 25 else 3 if input_length <= 100 else 5
    progress_indicators = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    
    for i in range(countdown_seconds, -1, -1):
        current_progress = random.choice(progress_indicators)
        yield session_log, f"{current_progress} Afterthought countdown... {i}s", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
        await asyncio.sleep(1)
        if cancel_flag:
            session_log.pop()
            interaction_phase = "waiting_for_input"
            yield session_log, "Input cancelled.", update_action_button(interaction_phase), False, loaded_files, interaction_phase, gr.update(interactive=True, value=original_input), gr.update(), gr.update()
            return

    # Begin response generation
    prefix = "AI-Chat:"
    interaction_phase = "generating_response"
    settings = get_model_settings(temporary.MODEL_NAME)

    # Handle web search if enabled
    search_results = None
    if web_search_enabled:
        session_log[-1]['content'] = "WORKFLOW:\nPRODUCE WEBSEARCH ---> PRODUCE ASSESSMENT"
        yield session_log, "🔍 Performing web search...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
        search_results = await asyncio.to_thread(utility.web_search, user_input)
        yield session_log, "✅ Web search completed." if search_results else "⚠️ No web results.", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()

    # Set up streaming
    q = queue.Queue()
    cancel_event = threading.Event()
    visible_response = ""

    def run_generator():
        try:
            for chunk in get_response_stream(
                session_log,
                settings=settings,
                web_search_enabled=web_search_enabled,
                search_results=search_results,
                cancel_event=cancel_event,
                llm_state=llm_state,
                models_loaded_state=models_loaded_state
            ):
                q.put(chunk)
            q.put(None)
        except Exception as e:
            q.put(f"Error: {str(e)}")

    thread = threading.Thread(target=run_generator, daemon=True)
    thread.start()

    # Process stream with immediate UI updates
    while True:
        try:
            chunk = q.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.01)  # Optimized delay when queue is empty
            continue

        if chunk is None:
            break

        if cancel_flag:
            cancel_event.set()
            session_log[-1]['content'] = "Generation cancelled."
            yield session_log, "Cancelled", update_action_button(interaction_phase), False, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
            break

        if chunk == "<CANCELLED>":
            session_log[-1]['content'] = "Generation cancelled."
            yield session_log, "Cancelled", update_action_button(interaction_phase), False, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
            break

        if isinstance(chunk, str) and chunk.startswith("Error:"):
            session_log[-1]['content'] = chunk
            yield session_log, f"⚠️ {chunk}", update_action_button("waiting_for_input"), False, loaded_files, "waiting_for_input", gr.update(interactive=True), gr.update(), gr.update()
            return

        # Stream updates (no hidden sentences or TOT logic)
        visible_response += chunk
        session_log[-1]['content'] = f"{prefix}\n{visible_response}"
        yield session_log, "Streaming Response...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update(), gr.update()
        await asyncio.sleep(0.05)  # Allow UI to update

    # Finalize response
    if visible_response:
        final_content = filter_operational_content(f"{prefix}\n{visible_response}")
        session_log[-1]['content'] = final_content
        utility.save_session_history(session_log, temporary.session_attached_files)  # No vector files
    else:
        session_log[-1]['content'] = f"{prefix}\n(Empty response)"

    # Handle text-to-speech if enabled
    if speak_enabled and visible_response:
        summary_prompt = f"Summarize the following response in under 256 characters:\n\n{visible_response}"
        summary_response = llm_state.create_chat_completion(
            messages=[{"role": "user", "content": summary_prompt}],
            max_tokens=256,
            temperature=0.5,
            stream=False
        )
        summary = summary_response['choices'][0]['message']['content'].strip()
        if len(summary) > 256:
            summary = summary[:253] + "..."
        threading.Thread(target=speak_text, args=(summary,), daemon=True).start()

    interaction_phase = "waiting_for_input"
    yield session_log, "✅ Response ready", update_action_button(interaction_phase), False, loaded_files, interaction_phase, gr.update(interactive=True, value=""), gr.update(), gr.update()

# Core Gradio Interface    
def launch_interface():
    """Launch the Gradio interface for the Chat-Gradio-Gguf conversationbot with a split-screen layout."""
    global demo
    import tkinter as tk
    from tkinter import filedialog
    import os
    import gradio as gr
    from pathlib import Path
    from scripts import temporary, utility, models
    from scripts.temporary import (
        STATUS_TEXTS, MODEL_NAME, SESSION_ACTIVE,
        MAX_HISTORY_SLOTS, MAX_ATTACH_SLOTS, SESSION_LOG_HEIGHT, INPUT_LINES,
        MODEL_FOLDER, CONTEXT_SIZE, BATCH_SIZE, TEMPERATURE, REPEAT_PENALTY,
        VRAM_SIZE, SELECTED_GPU, SELECTED_CPU, MLOCK, BACKEND_TYPE,
        ALLOWED_EXTENSIONS, VRAM_OPTIONS, CTX_OPTIONS, BATCH_OPTIONS, TEMP_OPTIONS,
        REPEAT_OPTIONS, HISTORY_SLOT_OPTIONS, SESSION_LOG_HEIGHT_OPTIONS,
        INPUT_LINES_OPTIONS, ATTACH_SLOT_OPTIONS
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
        """
    ) as demo:
        # Initialize state variables early
        model_folder_state = gr.State(temporary.MODEL_FOLDER)
        
        states = dict(
            attached_files=gr.State([]),
            models_loaded=gr.State(False),
            llm=gr.State(None),
            cancel_flag=gr.State(False),
            interaction_phase=gr.State("waiting_for_input"),
            is_reasoning_model=gr.State(False),
            selected_panel=gr.State("History"),
            expanded_state=gr.State(True),
            model_settings=gr.State({})  # Store full model settings
        )
        # Define conversation_components once to avoid redefinition
        conversation_components = {}

        with gr.Tabs():
            with gr.Tab("Interaction"):
                with gr.Row():
                    # Expanded left column
                    with gr.Column(visible=True, min_width=350, elem_classes=["clean-elements"]) as left_column_expanded:
                        toggle_button_expanded = gr.Button("Chat-Gradio-Gguf", variant="secondary")
                        panel_toggle = gr.Radio(
                            choices=["History", "Attach"],  # Removed "Vector"
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
                        with gr.Group(visible=True) as history_slots_group:
                            start_new_session_btn = gr.Button("Start New Session...", variant="secondary")
                            buttons = dict(
                                session=[gr.Button(
                                    f"History Slot {i+1}",
                                    variant="huggingface",
                                    visible=False
                                ) for i in range(temporary.MAX_POSSIBLE_HISTORY_SLOTS)]
                            )
                    
                    # Collapsed left column
                    with gr.Column(visible=False, min_width=60, elem_classes=["clean-elements"]) as left_column_collapsed:
                        toggle_button_collapsed = gr.Button("CGG", variant="secondary", elem_classes=["clean-elements-normbot"])
                        new_session_btn_collapsed = gr.Button("New", variant="secondary")
                    
                    # Main interaction column (split-screen effect)
                    with gr.Column(scale=30, elem_classes=["clean-elements"]):
                        with gr.Row(elem_classes=["clean-elements"]):
                            # Left side: Input area and controls
                            with gr.Row(elem_classes=["clean-elements"]):
                                with gr.Column(elem_classes=["clean-elements"]):
                                    with gr.Row(elem_classes=["clean-elements"]):
                                        switches = dict(
                                            web_search=gr.Checkbox(label="Search", value=False, visible=True),
                                            speak=gr.Checkbox(label="Speak", value=False, visible=True)  # Removed "tot"
                                        )
                                    with gr.Row(elem_classes=["clean-elements"]):
                                        conversation_components["user_input"] = gr.Textbox(
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
                                            elem_classes=["send-button-green"],
                                            scale=10
                                        )
                                # Right side: Session Log
                                with gr.Column(elem_classes=["clean-elements"]):
                                    with gr.Row(elem_classes=["clean-elements"]):
                                        conversation_components["session_log"] = gr.Chatbot(
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
                    exit_button = gr.Button("Exit", variant="stop", elem_classes=["double-height"], min_width=110)
                    exit_button.click(
                        fn=shutdown_program,
                        inputs=[states["llm"], states["models_loaded"]]
                    )

            # Configuration tab
            with gr.Tab("Configuration"):
                with gr.Column(scale=1, elem_classes=["clean-elements"]):
                    is_cpu_only = temporary.BACKEND_TYPE in ["CPU Only - AVX2", "CPU Only - AVX512", "CPU Only - NoAVX", "CPU Only - OpenBLAS"]
                    config_components = {}
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("CPU/GPU Options...")
                    # GPU options row, visible when not CPU-only
                    with gr.Row(visible=not is_cpu_only, elem_classes=["clean-elements"]):
                        config_components.update(
                            backend_type=gr.Textbox(label="Backend Type", value=temporary.BACKEND_TYPE, interactive=False, scale=3),
                        )
                        gpu_choices = utility.get_available_gpus()
                        if len(gpu_choices) == 1:
                            default_gpu = gpu_choices[0]
                        else:
                            gpu_choices = ["Select_processing_device..."] + gpu_choices
                            default_gpu = temporary.SELECTED_GPU if temporary.SELECTED_GPU in gpu_choices else "Select_processing_device..."
                        config_components.update(
                            gpu=gr.Dropdown(choices=gpu_choices, label="Select GPU", value=default_gpu, scale=4),
                            vram=gr.Dropdown(choices=temporary.VRAM_OPTIONS, label="Assign Free VRam", value=temporary.VRAM_SIZE, scale=2),
                        )
                    # CPU options row, visible when CPU-only
                    with gr.Row(visible=is_cpu_only, elem_classes=["clean-elements"]):
                        config_components.update(
                            backend_type=gr.Textbox(label="Backend Type", value=temporary.BACKEND_TYPE, interactive=False, scale=3),
                        )
                        cpu_choices = [cpu["label"] for cpu in utility.get_cpu_info()] or ["Default CPU"]
                        if len(cpu_choices) == 1:
                            default_cpu = cpu_choices[0]
                        else:
                            cpu_choices = ["Select_processing_device..."] + cpu_choices
                            default_cpu = temporary.SELECTED_CPU if temporary.SELECTED_CPU in cpu_choices else "Select_processing_device..."
                        config_components.update(
                            cpu=gr.Dropdown(choices=cpu_choices, label="Select CPU", value=default_cpu, scale=4),
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("Model Options...")
                    with gr.Row(elem_classes=["clean-elements"]):
                        available_models = temporary.AVAILABLE_MODELS
                        if available_models is None:
                            available_models = models.get_available_models()
                            print("Warning: AVAILABLE_MODELS was None, scanned models directory as fallback.")
                        # Ensure "Browse_for_model_folder..." is always an option
                        base_choices = ["Select_a_model...", "Browse_for_model_folder..."]
                        if available_models and available_models != ["Browse_for_model_folder..."]:
                            available_models = [m for m in available_models if m not in base_choices]
                            available_models = base_choices + available_models
                        else:
                            available_models = base_choices
                        # Set default_model, preferring temporary.MODEL_NAME if valid
                        if temporary.MODEL_NAME in available_models and temporary.MODEL_NAME not in ["Select_a_model...", "Browse_for_model_folder..."]:
                            default_model = temporary.MODEL_NAME
                        elif len(available_models) > 2:  # More than just base choices
                            default_model = available_models[2]  # First actual model
                        else:
                            default_model = "Select_a_model..."
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
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("Critical Actions...")
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            save_settings=gr.Button("Save Settings", variant="primary")
                        )
                        custom_components.update(
                            delete_all_history=gr.Button("Delete All History", variant="stop")  # Renamed from "Delete All History/Vectors"
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        with gr.Column(scale=1, elem_classes=["clean-elements"]):
                            gr.Markdown("About Program...")
                            gr.Markdown("[Chat-Gradio-Gguf](https://github.com/wiseman-timelord/Chat-Gradio-Gguf) by [Wiseman-Timelord](https://github.com/wiseman-timelord).")
                            gr.Markdown("Donations through, [Patreon](https://patreon.com/WisemanTimelord) or [Ko-fi](https://ko-fi.com/WisemanTimelord).")
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            status_settings=gr.Textbox(
                                label="Status",
                                interactive=False,
                                value="Select model on Configuration page.",
                                scale=20
                            ),
                            shutdown=gr.Button("Exit", variant="stop", elem_classes=["double-height"], min_width=110).click(
                                fn=shutdown_program,
                                inputs=[states["llm"], states["models_loaded"]]
                            )
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
            outputs=[status_text]
        )

        start_new_session_btn.click(
            fn=start_new_session,
            inputs=[states["models_loaded"]],
            outputs=[conversation_components["session_log"], status_text, conversation_components["user_input"], switches["web_search"], switches["speak"]]  # Removed "tot"
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        ).then(
            fn=lambda: [],
            inputs=[],
            outputs=[states["attached_files"]]  # Removed "vector_files"
        )

        new_session_btn_collapsed.click(
            fn=start_new_session,
            inputs=[states["models_loaded"]],
            outputs=[conversation_components["session_log"], status_text, conversation_components["user_input"], switches["web_search"], switches["speak"]]  # Removed "tot"
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        ).then(
            fn=lambda: [],
            inputs=[],
            outputs=[states["attached_files"]]  # Removed "vector_files"
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
                switches["web_search"],
                states["interaction_phase"],
                switches["speak"],
                states["llm"],
                states["models_loaded"]
            ],
            outputs=[
                conversation_components["session_log"],
                status_text,
                action_buttons["action"],
                states["cancel_flag"],
                states["attached_files"],
                states["interaction_phase"],
                conversation_components["user_input"],
                switches["web_search"],
                switches["speak"]  # Removed "tot"
            ]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        action_buttons["copy_response"].click(
            fn=copy_last_response,
            inputs=[conversation_components["session_log"]],
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

        for i, btn in enumerate(attach_slots):
            btn.click(
                fn=lambda files, idx=i: eject_file(files, idx, True),
                inputs=[states["attached_files"]],
                outputs=[states["attached_files"], status_text] + attach_slots + [attach_files]
            )

        for i, btn in enumerate(buttons["session"]):
            btn.click(
                fn=load_session_by_index,
                inputs=[gr.State(value=i)],
                outputs=[conversation_components["session_log"], states["attached_files"], status_text]  # Removed "vector_files"
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
            outputs=[model_folder_state, config_components["model"], status_text]
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
        )

        states["selected_panel"].change(
            fn=lambda panel: (
                gr.update(visible=panel == "Attach"),
                gr.update(visible=panel == "History")
            ),
            inputs=[states["selected_panel"]],
            outputs=[attach_group, history_slots_group]  # Removed "vector_group"
        )

        for comp in [config_components[k] for k in ["ctx", "batch", "temp", "repeat", "vram", "gpu", "cpu", "model"]]:
            comp.change(
                fn=update_config_settings,
                inputs=[config_components[k] for k in ["ctx", "batch", "temp", "repeat", "vram", "gpu", "cpu", "model"]],
                outputs=[status_text]
            )

        config_components["browse"].click(
            fn=browse_for_model_folder,
            inputs=[model_folder_state],
            outputs=[model_folder_state, config_components["model"]]
        ).then(
            fn=lambda f: f"Model directory updated to: {f}",
            inputs=[model_folder_state],
            outputs=[status_text]
        )

        config_components["model"].change(
            fn=lambda model: (setattr(temporary, "MODEL_NAME", model), f"Selected model: {model}")[1],
            inputs=[config_components["model"]],
            outputs=[status_text]
        )

        config_components["unload"].click(
            fn=unload_models,
            inputs=[states["llm"], states["models_loaded"]],
            outputs=[status_text, states["llm"], states["models_loaded"]]
        ).then(
            fn=lambda: gr.update(interactive=False),
            outputs=[conversation_components["user_input"]]
        )

        config_components["load_models"].click(
            fn=set_loading_status,
            outputs=[status_text]
        ).then(
            fn=load_models,
            inputs=[model_folder_state, config_components["model"], config_components["vram"], states["llm"], states["models_loaded"]],
            outputs=[status_text, states["models_loaded"], states["llm"], states["models_loaded"]]
        ).then(
            fn=lambda status, ml: (status, gr.update(interactive=ml)),
            inputs=[status_text, states["models_loaded"]],
            outputs=[status_text, conversation_components["user_input"]]
        )

        config_components["save_settings"].click(
            fn=save_all_settings,
            outputs=[status_text]
        )

        custom_components["delete_all_history"].click(
            fn=utility.delete_all_session_histories,  # Updated to match simplified utility function
            outputs=[status_text]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        custom_components["session_log_height"].change(
            fn=update_session_log_height,
            inputs=[custom_components["session_log_height"]],
            outputs=[conversation_components["session_log"]]
        )

        custom_components["input_lines"].change(
            fn=update_input_lines,
            inputs=[custom_components["input_lines"]],
            outputs=[conversation_components["user_input"]]
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
        )

        status_text.change(
            fn=lambda status: status,
            inputs=[status_text],
            outputs=[config_components["status_settings"]]
        )

    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, show_api=False)
    
if __name__ == "__main__":
    launch_interface()