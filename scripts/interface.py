# Script: `.\scripts\interface.py`

# Imports...
import gradio as gr
from gradio import themes
import re, os, json, pyperclip, yake, random, asyncio
from pathlib import Path
from datetime import datetime  # Added for timestamp formatting
import tkinter as tk
from tkinter import filedialog
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from queue import Queue
import scripts.temporary as temporary
from scripts.temporary import (
    USER_COLOR, THINK_COLOR, RESPONSE_COLOR, SEPARATOR, MID_SEPARATOR,
    ALLOWED_EXTENSIONS, N_CTX, VRAM_SIZE, SELECTED_GPU, SELECTED_CPU,
    current_model_settings, N_GPU_LAYERS, VRAM_OPTIONS, REPEAT_PENALTY,
    MLOCK, HISTORY_DIR, BATCH_OPTIONS, N_BATCH, MODEL_FOLDER,
    MODEL_NAME, STATUS_TEXTS, CTX_OPTIONS, RP_LOCATION, USER_PC_NAME, USER_PC_ROLE,
    AI_NPC_NAME, AI_NPC_ROLE, SESSION_ACTIVE, TOT_VARIATIONS,
    MAX_HISTORY_SLOTS, MAX_ATTACH_SLOTS, HISTORY_SLOT_OPTIONS, ATTACH_SLOT_OPTIONS,
    BACKEND_TYPE, STREAM_OUTPUT
)
from scripts import utility
from scripts.utility import (
    delete_all_session_vectorstores, create_session_vectorstore, web_search, get_saved_sessions,
    load_session_history, save_session_history, load_and_chunk_documents,
    get_available_gpus, save_config
)
from scripts.models import (
    get_response_stream, get_available_models, unload_models, get_model_settings,
    context_injector, inspect_model, load_models
)
from langchain_core.documents import Document

# Functions...
def set_loading_status():
    return "Loading model..."

def format_response(output: str) -> str:
    formatted = []
    code_blocks = re.findall(r'```(\w+)?\n(.*?)```', output, re.DOTALL)
    for lang, code in code_blocks:
        lexer = get_lexer_by_name(lang, stripall=True)
        formatted_code = highlight(code, lexer, HtmlFormatter())
        output = output.replace(f'```{lang}\n{code}```', formatted_code)
    return f'<span style="color: {RESPONSE_COLOR}">{output}</span>'

def select_directory():
    print("Opening directory selection dialog...")
    root = tk.Tk()
    root.withdraw()
    initial_dir = r"C:\Program_Filez\Text-Gradio-Gguf\Text-Gradio-Gguf-main\models"
    if not os.path.exists(initial_dir):
        initial_dir = os.path.expanduser("~")
    path = filedialog.askdirectory(initialdir=initial_dir)
    root.destroy()
    if path:
        print(f"Selected path: {path}")
        return path
    else:
        print("No directory selected")
        return None

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

def eject_file(loaded_files, slot_index):
    from scripts.utility import create_session_vectorstore
    from pathlib import Path
    if 0 <= slot_index < len(loaded_files):
        removed_file = loaded_files.pop(slot_index)
        try:
            Path(removed_file).unlink()
        except Exception as e:
            print(f"Error unlinking file: {e}")
        session_vectorstore = create_session_vectorstore(loaded_files)
        context_injector.set_session_vectorstore(session_vectorstore)
        status_msg = f"Ejected {Path(removed_file).name}"
    else:
        status_msg = "No file to eject"
    updates = update_file_slot_ui(loaded_files)
    return [loaded_files, status_msg] + updates

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

def start_new_session():
    from scripts import temporary
    temporary.current_session_id = None
    temporary.session_label = ""
    temporary.SESSION_ACTIVE = True
    return [], "Type input and click Send to begin...", gr.update(interactive=True)

def load_session_by_index(index):
    sessions = utility.get_saved_sessions()
    if index < len(sessions):
        session_file = sessions[index]
        session_id, label, history, attached_files = utility.load_session_history(Path(HISTORY_DIR) / session_file)
        temporary.current_session_id = session_id
        temporary.session_label = label
        temporary.SESSION_ACTIVE = True
        return history, f"Loaded session: {label}"
    return [], "No session to load"

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

def update_mode_based_options(mode, showing_rpg_right):
    mode = mode.lower()
    is_rpg = mode == "rpg"
    if not is_rpg:
        showing_rpg_right = False
    if mode == "code":
        tot_visible = False
        web_visible = True
        file_visible = True
    elif is_rpg:
        tot_visible = False
        web_visible = False
        file_visible = True
    else:
        tot_visible = True
        web_visible = True
        file_visible = True
    toggle_visible = is_rpg
    toggle_label = "Show File Attachments" if showing_rpg_right else "Show RPG Settings"
    file_attachments_visible = not showing_rpg_right if is_rpg else True
    rpg_settings_visible = showing_rpg_right if is_rpg else False
    return [
        gr.update(visible=tot_visible),
        gr.update(visible=web_visible),
        gr.update(visible=file_visible),
        gr.update(visible=toggle_visible, value=toggle_label),
        gr.update(visible=file_attachments_visible),
        gr.update(visible=rpg_settings_visible),
        showing_rpg_right
    ]

def update_model_based_options(model_name):
    if model_name in ["Select_a_model...", "No models found"]:
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

def update_file_slot_ui(loaded_files):
    from pathlib import Path
    import scripts.temporary as temporary
    button_updates = []
    for i in range(temporary.MAX_POSSIBLE_ATTACH_SLOTS):
        if i < temporary.MAX_ATTACH_SLOTS:
            if i < len(loaded_files):
                filename = Path(loaded_files[i]).name
                short_name = (filename[:36] + ".." if len(filename) > 38 else filename)
                label = f"{short_name}"
                variant = "primary"  # Changed from "secondary" to "primary"
            else:
                label = "File Slot Free"
                variant = "huggingface"  # Unchanged
            visible = True
        else:
            label = ""
            variant = "primary"  # This is for hidden slots, can remain as is
            visible = False
        button_updates.append(gr.update(value=label, visible=visible, variant=variant))
    attach_files_visible = len(loaded_files) < temporary.MAX_ATTACH_SLOTS
    return button_updates + [gr.update(visible=attach_files_visible)]


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
    sessions = utility.get_saved_sessions()
    button_updates = []
    for i in range(temporary.MAX_POSSIBLE_HISTORY_SLOTS):
        if i < temporary.MAX_HISTORY_SLOTS:
            if i < len(sessions):
                session_path = Path(HISTORY_DIR) / sessions[i]
                try:
                    stat = session_path.stat()
                    update_time = stat.st_mtime if stat.st_mtime else stat.st_ctime
                    formatted_time = datetime.fromtimestamp(update_time).strftime("%Y-%m-%d %H:%M")
                    session_id, label, history, _ = utility.load_session_history(session_path)
                    if history:
                        text = " ".join([msg['content'] for msg in history])
                        text = filter_operational_content(text)
                        kw_extractor = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.9, top=1)
                        keywords = kw_extractor.extract_keywords(text)
                        description = keywords[0][0][:20] if keywords else "No description"
                    else:
                        description = "No history"
                    temporary.yake_history_detail[i] = description
                    btn_label = f"{formatted_time} - {description}"
                except Exception as e:
                    print(f"Error loading session {session_path}: {e}")
                    btn_label = f"Session {i+1}"
                    temporary.yake_history_detail[i] = None
            else:
                btn_label = "History Slot Empty"
                temporary.yake_history_detail[i] = None
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
                        interaction_phase):
    from scripts import temporary, utility, models
    from scripts.temporary import STATUS_TEXTS, MODEL_NAME, SESSION_ACTIVE, TOT_VARIATIONS
    import asyncio
    
    # Check if models are loaded
    if not models_loaded:
        yield session_log, "Please load a model first.", gr.update(), cancel_flag, loaded_files, interaction_phase, gr.update()
        return
    
    # Start a new session if not active
    if not temporary.SESSION_ACTIVE:
        temporary.current_session_id = None
        temporary.session_label = ""
        temporary.SESSION_ACTIVE = True
        session_log = []
        yield session_log, "New session started.", gr.update(), cancel_flag, loaded_files, interaction_phase, gr.update()
        await asyncio.sleep(0.1)
    
    # Validate input
    if not user_input.strip():
        yield session_log, "No input provided.", gr.update(), cancel_flag, loaded_files, interaction_phase, gr.update()
        return

    # Append user input and clear the textbox
    session_log.append({'role': 'user', 'content': f"User:\n{user_input}"})
    if len(session_log) == 1 and session_log[0]['role'] == 'user':
        temporary.session_label = create_session_label(user_input)
    session_log.append({'role': 'assistant', 'content': "Afterthought countdown... "})
    interaction_phase = "afterthought_countdown"
    yield session_log, "Processing...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(value="")  # Clear user input here

    # Afterthought countdown
    countdown_seconds = 6 if len(user_input.split('\n')) >= 10 else 4 if len(user_input.split('\n')) >= 5 else 2 if temporary.AFTERTHOUGHT_TIME else 1
    for i in range(countdown_seconds, -1, -1):
        session_log[-1]['content'] = f"Afterthought countdown... {i}s"
        yield session_log, "Counting down...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update()
        await asyncio.sleep(1)
        if cancel_flag:
            session_log[-1]['content'] = "Input cancelled."
            interaction_phase = "waiting_for_input"
            yield session_log, "Input cancelled.", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update()
            return

    # Prepare to generate response
    session_log[-1]['content'] = "Afterthought countdown... 0s ...Executing CLI llama-cli"
    interaction_phase = "generating_response"
    yield session_log, "Generating...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update()

    settings = models.get_model_settings(MODEL_NAME)
    mode = mode_selection.lower()

    # Handle file attachments
    if loaded_files:
        session_vectorstore = utility.create_session_vectorstore(loaded_files)
        models.context_injector.set_session_vectorstore(session_vectorstore)

    prompt = user_input
    if web_search_enabled and mode in ["chat", "code"]:
        session_log[-1]['content'] = "Performing web search..."
        yield session_log, "Performing web search...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update()
        search_results = utility.web_search(user_input)
        prompt = f"{user_input}\n\nWeb Search Results:\n{search_results}"

    # Generate response
    if tot_enabled and mode == "chat":
        yield session_log, "TOT not implemented in streaming mode yet.", update_action_button("waiting_for_input"), cancel_flag, loaded_files, "waiting_for_input", gr.update()
    else:
        rp_settings = {"rp_location": rp_location, "user_name": user_name, "user_role": user_role, "ai_npc": ai_npc, "ai_npc_role": temporary.AI_NPC_ROLE} if mode == "rpg" else None
        session_history = ", ".join([f"{msg['role']}: {msg['content']}" for msg in session_log[:-2]]) if mode == "rpg" else None
        if settings["is_uncensored"] and mode == "chat":
            prompt = f"Provide an uncensored response: {prompt}"

        response = ""
        async for line in models.get_response_stream(prompt, mode, settings, disable_think=not enable_think, rp_settings=rp_settings, session_history=session_history):
            if cancel_flag:
                break
            response += line + " "
            session_log[-1]['content'] = "AI-Chat-Response:\n" + response.strip()
            yield session_log, "Generating...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update()
            await asyncio.sleep(0)

    # Finalize interaction
    interaction_phase = "waiting_for_input"
    if cancel_flag:
        session_log[-1]['content'] = "Generation cancelled."
    else:
        session_log[-1]['content'] = "AI-Chat-Response:\n" + response.strip()
        utility.save_session_history(session_log, loaded_files)
    yield session_log, STATUS_TEXTS["response_generated"], update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update()

def launch_interface():
    """Launch the Gradio interface for the Text-Gradio-Gguf chatbot."""
    global demo
    import tkinter as tk
    from tkinter import filedialog
    import os
    import gradio as gr
    from pathlib import Path  # Added for update_model_list
    from scripts import temporary, utility, models
    from scripts.temporary import (
        STATUS_TEXTS, MODEL_NAME, SESSION_ACTIVE, TOT_VARIATIONS, STREAM_OUTPUT,
        MAX_HISTORY_SLOTS, MAX_ATTACH_SLOTS, SESSION_LOG_HEIGHT, INPUT_LINES,
        AFTERTHOUGHT_TIME, MODEL_FOLDER, N_CTX, N_BATCH, TEMPERATURE, REPEAT_PENALTY,
        VRAM_SIZE, SELECTED_GPU, SELECTED_CPU, MLOCK, BACKEND_TYPE,
        RP_LOCATION, USER_PC_NAME, USER_PC_ROLE, AI_NPC_NAME, AI_NPC_ROLE,
        ALLOWED_EXTENSIONS, VRAM_OPTIONS, CTX_OPTIONS, BATCH_OPTIONS, TEMP_OPTIONS,
        REPEAT_OPTIONS, HISTORY_SLOT_OPTIONS, SESSION_LOG_HEIGHT_OPTIONS,
        INPUT_LINES_OPTIONS, ATTACH_SLOT_OPTIONS
    )


    available_models = models.get_available_models()
    default_model = temporary.MODEL_NAME if temporary.MODEL_NAME in available_models else available_models[0]  # First choice is always valid

    with gr.Blocks(title="Chat-Gradio-Gguf", css=".scrollable{overflow-y:auto}.send-button{background-color:green!important;color:white!important;height:80px!important}.double-height{height:80px!important}.clean-elements{gap:4px!important;margin-bottom:4px!important}.clean-elements-normbot{gap:4px!important;margin-bottom:20px!important}") as demo:
        model_folder_state = gr.State(temporary.MODEL_FOLDER)

        states = dict(
            loaded_files=gr.State([]),
            models_loaded=gr.State(False),
            showing_rpg_right=gr.State(False),
            cancel_flag=gr.State(False),
            interaction_phase=gr.State("waiting_for_input"),
            is_reasoning_model=gr.State(False)
        )

        with gr.Tabs():
            with gr.Tab("Conversation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Column(visible=True, elem_classes=["clean-elements"]):
                            buttons = dict(
                                start_new_session=gr.Button("Start New Session...", variant="secondary"),
                                session=[gr.Button(f"History Slot {i+1}", variant="huggingface", visible=False) for i in range(temporary.MAX_POSSIBLE_HISTORY_SLOTS)]
                            )

                    with gr.Column(scale=30, elem_classes=["clean-elements"]):
                        chat_components = dict(
                            session_log=gr.Chatbot(label="Session Log", height=temporary.SESSION_LOG_HEIGHT, elem_classes=["scrollable"], type="messages"),
                            user_input=gr.Textbox(label="User Input", lines=temporary.INPUT_LINES, interactive=False, placeholder="Enter text here...")
                        )
                        with gr.Row(elem_classes=["clean-elements"]):
                            action_buttons = {}
                            with gr.Column(elem_classes=["clean-elements"]):
                                action_buttons.update(
                                    action=gr.Button("Send Input", variant="secondary", scale=30, elem_classes=["send-button"])
                                )
                            with gr.Column(elem_classes=["clean-elements"]):
                                action_buttons.update(
                                    edit_previous=gr.Button("Edit Last Input", variant="huggingface"),
                                    copy_response=gr.Button("Copy Last Output", variant="huggingface")
                                )
                            with gr.Column(elem_classes=["clean-elements"]):
                                switches = dict(
                                    web_search=gr.Checkbox(label="Web-Search", value=False, visible=True),
                                    tot=gr.Checkbox(label="Enable TOT", value=False, visible=True),
                                    enable_think=gr.Checkbox(label="Enable THINK", value=True, visible=False)
                                )

                    with gr.Column(scale=1):
                        right_panel = dict(
                            mode_selection=gr.Radio(choices=["Chat", "Code", "Rpg"], label="Select Operation Mode", value="Chat"),
                            toggle_rpg_settings=gr.Button("Show RPG Settings", variant="secondary", elem_classes=["clean-elements"], visible=False),
                            file_attachments=gr.Group(visible=True, elem_classes=["clean-elements"]),
                            rpg_settings=gr.Group(visible=False, elem_classes=["clean-elements"])
                        )
                        with right_panel["file_attachments"]:
                            right_panel.update(
                                attach_files=gr.UploadButton("Attach New Files", file_types=[f".{ext}" for ext in temporary.ALLOWED_EXTENSIONS], file_count="multiple", variant="secondary", elem_classes=["clean-elements"]),
                                file_slots=[gr.Button("File Slot Free", variant="huggingface", visible=False) for _ in range(temporary.MAX_POSSIBLE_ATTACH_SLOTS)],
                            )
                        with right_panel["rpg_settings"]:
                            rpg_fields = dict(
                                rp_location=gr.Textbox(label="RP Location", value=temporary.RP_LOCATION),
                                user_name=gr.Textbox(label="User Name", value=temporary.USER_PC_NAME),
                                user_role=gr.Textbox(label="User Role", value=temporary.USER_PC_ROLE),
                                ai_npc=gr.Textbox(label="AI NPC", value=temporary.AI_NPC_NAME),
                                ai_npc_role=gr.Textbox(label="AI Role", value=temporary.AI_NPC_ROLE),
                                save_rpg=gr.Button("Save RPG Settings", variant="primary")
                            )
                with gr.Row():
                    status_text = gr.Textbox(label="Status", interactive=False, value="Select model on Configuration page.", scale=30)
                    gr.Button("Exit Program", variant="stop", elem_classes=["double-height"]).click(fn=shutdown_program, inputs=[states["models_loaded"]])

            with gr.Tab("Configuration"):
                with gr.Column(scale=1, elem_classes=["clean-elements"]):
                    is_cpu_only = temporary.BACKEND_TYPE in ["CPU Only - AVX2", "CPU Only - AVX512", "CPU Only - NoAVX", "CPU Only - OpenBLAS"]
                    config_components = {}
                    
                    # CPU/GPU Options
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("CPU/GPU Options...")
                    with gr.Row(visible=not is_cpu_only, elem_classes=["clean-elements"]):
                        config_components.update(
                            gpu=gr.Dropdown(choices=utility.get_available_gpus(), label="Select GPU", value=temporary.SELECTED_GPU, scale=5),
                            backend_type=gr.Textbox(label="Installed Backend", value=temporary.BACKEND_TYPE, interactive=False, scale=5),
                            vram=gr.Dropdown(choices=temporary.VRAM_OPTIONS, label="Assign Free VRam", value=temporary.VRAM_SIZE, scale=3),
                        )
                    with gr.Row(visible=is_cpu_only, elem_classes=["clean-elements"]):
                        cpu_choices = [cpu["label"] for cpu in utility.get_cpu_info()] or ["Default CPU"]
                        config_components.update(
                            backend_type=gr.Textbox(label="Backend Type", value=temporary.BACKEND_TYPE, interactive=False, scale=5),
                            cpu=gr.Dropdown(choices=cpu_choices, label="Select CPU", value=temporary.SELECTED_CPU if temporary.SELECTED_CPU in cpu_choices else cpu_choices[0], scale=5),
                        )
                    
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("Model Options...")
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            model_dir=gr.Textbox(label="Model Folder", value=temporary.MODEL_FOLDER, interactive=False, scale=10),
                            model=gr.Dropdown(
                                choices=available_models,
                                label="Select Model",
                                value=default_model,  # Will be "Select_a_model..." or "No models found"
                                allow_custom_value=False,
                                scale=10
                            )
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            ctx=gr.Dropdown(choices=temporary.CTX_OPTIONS, label="Context Size", value=temporary.N_CTX),
                            batch=gr.Dropdown(choices=temporary.BATCH_OPTIONS, label="Batch Size", value=temporary.N_BATCH),
                            temp=gr.Dropdown(choices=temporary.TEMP_OPTIONS, label="Temperature", value=temporary.TEMPERATURE),
                            repeat=gr.Dropdown(choices=temporary.REPEAT_OPTIONS, label="Repeat Penalty", value=temporary.REPEAT_PENALTY),
                        )
                        with gr.Column(elem_classes=["clean-elements"]):
                            config_components.update(
                                stream_output=gr.Checkbox(label="Stream Output", value=temporary.STREAM_OUTPUT),
                                # Removed: mlock_cpu and use_python_bindings checkboxes
                            )
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            browse=gr.Button("Browse", variant="secondary", elem_classes=["double-height"]), 
                            load_models=gr.Button("Load Model", variant="secondary", elem_classes=["double-height"]),
                            inspect_model=gr.Button("Inspect Model", variant="huggingface", elem_classes=["double-height"]),
                            unload=gr.Button("Unload Model", elem_classes=["double-height"], variant="huggingface"),
                        )
                    
                    # Interface Options
                    custom_components = {}
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("Interface Options...")
                    with gr.Row(elem_classes=["clean-elements"]):
                        custom_components.update(
                            max_history_slots=gr.Dropdown(choices=temporary.HISTORY_SLOT_OPTIONS, label="Max History Slots", value=temporary.MAX_HISTORY_SLOTS),
                            session_log_height=gr.Dropdown(choices=temporary.SESSION_LOG_HEIGHT_OPTIONS, label="Session Log Height", value=temporary.SESSION_LOG_HEIGHT),
                            input_lines=gr.Dropdown(choices=temporary.INPUT_LINES_OPTIONS, label="Input Lines", value=temporary.INPUT_LINES),
                            max_attach_slots=gr.Dropdown(choices=temporary.ATTACH_SLOT_OPTIONS, label="Max Attach Slots", value=temporary.MAX_ATTACH_SLOTS),
                        )
                        with gr.Column(elem_classes=["clean-elements"]):
                            custom_components.update(
                                afterthought_time=gr.Checkbox(label="After-Thought Time", value=temporary.AFTERTHOUGHT_TIME)
                            )
                    
                    # Critical Actions
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("Critical Actions...")
                    with gr.Row(elem_classes=["clean-elements-normbot"]):
                        custom_components.update(
                            delete_all_vectorstores=gr.Button("Delete All History/Vectors", variant="stop", elem_classes=["double-height"])
                        )
                        config_components.update(
                            save_settings=gr.Button("Save Settings", variant="primary", elem_classes=["double-height"])
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            status_settings=gr.Textbox(label="Status", interactive=False, scale=20),
                            shutdown=gr.Button("Exit Program", variant="stop", elem_classes=["double-height"]).click(fn=shutdown_program, inputs=[states["models_loaded"]])
                        )

        # Helper Functions
        def select_model_folder():
            print("Opening directory selection dialog...")
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            initial_dir = temporary.MODEL_FOLDER if os.path.exists(temporary.MODEL_FOLDER) else os.path.expanduser("~")
            folder = filedialog.askdirectory(initialdir=initial_dir, title="Select Model Folder")
            root.destroy()
            if folder:
                print(f"Selected folder: {folder}")
                temporary.MODEL_FOLDER = folder
                return folder, folder  # Return for model_dir and model_folder_state
            else:
                print("No folder selected")
                return temporary.MODEL_FOLDER, temporary.MODEL_FOLDER  # Return current folder if none selected

        def update_model_list(folder):
            model_dir = Path(folder)
            models = [f.name for f in model_dir.glob("*.gguf") if f.is_file()]
            if models:
                choices = ["Select_a_model..."] + models
                value = "Select_a_model..."  # Reset to default when folder changes
            else:
                choices = ["No models found"]
                value = "No models found"  # Set to indicate no models
            print(f"Updating model list for {folder}: {choices}")
            return gr.update(choices=choices, value=value)

        def update_config_settings(*args):
            keys = ["ctx", "batch", "temp", "repeat", "vram", "gpu", "cpu", "model"]
            attr_map = {
                "ctx": "N_CTX", "batch": "N_BATCH", "temp": "TEMPERATURE", "repeat": "REPEAT_PENALTY",
                "vram": "VRAM_SIZE", "gpu": "SELECTED_GPU", "cpu": "SELECTED_CPU", "model": "MODEL_NAME"
            }
            for k, v in zip(keys, args):
                if k in attr_map:
                    setattr(temporary, attr_map[k], int(v) if k in ["ctx", "batch", "vram"] else float(v) if k in ["temp", "repeat"] else v)
            return "Settings updated in memory. Click 'Save Settings' to commit."

        def update_mlock(mlock):
            temporary.MLOCK = mlock
            return "MLock updated."

        def update_stream_output(value):
            temporary.STREAM_OUTPUT = value
            return f"Stream Output {'enabled' if value else 'disabled'}."

        def save_all_settings():
            utility.save_config()
            return "Settings saved to persistent.json"

        async def action_handler(phase, user_input, session_log, tot_enabled, loaded_files,
                                 enable_think, is_reasoning_model, rp_location, user_name, user_role, ai_npc,
                                 cancel_flag, mode_selection, web_search_enabled, models_loaded, interaction_phase):
            if phase == "waiting_for_input":
                # Pass through all 7 values from chat_interface
                async for output in chat_interface(user_input, session_log, tot_enabled, loaded_files, enable_think,
                                                   is_reasoning_model, rp_location, user_name, user_role, ai_npc,
                                                   cancel_flag, mode_selection, web_search_enabled, models_loaded,
                                                   interaction_phase):
                    yield output
            else:
                # Handle cancellation or other phases with 7 outputs
                if phase == "afterthought_countdown":
                    cancel_flag = True
                    new_phase = "waiting_for_input"
                    status_msg = "Input cancelled."
                elif phase == "generating_response":
                    cancel_flag = True
                    new_phase = "waiting_for_input"
                    status_msg = "Response generation cancelled."
                else:
                    new_phase = interaction_phase
                    status_msg = "No action taken."
                yield session_log, status_msg, update_action_button(new_phase), cancel_flag, loaded_files, new_phase, gr.update()

        # Event Handlers
        buttons["start_new_session"].click(
            fn=start_new_session,
            outputs=[chat_components["session_log"], status_text, chat_components["user_input"]]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        action_buttons["action"].click(
            fn=action_handler,
            inputs=[
                states["interaction_phase"],
                chat_components["user_input"],
                chat_components["session_log"],
                switches["tot"],
                states["loaded_files"],
                switches["enable_think"],
                states["is_reasoning_model"],
                rpg_fields["rp_location"],
                rpg_fields["user_name"],
                rpg_fields["user_role"],
                rpg_fields["ai_npc"],
                states["cancel_flag"],
                right_panel["mode_selection"],
                switches["web_search"],
                states["models_loaded"],
                states["interaction_phase"]
            ],
            outputs=[
                chat_components["session_log"],
                status_text,
                action_buttons["action"],
                states["cancel_flag"],
                states["loaded_files"],
                states["interaction_phase"],
                chat_components["user_input"]
            ]
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

        right_panel["attach_files"].upload(
            fn=process_uploaded_files,
            inputs=[right_panel["attach_files"], states["loaded_files"], states["models_loaded"]],
            outputs=[status_text, states["loaded_files"]]
        ).then(
            fn=update_file_slot_ui,
            inputs=[states["loaded_files"]],
            outputs=right_panel["file_slots"] + [right_panel["attach_files"]]
        )

        for i, btn in enumerate(right_panel["file_slots"]):
            btn.click(
                fn=eject_file,
                inputs=[states["loaded_files"], gr.State(value=i)],
                outputs=[states["loaded_files"], status_text] + right_panel["file_slots"] + [right_panel["attach_files"]]
            )

        for i, btn in enumerate(buttons["session"]):
            btn.click(
                fn=load_session_by_index,
                inputs=[gr.State(value=i)],
                outputs=[chat_components["session_log"], status_text]
            ).then(
                fn=update_session_buttons,
                inputs=[],
                outputs=buttons["session"]
            )

        for comp in [config_components[k] for k in ["ctx", "batch", "temp", "repeat", "vram", "gpu", "cpu", "model"]]:
            comp.change(
                fn=update_config_settings,
                inputs=[config_components[k] for k in ["ctx", "batch", "temp", "repeat", "vram", "gpu", "cpu", "model"]] + [config_components["model_dir"]],
                outputs=[config_components["status_settings"]]
            )

        config_components["browse"].click(
            fn=select_model_folder,
            outputs=[config_components["model_dir"], model_folder_state]
        ).then(
            fn=lambda f: f"Model directory updated to: {f}",
            inputs=[model_folder_state],
            outputs=[config_components["status_settings"]]
        )

        model_folder_state.change(
            fn=update_model_list,
            inputs=[model_folder_state],
            outputs=[config_components["model"]]
        )

        config_components["stream_output"].change(
            fn=update_stream_output,
            inputs=[config_components["stream_output"]],
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

        config_components["inspect_model"].click(
            fn=inspect_model,
            inputs=[config_components["model_dir"], config_components["model"], config_components["vram"]],
            outputs=[config_components["status_settings"]]
        )

        config_components["load_models"].click(
            fn=set_loading_status,
            outputs=[config_components["status_settings"]]
        ).then(
            fn=load_models,
            inputs=[config_components["model_dir"], config_components["model"], config_components["vram"]],
            outputs=[config_components["status_settings"], states["models_loaded"]]
        ).then(
            fn=lambda status, ml: (status, gr.update(interactive=ml)),
            inputs=[config_components["status_settings"], states["models_loaded"]],
            outputs=[config_components["status_settings"], chat_components["user_input"]]
        )

        config_components["save_settings"].click(
            fn=save_all_settings,
            outputs=[config_components["status_settings"]]
        )

        right_panel["toggle_rpg_settings"].click(
            fn=toggle_rpg_settings,
            inputs=[states["showing_rpg_right"]],
            outputs=[right_panel["toggle_rpg_settings"], right_panel["file_attachments"], right_panel["rpg_settings"], states["showing_rpg_right"]]
        )

        rpg_fields["save_rpg"].click(
            fn=save_rp_settings,
            inputs=list(rpg_fields.values())[:-1],
            outputs=list(rpg_fields.values())[:-1]
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
            fn=lambda h: gr.update(height=h),
            inputs=[custom_components["session_log_height"]],
            outputs=[chat_components["session_log"]]
        )

        custom_components["input_lines"].change(
            fn=lambda l: gr.update(lines=l),
            inputs=[custom_components["input_lines"]],
            outputs=[chat_components["user_input"]]
        )

        custom_components["max_history_slots"].change(
            fn=lambda s: setattr(temporary, "MAX_HISTORY_SLOTS", s),
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
            fn=update_file_slot_ui,
            inputs=[states["loaded_files"]],
            outputs=right_panel["file_slots"] + [right_panel["attach_files"]]
        )

        custom_components["afterthought_time"].change(
            fn=lambda v: setattr(temporary, "AFTERTHOUGHT_TIME", v),
            inputs=[custom_components["afterthought_time"]],
            outputs=[config_components["status_settings"]]
        )

        # Corrected event handler: Fixed typo and removed duplicate
        right_panel["mode_selection"].change(
            fn=update_mode_based_options,
            inputs=[right_panel["mode_selection"], states["showing_rpg_right"]],
            outputs=[switches["tot"], switches["web_search"], right_panel["attach_files"], right_panel["toggle_rpg_settings"], right_panel["file_attachments"], right_panel["rpg_settings"], states["showing_rpg_right"]]
        )

        demo.load(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        ).then(
            fn=update_file_slot_ui,
            inputs=[states["loaded_files"]],
            outputs=right_panel["file_slots"] + [right_panel["attach_files"]]
        )

        demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, show_api=False)

if __name__ == "__main__":
    launch_interface()