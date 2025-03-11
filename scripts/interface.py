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
    context_injector, inspect_model, load_models, get_available_models
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

def update_model_list(new_dir):
    print(f"Updating model list with new_dir: {new_dir}")
    temporary.MODEL_FOLDER = new_dir
    choices = get_available_models()
    print(f"Choices returned: {choices}")
    return gr.update(choices=choices)

def select_directory(current_model_folder):
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
        return path, path
    else:
        print("No directory selected")
        return current_model_folder, current_model_folder

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

def update_mode_based_options(mode, showing_rpg_right, is_reasoning_model):
    mode = mode.lower()
    if mode == "code":
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
        if i < len(loaded_files):  # Only show buttons for loaded files
            filename = Path(loaded_files[i]).name
            short_name = (filename[:36] + ".." if len(filename) > 38 else filename)
            label = f"{short_name}"
            variant = "primary"
            visible = True
        else:
            label = ""
            variant = "primary"
            visible = False  # Hide slots beyond the number of loaded files
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
        if i < len(sessions):  # Only show buttons for existing sessions
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
                        interaction_phase):
    from scripts import temporary, utility, models
    from scripts.temporary import STATUS_TEXTS, MODEL_NAME, SESSION_ACTIVE, TOT_VARIATIONS
    import asyncio
    
    # Check if models are loaded
    if not models_loaded:
        yield session_log, "Please load a model first.", update_action_button("waiting_for_input"), cancel_flag, loaded_files, "waiting_for_input", gr.update(), gr.update()
        return
    
    # Start a new session if not active
    if not temporary.SESSION_ACTIVE:
        temporary.current_session_id = None
        temporary.session_label = ""
        temporary.SESSION_ACTIVE = True
        session_log = []
        yield session_log, "New session started.", update_action_button("waiting_for_input"), cancel_flag, loaded_files, "waiting_for_input", gr.update(), gr.update()
        await asyncio.sleep(0.1)
    
    # Validate input
    if not user_input.strip():
        yield session_log, "No input provided.", update_action_button("waiting_for_input"), cancel_flag, loaded_files, "waiting_for_input", gr.update(), gr.update()
        return

    # Append user input and prepare assistant response
    session_log.append({'role': 'user', 'content': f"User:\n{user_input}"})
    if len(session_log) == 1 and session_log[0]['role'] == 'user':
        temporary.session_label = create_session_label(user_input)
    session_log.append({'role': 'assistant', 'content': ""})  # Start with empty content
    interaction_phase = "afterthought_countdown"
    yield session_log, "Processing...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(value=""), gr.update()

    # Afterthought countdown
    countdown_seconds = 6 if len(user_input.split('\n')) >= 10 else 4 if len(user_input.split('\n')) >= 5 else 2 if temporary.AFTERTHOUGHT_TIME else 1
    for i in range(countdown_seconds, -1, -1):
        session_log[-1]['content'] = f"Afterthought countdown... {i}s"
        yield session_log, "Counting down...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update()
        await asyncio.sleep(1)
        if cancel_flag:
            session_log[-1]['content'] = "Input cancelled."
            interaction_phase = "waiting_for_input"
            yield session_log, "Input cancelled.", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update()
            return

    # Prepare to generate response
    session_log[-1]['content'] = "Afterthought countdown... 0s ...Executing CLI llama-cli"
    interaction_phase = "generating_response"
    yield session_log, "Generating...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update()

    settings = models.get_model_settings(MODEL_NAME)
    mode = mode_selection.lower()

    # Handle file attachments
    if loaded_files:
        session_vectorstore = utility.create_session_vectorstore(loaded_files)
        models.context_injector.set_session_vectorstore(session_vectorstore)

    # Handle web search
    if web_search_enabled and mode in ["chat", "code"]:
        session_log[-1]['content'] = "Performing web search..."
        yield session_log, "Performing web search...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update()
        search_results = utility.web_search(user_input)
        session_log[-2]['content'] += f"\n\nWeb Search Results:\n{search_results}"

    # Generate response
    if tot_enabled and mode == "chat":
        yield session_log, "TOT not implemented in streaming mode yet.", update_action_button("waiting_for_input"), cancel_flag, loaded_files, "waiting_for_input", gr.update(), gr.update()
    else:
        rp_settings = {"rp_location": rp_location, "user_name": user_name, "user_role": user_role, "ai_npc": ai_npc, "ai_npc_role": temporary.AI_NPC_ROLE} if mode == "rpg" else None
        async for line in models.get_response_stream(
            session_log,
            mode,
            settings,
            disable_think=not enable_think,
            rp_settings=rp_settings
        ):
            if cancel_flag:
                break
            # Overwrite the assistant's content with each streamed update
            session_log[-1]['content'] = line.strip()
            yield session_log, "Generating...", update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), gr.update()
            await asyncio.sleep(0)

    # Finalize interaction
    interaction_phase = "waiting_for_input"
    if cancel_flag:
        session_log[-1]['content'] = "Generation cancelled."
    else:
        utility.save_session_history(session_log, loaded_files)
    
    # Set THINK checkbox state
    think_update = gr.update(value=False) if is_reasoning_model else gr.update()
    yield session_log, STATUS_TEXTS["response_generated"], update_action_button(interaction_phase), cancel_flag, loaded_files, interaction_phase, gr.update(), think_update

def launch_interface():
    """Launch the Gradio interface for the Text-Gradio-Gguf chatbot with updated Conversation tab layout."""
    global demo
    import tkinter as tk
    from tkinter import filedialog
    import os
    import gradio as gr
    from pathlib import Path
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
    default_model = temporary.MODEL_NAME if temporary.MODEL_NAME in available_models else available_models[0]

    with gr.Blocks(title="Chat-Gradio-Gguf", css=".scrollable{overflow-y:auto}.send-button{background-color:green!important;color:white!important;height:80px!important}.half-width{width:80px!important}.double-height{height:80px!important}.clean-elements{gap:4px!important;margin-bottom:4px!important}.clean-elements-normbot{gap:4px!important;margin-bottom:20px!important}") as demo:
        model_folder_state = gr.State(temporary.MODEL_FOLDER)

        # Updated states to include selected_panel for panel toggle
        states = dict(
            loaded_files=gr.State([]),
            models_loaded=gr.State(False),
            showing_rpg_right=gr.State(False),
            cancel_flag=gr.State(False),
            interaction_phase=gr.State("waiting_for_input"),
            is_reasoning_model=gr.State(False),
            selected_panel=gr.State("History")  # Changed from "Files" to "History"
        )

        with gr.Tabs():
            with gr.Tab("Conversation"):
                with gr.Row():
                    with gr.Column(min_width=325, elem_classes=["clean-elements"]):
                        mode_selection = gr.Radio(
                            choices=["Chat", "Code", "Rpg"],
                            label="Operation Mode",
                            value="Chat"
                        )
                        panel_toggle = gr.Radio(
                            choices=["History", "Files"],
                            label="Panel Mode",
                            value="History"
                        )
                        # Attachments group for file uploads (moved up)
                        with gr.Group(visible=False) as attachments_group:
                            attach_files = gr.UploadButton(
                                "Attach New Files",
                                file_types=[f".{ext}" for ext in temporary.ALLOWED_EXTENSIONS],
                                file_count="multiple",
                                variant="secondary",
                                elem_classes=["clean-elements"]
                            )
                            file_slots = [gr.Button(
                                "File Slot Free",
                                variant="huggingface",
                                visible=False
                            ) for _ in range(temporary.MAX_POSSIBLE_ATTACH_SLOTS)]
                        # History slots group for session history
                        with gr.Group(visible=True) as history_slots_group:
                            start_new_session_btn = gr.Button("Start New Session...", variant="secondary")
                            buttons = dict(
                                session=[gr.Button(
                                    f"History Slot {i+1}",
                                    variant="huggingface",
                                    visible=False
                                ) for i in range(temporary.MAX_POSSIBLE_HISTORY_SLOTS)]
                            )
                        # RPG config group for roleplay settings
                        with gr.Group(visible=False) as rpg_config_group:
                            rpg_fields = dict(
                                rp_location=gr.Textbox(label="RP Location", value=temporary.RP_LOCATION),
                                user_name=gr.Textbox(label="User Name", value=temporary.USER_PC_NAME),
                                user_role=gr.Textbox(label="User Role", value=temporary.USER_PC_ROLE),
                                ai_npc=gr.Textbox(label="AI NPC", value=temporary.AI_NPC_NAME),
                                ai_npc_role=gr.Textbox(label="AI Role", value=temporary.AI_NPC_ROLE),
                                save_rpg=gr.Button("Save RPG Settings", variant="primary")
                            )


                    with gr.Column(scale=30, elem_classes=["clean-elements"]):
                        chat_components = {}
                        with gr.Row(elem_classes=["clean-elements"]):
                            chat_components["session_log"] = gr.Chatbot(
                                label="Session Log",
                                height=temporary.SESSION_LOG_HEIGHT,
                                elem_classes=["scrollable"],
                                type="messages"
                            )
                        with gr.Row(elem_classes=["clean-elements"]):
                            with gr.Column(scale=10, elem_classes=["clean-elements"]):
                                chat_components["user_input"] = gr.Textbox(
                                    label="User Input",
                                    lines=temporary.INPUT_LINES,
                                    interactive=False,
                                    placeholder="Enter text here..."
                                )
                            with gr.Column(min_width=166, elem_classes=["clean-elements"]):
                                switches = dict(
                                    web_search=gr.Checkbox(label="Internet", value=False, visible=True),
                                    tot=gr.Checkbox(label="T.O.T.", value=False, visible=True),
                                    enable_think=gr.Checkbox(label="THINK", value=False, visible=False)
                                )
                        with gr.Row(elem_classes=["clean-elements"]):
                            action_buttons = {}
                            with gr.Column(elem_classes=["clean-elements"], scale=40):
                                action_buttons["action"] = gr.Button(
                                    "Send Input",
                                    variant="secondary",
                                    elem_classes=["send-button"]
                                )
                            with gr.Column(elem_classes=["clean-elements"], min_width=166):
                                action_buttons["edit_previous"] = gr.Button("Edit Last Input", variant="huggingface")
                                action_buttons["copy_response"] = gr.Button("Copy Last Output", variant="huggingface")

                with gr.Row():
                    status_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        value="Select model on Configuration page.",
                        scale=30
                    )
                    gr.Button("Exit Program", variant="stop", elem_classes=["double-height"]).click(
                        fn=shutdown_program,
                        inputs=[states["models_loaded"]]
                    )

            # Configuration tab remains unchanged for this update
            with gr.Tab("Configuration"):
                with gr.Column(scale=1, elem_classes=["clean-elements"]):
                    is_cpu_only = temporary.BACKEND_TYPE in ["CPU Only - AVX2", "CPU Only - AVX512", "CPU Only - NoAVX", "CPU Only - OpenBLAS"]
                    config_components = {}
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("CPU/GPU Options...")
                    with gr.Row(visible=not is_cpu_only, elem_classes=["clean-elements"]):
                        config_components.update(
                            gpu=gr.Dropdown(choices=utility.get_available_gpus(), label="Select GPU", value=temporary.SELECTED_GPU, scale=5),
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
                            model=gr.Dropdown(choices=available_models, label="Select Model", value=default_model, allow_custom_value=False, scale=10)
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            ctx=gr.Dropdown(choices=temporary.CTX_OPTIONS, label="Context Size", value=temporary.N_CTX, scale=5),
                            batch=gr.Dropdown(choices=temporary.BATCH_OPTIONS, label="Batch Size", value=temporary.N_BATCH, scale=5),
                            temp=gr.Dropdown(choices=temporary.TEMP_OPTIONS, label="Temperature", value=temporary.TEMPERATURE, scale=5),
                            repeat=gr.Dropdown(choices=temporary.REPEAT_OPTIONS, label="Repeat Penalty", value=temporary.REPEAT_PENALTY, scale=5),
                        )
                        with gr.Column(elem_classes=["clean-elements"]):
                            config_components.update(
                                stream_output=gr.Checkbox(label="Stream Output", value=temporary.STREAM_OUTPUT),
                            )
                    with gr.Row(elem_classes=["clean-elements"]):
                        config_components.update(
                            browse=gr.Button("Browse", variant="secondary", elem_classes=["double-height"]), 
                            load_models=gr.Button("Load Model", variant="secondary", elem_classes=["double-height"]),
                            inspect_model=gr.Button("Inspect Model", variant="huggingface", elem_classes=["double-height"]),
                            unload=gr.Button("Unload Model", elem_classes=["double-height"], variant="huggingface"),
                        )
                    with gr.Row(elem_classes=["clean-elements"]):
                        gr.Markdown("Interface Options...")
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
                                afterthought_time=gr.Checkbox(label="After-Thought Time", value=temporary.AFTERTHOUGHT_TIME)
                            )
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

        # Define the start_new_session function
        def start_new_session():
            from scripts import temporary
            temporary.current_session_id = None
            temporary.session_label = ""
            temporary.SESSION_ACTIVE = True
            return [], "Type input and click Send to begin...", gr.update(interactive=True)

        # New function to update panel options based on mode
        def update_panel_on_mode_change(mode, current_panel):
            mode = mode.lower()
            if mode == "rpg":
                choices = ["History", "Files", "Roleplay"]
            else:
                choices = ["History", "Files"]
            
            # Set the default panel to "History" if available
            new_panel = "History" if "History" in choices else choices[0]
            
            # Define visibility based on the selected panel
            attachments_visible = new_panel == "Files"
            rpg_visible = new_panel == "Roleplay" and mode == "rpg"
            history_visible = new_panel == "History"
            attach_files_visible = new_panel == "Files"  # Explicitly control "Attach New Files"
            
            return (
                gr.update(choices=choices, value=new_panel),  # Update panel toggle options
                gr.update(visible=attachments_visible),       # attachments_group visibility
                gr.update(visible=rpg_visible),               # rpg_config_group visibility
                gr.update(visible=history_visible),           # history_slots_group visibility
                new_panel,                                    # Update selected_panel state
                gr.update(visible=attach_files_visible)       # Explicitly set attach_files visibility
            )

        # Define placeholder action_handler to resolve NameError
        def action_handler(
            interaction_phase,
            user_input,
            session_log,
            tot,
            loaded_files,
            enable_think,
            is_reasoning_model,
            rp_location,
            user_name,
            user_role,
            ai_npc,
            cancel_flag,
            mode_selection,
            web_search,
            models_loaded,
            interaction_phase_state
        ):
            print("Inputs received:", user_input)
            # Initial return before async processing
            return (
                session_log,              # chat_components["session_log"]
                "Processing input...",    # status_text
                gr.update(value="Send Input", variant="secondary"),  # action_buttons["action"]
                False,                    # states["cancel_flag"]
                loaded_files,             # states["loaded_files"]
                "waiting_for_input",      # states["interaction_phase"]
                gr.update(value="")       # chat_components["user_input"]
            )

        def update_config_settings(ctx, batch, temp, repeat, vram, gpu, cpu, model, model_dir):
            print(f"Updating temporary: model_dir={model_dir}, model_name={model}")
            temporary.N_CTX = int(ctx)
            temporary.N_BATCH = int(batch)
            temporary.TEMPERATURE = float(temp)
            temporary.REPEAT_PENALTY = float(repeat)
            temporary.VRAM_SIZE = int(vram)
            temporary.SELECTED_GPU = gpu
            temporary.SELECTED_CPU = cpu
            temporary.MODEL_NAME = model
            temporary.MODEL_FOLDER = model_dir
            print(f"Set temporary.MODEL_FOLDER={temporary.MODEL_FOLDER}, MODEL_NAME={temporary.MODEL_NAME}")
            return "Configuration updated"

        def update_stream_output(stream_output_value):
            temporary.STREAM_OUTPUT = stream_output_value
            return f"Stream Output set to: {stream_output_value}"

        def save_all_settings():
            return utility.save_config()

        # Updated Event Handlers
        start_new_session_btn.click(
            fn=start_new_session,
            outputs=[chat_components["session_log"], status_text, chat_components["user_input"]]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        )

        action_buttons["action"].click(
            fn=chat_interface,
            inputs=[
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
                mode_selection,
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
                chat_components["user_input"],
                switches["enable_think"]  # Added to handle auto-disable
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

        attach_files.upload(
            fn=process_uploaded_files,
            inputs=[attach_files, states["loaded_files"], states["models_loaded"]],
            outputs=[status_text, states["loaded_files"]]
        ).then(
            fn=update_file_slot_ui,
            inputs=[states["loaded_files"]],
            outputs=file_slots + [attach_files]
        )

        for i, btn in enumerate(file_slots):
            btn.click(
                fn=eject_file,
                inputs=[states["loaded_files"], gr.State(value=i)],
                outputs=[states["loaded_files"], status_text] + file_slots + [attach_files]
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

            mode_selection.change(
                fn=lambda mode: (
                    gr.update(value=False),  # Reset web_search
                    gr.update(value=False),  # Reset tot
                    gr.update(value=False) if models.get_model_settings(temporary.MODEL_NAME)["is_reasoning"] else gr.update(),  # Reset enable_think if reasoning model
                ),
                inputs=[mode_selection],
                outputs=[switches["web_search"], switches["tot"], switches["enable_think"]]
            ).then(
                fn=update_panel_on_mode_change,
                inputs=[mode_selection, states["selected_panel"]],
                outputs=[panel_toggle, attachments_group, rpg_config_group, history_slots_group, states["selected_panel"], attach_files]
            ).then(
                fn=update_mode_based_options,
                inputs=[mode_selection, states["showing_rpg_right"], states["is_reasoning_model"]],
                outputs=[switches["tot"], switches["web_search"], switches["enable_think"]]
            )

        panel_toggle.change(
            fn=lambda panel: panel,
            inputs=[panel_toggle],
            outputs=[states["selected_panel"]]
        ).then(
            fn=lambda panel, mode: (
                gr.update(visible=panel == "Files"),
                gr.update(visible=panel == "Roleplay" and mode.lower() == "rpg"),
                gr.update(visible=panel == "History"),
                gr.update(visible=panel == "Files")  # Update "Attach New Files" visibility
            ),
            inputs=[states["selected_panel"], mode_selection],
            outputs=[attachments_group, rpg_config_group, history_slots_group, attach_files]
        )

        rpg_fields["save_rpg"].click(
            fn=save_rp_settings,
            inputs=list(rpg_fields.values())[:-1],
            outputs=list(rpg_fields.values())[:-1]
        )

        config_components["model"].change(
            fn=lambda model_name: gr.update(visible=models.get_model_settings(model_name)["is_reasoning"]),
            inputs=[config_components["model"]],
            outputs=[switches["enable_think"]]
        ).then(
            fn=lambda model_name: models.get_model_settings(model_name)["is_reasoning"],
            inputs=[config_components["model"]],
            outputs=[states["is_reasoning_model"]]
        )

        # Remaining event handlers for Configuration tab remain unchanged
        for comp in [config_components[k] for k in ["ctx", "batch", "temp", "repeat", "vram", "gpu", "cpu", "model"]]:
            comp.change(
                fn=update_config_settings,
                inputs=[config_components[k] for k in ["ctx", "batch", "temp", "repeat", "vram", "gpu", "cpu", "model"]] + [config_components["model_dir"]],
                outputs=[config_components["status_settings"]]
            )

        config_components["browse"].click(
            fn=select_directory,
            inputs=[model_folder_state],
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

        model_folder_state.change(
            fn=lambda f: f"Model directory updated to: {f}",
            inputs=[model_folder_state],
            outputs=[config_components["status_settings"]]
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
            outputs=file_slots + [attach_files]
        )

        custom_components["afterthought_time"].change(
            fn=lambda v: setattr(temporary, "AFTERTHOUGHT_TIME", v),
            inputs=[custom_components["afterthought_time"]],
            outputs=[config_components["status_settings"]]
        )

        demo.load(
            fn=lambda model_name: models.get_model_settings(model_name)["is_reasoning"],
            inputs=[config_components["model"]],
            outputs=[states["is_reasoning_model"]]
        ).then(
            fn=update_panel_on_mode_change,
            inputs=[mode_selection, states["selected_panel"]],
            outputs=[panel_toggle, attachments_group, rpg_config_group, history_slots_group, states["selected_panel"], attach_files]
        ).then(
            fn=update_mode_based_options,
            inputs=[mode_selection, states["showing_rpg_right"], states["is_reasoning_model"]],
            outputs=[switches["tot"], switches["web_search"], switches["enable_think"]]
        ).then(
            fn=update_session_buttons,
            inputs=[],
            outputs=buttons["session"]
        ).then(
            fn=update_file_slot_ui,
            inputs=[states["loaded_files"]],
            outputs=file_slots + [attach_files]
        )

        demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, show_api=False)

if __name__ == "__main__":
    launch_interface()