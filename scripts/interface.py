# Script: `.\scripts\interface.py`

# Imports...
import gradio as gr
from gradio import themes
import re, os, json, pyperclip, yake, random, asyncio
from pathlib import Path
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import scripts.temporary as temporary
from scripts.temporary import (
    USER_COLOR, THINK_COLOR, RESPONSE_COLOR, SEPARATOR, MID_SEPARATOR,
    ALLOWED_EXTENSIONS, N_CTX, VRAM_SIZE, SELECTED_GPU, SELECTED_CPU,
    current_model_settings, N_GPU_LAYERS, VRAM_OPTIONS, REPEAT_OPTIONS,
    REPEAT_PENALTY, MLOCK, HISTORY_DIR, BATCH_OPTIONS, N_BATCH, MODEL_FOLDER,
    MODEL_NAME, STATUS_TEXTS, CTX_OPTIONS, RP_LOCATION, USER_PC_NAME, USER_PC_ROLE,
    AI_NPC1_NAME, AI_NPC2_NAME, AI_NPC3_NAME, AI_NPCS_ROLES, SESSION_ACTIVE, TOT_VARIATIONS, RAG_MAX_DOCS
)
from scripts import utility
from scripts.utility import (
    delete_vectorstore, delete_all_vectorstores, web_search, get_saved_sessions,
    load_session_history, save_session_history, load_and_chunk_documents,
    create_vectorstore, get_available_gpus, create_session_vectorstore, save_config
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

def web_search_trigger(query):
    try:
        result = utility.web_search(query)
        return result if result else "No results found"
    except Exception as e:
        return f"Error: {str(e)}"

def delete_npc(npc_index, ai_npc1, ai_npc2, ai_npc3):
    npc_values = [ai_npc1, ai_npc2, ai_npc3]
    npc_values[npc_index] = "Unused"
    return npc_values[0], npc_values[1], npc_values[2]

def save_rp_settings(rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles):
    from scripts import temporary
    config_path = Path("data/persistent.json")
    
    # Update temporary variables
    temporary.RP_LOCATION = rp_location
    temporary.USER_PC_NAME = user_name
    temporary.USER_PC_ROLE = user_role
    temporary.AI_NPC1_NAME = ai_npc1
    temporary.AI_NPC2_NAME = ai_npc2
    temporary.AI_NPC3_NAME = ai_npc3
    temporary.AI_NPCS_ROLES = ai_npcs_roles
    
    # Save to config file and return updated values
    utility.save_config()
    return (
        rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles,
        rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles  # Sync both sides
    )

def process_uploaded_files(files, loaded_files, operation_mode, models_loaded):
    """
    Process uploaded files for RAG integration and update session vector store.
    """
    from scripts.utility import create_session_vectorstore
    if not models_loaded:
        return "Error: Load a model first.", loaded_files
    
    max_files = RAG_MAX_DOCS  # Always 6
    if len(loaded_files) >= max_files:
        return f"Max files ({max_files}) reached.", loaded_files
    
    new_files = [f for f in files if f not in loaded_files]
    available_slots = max_files - len(loaded_files)
    loaded_files.extend(new_files[:available_slots])
    
    # Update session vector store with new files
    session_vectorstore = create_session_vectorstore(loaded_files)
    context_injector.set_session_vectorstore(session_vectorstore)
    
    return f"Processed {min(len(new_files), available_slots)} new files.", loaded_files

def eject_file(loaded_files, slot_index, rag_max_docs):
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
    button_updates, attach_files_update = update_file_slot_ui(loaded_files, rag_max_docs)
    return [loaded_files, status_msg] + button_updates + [attach_files_update]

def remove_all_attachments(loaded_files, rag_max_docs):
    from scripts.utility import create_session_vectorstore
    loaded_files = []
    session_vectorstore = create_session_vectorstore(loaded_files)
    context_injector.set_session_vectorstore(session_vectorstore)
    status_msg = "All attachments removed."
    button_updates, attach_files_update = update_file_slot_ui(loaded_files, rag_max_docs)
    return [loaded_files, status_msg] + button_updates + [attach_files_update]

def toggle_rpg_settings(mode, showing_rpg_right):
    if mode != "Rpg":
        # If not in RPG mode, button should be hidden, so return defaults (though click shouldn't occur)
        return (
            gr.update(visible=False),  # toggle_rpg_settings_btn
            gr.update(visible=True),   # file_attachments_group
            gr.update(visible=False),  # rpg_settings_group
            False                      # showing_rpg_right
        )
    # Toggle the state
    new_showing_rpg_right = not showing_rpg_right
    toggle_label = "Show File Attachments" if new_showing_rpg_right else "Show RPG Settings"
    return (
        gr.update(visible=True, value=toggle_label),  # toggle_rpg_settings_btn
        gr.update(visible=not new_showing_rpg_right), # file_attachments_group
        gr.update(visible=new_showing_rpg_right),     # rpg_settings_group
        new_showing_rpg_right                         # Update state
    )

def handle_slot_click(slot_index, loaded_files, rag_max_docs):
    if slot_index < len(loaded_files):
        removed_file = loaded_files.pop(slot_index)
        try:
            Path(removed_file).unlink()
        except Exception as e:
            print(f"Error unlinking file: {e}")
        docs = load_and_chunk_documents([Path(f) for f in loaded_files])
        create_vectorstore(docs, "chat")
        status_msg = f"Ejected {Path(removed_file).name}"
    else:
        status_msg = "Click 'Attach Files' to add files."
    return [loaded_files, status_msg] + update_file_slot_ui(loaded_files, rag_max_docs)

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
        return utility.load_session_history(Path(HISTORY_DIR) / session_file)
    return [], "No session to load"

def copy_last_response(session_log):
    if session_log and session_log[-1][1]:
        response = session_log[-1][1]
        clean_response = re.sub(r'<[^>]+>', '', response)
        pyperclip.copy(clean_response)
        return "AI Response copied to clipboard."
    return "No response available to copy."

def cancel_input():
    return True, gr.update(visible=True), gr.update(visible=False), "Input cancelled."

def determine_operation_mode(quality_model):
    if quality_model == "Select_a_model...":
        return "Select models to enable mode detection."
    settings = get_model_settings(quality_model)
    category = settings["category"]
    if category == "code":
        return "Code"
    elif category == "rpg":
        return "Rpg"
    elif category == "uncensored":
        return "Chat"
    return "Chat"

def update_dynamic_options(quality_model, loaded_files_state, showing_rpg_right):
    if quality_model == "Select_a_model...":
        mode = "Select models"
        settings = {"is_reasoning": False}
    else:
        settings = get_model_settings(quality_model)
        mode = settings["category"].capitalize()
    
    think_visible = settings["is_reasoning"]
    
    if mode == "Code":
        tot_visible = False
        web_visible = False
        file_visible = True
    elif mode == "Rpg":
        tot_visible = False
        web_visible = False
        file_visible = True
    else:  # Chat mode
        tot_visible = True
        web_visible = True
        file_visible = True

    # Right column visibility logic
    toggle_visible = True  # Always visible for development
    toggle_label = "Show File Attachments" if showing_rpg_right else "Show RPG Settings"
    
    return [
        gr.update(visible=tot_visible),      # TOT checkbox
        gr.update(visible=web_visible),      # Web search
        gr.update(visible=think_visible),    # Think switch
        gr.update(value=mode),               # Theme status
        gr.update(visible=file_visible),     # attach_files_btn
        gr.update(visible=toggle_visible, value=toggle_label),  # toggle_rpg_settings_btn
        gr.update(visible=not showing_rpg_right),  # file_attachments_group
        gr.update(visible=showing_rpg_right),      # rpg_settings_group
        showing_rpg_right                          # Update the state
    ]

def update_file_slot_ui(loaded_files, rag_max_docs):
    from pathlib import Path
    button_updates = []
    for i in range(6):
        if i < len(loaded_files):
            filename = Path(loaded_files[i]).name
            short_name = (filename[:13] + ".." if len(filename) > 15 else filename)
            label = f"{short_name}"
            variant = "secondary"
            visible = True
        else:
            label = ""
            variant = "primary"
            visible = False
        button_updates.append(gr.update(value=label, visible=visible, variant=variant))
    attach_files_visible = len(loaded_files) < rag_max_docs
    return button_updates + [gr.update(visible=attach_files_visible)]

def update_rp_settings(rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles):
    temporary.RP_LOCATION = rp_location
    temporary.USER_PC_NAME = user_name
    temporary.USER_PC_ROLE = user_role
    temporary.AI_NPC1_NAME = ai_npc1
    temporary.AI_NPC2_NAME = ai_npc2
    temporary.AI_NPC3_NAME = ai_npc3
    temporary.AI_NPCS_ROLES = ai_npcs_roles
    return rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles

def save_rpg_settings_to_json():
    config_path = Path("data/persistent.json")
    if config_path.exists():
        with open(config_path, "r+") as f:
            config = json.load(f)
            config["rp_settings"] = {
                "rp_location": temporary.RP_LOCATION,
                "user_name": temporary.USER_PC_NAME,
                "user_role": temporary.USER_PC_ROLE,
                "ai_npc1": temporary.AI_NPC1_NAME,
                "ai_npc2": temporary.AI_NPC2_NAME,
                "ai_npc3": temporary.AI_NPC3_NAME,
                "ai_npcs_roles": temporary.AI_NPCS_ROLES
            }
            f.seek(0)
            json.dump(config, f, indent=2)
            f.truncate()
        return "RPG settings saved to persistent.json"
    else:
        return "Configuration file not found."

def update_session_buttons():
    sessions = utility.get_saved_sessions()
    button_updates = []
    for i in range(9):
        if i < len(sessions):
            session_path = Path(HISTORY_DIR) / sessions[i]
            try:
                label, _ = utility.load_session_history(session_path)
                btn_label = f"{label}" if label else f"Session {i+1}"
            except Exception:
                btn_label = f"Session {i+1}"
        else:
            btn_label = "History Slot Empty"
        button_updates.append(gr.update(value=btn_label, visible=True))
    return button_updates

def update_left_panel_visibility(mode, showing_rpg_settings):
    is_rpg_mode = (mode.lower() == "rpg")
    if is_rpg_mode and showing_rpg_settings:
        return gr.update(visible=False), gr.update(visible=True)
    else:
        return gr.update(visible=True), gr.update(visible=False)

async def chat_interface(user_input, session_log, tot_enabled, loaded_files_state, disable_think, 
                        rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, cancel_flag):
    from scripts import temporary, utility, models
    from scripts.temporary import STATUS_TEXTS, MODEL_NAME, SESSION_ACTIVE, TOT_VARIATIONS
    
    if not SESSION_ACTIVE:
        yield session_log, "Please start a new session first.", gr.update(), gr.update(), cancel_flag, loaded_files_state
        return
    if not user_input.strip():
        yield session_log, "No input provided.", gr.update(), gr.update(), cancel_flag, loaded_files_state
        return

    # Append user input and initialize response
    session_log.append((user_input, ""))
    yield session_log, "Processing...", gr.update(visible=False), gr.update(visible=True), False, loaded_files_state

    # Determine countdown based on input lines, respecting AFTERTHOUGHT_TIME
    if temporary.AFTERTHOUGHT_TIME:
        num_lines = len(user_input.split('\n'))
        if num_lines >= 10:
            countdown_seconds = 6
        elif num_lines >= 5:
            countdown_seconds = 4
        else:
            countdown_seconds = 2
    else:
        countdown_seconds = 1

    # Countdown with cancellation support
    for i in range(countdown_seconds, 0, -1):
        session_log[-1] = (user_input, f"Afterthought countdown... {i}s")
        yield session_log, "Counting down...", gr.update(visible=False), gr.update(visible=True), False, loaded_files_state
        await asyncio.sleep(1)
        if cancel_flag:
            session_log[-1] = (user_input, "Input cancelled.")
            yield session_log, "Input cancelled.", gr.update(visible=True), gr.update(visible=False), False, loaded_files_state
            return

    # Generate response
    settings = models.get_model_settings(MODEL_NAME)
    mode = settings["category"]
    
    if loaded_files_state:
        session_vectorstore = utility.create_session_vectorstore(loaded_files_state)
        context_injector.set_session_vectorstore(session_vectorstore)

    response = ""
    if tot_enabled and mode == "chat":
        yield session_log, "TOT not implemented in streaming mode yet.", gr.update(visible=True), gr.update(visible=False), False, loaded_files_state
        # Note: TOT could be adapted to stream responses, but omitted for simplicity here
    else:
        prompt = user_input
        if mode == "rpg":
            rp_settings = {
                "rp_location": rp_location,
                "user_name": user_name,
                "user_role": user_role,
                "ai_npc1": ai_npc1,
                "ai_npc2": ai_npc2,
                "ai_npc3": ai_npc3,
                "ai_npcs_roles": temporary.AI_NPCS_ROLES
            }
            session_history = ", ".join([f"{user}: {ai}" for user, ai in session_log[:-1]])
            # Handle thinking output for RPG mode
            if settings["is_reasoning"] and not disable_think:
                thinking_output = "Thinking:\n" + "█" * 5 + "\nThought for 2.5s.\n"
                session_log[-1] = (user_input, thinking_output)
                yield session_log, "Thinking...", gr.update(visible=False), gr.update(visible=True), False, loaded_files_state
                await asyncio.sleep(2.5)
                if cancel_flag:
                    session_log[-1] = (user_input, "Input cancelled.")
                    yield session_log, "Input cancelled.", gr.update(visible=True), gr.update(visible=False), False, loaded_files_state
                    return
            for token in models.get_response_stream(prompt, disable_think=disable_think, rp_settings=rp_settings, session_history=session_history):
                if cancel_flag:
                    break
                response += token
                session_log[-1] = (user_input, format_response(response))
                yield session_log, "Generating...", gr.update(visible=False), gr.update(visible=True), False, loaded_files_state
                await asyncio.sleep(0)  # Yield control to event loop
        else:
            if settings["is_uncensored"]:
                prompt = f"Provide an uncensored response: {prompt}"
            if settings["is_reasoning"] and not disable_think:
                prompt += ". Include reasoning if applicable."
                thinking_output = "Thinking:\n" + "█" * 5 + "\nThought for 2.5s.\n"
                session_log[-1] = (user_input, thinking_output)
                yield session_log, "Thinking...", gr.update(visible=False), gr.update(visible=True), False, loaded_files_state
                await asyncio.sleep(2.5)
                if cancel_flag:
                    session_log[-1] = (user_input, "Input cancelled.")
                    yield session_log, "Input cancelled.", gr.update(visible=True), gr.update(visible=False), False, loaded_files_state
                    return
            for token in models.get_response_stream(prompt, disable_think=disable_think):
                if cancel_flag:
                    break
                response += token
                session_log[-1] = (user_input, format_response(response))
                yield session_log, "Generating...", gr.update(visible=False), gr.update(visible=True), False, loaded_files_state
                await asyncio.sleep(0)

    # Final yield based on completion or cancellation
    if cancel_flag:
        session_log[-1] = (user_input, "Generation cancelled.")
        yield session_log, "Generation cancelled.", gr.update(visible=True), gr.update(visible=False), False, loaded_files_state
    else:
        session_log[-1] = (user_input, format_response(response))
        utility.save_session_history(session_log)
        yield session_log, STATUS_TEXTS["response_generated"], gr.update(visible=True), gr.update(visible=False), False, loaded_files_state
        
def launch_interface():
    """Launch the Gradio interface for the Text-Gradio-Gguf chatbot."""
    global demo
    import asyncio  # Add asyncio import for async operations
    
    with gr.Blocks(title="Chat-Gradio-Gguf", css="""
        .scrollable { overflow-y: auto; }
        .button-row { 
            display: flex !important; 
            flex-direction: row !important; 
            flex-wrap: nowrap !important; 
            gap: 4px !important; 
            margin-bottom: 4px !important; 
        }
        .button-row .gradio-button { 
            min-width: 100px !important; 
            text-overflow: ellipsis; 
            overflow: hidden; 
            white-space: nowrap; 
        }
        .collapsible { margin-bottom: 10px; border: 1px solid #444; padding: 5px; border-radius: 5px; }
        .gradio-button {
            border: 1px solid #ccc;
        }
    """) as demo:
        # State variables
        loaded_files_state = gr.State(value=[])
        rag_max_docs_state = gr.State(value=6)
        models_loaded_state = gr.State(value=False)
        showing_rpg_right = gr.State(False)
        cancel_flag = gr.State(False)  # New state for cancellation tracking

        with gr.Tabs():
            with gr.Tab("Conversation"):
                with gr.Row():
                    # Left column (controls)
                    with gr.Column(scale=1):
                        theme_status = gr.Textbox(label="Operation Mode", interactive=False, value="No model loaded.")
                        
                        with gr.Column(visible=True) as session_history_column:
                            start_new_session_btn = gr.Button("Start New Session...", variant="secondary")
                            session_buttons = [gr.Button(f"History Slot {i+1}", visible=True, variant="huggingface") for i in range(9)]
                            delete_all_history_btn = gr.Button("Delete All History", variant="primary")
                        
                        with gr.Column(visible=False) as rpg_settings_column:
                            show_session_history_btn = gr.Button("Show Session History")
                            rp_location = gr.Textbox(label="RP Location", value=temporary.RP_LOCATION)
                            user_name = gr.Textbox(label="User Name", value=temporary.USER_PC_NAME)
                            user_role = gr.Textbox(label="User Role", value=temporary.USER_PC_ROLE)
                            ai_npc1 = gr.Textbox(label="AI NPC 1", value=temporary.AI_NPC1_NAME)
                            ai_npc2 = gr.Textbox(label="AI NPC 2", value=temporary.AI_NPC2_NAME)
                            ai_npc3 = gr.Textbox(label="AI NPC 3", value=temporary.AI_NPC3_NAME)
                            ai_npcs_roles = gr.Textbox(label="AI Roles", value=temporary.AI_NPCS_ROLES)
                            save_rpg_btn = gr.Button("Save RPG Settings")

                        with gr.Row():
                            web_search_switch = gr.Checkbox(label="Web-Search", value=False, visible=False)
                            tot_checkbox = gr.Checkbox(label="Enable TOT", value=False, visible=False)
                            disable_think_switch = gr.Checkbox(label="Disable THINK", value=False, visible=False)

                    # Middle column (chat area)
                    with gr.Column(scale=30):
                        session_log = gr.Chatbot(label="Session Log", height=475, elem_classes=["scrollable"], type="messages")             
                        with gr.Row():
                            user_input = gr.Textbox(label="User Input", lines=5, interactive=False, placeholder="Enter text here...")
                        with gr.Row():
                            send_btn = gr.Button("Send Input", variant="secondary", scale=20)
                            cancel_btn = gr.Button("Cancel Input", variant="stop", visible=False)  # New Cancel button
                            edit_previous_btn = gr.Button("Edit Last Input", variant="huggingface")
                            copy_response_btn = gr.Button("Copy Last Output", variant="huggingface")

                    # Right column (file attachments and RPG settings)
                    with gr.Column(scale=1):
                        toggle_rpg_settings_btn = gr.Button("Show RPG Settings", visible=True, variant="secondary")
                        
                        with gr.Group(visible=True) as file_attachments_group:
                            attach_files_btn = gr.UploadButton("Attach New Files", 
                                                              file_types=[f".{ext}" for ext in temporary.ALLOWED_EXTENSIONS], 
                                                              file_count="multiple", 
                                                              variant="secondary")
                            file_slot_buttons = []
                            for row in range(6):
                                with gr.Row(elem_classes=["button-row"]):
                                    file_slot_buttons.append(gr.Button("File Slot Free", visible=True, variant="huggingface"))
                            remove_all_attachments_btn = gr.Button("Remove All Attachments", variant="primary")
                        
                        with gr.Group(visible=False) as rpg_settings_group:
                            rp_location_right = gr.Textbox(label="RP Location", value=temporary.RP_LOCATION)
                            user_name_right = gr.Textbox(label="User Name", value=temporary.USER_PC_NAME)
                            user_role_right = gr.Textbox(label="User Role", value=temporary.USER_PC_ROLE)
                            ai_npc1_right = gr.Textbox(label="AI NPC 1", value=temporary.AI_NPC1_NAME)
                            ai_npc2_right = gr.Textbox(label="AI NPC 2", value=temporary.AI_NPC2_NAME)
                            ai_npc3_right = gr.Textbox(label="AI NPC 3", value=temporary.AI_NPC3_NAME)
                            ai_npcs_roles_right = gr.Textbox(label="AI Roles", value=temporary.AI_NPCS_ROLES)
                            save_rpg_right_btn = gr.Button("Save RPG Settings")

                with gr.Row():
                    status_text = gr.Textbox(label="Status", interactive=False, value="Select model on Configuration page.", scale=20)
                    shutdown_btn = gr.Button("Exit Program", variant="stop")

            # Configuration Tab
            with gr.Tab("Configuration"):
                with gr.Row():
                    gpu_dropdown = gr.Dropdown(
                        choices=utility.get_available_gpus(),
                        label="Select GPU",
                        value=temporary.SELECTED_GPU,
                        visible=not (temporary.BACKEND_TYPE in [
                            "CPU Only - AVX2", "CPU Only - AVX512", "CPU Only - NoAVX", "CPU Only - OpenBLAS"
                        ]),
                        scale=20
                    )
                    cpu_info = utility.get_cpu_info()
                    cpu_choices = [cpu["label"] for cpu in cpu_info]
                    cpu_dropdown = gr.Dropdown(
                        choices=cpu_choices,
                        label="Select CPU",
                        value=(temporary.SELECTED_CPU if temporary.SELECTED_CPU in cpu_choices 
                               else cpu_choices[0] if cpu_choices else None),
                        visible=temporary.BACKEND_TYPE in [
                            "CPU Only - AVX2", "CPU Only - AVX512", "CPU Only - NoAVX", "CPU Only - OpenBLAS"
                        ],
                        scale=20
                    )
                with gr.Row():                
                    vram_dropdown = gr.Dropdown(choices=temporary.VRAM_OPTIONS, label="Assign Free VRam", value=temporary.VRAM_SIZE, interactive=True)
                    mlock_checkbox = gr.Checkbox(label="MLock Enabled", value=temporary.MLOCK)
                with gr.Row():
                    model_dir_text = gr.Textbox(label="Model Folder", value=temporary.MODEL_FOLDER, scale=20)
                    browse_btn = gr.Button("Browse", variant="secondary")
                with gr.Row():
                    mode_text = gr.Textbox(label="Mode Detected", interactive=False, value="No Model Selected", scale=1)
                    model_dropdown = gr.Dropdown(
                        choices=get_available_models(),
                        label="Select Model",
                        value=temporary.MODEL_NAME,
                        allow_custom_value=True,
                        scale=20
                    )
                    refresh_btn = gr.Button("Refresh")
                with gr.Row():
                    ctx_dropdown = gr.Dropdown(choices=temporary.CTX_OPTIONS, label="Context Size", value=temporary.N_CTX, interactive=True)
                    batch_dropdown = gr.Dropdown(choices=temporary.BATCH_OPTIONS, label="Batch Size", value=temporary.N_BATCH, interactive=True)
                    temp_dropdown = gr.Dropdown(choices=temporary.TEMP_OPTIONS, label="Temperature", value=temporary.TEMPERATURE, interactive=True)
                    repeat_dropdown = gr.Dropdown(choices=temporary.REPEAT_OPTIONS, label="Repeat Penalty", value=temporary.REPEAT_PENALTY, interactive=True)
                with gr.Row():
                    load_models_btn = gr.Button("Load Model", variant="secondary")
                    inspect_model_btn = gr.Button("Inspect Model", variant="huggingface")
                    unload_btn = gr.Button("Unload Model")
                with gr.Row():
                    afterthought_checkbox = gr.Checkbox(label="Afterthought Time", value=temporary.AFTERTHOUGHT_TIME)  # New checkbox
                with gr.Row():
                    erase_chat_btn = gr.Button("Erase Chat Data", variant="primary")
                    erase_rpg_btn = gr.Button("Erase RPG Data", variant="primary")
                    erase_code_btn = gr.Button("Erase Code Data", variant="primary")
                    erase_all_btn = gr.Button("Erase All Data", variant="primary")
                with gr.Row():
                    status_text_settings = gr.Textbox(label="Status", interactive=False, scale=20)
                    save_settings_btn = gr.Button("Save Settings", variant="stop")

        # Event Handlers
        def update_config_settings(ctx, batch, temp, repeat, vram, mlock, afterthought, gpu, cpu, model_dir, model):
            temporary.N_CTX = ctx
            temporary.N_BATCH = batch
            temporary.TEMPERATURE = temp
            temporary.REPEAT_PENALTY = repeat
            temporary.VRAM_SIZE = vram
            temporary.MLOCK = mlock
            temporary.AFTERTHOUGHT_TIME = afterthought  # Update temporary variable
            temporary.SELECTED_GPU = gpu if gpu else temporary.SELECTED_GPU
            temporary.SELECTED_CPU = cpu if cpu else temporary.SELECTED_CPU
            temporary.MODEL_FOLDER = model_dir
            temporary.MODEL_NAME = model
            return "Settings updated in memory. Click 'Save Settings' to persist."

        def save_all_settings():
            utility.save_config()
            return "Settings saved to persistent.json"

        def update_left_panel_visibility(theme_status):
            is_rpg_mode = "rpg" in theme_status.lower()
            return gr.update(visible=not is_rpg_mode), gr.update(visible=is_rpg_mode)

        def update_dynamic_options(quality_model, loaded_files_state, showing_rpg_right):
            if quality_model == "Select_a_model...":
                mode = "Select models"
                settings = {"is_reasoning": False}
            else:
                settings = get_model_settings(quality_model)
                mode = settings["category"].capitalize()
            
            think_visible = settings["is_reasoning"]
            if mode == "Code":
                tot_visible = False
                web_visible = False
                file_visible = True
            elif mode == "Rpg":
                tot_visible = False
                web_visible = False
                file_visible = True
            else:  # Chat mode
                tot_visible = True
                web_visible = True
                file_visible = True

            toggle_visible = True
            toggle_label = "Show File Attachments" if showing_rpg_right else "Show RPG Settings"
            return [
                gr.update(visible=tot_visible),
                gr.update(visible=web_visible),
                gr.update(visible=think_visible),
                gr.update(value=mode),
                gr.update(visible=file_visible),
                gr.update(visible=toggle_visible, value=toggle_label),
                gr.update(visible=not showing_rpg_right),
                gr.update(visible=showing_rpg_right),
                showing_rpg_right
            ]

        def toggle_rpg_settings(showing_rpg_right):
            showing_rpg_right = not showing_rpg_right
            toggle_label = "Show File Attachments" if showing_rpg_right else "Show RPG Settings"
            return (
                gr.update(visible=not showing_rpg_right),
                gr.update(visible=showing_rpg_right),
                gr.update(value=toggle_label),
                showing_rpg_right
            )

        # New cancel function
        def cancel_input():
            return True, gr.update(visible=True), gr.update(visible=False), "Input cancelled."

        # Event handlers for UI interactions
        start_new_session_btn.click(
            fn=start_new_session,
            inputs=None,
            outputs=[session_log, status_text, user_input]
        )
        
        send_btn.click(
            fn=chat_interface,
            inputs=[user_input, session_log, tot_checkbox, loaded_files_state, disable_think_switch, 
                    rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, cancel_flag],
            outputs=[session_log, status_text, send_btn, cancel_btn, cancel_flag, loaded_files_state]
        )
        
        cancel_btn.click(
            fn=cancel_input,
            inputs=None,
            outputs=[cancel_flag, send_btn, cancel_btn, status_text]
        )
        
        copy_response_btn.click(
            fn=copy_last_response,
            inputs=[session_log],
            outputs=[status_text]
        )
        
        attach_files_btn.upload(
            fn=process_uploaded_files,
            inputs=[attach_files_btn, loaded_files_state, theme_status, models_loaded_state],
            outputs=[status_text, loaded_files_state]
        )
        
        remove_all_attachments_btn.click(
            fn=remove_all_attachments,
            inputs=[loaded_files_state, rag_max_docs_state],
            outputs=[loaded_files_state, status_text] + file_slot_buttons + [attach_files_btn]
        )
        
        for i, btn in enumerate(file_slot_buttons):
            btn.click(
                fn=eject_file,
                inputs=[loaded_files_state, gr.State(value=i), rag_max_docs_state],
                outputs=[loaded_files_state, status_text] + file_slot_buttons + [attach_files_btn]
            )

        model_dropdown.change(
            fn=determine_operation_mode,
            inputs=[model_dropdown],
            outputs=[theme_status]
        ).then(
            fn=lambda mode: context_injector.set_mode(mode.lower()),
            inputs=[theme_status],
            outputs=None
        ).then(
            fn=update_dynamic_options,
            inputs=[model_dropdown, loaded_files_state, showing_rpg_right],
            outputs=[
                tot_checkbox,
                web_search_switch,
                disable_think_switch,
                theme_status,
                attach_files_btn,
                toggle_rpg_settings_btn,
                file_attachments_group,
                rpg_settings_group,
                showing_rpg_right
            ]
        ).then(
            fn=update_left_panel_visibility,
            inputs=[theme_status],
            outputs=[session_history_column, rpg_settings_column]
        ).then(
            fn=update_file_slot_ui,
            inputs=[loaded_files_state, rag_max_docs_state],
            outputs=file_slot_buttons + [attach_files_btn]
        )

        show_session_history_btn.click(
            fn=update_left_panel_visibility,
            inputs=[theme_status],
            outputs=[session_history_column, rpg_settings_column]
        )

        delete_all_history_btn.click(
            fn=lambda: ("All history deleted.", *update_session_buttons()),
            inputs=None,
            outputs=[status_text] + session_buttons
        )

        toggle_rpg_settings_btn.click(
            fn=toggle_rpg_settings,
            inputs=[showing_rpg_right],
            outputs=[file_attachments_group, rpg_settings_group, toggle_rpg_settings_btn, showing_rpg_right]
        )

        # Synchronize RPG settings
        for left, right in [
            (rp_location, rp_location_right),
            (user_name, user_name_right),
            (user_role, user_role_right),
            (ai_npc1, ai_npc1_right),
            (ai_npc2, ai_npc2_right),
            (ai_npc3, ai_npc3_right),
            (ai_npcs_roles, ai_npcs_roles_right)
        ]:
            left.change(fn=lambda x: gr.update(value=x), inputs=[left], outputs=[right])
            right.change(fn=lambda x: gr.update(value=x), inputs=[right], outputs=[left])

        save_rpg_btn.click(
            fn=save_rp_settings,
            inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles],
            outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles]
        )
        save_rpg_right_btn.click(
            fn=save_rp_settings,
            inputs=[rp_location_right, user_name_right, user_role_right, ai_npc1_right, ai_npc2_right, ai_npc3_right, ai_npcs_roles_right],
            outputs=[rp_location_right, user_name_right, user_role_right, ai_npc1_right, ai_npc2_right, ai_npc3_right, ai_npcs_roles_right,
                     rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles]
        )

        load_models_btn.click(
            fn=set_loading_status,
            inputs=None,
            outputs=[status_text_settings]
        ).then(
            fn=load_models,
            inputs=[model_dropdown, vram_dropdown],
            outputs=[status_text_settings, models_loaded_state]
        )

        # Config input event handlers - updated to include afterthought_checkbox
        ctx_dropdown.change(
            fn=update_config_settings,
            inputs=[ctx_dropdown, batch_dropdown, temp_dropdown, repeat_dropdown, vram_dropdown, mlock_checkbox, afterthought_checkbox, gpu_dropdown, cpu_dropdown, model_dir_text, model_dropdown],
            outputs=[status_text_settings]
        )
        batch_dropdown.change(
            fn=update_config_settings,
            inputs=[ctx_dropdown, batch_dropdown, temp_dropdown, repeat_dropdown, vram_dropdown, mlock_checkbox, afterthought_checkbox, gpu_dropdown, cpu_dropdown, model_dir_text, model_dropdown],
            outputs=[status_text_settings]
        )
        temp_dropdown.change(
            fn=update_config_settings,
            inputs=[ctx_dropdown, batch_dropdown, temp_dropdown, repeat_dropdown, vram_dropdown, mlock_checkbox, afterthought_checkbox, gpu_dropdown, cpu_dropdown, model_dir_text, model_dropdown],
            outputs=[status_text_settings]
        )
        repeat_dropdown.change(
            fn=update_config_settings,
            inputs=[ctx_dropdown, batch_dropdown, temp_dropdown, repeat_dropdown, vram_dropdown, mlock_checkbox, afterthought_checkbox, gpu_dropdown, cpu_dropdown, model_dir_text, model_dropdown],
            outputs=[status_text_settings]
        )
        vram_dropdown.change(
            fn=update_config_settings,
            inputs=[ctx_dropdown, batch_dropdown, temp_dropdown, repeat_dropdown, vram_dropdown, mlock_checkbox, afterthought_checkbox, gpu_dropdown, cpu_dropdown, model_dir_text, model_dropdown],
            outputs=[status_text_settings]
        )
        mlock_checkbox.change(
            fn=update_config_settings,
            inputs=[ctx_dropdown, batch_dropdown, temp_dropdown, repeat_dropdown, vram_dropdown, mlock_checkbox, afterthought_checkbox, gpu_dropdown, cpu_dropdown, model_dir_text, model_dropdown],
            outputs=[status_text_settings]
        )
        afterthought_checkbox.change(
            fn=update_config_settings,
            inputs=[ctx_dropdown, batch_dropdown, temp_dropdown, repeat_dropdown, vram_dropdown, mlock_checkbox, afterthought_checkbox, gpu_dropdown, cpu_dropdown, model_dir_text, model_dropdown],
            outputs=[status_text_settings]
        )
        gpu_dropdown.change(
            fn=update_config_settings,
            inputs=[ctx_dropdown, batch_dropdown, temp_dropdown, repeat_dropdown, vram_dropdown, mlock_checkbox, afterthought_checkbox, gpu_dropdown, cpu_dropdown, model_dir_text, model_dropdown],
            outputs=[status_text_settings]
        )
        cpu_dropdown.change(
            fn=update_config_settings,
            inputs=[ctx_dropdown, batch_dropdown, temp_dropdown, repeat_dropdown, vram_dropdown, mlock_checkbox, afterthought_checkbox, gpu_dropdown, cpu_dropdown, model_dir_text, model_dropdown],
            outputs=[status_text_settings]
        ).then(
            fn=save_all_settings,
            inputs=None,
            outputs=[status_text_settings]
        )
        model_dropdown.change(
            fn=update_config_settings,
            inputs=[ctx_dropdown, batch_dropdown, temp_dropdown, repeat_dropdown, vram_dropdown, mlock_checkbox, afterthought_checkbox, gpu_dropdown, cpu_dropdown, model_dir_text, model_dropdown],
            outputs=[status_text_settings]
        ).then(
            fn=determine_operation_mode,
            inputs=[model_dropdown],
            outputs=[theme_status]
        ).then(
            fn=lambda mode: context_injector.set_mode(mode.lower()),
            inputs=[theme_status],
            outputs=None
        ).then(
            fn=update_dynamic_options,
            inputs=[model_dropdown, loaded_files_state, showing_rpg_right],
            outputs=[
                tot_checkbox,
                web_search_switch,
                disable_think_switch,
                theme_status,
                attach_files_btn,
                toggle_rpg_settings_btn,
                file_attachments_group,
                rpg_settings_group,
                showing_rpg_right
            ]
        ).then(
            fn=update_left_panel_visibility,
            inputs=[theme_status],
            outputs=[session_history_column, rpg_settings_column]
        ).then(
            fn=update_file_slot_ui,
            inputs=[loaded_files_state, rag_max_docs_state],
            outputs=file_slot_buttons + [attach_files_btn]
        )

        # Erase button event handlers
        erase_chat_btn.click(fn=lambda: utility.delete_vectorstore("chat"), inputs=None, outputs=[status_text_settings])
        erase_rpg_btn.click(fn=lambda: utility.delete_vectorstore("rpg"), inputs=None, outputs=[status_text_settings])
        erase_code_btn.click(fn=lambda: utility.delete_vectorstore("code"), inputs=None, outputs=[status_text_settings])
        erase_all_btn.click(fn=utility.delete_all_vectorstores, inputs=None, outputs=[status_text_settings])

        save_settings_btn.click(fn=save_all_settings, inputs=None, outputs=[status_text_settings])

        demo.launch(
            server_name="127.0.0.1",
            server_port=7860,
            show_error=True,
            show_api=False
        )
        
if __name__ == "__main__":
    launch_interface()