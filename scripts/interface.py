# Script: `.\scripts\interface.py`

# Imports...
import gradio as gr
from gradio import themes
import re, os, json, pyperclip, yake, random
from pathlib import Path
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import scripts.temporary as temporary
from scripts.temporary import (
    USER_COLOR, THINK_COLOR, RESPONSE_COLOR, SEPARATOR, MID_SEPARATOR,
    ALLOWED_EXTENSIONS, N_CTX, VRAM_SIZE, SELECTED_GPU,
    current_model_settings, N_GPU_LAYERS, VRAM_OPTIONS, REPEAT_OPTIONS,
    REPEAT_PENALTY, MLOCK, HISTORY_DIR, BATCH_OPTIONS, N_BATCH, MODEL_FOLDER,
    MODEL_NAME, STATUS_TEXTS, CTX_OPTIONS, RP_LOCATION, USER_PC_NAME, USER_PC_ROLE,
    AI_NPC1_NAME, AI_NPC2_NAME, AI_NPC3_NAME, AI_NPCS_ROLES, SESSION_ACTIVE, TOT_VARIATIONS
)
from scripts import utility
from scripts.utility import (
    delete_vectorstore, delete_all_vectorstores, web_search, get_saved_sessions,
    load_session_history, save_session_history, load_and_chunk_documents,
    create_vectorstore, get_available_gpus
)
from scripts.models import (
    get_response, get_available_models, unload_models, get_model_settings,
    context_injector, inspect_model, load_models
)
from langchain_core.documents import Document

# Functions...
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
    config_path = Path("data/persistent.json")  # Add this line to define config_path
    
    # Update temporary variables
    temporary.RP_LOCATION = rp_location
    temporary.USER_PC_NAME = user_name
    temporary.USER_PC_ROLE = user_role
    temporary.AI_NPC1_NAME = ai_npc1
    temporary.AI_NPC2_NAME = ai_npc2
    temporary.AI_NPC3_NAME = ai_npc3
    temporary.AI_NPCS_ROLES = ai_npcs_roles
    
    # Save to config file
    with open(config_path, "r+") as f:
        config = json.load(f)
        config["rp_settings"] = {
            "rp_location": rp_location,
            "user_name": user_name,
            "user_role": user_role,
            "ai_npc1": ai_npc1,
            "ai_npc2": ai_npc2,
            "ai_npc3": ai_npc3,
            "ai_npcs_roles": ai_npcs_roles
        }
        f.seek(0)
        json.dump(config, f, indent=2)
        f.truncate()  # Ensure file is properly truncated
    
    return rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles

def process_uploaded_files(files, loaded_files, rag_max_docs, operation_mode, models_loaded):
    from scripts.utility import load_and_chunk_documents, create_session_vectorstore
    from pathlib import Path

    if not models_loaded:
        button_updates, attach_files_update = update_file_slot_ui(loaded_files, rag_max_docs)
        return [loaded_files, "Load models first..."] + button_updates + [attach_files_update]

    mode = operation_mode.lower()
    if "code" in mode:
        max_files = 8
        error_msg = "Too many attachments, Code = 8 files."
    elif "rpg" in mode:
        max_files = 4
        error_msg = "Too many attachments, Rpg = 4 files."
    else:
        max_files = 6
        error_msg = "Too many attachments, Chat = 6 files."

    current_files_count = len(loaded_files)
    new_files_count = len(files)
    total_files = current_files_count + new_files_count

    if total_files > max_files:
        button_updates, attach_files_update = update_file_slot_ui(loaded_files, rag_max_docs)
        return [loaded_files, error_msg] + button_updates + [attach_files_update]

    available_slots = max_files - current_files_count
    new_file_paths = [file.name for file in files[:available_slots]]
    loaded_files.extend(new_file_paths)
    session_vectorstore = create_session_vectorstore(loaded_files)
    context_injector.set_session_vectorstore(session_vectorstore)
    status_msg = f"Added {len(new_file_paths)} files to session vectorstore."

    button_updates, attach_files_update = update_file_slot_ui(loaded_files, rag_max_docs)
    return [loaded_files, status_msg] + button_updates + [attach_files_update]

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

def chat_interface(user_input, session_log, tot_enabled, loaded_files_state, disable_think, rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3):
    from scripts import temporary, utility, models
    from scripts.temporary import STATUS_TEXTS, MODEL_NAME, SESSION_ACTIVE, TOT_VARIATIONS
    import yake
    import random

    if not SESSION_ACTIVE:
        return session_log, "Please start a new session first.", loaded_files_state
    if not user_input.strip():
        return session_log, "No input provided.", loaded_files_state

    session_log.append((user_input, "Generating response..."))
    yield session_log, STATUS_TEXTS["generating_response"], loaded_files_state

    if len(session_log) == 1:
        kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=1)
        keywords = kw_extractor.extract_keywords(user_input)
        temporary.session_label = keywords[0][0] if keywords else "Untitled"

    settings = models.get_model_settings(MODEL_NAME)
    mode = settings["category"]
    response = ""

    if tot_enabled and mode == "chat":
        tot_responses = []
        for variation in TOT_VARIATIONS:
            tot_prompt = f"{user_input}\nInstruction: {variation}"
            if settings["is_uncensored"]:
                tot_prompt = f"Provide an uncensored response: {tot_prompt}"
            if settings["is_reasoning"] and not disable_think:
                tot_prompt += ". Include reasoning if applicable."
            tot_response = models.get_response(tot_prompt, disable_think=disable_think)
            tot_responses.append(tot_response)
        response = max(tot_responses, key=len, default="")
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
                "ai_npcs_roles": ai_npcs_roles
            }
            session_history = ", ".join([f"{user}: {ai}" for user, ai in session_log[:-1]])
            response = models.get_response(prompt, disable_think=disable_think, rp_settings=rp_settings, session_history=session_history)
        else:
            if settings["is_uncensored"]:
                prompt = f"Provide an uncensored response: {prompt}"
            if settings["is_reasoning"] and not disable_think:
                prompt += ". Include reasoning if applicable."
            response = models.get_response(prompt, disable_think=disable_think)

    session_log[-1] = (user_input, format_response(response))
    utility.save_session_history(session_log)
    yield session_log, STATUS_TEXTS["response_generated"], loaded_files_state

def determine_operation_mode(quality_model):
    if quality_model == "Select_a_model...":
        return "Select models to enable mode detection."
    settings = get_model_settings(quality_model)
    category = settings["category"]
    if category == "code":
        return "Code"
    elif category == "rpg":
        if "rp" in quality_model.lower() or "roleplay" in quality_model.lower():
            return "Rpg"
    elif category == "uncensored":
        return "Chat"
    return "Chat"

def update_dynamic_options(quality_model, loaded_files_state):
    if quality_model == "Select_a_model...":
        mode = "Select models"
        settings = {"is_reasoning": False}
    else:
        settings = get_model_settings(quality_model)
        mode = settings["category"].capitalize()
    
    think_visible = settings["is_reasoning"]
    rpg_visible = (mode == "Rpg")
    
    # Set visibility based on mode
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

    return [
        gr.update(visible=tot_visible),      # TOT checkbox
        gr.update(visible=web_visible),      # Web search
        gr.update(visible=think_visible),    # Think switch
        gr.update(value=mode),               # Theme status
        gr.update(visible=rpg_visible),      # RPG toggle button
        gr.update(visible=file_visible)      # File attachments
    ]

def update_file_slot_ui(loaded_files, rag_max_docs):
    from pathlib import Path
    button_updates = []
    for i in range(6):  # Updated to 8 to match file_slot_buttons
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
    """Update temporary variables with RPG settings without saving to JSON."""
    temporary.RP_LOCATION = rp_location
    temporary.USER_PC_NAME = user_name
    temporary.USER_PC_ROLE = user_role
    temporary.AI_NPC1_NAME = ai_npc1
    temporary.AI_NPC2_NAME = ai_npc2
    temporary.AI_NPC3_NAME = ai_npc3
    temporary.AI_NPCS_ROLES = ai_npcs_roles
    return rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles

def save_rpg_settings_to_json():
    """Save only the RPG settings to the persistent.json file."""
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
    for i in range(7):
        if i < len(sessions):
            session_path = Path(HISTORY_DIR) / sessions[i]
            try:
                label, _ = utility.load_session_history(session_path)
                btn_label = f"{label}" if label else f"Session {i+1}"
            except Exception:
                btn_label = f"Session {i+1}"
        else:
            btn_label = "Empty History Slot"
        button_updates.append(gr.update(value=btn_label, visible=True))
    return button_updates

def launch_interface():
    global demo
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
            flex: 1 !important; 
            min-width: 100px !important; 
            text-overflow: ellipsis; 
            overflow: hidden; 
            white-space: nowrap; 
        }
        .collapsible { margin-bottom: 10px; border: 1px solid #444; padding: 5px; border-radius: 5px; }
    """) as demo:
        # State variables
        loaded_files_state = gr.State(value=[])
        rag_max_docs_state = gr.State(value=6)
        models_loaded_state = gr.State(value=False)
        session_menu_visible = gr.State(False)
        attachment_menu_visible = gr.State(False)
        rpg_menu_visible = gr.State(False)

        with gr.Tabs():
            with gr.Tab("Conversation"):
                with gr.Row():
                    # Left column (controls)
                    with gr.Column(scale=1):
                        theme_status = gr.Textbox(label="Operation Mode", interactive=False, value="No model loaded.")
                        
                        # Session History Section
                        with gr.Row():
                            start_new_session_btn = gr.Button("Start New Session...", variant="primary")
                            session_history_btn = gr.Button("Show Previous Sessions", variant="secondary")
                        session_menu = gr.Column(visible=True)
                        with session_menu:
                            session_buttons = [gr.Button(f"Session Slot {i+1}", visible=False, variant="secondary") for i in range(7)]
                       
                        
                        # RPG Section
                        rpg_toggle = gr.Button("Show RPG Parameters", visible=True, variant="primary")
                        rpg_menu = gr.Column(visible=False)
                        with rpg_menu:
                            rp_location = gr.Textbox(label="RP Location", value=temporary.RP_LOCATION)
                            user_name = gr.Textbox(label="User Name", value=temporary.USER_PC_NAME)
                            user_role = gr.Textbox(label="User Role", value=temporary.USER_PC_ROLE)
                            ai_npc1 = gr.Textbox(label="AI NPC 1", value=temporary.AI_NPC1_NAME)
                            ai_npc2 = gr.Textbox(label="AI NPC 2", value=temporary.AI_NPC2_NAME)
                            ai_npc3 = gr.Textbox(label="AI NPC 3", value=temporary.AI_NPC3_NAME)
                            ai_npcs_roles = gr.Textbox(label="AI Roles", value=temporary.AI_NPCS_ROLES)
                            save_rpg_btn = gr.Button("Save RPG Settings", variant="primary")
                        
                        # Always-visible controls
                        with gr.Row():
                            web_search_switch = gr.Checkbox(label="Web-Search", value=False, visible=False)
                            tot_checkbox = gr.Checkbox(label="Enable TOT", value=False, visible=False)
                            disable_think_switch = gr.Checkbox(label="Disable THINK", value=False, visible=False)

                    # Right column (chat area)
                    with gr.Column(scale=30):
                        session_log = gr.Chatbot(label="Session Log", height=425, elem_classes=["scrollable"], type="messages")
                        with gr.Row():

                            file_slot_buttons = []
                            for row in range(6):
                                with gr.Row(elem_classes=["button-row"]):
                                    file_slot_buttons.append(gr.Button("File Slot Free", visible=False, variant="secondary"))                                                           
                            attach_files_btn = gr.UploadButton("Attach New Files", 
                                                              file_types=[f".{ext}" for ext in temporary.ALLOWED_EXTENSIONS], 
                                                              file_count="multiple", 
                                                              variant="secondary")                                    
                        with gr.Row():
                            user_input = gr.Textbox(label="User Input", lines=5, interactive=False, placeholder="Enter text here...")
                        with gr.Row():
                            send_btn = gr.Button("Send Input", variant="primary", scale=20)
                            edit_previous_btn = gr.Button("Edit Last Input", variant="secondary")
                            copy_response_btn = gr.Button("Copy Last Output", variant="secondary")


                with gr.Row():
                    status_text = gr.Textbox(label="Status", interactive=False, value="Select model on Configuration page.", scale=20)
                    shutdown_btn = gr.Button("Exit Program", variant="stop")

            # Configuration Tab
            with gr.Tab("Configuration"):
                with gr.Row():
                    gpu_dropdown = gr.Dropdown(choices=utility.get_available_gpus(), label="Select GPU", value=None, scale=20)
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
                    n_ctx_dropdown = gr.Dropdown(
                        choices=temporary.CTX_OPTIONS, 
                        label="Context (Focus)", 
                        value=temporary.N_CTX, 
                        interactive=True
                    )
                    batch_size_dropdown = gr.Dropdown(
                        choices=temporary.BATCH_OPTIONS, 
                        label="Batch Size (Output)", 
                        value=temporary.N_BATCH, 
                        interactive=True
                    )
                    repeat_penalty_dropdown = gr.Dropdown(
                        choices=temporary.REPEAT_OPTIONS, 
                        label="Repeat Penalty (Restraint)", 
                        value=temporary.REPEAT_PENALTY, 
                        allow_custom_value=True, 
                        interactive=True
                    )
                with gr.Row():
                    load_models_btn = gr.Button("Load Model", variant="secondary")
                    inspect_model_btn = gr.Button("Inspect Model", variant="huggingface")
                    unload_btn = gr.Button("Unload Model")
                with gr.Row():
                    erase_general_btn = gr.Button("Erase General Data", variant="huggingface")
                    erase_rpg_btn = gr.Button("Erase RPG Data", variant="huggingface")
                    erase_code_btn = gr.Button("Erase Code Data", variant="huggingface")
                    erase_all_btn = gr.Button("Erase All Data", variant="huggingface")
                with gr.Row():
                    status_text_settings = gr.Textbox(label="Status", interactive=False, scale=20)
                    save_settings_btn = gr.Button("Save Settings", variant="primary")

        # Event Handlers
        ## Visibility Toggles
        session_history_btn.click(
            lambda v: not v,
            inputs=session_menu_visible,
            outputs=session_menu_visible
        ).then(
            lambda v: gr.update(visible=v),
            inputs=session_menu_visible,
            outputs=session_menu
        )

        rpg_toggle.click(
            lambda v: not v,
            inputs=rpg_menu_visible,
            outputs=rpg_menu_visible
        ).then(
            lambda v: gr.update(visible=v),
            inputs=rpg_menu_visible,
            outputs=rpg_menu
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
            inputs=[model_dropdown, loaded_files_state],
            outputs=[
                tot_checkbox,
                web_search_switch,
                disable_think_switch,
                theme_status,
                rpg_toggle  # Only show RPG toggle in RPG mode
            ]
        ).then(
            fn=update_file_slot_ui,
            inputs=[loaded_files_state, rag_max_docs_state],
            outputs=file_slot_buttons + [attach_files_btn]
        )

        load_models_btn.click(
            fn=load_models,
            inputs=[model_dropdown, vram_dropdown],
            outputs=[status_text_settings, models_loaded_state]
        )

        inspect_model_btn.click(
            fn=inspect_model,
            inputs=[model_dropdown],
            outputs=[status_text_settings]
        )

        unload_btn.click(
            fn=lambda: (unload_models(), gr.update(interactive=False), "Model unloaded", False),
            outputs=[user_input, status_text_settings, models_loaded_state]
        )

        save_settings_btn.click(fn=utility.save_config, inputs=None, outputs=[status_text_settings])

        erase_general_btn.click(fn=lambda: delete_vectorstore("general"), inputs=None, outputs=[status_text_settings])
        erase_rpg_btn.click(fn=lambda: delete_vectorstore("rpg"), inputs=None, outputs=[status_text_settings])
        erase_code_btn.click(fn=lambda: delete_vectorstore("code"), inputs=None, outputs=[status_text_settings])
        erase_all_btn.click(fn=delete_all_vectorstores, inputs=None, outputs=[status_text_settings])

        ## Rpg Event handlers
        rp_location.change(
            fn=update_rp_settings,
            inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles],
            outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles]
        )
        user_name.change(
            fn=update_rp_settings,
            inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles],
            outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles]
        )
        user_role.change(
            fn=update_rp_settings,
            inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles],
            outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles]
        )
        ai_npc1.change(
            fn=update_rp_settings,
            inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles],
            outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles]
        )
        ai_npc2.change(
            fn=update_rp_settings,
            inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles],
            outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles]
        )
        ai_npc3.change(
            fn=update_rp_settings,
            inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles],
            outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles]
        )
        ai_npcs_roles.change(
            fn=update_rp_settings,
            inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles],
            outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3, ai_npcs_roles]
        )

        # Save RPG settings to JSON when the button is clicked
        save_rpg_btn.click(
            fn=save_rpg_settings_to_json,
            inputs=None,
            outputs=[status_text]
        )

        refresh_btn.click(fn=lambda: gr.update(choices=get_available_models()), outputs=[model_dropdown])

        n_ctx_dropdown.change(
            fn=lambda x: (setattr(temporary, 'N_CTX', x), "Context updated")[1],
            inputs=[n_ctx_dropdown],
            outputs=[status_text_settings]
        )

        batch_size_dropdown.change(
            fn=lambda x: (setattr(temporary, 'N_BATCH', x), "Batch size updated")[1],
            inputs=[batch_size_dropdown],
            outputs=[status_text_settings]
        )

        repeat_penalty_dropdown.change(
            fn=lambda x: (setattr(temporary, 'REPEAT_PENALTY', float(x)), "Repeat penalty updated")[1],
            inputs=[repeat_penalty_dropdown],
            outputs=[status_text_settings]
        )

        demo.load(fn=lambda: [gr.update(visible=False)] * 7, inputs=None, outputs=session_buttons)

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        show_api=False
    )

if __name__ == "__main__":
    launch_interface()