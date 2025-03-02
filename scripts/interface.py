# Script: `.\scripts\interface.py`

# Imports...
import gradio as gr
from gradio import themes
import re, os, json, paperclip, yake, random
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
    MODEL_NAME, STATUS_TEXTS, CTX_OPTIONS, RP_LOCATION, USER_NAME, USER_ROLE,
    AI_NPC1, AI_NPC2, AI_NPC3, SESSION_ACTIVE, TOT_VARIATIONS
)
from scripts import utility
from scripts.utility import (
    delete_vectorstore, delete_all_vectorstores, web_search, get_saved_sessions,
    load_session_history, save_session_history, load_and_chunk_documents,
    create_vectorstore, get_available_gpus
)
from scripts.models import (
    get_response, get_available_models, unload_models, get_model_settings,
    context_injector, inspect_model
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

def save_rp_settings(rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3):
    config_path = Path("data/persistent.json")
    with open(config_path, "r+") as f:
        config = json.load(f)
        config["rp_settings"] = {
            "rp_location": rp_location,
            "user_name": user_name,
            "user_role": user_role,
            "ai_npc1": ai_npc1,
            "ai_npc2": ai_npc2,
            "ai_npc3": ai_npc3
        }
        f.seek(0)
        json.dump(config, f, indent=2)
    return rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3

def process_uploaded_files(files, loaded_files, rag_max_docs, operation_mode, models_loaded):
    from scripts.utility import map_operation_mode_to_vectorstore_mode, load_and_chunk_documents, create_session_vectorstore
    from pathlib import Path

    # Check if model is loaded
    if not models_loaded:
        button_updates, attach_files_update = update_file_slot_ui(loaded_files, rag_max_docs)
        return [loaded_files, "Load models first..."] + button_updates + [attach_files_update]

    # Map operation mode to category and determine max files
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

    # Calculate total files after adding new ones
    current_files_count = len(loaded_files)
    new_files_count = len(files)
    total_files = current_files_count + new_files_count

    # Check if total exceeds mode-specific limit
    if total_files > max_files:
        button_updates, attach_files_update = update_file_slot_ui(loaded_files, rag_max_docs)
        return [loaded_files, error_msg] + button_updates + [attach_files_update]

    # Process files if within limit
    available_slots = max_files - current_files_count
    new_file_paths = [file.name for file in files[:available_slots]]  # Full paths, limited to available slots
    loaded_files.extend(new_file_paths)
    session_vectorstore = create_session_vectorstore(loaded_files)
    context_injector.set_session_vectorstore(session_vectorstore)
    status_msg = f"Added {len(new_file_paths)} files to session vectorstore."

    # Update UI
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

def update_file_slot_ui(loaded_files, rag_max_docs):
    from pathlib import Path
    button_updates = []
    for i in range(8):
        if i < len(loaded_files):  # Show button only if a file is attached at this index
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

def handle_slot_click(slot_index, loaded_files, rag_max_docs):
    if slot_index < len(loaded_files):
        removed_file = loaded_files.pop(slot_index)
        try:
            Path(removed_file).unlink()
        except Exception as e:
            print(f"Error unlinking file: {e}")
        docs = load_and_chunk_documents([Path(f) for f in loaded_files])
        create_vectorstore(docs, "chat")  # Default mode, adjust as needed
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

def update_session_buttons():
    sessions = utility.get_saved_sessions()
    button_updates = []
    for i in range(11):
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

    # Session label generation (only set once, no intermediate message)
    if len(session_log) == 1:
        kw_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.9, top=1)
        keywords = kw_extractor.extract_keywords(user_input)
        temporary.session_label = keywords[0][0] if keywords else "Untitled"

    # Main response generation
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
                "ai_npc3": ai_npc3
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
    elif category == "rpg":  # Changed from "nsfw"
        if "rp" in quality_model.lower() or "roleplay" in quality_model.lower():
            return "Chat-Rp"  # For roleplay-specific models
        else:
            return "Chat-Nsfw"  # For general mature content
    elif category == "uncensored":
        return "Chat-Uncensored"
    return "Chat"

def update_dynamic_options(quality_model, loaded_files_state):
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
        rp_visible = False
        rag_max_docs = 8
    elif mode == "Rpg":
        tot_visible = False
        web_visible = False
        file_visible = True  # Enable file slots for RPG mode
        rp_visible = True
        rag_max_docs = 4
    elif mode == "Chat":
        tot_visible = True
        web_visible = True
        file_visible = True
        rp_visible = False
        rag_max_docs = 6
    else:
        tot_visible = False
        web_visible = False
        file_visible = True
        rp_visible = False
        rag_max_docs = 4
    updates = [
        gr.update(visible=tot_visible),
        gr.update(visible=web_visible),
        gr.update(visible=think_visible),
        gr.update(value=mode)
    ]
    rp_updates = [gr.update(visible=rp_visible) for _ in range(8)]  # Matches 8 RPG components
    return [rag_max_docs] + updates + rp_updates

def toggle_model_rows(quality_model):
    visible = quality_model != "Select_a_model..."
    return [gr.update(visible=visible)] * 2

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
    """) as demo:
        loaded_files_state = gr.State(value=[])
        rag_max_docs_state = gr.State(value=4)
        models_loaded_state = gr.State(value=False)

        with gr.Tabs():
            with gr.Tab("Conversation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        start_new_session_btn = gr.Button("Start New Session", variant="primary")
                        session_buttons = [gr.Button("Empty History Slot", visible=False, variant="primary") for _ in range(11)]
                    with gr.Column(scale=20):
                        session_log = gr.Chatbot(label="Session Log", height=425, elem_classes=["scrollable"])
                        user_input = gr.Textbox(label="User Input", lines=5, interactive=False, placeholder="Enter text here...")
                        with gr.Row():
                            send_btn = gr.Button("Send Input", variant="primary")
                            edit_previous_btn = gr.Button("Edit Last Input", variant="secondary")
                            copy_response_btn = gr.Button("Copy Last Output", variant="secondary")
                    with gr.Column(scale=1):
                        theme_status = gr.Textbox(label="Operation Mode", interactive=False, value="No model loaded.")
                        web_search_switch = gr.Checkbox(label="Web-Search", value=False, visible=False)
                        tot_checkbox = gr.Checkbox(label="Enable TOT", value=False, visible=False)
                        disable_think_switch = gr.Checkbox(label="Disable THINK", value=False, visible=False)
                        rp_location = gr.Textbox(label="RP Location", value=temporary.RP_LOCATION, visible=False)
                        user_name = gr.Textbox(label="User Name", value=temporary.USER_NAME, visible=False)
                        user_role = gr.Textbox(label="User Role", value=temporary.USER_ROLE, visible=False)
                        ai_npc1 = gr.Textbox(label="AI NPC 1", value=temporary.AI_NPC1, visible=False)
                        ai_npc2 = gr.Textbox(label="AI NPC 2", value=temporary.AI_NPC2, visible=False)
                        delete_npc2_btn = gr.Button("Delete NPC 2", visible=False, variant="secondary")
                        ai_npc3 = gr.Textbox(label="AI NPC 3", value=temporary.AI_NPC3, visible=False)
                        delete_npc3_btn = gr.Button("Delete NPC 3", visible=False, variant="secondary")
                        file_slot_buttons = []
                        attach_files_btn = gr.UploadButton("Attach Files", file_types=[f".{ext}" for ext in temporary.ALLOWED_EXTENSIONS], file_count="multiple", variant="primary")
                        for row in range(4):
                            with gr.Row(elem_classes=["button-row"]):
                                for col in range(2):
                                    file_slot_buttons.append(gr.Button("File Slot Free", visible=False, variant="secondary"))
                with gr.Row():
                    status_text = gr.Textbox(label="Status", interactive=False, value="Select model on Configuration page.", scale=20)
                    shutdown_btn = gr.Button("Exit Program", variant="stop")

            with gr.Tab("Configuration"):
                with gr.Row():
                    gpu_dropdown = gr.Dropdown(choices=utility.get_available_gpus(), label="Select GPU", value=None, scale=20)
                with gr.Row():                
                    vram_dropdown = gr.Dropdown(choices=temporary.VRAM_OPTIONS, label="Assign Free VRam", value=temporary.VRAM_SIZE)
                    mlock_checkbox = gr.Checkbox(label="MLock Enabled", value=temporary.MLOCK)                
                
                
                with gr.Row():
                    model_dir_text = gr.Textbox(label="Model Folder", value=temporary.MODEL_FOLDER, scale=20)
                    browse_btn = gr.Button("Browse", variant="primary")
                    
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
                    n_ctx_dropdown = gr.Dropdown(choices=temporary.CTX_OPTIONS, label="Context (Focus)", value=temporary.N_CTX)
                    batch_size_dropdown = gr.Dropdown(choices=temporary.BATCH_OPTIONS, label="Batch Size (Output)", value=temporary.N_BATCH)
                    repeat_penalty_dropdown = gr.Dropdown(choices=temporary.REPEAT_OPTIONS, label="Repeat Penalty (Restraint)", value=temporary.REPEAT_PENALTY, allow_custom_value=True)

                with gr.Row():
                    load_models_btn = gr.Button("Load Model", variant="primary")
                    inspect_model_btn = gr.Button("Inspect Model")
                    unload_btn = gr.Button("Unload Model")
                with gr.Row():
                    erase_general_btn = gr.Button("Erase General Data", variant="secondary")
                    erase_rpg_btn = gr.Button("Erase RPG Data", variant="secondary")
                    erase_code_btn = gr.Button("Erase Code Data", variant="secondary")
                    erase_all_btn = gr.Button("Erase All Data", variant="secondary")
                with gr.Row():
                    status_text_settings = gr.Textbox(label="Status", interactive=False, scale=20)
                    save_settings_btn = gr.Button("Save Settings", variant="primary")

        # Event Handlers
        def handle_start_session_click(models_loaded):
            from scripts import temporary
            if not models_loaded:
                return "Load model first on Configuration page...", models_loaded, [], "", gr.update(interactive=False), []
            else:
                temporary.current_session_id = None
                temporary.session_label = ""
                temporary.SESSION_ACTIVE = True
                temporary.context_injector.set_session_vectorstore(None)
                return "Type input and click Send to begin...", models_loaded, [], "", gr.update(interactive=True), []

        def load_models(model_name, vram_size):
            from scripts import temporary, models
            if model_name == "Select_a_model...":
                return "Select a model to load.", False
            gpu_layers = models.calculate_gpu_layers([model_name], vram_size)
            temporary.N_GPU_LAYERS = gpu_layers.get(model_name, 0)
            model_path = Path(temporary.MODEL_FOLDER) / model_name
            temporary.llm = temporary.Llama(
                model_path=str(model_path),
                n_ctx=temporary.N_CTX,
                n_gpu_layers=temporary.N_GPU_LAYERS,
                n_batch=temporary.N_BATCH,
                mmap=temporary.MMAP,
                mlock=temporary.MLOCK,
                verbose=False
            )
            temporary.MODELS_LOADED = True
            temporary.MODEL_NAME = model_name
            status = f"Model loaded, layer distribution: VRAM={temporary.N_GPU_LAYERS}"
            return status, True

        edit_previous_btn.click(
            fn=lambda h, i: (h[:-2], h[-2][0]) if len(h) >= 2 else ([], h[0][0]) if len(h) == 1 else (h, ""),
            inputs=[session_log, user_input],
            outputs=[session_log, user_input]
        )
        send_btn.click(
            fn=chat_interface,
            inputs=[user_input, session_log, tot_checkbox, loaded_files_state, disable_think_switch, rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3],
            outputs=[session_log, status_text, loaded_files_state]
        ).then(
            fn=update_session_buttons, inputs=None, outputs=session_buttons
        )
        attach_files_btn.upload(
            fn=process_uploaded_files,
            inputs=[attach_files_btn, loaded_files_state, rag_max_docs_state, theme_status, models_loaded_state],
            outputs=[loaded_files_state, status_text] + file_slot_buttons + [attach_files_btn]
        )
        start_new_session_btn.click(
            fn=handle_start_session_click,
            inputs=[models_loaded_state],
            outputs=[status_text, models_loaded_state, session_log, user_input, user_input]
        ).then(
            fn=update_session_buttons, inputs=None, outputs=session_buttons
        )
        shutdown_btn.click(
            fn=lambda: (temporary.unload_models(), demo.close(), os._exit(0), "Program terminated")[3],
            inputs=None,
            outputs=[status_text]
        )
        for i, btn in enumerate(file_slot_buttons):
            btn.click(
                fn=lambda s, r, idx=i: temporary.handle_slot_click(idx, s, r),
                inputs=[loaded_files_state, rag_max_docs_state],
                outputs=[loaded_files_state, status_text] + file_slot_buttons + [attach_files_btn]
            )
        for i, btn in enumerate(session_buttons):
            btn.click(
                fn=lambda idx=i: temporary.load_session_by_index(idx),
                inputs=[],
                outputs=[session_log, status_text]
            ).then(
                fn=update_session_buttons, inputs=None, outputs=session_buttons
            )
        model_dropdown.change(
            fn=determine_operation_mode,
            inputs=[model_dropdown],
            outputs=[theme_status]
        ).then(
            fn=lambda mode: temporary.context_injector.set_mode(mode.lower()),
            inputs=[theme_status],
            outputs=None
        ).then(
            fn=update_dynamic_options,
            inputs=[model_dropdown, loaded_files_state],
            outputs=[
                rag_max_docs_state,
                tot_checkbox,
                web_search_switch,
                disable_think_switch,
                theme_status,
                rp_location,
                user_name,
                user_role,
                ai_npc1,
                ai_npc2,
                delete_npc2_btn,
                ai_npc3,
                delete_npc3_btn
            ]
        ).then(
            fn=update_file_slot_ui,
            inputs=[loaded_files_state, rag_max_docs_state],
            outputs=file_slot_buttons + [attach_files_btn]
        )
        erase_general_btn.click(fn=lambda: delete_vectorstore("general"), inputs=None, outputs=[status_text_settings])
        load_models_btn.click(
            fn=load_models,
            inputs=[model_dropdown, vram_dropdown],
            outputs=[status_text_settings, models_loaded_state]
        )
        erase_rpg_btn.click(fn=lambda: delete_vectorstore("rpg"), inputs=None, outputs=[status_text_settings])
        erase_code_btn.click(fn=lambda: delete_vectorstore("code"), inputs=None, outputs=[status_text_settings])
        erase_all_btn.click(fn=delete_all_vectorstores, inputs=None, outputs=[status_text_settings])
        rp_location.change(fn=save_rp_settings, inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3], outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3])
        user_name.change(fn=save_rp_settings, inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3], outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3])
        user_role.change(fn=save_rp_settings, inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3], outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3])
        ai_npc1.change(fn=save_rp_settings, inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3], outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3])
        ai_npc2.change(fn=save_rp_settings, inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3], outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3])
        ai_npc3.change(fn=save_rp_settings, inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3], outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3])
        delete_npc2_btn.click(fn=lambda x, y, z: temporary.delete_npc(1, x, y, z), inputs=[ai_npc1, ai_npc2, ai_npc3], outputs=[ai_npc1, ai_npc2, ai_npc3])
        delete_npc3_btn.click(fn=lambda x, y, z: temporary.delete_npc(2, x, y, z), inputs=[ai_npc1, ai_npc2, ai_npc3], outputs=[ai_npc1, ai_npc2, ai_npc3])
        refresh_btn.click(fn=lambda: gr.update(choices=temporary.get_available_models()), outputs=[model_dropdown])
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
        demo.load(fn=lambda: [gr.update(visible=False)] * 11, inputs=None, outputs=session_buttons)

    demo.launch()

if __name__ == "__main__":
    launch_interface()