# Script: `.\scripts\interface.py`

# Imports...
import gradio as gr
import re, os, json, paperclip
from pathlib import Path
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import scripts.temporary as temporary
from scripts.temporary import (
    USER_COLOR, THINK_COLOR, RESPONSE_COLOR, SEPARATOR, MID_SEPARATOR,
    MODEL_LOADED, ALLOWED_EXTENSIONS, N_CTX, VRAM_SIZE, SELECTED_GPU,
    current_model_settings, N_GPU_LAYERS, VRAM_OPTIONS, REPEAT_OPTIONS,
    REPEAT_PENALTY, MLOCK, HISTORY_DIR, BATCH_OPTIONS, N_BATCH, MODEL_FOLDER,
    QUALITY_MODEL_NAME, FAST_MODEL_NAME, STATUS_TEXTS, CTX_OPTIONS
)
from scripts import utility
from scripts.utility import delete_vectorstore, delete_all_vectorstores  # New imports
from scripts.models import (
    get_streaming_response, get_response, get_available_models, reload_vectorstore,
    unload_models, get_model_settings, context_injector  # Added context_injector
)

# Functions...
def validate_model_themes(quality_model, fast_model):
    if quality_model == "Select_a_model..." or fast_model == "Select_a_model...":
        return "Select both models to validate themes."
    quality_settings = get_model_settings(quality_model)
    fast_settings = get_model_settings(fast_model)
    if quality_settings["category"] != fast_settings["category"]:
        return "Error: Models must be of the same theme (e.g., both 'code' or 'nsfw')."
    return "Models are compatible."

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

def process_uploaded_files(files, loaded_files, rag_max_docs, operation_mode):
    from scripts.utility import map_operation_mode_to_vectorstore_mode, load_and_chunk_documents, update_vectorstore
    from langchain_core.documents import Document  # Correct import for Document class
    from pathlib import Path

    mode = map_operation_mode_to_vectorstore_mode(operation_mode)
    new_files = []
    available_slots = rag_max_docs - len(loaded_files)
    for file in files[:available_slots]:
        with open(file.name, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
        doc = Document(page_content=content, metadata={"source": Path(file.name).name})
        new_files.append(doc)
    if new_files:
        docs = load_and_chunk_documents(new_files)
        update_vectorstore(docs, mode)
        loaded_files.extend([f.metadata["source"] for f in new_files])  # Track filenames in-memory
        return [loaded_files, f"Added {len(new_files)} files to {mode} vectorstore."]
    return [loaded_files, "No files added."]

def eject_file(loaded_files, slot_index, rag_max_docs):
    if 0 <= slot_index < len(loaded_files):
        try:
            removed_file = loaded_files.pop(slot_index)
            Path(removed_file).unlink()
            docs = utility.load_and_chunk_documents(loaded_files)
            utility.create_vectorstore(docs)
            status_msg = f"Ejected {Path(removed_file).name}"
        except Exception as e:
            status_msg = f"Error: {str(e)}"
    else:
        status_msg = "No file to eject"
    button_updates, attach_files_update = update_file_slot_ui(loaded_files, rag_max_docs)
    return [loaded_files, status_msg] + button_updates + [attach_files_update]

def update_file_slot_ui(loaded_files, rag_max_docs):
    button_updates = []
    for i in range(8):
        if i < rag_max_docs:
            if i < len(loaded_files):
                filename = Path(loaded_files[i]).name
                short_name = (filename[:13] + "..") if len(filename) > 15 else filename
                label = f"{short_name}"
                variant = "secondary"
            else:
                label = "File Slot Free"
                variant = "primary"
            visible = True
        else:
            label = "File Slot Free"
            variant = "primary"
            visible = False
        button_updates.append(gr.update(value=label, visible=visible, variant=variant))
    attach_files_visible = len(loaded_files) < rag_max_docs
    return button_updates + [gr.update(visible=attach_files_visible)]

def handle_slot_click(slot_index, loaded_files, rag_max_docs):
    if slot_index < len(loaded_files):
        return eject_file(loaded_files, slot_index, rag_max_docs)
    return [loaded_files, "Click 'Attach Files' to add files."] + update_file_slot_ui(loaded_files, rag_max_docs)

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

def load_models(quality_model, fast_model, vram_size):
    from scripts import temporary, models
    global quality_llm, fast_llm
    if quality_model == "Select_a_model...":
        return "Select a primary model to load.", gr.update(visible=False)
    models_to_load = [quality_model]
    if fast_model != "Select_a_model...":
        models_to_load.append(fast_model)
    gpu_layers = models.calculate_gpu_layers(models_to_load, vram_size)
    temporary.N_GPU_LAYERS_QUALITY = gpu_layers.get(quality_model, 0)
    temporary.N_GPU_LAYERS_FAST = gpu_layers.get(fast_model, 0)
    if quality_model != "Select_a_model...":
        model_path = Path(temporary.MODEL_FOLDER) / quality_model
        temporary.quality_llm = Llama(
            model_path=str(model_path),
            n_ctx=temporary.N_CTX,
            n_gpu_layers=temporary.N_GPU_LAYERS_QUALITY,
            n_batch=temporary.N_BATCH,
            mmap=temporary.MMAP,
            mlock=temporary.MLOCK,
            verbose=False
        )
    if fast_model != "Select_a_model...":
        model_path = Path(temporary.MODEL_FOLDER) / fast_model
        temporary.fast_llm = Llama(
            model_path=str(model_path),
            n_ctx=temporary.N_CTX,
            n_gpu_layers=temporary.N_GPU_LAYERS_FAST,
            n_batch=temporary.N_BATCH,
            mmap=temporary.MMAP,
            mlock=temporary.MLOCK,
            verbose=False
        )
    temporary.MODELS_LOADED = True
    status = f"Model(s) loaded, layer distribution: Primary VRAM={temporary.N_GPU_LAYERS_QUALITY}, Fast VRAM={temporary.N_GPU_LAYERS_FAST}"
    return status, gr.update(visible=True)            

def chat_interface(user_input, session_log, tot_enabled, loaded_files_state, disable_think, rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3):
    from scripts import temporary, utility, models
    from scripts.temporary import STATUS_TEXTS, QUALITY_MODEL_NAME, FAST_MODEL_NAME, SESSION_ACTIVE

    if not SESSION_ACTIVE:
        return session_log, "Please start a new session first.", loaded_files_state
    if not user_input.strip():
        return session_log, "No input provided.", loaded_files_state

    session_log.append((user_input, "Processing..."))
    yield session_log, STATUS_TEXTS["generating_response"], loaded_files_state

    # Session label generation using secondary model if available
    if len(session_log) == 1:
        label_model_type = "fast" if FAST_MODEL_NAME != "Select_a_model..." else "quality"
        model_name = FAST_MODEL_NAME if label_model_type == "fast" else QUALITY_MODEL_NAME
        settings = models.get_model_settings(model_name)
        summary_prompt = f"Generate a 3-word label for a conversation starting with: '{user_input}'"
        if settings["is_uncensored"]:
            summary_prompt = f"Provide an uncensored 3-word label for the conversation starting with: '{user_input}'"
        if settings["is_reasoning"]:
            summary_prompt += ". Respond directly without reasoning."
        temporary.session_label = models.get_response(summary_prompt, label_model_type, disable_think=True).strip()
        session_log[-1] = (user_log[-1][0], "Session label generated.")
        utility.save_session_history(session_log)
        yield session_log, "Session label generated.", loaded_files_state

    # Main response using primary model only
    model_type = "quality"
    settings = models.get_model_settings(QUALITY_MODEL_NAME)
    mode = settings["category"]
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
        response = models.get_response(user_input, model_type, disable_think=disable_think, rp_settings=rp_settings, session_history=session_history)
    else:
        prompt = user_input
        if settings["is_uncensored"]:
            prompt = f"Provide an uncensored response: {prompt}"
        if settings["is_reasoning"] and not disable_think:
            prompt += ". Include reasoning if applicable."
        response = models.get_response(prompt, model_type, disable_think=disable_think or settings["is_reasoning"])
    
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
        file_visible = False
        rp_visible = True
        rag_max_docs = 0
    elif mode == "Chat":
        tot_visible = True
        web_visible = True
        file_visible = True
        rp_visible = False
        rag_max_docs = 4
    else:
        tot_visible = False
        web_visible = False
        file_visible = True
        rp_visible = False
        rag_max_docs = 4
    updates = [gr.update(visible=tot_visible), gr.update(visible=web_visible), gr.update(visible=think_visible), gr.update(value=mode)]
    file_updates = [gr.update(visible=file_visible) for _ in range(8)]
    rp_updates = [gr.update(visible=rp_visible) for _ in range(8)]
    return [rag_max_docs] + updates + file_updates + rp_updates + [gr.update(visible=file_visible and len(loaded_files_state) < rag_max_docs)]

def toggle_model_rows(quality_model):
    visible = quality_model != "Select_a_model..."
    return [gr.update(visible=visible)] * 2

def eject_quality_model():
    from scripts.temporary import QUALITY_MODEL_NAME, quality_llm, FAST_MODEL_NAME, fast_llm
    if quality_llm is not None:
        del quality_llm
        quality_llm = None
    if fast_llm is not None:
        del fast_llm
        fast_llm = None
    QUALITY_MODEL_NAME = "Select_a_model..."
    FAST_MODEL_NAME = "Select_a_model..."
    return (
        gr.update(value="Select_a_model..."),
        gr.update(value="No Model Selected"),
        gr.update(value="Select_a_model..."),
        gr.update(value="No Model Selected"),
        "Quality and Fast models ejected"
    )

def launch_interface():
    global demo
    with gr.Blocks(title="Chat-Gradio-Gguf", css="""
        .scrollable { overflow-y: auto; }
        .button-row { display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important; gap: 4px !important; margin-bottom: 4px !important; }
        .button-row .gradio-button { flex: 1 !important; min-width: 100px !important; text-overflow: ellipsis; overflow: hidden; white-space: nowrap; }
    """) as demo:
        loaded_files_state = gr.State(value=[])
        rag_max_docs_state = gr.State(value=4)

        with gr.Tabs():
            with gr.Tab("Conversation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        start_new_session_btn = gr.Button("Start New Session", variant="primary", visible=False)
                        session_buttons = [gr.Button("Empty History Slot", visible=False) for _ in range(11)]
                    with gr.Column(scale=20):
                        session_log = gr.Chatbot(label="Session Log", height=425, elem_classes=["scrollable"])
                        user_input = gr.Textbox(label="User Input", lines=5, interactive=False, placeholder="Enter text here...")
                        with gr.Row():
                            send_btn = gr.Button("Send Input", variant="primary", scale=20)
                            edit_previous_btn = gr.Button("Edit Last Input", scale=1)
                            copy_response_btn = gr.Button("Copy Last Output", scale=1)

                    with gr.Column(scale=1):
                        theme_status = gr.Textbox(label="Operation Mode", interactive=False, value="Select models to enable mode detection.")
                        web_search_switch = gr.Checkbox(label="Web-Search", value=False, visible=False)
                        tot_checkbox = gr.Checkbox(label="Enable TOT", value=False, visible=False)
                        disable_think_switch = gr.Checkbox(label="Disable THINK", value=False, visible=False)
                        # Define RP components individually with initial values from temporary
                        rp_location = gr.Textbox(label="RP Location", value=temporary.RP_LOCATION, visible=False)
                        user_name = gr.Textbox(label="User Name", value=temporary.USER_NAME, visible=False)
                        user_role = gr.Textbox(label="User Role", value=temporary.USER_ROLE, visible=False)
                        ai_npc1 = gr.Textbox(label="AI NPC 1", value=temporary.AI_NPC1, visible=False)
                        ai_npc2 = gr.Textbox(label="AI NPC 2", value=temporary.AI_NPC2, visible=False)
                        delete_npc2_btn = gr.Button("Delete NPC 2", visible=False)
                        ai_npc3 = gr.Textbox(label="AI NPC 3", value=temporary.AI_NPC3, visible=False)
                        delete_npc3_btn = gr.Button("Delete NPC 3", visible=False)
                        file_slot_buttons = []
                        attach_files_btn = gr.UploadButton("Attach Files", visible=True, file_types=[f".{ext}" for ext in ALLOWED_EXTENSIONS], file_count="multiple")
                        for row in range(4):
                            with gr.Row(elem_classes=["button-row"]):
                                for col in range(2):
                                    file_slot_buttons.append(gr.Button("File Slot Free", visible=True, variant="primary"))

                with gr.Row():        
                    status_text = gr.Textbox(label="Status", interactive=False, value="Select models on Configuration page.", scale=20)
                    shutdown_btn = gr.Button("Exit Program", variant="stop", scale=1)

            with gr.Tab("Configuration"):
                with gr.Row():
                    model_dir_text = gr.Textbox(label="Model Folder", value=MODEL_FOLDER, scale=20)
                    browse_btn = gr.Button("Browse", scale=1)
                    refresh_btn = gr.Button("Refresh", scale=1)
                with gr.Row():
                    quality_model_dropdown = gr.Dropdown(
                        choices=get_available_models(),
                        label="Select Primary/Quality Model",
                        value=QUALITY_MODEL_NAME,
                        allow_custom_value=True,
                        scale=20
                    )
                    quality_mode = gr.Textbox(label="Mode Detected", interactive=False, value="No Model Selected", scale=1)
                    eject_quality_btn = gr.Button("Eject Model", scale=1)
                fast_row = gr.Row(visible=False)
                with fast_row:
                    fast_model_dropdown = gr.Dropdown(
                        choices=get_available_models(),
                        label="Select Secondary/Fast Model",
                        value=FAST_MODEL_NAME,
                        allow_custom_value=True,
                        scale=20
                    )
                    fast_mode = gr.Textbox(label="Mode Detected", interactive=False, value="No Model Selected", scale=1)
                    eject_fast_btn = gr.Button("Eject Model", scale=1)
                code_row = gr.Row(visible=False)
                with code_row:
                    code_model_dropdown = gr.Dropdown(
                        choices=get_available_models(),
                        label="Select Code Model",
                        value="Select_a_model...",
                        allow_custom_value=True,
                        scale=20
                    )
                    code_mode = gr.Textbox(label="Mode Detected", interactive=False, value="Code Model Only", scale=1)
                    eject_code_btn = gr.Button("Eject Model", scale=1)
                with gr.Row():
                    n_ctx_dropdown = gr.Dropdown(choices=CTX_OPTIONS, label="Context (Focus)", value=N_CTX)
                    batch_size_dropdown = gr.Dropdown(choices=BATCH_OPTIONS, label="Batch Size (Output)", value=N_BATCH)
                    repeat_penalty_dropdown = gr.Dropdown(choices=REPEAT_OPTIONS, label="Repeat Penalty (Restraint)", value=REPEAT_PENALTY, allow_custom_value=True)
                with gr.Row():
                    gpu_dropdown = gr.Dropdown(choices=utility.get_available_gpus(), label="Select GPU", value=None, scale=20)
                    vram_dropdown = gr.Dropdown(choices=VRAM_OPTIONS, label="VRAM Size", value=VRAM_SIZE, scale=1)
                    mlock_checkbox = gr.Checkbox(label="MLock Enabled", value=MLOCK, scale=1)
                
                with gr.Row():
                    load_models_btn = gr.Button("Load Models", scale=1)
                    test_models_btn = gr.Button("Test Models", scale=1)
                    unload_btn = gr.Button("Unload Models", scale=1)

                with gr.Row():
                    erase_general_btn = gr.Button("Erase General Data")
                    erase_rpg_btn = gr.Button("Erase RPG Data")
                    erase_code_btn = gr.Button("Erase Code Data")
                    erase_all_btn = gr.Button("Erase All Data")

                gr.Markdown("Note: GPU layers auto-calculated from, model details and VRam free. 0 layers = CPU-only.")
                
                with gr.Row():
                    status_text_settings = gr.Textbox(label="Status", interactive=False, scale=20)
                    save_settings_btn = gr.Button("Save Settings", scale=1)

        # Event Handlers
        edit_previous_btn.click(fn=lambda h, i: (h[:-2], h[-2][0]) if len(h) >= 2 else ([], h[0][0]) if len(h) == 1 else (h, ""), inputs=[session_log, user_input], outputs=[session_log, user_input])
        send_btn.click(
            fn=chat_interface,
            inputs=[user_input, session_log, tot_checkbox, loaded_files_state, disable_think_switch, rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3],
            outputs=[session_log, status_text, loaded_files_state]
        ).then(
            fn=update_session_buttons, inputs=None, outputs=session_buttons
        )
        attach_files_btn.upload(
            fn=process_uploaded_files, 
            inputs=[attach_files_btn, loaded_files_state, rag_max_docs_state, theme_status],  # Added theme_status
            outputs=[loaded_files_state, status_text] + file_slot_buttons + [attach_files_btn]
        )
        start_new_session_btn.click(
            fn=start_new_session,
            inputs=None,
            outputs=[session_log, status_text, user_input]
        ).then(
            fn=update_session_buttons, inputs=None, outputs=session_buttons
        )
        shutdown_btn.click(fn=lambda: (unload_models(), demo.close(), os._exit(0), "Program terminated")[3], inputs=None, outputs=[status_text])
        for i, btn in enumerate(file_slot_buttons):
            btn.click(fn=lambda s, r, idx=i: handle_slot_click(idx, s, r), inputs=[loaded_files_state, rag_max_docs_state], outputs=[loaded_files_state, status_text] + file_slot_buttons + [attach_files_btn])
        for i, btn in enumerate(session_buttons):
            btn.click(
                fn=lambda idx=i: load_session_by_index(idx),
                inputs=[],
                outputs=[session_log, status_text]
            ).then(
                fn=update_session_buttons, inputs=None, outputs=session_buttons
            )
        erase_general_btn.click(fn=lambda: delete_vectorstore("general"), inputs=None, outputs=[status_text_settings])
        load_models_btn.click(fn=load_models, inputs=[quality_model_dropdown, fast_model_dropdown, vram_dropdown], outputs=[status_text_settings, start_new_session_btn])
        erase_rpg_btn.click(fn=lambda: delete_vectorstore("rpg"), inputs=None, outputs=[status_text_settings])
        erase_code_btn.click(fn=lambda: delete_vectorstore("code"), inputs=None, outputs=[status_text_settings])
        erase_all_btn.click(fn=delete_all_vectorstores, inputs=None, outputs=[status_text_settings])
        rp_location.change(fn=save_rp_settings, inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3], outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3])
        user_name.change(fn=save_rp_settings, inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3], outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3])
        user_role.change(fn=save_rp_settings, inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3], outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3])
        ai_npc1.change(fn=save_rp_settings, inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3], outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3])
        ai_npc2.change(fn=save_rp_settings, inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3], outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3])
        ai_npc3.change(fn=save_rp_settings, inputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3], outputs=[rp_location, user_name, user_role, ai_npc1, ai_npc2, ai_npc3])
        delete_npc2_btn.click(fn=lambda x, y, z: delete_npc(1, x, y, z), inputs=[ai_npc1, ai_npc2, ai_npc3], outputs=[ai_npc1, ai_npc2, ai_npc3])
        delete_npc3_btn.click(fn=lambda x, y, z: delete_npc(2, x, y, z), inputs=[ai_npc1, ai_npc2, ai_npc3], outputs=[ai_npc1, ai_npc2, ai_npc3])
        eject_quality_btn.click(fn=eject_quality_model, outputs=[quality_model_dropdown, quality_mode, fast_model_dropdown, fast_mode, status_text_settings]).then(fn=toggle_model_rows, inputs=[quality_model_dropdown], outputs=[fast_row, code_row])
        refresh_btn.click(fn=lambda: [gr.update(choices=get_available_models()), gr.update(choices=get_available_models())], outputs=[quality_model_dropdown, fast_model_dropdown])
        test_models_btn.click(fn=lambda: "Model testing not implemented", outputs=[status_text_settings])
        unload_btn.click(fn=lambda: (gr.update(interactive=False), "Models unloaded"), outputs=[user_input, status_text_settings])
        save_settings_btn.click(fn=utility.save_config, inputs=None, outputs=[status_text_settings])
        demo.load(fn=lambda: [gr.update(visible=False)] * 11, inputs=None, outputs=session_buttons)

    demo.launch()

if __name__ == "__main__":
    launch_interface()