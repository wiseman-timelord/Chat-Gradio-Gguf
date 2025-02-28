# Script: `.\scripts\interface.py` 

# Imports...
import gradio as gr
import re, os
from pathlib import Path
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from scripts.temporary import (
    USER_COLOR, THINK_COLOR, RESPONSE_COLOR, SEPARATOR, MID_SEPARATOR,
    MODEL_LOADED, ALLOWED_EXTENSIONS, CTX_OPTIONS, MODEL_PATH, N_CTX,
    TEMPERATURE, TEMP_OPTIONS, VRAM_SIZE, SELECTED_GPU, current_model_settings,
    N_GPU_LAYERS, VRAM_OPTIONS, REPEAT_OPTIONS, REPEAT_PENALTY, MLOCK,
    HISTORY_DIR, BATCH_OPTIONS, N_BATCH, MODEL_FOLDER
)
from scripts import utility
from scripts.models import (
    get_streaming_response, get_response, get_available_models, reload_vectorstore,
    initialize_model, unload_model, get_model_settings
)

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

def process_uploaded_files(files, loaded_files, rag_max_docs):
    """Process uploaded files, add to available slots up to rag_max_docs."""
    try:
        files_dir = Path("files")
        files_dir.mkdir(exist_ok=True)
        new_files = []
        
        available_slots = [i for i in range(rag_max_docs) if i >= len(loaded_files)]
        for idx, file in enumerate(files[:len(available_slots)]):
            original_name = Path(file.name).name
            dest_path = files_dir / original_name
            with open(file.name, 'rb') as src, open(dest_path, 'wb') as dst:
                dst.write(src.read())
            new_files.append(str(dest_path))
        
        loaded_files.extend(new_files)
        ignored_files = len(files) - len(available_slots)
        status_msg = f"Added {len(new_files)} files"
        if ignored_files > 0:
            status_msg += f", ignored {ignored_files} files (max slots reached)"
        
        docs = utility.load_and_chunk_documents(loaded_files)
        utility.create_vectorstore(docs)
        
        button_updates, attach_files_update = update_file_slot_ui(loaded_files, rag_max_docs)
        return [loaded_files, status_msg] + button_updates + [attach_files_update]
    except Exception as e:
        button_updates, attach_files_update = update_file_slot_ui(loaded_files, rag_max_docs)
        return [loaded_files, f"Error: {str(e)}"] + button_updates + [attach_files_update]


def eject_file(loaded_files, slot_index, rag_max_docs):
    """Eject a file from the specified slot."""
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
    """Update file slot buttons based on current state."""
    button_updates = []
    for i in range(6):  # Fixed 6 buttons
        if i < len(loaded_files):
            filename = Path(loaded_files[i]).name
            short_name = (filename[:13] + "..") if len(filename) > 15 else filename
            label = f"{short_name}"
            variant = "secondary"
        else:
            label = "Slot Empty"
            variant = "primary"
        button_updates.append(gr.update(value=label, visible=True, variant=variant))
    
    # Show attach files button only if there are empty slots
    attach_files_visible = len(loaded_files) < 6
    return button_updates + [gr.update(visible=attach_files_visible)]

def update_rag_max_docs(rag_max_docs, loaded_files):
    """Update RAG_MAX_DOCS, truncate loaded_files if necessary, and refresh UI."""
    if len(loaded_files) > rag_max_docs:
        loaded_files = loaded_files[:rag_max_docs]
        docs = utility.load_and_chunk_documents(loaded_files)
        utility.create_vectorstore(docs)
    return [loaded_files] + update_file_slot_ui(loaded_files, rag_max_docs)

def handle_slot_click(slot_index, loaded_files):
    """Handle slot button clicks - either eject file or trigger upload"""
    if slot_index < len(loaded_files):
        # Eject existing file
        return eject_file(loaded_files, slot_index, 6)
    else:
        # Trigger file upload for specific slot
        return [loaded_files, "Click 'Attach Files' to add files to available slots."] + \
               update_file_slot_ui(loaded_files, 6)

def update_file_slot_ui(loaded_files, rag_max_docs):
    """Update slot buttons based on current state."""
    button_updates = []
    for i in range(10):
        if i < rag_max_docs:
            if i < len(loaded_files):
                filename = Path(loaded_files[i]).name
                short_name = (filename[:13] + "..") if len(filename) > 12 else filename
                label = f"{short_name}"
                variant = "secondary"
            else:
                label = "Slot Empty"
                variant = "primary"
            visible = True
        else:
            label = ""
            visible = False
        button_updates.append(gr.update(value=label, visible=visible, variant=variant))
    
    # Show attach files button only if there are empty slots
    attach_files_visible = rag_max_docs > 0 and len(loaded_files) < rag_max_docs
    return button_updates + [gr.update(visible=attach_files_visible)]

def handle_slot_click(slot_index, loaded_files, rag_max_docs):
    """Handle slot button clicks - either eject file or trigger upload"""
    if slot_index < len(loaded_files):
        # Eject existing file
        return eject_file(loaded_files, slot_index, rag_max_docs)
    else:
        # Trigger file upload for specific slot
        return [loaded_files, "Click 'Attach Files' to add files to available slots."] + \
               update_file_slot_ui(loaded_files, rag_max_docs)

def get_saved_sessions():
    return utility.get_saved_sessions()

def start_new_session():
    from scripts import temporary
    temporary.current_session_id = datetime.now().strftime(temporary.SESSION_FILE_FORMAT)
    temporary.session_label = ""
    return gr.Chatbot(value=[]), gr.Textbox(value="")

def load_session(session_file):
    from scripts import temporary
    from scripts import utility
    if session_file:
        session_id = session_file.replace("session_", "").replace(".json", "")
        temporary.current_session_id = session_id
        label, history = utility.load_session_history(Path(temporary.HISTORY_DIR) / session_file)
        temporary.session_label = label
        return history, "Session loaded"
    return [], "Session loaded"

def unload_model_ui():
    from scripts.temporary import STATUS_TEXTS
    yield [
        gr.Textbox.update(interactive=False),
        gr.Button.update(interactive=False),
        gr.Textbox.update(value=STATUS_TEXTS["model_unloading"]),
        gr.Textbox.update(value=STATUS_TEXTS["model_unloading"])
    ]
    unload_model()
    yield [
        gr.Textbox.update(interactive=False, placeholder="Check Status box..."),
        gr.Button.update(interactive=True),
        gr.Textbox.update(value="No model loaded"),
        gr.Textbox.update(value=STATUS_TEXTS["model_unloaded"]),
        gr.Textbox.update(value=STATUS_TEXTS["model_unloaded"])
    ]

def change_model(model_name):
    from scripts.temporary import MODEL_PATH, TEMPERATURE, current_model_settings, MODEL_FOLDER
    from scripts import utility
    MODEL_PATH = f"{MODEL_FOLDER}/{model_name}"
    settings = get_model_settings(model_name)
    current_model_settings.update(settings)
    TEMPERATURE = settings["temperature"]
    unload_model()
    initialize_model(None)
    utility.save_config()
    model_info_text = f"Loaded model: {model_name}, Category: {settings['category']}"
    return gr.Textbox(interactive=True), gr.Button(interactive=False), model_info_text

def load_model():
    from scripts.temporary import STATUS_TEXTS
    yield [
        gr.Textbox.update(interactive=False),
        gr.Button.update(interactive=False),
        gr.Textbox.update(value=STATUS_TEXTS["model_loading"]),
        gr.Textbox.update(value=STATUS_TEXTS["model_loading"])
    ]
    try:
        if MODEL_LOADED:
            unload_model()
        status_msg = initialize_model(None)
        model_name = Path(MODEL_PATH).name
        category = current_model_settings["category"]
        yield [
            gr.Textbox.update(interactive=True, placeholder="Type your message..."),
            gr.Button.update(interactive=False),
            gr.Textbox.update(value=f"Loaded model: {model_name}, Category: {category}"),
            gr.Textbox.update(value=STATUS_TEXTS["model_loaded"]),
            gr.Textbox.update(value=STATUS_TEXTS["model_loaded"])
        ]
    except Exception as e:
        yield [
            gr.Textbox.update(interactive=False),
            gr.Button.update(interactive=True),
            gr.Textbox.update(value="No model loaded"),
            gr.Textbox.update(value=f"Error: {str(e)}"),
            gr.Textbox.update(value=f"Error: {str(e)}")
        ]

def shutdown_program():
    """Unload the model and terminate the program."""
    from scripts.models import unload_model
    from scripts.temporary import STATUS_TEXTS
    try:
        print("Shutting down: Unloading model...")
        unload_model()
        print("Model unloaded. Terminating program...")
        demo.close()
        import os
        os._exit(0)
    except Exception as e:
        print(f"Error during shutdown: {str(e)}")
        return STATUS_TEXTS["error"]
    return STATUS_TEXTS["model_unloaded"]

def chat_interface(message: str, history):
    from scripts import temporary
    from scripts import utility
    from scripts.models import get_response, get_streaming_response
    history.append((f'<span style="color: {temporary.USER_COLOR}">{message}</span>', ""))
    yield history, temporary.STATUS_TEXTS["generating_response"]
    try:
        full_response = ""
        for token in get_streaming_response(message):
            full_response += token
            history[-1] = (history[-1][0], format_response(full_response))
            yield history, temporary.STATUS_TEXTS["generating_response"]
        if len(history) == 1:
            # Generate a 3-word label based only on the user's first input
            summary_prompt = f"Generate a 3-word label for a conversation starting with: '{message}'"
            temporary.session_label = get_response(summary_prompt).strip()
        utility.save_session_history(history)
        yield history, temporary.STATUS_TEXTS["response_generated"]
    except Exception as e:
        history[-1] = (history[-1][0], f'<span style="color: red;">Error: {str(e)}</span>')
        yield history, f"Error: {str(e)}"

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
        # State variables
        loaded_files_state = gr.State(value=[])
        rag_max_docs_state = gr.State(value=6)  # Fixed to 6

        with gr.Tabs() as tabs:
            with gr.Tab("Conversation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # Start New Session button at the top
                        start_new_session_btn = gr.Button("Start New Session", variant="primary")
                        
                        # Session history slots - Fixed 8 buttons
                        with gr.Column():
                            session_buttons = [gr.Button("Empty History Slot") for _ in range(8)]
                        
                        # File slot buttons - Fixed 6 buttons (3 rows of 2)
                        with gr.Column():
                            file_slot_buttons = []
                            for row in range(3):
                                with gr.Row(elem_classes=["button-row"]):
                                    file_slot_buttons.append(gr.Button("Slot Empty", visible=True, variant="primary"))
                                    file_slot_buttons.append(gr.Button("Slot Empty", visible=True, variant="primary"))

                    with gr.Column(scale=20):
                        session_log = gr.Chatbot(label="Session Log", height=375, elem_classes=["scrollable"])
                        user_input = gr.Textbox(
                            label="User Input",
                            lines=4,
                            interactive=False,
                            placeholder="Enter text here...",
                            elem_classes=["scrollable"]
                        )
                        status_text = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Select and load a model on Configuration page."
                        )
                        with gr.Row():
                            send_btn = gr.Button("Send Input", variant="primary", scale=4)
                            edit_previous_btn = gr.Button("Edit Previous", scale=1)                      
                            attach_files_btn = gr.UploadButton(
                                "Attach Files",
                                visible=True,
                                file_types=[f".{ext}" for ext in ALLOWED_EXTENSIONS],
                                file_count="multiple"
                            )
                            web_search_switch = gr.Checkbox(label="Web-Search", value=False, scale=1)
                            shutdown_btn = gr.Button("Shutdown Program", variant="stop", scale=1)

            with gr.Tab("Configuration"):
                with gr.Row():
                    model_dir_text = gr.Textbox(label="Model Folder", value=MODEL_FOLDER, scale=20)
                    browse_btn = gr.Button("Browse", scale=1)
                model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    label="Select Model",
                    value=MODEL_PATH.split('/')[-1],
                    allow_custom_value=True
                )
                with gr.Row():
                    temperature_dropdown = gr.Dropdown(
                        choices=TEMP_OPTIONS,
                        label="Temperature (Creativity)",
                        value=TEMPERATURE,
                        allow_custom_value=True
                    )
                    repeat_penalty_dropdown = gr.Dropdown(
                        choices=REPEAT_OPTIONS,
                        label="Repeat Penalty (Do not repeat)",
                        value=REPEAT_PENALTY,
                        allow_custom_value=True
                    )
                    n_ctx_dropdown = gr.Dropdown(
                        choices=CTX_OPTIONS,
                        label="Context Window (Speed vs Context)",
                        value=N_CTX
                    )
                    batch_size_dropdown = gr.Dropdown(
                        choices=BATCH_OPTIONS,
                        label="Batch Size (Output Length)",
                        value=N_BATCH
                    )
                with gr.Row():
                    gpu_dropdown = gr.Dropdown(
                        choices=utility.get_available_gpus(),
                        label="Select GPU",
                        value=SELECTED_GPU
                    )
                    vram_dropdown = gr.Dropdown(
                        choices=VRAM_OPTIONS,
                        label="VRAM Size (Free GPU Memory)",
                        value=VRAM_SIZE
                    )
                    mlock_checkbox = gr.Checkbox(
                        label="MLock Enabled (Keep model loaded)",
                        value=MLOCK
                    )
                gr.Markdown("Note: GPU layers calculated from model details and VRAM. 0 layers = CPU-only.")
                status_text_settings = gr.Textbox(label="Status", interactive=False)
                with gr.Row():
                    load_btn = gr.Button("Load Model", variant="secondary", scale=1)
                    unload_btn = gr.Button("Unload Model", variant="secondary", scale=1)
                    save_settings_btn = gr.Button("Save Settings", scale=1)

        # Helper functions
        def update_session_buttons():
            """Update session buttons with labels from saved sessions."""
            sessions = utility.get_saved_sessions()
            button_updates = []
            for i in range(8):  # Fixed 8 buttons
                if i < len(sessions):
                    session_path = Path(HISTORY_DIR) / sessions[i]
                    try:
                        label, _ = utility.load_session_history(session_path)
                        btn_label = f"{label}" if label else f"Session {i+1}"
                    except:
                        btn_label = f"Session {i+1}"
                else:
                    btn_label = "Empty History Slot"
                button_updates.append(gr.update(value=btn_label, visible=True))
            return button_updates

        def load_session_by_index(index):
            sessions = utility.get_saved_sessions()
            if index < len(sessions):
                session_file = sessions[index]
                return load_session(session_file)
            else:
                return [], "No session to load"

        def update_file_slot_ui(loaded_files, rag_max_docs):
            """Update file slot buttons based on current state."""
            button_updates = []
            for i in range(6):  # Fixed 6 buttons
                if i < len(loaded_files):
                    filename = Path(loaded_files[i]).name
                    short_name = (filename[:13] + "..") if len(filename) > 15 else filename
                    label = f"{short_name}"
                    variant = "secondary"
                else:
                    label = "Slot Empty"
                    variant = "primary"
                button_updates.append(gr.update(value=label, visible=True, variant=variant))
            attach_files_visible = len(loaded_files) < 6
            return button_updates, gr.update(visible=attach_files_visible)

        def process_uploaded_files(files, loaded_files, rag_max_docs):
            """Process uploaded files, add to available slots up to rag_max_docs."""
            try:
                files_dir = Path("files")
                files_dir.mkdir(exist_ok=True)
                new_files = []
                
                available_slots = [i for i in range(rag_max_docs) if i >= len(loaded_files)]
                for idx, file in enumerate(files[:len(available_slots)]):
                    original_name = Path(file.name).name
                    dest_path = files_dir / original_name
                    with open(file.name, 'rb') as src, open(dest_path, 'wb') as dst:
                        dst.write(src.read())
                    new_files.append(str(dest_path))
                
                loaded_files.extend(new_files)
                ignored_files = len(files) - len(available_slots)
                status_msg = f"Added {len(new_files)} files"
                if ignored_files > 0:
                    status_msg += f", ignored {ignored_files} files (max slots reached)"
                
                docs = utility.load_and_chunk_documents(loaded_files)
                utility.create_vectorstore(docs)
                
                button_updates, attach_files_update = update_file_slot_ui(loaded_files, rag_max_docs)
                return [loaded_files, status_msg] + button_updates + [attach_files_update]
            except Exception as e:
                button_updates, attach_files_update = update_file_slot_ui(loaded_files, rag_max_docs)
                return [loaded_files, f"Error: {str(e)}"] + button_updates + [attach_files_update]

        def eject_file(loaded_files, slot_index, rag_max_docs):
            """Eject a file from the specified slot."""
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

        def handle_slot_click(slot_index, loaded_files, rag_max_docs):
            """Handle slot button clicks - only eject files, prompt Attach Files for empty slots."""
            if slot_index < len(loaded_files):
                return eject_file(loaded_files, slot_index, rag_max_docs)
            else:
                button_updates, attach_files_update = update_file_slot_ui(loaded_files, rag_max_docs)
                return [loaded_files, "Click 'Attach Files' to add files."] + button_updates + [attach_files_update]

        def edit_previous(history, input_box):
            if len(history) >= 2:
                history = history[:-2]
                last_user_input = history[-1][0] if history else ""
                return history, gr.Textbox.update(value=last_user_input)
            elif len(history) == 1:
                last_user_input = history[0][0]
                history = []
                return history, gr.Textbox.update(value=last_user_input)
            else:
                return history, gr.Textbox.update(value="")

        # Event handlers for Conversation tab
        edit_previous_btn.click(
            fn=edit_previous,
            inputs=[session_log, user_input],
            outputs=[session_log, user_input],
            api_name="edit_previous"
        )

        send_btn.click(
            fn=chat_interface,
            inputs=[user_input, session_log],
            outputs=[session_log, status_text],
            api_name="chat_interface"
        ).then(
            fn=update_session_buttons,
            inputs=None,
            outputs=session_buttons,
            api_name="update_session_buttons_after_send"
        )

        attach_files_btn.upload(
            fn=process_uploaded_files,
            inputs=[attach_files_btn, loaded_files_state, rag_max_docs_state],
            outputs=[loaded_files_state, status_text] + file_slot_buttons + [attach_files_btn],
            api_name="process_uploaded_files"
        )

        start_new_session_btn.click(
            fn=start_new_session,
            inputs=None,
            outputs=[session_log, user_input],
            api_name="restart_session"
        ).then(
            fn=lambda: "New session started",
            outputs=[status_text],
            api_name="new_session_started"
        )

        shutdown_btn.click(
            fn=shutdown_program,
            inputs=None,
            outputs=[status_text],
            api_name="shutdown_program"
        )

        # Slot button click handlers
        for i, btn in enumerate(file_slot_buttons):
            btn.click(
                fn=lambda state, rag_max, idx=i: handle_slot_click(idx, state, rag_max),
                inputs=[loaded_files_state, rag_max_docs_state],
                outputs=[loaded_files_state, status_text] + file_slot_buttons + [attach_files_btn],
                api_name=f"slot_{i}_action"
            )

        # Session button click handlers
        for i, btn in enumerate(session_buttons):
            btn.click(
                fn=lambda idx=i: load_session_by_index(idx),
                inputs=[],
                outputs=[session_log, status_text],
                api_name=f"load_session_{i}"
            )

        # Configuration tab event handlers
        load_btn.click(
            fn=load_model,
            outputs=[user_input, load_btn, status_text, status_text_settings],
            api_name="load_model"
        )

        unload_btn.click(
            fn=unload_model_ui,
            outputs=[user_input, load_btn, status_text, status_text_settings],
            api_name="unload_model"
        )

        save_settings_btn.click(
            fn=utility.save_config,
            inputs=None,
            outputs=[status_text_settings],
            api_name="save_config"
        )

        temperature_dropdown.change(
            fn=lambda x: utility.update_setting("temperature", x),
            inputs=[temperature_dropdown],
            outputs=[user_input, load_btn],
            api_name="update_temperature"
        )

        model_dir_text.change(
            fn=lambda x: utility.update_setting("model_folder", x),
            inputs=[model_dir_text],
            outputs=[user_input, load_btn],
            api_name="update_model_folder"
        ).then(
            fn=lambda: gr.Dropdown.update(choices=get_available_models()),
            outputs=[model_dropdown],
            api_name="update_model_dropdown"
        )

        # Initial UI setup
        demo.load(
            fn=update_file_slot_ui,
            inputs=[loaded_files_state, rag_max_docs_state],
            outputs=file_slot_buttons + [attach_files_btn],
            api_name="initial_file_slot_ui"
        ).then(
            fn=update_session_buttons,
            inputs=None,
            outputs=session_buttons,
            api_name="initial_session_buttons"
        )

    demo.launch()

if __name__ == "__main__":
    launch_interface()