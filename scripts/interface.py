# Imports...
import gradio as gr
import re
from pathlib import Path
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from scripts.temporary import (
    USER_COLOR, THINK_COLOR, RESPONSE_COLOR, SEPARATOR, MID_SEPARATOR,
    MODEL_LOADED, ALLOWED_EXTENSIONS, CTX_OPTIONS, MODEL_PATH, N_CTX,
    TEMPERATURE, TEMP_OPTIONS, VRAM_SIZE, SELECTED_GPU, HISTORY_OPTIONS,
    MAX_SESSIONS, current_model_settings, N_GPU_LAYERS
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

def chat_interface(message: str, history):
    from scripts.temporary import STATUS_TEXTS
    history.append((f'<span style="color: {USER_COLOR}">{message}</span>', ""))
    yield history, STATUS_TEXTS["generating_response"]
    try:
        full_response = ""
        for token in get_streaming_response(message):
            full_response += token
            history[-1] = (history[-1][0], format_response(full_response))
            yield history, STATUS_TEXTS["generating_response"]
        if len(history) == 1:
            summary_prompt = f"Summarize the following conversation in exactly three words:\nUser: {message}\nAssistant: {full_response}"
            globals()["session_label"] = get_response(summary_prompt).strip()
            utility.save_session_history(history)
        yield history, STATUS_TEXTS["response_generated"]
    except Exception as e:
        history[-1] = (history[-1][0], f'<span style="color: red;">Error: {str(e)}</span>')
        yield history, f"Error: {str(e)}"

def unload_model_ui():
    from scripts.temporary import STATUS_TEXTS
    yield [
        gr.Textbox.update(interactive=False),
        gr.Button.update(interactive=False),
        gr.Textbox.update(value=STATUS_TEXTS["model_unloading"]),
        gr.Textbox.update(value=STATUS_TEXTS["model_unloading"])
    ]
    unload_model()  # Call the models.py version
    yield [
        gr.Textbox.update(interactive=False, placeholder="Load a model to start chatting..."),
        gr.Button.update(interactive=True),
        gr.Textbox.update(value="No model loaded"),
        gr.Textbox.update(value=STATUS_TEXTS["model_unloaded"]),
        gr.Textbox.update(value=STATUS_TEXTS["model_unloaded"])
    ]

def change_model(model_name):
    from scripts.temporary import MODEL_PATH, TEMPERATURE, current_model_settings
    from scripts import utility  # Add this for save_config
    MODEL_PATH = f"models/{model_name}"
    settings = get_model_settings(model_name)
    current_model_settings.update(settings)
    TEMPERATURE = settings["temperature"]
    unload_model()
    initialize_model(None)
    utility.save_config()  # Persist changes to config
    model_info_text = f"Loaded model: {model_name}, Category: {settings['category']}"
    return gr.Textbox(interactive=True), gr.Button(interactive=False), model_info_text

def web_search_trigger(query):
    try:
        result = utility.web_search(query)
        return result if result else "No results found"
    except Exception as e:
        return f"Error: {str(e)}"

def process_uploaded_files(files):
    try:
        files_dir = Path("files")
        files_dir.mkdir(exist_ok=True)
        for file in files:
            original_name = Path(file.name).name
            dest_path = files_dir / original_name
            with open(file.name, 'rb') as src, open(dest_path, 'wb') as dst:
                dst.write(src.read())
        docs = utility.load_and_chunk_documents(files_dir)
        if docs:
            utility.create_vectorstore(docs)
            reload_vectorstore("general_knowledge")
            yield "Files loaded and Ready"
        else:
            yield "Error: No valid documents"
    except Exception as e:
        yield f"Error: {str(e)}"

def create_control_row():
    return gr.Row(
        gr.Button("Send Input", variant="primary"),
        gr.UploadButton("Attach Files", file_types=list(ALLOWED_EXTENSIONS), file_count="multiple"),
        gr.Button("Edit Previous"),
        gr.Button("Remove Last")
    )

def create_session_controls():
    with gr.Row() as row:
        load_btn = gr.Button("Load Model", variant="secondary")
        unload_btn = gr.Button("Unload Model", variant="secondary")
        restart_btn = gr.Button("Restart Session")
        shutdown_btn = gr.Button("Shutdown Program", variant="stop")
    return row, load_btn, unload_btn, shutdown_btn

def get_saved_sessions():
    return utility.get_saved_sessions()

def start_new_session():
    global message_history, session_label
    message_history = []
    session_label = ""
    return gr.Chatbot(value=[]), gr.Textbox(value="")

def load_session(session_file):
    if session_file:
        label, history = utility.load_session_history(Path(HISTORY_DIR) / session_file)
        global session_label
        session_label = label
        return history, "Session loaded"
    return history, "Session loaded"

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

def launch_interface():
    with gr.Blocks() as demo:
        # Chat Interface
        chatbot = gr.Chatbot(height=500)
        input_box = gr.Textbox(interactive=False, placeholder="Load a model to start chatting...")
        status_text = gr.Textbox(label="Status", interactive=False, value="Select and load a model on Configuration page")

        with gr.Row():
            with gr.Column(scale=1):
                session_tabs = gr.Tabs()
                with session_tabs:
                    with gr.Tab("Start New Session", id="new_session"):
                        pass

            with gr.Column(scale=4):
                with gr.Tab("Chat"):
                    model_info = gr.Textbox(label="Current Model", interactive=False, value="No model loaded")
                    # Create buttons explicitly
                    send_btn = gr.Button("Send Input", variant="primary")
                    upload_btn = gr.UploadButton("Attach Files", file_types=list(ALLOWED_EXTENSIONS), file_count="multiple")
                    edit_btn = gr.Button("Edit Previous")
                    remove_btn = gr.Button("Remove Last")
                    control_row = gr.Row(send_btn, upload_btn, edit_btn, remove_btn)
                    web_btn = gr.Button("Web Search")
                    web_output = gr.Textbox(label="Web Results", interactive=False)
                    upload_output = gr.Textbox(label="Upload Status", interactive=False)
                    save_session_btn = gr.Button("Save Session")
                    save_session_output = gr.Textbox(label="Save Status", interactive=False)
                    session_controls, load_btn, unload_btn, shutdown_btn = create_session_controls()

            with gr.Tab("Settings"):
                status_text_settings = gr.Textbox(label="Status", interactive=False)
                model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    label="Select Model",
                    value=MODEL_PATH.split('/')[-1],
                    allow_custom_value=True  # Added to prevent warnings
                )
                temperature_dropdown = gr.Dropdown(
                    choices=TEMP_OPTIONS,
                    label="Temperature",
                    value=TEMPERATURE,
                    allow_custom_value=True  # Added to prevent warnings
                )
                n_ctx_dropdown = gr.Dropdown(
                    choices=CTX_OPTIONS,
                    label="Context Window",
                    value=N_CTX
                )
                vram_dropdown = gr.Dropdown(
                    choices=[1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768],
                    label="VRAM Size (MB)",
                    value=VRAM_SIZE
                )
                gpu_dropdown = gr.Dropdown(
                    choices=utility.get_available_gpus(),
                    label="Select GPU",
                    value=SELECTED_GPU
                )
                gpu_layers_info = gr.Textbox(
                    label="GPU Layers to Offload",
                    value=f"Calculated: {N_GPU_LAYERS} (0 means all in system RAM)",
                    interactive=False
                )
                gr.Markdown("Note: GPU layers auto-calculated based on, model size and VRAM. 0 layers = CPU-only.")
                save_btn = gr.Button("Save Settings")

        # Event handlers
        def update_session_tabs():
            sessions = utility.get_saved_sessions()
            tabs = [gr.Tab("Start New Session", id="new_session")]
            for session in sessions[:MAX_SESSIONS]:
                label, _ = utility.load_session_history(Path(HISTORY_DIR) / session)
                tabs.append(gr.Tab(label, id=session))
            return gr.Tabs.update(children=tabs)

        def update_status(msg):
            """Update both status textboxes with the same message."""
            return [gr.Textbox.update(value=msg), gr.Textbox.update(value=msg)]

        # Chat interface with error handling
        def safe_chat_interface(message: str, history):
            try:
                return chat_interface(message, history)
            except Exception as e:
                history.append((history[-1][0], f'<span style="color: red;">Error: {str(e)}</span>'))
                return history, f"Error: {str(e)}"

        send_btn.click(
            fn=safe_chat_interface,
            inputs=[input_box, chatbot],
            outputs=[chatbot, status_text],
            api_name="chat_interface"  # Explicit api_name
        ).then(
            fn=update_session_tabs,
            inputs=None,
            outputs=[session_tabs],
            api_name="update_session_tabs_1"  # Explicit api_name
        )

        # File upload with error handling
        def safe_process_uploaded_files(files):
            try:
                return process_uploaded_files(files)
            except Exception as e:
                return f"Error: {str(e)}"

        upload_btn.upload(
            fn=safe_process_uploaded_files,
            inputs=[upload_btn],
            outputs=[upload_output, status_text],
            api_name="process_uploaded_files"  # Explicit api_name
        )

        # Web search with error handling
        def safe_web_search_trigger(query):
            try:
                return web_search_trigger(query)
            except Exception as e:
                return f"Error: {str(e)}"

        web_btn.click(
            fn=lambda: "Searching web...",
            outputs=status_text,
            api_name="web_search_start"  # Explicit api_name
        ).then(
            fn=safe_web_search_trigger,
            inputs=[input_box],
            outputs=[web_output],
            api_name="web_search_trigger"  # Explicit api_name
        ).then(
            fn=lambda: "Web results ready",
            outputs=status_text,
            api_name="web_search_end"  # Explicit api_name
        )

        # Session save with error handling
        def safe_save_session_history(history):
            try:
                return utility.save_session_history(history)
            except Exception as e:
                return f"Error: {str(e)}"

        save_session_btn.click(
            fn=safe_save_session_history,
            inputs=[chatbot],
            outputs=[save_session_output, status_text],
            api_name="save_session_history"  # Explicit api_name
        ).then(
            fn=update_session_tabs,
            inputs=None,
            outputs=[session_tabs],
            api_name="update_session_tabs_2"  # Explicit api_name
        )

        # Model loading with error handling
        def safe_load_model():
            try:
                return load_model()
            except Exception as e:
                return [
                    gr.Textbox(interactive=False),
                    gr.Button(interactive=True),
                    "No model loaded",
                    f"Error: {str(e)}",
                    f"Error: {str(e)}"
                ]

        load_btn.click(
            fn=safe_load_model,
            outputs=[input_box, load_btn, model_info, status_text, status_text_settings],
            api_name="load_model"  # Explicit api_name
        )

        # Model unloading with error handling
        def safe_unload_model_ui():
            try:
                return unload_model_ui()
            except Exception as e:
                return [
                    gr.Textbox(interactive=False),
                    gr.Button(interactive=True),
                    "No model loaded",
                    f"Error: {str(e)}",
                    f"Error: {str(e)}"
                ]

        unload_btn.click(
            fn=safe_unload_model_ui,
            outputs=[input_box, load_btn, model_info, status_text, status_text_settings],
            api_name="unload_model"  # Explicit api_name
        )

        # Shutdown button
        shutdown_btn.click(
            fn=lambda: exit(0),
            inputs=None,
            outputs=None,
            api_name="shutdown"  # Explicit api_name
        )

        # Settings updates
        temperature_dropdown.change(
            fn=lambda x: utility.update_setting("temperature", x),
            inputs=[temperature_dropdown],
            outputs=[input_box, load_btn],
            api_name="update_temperature"  # Explicit api_name
        )

        n_ctx_dropdown.change(
            fn=lambda x: utility.update_setting("n_ctx", x),
            inputs=[n_ctx_dropdown],
            outputs=[input_box, load_btn],
            api_name="update_n_ctx"  # Explicit api_name
        )

        vram_dropdown.change(
            fn=lambda x: utility.update_setting("vram_size", x),
            inputs=[vram_dropdown],
            outputs=[input_box, load_btn],
            api_name="update_vram_size"  # Explicit api_name
        ).then(
            fn=lambda: gr.Textbox.update(value=f"Calculated: {N_GPU_LAYERS} (0 means all in system RAM)"),
            inputs=None,
            outputs=[gpu_layers_info],
            api_name="update_gpu_layers_info"  # Explicit api_name
        )

        model_dropdown.change(
            fn=change_model,
            inputs=[model_dropdown],
            outputs=[input_box, load_btn, model_info],
            api_name="change_model"  # Explicit api_name
        ).then(
            fn=lambda: gr.Textbox.update(value=f"Calculated: {N_GPU_LAYERS} (0 means all in system RAM)"),
            inputs=None,
            outputs=[gpu_layers_info],
            api_name="update_gpu_layers_info_2"  # Explicit api_name
        )

        gpu_dropdown.change(
            fn=lambda x: utility.update_setting("selected_gpu", x),
            inputs=[gpu_dropdown],
            outputs=[input_box, load_btn],
            api_name="update_selected_gpu"  # Explicit api_name
        )

        # Save settings with error handling
        def safe_save_config():
            try:
                utility.save_config()
                return "Settings saved"
            except Exception as e:
                return f"Error: {str(e)}"

        save_btn.click(
            fn=safe_save_config,
            inputs=None,
            outputs=[status_text, status_text_settings],
            api_name="save_config"  # Explicit api_name
        )

        # Session tab selection with error handling
        def safe_load_session(tab_id):
            try:
                return load_session(tab_id) if tab_id != "new_session" else start_new_session()
            except Exception as e:
                return [chatbot, f"Error: {str(e)}"]

        session_tabs.select(
            fn=safe_load_session,
            inputs=[session_tabs],
            outputs=[chatbot, input_box],
            api_name="load_session"  # Explicit api_name
        ).then(
            fn=update_session_tabs,
            inputs=None,
            outputs=[session_tabs],
            api_name="update_session_tabs_3"  # Explicit api_name
        )

        # Restart session with error handling
        def safe_start_new_session():
            try:
                return start_new_session()
            except Exception as e:
                return [chatbot, f"Error: {str(e)}"]

        # Ensure restart_btn is defined and accessible
        restart_btn = session_controls.children[2]  # Assuming restart_btn is the third button in session_controls

        restart_btn.click(
            fn=safe_start_new_session,
            inputs=None,
            outputs=[chatbot, input_box],
            api_name="restart_session"  # Explicit api_name
        ).then(
            fn=lambda: "New session started",
            outputs=[status_text, status_text_settings],
            api_name="new_session_started"  # Explicit api_name
        )

    demo.launch()

    
if __name__ == "__main__":
    launch_interface()