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
    MAX_SESSIONS, current_model_settings, N_GPU_LAYERS, VRAM_OPTIONS
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
    unload_model()
    yield [
        gr.Textbox.update(interactive=False, placeholder="Check Status box..."),
        gr.Button.update(interactive=True),
        gr.Textbox.update(value="No model loaded"),
        gr.Textbox.update(value=STATUS_TEXTS["model_unloaded"]),
        gr.Textbox.update(value=STATUS_TEXTS["model_unloaded"])
    ]

def change_model(model_name):
    from scripts.temporary import MODEL_PATH, TEMPERATURE, current_model_settings
    from scripts import utility
    MODEL_PATH = f"models/{model_name}"
    settings = get_model_settings(model_name)
    current_model_settings.update(settings)
    TEMPERATURE = settings["temperature"]
    unload_model()
    initialize_model(None)
    utility.save_config()
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
            return "Files loaded and Ready"
        else:
            return "Error: No valid documents"
    except Exception as e:
        return f"Error: {str(e)}"

def create_control_row():
    return gr.Row(
        gr.Button("Send Input", variant="primary"),
        gr.UploadButton("Attach Files", file_types=list(ALLOWED_EXTENSIONS), file_count="multiple"),
        gr.Button("Edit Previous"),
        gr.Button("Remove Last")
    )

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
    with gr.Blocks(title="Chat-Gradio-Gguf") as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Conversation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        session_tabs = gr.Tabs()
                        with session_tabs:
                            with gr.Tab("Session History Index", id="session_history"):
                                start_new_session_btn = gr.Button("Start New Session")
                    with gr.Column(scale=4):
                        session_log = gr.Chatbot(label="Session Log", height=500)
                        user_input = gr.Textbox(label="User Input", interactive=False, placeholder="Enter text here...")
                        status_text = gr.Textbox(label="Status", interactive=False, value="Select and load a model on Configuration page.")
                        
                        # Buttons and switch row, defined like the working version
                        with gr.Row():
                            send_btn = gr.Button("Send Input", variant="primary", scale=2)
                            edit_previous_btn = gr.Button("Edit Previous", scale=1)
                            attach_files_btn = gr.UploadButton("Attach Files", file_types=list(ALLOWED_EXTENSIONS), file_count="multiple", scale=1)
                            save_session_btn = gr.Button("Save Session", scale=1)  # Reintroduced
                            web_search_switch = gr.Checkbox(label="Web-Search", value=False, scale=1)

            with gr.Tab("Configuration"):
                model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    label="Select Model",
                    value=MODEL_PATH.split('/')[-1],
                    allow_custom_value=True
                )
                temperature_dropdown = gr.Dropdown(
                    choices=TEMP_OPTIONS,
                    label="Temperature",
                    value=TEMPERATURE,
                    allow_custom_value=True
                )
                n_ctx_dropdown = gr.Dropdown(
                    choices=CTX_OPTIONS,
                    label="Context Window",
                    value=N_CTX
                )
                vram_dropdown = gr.Dropdown(
                    choices=VRAM_OPTIONS,
                    label="VRAM Size (MB)",
                    value=VRAM_SIZE
                )
                gpu_dropdown = gr.Dropdown(
                    choices=utility.get_available_gpus(),
                    label="Select GPU",
                    value=SELECTED_GPU
                )
                gr.Markdown("Note: GPU layers calculated from model details and VRAM. 0 layers = CPU-only.")
                status_text_settings = gr.Textbox(label="Status", interactive=False)
                
                # Buttons row, defined like the working version
                with gr.Row():
                    load_btn = gr.Button("Load Model", variant="secondary", scale=1)
                    unload_btn = gr.Button("Unload Model", variant="secondary", scale=1)
                    save_settings_btn = gr.Button("Save Settings", scale=1)

        # Helper functions
        def update_session_tabs():
            sessions = utility.get_saved_sessions()
            tabs = [gr.Tab("Session History Index", id="session_history")]
            for session in sessions[:MAX_SESSIONS]:
                label, _ = utility.load_session_history(Path(HISTORY_DIR) / session)
                tabs.append(gr.Tab(label, id=session))
            return gr.Tabs.update(children=tabs)

        def update_status(msg):
            return gr.Textbox.update(value=msg)

        def safe_chat_interface(message: str, history):
            from scripts.temporary import STATUS_TEXTS
            history.append((message, ""))
            yield history, STATUS_TEXTS["generating_response"]
            try:
                full_response = ""
                for token in get_streaming_response(message):
                    full_response += token
                    history[-1] = (history[-1][0], full_response)
                    yield history, STATUS_TEXTS["generating_response"]
                if len(history) == 1:
                    summary_prompt = f"Summarize the following conversation in exactly three words:\nUser: {message}\nAssistant: {full_response}"
                    globals()["session_label"] = get_response(summary_prompt).strip()
                    utility.save_session_history(history)
                yield history, STATUS_TEXTS["response_generated"]
            except Exception as e:
                history[-1] = (history[-1][0], f'<span style="color: red;">Error: {str(e)}</span>')
                yield history, f"Error: {str(e)}"

        def safe_process_uploaded_files(files):
            try:
                return process_uploaded_files(files)
            except Exception as e:
                return f"Error: {str(e)}"

        def safe_save_session_history(history):
            try:
                return utility.save_session_history(history)
            except Exception as e:
                return f"Error: {str(e)}"

        def safe_load_model():
            try:
                return load_model()
            except Exception as e:
                return [
                    gr.Textbox.update(interactive=False),
                    gr.Button.update(interactive=True),
                    gr.Textbox.update(value="No model loaded"),
                    gr.Textbox.update(value=f"Error: {str(e)}"),
                    gr.Textbox.update(value=f"Error: {str(e)}")
                ]

        def safe_unload_model_ui():
            try:
                return unload_model_ui()
            except Exception as e:
                return [
                    gr.Textbox.update(interactive=False),
                    gr.Button.update(interactive=True),
                    gr.Textbox.update(value="No model loaded"),
                    gr.Textbox.update(value=f"Error: {str(e)}"),
                    gr.Textbox.update(value=f"Error: {str(e)}")
                ]

        def safe_load_session(tab_id):
            try:
                return load_session(tab_id) if tab_id != "session_history" else start_new_session()
            except Exception as e:
                return [session_log, f"Error: {str(e)}"]

        def safe_start_new_session():
            try:
                return start_new_session()
            except Exception as e:
                return [session_log, f"Error: {str(e)}"]

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
            fn=safe_chat_interface,
            inputs=[user_input, session_log],
            outputs=[session_log, status_text],
            api_name="chat_interface"
        ).then(
            fn=update_session_tabs,
            inputs=None,
            outputs=[session_tabs],
            api_name="update_session_tabs_1"
        )

        attach_files_btn.upload(
            fn=safe_process_uploaded_files,
            inputs=[attach_files_btn],
            outputs=[status_text],
            api_name="process_uploaded_files"
        )

        save_session_btn.click(
            fn=safe_save_session_history,
            inputs=[session_log],
            outputs=[status_text],
            api_name="save_session_history"
        ).then(
            fn=update_session_tabs,
            inputs=None,
            outputs=[session_tabs],
            api_name="update_session_tabs_2"
        )

        start_new_session_btn.click(
            fn=safe_start_new_session,
            inputs=None,
            outputs=[session_log, user_input],
            api_name="restart_session"
        ).then(
            fn=lambda: "New session started",
            outputs=[status_text],
            api_name="new_session_started"
        )

        # Event handlers for Configuration tab
        load_btn.click(
            fn=safe_load_model,
            outputs=[user_input, load_btn, status_text, status_text_settings],
            api_name="load_model"
        )

        unload_btn.click(
            fn=safe_unload_model_ui,
            outputs=[user_input, load_btn, status_text, status_text_settings],
            api_name="unload_model"
        )

        save_settings_btn.click(
            fn=utility.save_config,
            inputs=None,
            outputs=[status_text_settings],
            api_name="save_config"
        )

        # Session tab selection
        session_tabs.select(
            fn=safe_load_session,
            inputs=[session_tabs],
            outputs=[session_log, user_input],
            api_name="load_session"
        ).then(
            fn=update_session_tabs,
            inputs=None,
            outputs=[session_tabs],
            api_name="update_session_tabs_3"
        )

        # Settings updates
        temperature_dropdown.change(
            fn=lambda x: utility.update_setting("temperature", x),
            inputs=[temperature_dropdown],
            outputs=[user_input, load_btn],
            api_name="update_temperature"
        )

        n_ctx_dropdown.change(
            fn=lambda x: utility.update_setting("n_ctx", x),
            inputs=[n_ctx_dropdown],
            outputs=[user_input, load_btn],
            api_name="update_n_ctx"
        )

        vram_dropdown.change(
            fn=lambda x: utility.update_setting("vram_size", x),
            inputs=[vram_dropdown],
            outputs=[user_input, load_btn],
            api_name="update_vram_size"
        ).then(
            fn=lambda: gr.Textbox.update(value=f"Calculated: {N_GPU_LAYERS} (0 means all in system RAM)"),
            inputs=None,
            outputs=[status_text_settings],
            api_name="update_gpu_layers_info"
        )

        model_dropdown.change(
            fn=change_model,
            inputs=[model_dropdown],
            outputs=[user_input, load_btn],
            api_name="change_model"
        ).then(
            fn=lambda: gr.Textbox.update(value=f"Calculated: {N_GPU_LAYERS} (0 means all in system RAM)"),
            inputs=None,
            outputs=[status_text_settings],
            api_name="update_gpu_layers_info_2"
        )

        gpu_dropdown.change(
            fn=lambda x: utility.update_setting("selected_gpu", x),
            inputs=[gpu_dropdown],
            outputs=[user_input, load_btn],
            api_name="update_selected_gpu"
        )

    demo.launch()

if __name__ == "__main__":
    launch_interface()