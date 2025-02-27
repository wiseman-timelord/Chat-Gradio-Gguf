# Script: `.\scripts\interface.py`

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
    MAX_SESSIONS, current_model_settings, N_GPU_LAYERS, VRAM_OPTIONS,
    REPEAT_OPTIONS, REPEAT_PENALTY, MLOCK, HISTORY_DIR, BATCH_OPTIONS,
    N_BATCH, MAX_DOCS_OPTIONS, RAG_MAX_DOCS, MODEL_FOLDER
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
            # Calculate context size: Total Characters * 1.25
            summary_prompt = f"Summarize the following conversation in exactly three words:\nUser: {message}\nAssistant: {full_response}"
            prompt_length = len(summary_prompt)
            n_ctx_to_use = int(prompt_length * 1.25)
            # Ensure n_ctx_to_use is within model limits, default to N_CTX if exceeded
            n_ctx_to_use = min(n_ctx_to_use, temporary.N_CTX)
            # Note: We assume get_response uses the model's n_ctx, no direct way to set it per call
            temporary.session_label = get_response(summary_prompt).strip()
        utility.save_session_history(history)
        yield history, temporary.STATUS_TEXTS["response_generated"]
    except Exception as e:
        history[-1] = (history[-1][0], f'<span style="color: red;">Error: {str(e)}</span>')
        yield history, f"Error: {str(e)}"

def launch_interface():
    with gr.Blocks(title="Chat-Gradio-Gguf") as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Conversation"):
                with gr.Row():
                    with gr.Column(scale=1):  # Adjusted for session history width
                        session_tabs = gr.Tabs()
                        with session_tabs:
                            with gr.Tab("Session History Index", id="session_history"):
                                start_new_session_btn = gr.Button("Start New Session")
                    with gr.Column(scale=20):  # Adjusted for session history width
                        session_log = gr.Chatbot(label="Session Log", height=500)
                        user_input = gr.Textbox(
                            label="User Input",
                            lines=5,  # Expand to 5 lines with scrolling
                            interactive=False,
                            placeholder="Enter text here..."
                        )
                        status_text = gr.Textbox(
                            label="Status",
                            interactive=False,
                            value="Select and load a model on Configuration page."
                        )
                        with gr.Row():
                            send_btn = gr.Button("Send Input", variant="primary", scale=2)
                            edit_previous_btn = gr.Button("Edit Previous", scale=1)
                            attach_files_btn = gr.UploadButton(
                                "Attach Files",
                                file_types=list(ALLOWED_EXTENSIONS),
                                file_count="multiple",
                                scale=1
                            )
                            web_search_switch = gr.Checkbox(label="Web-Search", value=False, scale=1)

            with gr.Tab("Configuration"):
                model_dropdown = gr.Dropdown(
                    choices=get_available_models(),
                    label="Select Model",
                    value=MODEL_PATH.split('/')[-1],
                    allow_custom_value=True
                )
                with gr.Row():
                    temperature_dropdown = gr.Dropdown(
                        choices=TEMP_OPTIONS,
                        label="Temperature",
                        value=TEMPERATURE,
                        allow_custom_value=True
                    )
                    repeat_penalty_dropdown = gr.Dropdown(
                        choices=REPEAT_OPTIONS,
                        label="Repeat Penalty",
                        value=REPEAT_PENALTY,
                        allow_custom_value=True
                    )
                    n_ctx_dropdown = gr.Dropdown(
                        choices=CTX_OPTIONS,
                        label="Context Window",
                        value=N_CTX
                    )
                    batch_size_dropdown = gr.Dropdown(
                        choices=BATCH_OPTIONS,
                        label="Batch Size",
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
                        label="VRAM Size (MB)",
                        value=VRAM_SIZE
                    )
                    mlock_checkbox = gr.Checkbox(
                        label="MLock Enabled",
                        value=MLOCK
                    )
                with gr.Row():
                    max_sessions_dropdown = gr.Dropdown(
                        choices=HISTORY_OPTIONS,
                        label="Max Session History",
                        value=MAX_SESSIONS
                    )
                    max_docs_dropdown = gr.Dropdown(
                        choices=MAX_DOCS_OPTIONS,
                        label="Max RAG Documents",
                        value=RAG_MAX_DOCS
                    )
                with gr.Row():
                    model_dir_text = gr.Textbox(label="Model Folder", value=MODEL_FOLDER)
                gr.Markdown("Note: GPU layers calculated from model details and VRAM. 0 layers = CPU-only.")
                status_text_settings = gr.Textbox(label="Status", interactive=False)
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
            outputs=[user_input, load_btn, status_text_settings],  # Updated outputs
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

        repeat_penalty_dropdown.change(
            fn=lambda x: utility.update_setting("repeat_penalty", x),
            inputs=[repeat_penalty_dropdown],
            outputs=[user_input, load_btn],
            api_name="update_repeat_penalty"
        )

        mlock_checkbox.change(
            fn=lambda x: utility.update_setting("mlock", x),
            inputs=[mlock_checkbox],
            outputs=[user_input, load_btn],
            api_name="update_mlock"
        )

        max_sessions_dropdown.change(
            fn=lambda x: utility.update_setting("max_sessions", x),
            inputs=[max_sessions_dropdown],
            outputs=[user_input, load_btn],
            api_name="update_max_sessions"
        ).then(
            fn=update_session_tabs,
            inputs=None,
            outputs=[session_tabs],
            api_name="update_session_tabs_4"
        )

        batch_size_dropdown.change(
            fn=lambda x: utility.update_setting("n_batch", x),
            inputs=[batch_size_dropdown],
            outputs=[user_input, load_btn],
            api_name="update_n_batch"
        )

        max_docs_dropdown.change(
            fn=lambda x: utility.update_setting("rag_max_docs", x),
            inputs=[max_docs_dropdown],
            outputs=[user_input, load_btn],
            api_name="update_rag_max_docs"
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

    demo.launch()

if __name__ == "__main__":
    launch_interface()