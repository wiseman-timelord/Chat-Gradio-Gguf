# Script: `.\scripts\interface.py`

import gradio as gr
import re
from pathlib import Path
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
from scripts.temporary import (
    USER_COLOR, THINK_COLOR, RESPONSE_COLOR, SEPARATOR, MID_SEPARATOR,
    MODEL_LOADED, ACTIVE_SESSION, ALLOWED_EXTENSIONS, CTX_OPTIONS,
    MODEL_PATH, N_GPU_LAYERS, N_CTX, TEMPERATURE, TEMP_OPTIONS,
    VRAM_SIZE, SELECTED_GPU, HISTORY_OPTIONS, MAX_SESSIONS
)
from scripts import utility  # Updated import for utility functions

def format_response(output: str) -> str:
    formatted = []
    code_blocks = re.findall(r'```(\w+)?\n(.*?)```', output, re.DOTALL)
    for lang, code in code_blocks:
        lexer = get_lexer_by_name(lang, stripall=True)
        formatted_code = highlight(code, lexer, HtmlFormatter())
        output = output.replace(f'```{lang}\n{code}```', formatted_code)
    return f'<span style="color: {RESPONSE_COLOR}">{output}</span>'

def chat_interface(message: str, history):
    history.append((f'<span style="color: {USER_COLOR}">{message}</span>', ""))
    try:
        full_response = ""
        for token in get_streaming_response(message):
            full_response += token
            history[-1] = (history[-1][0], format_response(full_response))
            yield history
        if len(history) == 1:
            summary_prompt = f"Summarize the following conversation in exactly three words:\nUser: {message}\nAssistant: {full_response}"
            globals()["session_label"] = get_response(summary_prompt).strip()
            utility.save_session_history(history)  # Save with label
    except Exception as e:
        history[-1] = (history[-1][0], f'<span style="color: red;">Error: {str(e)}</span>')
        yield history

def get_available_models():
    model_dir = Path("models")
    return [f.name for f in model_dir.glob("*.gguf")]

def change_model(model_name):
    """Switch to a different model."""
    from scripts.temporary import MODEL_PATH
    globals()["MODEL_PATH"] = f"data/models/{model_name}"
    unload_model()
    load_model()
    return gr.Textbox(interactive=True), gr.Button(interactive=False)

def web_search_trigger(query):
    """Trigger a web search."""
    return utility.web_search(query)  # Updated to use utility

def process_uploaded_files(files):
    """Process uploaded files."""
    files_dir = Path("files")
    files_dir.mkdir(exist_ok=True)
    for file in files:
        original_name = Path(file.name).name
        dest_path = files_dir / original_name
        with open(file.name, 'rb') as src, open(dest_path, 'wb') as dst:
            dst.write(src.read())
    docs = utility.load_and_chunk_documents(files_dir)  # Updated to use utility
    if docs:
        utility.create_vectorstore(docs)  # Updated to use utility
    from scripts.models import reload_vectorstore
    reload_vectorstore("general_knowledge")
    return "Files uploaded and processed successfully"

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
    return utility.get_saved_sessions()  # Updated to use utility

def start_new_session():
    global message_history, session_label
    message_history = []
    session_label = ""
    return gr.Chatbot(value=[]), gr.Textbox(value="")

def load_session(session_file):
    if session_file:
        label, history = utility.load_session_history(Path(HISTORY_DIR) / session_file)  # Updated to use utility
        global session_label
        session_label = label
        return history
    return []

def launch_interface():
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(height=500)  # Increased height for better visibility
        input_box = gr.Textbox(interactive=False, placeholder="Load a model to start chatting...")

        with gr.Row():
            with gr.Column(scale=1):  # Left sidebar for session tabs
                session_tabs = gr.Tabs()
                with session_tabs:
                    with gr.Tab("Start New Session", id="new_session"):
                        pass  # Placeholder for new session tab
                    # Dynamic session tabs will be updated via a function

            with gr.Column(scale=4):  # Main chat area
                with gr.Tab("Chat"):
                    control_row = create_control_row()
                    web_btn = gr.Button("Web Search")
                    web_output = gr.Textbox(label="Web Results", interactive=False)
                    upload_output = gr.Textbox(label="Upload Status", interactive=False)
                    save_session_btn = gr.Button("Save Session")
                    save_session_output = gr.Textbox(label="Save Status", interactive=False)
                    session_controls, load_btn, unload_btn, shutdown_btn = create_session_controls()

        with gr.Tab("Settings"):
            model_dropdown = gr.Dropdown(
                choices=get_available_models(),
                label="Select Model",
                value=MODEL_PATH.split('/')[-1]
            )
            temperature_dropdown = gr.Dropdown(
                choices=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],  # Per readme.md
                label="Temperature",
                value=TEMPERATURE
            )
            n_ctx_dropdown = gr.Dropdown(
                choices=CTX_OPTIONS,
                label="Context Window",
                value=N_CTX
            )
            vram_dropdown = gr.Dropdown(
                choices=[1024, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 16384, 20480, 24576, 32768],  # Per readme.md
                label="VRAM Size (MB)",
                value=VRAM_SIZE
            )
            gpu_dropdown = gr.Dropdown(
                choices=utility.get_available_gpus(),
                label="Select GPU",
                value=SELECTED_GPU
            )
            gr.Markdown("Note: For a 14B model on an 8GB GPU, try 20-30 layers.")
            save_btn = gr.Button("Save Settings")

        # Event handlers
        send_btn = control_row.children[0]
        upload_btn = control_row.children[1]

        def update_session_tabs():
            sessions = utility.get_saved_sessions()
            tabs = [gr.Tab("Start New Session", id="new_session")]
            for session in sessions[:MAX_SESSIONS]:
                label, _ = utility.load_session_history(Path(HISTORY_DIR) / session)
                tabs.append(gr.Tab(label, id=session))
            return gr.Tabs.update(children=tabs)

        send_btn.click(
            fn=chat_interface,
            inputs=[input_box, chatbot],
            outputs=[chatbot]
        ).then(
            fn=update_session_tabs,
            inputs=None,
            outputs=[session_tabs]
        )
        upload_btn.upload(
            fn=process_uploaded_files,
            inputs=[upload_btn],
            outputs=[upload_output]
        )
        web_btn.click(
            fn=web_search_trigger,
            inputs=[input_box],
            outputs=[web_output]
        )
        save_session_btn.click(
            fn=lambda history: utility.save_session_history(history),
            inputs=[chatbot],
            outputs=[save_session_output]
        ).then(
            fn=update_session_tabs,
            inputs=None,
            outputs=[session_tabs]
        )
        load_btn.click(
            fn=load_model,
            outputs=[input_box, load_btn]
        )
        unload_btn.click(
            fn=unload_model,
            outputs=[input_box, load_btn]
        )
        shutdown_btn.click(
            fn=lambda: exit(0),
            inputs=None,
            outputs=None
        )
        model_dropdown.change(
            fn=change_model,
            inputs=[model_dropdown],
            outputs=[input_box, load_btn]
        )
        temperature_dropdown.change(
            fn=lambda x: utility.update_setting("temperature", x),
            inputs=[temperature_dropdown],
            outputs=[input_box, load_btn]
        )
        n_ctx_dropdown.change(
            fn=lambda x: utility.update_setting("n_ctx", x),
            inputs=[n_ctx_dropdown],
            outputs=[input_box, load_btn]
        )
        vram_dropdown.change(
            fn=lambda x: utility.update_setting("vram_size", x),
            inputs=[vram_dropdown],
            outputs=[input_box, load_btn]
        )
        gpu_dropdown.change(
            fn=lambda x: utility.update_setting("selected_gpu", x),
            inputs=[gpu_dropdown],
            outputs=[input_box, load_btn]
        )
        save_btn.click(
            fn=utility.save_config,
            inputs=None,
            outputs=None
        )
        session_tabs.select(
            fn=lambda tab_id: load_session(tab_id) if tab_id != "new_session" else start_new_session(),
            inputs=[session_tabs],
            outputs=[chatbot, input_box]
        ).then(
            fn=update_session_tabs,
            inputs=None,
            outputs=[session_tabs]
        )

    demo.launch()

def load_model():
    from scripts.models import initialize_model
    initialize_model(None)
    return gr.Textbox(interactive=True), gr.Button(interactive=False)

def unload_model():
    from scripts.models import unload_model
    unload_model()
    return gr.Textbox(interactive=False), gr.Button(interactive=True)

if __name__ == "__main__":
    launch_interface()