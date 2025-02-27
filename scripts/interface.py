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
        gr.Textbox.update(interactive=False, placeholder="Load a model to start chatting..."),
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
                        user_input = gr.Textbox(label="User Input", interactive=False, placeholder="Load a model to start chatting...")
                        status_text = gr.Textbox(label="Status", interactive=False, value="Select and load a model on Configuration page")
                        
                        # Buttons and switch row
                        send_btn = gr.Button("Send Input", variant="primary")
                        edit_previous_btn = gr.Button("Edit Previous")
                        attach_files_btn = gr.UploadButton("Attach Files", file_types=list(ALLOWED_EXTENSIONS), file_count="multiple")
                        web_search_switch = gr.Checkbox(label="Web-Search", value=False)
                        buttons_row = gr.Row(send_btn, edit_previous_btn, attach_files_btn, web_search_switch)

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
                
                # Buttons row
                load_btn = gr.Button("Load Model", variant="secondary")
                unload_btn = gr.Button("Unload Model", variant="secondary")
                save_settings_btn = gr.Button("Save Settings")
                config_buttons_row = gr.Row(load_btn, unload_btn, save_settings_btn)

        # Helper function for Edit Previous (assumed defined elsewhere)
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

        # Event handlers (simplified for brevity, assumed defined elsewhere)
        edit_previous_btn.click(fn=edit_previous, inputs=[session_log, user_input], outputs=[session_log, user_input])
        send_btn.click(fn=chat_interface, inputs=[user_input, session_log], outputs=[session_log, status_text])
        attach_files_btn.upload(fn=process_uploaded_files, inputs=[attach_files_btn], outputs=[status_text])
        start_new_session_btn.click(fn=start_new_session, outputs=[session_log, user_input])
        load_btn.click(fn=load_model, outputs=[user_input, load_btn, status_text, status_text_settings])
        unload_btn.click(fn=unload_model_ui, outputs=[user_input, load_btn, status_text, status_text_settings])
        save_settings_btn.click(fn=utility.save_config, outputs=[status_text_settings])

    demo.launch()

if __name__ == "__main__":
    launch_interface()