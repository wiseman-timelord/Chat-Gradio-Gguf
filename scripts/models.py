# Script: `.\scripts\models.py`

# Imports...
import time, re
from pathlib import Path
import gradio as gr
from scripts.prompts import get_system_message, get_reasoning_instruction, get_tot_instruction
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
import scripts.temporary as temporary  # Import module instead of specific variables
from scripts.prompts import prompt_templates
from scripts.temporary import (
    CONTEXT_SIZE, GPU_LAYERS, BATCH_SIZE, LLAMA_CLI_PATH, BACKEND_TYPE, VRAM_SIZE,
    DYNAMIC_GPU_LAYERS, MMAP, current_model_settings, llm,
    MODEL_NAME, REPEAT_PENALTY, TEMPERATURE, MODELS_LOADED
)

# Classes...
class ContextInjector:
    def __init__(self):
        self.vectorstores = {}
        self.current_vectorstore = None
        self.current_mode = None
        self.session_vectorstore = None
        print("VectorStore Injector initialized.")

    def set_session_vectorstore(self, vectorstore):
        self.session_vectorstore = vectorstore
        if vectorstore:
            print("Session-specific vectorstore set.")
        else:
            print("Session-specific vectorstore cleared.")

    def load_session_vectorstore(self, session_id):
        vs_path = Path("data/vectors") / f"session_{session_id}"  # Updated path
        if vs_path.exists():
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.session_vectorstore = FAISS.load_local(
                str(vs_path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Loaded session vectorstore for session {session_id} from {vs_path}.")
        else:
            self.session_vectorstore = None
            print(f"No session vectorstore found for session {session_id} at {vs_path}.")

context_injector = ContextInjector()

# Functions...
def get_model_size(model_path: str) -> float:
    return Path(model_path).stat().st_size / (1024 * 1024)

def clean_content(role, content):
    """Remove prefixes from session_log content for model input."""
    if role == 'user':
        return content.replace("User:\n", "", 1).strip()
    return content.strip()

def set_cpu_affinity():
    from scripts import utility
    cpu_only_backends = ["CPU Only - AVX2", "CPU Only - AVX512", "CPU Only - NoAVX", "CPU Only - OpenBLAS"]
    if temporary.BACKEND_TYPE in cpu_only_backends and temporary.SELECTED_CPU:
        cpus = utility.get_cpu_info()
        selected_cpu = next((cpu for cpu in cpus if cpu["label"] == temporary.SELECTED_CPU), None)
        if selected_cpu:
            try:
                p = psutil.Process()
                p.cpu_affinity(selected_cpu["core_range"])
                print(f"Set CPU affinity to {selected_cpu['label']}")
            except Exception as e:
                print(f"Failed to set CPU affinity: {e}")

def get_available_models():
    model_dir = Path(temporary.MODEL_FOLDER)
    print(f"Scanning directory: {model_dir}")
    files = list(model_dir.glob("*.gguf"))
    models = [f.name for f in files if f.is_file()]
    if models:
        choices = models
    else:
        choices = ["Browse_for_model_folder..."]
    print(f"Models Found: {choices}")
    return choices

def get_model_settings(model_name):
    from .temporary import category_keywords, handling_keywords
    model_name_lower = model_name.lower()
    category = "chat"
    for cat, keywords in category_keywords.items():
        if any(keyword in model_name_lower for keyword in keywords):
            category = cat
            break
    is_uncensored = any(keyword in model_name_lower for keyword in handling_keywords["uncensored"])
    is_reasoning = any(keyword in model_name_lower for keyword in handling_keywords["reasoning"])
    return {
        "category": category,
        "is_uncensored": is_uncensored,
        "is_reasoning": is_reasoning,
        "detected_keywords": []
    }

def determine_operation_mode(model_name):
    if model_name == "Browse_for_model_folder...":
        return "Select models to enable mode detection.", "Select models to enable mode detection."
    settings = get_model_settings(model_name)
    mode = settings["category"].capitalize()
    return mode, mode

def calculate_gpu_layers(models, available_vram):
    from math import floor
    if not models or available_vram <= 0:
        return {model: 0 for model in models}
    total_size = sum(get_model_size(Path(MODEL_FOLDER) / model) for model in models if model != "Browse_for_model_folder...")
    if total_size == 0:
        return {model: 0 for model in models}
    vram_allocations = {
        model: (get_model_size(Path(MODEL_FOLDER) / model) / total_size) * available_vram
        for model in models if model != "Browse_for_model_folder..."
    }
    gpu_layers = {}
    for model in models:
        if model == "Browse_for_model_folder...":
            gpu_layers[model] = 0
            continue
        model_path = Path(MODEL_FOLDER) / model
        num_layers = get_model_layers(str(model_path))
        if num_layers == 0:
            gpu_layers[model] = 0
            continue
        model_file_size = get_model_size(str(model_path))
        adjusted_model_size = model_file_size * 1.1
        layer_size = adjusted_model_size / num_layers if num_layers > 0 else 0
        max_layers = floor(vram_allocations[model] / layer_size) if layer_size > 0 else 0
        gpu_layers[model] = min(max_layers, num_layers) if DYNAMIC_GPU_LAYERS else num_layers
    return gpu_layers

def get_model_layers(model_path: str) -> int:
    """Determine the number of layers in a GGUF model.

    Args:
        model_path (str): Path to the model file.

    Returns:
        int: Number of layers, or 0 if undetermined.
    """
    try:
        from llama_cpp import Llama
        import re
        import io
        from contextlib import redirect_stdout, redirect_stderr
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            model = Llama(model_path=model_path, verbose=True, n_ctx=8, n_batch=1, n_gpu_layers=0)
            del model
        output = output_buffer.getvalue()
        patterns = [
            r'block_count\s*=\s*(\d+)',
            r'n_layer\s*=\s*(\d+)',
            r'- kv\s+\d+:\s+.*\.block_count\s+u\d+\s+=\s+(\d+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                num_layers = int(match.group(1))
                print(f"Found layers with pattern '{pattern}': {num_layers}")
                return num_layers
        print("Could not determine layer count from output.")
        return 0
    except Exception as e:
        print(f"Error reading model layers: {e}")
        return 0

def get_model_metadata(model_path: str) -> dict:
    try:
        import re
        import io
        from contextlib import redirect_stdout, redirect_stderr
        from llama_cpp import Llama
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            model = Llama(
                model_path=model_path,
                verbose=True,
                n_ctx=8,  # Changed from CONTEXT_SIZE=8
                n_batch=1,
                n_gpu_layers=0
            )
            del model
        output = output_buffer.getvalue()
        metadata = {}
        for line in output.splitlines():
            if line.startswith("llama_model_loader: - kv"):
                match = re.search(r'llama_model_loader: - kv\s+\d+:\s+([\w\.]+)\s+(\w+(?:\[.*?\])?)\s+=\s+(.*)', line)
                if match:
                    key = match.group(1)
                    type_str = match.group(2).split('[')[0]
                    value_str = match.group(3).strip()
                    if type_str == 'u32':
                        value = int(value_str)
                    elif type_str == 'f32':
                        value = float(value_str)
                    elif type_str == 'str':
                        value = value_str
                    elif type_str == 'bool':
                        value = value_str.lower() == 'true'
                    else:
                        value = value_str
                    metadata[key] = value
        return metadata
    except Exception as e:
        print(f"Debug: Error reading model metadata: {e}")
        return {}

def inspect_model(model_dir, model_name, vram_size):
    from scripts.utility import save_config
    if model_name == "Browse_for_model_folder...":
        return "Select a model to inspect."
    model_path = Path(model_dir) / model_name
    if not model_path.exists():
        return f"Model file '{model_path}' not found."
    save_config()
    try:
        metadata = get_model_metadata(str(model_path))
        architecture = metadata.get('general.architecture', 'unknown')
        params_str = metadata.get('general.size_label', 'Unknown')
        layers = metadata.get(f'{architecture}.block_count', 'Unknown')
        max_ctx = metadata.get(f'{architecture}.context_length', 'Unknown')
        embed = metadata.get(f'{architecture}.embedding_length', 'Unknown')
        model_size_mb = get_model_size(str(model_path))
        model_size_gb = model_size_mb / 1024
        if isinstance(layers, int) and layers > 0:
            fit_layers = calculate_single_model_gpu_layers_with_layers(
                str(model_path), vram_size, layers, DYNAMIC_GPU_LAYERS
            )
        else:
            fit_layers = "Unknown"
        author = metadata.get('general.organization', 'Unknown')
        return (
            f"Results: Params = {params_str}, "
            f"Fit/Layers = {fit_layers}/{layers}, "
            f"Size = {model_size_gb:.2f} GB, "
            f"Max Ctx = {max_ctx}, "
            f"Embed = {embed}, "
            f"Author = {author}"
        )
    except Exception as e:
        return f"Error inspecting model: {str(e)}"

def load_models(model_folder, model, vram_size):
    from scripts.temporary import CONTEXT_SIZE, BATCH_SIZE, MMAP, DYNAMIC_GPU_LAYERS
    from scripts.utility import save_config
    from pathlib import Path
    import traceback

    save_config()

    if model in ["Browse_for_model_folder...", "No models found"]:
        temporary.MODELS_LOADED = False
        return "Select a model to load.", False

    model_path = Path(model_folder) / model
    if not model_path.exists():
        temporary.MODELS_LOADED = False
        return f"Error: Model file '{model_path}' not found.", False

    num_layers = get_model_layers(str(model_path))
    if num_layers <= 0:
        temporary.MODELS_LOADED = False
        return f"Error: Could not determine layer count for model '{model}'.", False

    temporary.GPU_LAYERS = calculate_single_model_gpu_layers_with_layers(
        str(model_path), vram_size, num_layers, DYNAMIC_GPU_LAYERS
    )

    try:
        from llama_cpp import Llama
    except ImportError:
        temporary.MODELS_LOADED = False
        return "Error: llama-cpp-python not installed. Python bindings are required.", False

    try:
        if temporary.MODELS_LOADED:
            unload_models()

        print(f"Debug: Loading model '{model}' from '{model_folder}' with Python bindings, mlock=True")
        temporary.llm = Llama(
            model_path=str(model_path),
            n_ctx=CONTEXT_SIZE,
            n_gpu_layers=temporary.GPU_LAYERS,
            n_batch=BATCH_SIZE,
            mmap=MMAP,
            mlock=True,
            verbose=True
        )

        test_output = temporary.llm.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=5,
            stream=False
        )
        print(f"Debug: Test inference successful: {test_output}")

        temporary.MODELS_LOADED = True
        temporary.MODEL_NAME = model
        status = f"Model '{model}' loaded successfully. GPU layers: {temporary.GPU_LAYERS}/{num_layers}"
        return status, True

    except Exception as e:
        error_msg = f"Error loading model: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        temporary.MODELS_LOADED = False
        return error_msg, False

def calculate_single_model_gpu_layers_with_layers(model_path: str, available_vram: int, num_layers: int, dynamic_gpu_layers: bool = True) -> int:
    from math import floor
    if num_layers <= 0 or available_vram <= 0:
        print("Debug: Invalid input (layers or VRAM), returning 0 layers")
        return 0
    model_file_size = get_model_size(model_path)
    print(f"Debug: Model size = {model_file_size:.2f} MB, Layers = {num_layers}, VRAM = {available_vram} MB")
    adjusted_model_size = model_file_size * 1.1
    layer_size = adjusted_model_size / num_layers
    print(f"Debug: Adjusted size = {adjusted_model_size:.2f} MB, Layer size = {layer_size:.2f} MB")
    max_layers = floor(available_vram / layer_size)
    result = min(max_layers, num_layers) if dynamic_gpu_layers else num_layers
    print(f"Debug: Max layers with VRAM = {max_layers}, Final result = {result}")
    return result

def unload_models():
    global llm, MODELS_LOADED, MODEL_NAME
    import gc
    if MODELS_LOADED:
        del llm
        gc.collect()
        MODELS_LOADED = False
        print(f"Model {MODEL_NAME} unloaded.")
        return "Model unloaded successfully."
    print("Warning: No model was loaded to unload.")
    return "No model loaded to unload."

def clean_content(role, content):
    """Remove prefixes from session_log content for model input."""
    if role == 'user':
        return content.replace("User:\n", "", 1).strip()
    elif role == 'assistant':
        return content.replace("AI-Chat-Response:\n", "", 1).strip()
    return content

def generate_summary(text):
    summary_prompt = (
        "Summarize the following response in under 256 characters, focusing on critical information and conclusions:\n\n"
        f"{text}"
    )
    response = temporary.llm.create_chat_completion(
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=512,  # Batch size for efficiency
        temperature=0.5,
        stream=False
    )
    summary = response['choices'][0]['message']['content'].strip()
    if len(summary) > 256:
        summary = summary[:253] + "..."  # Truncate with ellipsis
    return summary

# aSync Functions...
async def get_response_stream(session_log, mode, settings, disable_think=False, rp_settings=None, tot_enabled=False, web_search_enabled=False, search_results=None):
    if not temporary.MODELS_LOADED or temporary.llm is None:
        yield "Error: No model loaded. Please load a model first."
        return

    messages = []
    for msg in session_log[:-1]:
        role = msg['role']
        content = clean_content(role, msg['content'])
        messages.append({"role": role, "content": content})

    system_message = get_system_message(mode, settings.get("is_uncensored", False), rp_settings)
    messages.insert(0, {"role": "system", "content": system_message})

    if web_search_enabled and mode in ["chat", "code"]:
        web_prompt = "Use the following web search results to inform your response if relevant:\n" + str(search_results) if search_results else "No web search results were found. Proceed with your best response."
        messages.insert(1, {"role": "system", "content": web_prompt})

    if settings.get("is_reasoning", False) and not disable_think and mode == "chat":
        messages[-1]["content"] += get_reasoning_instruction()
    if tot_enabled and mode == "chat":
        messages[-1]["content"] += get_tot_instruction()

    try:
        response_stream = temporary.llm.create_chat_completion(
            messages=messages,
            max_tokens=1024,
            temperature=temporary.TEMPERATURE,
            repeat_penalty=temporary.REPEAT_PENALTY,
            stream=True
        )
        for chunk in response_stream:
            if 'choices' in chunk and chunk['choices']:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    chunk_content = delta['content']
                    print(chunk_content, end='', flush=True)
                    yield chunk_content
    except Exception as e:
        yield f"Error generating response: {str(e)}"