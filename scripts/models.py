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
    DYNAMIC_GPU_LAYERS, MMAP, current_model_settings, handling_keywords, llm,
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
def get_model_metadata(model_path: str) -> dict:
    """
    Retrieve metadata from a GGUF model, including the number of layers.
    
    Args:
        model_path (str): Path to the GGUF model file.
    
    Returns:
        dict: Metadata with 'layers' key, or empty dict on failure.
    """
    try:
        from llama_cpp import Llama
        # First attempt: Use model.metadata (more reliable than verbose output)
        model = Llama(
            model_path=model_path,
            n_ctx=512,  # Minimal context for metadata
            n_batch=1,
            n_gpu_layers=0,
            verbose=False  # Avoid verbose output unless needed
        )
        metadata = model.metadata  # Direct access to GGUF metadata
        del model
        print(f"Debug: Metadata keys for '{model_path}': {list(metadata.keys())}")

        # Extract architecture and layers
        architecture = metadata.get('general.architecture', 'unknown')
        print(f"Debug: Detected architecture: {architecture}")
        layers = metadata.get(f'{architecture}.block_count', 0)

        # Fallback: Search for alternative layer count keys
        if layers == 0:
            for key in metadata:
                if 'block_count' in key or 'layer_count' in key:
                    layers = metadata[key]
                    print(f"Debug: Found layers ({layers}) in key '{key}'")
                    break
            else:
                layers = 0

        metadata['layers'] = layers
        if layers == 0:
            print(f"Warning: Could not determine layer count for '{model_path}'. Metadata: {metadata}")
        else:
            print(f"Debug: Found {layers} layers for '{model_path}'")
        return metadata

    except AttributeError:
        # Fallback to verbose output if metadata attribute is unavailable
        print("Debug: model.metadata not available, falling back to verbose output")
        try:
            import re
            import io
            from contextlib import redirect_stdout, redirect_stderr
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
                model = Llama(
                    model_path=model_path,
                    n_ctx=512,
                    n_batch=1,
                    n_gpu_layers=0,
                    verbose=True
                )
                del model
            output = output_buffer.getvalue()
            print(f"Debug: Raw output for '{model_path}':\n{output}")
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

            architecture = metadata.get('general.architecture', 'unknown')
            layers = metadata.get(f'{architecture}.block_count', 0)
            if layers == 0:
                layers = next((value for key, value in metadata.items() if 'block_count' in key or 'layer_count' in key), 0)
            metadata['layers'] = layers
            if layers == 0:
                print(f"Warning: Could not determine layer count. Metadata keys: {list(metadata.keys())}")
            return metadata
        except Exception as e:
            print(f"Error reading metadata (verbose fallback): {e}")
            return {}
    except Exception as e:
        print(f"Error reading model metadata for '{model_path}': {e}")
        return {}

def get_model_layers(model_path: str) -> int:
    """
    Get the number of layers for a GGUF model.
    
    Args:
        model_path (str): Path to the GGUF model file.
    
    Returns:
        int: Number of layers, or 0 if not determined.
    """
    metadata = get_model_metadata(model_path)
    layers = metadata.get('layers', 0)
    return int(layers)  # Ensure conversion to integer

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
    model_name_lower = model_name.lower()
    is_uncensored = any(keyword in model_name_lower for keyword in handling_keywords["uncensored"])
    is_reasoning = any(keyword in model_name_lower for keyword in handling_keywords["reasoning"])
    is_nsfw = any(keyword in model_name_lower for keyword in handling_keywords["nsfw"])
    is_code = any(keyword in model_name_lower for keyword in handling_keywords["code"])
    is_roleplay = any(keyword in model_name_lower for keyword in handling_keywords["roleplay"])
    return {
        "category": "chat",
        "is_uncensored": is_uncensored,
        "is_reasoning": is_reasoning,
        "is_nsfw": is_nsfw,
        "is_code": is_code,
        "is_roleplay": is_roleplay,
        "detected_keywords": [kw for kw in handling_keywords if any(k in model_name_lower for k in handling_keywords[kw])]
    }

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

def load_models(model_folder, model, vram_size, llm_state, models_loaded_state):
    from scripts.temporary import CONTEXT_SIZE, BATCH_SIZE, MMAP, DYNAMIC_GPU_LAYERS
    from scripts.utility import save_config
    from pathlib import Path
    import traceback

    save_config()

    if model in ["Browse_for_model_folder...", "No models found"]:
        return "Select a model to load.", False, llm_state, models_loaded_state

    model_path = Path(model_folder) / model
    if not model_path.exists():
        return f"Error: Model file '{model_path}' not found.", False, llm_state, models_loaded_state

    num_layers = get_model_layers(str(model_path))
    if num_layers <= 0:
        return f"Error: Could not determine layer count for model '{model}'.", False, llm_state, models_loaded_state

    temporary.GPU_LAYERS = calculate_single_model_gpu_layers_with_layers(
        str(model_path), vram_size, num_layers, DYNAMIC_GPU_LAYERS
    )

    try:
        from llama_cpp import Llama
    except ImportError:
        return "Error: llama-cpp-python not installed. Python bindings are required.", False, llm_state, models_loaded_state

    try:
        if models_loaded_state:
            unload_models(llm_state, models_loaded_state)

        print(f"Debug: Loading model '{model}' from '{model_folder}' with Python bindings")
        new_llm = Llama(
            model_path=str(model_path),
            n_ctx=temporary.CONTEXT_SIZE,
            n_gpu_layers=temporary.GPU_LAYERS,
            n_batch=temporary.BATCH_SIZE,
            mmap=temporary.MMAP,
            mlock=temporary.MLOCK,
            verbose=True
        )

        test_output = new_llm.create_chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=temporary.BATCH_SIZE,
            stream=False
        )
        print(f"Debug: Test inference successful: {test_output}")

        temporary.MODEL_NAME = model  # Keep for settings
        status = f"Model '{model}' loaded successfully. GPU layers: {temporary.GPU_LAYERS}/{num_layers}"
        return status, True, new_llm, True

    except Exception as e:
        error_msg = f"Error loading model: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return error_msg, False, None, False

def calculate_single_model_gpu_layers_with_layers(model_path: str, available_vram: int, num_layers: int, dynamic_gpu_layers: bool = True) -> int:
    from math import floor
    if num_layers <= 0 or available_vram <= 0:
        print("Debug: Invalid input (layers or VRAM), returning 0 layers")
        return 0
    model_file_size = get_model_size(model_path)
    print(f"Debug: Model size = {model_file_size:.2f} MB, Layers = {num_layers}, VRAM = {available_vram} MB")
    adjusted_model_size = model_file_size * 1.125
    layer_size = adjusted_model_size / num_layers
    print(f"Debug: Adjusted size = {adjusted_model_size:.2f} MB, Layer size = {layer_size:.2f} MB")
    max_layers = floor(available_vram / layer_size)
    result = min(max_layers, num_layers) if dynamic_gpu_layers else num_layers
    print(f"Debug: Max layers with VRAM = {max_layers}, Final result = {result}")
    return result

def unload_models(llm_state, models_loaded_state):
    import gc
    if models_loaded_state:
        del llm_state
        gc.collect()
        print(f"Model {temporary.MODEL_NAME} unloaded.")
        return "Model unloaded successfully.", None, False
    print("Warning: No model was loaded to unload.")
    return "No model loaded to unload.", llm_state, models_loaded_state

def generate_summary(text):
    summary_prompt = (
        "Summarize the following response in under 256 characters, focusing on critical information and conclusions:\n\n"
        f"{text}"
    )
    response = temporary.llm.create_chat_completion(
        messages=[{"role": "user", "content": summary_prompt}],
        max_tokens=temporary.BATCH_SIZE,  # Fixed from 512
        temperature=temporary.TEMPERATURE,  # Fixed from 0.5
        stream=False  # Reasonable for summary
    )
    summary = response['choices'][0]['message']['content'].strip()
    if len(summary) > 256:
        summary = summary[:253] + "..."  # Truncate with ellipsis
    return summary


# aSync Functions...
def get_response_stream(session_log, settings, disable_think=False, tot_enabled=False, 
                       web_search_enabled=False, search_results=None, cancel_event=None, 
                       llm_state=None, models_loaded_state=False):
    if not models_loaded_state or llm_state is None:
        yield "Error: No model loaded. Please load a model first."
        return

    print("Debug: Entering get_response_stream")
    print(f"Debug: session_log = {session_log}")

    # Build messages
    messages = []
    system_message = get_system_message(
        is_uncensored=settings.get("is_uncensored", False),
        is_nsfw=settings.get("is_nsfw", False),
        web_search_enabled=web_search_enabled,
        tot_enabled=tot_enabled,
        is_reasoning=settings.get("is_reasoning", False),
        disable_think=disable_think,
        is_roleplay=settings.get("is_roleplay", False)  # Added is_roleplay parameter
    )
    if web_search_enabled and search_results:
        system_message += f"\n\n=== Web Results ===\n{search_results}"
    messages.append({"role": "system", "content": system_message})

    if session_log and len(session_log) >= 2 and session_log[-2]['role'] == 'user':
        user_content = clean_content('user', session_log[-2]['content'])
        if context_injector.session_vectorstore:
            query = user_content
            docs = context_injector.session_vectorstore.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            user_content = f"{user_content}\n\nRelevant context from attached documents:\n{context}"
        messages.append({"role": "user", "content": user_content})
    else:
        print("Debug: No valid user message in session_log")
        yield "Error: No user input to process."
        return

    print("\n" + "="*40 + " FULL PROMPT " + "="*40)
    for msg in messages:
        print(f"{msg['role'].upper()}:\n{msg['content']}\n")
    print("="*93 + "\n")

    try:
        print("Debug: Calling llm_state.create_chat_completion")
        response_stream = llm_state.create_chat_completion(
            messages=messages,
            max_tokens=BATCH_SIZE,
            temperature=TEMPERATURE,
            repeat_penalty=REPEAT_PENALTY,
            stream=True
        )
        
        buffer = ""
        has_content = False
        in_thinking_phase = True
        last_pos = 0  # Track the last processed position for periods in thinking phase
        sentence_endings = ['.', '!', '?']

        for chunk in response_stream:
            if cancel_event and cancel_event.is_set():
                yield "<CANCELLED>"
                return
            if 'choices' in chunk and chunk['choices']:
                content = chunk['choices'][0].get('delta', {}).get('content', '')
                if content:
                    has_content = True
                    buffer += content

                    if in_thinking_phase:
                        if "</think>" in buffer:
                            in_thinking_phase = False
                            parts = buffer.split("</think>", 1)
                            buffer = parts[1].strip()  # Keep only post-thinking content
                            yield "<THINKING_DONE>"
                            print("Debug: Thinking phase ended")
                        else:
                            # Check for periods to yield progress
                            while True:
                                period_pos = buffer.find('.', last_pos)
                                if period_pos != -1 and (period_pos + 1 == len(buffer) or buffer[period_pos + 1] in [' ', '\n']):
                                    yield "<THINKING_PROGRESS>"
                                    last_pos = period_pos + 1
                                    print(f"Debug: Yielded <THINKING_PROGRESS> at period position {period_pos}")
                                else:
                                    break
                    else:
                        # Streaming phase: yield complete sentences
                        while True:
                            # Find the earliest sentence ending followed by space, newline, or end
                            sentence_end_pos = -1
                            for ending in sentence_endings:
                                pos = buffer.find(ending)
                                if pos != -1:
                                    # Check if the ending marks a complete sentence
                                    if pos + 1 < len(buffer) and buffer[pos + 1] in [' ', '\n'] or pos + 1 == len(buffer):
                                        if sentence_end_pos == -1 or pos < sentence_end_pos:
                                            sentence_end_pos = pos
                                    elif pos + 1 == len(buffer):  # End of buffer, incomplete sentence
                                        break

                            if sentence_end_pos != -1:
                                sentence = buffer[:sentence_end_pos + 1].strip()
                                buffer = buffer[sentence_end_pos + 1:].strip()
                                if sentence:
                                    yield sentence
                                    print(f"Debug: Yielded streaming sentence: {sentence}")
                            else:
                                break  # No complete sentence found yet

        # Yield any remaining complete sentence or final buffer
        if buffer.strip():
            if in_thinking_phase:
                yield "<THINKING_DONE>"  # Treat remaining buffer as end of thinking if not closed
                print("Debug: Final thinking buffer processed")
            else:
                yield buffer.strip()  # Yield final content as-is
                print(f"Debug: Yielded final streaming buffer: {buffer.strip()}")
        elif not has_content:
            print("Debug: Model generated no content")
            yield "Error: Model generated an empty response."

    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        print(f"Debug: {error_msg}")
        yield error_msg