# Script: `.\scripts\models.py`

# Imports...
import time, subprocess
from llama_cpp import Llama
from pathlib import Path
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from . import temporary
from scripts.temporary import (
    N_CTX, N_GPU_LAYERS, N_BATCH, USE_PYTHON_BINDINGS,
    LLAMA_CLI_PATH, BACKEND_TYPE, VRAM_SIZE,
    DYNAMIC_GPU_LAYERS, MMAP, MLOCK, current_model_settings,
    prompt_templates, llm, MODEL_NAME, MODEL_FOLDER, REPEAT_PENALTY,
    TEMPERATURE, MODELS_LOADED
)


# Classes...
class ContextInjector:
    def __init__(self):
        self.vectorstores = {}  # Retain for potential future use, but not mode-specific by default
        self.current_vectorstore = None
        self.current_mode = None
        self.session_vectorstore = None  # Primary vectorstore for session-specific RAG
        print("ContextInjector initialized with session-specific vectorstore support.")

    def set_session_vectorstore(self, vectorstore):
        self.session_vectorstore = vectorstore
        if vectorstore:
            print("Session-specific vectorstore set successfully.")
        else:
            print("Session-specific vectorstore cleared.")

    def load_session_vectorstore(self, session_id):
        """Load a session-specific vectorstore based on session_id."""
        vs_path = Path("data/vectors/session") / f"session_{session_id}"
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

    def inject_context(self, prompt: str) -> str:
        """Inject context from session-specific vectorstore only."""
        if self.session_vectorstore is None:
            print("No session vectorstore available; returning prompt unchanged.")
            return prompt
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = embedding_model.encode([prompt])[0]
        docs = self.session_vectorstore.similarity_search_by_vector(query_embedding, k=2)
        context = "\n".join([doc.page_content for doc in docs])
        print(f"Injected context from session vectorstore: {context[:50]}...")
        return f"Relevant information:\n{context}\n\nQuery: {prompt}"

context_injector = ContextInjector()

# Functions...
def reload_vectorstore(name: str):
    context_injector.load_vectorstore(name)
    return "Knowledge updated"

def get_model_size(model_path: str) -> float:
    """Get model file size in MB."""
    return Path(model_path).stat().st_size / (1024 * 1024)

def set_cpu_affinity():
    """Set processor affinity to the selected CPU's cores for CPU-only backends."""
    from scripts import utility  # Delayed import to avoid circularity
    cpu_only_backends = [
        "CPU Only - AVX2", "CPU Only - AVX512", "CPU Only - NoAVX", "CPU Only - OpenBLAS"
    ]
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
                

def get_llm():
    from scripts.temporary import llm, MODEL_NAME, MODEL_FOLDER, N_CTX, N_GPU_LAYERS, N_BATCH, MMAP, MLOCK
    if MODEL_NAME != "Select_a_model...":
        if llm is None:
            model_path = Path(MODEL_FOLDER) / MODEL_NAME
            llm = Llama(
                model_path=str(model_path),
                n_ctx=N_CTX,
                n_gpu_layers=N_GPU_LAYERS,
                n_batch=N_BATCH,
                mmap=MMAP,
                mlock=MLOCK,
                verbose=False
            )
        return llm
    return None

def get_available_models():
    from .temporary import MODEL_FOLDER
    model_dir = Path(MODEL_FOLDER)
    models = [f.name for f in model_dir.glob("*.gguf") if f.is_file()]
    print(f"Available models in {MODEL_FOLDER}: {models}")
    return models

def get_model_settings(model_name):
    from .temporary import category_keywords, handling_keywords
    print(f"Category keywords: {category_keywords}")
    print(f"Handling keywords: {handling_keywords}")
    model_name_lower = model_name.lower()
    detected_keywords = []
    category = "chat"  # Default category
    # Check category keywords
    for cat, keywords in category_keywords.items():
        for keyword in keywords:
            if keyword in model_name_lower:
                detected_keywords.append(keyword)
                category = cat
                break  # Assuming one category per model
    # Check handling keywords
    is_uncensored = any(keyword in model_name_lower for keyword in handling_keywords["uncensored"])
    if is_uncensored:
        detected_keywords.append("uncensored")
    is_reasoning = any(keyword in model_name_lower for keyword in handling_keywords["reasoning"])
    if is_reasoning:
        detected_keywords.append("reasoning")
    settings = {
        "category": category,
        "is_uncensored": is_uncensored,
        "is_reasoning": is_reasoning,
        "detected_keywords": detected_keywords
    }
    return settings

def determine_operation_mode(model_name):
    if model_name == "Select_a_model...":
        return "Select models to enable mode detection.", "Select models to enable mode detection."
    settings = get_model_settings(model_name)
    mode = settings["category"].capitalize()
    return mode, mode

def calculate_gpu_layers(models, available_vram):
    """
    Calculate the number of layers to load onto the GPU for each model based on available VRAM.

    This function estimates how many layers of each model can be loaded onto the GPU given the available VRAM.
    It uses a proportional allocation of VRAM based on the size of each model and calculates the number of layers
    that can fit into the allocated VRAM for each model.

    Calculation Steps:
    1. Compute the total size of all models to be loaded.
    2. Allocate a portion of the available VRAM to each model proportional to its size.
    3. For each model, estimate the memory size per layer:
       - LayerSize = (ModelFileSize * 1.1875) / NumLayers
       - ModelFileSize is the size of the model file in MB.
       - 1.1875 is a factor to account for additional memory overhead (e.g., weights, buffers).
       - NumLayers is the total number of layers in the model.
    4. Calculate the number of layers that can fit into the allocated VRAM:
       - NumLayersForGpu = floor(AllocatedVRam / LayerSize)
    5. Ensure the number of layers does not exceed the total layers in the model:
       - NumLayersForGpu = min(NumLayersForGpu, NumLayers)

    Args:
        models (list): List of model names to load (e.g., ["model1.gguf", "model2.gguf"]).
        available_vram (int): Available VRAM in MB (e.g., 8192 for 8GB).

    Returns:
        dict: A dictionary mapping each model name to the number of layers to load onto the GPU.
    """
    from math import floor
    from pathlib import Path
    from scripts.temporary import MODEL_FOLDER, DYNAMIC_GPU_LAYERS

    # Handle edge cases
    if not models or available_vram <= 0:
        return {model: 0 for model in models}

    # Step 1: Calculate total size of all selected models
    total_size = sum(get_model_size(Path(MODEL_FOLDER) / model) for model in models if model != "Select_a_model...")
    if total_size == 0:
        return {model: 0 for model in models}

    # Step 2: Allocate VRAM to each model proportional to its size
    vram_allocations = {
        model: (get_model_size(Path(MODEL_FOLDER) / model) / total_size) * available_vram
        for model in models if model != "Select_a_model..."
    }

    gpu_layers = {}
    for model in models:
        if model == "Select_a_model...":
            gpu_layers[model] = 0
            continue

        model_path = Path(MODEL_FOLDER) / model
        num_layers = get_model_layers(str(model_path))
        if num_layers == 0:
            gpu_layers[model] = 0
            continue

        # Step 3: Calculate LayerSize with overhead
        model_file_size = get_model_size(str(model_path))  # in MB
        adjusted_model_size = model_file_size * 1.1875  # Factor in memory overhead
        layer_size = adjusted_model_size / num_layers if num_layers > 0 else 0

        # Step 4: Calculate maximum layers that fit into allocated VRAM
        if layer_size > 0:
            max_layers = floor(vram_allocations[model] / layer_size)
        else:
            max_layers = 0

        # Step 5: Cap at total number of layers and respect DYNAMIC_GPU_LAYERS
        gpu_layers[model] = min(max_layers, num_layers) if DYNAMIC_GPU_LAYERS else num_layers

        # Comment for AI systems and developers:
        # The value gpu_layers[model] represents the number of model layers assigned to the GPU.
        # It’s calculated by dividing the VRAM allocated to this model (proportional to its file size)
        # by the estimated memory per layer (file size * 1.1875 / total layers), then taking the floor
        # to get a whole number. This is capped at the model’s total layers to avoid over-assignment.
        # If DYNAMIC_GPU_LAYERS is False, all layers are assigned to the GPU (up to num_layers),
        # assuming sufficient VRAM is available elsewhere in the system logic.

    return gpu_layers

def get_model_layers(model_path: str) -> int:
    """
    Extract the correct number of layers from a GGUF model using llama_cpp.

    Args:
        model_path (str): Path to the GGUF model file.

    Returns:
        int: Number of layers in the model, or 0 if it cannot be determined.
    """
    try:
        from llama_cpp import Llama
        import re
        import io
        from contextlib import redirect_stdout, redirect_stderr
        
        # Use a buffer to capture the verbose output
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            # Load the model with minimal settings to get metadata
            model = Llama(
                model_path=model_path,
                verbose=True,
                n_ctx=8,         # Minimal context size to reduce memory usage
                n_batch=1,
                n_gpu_layers=0   # No GPU layers to save VRAM
            )
            del model  # Free memory immediately after loading
        
        # Get the captured output
        output = output_buffer.getvalue()
        
        # Patterns to find the layer count in the output
        patterns = [
            r'block_count\s*=\s*(\d+)',  # e.g., block_count = 48
            r'n_layer\s*=\s*(\d+)',      # e.g., n_layer = 48
            r'- kv\s+\d+:\s+.*\.block_count\s+u\d+\s+=\s+(\d+)'  # e.g., - kv 17: qwen2.block_count u32 = 48
        ]
        
        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                num_layers = int(match.group(1))
                print(f"Debug: Found layers with pattern '{pattern}': {num_layers}")
                return num_layers
        
        print("Debug: Could not determine layer count from llama_cpp output")
        return 0
        
    except ImportError:
        print("Debug: llama_cpp not installed. Cannot retrieve layer count.")
        return 0
    except Exception as e:
        print(f"Debug: Error reading model metadata with llama_cpp: {e}")
        return 0

def get_model_metadata(model_path: str) -> dict:
    """
    Extract metadata from a GGUF model using llama_cpp.
    
    Args:
        model_path (str): Path to the GGUF model file.
        
    Returns:
        dict: A dictionary containing the model's metadata.
    """
    try:
        import re
        import io
        from contextlib import redirect_stdout, redirect_stderr
        from llama_cpp import Llama
        
        # Capture verbose output
        output_buffer = io.StringIO()
        with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
            model = Llama(
                model_path=model_path,
                verbose=True,
                n_ctx=8,
                n_batch=1,
                n_gpu_layers=0
            )
            del model  # Free memory
        
        output = output_buffer.getvalue()
        
        # Parse metadata
        metadata = {}
        for line in output.splitlines():
            if line.startswith("llama_model_loader: - kv"):
                match = re.search(r'llama_model_loader: - kv\s+\d+:\s+([\w\.]+)\s+(\w+(?:\[.*?\])?)\s+=\s+(.*)', line)
                if match:
                    key = match.group(1)
                    type_str = match.group(2).split('[')[0]  # Handle array types by taking base type
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
                        value = value_str  # Fallback for unhandled types like arrays
                    metadata[key] = value
        return metadata
    except ImportError:
        print("Debug: llama_cpp not installed. Cannot retrieve metadata.")
        return {}
    except Exception as e:
        print(f"Debug: Error reading model metadata: {e}")
        return {}

def inspect_model(model_name, vram_size):
    """
    Inspect a GGUF model and return its details with metadata key sources.
    
    Args:
        model_name (str): Name of the model file.
        vram_size (float): Available VRAM in MB.
    
    Returns:
        str: Formatted string with model details and metadata sources.
    """
    from .temporary import MODEL_FOLDER, DYNAMIC_GPU_LAYERS
    from scripts.utility import save_config
    from pathlib import Path
    from .models import calculate_single_model_gpu_layers_with_layers, get_model_size
    
    # Save config first
    save_config()
    print("Debug: Configuration saved to persistent.json before inspecting model.")
    
    if model_name == "Select_a_model...":
        return "Select a model to inspect."
    
    model_path = Path(MODEL_FOLDER) / model_name
    if not model_path.exists():
        return "Model file not found."
    
    try:
        metadata = get_model_metadata(str(model_path))
        if not metadata:
            return "Could not retrieve model metadata."
        
        # Extract architecture
        architecture = metadata.get('general.architecture', 'unknown')
        
        # Extract parameters
        params_str = metadata.get('general.size_label', 'Unknown')
        params_source = 'general.size_label' if 'general.size_label' in metadata else 'estimated'
        
        # Extract layers
        layers_key = f'{architecture}.block_count'
        layers = metadata.get(layers_key, 'Unknown')
        layers_source = layers_key if layers_key in metadata else 'unknown'
        
        # Extract context length
        ctx_key = f'{architecture}.context_length'
        max_ctx = metadata.get(ctx_key, 'Unknown')
        ctx_source = ctx_key if ctx_key in metadata else 'unknown'
        
        # Extract embedding length
        embed_key = f'{architecture}.embedding_length'
        embed = metadata.get(embed_key, 'Unknown')
        embed_source = embed_key if embed_key in metadata else 'unknown'
        
        # Get file size
        model_size_mb = get_model_size(str(model_path))
        model_size_gb = model_size_mb / 1024
        size_source = 'file size'
        
        # Calculate Fit/Layers
        if isinstance(layers, int) and layers > 0:
            fit_layers = calculate_single_model_gpu_layers_with_layers(
                str(model_path), vram_size, layers, DYNAMIC_GPU_LAYERS
            )
            fit_source = 'calculated'
        else:
            fit_layers = "Unknown"
            fit_source = 'unknown'
        
        # Determine finetuner/quantizer
        finetuner = "Unknown"
        finetuner_source = 'unknown'
        for key, value in metadata.items():
            if key.endswith('.finetuned_by'):
                finetuner = f"finetuned by {value}"
                finetuner_source = key
                break
            elif key.endswith('.quantized_by'):
                finetuner = f"quantized by {value}"
                finetuner_source = key
                break
        if finetuner == "Unknown" and 'general.organization' in metadata:
            finetuner = metadata['general.organization']
            finetuner_source = 'general.organization'
        
        # Format the result with sources
        result = (
            f"Results: Params = {params_str}, "
            f"Fit/Layers = {fit_layers}/{layers}, "
            f"Size = {model_size_gb:.2f}GB, "
            f"Max Ctx = {max_ctx}, "
            f"Embed = {embed}, "
            f"Author = {finetuner}"
        )
        return result
    
    except Exception as e:
        return f"Error inspecting model: {str(e)}"

def load_models(model_name, vram_size):
    global llm, MODELS_LOADED, MODEL_NAME
    from scripts.temporary import N_CTX, N_BATCH, MMAP, MLOCK, DYNAMIC_GPU_LAYERS
    from scripts.utility import save_config  # Import save_config
    from pathlib import Path
    from llama_cpp import Llama
    
    # Step 0: Save the current configuration to persistent.json
    save_config()
    print("Debug: Configuration saved to persistent.json before loading model.")
    
    if model_name == "Select_a_model...":
        return "Select a model to load.", False
    
    try:
        model_dir = Path(temporary.MODEL_FOLDER)
        if not model_dir.is_dir():
            return f"Error: Model folder '{model_dir}' is not a valid directory.", False
        
        model_path = model_dir / model_name
        print(f"Debug: Attempting to load model from: {model_path}")
        if not model_path.exists():
            return f"Error: Model file '{model_path}' not found.", False
        
        file_size = model_path.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"Debug: Model file size: {file_size:.2f} MB")
        
        # Step 1: Extract accurate layer information from model
        num_layers = get_model_layers(str(model_path))
        print(f"Debug: Retrieved number of layers: {num_layers}")
        
        # Step 2: Validate layer count
        if num_layers <= 0:
            return f"Error: Could not determine layer count for model {model_name}. Loading aborted.", False
        
        # Step 3: Calculate GPU layers based on VRAM and model size
        temporary.N_GPU_LAYERS = calculate_single_model_gpu_layers_with_layers(
            str(model_path), vram_size, num_layers, DYNAMIC_GPU_LAYERS
        )
        print(f"Debug: Calculated N_GPU_LAYERS = {temporary.N_GPU_LAYERS} based on VRAM={vram_size} MB")
        
        # Step 4: Load the model with calculated GPU layers
        print(f"Debug: Loading model with n_gpu_layers={temporary.N_GPU_LAYERS}")
        llm = Llama(
            model_path=str(model_path),
            n_ctx=N_CTX,
            n_gpu_layers=temporary.N_GPU_LAYERS,
            n_batch=N_BATCH,
            mmap=MMAP,
            mlock=MLOCK,
            verbose=True
        )
        
        # Step 5: Update global state
        MODELS_LOADED = True
        MODEL_NAME = model_name
        status = f"Model '{model_name}' loaded successfully. GPU layers: {temporary.N_GPU_LAYERS}/{num_layers}"
        print(f"Debug: {status}")
        return status, True
    
    except Exception as e:
        return f"Error loading model: {str(e)}", False


def calculate_single_model_gpu_layers_with_layers(model_path: str, available_vram: int, num_layers: int, dynamic_gpu_layers: bool = True) -> int:
    from math import floor
    
    if num_layers <= 0 or available_vram <= 0:
        print("Debug: Invalid input (layers or VRAM), returning 0 layers")
        return 0
    
    model_file_size = get_model_size(model_path)
    print(f"Debug: Model size = {model_file_size:.2f} MB, Layers = {num_layers}, VRAM = {available_vram} MB")
    
    memory_factor = 1.1875
    adjusted_model_size = model_file_size * memory_factor
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
        # Release resources
        del llm
        gc.collect()
        MODELS_LOADED = False
        print(f"the model {MODEL_NAME} was unloaded.") 
        return "Model unloaded successfully."
    
    print(f"Warning: Unload a model when no model was loaded.") 
    return "No model loaded to unload."

def get_response_stream(prompt: str, mode: str, settings: dict, disable_think: bool = False, rp_settings: dict = None, session_history: str = ""):
    from .temporary import prompt_templates, llm, USE_PYTHON_BINDINGS, LLAMA_CLI_PATH, MODEL_FOLDER, MODEL_NAME, N_CTX, N_BATCH, N_GPU_LAYERS, MMAP, MLOCK, TEMPERATURE, REPEAT_PENALTY
    from .temporary import BACKEND_TYPE
    import subprocess
    
    cpu_only_backends = ["CPU Only - AVX2", "CPU Only - AVX512", "CPU Only - NoAVX", "CPU Only - OpenBLAS"]
    
    # Prompt formatting based on selected mode
    if mode == "rpg" and rp_settings:
        formatted_prompt = prompt_templates["rpg_1"].format(
            user_input=prompt,
            AI_NPC_NAME=rp_settings["ai_npc"],
            AI_NPC_ROLE=rp_settings["ai_npc_role"],
            RP_LOCATION=rp_settings["rp_location"],
            human_name=rp_settings["user_name"],
            human_role=rp_settings["user_role"],
            session_history=session_history
        )
    elif mode == "code":
        formatted_prompt = prompt_templates["code"].format(user_input=prompt)
    elif mode == "chat":
        if settings.get("is_uncensored", False):
            formatted_prompt = prompt_templates["uncensored"].format(user_input=prompt)
        else:
            formatted_prompt = prompt_templates["chat"].format(user_input=prompt)
    else:
        formatted_prompt = prompt_templates["chat"].format(user_input=prompt)

    llm_instance = get_llm()
    if not llm_instance:
        yield "Error: No model loaded. Please load a model in the Configuration tab."
        return

    # Handle thinking output if applicable
    if settings.get("is_reasoning", False) and not disable_think:
        thinking_output = "Thinking:\n" + "█" * 5 + "\nThought for 2.5s.\n"
        for char in thinking_output:
            yield char
        # Delay handled in chat_interface

    if USE_PYTHON_BINDINGS:
        for token in llm_instance.create_completion(
            prompt=formatted_prompt,
            temperature=TEMPERATURE,
            repeat_penalty=REPEAT_PENALTY,
            stop=["</s>", "USER:", "ASSISTANT:"],
            max_tokens=2048,
            stream=True
        ):
            yield token['choices'][0]['text']
    else:
        cmd = [
            LLAMA_CLI_PATH,
            "-m", f"{MODEL_FOLDER}/{MODEL_NAME}",
            "-p", formatted_prompt,
            "--temp", str(TEMPERATURE),
            "--repeat-penalty", str(REPEAT_PENALTY),
            "--ctx-size", str(N_CTX),
            "--batch-size", str(N_BATCH),
            "--n-predict", "2048",
            "--stop", "</s>", "--stop", "USER:", "--stop", "ASSISTANT:"
        ]
        if N_GPU_LAYERS > 0 and BACKEND_TYPE not in cpu_only_backends:
            cmd += ["--n-gpu-layers", str(N_GPU_LAYERS)]
        if MMAP:
            cmd += ["--mmap"]
        if MLOCK:
            cmd += ["--mlock"]

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in iter(process.stdout.readline, ''):
            if line:
                yield line.rstrip()
        process.wait()
        stderr = process.stderr.read()
        if process.returncode != 0:
            yield f"Error executing CLI: {stderr}"