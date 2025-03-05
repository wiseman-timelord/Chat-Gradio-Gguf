# Script: `.\scripts\models.py`

# Imports...
import time, subprocess
from llama_cpp import Llama
from pathlib import Path
import gradio as gr
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from scripts.temporary import (
    N_CTX, N_GPU_LAYERS, N_BATCH, USE_PYTHON_BINDINGS,
    LLAMA_CLI_PATH, BACKEND_TYPE, VRAM_SIZE,
    DYNAMIC_GPU_LAYERS, MMAP, MLOCK, current_model_settings,
    prompt_templates, llm, MODEL_NAME, MODEL_FOLDER, REPEAT_PENALTY,
    TEMPERATURE
)


# Classes...
class ContextInjector:
    def __init__(self):
        self.vectorstores = {}
        self.current_vectorstore = None
        self.current_mode = None
        self.session_vectorstore = None  # Added for session-specific RAG
        self._load_default_vectorstores()

    def _load_default_vectorstores(self):
        modes = ["code", "rpg", "chat"]
        for mode in modes:
            vs_path = Path("data/vectors") / mode / "knowledge"
            if vs_path.exists():
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                self.vectorstores[mode] = FAISS.load_local(
                    str(vs_path),
                    embeddings=embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded {mode} vectorstore.")

    def set_mode(self, mode: str):
        valid_modes = ["code", "rpg", "chat"]
        if mode in valid_modes:
            if mode not in self.vectorstores:
                self.load_vectorstore(mode)
            if mode in self.vectorstores:
                self.current_vectorstore = self.vectorstores[mode]
                self.current_mode = mode
            else:
                self.current_vectorstore = None
                self.current_mode = None
                print(f"No vectorstore found for mode: {mode}")
        else:
            print(f"Invalid mode: {mode}. Expected one of {valid_modes}")

    def load_vectorstore(self, mode: str):
        vs_path = Path("data/vectors") / mode / "knowledge"
        if vs_path.exists():
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vectorstores[mode] = FAISS.load_local(
                str(vs_path),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"Loaded {mode} vectorstore.")
        else:
            print(f"Vectorstore not found for mode: {mode}")

    def set_session_vectorstore(self, vectorstore):
        self.session_vectorstore = vectorstore

    def inject_context(self, prompt: str) -> str:
        if self.current_vectorstore is None and self.session_vectorstore is None:
            return prompt
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = embedding_model.encode([prompt])[0]
        docs = []
        if self.current_vectorstore:
            docs.extend(self.current_vectorstore.similarity_search_by_vector(query_embedding, k=2))
        if self.session_vectorstore:
            docs.extend(self.session_vectorstore.similarity_search_by_vector(query_embedding, k=2))
        context = "\n".join([doc.page_content for doc in docs])
        return f"Relevant information:\n{context}\n\nQuery: {prompt}"

context_injector = ContextInjector()

# Functions...
def reload_vectorstore(name: str):
    context_injector.load_vectorstore(name)
    return "Knowledge updated"

def get_model_size(model_path: str) -> float:
    """Get model file size in MB."""
    return Path(model_path).stat().st_size / (1024 * 1024)

def get_model_layers(model_path: str) -> int:
    """Get number of layers from GGUF model metadata efficiently."""
    try:
        from gguf import GGUFReader  # Assuming gguf library is available
        reader = GGUFReader(model_path)
        num_layers = reader.fields.get('llama.block_count', [0])[0]
        return int(num_layers)
    except Exception as e:
        print(f"Error reading model metadata: {e}")
        return 0  # Fallback if metadata unavailable
        
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

def get_model_settings(model_name):
    from .temporary import category_keywords, handling_keywords
    model_name_lower = model_name.lower()
    category = "chat"  # Default category
    is_uncensored = any(keyword in model_name_lower for keyword in handling_keywords["uncensored"])
    is_reasoning = any(keyword in model_name_lower for keyword in handling_keywords["reasoning"])
    for cat, keywords in category_keywords.items():
        if any(keyword in model_name_lower for keyword in keywords):
            category = cat
            break
    settings = {
        "category": category,
        "is_uncensored": is_uncensored,
        "is_reasoning": is_reasoning,
    }
    return settings

def determine_operation_mode(quality_model):
    if quality_model == "Select_a_model...":
        return "Select models to enable mode detection."
    settings = get_model_settings(quality_model)
    return settings["category"].capitalize()

def get_available_models():
    from .temporary import MODEL_FOLDER
    model_dir = Path(MODEL_FOLDER)
    return [f.name for f in model_dir.glob("*.gguf") if f.is_file()]

def inspect_model(model_name):
    from .temporary import MODEL_FOLDER
    from pathlib import Path
    
    if model_name == "Select_a_model...":
        return "Select a model to inspect."
    
    model_path = Path(MODEL_FOLDER) / model_name
    if not model_path.exists():
        return f"Model file not found: {model_path}"
    
    try:
        model_size_mb = get_model_size(str(model_path))
        num_layers = get_model_layers(str(model_path))
        settings = get_model_settings(model_name)
        model_type = settings["category"].capitalize()
        if num_layers > 0:
            model_size_gb = model_size_mb / 1024
            memory_per_layer_gb = (model_size_gb * 1.1875) / num_layers
            memory_per_layer_str = f"{memory_per_layer_gb:.3f} GB"
        else:
            memory_per_layer_str = "N/A"
        return f"Model: {model_name} | Type: {model_type} | Size: {model_size_mb:.2f} MB | Layers: {num_layers} | Memory/Layer: {memory_per_layer_str}"
    except Exception as e:
        return f"Error inspecting model: {str(e)}"

def load_models(quality_model, vram_size):
    """Load the selected model based on quality and VRAM settings."""
    global llm
    cpu_only_backends = [
        "CPU Only - AVX2", "CPU Only - AVX512", "CPU Only - NoAVX", "CPU Only - OpenBLAS"
    ]
    
    if quality_model == "Select_a_model...":
        return "Select a model to load.", False
    
    try:
        models_to_load = [quality_model]
        if temporary.BACKEND_TYPE in cpu_only_backends:
            set_cpu_affinity()
            temporary.N_GPU_LAYERS = 0  # No GPU layers for CPU-only backends
        else:
            gpu_layers = calculate_gpu_layers(models_to_load, vram_size)
            temporary.N_GPU_LAYERS = gpu_layers.get(quality_model, 0)
        
        model_path = Path(temporary.MODEL_FOLDER) / quality_model
        
        llm = Llama(
            model_path=str(model_path),
            n_ctx=temporary.N_CTX,
            n_gpu_layers=temporary.N_GPU_LAYERS,
            n_batch=temporary.N_BATCH,
            mmap=temporary.MMAP,
            mlock=temporary.MLOCK,
            verbose=False
        )
        
        temporary.MODELS_LOADED = True
        temporary.MODEL_NAME = quality_model
        status = f"Model '{quality_model}' loaded, layer distribution: VRAM={temporary.N_GPU_LAYERS} layers"
        return status, True
    
    except Exception as e:
        return f"Error loading model: {str(e)}", False
def unload_models():
    from scripts.temporary import llm
    if llm is not None:
        del llm
        llm = None
    print("Models unloaded successfully.")

def get_response_stream(prompt: str, disable_think: bool = False, rp_settings: dict = None, session_history: str = ""):
    from .temporary import llm, USE_PYTHON_BINDINGS, LLAMA_CLI_PATH, MODEL_FOLDER, MODEL_NAME, N_CTX, N_BATCH, N_GPU_LAYERS, MMAP, MLOCK, TEMPERATURE, REPEAT_PENALTY
    from .temporary import BACKEND_TYPE, prompt_templates
    import subprocess
    
    cpu_only_backends = ["CPU Only - AVX2", "CPU Only - AVX512", "CPU Only - NoAVX", "CPU Only - OpenBLAS"]
    
    try:
        enhanced_prompt = context_injector.inject_context(prompt)
        settings = get_model_settings(MODEL_NAME)
        mode = settings["category"]

        if mode == "rpg" and rp_settings:
            active_npcs = sum(1 for npc in [rp_settings["ai_npc1"], rp_settings["ai_npc2"], rp_settings["ai_npc3"]] if npc != "Unused")
            template_key = f"rpg_{active_npcs}" if active_npcs in [1, 2, 3] else "rpg_1"
            formatted_prompt = prompt_templates[template_key].format(
                user_input=enhanced_prompt,
                AI_NPC1_NAME=rp_settings["ai_npc1"],
                AI_NPC2_NAME=rp_settings["ai_npc2"],
                AI_NPC3_NAME=rp_settings["ai_npc3"],
                AI_NPCS_ROLES=rp_settings["ai_npcs_roles"],
                RP_LOCATION=rp_settings["rp_location"],
                human_name=rp_settings["user_name"],
                human_role=rp_settings["user_role"],
                session_history=session_history
            )
        elif mode == "chat":
            template_key = "uncensored" if settings["is_uncensored"] else "chat"
            formatted_prompt = prompt_templates[template_key].format(user_input=enhanced_prompt)
        else:
            formatted_prompt = prompt_templates.get(mode, prompt_templates["chat"]).format(user_input=enhanced_prompt)

        llm = get_llm()
        if not llm:
            yield "Error: No model loaded. Please load a model in the Configuration tab."
            return

        # Handle thinking output if applicable
        if settings["is_reasoning"] and not disable_think:
            thinking_output = "Thinking:\n" + "█" * 5 + "\nThought for 2.5s.\n"
            for char in thinking_output:
                yield char
            # Removed await asyncio.sleep(2.5) - delay is handled in chat_interface

        if USE_PYTHON_BINDINGS:
            for token in llm.create_completion(
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
    except Exception as e:
        yield f"Error generating response: {str(e)}"