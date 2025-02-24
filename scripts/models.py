# Script: `.\scripts\models.py`

import subprocess
from llama_cpp import Llama
from pathlib import Path
from scripts.temporary import (
    MODEL_PATH, N_CTX, N_GPU_LAYERS, TEMPERATURE, USE_PYTHON_BINDINGS,
    LLAMA_CLI_PATH, MODEL_LOADED, BACKEND_TYPE, llm, VRAM_SIZE,
    DYNAMIC_GPU_LAYERS, MMAP, MLOCK
)

LAMARCK_PROMPT_TEMPLATE = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives detailed, practical answers with technical depth.\n\n"
    "USER: {prompt}\n"
    "ASSISTANT:"
)

class ContextInjector:
    def __init__(self):
        self.vectorstores = {}
        self.active_context = []
        self._load_default_vectorstores()

    def _load_default_vectorstores(self):
        from scripts.temporary import RAG_AUTO_LOAD, VECTORSTORE_DIR
        for vs_name in RAG_AUTO_LOAD:
            self.load_vectorstore(vs_name)

    def load_vectorstore(self, name: str):
        from scripts.temporary import VECTORSTORE_DIR
        vs_path = Path(VECTORSTORE_DIR) / f"{name}"
        if vs_path.exists():
            from langchain.vectorstores import FAISS
            from langchain.embeddings import HuggingFaceEmbeddings
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vectorstores[name] = FAISS.load_local(str(vs_path), embeddings)
        else:
            print(f"Vectorstore not found: {name}")

    def inject_context(self, prompt: str) -> str:
        if not self.vectorstores:
            return prompt
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = embedding_model.encode([prompt])[0]
        context = []
        for vs_name, vs in self.vectorstores.items():
            docs = vs.similarity_search_by_vector(query_embedding, k=RAG_MAX_DOCS)
            context.extend([f"{vs_name}: {doc.page_content}" for doc in docs])
        context_str = "\n".join(context)
        return f"Relevant information:\n{context_str}\n\nQuery: {prompt}"t

context_injector = ContextInjector()

def reload_vectorstore(name: str):
    context_injector.load_vectorstore(name)

def get_model_size(model_path: str) -> float:
    """Get model file size in MB."""
    return Path(model_path).stat().st_size / (1024 * 1024)

def get_model_layers(model_path: str) -> int:
    """Get number of layers from GGUF model metadata."""
    try:
        llm_temp = Llama(model_path=model_path, verbose=False)
        metadata = llm_temp.metadata
        num_layers = metadata.get('llama.block_count', 0)
        del llm_temp
        return num_layers
    except Exception:
        return 0

def calculate_gpu_layers(model_path: str, vram_size: float) -> int:
    """Calculate number of layers to offload to GPU based on VRAM size."""
    model_size = get_model_size(model_path)  # Model size in MB
    num_layers = get_model_layers(model_path)
    if num_layers == 0:
        return 0
    layer_size = (model_size * 1.25) / num_layers  # Per readme.md notation
    available_vram = (vram_size / 1024) * 0.9  # Convert MB to GB, use 90% of VRAM
    max_layers = int(available_vram / layer_size)
    return min(max_layers, num_layers)

def initialize_model(_):
    """Initialize the GGUF model with dynamic GPU layer offloading."""
    global llm, MODEL_LOADED
    try:
        if USE_PYTHON_BINDINGS:
            model_path = Path(MODEL_PATH)
            n_gpu_layers = 0  # Default to 0, dynamically calculated if enabled
            if DYNAMIC_GPU_LAYERS:
                n_gpu_layers = calculate_gpu_layers(str(model_path), VRAM_SIZE / 1024)
            llm = Llama(
                model_path=str(model_path),
                n_ctx=N_CTX,
                n_gpu_layers=n_gpu_layers,
                mmap=MMAP,
                mlock=MLOCK,
                verbose=False
            )
        else:
            if not Path(LLAMA_CLI_PATH).exists():
                raise FileNotFoundError("llama-cli executable not found")
        MODEL_LOADED = True
    except Exception as e:
        MODEL_LOADED = False
        raise RuntimeError(f"Model initialization failed: {str(e)}")

def unload_model():
    global llm, MODEL_LOADED
    if llm:
        try:
            del llm
            llm = None
            MODEL_LOADED = False
        except Exception as e:
            print(f"Error unloading model: {str(e)}")
    else:
        print("No model is currently loaded")

def get_streaming_response(prompt: str):
    global llm
    enhanced_prompt = context_injector.inject_context(prompt)
    formatted_prompt = LAMARCK_PROMPT_TEMPLATE.format(prompt=enhanced_prompt)
    if USE_PYTHON_BINDINGS:
        if not llm:
            raise RuntimeError("Model not loaded")
        stream = llm.create_completion(
            prompt=formatted_prompt,
            temperature=TEMPERATURE,
            max_tokens=2048,
            stream=True
        )
        for output in stream:
            yield output["choices"][0]["text"]
    else:
        cmd = [
            LLAMA_CLI_PATH,
            "-m", MODEL_PATH,
            "-p", formatted_prompt,
            "--temp", str(TEMPERATURE),
            "--ctx-size", str(N_CTX),
            "--n-predict", "2048",
            "--log-disable"
        ]
        if MMAP:  # Added (assumes --mmap exists)
            cmd += ["--mmap"]
        if MLOCK:  # Added (assumes --mlock exists)
            cmd += ["--mlock"]
        if "vulkan" in BACKEND_TYPE.lower():
            cmd += ["--vulkan", "--gpu-layers", str(N_GPU_LAYERS)]
        elif "kompute" in BACKEND_TYPE.lower():
            cmd += ["--kompute", "--gpu-layers", str(N_GPU_LAYERS)]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
        buffer = ""
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            buffer += line
            yield buffer

def get_response(prompt: str) -> str:
    global llm
    enhanced_prompt = context_injector.inject_context(prompt)
    formatted_prompt = LAMARCK_PROMPT_TEMPLATE.format(prompt=enhanced_prompt)
    if USE_PYTHON_BINDINGS:
        if not llm:
            raise RuntimeError("Model not loaded")
        output = llm.create_completion(
            prompt=formatted_prompt,
            temperature=TEMPERATURE,
            stop=["</s>", "USER:", "ASSISTANT:"],
            max_tokens=2048
        )
        return output["choices"][0]["text"]
    else:
        cmd = [
            LLAMA_CLI_PATH,
            "-m", MODEL_PATH,
            "-p", formatted_prompt,
            "--temp", str(TEMPERATURE),
            "--ctx-size", str(N_CTX),
            "--n-predict", "2048",
            "--log-disable"
        ]
        if MMAP:  # Added
            cmd += ["--mmap"]
        if MLOCK:  # Added
            cmd += ["--mlock"]
        if "vulkan" in BACKEND_TYPE.lower():
            cmd += ["--vulkan", "--gpu-layers", str(N_GPU_LAYERS)]
        elif "kompute" in BACKEND_TYPE.lower():
            cmd += ["--kompute", "--gpu-layers", str(N_GPU_LAYERS)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        return proc.stdout