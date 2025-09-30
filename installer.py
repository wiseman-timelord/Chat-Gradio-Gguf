# Script: installer.py (Installation script for Chat-Gradio-Gguf)
# The GPU Selection MUST retain options of, Vulkan, CUDA-12, HIP-Radeon.

# Imports
import os
import json
import subprocess
import sys
import contextlib
import time
from pathlib import Path
import shutil


# Constants
_PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
APP_NAME = "Chat-Gradio-Gguf"
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
TEMP_DIR = BASE_DIR / "data/temp"
LLAMACPP_TARGET_VERSION = "b6586"

DIRECTORIES = [
    "data", "scripts", "models",
    "data/history", "data/temp", "data/vectors",
    "data/fastembed_cache"
]

# Platform detection (windows / linux)
PLATFORM = None

def set_platform() -> None:
    global PLATFORM
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ["windows", "linux"]:
        print("ERROR: Platform argument required (windows/linux)")
        sys.exit(1)
    PLATFORM = sys.argv[1].lower()

set_platform()

# Backend definitions (PyTorch wheels removed)
if PLATFORM == "windows":
    BACKEND_OPTIONS = {
        "GPU/CPU - Vulkan": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli.exe",
            "needs_python_bindings": False,
            "vulkan_required": True,
            "build_flags": {}
        },
        "GPU/CPU - HIP-Radeon": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-hip-radeon-x64.zip",
            "dest": "data/llama-hip-radeon-bin",
            "cli_path": "data/llama-hip-radeon-bin/llama-cli.exe",
            "needs_python_bindings": False,
            "rocm_required": True,
            "build_flags": {}
        },
        "GPU/CPU - CUDA 12.x": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-cuda-12.4-x64.zip",
            "dest": "data/llama-cuda-12-bin",
            "cli_path": "data/llama-cuda-12-bin/llama-cli.exe",
            "needs_python_bindings": False,
            "cuda_required": True,
            "build_flags": {}
        }
    }
else:  # Linux
    BACKEND_OPTIONS = {
        "GPU/CPU - Vulkan": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-ubuntu-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli",
            "needs_python_bindings": False,
            "vulkan_required": True,
            "build_flags": {}
        }
    }


# Python requirements (CPU-only, no torch)
BASE_REQ = [
    "gradio",
    "requests==2.31.0",
    "pyperclip",
    "yake",
    "psutil",
    "ddgs",
    "newspaper3k",
    "langchain-community>=0.3.18",
    "faiss-cpu>=1.8.0",
    "langchain>=0.3.18",
    "pygments==2.17.2",
    "lxml[html_clean]",
    "pyttsx3",
    "onnxruntime",
    "fastembed",
    "tokenizers",
]

if PLATFORM == "windows":
    BASE_REQ.extend(["pywin32", "tk"])


# Utility helpers
def print_header(title: str) -> None:
    os.system('clear' if PLATFORM == "linux" else 'cls')
    width = shutil.get_terminal_size().columns
    print("=" * width)
    print(f"    {APP_NAME}: {title}")
    print("=" * width)
    print()

def print_status(message: str, success: bool = True) -> None:
    status = "[✓]" if success else "[✗]"
    print(f"{status} {message}")
    time.sleep(1 if success else 3)

def get_user_choice(prompt: str, options: list) -> str:
    print_header("Install Options")
    print(f" {prompt}")
    for i, option in enumerate(options, 1):
        print(f"    {i}. {option}")
    print("\n" + "-" * shutil.get_terminal_size().columns)

    while True:
        choice = input(f"Selection (1-{len(options)}, X to exit): ").strip().upper()
        if choice == "X":
            print("\nExiting installer...")
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice, please try again.")

@contextlib.contextmanager
def activate_venv():
    if not VENV_DIR.exists():
        raise FileNotFoundError(f"Virtual environment not found at {VENV_DIR}")

    if PLATFORM == "windows":
        bin_dir = VENV_DIR / "Scripts"
        python_exe = bin_dir / "python.exe"
    else:
        bin_dir = VENV_DIR / "bin"
        python_exe = bin_dir / "python"

    if not python_exe.exists():
        raise FileNotFoundError(f"Python executable not found at {python_exe}")

    old_path = os.environ["PATH"]
    old_python = sys.executable
    try:
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{old_path}"
        sys.executable = str(python_exe)
        yield
    finally:
        os.environ["PATH"] = old_path
        sys.executable = old_python

def create_directories() -> None:
    for dir_path in DIRECTORIES:
        full_path = BASE_DIR / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        try:
            test_file = full_path / "permission_test"
            test_file.touch()
            test_file.unlink()
            print_status(f"Verified directory: {dir_path}")
        except PermissionError:
            print_status(f"Permission denied for: {dir_path}", False)
            sys.exit(1)

def build_config(backend: str) -> dict:
    info = BACKEND_OPTIONS[backend]
    cli_relative = f"{info['dest']}/llama-cli{'.exe' if PLATFORM == 'windows' else ''}"
    return {
        "model_settings": {
            "model_dir": "models",
            "model_name": "",
            "context_size": 8192,
            "temperature": 0.66,
            "repeat_penalty": 1.1,
            "use_python_bindings": info["needs_python_bindings"],
            "llama_cli_path": str(BASE_DIR / cli_relative),
            "vram_size": 8192,
            "selected_gpu": None,
            "mmap": True,
            "mlock": True,
            "n_batch": 2048,
            "dynamic_gpu_layers": True,
            "max_history_slots": 12,
            "max_attach_slots": 6,
            "print_raw_output": False,
            "session_log_height": 500
        },
        "backend_config": {
            "backend_type": backend,
            "llama_bin_path": info["dest"],
            "cuda_required": info.get("cuda_required", False),
            "vulkan_required": info.get("vulkan_required", False)
        }
    }

def create_config(backend: str) -> None:
    config_path = BASE_DIR / "data" / "persistent.json"
    config = build_config(backend)
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print_status("Configuration file created")
    except Exception as e:
        print_status(f"Failed to create config: {str(e)}", False)

def create_venv() -> bool:
    try:
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
            print_status("Removed existing virtual environment")

        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print_status("Created new virtual environment")

        python_exe = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
        if not python_exe.exists():
            raise FileNotFoundError(f"Python executable not found at {python_exe}")

        print_status("Verified virtual environment setup")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to create venv: {e}", False)
        return False


# Progress helpers
def simple_progress_bar(current: int, total: int, width: int = 25) -> str:
    if total == 0:
        return "[" + "=" * width + "] 100%"
    filled_width = int(width * current // total)
    bar = "=" * filled_width + "-" * (width - filled_width)
    percent = 100 * current // total

    def format_bytes(b):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if b < 1024.0:
                return f"{b:.1f}{unit}"
            b /= 1024.0
        return f"{b:.1f}TB"

    return f"[{bar}] {percent}% ({format_bytes(current)}/{format_bytes(total)})"

def download_fastembed_model() -> bool:
    """Download and cache FastEmbed model during installation."""
    print_status("Downloading embedding model (FastEmbed)...")
    
    try:
        # Use the virtual environment's Python
        python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                        ("python.exe" if PLATFORM == "windows" else "python"))
        
        # Create a simple script to download the model
        download_script = '''
import os
from pathlib import Path
from fastembed import TextEmbedding

# Set cache directory
cache_dir = Path("data/fastembed_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# Override default cache location
os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir.absolute())

# Download the model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
print(f"Downloading {model_name}...")
embedding = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
print("Model downloaded successfully!")
'''
        
        # Write temporary download script
        download_script_path = TEMP_DIR / "download_fastembed.py"
        with open(download_script_path, 'w') as f:
            f.write(download_script)
        
        # Run the download script
        result = subprocess.run(
            [python_exe, str(download_script_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            print_status("Embedding model downloaded")
            download_script_path.unlink(missing_ok=True)
            return True
        else:
            print_status(f"Model download failed: {result.stderr}", False)
            return False
            
    except subprocess.TimeoutExpired:
        print_status("Model download timed out", False)
        return False
    except Exception as e:
        print_status(f"Model download error: {e}", False)
        return False

def download_with_progress(url: str, filepath: Path, description: str = "Downloading") -> None:
    import requests
    import time

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        chunk_size = 8192
        last_update = time.time()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    current_time = time.time()
                    if current_time - last_update > 0.35 or downloaded >= total_size:
                        if total_size > 0:
                            progress = simple_progress_bar(downloaded, total_size)
                            print(f"\r{description}: {progress}", end='', flush=True)
                        else:
                            downloaded_mb = downloaded / 1024 / 1024
                            print(f"\r{description}: {downloaded_mb:.1f} MB downloaded", end='', flush=True)
                        last_update = current_time
        print()
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise e


# Dependency checks
def is_vulkan_installed() -> bool:
    if PLATFORM == "windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Khronos\Vulkan\Drivers"):
                return True
        except:
            return False
    else:
        try:
            result1 = subprocess.run(["vulkaninfo", "--summary"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            result2 = subprocess.run(["ldconfig", "-p"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return result1.returncode == 0 or b"libvulkan" in result2.stdout
        except FileNotFoundError:
            return False

def is_cuda_installed() -> bool:
    if PLATFORM == "windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NVIDIA Corporation\CUDA"):
                return True
        except:
            return False
    else:
        try:
            result = subprocess.run(["nvcc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except FileNotFoundError:
            return False

def verify_backend_dependencies(backend: str) -> bool:
    info = BACKEND_OPTIONS[backend]
    if info.get("vulkan_required", False):
        if not is_vulkan_installed():
            print("\n" + "!" * 80)
            print(f"Vulkan not detected but required for {backend} backend!")
            if PLATFORM == "windows":
                print("  Download from: https://vulkan.lunarg.com/sdk/home")
            else:
                print("  Install with: sudo apt install vulkan-tools libvulkan-dev")
            print("!" * 80 + "\n")
            return False
    if info.get("cuda_required", False):
        if not is_cuda_installed():
            print("\n" + "!" * 80)
            print(f"CUDA not detected but required for {backend} backend!")
            if PLATFORM == "windows":
                print("  Download from: https://developer.nvidia.com/cuda-downloads")
            else:
                print("  Install with: sudo apt install nvidia-cuda-toolkit")
            print("!" * 80 + "\n")
            return False
    return True

def install_linux_system_dependencies(backend: str) -> bool:
    if PLATFORM != "linux":
        return True

    print_status("Installing Linux system dependencies...")

    packages = [
        "build-essential",
        "portaudio19-dev",
        "libasound2-dev",
        "python3-tk",
        "vulkan-tools",
        "libvulkan-dev",
        "espeak",
        "libespeak-dev",
        "ffmpeg",
        "xclip"
    ]

    if backend.startswith("GPU/CPU - Vulkan"):
        packages.extend(["mesa-utils", "libvulkan1"])
    elif backend.startswith("GPU/CPU - CUDA"):
        packages.extend(["nvidia-cuda-toolkit"])

    try:
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y"] + list(set(packages)), check=True)
        print_status("Linux dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"System dependencies failed: {e}", False)
        return False


# Python dependencies (no torch)
def install_python_deps(backend: str) -> bool:
    print_status("Installing Python dependencies...")
    try:
        python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python"))
        pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("pip.exe" if PLATFORM == "windows" else "pip"))

        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print_status("Upgraded pip")

        # Base CPU packages
        subprocess.run([pip_exe, "install", *BASE_REQ], check=True)

        # Backend-specific llama-cpp-python
        llama_cmd = [pip_exe, "install"]
        if backend.startswith("GPU/CPU - CUDA"):
            llama_cmd.extend(["llama-cpp-python[cuda]"])
        else:
            llama_cmd.extend(["llama-cpp-python"])

        max_retries = 3
        for attempt in range(max_retries):
            try:
                subprocess.run(llama_cmd, check=True)
                print_status("llama-cpp-python installed")
                break
            except subprocess.CalledProcessError:
                if attempt == max_retries - 1:
                    print_status("llama-cpp-python failed after retries", False)
                    return False
                time.sleep(5)

        print_status("Python dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Install failed: {e}", False)
        return False


# Backend download & extraction
def copy_linux_binaries(source_dir: Path, dest_dir: Path) -> None:
    build_bin_dir = source_dir / "build" / "bin"
    if not build_bin_dir.exists():
        raise FileNotFoundError(f"Build dir not found: {build_bin_dir}")

    copied = 0
    for file in build_bin_dir.iterdir():
        if file.is_file() and file.name.startswith("llama"):
            dest_file = dest_dir / file.name
            shutil.copy2(file, dest_file)
            os.chmod(dest_file, 0o755)
            copied += 1

    if copied:
        print_status(f"Copied {copied} binaries")
    else:
        raise FileNotFoundError("No llama binaries")

def download_extract_backend(backend: str) -> bool:
    print_status(f"Downloading llama.cpp ({backend})...")
    info = BACKEND_OPTIONS[backend]
    TEMP_DIR.mkdir(exist_ok=True)
    temp_zip = TEMP_DIR / "llama.zip"

    try:
        import zipfile
        download_with_progress(info["url"], temp_zip, f"Downloading {backend}")

        dest_path = BASE_DIR / info["dest"]
        dest_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(temp_zip, 'r') as zf:
            members = zf.namelist()
            total = len(members)
            for i, m in enumerate(members):
                zf.extract(m, dest_path)
                if i % 25 == 0 or i == total - 1:
                    print(f"\rExtracting: {simple_progress_bar(i + 1, total)}", end='', flush=True)
            print()

        if PLATFORM == "linux":
            copy_linux_binaries(dest_path, dest_path)

        cli_path = BASE_DIR / info["cli_path"]
        if not cli_path.exists():
            raise FileNotFoundError(f"llama-cli not found: {cli_path}")
        if PLATFORM == "linux":
            os.chmod(cli_path, 0o755)

        print_status("Backend ready")
        return True
    except Exception as e:
        print_status(f"Backend install failed: {e}", False)
        return False
    finally:
        temp_zip.unlink(missing_ok=True)


# Backend selection
def select_backend_type() -> str:
    if PLATFORM == "windows":
        opts = [
            "Vulkan - GPU/CPU – AMD/Intel/NVIDIA GPUs",
            "HIP-Radeon - GPU/CPU – AMD ROCm",
            "CUDA 12.x - GPU/CPU – NVIDIA GPUs"
        ]
        mapping = {
            opts[0]: "GPU/CPU - Vulkan",
            opts[1]: "GPU/CPU - HIP-Radeon",
            opts[2]: "GPU/CPU - CUDA 12.x"
        }
        choice = get_user_choice("Select backend:", opts)
        return mapping[choice]
    else:  # Linux
        return "GPU/CPU - Vulkan"


# Main install flow
def install():
    backend = select_backend_type()
    print_header("Installation")
    print(f"Installing {APP_NAME} on {PLATFORM} using {backend}")

    if sys.version_info < (3, 8):
        print_status("Python ≥3.8 required", False)
        sys.exit(1)

    if not verify_backend_dependencies(backend):
        print_status("Missing system dependencies", False)
        sys.exit(1)

    create_directories()
    if PLATFORM == "linux":
        install_linux_system_dependencies(backend)

    if not create_venv():
        print_status("Virtual environment failed", False)
        sys.exit(1)

    if not install_python_deps(backend):
        print_status("Python dependencies failed", False)
        sys.exit(1)

    # NEW: Download FastEmbed model
    if not download_fastembed_model():
        print_status("WARNING: Embedding model download failed", False)
        print("RAG features may not work until model is downloaded")
        # Don't exit - allow installation to continue

    info = BACKEND_OPTIONS[backend]
    if not info["needs_python_bindings"]:
        if not download_extract_backend(backend):
            print_status("Backend download failed", False)
            sys.exit(1)

    create_config(backend)
    print_status("Installation complete!")
    print("\nRun the launcher to start Chat-Gradio-Gguf\n")


if __name__ == "__main__":
    try:
        install()
    except KeyboardInterrupt:
        print("\nInstallation cancelled")
        sys.exit(1)