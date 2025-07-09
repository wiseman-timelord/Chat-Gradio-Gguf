# Script: installer.py (Installation script for Chat-Gradio-Gguf)

# Imports
import os
import json
import subprocess
import sys
import contextlib
import time
from pathlib import Path
import shutil

# Validate platform argument


# Constants
APP_NAME = "Chat-Gradio-Gguf"
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
TEMP_DIR = BASE_DIR / "data/temp"
LLAMACPP_TARGET_VERSION = "b5587"
BACKEND_TYPE = None
DIRECTORIES = [
    "data", "scripts", "models",
    "data/history", "data/temp"
]

# Platform-specific configurations
if PLATFORM == "windows":
    VULKAN_TARGET_VERSION = "1.4.304.1"
    BACKEND_OPTIONS = {
        "CPU Only - AVX2": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-avx2-x64.zip",
            "dest": "data/llama-avx2-bin",
            "cli_path": "data/llama-avx2-bin/llama-cli.exe",
            "needs_python_bindings": True
        },
        "CPU Only - AVX512": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-avx512-x64.zip",
            "dest": "data/llama-avx512-bin",
            "cli_path": "data/llama-avx512-bin/llama-cli.exe",
            "needs_python_bindings": True
        },
        "CPU Only - NoAVX": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-noavx-x64.zip",
            "dest": "data/llama-noavx-bin",
            "cli_path": "data/llama-noavx-bin/llama-cli.exe",
            "needs_python_bindings": True
        },
        "CPU Only - OpenBLAS": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-openblas-x64.zip",
            "dest": "data/llama-openblas-bin",
            "cli_path": "data/llama-openblas-bin/llama-cli.exe",
            "needs_python_bindings": True
        },
        "GPU/CPU - Vulkan": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli.exe",
            "needs_python_bindings": False,
            "vulkan_required": True
        },
        "GPU/CPU - HIP-Radeon": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-hip-radeon-x64.zip",
            "dest": "data/llama-hip-radeon-bin",
            "cli_path": "data/llama-hip-radeon-bin/llama-cli.exe",
            "needs_python_bindings": False,
            "vulkan_required": True
        },
        "GPU/CPU - CUDA 11.7": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-cuda-11.7-x64.zip",
            "dest": "data/llama-cuda-11.7-bin",
            "cli_path": "data/llama-cuda-11.7-bin/llama-cli.exe",
            "needs_python_bindings": False,
            "cuda_required": True
        },
        "GPU/CPU - CUDA 12.4": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-cuda-12.4-x64.zip",
            "dest": "data/llama-cuda-12.4-bin",
            "cli_path": "data/llama-cuda-12.4-bin/llama-cli.exe",
            "needs_python_bindings": False,
            "cuda_required": True
        }
    }
elif PLATFORM == "linux":
    BACKEND_OPTIONS = {
        "CPU Only": {
            "needs_python_bindings": True,
            "build_flags": {}
        },
        "GPU/CPU - Vulkan": {
            "needs_python_bindings": True,
            "build_flags": {"LLAMA_VULKAN": "1"},
            "vulkan_required": True
        },
        "GPU/CPU - CUDA": {
            "needs_python_bindings": True,
            "build_flags": {"LLAMA_CUBLAS": "1"},
            "cuda_required": True
        }
    }

# Python dependencies
REQUIREMENTS = [
    "gradio>=4.25.0",
    "requests==2.31.0",
    "pyperclip",
    "yake",
    "psutil",
    "duckduckgo-search",
    "newspaper3k",
    "llama-cpp-python",
    "langchain-community==0.3.18",
    "pygments==2.17.2",
    "lxml_html_clean",
    "pyttsx3",
    "pyobjc" if PLATFORM == "linux" else "pywin32"
]

CONFIG_TEMPLATE = """{
    "model_settings": {
        "model_dir": "models",
        "model_name": "",
        "context_size": 8192,
        "temperature": 0.66,
        "repeat_penalty": 1.1,
        "use_python_bindings": false,
        "llama_cli_path": "data/llama-vulkan-bin/llama-cli.exe",
        "vram_size": 8192,
        "selected_gpu": null,
        "selected_cpu": "null",
        "mmap": true,
        "mlock": true,
        "n_batch": 2048,
        "dynamic_gpu_layers": true,
        "max_history_slots": 12,
        "max_attach_slots": 6,
        "session_log_height": 500
    },
    "backend_config": {
        "backend_type": "GPU/CPU - Vulkan",
        "llama_bin_path": "data/llama-vulkan-bin"
    }
}"""

# Set Platform
def set_platform():
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ["windows", "linux"]:
        print("ERROR: Platform argument required (windows/linux)")
        sys.exit(1)
    temporary.PLATFORM = sys.argv[1].lower()

# Utility functions
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

def create_config(backend: str) -> None:
    config_path = BASE_DIR / "data" / "persistent.json"
    config = json.loads(CONFIG_TEMPLATE)
    backend_info = BACKEND_OPTIONS[backend]
    
    config["backend_config"]["backend_type"] = backend
    config["backend_config"]["llama_bin_path"] = backend_info.get("dest", "")
    config["model_settings"]["use_python_bindings"] = backend_info["needs_python_bindings"]
    
    if not backend_info["needs_python_bindings"]:
        config["model_settings"]["llama_cli_path"] = str(BASE_DIR / backend_info["cli_path"])
    else:
        config["model_settings"]["llama_cli_path"] = None
    
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
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

def install_python_deps(backend: str) -> bool:
    print_status("Installing Python dependencies...")
    try:
        python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python"))
        pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("pip.exe" if PLATFORM == "windows" else "pip"))
        
        if not os.path.exists(pip_exe):
            raise FileNotFoundError(f"Pip not found at {pip_exe}")
        
        # Upgrade pip first
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print_status("Upgraded pip successfully")
        
        # Linux-specific system dependencies
        if PLATFORM == "linux":
            print_status("Installing Linux system dependencies...")
            subprocess.run([
                "sudo", "apt-get", "update"
            ], check=True)
            
            subprocess.run([
                "sudo", "apt-get", "install", "-y",
                "espeak", "portaudio19-dev", "libasound2-dev"
            ], check=True)
            
            if backend == "GPU/CPU - Vulkan":
                subprocess.run([
                    "sudo", "apt-get", "install", "-y",
                    "vulkan-tools", "libvulkan-dev", "vulkan-utils"
                ], check=True)
            elif backend == "GPU/CPU - CUDA":
                subprocess.run([
                    "sudo", "apt-get", "install", "-y",
                    "nvidia-cuda-toolkit"
                ], check=True)
        
        # Install Python packages with appropriate build flags
        env = os.environ.copy()
        if backend in BACKEND_OPTIONS:
            env.update(BACKEND_OPTIONS[backend]["build_flags"])
        
        cmd = [pip_exe, "install"] + REQUIREMENTS
        if backend == "GPU/CPU - CUDA":
            cmd.append("llama-cpp-python[cuda]")
        
        subprocess.run(cmd, check=True, env=env)
        print_status("All Python dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Dependency installation failed: {e}", False)
        return False

def install_vulkan_sdk() -> bool:
    if PLATFORM != "windows":
        return True
        
    print_status("Preparing to install Vulkan SDK...")
    vulkan_url = f"https://sdk.lunarg.com/sdk/download/{VULKAN_TARGET_VERSION}/windows/VulkanSDK-{VULKAN_TARGET_VERSION}-Installer.exe?Human=true"
    TEMP_DIR.mkdir(exist_ok=True)
    installer_path = TEMP_DIR / "VulkanSDK.exe"
    
    try:
        import requests
        from tqdm import tqdm
        
        print_status("Downloading Vulkan SDK...")
        response = requests.get(vulkan_url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(installer_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading Vulkan SDK") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print_status("Running Vulkan SDK installer...")
        try:
            subprocess.run([str(installer_path), "/S"], check=True)
            print_status("Vulkan SDK installation completed")
            return True
        except PermissionError:
            print_status("Permission denied: Run installer as administrator", False)
            return False
        finally:
            installer_path.unlink(missing_ok=True)
    except Exception as e:
        print_status(f"Vulkan SDK installation failed: {str(e)}", False)
        installer_path.unlink(missing_ok=True)
        return False

def download_extract_backend(backend: str) -> bool:
    if PLATFORM != "windows":
        return True
        
    print_status(f"Downloading llama.cpp ({backend})...")
    backend_info = BACKEND_OPTIONS[backend]
    TEMP_DIR.mkdir(exist_ok=True)
    temp_zip = TEMP_DIR / "llama.zip"
    
    try:
        import requests
        from tqdm import tqdm
        import zipfile
        
        print_status(f"Downloading {backend} backend...")
        response = requests.get(backend_info["url"], stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(temp_zip, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {backend}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        dest_path = BASE_DIR / backend_info["dest"]
        dest_path.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        
        cli_path = BASE_DIR / backend_info["cli_path"]
        if not cli_path.exists():
            raise FileNotFoundError(f"llama-cli.exe not found at {cli_path}")
        
        print_status(f"llama.cpp ({backend}) installed successfully")
        return True
    except Exception as e:
        print_status(f"Unexpected error: {str(e)}", False)
        return False
    finally:
        temp_zip.unlink(missing_ok=True)

def select_backend_type() -> None:
    global BACKEND_TYPE
    if PLATFORM == "windows":
        options = [
            "AVX2 - CPU Only - Must be compatible with AVX2",
            "AVX512 - CPU Only - Must be compatible with AVX512",
            "NoAVX - CPU Only - For older CPUs without AVX support",
            "OpenBLAS - CPU Only - Optimized for linear algebra operations",
            "Vulkan - GPU/CPU - For AMD/nVidia/Intel GPU with x64 CPU",
            "Hip-Radeon - GPU/CPU - Experimental Vulkan for AMD-ROCM",
            "CUDA 11.7 - GPU/CPU - For CUDA 11.7 GPUs with CPU fallback",
            "CUDA 12.4 - GPU/CPU - For CUDA 12.4 GPUs with CPU fallback"
        ]
        mapping = {
            options[0]: "CPU Only - AVX2",
            options[1]: "CPU Only - AVX512",
            options[2]: "CPU Only - NoAVX",
            options[3]: "CPU Only - OpenBLAS",
            options[4]: "GPU/CPU - Vulkan",
            options[5]: "GPU/CPU - HIP-Radeon",
            options[6]: "GPU/CPU - CUDA 11.7",
            options[7]: "GPU/CPU - CUDA 12.4"
        }
    else:
        options = [
            "CPU Only - Standard CPU processing",
            "GPU/CPU - Vulkan - For AMD/Intel/NVIDIA GPUs",
            "GPU/CPU - CUDA - For NVIDIA GPUs with CUDA"
        ]
        mapping = {
            options[0]: "CPU Only",
            options[1]: "GPU/CPU - Vulkan",
            options[2]: "GPU/CPU - CUDA"
        }
    
    choice = get_user_choice("Select the Llama.Cpp backend type:", options)
    BACKEND_TYPE = mapping[choice]

def install():
    print_header("Installation")
    print(f"Installing {APP_NAME} on {PLATFORM}...")
    
    if sys.version_info < (3, 8):
        print_status("Python 3.8 or higher required", False)
        sys.exit(1)
    
    print_status(f"Selected backend: {BACKEND_TYPE}")
    backend_info = BACKEND_OPTIONS[BACKEND_TYPE]
    
    # Check for required system components
    if PLATFORM == "windows" and backend_info.get("vulkan_required", False):
        if not install_vulkan_sdk():
            print_status("Vulkan SDK installation failed!", False)
            sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Set up virtual environment
    if not create_venv():
        print_status("Virtual environment creation failed!", False)
        sys.exit(1)
    
    # Install Python dependencies
    with activate_venv():
        if not install_python_deps(BACKEND_TYPE):
            print_status("Python dependency installation failed!", False)
            sys.exit(1)
    
    # Download backend binaries (Windows only)
    if PLATFORM == "windows" and not backend_info["needs_python_bindings"]:
        if not download_extract_backend(BACKEND_TYPE):
            print_status("llama.cpp backend installation failed!", False)
            sys.exit(1)
    
    # Create configuration
    create_config(BACKEND_TYPE)
    
    print_status(f"\n{APP_NAME} installation completed successfully!")
    print("\nYou can now run the application using:")
    print(f"  {'Chat-Gradio-Gguf.bat' if PLATFORM == 'windows' else './Chat-Gradio-Gguf.sh'}")

if __name__ == "__main__":
    set_platform()
    try:
        select_backend_type()
        install()
    except KeyboardInterrupt:
        print("\nInstallation cancelled")
        sys.exit(1)