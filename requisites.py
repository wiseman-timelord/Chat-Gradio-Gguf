# Script: `.\requisites.py` (Dual purpose script, installer or validation)

# Imports
import os
import json
import platform
import subprocess
import sys
import contextlib
import time
import copy
import pkg_resources
from pathlib import Path
from typing import Dict, Any, Optional

# Globals
APP_NAME = "Text-Gradio-Gguf"
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
TEMP_DIR = BASE_DIR / "data/temp"
VULKAN_TARGET_VERSION = "1.4.304.1"
LLAMACPP_TARGET_VERSION = "b4837"
BACKEND_TYPE = None  # Will be set by the backend menu
DIRECTORIES = [
    "data", "scripts", "models",
    "data/vectors", "data/history", "data/temp",
    "data/vectors/code",
    "data/vectors/rpg",
    "data/vectors/chat"
]
VULKAN_PATHS = [
    Path("C:/VulkanSDK"),
    Path("C:/Program Files/VulkanSDK"),
    Path("C:/Program Files (x86)/VulkanSDK"),
    Path("C:/progra~1/VulkanSDK"),
    Path("C:/progra~2/VulkanSDK"),
    Path("C:/programs/VulkanSDK"),
    Path("C:/program_filez/VulkanSDK"),
    Path("C:/drivers/VulkanSDK")
]
REQUIREMENTS = [
    "gradio>=4.25.0",
    "langchain==0.3.19",
    "langchain-core>=0.3.35",
    "faiss-cpu==1.8.0",
    "requests==2.31.0",
    "tqdm==4.66.1",
    "pyperclip",
    "yake",
    "psutil",
    "llama-cpp-python",
    "langchain-community==0.3.18",
    "pygments==2.17.2",
    "sentence-transformers==2.6.0",
    "langchain_huggingface==0.1.2"
]
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
    "GPU/CPU - Kompute": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-kompute-x64.zip",
        "dest": "data/llama-kompute-bin",
        "cli_path": "data/llama-kompute-bin/llama-cli.exe",
        "needs_python_bindings": False,
        "vulkan_required": True
    },
    "GPU/CPU - CUDA 11.7": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-cuda-cu11.7-x64.zip",
        "dest": "data/llama-cuda-11.7-bin",
        "cli_path": "data/llama-cuda-11.7-bin/llama-cli.exe",
        "needs_python_bindings": False,
        "cuda_required": True
    },
    "GPU/CPU - CUDA 12.4": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-cuda-cu12.4-x64.zip",
        "dest": "data/llama-cuda-12.4-bin",
        "cli_path": "data/llama-cuda-12.4-bin/llama-cli.exe",
        "needs_python_bindings": False,
        "cuda_required": True
    }
}
CONFIG_TEMPLATE = """{
    "model_settings": {
        "model_dir": "E:/Models/qwen2.5-7b-cabs-v0.4-GGUF",
        "model_name": "qwen2.5-7b-cabs-v0.4.Q3_K_M.gguf",
        "n_ctx": 24576,
        "temperature": 0.75,
        "repeat_penalty": 1.0,
        "use_python_bindings": false,
        "llama_cli_path": "data/llama-vulkan-bin/llama-cli.exe",
        "vram_size": 8192,
        "selected_gpu": "Radeon (TM) RX 470 Graphics",
        "selected_cpu": "CPU 0: AMD Ryzen 9 3900X 12-Core Processor",
        "mmap": true,
        "mlock": true,
        "n_batch": 4096,
        "dynamic_gpu_layers": true,
        "afterthought_time": true,
        "max_history_slots": 15,
        "max_attach_slots": 6,
        "session_log_height": 475,
        "input_lines": 5
    },
    "backend_config": {
        "type": "GPU/CPU - Vulkan",
        "llama_bin_path": "data/llama-vulkan-bin"
    },
    "rp_settings": {
        "rp_location": "Most of the way to the top of the large hill",
        "user_name": "Human",
        "user_role": "Hiker",
        "ai_npc": "Llama",
        "ai_npc_role": "A wise llama"
    }
}"""

# Utility Functions
def clear_screen() -> None:
    os.system('cls')

def print_header(title: str) -> None:
    clear_screen()
    print(f"{'='*120}\n    {APP_NAME}: {title}\n{'='*120}\n")

def print_status(message: str, success: bool = True) -> None:
    status = "[GOOD]" if success else "[FAIL]"
    print(f"{message.ljust(60)} {status}")
    time.sleep(1 if success else 3)

# Configuration
def get_user_choice(prompt: str, options: list) -> str:
    print_header("Install Options")
    print(f"\n\n {prompt}\n\b")
    for i, option in enumerate(options, 1):
        print(f"    {i}. {option}\n")
    print(f"\n\n{'='*120}")
    while True:
        choice = input(" Selection; Menu Options = 1-{}, Exit Installer = X: ".format(len(options))).strip().upper()
        if choice == "X":
            print("\nExiting installer...")
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print(" Invalid choice, please try again.")

# Virtual Environment Management
@contextlib.contextmanager
def activate_venv():
    if not VENV_DIR.exists():
        raise FileNotFoundError(f"Virtual environment not found at {VENV_DIR}")
    activate_script = VENV_DIR / "Scripts" / "activate.bat"
    if not activate_script.exists():
        raise FileNotFoundError(f"Virtual environment activation script not found at {activate_script}")
    old_path = os.environ["PATH"]
    old_python = sys.executable
    try:
        os.environ["PATH"] = f"{VENV_DIR / 'Scripts'}{os.pathsep}{old_path}"
        sys.executable = str(VENV_DIR / "Scripts" / "python.exe")
        yield
    finally:
        os.environ["PATH"] = old_path
        sys.executable = old_python

# Check Functions
def get_python_wheel_tag() -> str:
    major = sys.version_info.major
    minor = sys.version_info.minor
    return f"cp{major}{minor}"

def check_llama_conflicts() -> bool:
    try:
        import llama_cpp
        print_status("llama-cpp-python is installed and compatible")
        return True
    except ImportError:
        print_status("llama-cpp-python not found", False)
        return False
    except Exception as e:
        print_status(f"llama-cpp-python conflict detected: {str(e)}", False)
        return False

def find_vulkan_versions() -> Dict[str, Path]:
    vulkan_versions = {}
    env_sdk = os.environ.get("VULKAN_SDK")
    env_sdk_base = None
    
    if env_sdk:
        env_path = Path(env_sdk)
        env_sdk_base = env_path.parent
        lib_path = env_path / "Lib/vulkan-1.lib"
        if lib_path.exists():
            version = env_path.name
            vulkan_versions[version] = env_path
            print(f"Found Vulkan SDK at {env_path} with version {version}")
        else:
            print(f"No vulkan-1.lib at {lib_path}")
    
    for base_path in VULKAN_PATHS:
        if env_sdk_base and str(base_path.resolve()).lower() == str(env_sdk_base.resolve()).lower():
            continue
        if base_path.exists():
            for sdk_dir in base_path.iterdir():
                if sdk_dir.is_dir():
                    version = sdk_dir.name
                    lib_path = sdk_dir / "Lib/vulkan-1.lib"
                    if lib_path.exists():
                        vulkan_versions[version] = sdk_dir
                        print(f"Found Vulkan SDK at {sdk_dir} with version {version}")
    
    print(f"Detected Vulkan versions: {vulkan_versions}")
    return vulkan_versions

# Installation Functions
def create_directories() -> None:
    for dir_path in DIRECTORIES:
        full_path = BASE_DIR / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        # Verify directory permissions
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
    config["backend_config"]["type"] = backend
    config["backend_config"]["llama_bin_path"] = backend_info["dest"]
    config["model_settings"]["llama_cli_path"] = backend_info["cli_path"]
    config["model_settings"]["use_python_bindings"] = backend_info["needs_python_bindings"]
    # Save the config to file
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print_status("Configuration file created or updated")
    except Exception as e:
        print_status(f"Failed to create config: {str(e)}", False)

def create_venv() -> bool:
    try:
        if VENV_DIR.exists():
            import shutil
            shutil.rmtree(VENV_DIR)
            print_status("Removed existing virtual environment")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print_status("Created fresh virtual environment")
        python_exe = VENV_DIR / "Scripts" / "python.exe"
        if not python_exe.exists():
            raise FileNotFoundError(f"Python executable not found at {python_exe}")
        print_status("Verified virtual environment setup")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to create venv: {e}", False)
        return False
    except Exception as e:
        print_status(f"Unexpected error: {str(e)}", False)
        return False

def check_vulkan_support() -> bool:
    vulkan_versions = find_vulkan_versions()
    for version in vulkan_versions.keys():
        if version.startswith("1.4."):
            print(f"Confirmed Vulkan SDK 1.4.x version: {version}")
            return True
    
    if vulkan_versions:
        print("\nWARNING: Found Vulkan SDK versions but not 1.4.x:")
        for i, (ver, path) in enumerate(vulkan_versions.items(), 1):
            print(f" {i}. {ver} at {path}")
        while True:
            choice = input("\nChoose: [1-{}] to use existing, [I] to install 1.4.x, [Q] to quit: ".format(len(vulkan_versions))).strip().upper()
            if choice == "Q":
                sys.exit(0)
            elif choice == "I":
                return False
            elif choice.isdigit() and 1 <= int(choice) <= len(vulkan_versions):
                selected_version = list(vulkan_versions.keys())[int(choice)-1]
                print(f"Using Vulkan SDK version {selected_version} - compatibility not guaranteed!")
                time.sleep(2)
                return False
    return False

def download_extract_backend(backend: str) -> bool:
    print_status(f"Downloading llama.cpp ({backend})...")
    backend_info = BACKEND_OPTIONS[backend]
    try:
        TEMP_DIR.mkdir(exist_ok=True)
        temp_zip = TEMP_DIR / "llama.zip"
        import requests
        from tqdm import tqdm
        response = requests.get(backend_info["url"], stream=True)
        response.raise_for_status()
        with open(temp_zip, 'wb') as f:
            with tqdm(total=int(response.headers.get('content-length', 0)),
                      unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        import zipfile
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR / backend_info["dest"])
        if temp_zip.exists():
            temp_zip.unlink()
        if not (BASE_DIR / backend_info["cli_path"]).exists():
            raise FileNotFoundError(f"llama-cli.exe not found at {backend_info['cli_path']}")
        print_status("llama.cpp installed successfully")
        return True
    except requests.exceptions.RequestException as e:
        print_status(f"Download failed: {str(e)}", False)
        return False
    except zipfile.BadZipFile:
        print_status("Downloaded file is not a valid ZIP archive", False)
        return False
    except Exception as e:
        print_status(f"Unexpected error: {str(e)}", False)
        return False

def install_vulkan_sdk() -> bool:
    print_status("Installing Vulkan SDK...")
    vulkan_url = f"https://sdk.lunarg.com/sdk/download/{VULKAN_TARGET_VERSION}/windows/VulkanSDK-{VULKAN_TARGET_VERSION}-Installer.exe?Human=true"
    TEMP_DIR.mkdir(exist_ok=True)
    installer_path = TEMP_DIR / "VulkanSDK.exe"
    try:
        import requests
        from tqdm import tqdm
        response = requests.get(vulkan_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(installer_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading Vulkan SDK") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        result = subprocess.run([str(installer_path), "/S"], check=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Installer exited with code {result.returncode}")
        print_status("Vulkan SDK installation completed")
        installer_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        print_status(f"Installation failed: {str(e)}", False)
        return False

def install_python_deps(backend: str) -> bool:
    print_status("Installing Python Dependencies...")
    try:
        python_exe = str(VENV_DIR / "Scripts" / "python.exe")
        pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
        if not os.path.exists(pip_exe):
            raise FileNotFoundError(f"Pip not found at {pip_exe}. Venv creation may have failed.")
        print_status("Upgrading pip in venv...")
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Pip upgrade warnings/errors: {result.stderr}")
        print_status("Pip upgraded successfully")
        print_status("Installing dependencies with custom wheel index...")
        cmd = (
            [pip_exe, "install"] + REQUIREMENTS + [
                "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu",
                "--prefer-binary",
                "--verbose"
            ]
        )
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Installation warnings/errors: {result.stderr}")
        print_status("Verifying sentence-transformers installation...")
        check_result = subprocess.run(
            [python_exe, "-c", "import sentence_transformers; print('sentence-transformers installed')"],
            check=True,
            capture_output=True,
            text=True
        )
        print(check_result.stdout)
        if check_result.stderr:
            print(f"Verification warnings/errors: {check_result.stderr}")
        print_status("Dependencies installed in venv")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Dependency install failed: {e}", False)
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        if "llama_cpp_python" in str(e.stderr):
            print("Error: Could not install llama-cpp-python from pre-built wheels.")
            print(f"Python version: {sys.version}")
            print("Solutions:")
            print("- Ensure Python is 3.8-3.11.")
            print("- Install Visual Studio Build Tools for source build if needed.")
        elif "sentence-transformers" in str(e.stderr):
            print("Error: Could not install sentence-transformers.")
            print("Solutions:")
            print("- Check internet connection.")
            print(f"- Run manually: {pip_exe} install sentence-transformers==2.6.0")
        return False
    except Exception as e:
        print_status(f"Unexpected error: {str(e)}", False)
        return False

# Menu Functions
def select_backend_type() -> None:
    global BACKEND_TYPE
    options = [
        "AVX2 - CPU Only - Must be compatible with AVX2",
        "AVX512 - CPU Only - Must be compatible with AVX512",
        "NoAVX - CPU Only - For older CPUs without AVX support",
        "OpenBLAS - CPU Only - Optimized for linear algebra operations",
        "Vulkan - GPU/CPU - For AMD/nVidia/Intel GPU with x64 CPU",
        "Kompute - GPU/CPU - Experimental Vulkan for AMD/nVidia/Intel",
        "CUDA 11.7 - GPU/CPU - For CUDA 11.7 GPUs with CPU fallback",
        "CUDA 12.4 - GPU/CPU - For CUDA 12.4 GPUs with CPU fallback"
    ]
    mapping = {
        options[0]: "CPU Only - AVX2",
        options[1]: "CPU Only - AVX512",
        options[2]: "CPU Only - NoAVX",
        options[3]: "CPU Only - OpenBLAS",
        options[4]: "GPU/CPU - Vulkan",
        options[5]: "GPU/CPU - Kompute",
        options[6]: "GPU/CPU - CUDA 11.7",
        options[7]: "GPU/CPU - CUDA 12.4"
    }
    choice = get_user_choice("Select the Llama.Cpp type:", options)
    BACKEND_TYPE = mapping[choice]

# Main Installation Flow
def install():
    print_header("Installation")
    print(f"Installing {APP_NAME}...")
    if platform.system() != "Windows":
        print_status("This installer is intended for Windows only.", False)
        time.sleep(2)
        sys.exit(1)
    if sys.version_info < (3, 8):
        print_status("Python 3.8 or higher required", False)
        time.sleep(2)
        sys.exit(1)
    backend_info = BACKEND_OPTIONS[BACKEND_TYPE]
    requires_vulkan = backend_info.get("vulkan_required", False)
    if requires_vulkan and not check_vulkan_support():
        if not install_vulkan_sdk():
            print_status("Vulkan installation failed!", False)
            time.sleep(2)
            sys.exit(1)
    create_directories()
    if not create_venv():
        time.sleep(2)
        sys.exit(1)
    with activate_venv():
        if not install_python_deps(BACKEND_TYPE):
            time.sleep(2)
            sys.exit(1)
        if backend_info.get("needs_python_bindings", False):
            if not check_llama_conflicts():
                time.sleep(2)
                sys.exit(1)
    if not download_extract_backend(BACKEND_TYPE):
        time.sleep(2)
        sys.exit(1)
    create_config(BACKEND_TYPE)
    print_status("Install processes completed.")

# Test Libraries Function
def test_libraries():
    """
    Tests if each library in REQUIREMENTS is installed in the virtual environment.
    Prints library name with [GOOD] or [FAIL], then a final status message.
    Exits with code 0 if all succeed, 1 if any fail.
    """
    failed = False
    for req in REQUIREMENTS:
        try:
            pkg_resources.require(req)
            # do nothing 
        except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
            print(f"{req} [FAIL]")
            failed = True
    if failed:
        print("Error: Libraries incomplete (re-run installer).")
        sys.exit(1)
    else:
        print("Success: Python libraries verified.")
        sys.exit(0)

# Main Entry Point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python requisite.py [installer|testlibs]")
        sys.exit(1)
    arg = sys.argv[1]
    if arg == "installer":
        try:
            select_backend_type()
            install()
        except KeyboardInterrupt:
            print("\nInstallation cancelled")
            sys.exit(1)
    elif arg == "testlibs":
        test_libraries()
    else:
        print("Invalid argument. Use 'installer' or 'testlibs'.")
        sys.exit(1)