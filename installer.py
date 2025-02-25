# Script: `.\installer.py`

# Imports...
import os, json, platform, subprocess, sys, contextlib, time, copy
from pathlib import Path
from typing import Dict, Any, Optional

# Constants...
APP_NAME = "Chat-Gradio-Gguf"
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
VULKAN_TARGET_VERSION = "1.4.304.1"  # Consider updating if this version isn’t available
DIRECTORIES = [
    "data", "files", "temp", "scripts",
    "data/models", "data/vectorstores"
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
    "gradio==4.30.0",
    "langchain==0.2.1",
    "faiss-cpu==1.8.0",
    "requests==2.31.0",
    "tqdm==4.66.1",
    "llama-cpp-python==0.2.61",
    "pygments==2.17.2",
    "sentence-transformers==2.2.2"
]
BACKEND_OPTIONS = {
    "CPU Only - AVX2": {
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b4762/llama-b4762-bin-win-avx2-x64.zip",
        "dest": "data/llama-avx2-bin",
        "cli_path": "data/llama-avx2-bin/bin/llama-cli.exe",
        "needs_python_bindings": True
    },
    "GPU/CPU - Vulkan": {
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b4762/llama-b4762-bin-win-vulkan-x64.zip",
        "dest": "data/llama-vulkan-bin",
        "cli_path": "data/llama-vulkan-bin/bin/llama-cli.exe",
        "needs_python_bindings": False,
        "vulkan_required": True
    },
    "GPU/CPU - Kompute": {
        "url": "https://github.com/ggml-org/llama.cpp/releases/download/b4762/llama-b4762-bin-win-kompute-x64.zip",
        "dest": "data/llama-kompute-bin",
        "cli_path": "data/llama-kompute-bin/bin/llama-cli.exe",
        "needs_python_bindings": False,
        "vulkan_required": True
    }
}
CONFIG_TEMPLATE = {
    "model_settings": {
        "model_path": "models/Lamarckvergence-14B-GGUF",
        "n_gpu_layers": 35,
        "n_ctx": 8192,
        "temperature": 0.7,
        "llama_cli_path": "",
        "use_python_bindings": True,
        "mmap": True,
        "mlock": False,
        "vram_size": 8192,
        "selected_gpu": None,
        "dynamic_gpu_layers": True
    },
    "rag_settings": {
        "chunk_size": 2048,
        "chunk_overlap": 256,
        "max_docs": 5
    },
    "history_settings": {
        "max_sessions": 10
    },
    "backend_config": {
        "type": "",
        "llama_bin_path": ""
    }
}

# Global Configuration Variables
BACKEND_TYPE = None  # Will be set by the backend menu

# Utility Functions...
def clear_screen() -> None:
    os.system('cls')  # Clears the screen on Windows

def print_header(title: str) -> None:
    clear_screen()    # <<<<------- added here
    print(f"{'='*120}\n    {APP_NAME}: {title}\n{'='*120}\n")

def print_status(message: str, success: bool = True) -> None:
    status = "[OK]" if success else "[FAIL]"
    print(f"{message.ljust(60)} {status}")
    time.sleep(1 if success else 3)

def get_user_choice(prompt: str, options: list) -> str:
    print_header("Install Options")
    print(f"\n\n\n\n\n\n\n {prompt}\n\b")  # Added extra newlines before options
    for i, option in enumerate(options, 1):
        print(f"    {i}. {option}\n")  # Added newline after each option
    print(f"\n\n\n\n\n\n\n{'='*120}")  # Added extra newlines before bottom border
    while True:
        choice = input(" Selection; Menu Options = 1-{}, Exit Installer = X: ".format(len(options))).strip().upper()
        if choice == "X":
            print("\nExiting installer...")
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print(" Invalid choice, please try again.")

# Installation Functions...
def find_vulkan_versions() -> Dict[str, Path]:
    vulkan_versions = {}
    env_sdk = os.environ.get("VULKAN_SDK")
    if env_sdk:
        env_path = Path(env_sdk)
        lib_path = env_path / "Lib/vulkan-1.lib"  # Check Lib for SDK presence
        print(f"Checking VULKAN_SDK: {env_path}")
        if lib_path.exists():
            version = env_path.name
            vulkan_versions[version] = env_path
            print(f"Found Vulkan SDK at {env_path} with version {version}")
        else:
            print(f"No vulkan-1.lib at {lib_path}")
    
    for base_path in VULKAN_PATHS:
        print(f"Checking base path: {base_path}")
        if base_path.exists():
            print(f"Base path exists: {base_path}")
            for sdk_dir in base_path.iterdir():
                if sdk_dir.is_dir():
                    version = sdk_dir.name
                    lib_path = sdk_dir / "Lib/vulkan-1.lib"
                    print(f"Checking directory: {sdk_dir}")
                    if lib_path.exists():
                        vulkan_versions[version] = sdk_dir
                        print(f"Found Vulkan SDK at {sdk_dir} with version {version}")
                    else:
                        print(f"No vulkan-1.lib at {lib_path}")
        else:
            print(f"Base path does not exist: {base_path}")
    print(f"Detected Vulkan versions: {vulkan_versions}")
    return vulkan_versions

def check_vulkan_support() -> bool:  # Now returns only a boolean
    vulkan_versions = find_vulkan_versions()
    # Check for any 1.4.x version
    for version in vulkan_versions.keys():
        if version.startswith("1.4."):
            print(f"Confirmed Vulkan SDK 1.4.x version: {version}")
            return True
    
    # If no 1.4.x but other versions exist, prompt user
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
                return False  # Still proceed with install if < 1.4.x
    return False

def install_vulkan_sdk() -> bool:  # Returns success/failure
    print_status("Installing Vulkan SDK...")
    vulkan_url = f"https://sdk.lunarg.com/sdk/download/{VULKAN_TARGET_VERSION}/windows/VulkanSDK-{VULKAN_TARGET_VERSION}-Installer.exe?Human=true"
    temp_dir = BASE_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    installer_path = temp_dir / "VulkanSDK.exe"
    
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

def create_directories() -> None:
    print_status("Creating Directory Structure...")
    for dir_path in DIRECTORIES:
        full_path = BASE_DIR / dir_path
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            print_status(f"Created directory: {dir_path}")
        except Exception as e:
            print_status(f"Failed to create {dir_path}: {str(e)}", False)

def create_venv() -> bool:
    print_status("Creating Virtual Environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print_status("Virtual environment created at .venv")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to create venv: {e}", False)
        return False

def install_python_deps(backend: str) -> bool:
    print_status("Installing Python Dependencies...")
    try:
        pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
        subprocess.run([pip_exe, "install", "--upgrade", "pip"], check=True)
        subprocess.run([pip_exe, "install"] + REQUIREMENTS, check=True)
        backend_info = BACKEND_OPTIONS[backend]
        if backend_info.get("needs_python_bindings", False):
            if "Vulkan" in backend:
                subprocess.run([
                    pip_exe, "install", "llama-cpp-python",
                    "--no-binary", "llama-cpp-python",
                    "--config-settings=cmake.define.LLAMA_VULKAN=ON"
                ], check=True)
        print_status("Dependencies installed in venv")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Dependency install failed: {e}", False)
        return False

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

def download_extract_backend(backend: str) -> bool:
    print_status(f"Downloading llama.cpp ({backend})...")
    backend_info = BACKEND_OPTIONS[backend]
    try:
        temp_dir = BASE_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_zip = temp_dir / "llama.zip"
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

def create_config(backend: str) -> None:  # No vulkan_path needed
    print_status("Creating configuration file...")
    config_path = BASE_DIR / "data" / "config.json"
    config = copy.deepcopy(CONFIG_TEMPLATE)
    backend_info = BACKEND_OPTIONS[backend]
    config["backend_config"]["type"] = backend
    config["backend_config"]["llama_bin_path"] = backend_info["dest"]
    config["model_settings"]["llama_cli_path"] = backend_info["cli_path"]
    config["model_settings"]["use_python_bindings"] = backend_info["needs_python_bindings"]
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print_status("Configuration file created")
    except Exception as e:
        print_status(f"Failed to create config: {str(e)}", False)

# Menu Functions...
def select_backend_type() -> None:
    global BACKEND_TYPE
    options = [
        "AVX2 - CPU Only - Must be compatible with AVX2 (slowest)",
        "Vulkan - GPU/CPU - For AMD/nVidia/Intel GPU with x64 CPU fallback",
        "Kompute - Enhanced Vulkan (experimental)"
    ]
    mapping = {
        options[0]: "CPU Only - AVX2",
        options[1]: "GPU/CPU - Vulkan",
        options[2]: "GPU/CPU - Kompute"
    }
    choice = get_user_choice("Select the Llama.Cpp Backend Type...", options)
    BACKEND_TYPE = mapping[choice]

# Main Installation Flow...
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
    
    create_config(BACKEND_TYPE)  # No vulkan_path passed
    
    print_status(f"{APP_NAME} installed successfully!")
    input(" Press Enter to exit...")

# Main Entry Point...
def main():
    # Step 1: Configuration Menus
    select_backend_type()
    
    # Step 2: Proceed to Installation
    install()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInstallation cancelled")
        sys.exit(1)