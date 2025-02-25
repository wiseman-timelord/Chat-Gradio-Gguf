# Script: `.\installer.py`

# Imports...
import os, json, platform, subprocess, sys, contextlib, time
from pathlib import Path
from typing import Dict, Any, Optional

# Constants...
APP_NAME = "Chat-Gradio-Deep"
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
VULKAN_TARGET_VERSION = "1.4.304.1" 
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
    "python-magic==0.4.27",
    "requests==2.31.0",
    "tqdm==4.66.1",
    "httpx==0.27.0",
    "llama-cpp-python==0.2.61",
    "pygments==2.17.2",
    "sentence-transformers==2.2.2"  # Added for ContextInjector
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
        "vulkan_required": True  # Updated to ensure Vulkan SDK installation
    }
}
CONFIG_TEMPLATE = {
    "model_settings": {
        "model_path": "models/deepseek-r2-distill.Q4_K_M.gguf",
        "n_gpu_layers": 35,
        "n_ctx": 4096,
        "temperature": 0.7,
        "llama_cli_path": "",
        "use_python_bindings": True,
        "mmap": True,
        "mlock": False,
        "vram_size": 8192,
        "selected_gpu": None,
        "dynamic_gpu_layers": True
    },
    "ui_settings": {
        "font_name": "Arial",
        "font_size": 14,
        "background_color": "#1a1a1a",
        "accent_color": "#4CAF50"
    },
    "rag_settings": {
        "chunk_size": 2048,
        "chunk_overlap": 256,
        "max_docs": 5
    },
    "history_settings": {  # Add this section
        "max_sessions": 10
    },
    "backend_config": {
        "type": "",
        "llama_bin_path": ""
    }
}

# Functions...
def print_header(title: str) -> None:
    print(f"\n=== {title} ===")

def print_status(message: str, success: bool = True) -> None:
    status = "[OK]" if success else "[FAIL]"
    print(f" {message.ljust(60)} {status}")
    time.sleep(1 if success else 3)

def get_user_choice(prompt: str, options: list) -> str:
    print_header(prompt)
    for i, option in enumerate(options, 1):
        print(f" {i}. {option}")
    while True:
        choice = input(f"\n Enter your choice (1-{len(options)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print(" Invalid choice, please try again.")

def check_vulkan_support() -> tuple[bool, Optional[Path]]:
    """Returns (has_14x, install_path)"""
    vulkan_versions = find_vulkan_versions()
    
    # Check for any 1.4.x version
    for version, path in vulkan_versions.items():
        if version.startswith("1.4."):
            return (True, path)
    
    # If we have any versions (but not 1.4.x)
    if vulkan_versions:
        print("\nWARNING: Found Vulkan SDK versions but not 1.4.x:")
        for i, (ver, path) in enumerate(vulkan_versions.items(), 1):
            print(f" {i}. {ver} at {path}")
        
        while True:
            choice = input("\nChoose: [1-9] to use existing, [I] to install 1.4.x, [Q] to quit: ").strip().upper()
            
            if choice == "Q":
                sys.exit(0)
            elif choice == "I":
                return (False, None)
            elif choice.isdigit() and 1 <= int(choice) <= len(vulkan_versions):
                selected_version = list(vulkan_versions.values())[int(choice)-1]
                print(f"Using Vulkan SDK at {selected_version} - compatibility not guaranteed!")
                time.sleep(2)
                return (False, selected_version)
    
    return (False, None)

def find_vulkan_versions() -> Dict[str, Path]:
    """Search all possible paths for Vulkan SDK versions, returns {version: path}"""
    vulkan_versions = {}
    
    # Check environment variable first
    env_sdk = os.environ.get("VULKAN_SDK")
    if env_sdk:
        env_path = Path(env_sdk)
        dll_path = env_path / "Bin/vulkan-1.dll"
        if dll_path.exists():
            version = env_path.name
            vulkan_versions[version] = env_path
    
    # Check all possible installation paths
    for base_path in VULKAN_PATHS:
        if base_path.exists():
            for sdk_dir in base_path.iterdir():
                if sdk_dir.is_dir():
                    version = sdk_dir.name
                    dll_path = sdk_dir / "Bin/vulkan-1.dll"
                    if dll_path.exists():
                        vulkan_versions[version] = sdk_dir
    
    return vulkan_versions

def install_vulkan_sdk() -> Optional[Path]:
    """Returns installed path if successful"""
    print_header("Installing Vulkan SDK")
    vulkan_url = f"https://sdk.lunarg.com/sdk/download/{VULKAN_TARGET_VERSION}/windows/VulkanSDK-{VULKAN_TARGET_VERSION}-Installer.exe?Human=true"
    temp_dir = BASE_DIR / "temp"
    temp_dir.mkdir(exist_ok=True)
    installer_path = temp_dir / "VulkanSDK.exe"
    
    try:
        import requests
        from tqdm import tqdm
        
        # Download with progress bar
        response = requests.get(vulkan_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading Vulkan SDK")
        
        with open(installer_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        progress_bar.close()
        
        # Silent install
        print_status("Running Vulkan SDK installer...")
        result = subprocess.run([str(installer_path), "/S"], check=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"Installer exited with code {result.returncode}")
        
        # Verify installation
        installed_path = Path(f"C:/VulkanSDK/{VULKAN_TARGET_VERSION}")
        if (installed_path / "Bin/vulkan-1.dll").exists():
            print_status("Vulkan SDK installation verified")
            installer_path.unlink(missing_ok=True)  # Cleanup installer
            return installed_path
        else:
            print_status("Vulkan SDK files not found after installation", False)
            print("A system restart may be required.")
            return None
            
    except Exception as e:
        print_status(f"Installation failed: {str(e)}", False)
        return None


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

# --- Core Installation Functions ---
def create_directories() -> None:
    print_header("Creating Directory Structure")
    for dir_path in DIRECTORIES:
        full_path = BASE_DIR / dir_path
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            print_status(f"Created directory: {dir_path}")
        except Exception as e:
            print_status(f"Failed to create {dir_path}: {str(e)}", False)

def create_venv() -> bool:
    print_header("Creating Virtual Environment")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print_status("Virtual environment created at .venv")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to create venv: {e}", False)
        return False

def install_python_deps(backend: str) -> bool:
    print_header("Installing Python Dependencies")
    try:
        pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
        subprocess.run([pip_exe, "install", "--upgrade", "pip"], check=True)
        
        # Install base requirements
        subprocess.run([pip_exe, "install"] + REQUIREMENTS, check=True)
        
        # Install backend-specific bindings
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
    print_header(f"Downloading llama.cpp ({backend})")
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

def create_config(backend: str, vulkan_path: Optional[Path] = None) -> None:
    config_path = BASE_DIR / "data" / "config.json"
    config = copy.deepcopy(CONFIG_TEMPLATE)
    backend_info = BACKEND_OPTIONS[backend]
    # Update backend config
    config["backend_config"]["type"] = backend
    config["backend_config"]["llama_bin_path"] = backend_info["dest"]
    config["model_settings"]["llama_cli_path"] = backend_info["cli_path"]
    config["model_settings"]["use_python_bindings"] = backend_info["needs_python_bindings"]

    if vulkan_path:
        config["vulkan_sdk"] = {
            "path": str(vulkan_path),
            "bin_path": str(vulkan_path / "Bin"),
            "lib_path": str(vulkan_path / "Lib"),
            "version": vulkan_path.name
        }
    
    try:
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print_status("Configuration file created")
    except Exception as e:
        print_status(f"Failed to create config: {str(e)}", False)

def display_backend_menu() -> str:
    """Display the backend selection menu and return the user's choice."""
    print_header("Llama.Cpp Backend")
    print(" Information:...")
    print("     Vulkan - For AMD/nVidia/Intel GPU with x64 CPU fallback")
    print("     Kompute - Better than and similar to, Vulkan, but EXPERIMENTAL")
    print("     Avx 2 - CPU ONLY, must be compatible with AVX2 (slowest)")
    
    options = list(BACKEND_OPTIONS.keys())
    for i, option in enumerate(options, 1):
        print(f" {i}. {option}")
    
    while True:
        choice = input(f"\n Enter your choice (1-{len(options)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print(" Invalid choice, please try again.")

# --- Main Flow ---
def main():
    print(f"\n{APP_NAME} Installer\n{'='*40}")
    
    # Check system compatibility
    if platform.system() != "Windows":
        print("This installer is intended for Windows only.")
        time.sleep(2)
        sys.exit(1)
    
    if sys.version_info < (3, 8):
        print(" Python 3.8 or higher required")
        time.sleep(2)
        sys.exit(1)
        
    # Display backend menu and get user's choice
    backend_choice = display_backend_menu()
    backend_info = BACKEND_OPTIONS[backend_choice]
    
    # Check if selected backend requires Vulkan
    requires_vulkan = backend_info.get("vulkan_required", False)
    
    vulkan_path = None
    if requires_vulkan:
        has_14x, detected_path = check_vulkan_support()
        
        if detected_path:  # User selected existing non-1.4.x version
            vulkan_path = detected_path
        elif not has_14x:
            # Proceed with installation
            vulkan_path = install_vulkan_sdk()
            if not vulkan_path:
                print("Vulkan installation failed!")
                time.sleep(2)
                sys.exit(1)
    
    # Create directory structure
    create_directories()
    
    # Create virtual environment
    if not create_venv():
        time.sleep(2)
        sys.exit(1)
    
    # Activate venv and install dependencies
    with activate_venv():
        if not install_python_deps(backend_choice):
            time.sleep(2)
            sys.exit(1)
        
        # Check for llama-cpp-python conflicts if needed
        if backend_info.get("needs_python_bindings", False):
            if not check_llama_conflicts():
                time.sleep(2)
                sys.exit(1)
    
    # Download and extract backend binaries
    if not download_extract_backend(backend_choice):
        time.sleep(2)
        sys.exit(1)
    
    # Create configuration file
    config_path = BASE_DIR / "data" / "config.json"  # Corrected path
    create_config(backend_choice, vulkan_path)
    
    # Completion message
    print_header("Installation Complete")
    input(" Press Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n Installation cancelled")
        sys.exit(1)