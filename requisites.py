# Script: `.\requisites.py` (Dual purpose script, installer or validation)

# Imports
import os, json, platform, subprocess, sys, contextlib, time, copy
import pkg_resources
from pathlib import Path
from typing import Dict, Any, Optional

# Globals
APP_NAME = "Chat-Gradio-Gguf"
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
TEMP_DIR = BASE_DIR / "data/temp"
VULKAN_TARGET_VERSION = "1.4.304.1"
LLAMACPP_TARGET_VERSION = "b5596"
BACKEND_TYPE = None  # Will be set by the backend menu
DIRECTORIES = [
    "data", "scripts", "models",
    "data/history", "data/temp"
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
    "requests==2.31.0",
    "pyperclip",
    "yake",
    "psutil",
    "pywin32",
    "duckduckgo-search",
    "newspaper3k",
    "llama-cpp-python",
    "langchain-community==0.3.18",
    "pygments==2.17.2",
    "lxml_html_clean",  # Added to support newspaper3k's HTML cleaning
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
    "GPU/CPU - HIP-Radeon": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-hip-radeon-x64.zip",  # Updated URL pattern
        "dest": "data/llama-hip-radeon-bin",
        "cli_path": "data/llama-hip-radeon-bin/llama-cli.exe",
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
    venv_python = str(VENV_DIR / "Scripts" / "python.exe")
    try:
        result = subprocess.run(
            [venv_python, "-c", "import llama_cpp"],
            capture_output=True,
            text=True,
            check=True
        )
        print_status("llama-cpp-python is installed and compatible")
        return True
    except subprocess.CalledProcessError as e:
        if "ModuleNotFoundError" in e.stderr:
            print_status("llama-cpp-python not found", False)
        else:
            print_status(f"llama-cpp-python conflict: {e.stderr}", False)
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

def check_vulkan_support() -> bool:
    """
    Check if a Vulkan SDK version 1.4.x is installed or prompt user to install it.
    
    Returns:
        bool: True if Vulkan 1.4.x is found or user accepts an existing version, False otherwise.
    """
    vulkan_versions = find_vulkan_versions()
    for version in vulkan_versions.keys():
        if version.startswith("1.4."):
            print_status(f"Confirmed Vulkan SDK 1.4.x version: {version}")
            return True
    
    if vulkan_versions:
        print("\nWARNING: Found Vulkan SDK versions but not 1.4.x:")
        for i, (ver, path) in enumerate(vulkan_versions.items(), 1):
            print(f" {i}. {ver} at {path}")
        while True:
            choice = input("\nChoose: [1-{}] to use existing, [I] to install 1.4.x, [Q] to quit: ".format(len(vulkan_versions))).strip().upper()
            if choice == "Q":
                print_status("User chose to exit due to Vulkan version mismatch", False)
                sys.exit(0)
            elif choice == "I":
                return False
            elif choice.isdigit() and 1 <= int(choice) <= len(vulkan_versions):
                selected_version = list(vulkan_versions.keys())[int(choice)-1]
                print_status(f"Using Vulkan SDK version {selected_version} - compatibility not guaranteed!", False)
                time.sleep(2)
                return True  # Allow proceeding with non-1.4.x version
    print_status("No Vulkan SDK found", False)
    return False

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
    config["backend_config"]["backend_type"] = backend
    config["backend_config"]["llama_bin_path"] = backend_info["dest"]
    config["model_settings"]["llama_cli_path"] = backend_info["cli_path"]
    config["model_settings"]["use_python_bindings"] = backend_info["needs_python_bindings"]
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

def install_vulkan_sdk() -> bool:
    """
    Download and install Vulkan SDK version 1.4.x.
    
    Returns:
        bool: True if installation succeeds, False otherwise.
    """
    print_status("Preparing to install Vulkan SDK...")
    vulkan_url = f"https://sdk.lunarg.com/sdk/download/{VULKAN_TARGET_VERSION}/windows/VulkanSDK-{VULKAN_TARGET_VERSION}-Installer.exe?Human=true"
    TEMP_DIR.mkdir(exist_ok=True)
    installer_path = TEMP_DIR / "VulkanSDK.exe"
    try:
        import requests
        from tqdm import tqdm
        print_status("Downloading Vulkan SDK...")
        for attempt in range(3):  # Retry up to 3 times
            try:
                response = requests.get(vulkan_url, stream=True, timeout=30)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                with open(installer_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading Vulkan SDK") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                break
            except requests.exceptions.RequestException as e:
                print_status(f"Download attempt {attempt + 1} failed: {str(e)}", False)
                if attempt == 2:
                    print_status("All download attempts failed", False)
                    return False
                time.sleep(5)
        
        print_status("Running Vulkan SDK installer...")
        try:
            result = subprocess.run([str(installer_path), "/S"], check=True, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"Installer exited with code {result.returncode}")
            print_status("Vulkan SDK installation completed")
        except PermissionError:
            print_status("Permission denied: Run installer as administrator", False)
            return False
        finally:
            installer_path.unlink(missing_ok=True)
        return True
    except Exception as e:
        print_status(f"Vulkan SDK installation failed: {str(e)}", False)
        installer_path.unlink(missing_ok=True)
        return False

def download_extract_backend(backend: str) -> bool:
    """
    Download and extract the llama.cpp backend binary for the specified backend.
    
    Args:
        backend (str): The selected backend type (e.g., "GPU/CPU - Vulkan").
    
    Returns:
        bool: True if download and extraction succeed, False otherwise.
    """
    print_status(f"Downloading llama.cpp ({backend})...")
    backend_info = BACKEND_OPTIONS[backend]
    TEMP_DIR.mkdir(exist_ok=True)
    temp_zip = TEMP_DIR / "llama.zip"
    try:
        import requests
        from tqdm import tqdm
        for attempt in range(3):  # Retry up to 3 times
            try:
                response = requests.get(backend_info["url"], stream=True, timeout=30)
                response.raise_for_status()
                total_size = int(response.headers.get('content-length', 0))
                with open(temp_zip, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {backend}") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))
                break
            except requests.exceptions.RequestException as e:
                print_status(f"Download attempt {attempt + 1} failed: {str(e)}", False)
                if attempt == 2:
                    print_status("All download attempts failed", False)
                    return False
                time.sleep(5)
        
        import zipfile
        dest_path = BASE_DIR / backend_info["dest"]
        dest_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(dest_path)
        temp_zip.unlink(missing_ok=True)
        
        cli_path = BASE_DIR / backend_info["cli_path"]
        if not cli_path.exists():
            raise FileNotFoundError(f"llama-cli.exe not found at {cli_path}")
        print_status(f"llama.cpp ({backend}) installed successfully")
        return True
    except zipfile.BadZipFile:
        print_status("Downloaded file is not a valid ZIP archive", False)
        temp_zip.unlink(missing_ok=True)
        return False
    except Exception as e:
        print_status(f"Unexpected error: {str(e)}", False)
        temp_zip.unlink(missing_ok=True)
        return False

def install_python_deps(backend: str) -> bool:
    """
    Install Python dependencies in the virtual environment.
    
    Args:
        backend (str): The selected backend type (e.g., "GPU/CPU - Vulkan").
    
    Returns:
        bool: True if all dependencies are installed successfully, False otherwise.
    """
    print_status("Installing Python dependencies...")
    try:
        python_exe = str(VENV_DIR / "Scripts" / "python.exe")
        pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
        if not os.path.exists(pip_exe):
            raise FileNotFoundError(f"Pip not found at {pip_exe}. Virtual environment creation may have failed.")
        
        print_status("Upgrading pip in virtual environment...")
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Pip upgrade warnings: {result.stderr}")
        print_status("Pip upgraded successfully")
        
        print_status("Installing dependencies...")
        cmd = [
            pip_exe, "install",
            "--no-warn-script-location",  # Suppress script location warnings
            "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu",
            "--prefer-binary",
            "--verbose"
        ] + REQUIREMENTS
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Installation warnings: {result.stderr}")
        print_status("All dependencies installed successfully")
        
        # Verify critical dependencies with retries
        print_status("Verifying critical dependencies...")
        with activate_venv():
            import site
            import sys
            # Ensure site-packages is in sys.path
            site.addsitedir(str(VENV_DIR / "Lib" / "site-packages"))
            print(f"sys.path during verification: {sys.path}")
            
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    import duckduckgo_search
                    import newspaper
                    import llama_cpp
                    print_status("Verified critical dependencies: duckduckgo-search, newspaper3k, llama-cpp-python")
                    return True
                except ImportError as e:
                    print(f"Verification attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                    if attempt < max_attempts - 1:
                        time.sleep(1)  # Wait before retrying
                        continue
                    # Final failure
                    site_packages = str(VENV_DIR / "Lib" / "site-packages")
                    print_status(
                        f"Critical dependency verification failed: {str(e)}\n"
                        f"sys.path: {sys.path}\n"
                        f"Expected site-packages: {site_packages}\n"
                        f"Check if packages are installed in {site_packages} with: {pip_exe} list",
                        False
                    )
                    return False
    except subprocess.CalledProcessError as e:
        print_status(f"Dependency installation failed: {e}", False)
        print(f"Command output: {e.stdout}")
        print(f"Command error: {e.stderr}")
        print("Solutions:")
        print("- Ensure a stable internet connection.")
        print("- Verify Python version is 3.8-3.11.")
        print(f"- Run manually: {pip_exe} install {' '.join(REQUIREMENTS)}")
        if "llama_cpp_python" in str(e.stderr):
            print("- Install Visual Studio Build Tools for source build if needed.")
        return False
    except Exception as e:
        print_status(f"Unexpected error during dependency installation: {str(e)}", False)
        return False

# Main Installation Flow
def install():
    """
    Main installation function to set up the Chat-Gradio-Gguf application.
    """
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
    
    print_status("Selected backend: " + BACKEND_TYPE)
    backend_info = BACKEND_OPTIONS[BACKEND_TYPE]
    requires_vulkan = backend_info.get("vulkan_required", False)
    
    if requires_vulkan and not check_vulkan_support():
        if not install_vulkan_sdk():
            print_status("Vulkan SDK installation failed!", False)
            time.sleep(2)
            sys.exit(1)
        else:
            print_status("Vulkan SDK installed successfully")
    
    print_status("Creating required directories...")
    create_directories()
    
    print_status("Setting up virtual environment...")
    if not create_venv():
        print_status("Virtual environment creation failed!", False)
        time.sleep(2)
        sys.exit(1)
    
    print_status("Installing Python dependencies...")
    with activate_venv():
        if not install_python_deps(BACKEND_TYPE):
            print_status("Python dependency installation failed!", False)
            time.sleep(2)
            sys.exit(1)
        if backend_info.get("needs_python_bindings", False):
            print_status("Verifying llama-cpp-python compatibility...")
            if not check_llama_conflicts():
                print_status("llama-cpp-python verification failed!", False)
                time.sleep(2)
                sys.exit(1)
    
    print_status("Downloading and installing llama.cpp backend...")
    if not download_extract_backend(BACKEND_TYPE):
        print_status("llama.cpp backend installation failed!", False)
        time.sleep(2)
        sys.exit(1)
    
    print_status("Creating configuration file...")
    create_config(BACKEND_TYPE)
    
    print_status(f"\n{APP_NAME} installation completed successfully!")
    time.sleep(2)

# Menu Functions
def select_backend_type() -> None:
    global BACKEND_TYPE
    options = [
        "AVX2 - CPU Only - Must be compatible with AVX2",
        "AVX512 - CPU Only - Must be compatible with AVX512",
        "NoAVX - CPU Only - For older CPUs without AVX support",
        "OpenBLAS - CPU Only - Optimized for linear algebra operations",
        "Vulkan - GPU/CPU - For AMD/nVidia/Intel GPU with x64 CPU",
        "HIP-Radeon - GPU/CPU - Experimental Vulkan for AMD-ROCM",
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
    choice = get_user_choice("Select the Llama.Cpp type:", options)
    BACKEND_TYPE = mapping[choice]

# Test Libraries Function
def test_libraries():
    """
    Test if each library in REQUIREMENTS is installed in the virtual environment.
    Prints library name with [GOOD] or [FAIL], then a final status message.
    Exits with code 0 if all succeed, 1 if any fail.
    """
    failed = False
    pip_exe = str(VENV_DIR / "Scripts" / "pip.exe")
    for req in REQUIREMENTS:
        try:
            pkg_resources.require(req)
            print_status(f"{req}")
        except pkg_resources.DistributionNotFound:
            print_status(f"{req} not found", False)
            print(f"Install it with: {pip_exe} install {req}")
            failed = True
        except pkg_resources.VersionConflict as e:
            print_status(f"{req} version mismatch: {str(e)}", False)
            print(f"Install correct version with: {pip_exe} install {req}")
            failed = True
    if failed:
        print_status("Error: Some libraries are missing or have incorrect versions", False)
        print("Re-run the installer with: python requisites.py installer")
        sys.exit(1)
    else:
        print_status("Success: All Python libraries verified")
        sys.exit(0)

# Main Entry Point
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python requisites.py [installer|testlibs]")
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