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

# Constants
APP_NAME = "Chat-Gradio-Gguf"
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
TEMP_DIR = BASE_DIR / "data/temp"
LLAMACPP_TARGET_VERSION = "b5587"
BACKEND_TYPE = None
PLATFORM = None  # Initialize PLATFORM variable

DIRECTORIES = [
    "data", "scripts", "models",
    "data/history", "data/temp"
]

# Set Platform
def set_platform():
    global PLATFORM
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ["windows", "linux"]:
        print("ERROR: Platform argument required (windows/linux)")
        sys.exit(1)
    PLATFORM = sys.argv[1].lower()

# Initialize platform before using it
set_platform()

# Platform-specific configurations
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
            "vulkan_required": True,
            "build_flags": {}
        },
        "GPU/CPU - CUDA 11.7": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-cuda-11.7-x64.zip",
            "dest": "data/llama-cuda-11.7-bin",
            "cli_path": "data/llama-cuda-11.7-bin/llama-cli.exe",
            "needs_python_bindings": False,
            "cuda_required": True,
            "build_flags": {}
        },
        "GPU/CPU - CUDA 12.4": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-cuda-12.4-x64.zip",
            "dest": "data/llama-cuda-12.4-bin",
            "cli_path": "data/llama-cuda-12.4-bin/llama-cli.exe",
            "needs_python_bindings": False,
            "cuda_required": True,
            "build_flags": {}
        }
    }
elif PLATFORM == "linux":
    BACKEND_OPTIONS = {
        "GPU/CPU - Vulkan": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-ubuntu-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli",
            "needs_python_bindings": False,
            "vulkan_required": True,
            "build_flags": {}
        },
        "GPU/CPU - CUDA": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-ubuntu-cuda-x64.zip",
            "dest": "data/llama-cuda-bin",
            "cli_path": "data/llama-cuda-bin/llama-cli",
            "needs_python_bindings": False,
            "cuda_required": True,
            "build_flags": {}
        }
    }

# Python dependencies
REQUIREMENTS = [
    "gradio>=4.25.0",
    "requests==2.31.0",
    "pyperclip",
    "yake",
    "psutil",
    "ddgs",
    "newspaper3k",
    "llama-cpp-python",
    "langchain-community==0.3.18",
    "pygments==2.17.2",
    "lxml[html_clean]",
    "pyttsx3"
]

# Add platform-specific requirements
if PLATFORM == "windows":
    REQUIREMENTS.append("pywin32")
    REQUIREMENTS.append("tk") 
elif PLATFORM == "linux":
    REQUIREMENTS.append("python3-tk")  # Required for tkinter

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

def build_config(backend: str) -> dict:
    """
    Build a complete, backend-specific configuration dictionary
    that will be written to data/persistent.json
    """
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
            "print_raw_model_output": False,
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

def simple_progress_bar(current: int, total: int, width: int = 25) -> str:
    """Simple progress bar without external dependencies"""
    if total == 0:
        return "[" + "=" * width + "] 100%"
    
    filled_width = int(width * current // total)
    bar = "=" * filled_width + "-" * (width - filled_width)
    percent = 100 * current // total
    
    # Format file sizes
    def format_bytes(bytes_val):
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f}{unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f}TB"
    
    return f"[{bar}] {percent}% ({format_bytes(current)}/{format_bytes(total)})"

def download_with_progress(url: str, filepath: Path, description: str = "Downloading") -> None:
    """Download file with progress display using only built-in libraries"""
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
                    
                    # Update progress every 0.35 seconds or when complete
                    current_time = time.time()
                    if current_time - last_update > 0.35 or downloaded >= total_size:
                        if total_size > 0:
                            progress = simple_progress_bar(downloaded, total_size)
                            print(f"\r{description}: {progress}", end='', flush=True)
                        else:
                            # Unknown size, show downloaded amount
                            downloaded_mb = downloaded / 1024 / 1024
                            print(f"\r{description}: {downloaded_mb:.1f} MB downloaded", end='', flush=True)
                        last_update = current_time
        
        print()  # New line after progress bar
        
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise e

def check_package_exists(package_name: str) -> bool:
    """Check if a package exists in the repositories"""
    try:
        result = subprocess.run(
            ["apt-cache", "search", "--names-only", f"^{package_name}$"],
            capture_output=True, text=True, check=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return False

def is_vulkan_installed() -> bool:
    """Check if Vulkan is properly installed"""
    if PLATFORM == "windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Khronos\Vulkan\Drivers") as key:
                return True
        except:
            return False
    else:  # Linux
        try:
            # Check both vulkaninfo and library presence
            result1 = subprocess.run(["vulkaninfo", "--summary"], 
                                  stdout=subprocess.DEVNULL, 
                                  stderr=subprocess.DEVNULL)
            result2 = subprocess.run(["ldconfig", "-p", "|", "grep", "libvulkan"],
                                  shell=True,
                                  stdout=subprocess.DEVNULL)
            return result1.returncode == 0 or result2.returncode == 0
        except FileNotFoundError:
            return False

def is_cuda_installed() -> bool:
    """Check if CUDA is properly installed"""
    if PLATFORM == "windows":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\NVIDIA Corporation\CUDA") as key:
                return True
        except:
            return False
    else:  # Linux
        try:
            result = subprocess.run(["nvcc", "--version"], 
                                 stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except FileNotFoundError:
            return False

def verify_backend_dependencies(backend: str) -> bool:
    """Check if required dependencies are installed for selected backend"""
    backend_info = BACKEND_OPTIONS[backend]
    
    if backend_info.get("vulkan_required", False):
        if not is_vulkan_installed():
            print("\n" + "!" * 80)
            print(f"Vulkan not detected but required for {backend} backend!")
            print("Please install Vulkan SDK before continuing:")
            if PLATFORM == "windows":
                print("  Download from: https://vulkan.lunarg.com/sdk/home")
            else:
                print("  Install with: sudo apt install vulkan-tools libvulkan-dev")
            print("!" * 80 + "\n")
            return False
    
    if backend_info.get("cuda_required", False):
        if not is_cuda_installed():
            print("\n" + "!" * 80)
            print(f"CUDA not detected but required for {backend} backend!")
            print("Please install CUDA Toolkit before continuing:")
            if PLATFORM == "windows":
                print("  Download from: https://developer.nvidia.com/cuda-downloads")
            else:
                print("  Install with: sudo apt install nvidia-cuda-toolkit")
            print("!" * 80 + "\n")
            return False
    
    return True

def install_linux_system_dependencies(backend: str) -> bool:
    """Install Linux system dependencies with improved error handling"""
    if PLATFORM != "linux":
        return True
    
    print_status("Installing Linux system dependencies...")
    
    essential_packages = [
        "build-essential",
        "portaudio19-dev",
        "libasound2-dev",
        "python3-tk",
        "vulkan-tools",
        "libvulkan-dev",
        "espeak",
        "libespeak-dev",
        "ffmpeg"  # For potential audio processing
    ]
    
    optional_packages = [
        "pulseaudio",
        "libpulse-dev"
    ]
    
    backend_packages = []
    if backend == "GPU/CPU - Vulkan":
        backend_packages = [
            "mesa-utils",
            "vulkan-utils",
            "libvulkan1",
            "vulkan-validationlayers"
        ]
    elif backend == "GPU/CPU - CUDA":
        backend_packages = [
            "nvidia-cuda-toolkit",
            "nvidia-driver-535"  # Or appropriate driver version
        ]
    
    # Unified installation process
    all_packages = essential_packages + optional_packages + backend_packages
    unique_packages = list(set(all_packages))  # Remove duplicates
    
    try:
        # Update package lists
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        
        # Install all packages in one command for efficiency
        install_cmd = ["sudo", "apt-get", "install", "-y"] + unique_packages
        subprocess.run(install_cmd, check=True)
        
        print_status("All Linux dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to install system dependencies: {str(e)}", False)
        print("\nYou may need to install these packages manually:")
        print("  sudo apt-get install " + " ".join(unique_packages))
        return False

def install_python_deps(backend: str) -> bool:
    print_status("Installing Python dependencies...")
    try:
        python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python"))
        pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("pip.exe" if PLATFORM == "windows" else "pip"))
        
        if not os.path.exists(pip_exe):
            raise FileNotFoundError(f"Pip not found at {pip_exe}")
        
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print_status("Upgraded pip successfully")

        # Linux-specific system dependencies
        if PLATFORM == "linux":
            try:
                # Install system packages for tkinter
                print_status("Installing system dependencies for tkinter...")
                subprocess.run([
                    "sudo", "apt-get", "install", "-y",
                    "python3-tk",
                    "python3.13-tk"  # Specific package for Python 3.13
                ], check=True)
                print_status("System dependencies for tkinter installed")

                # Verify tkinter works with Python 3.13
                print_status("Verifying tkinter installation...")
                subprocess.run([python_exe, "-c", "import tkinter; print('Tkinter verified')"], check=True)
                print_status("Tkinter verified with Python 3.13")
            except subprocess.CalledProcessError as e:
                print_status(f"Failed to install system dependencies for tkinter: {e}", False)
                # Continue installation but tkinter may not work
                
        # Windows-specific tk installation
        elif PLATFORM == "windows":
            try:
                subprocess.run([pip_exe, "install", "tk"], check=True)
                print_status("Installed tk for Windows")
            except subprocess.CalledProcessError as e:
                print_status(f"Failed to install tk for Windows: {e}", False)

        # Install Python packages (excluding python3-tk)
        requirements = [req for req in REQUIREMENTS if req != "python3-tk"]
        
        print_status("Installing Python packages...")
        for req in requirements:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    subprocess.run([pip_exe, "install", req], check=True)
                    print_status(f"Installed {req}")
                    break
                except subprocess.CalledProcessError as e:
                    if attempt == max_retries - 1:
                        print_status(f"Failed to install {req} after {max_retries} attempts: {e}", False)
                        continue
                    print_status(f"Retrying {req} (attempt {attempt + 2}/{max_retries})", False)
                    time.sleep(2)

        # Install llama-cpp-python with appropriate backend
        print_status("Installing llama-cpp-python...")
        llama_cmd = [pip_exe, "install"]
        if backend == "GPU/CPU - CUDA":
            llama_cmd.extend(["llama-cpp-python[cuda]"])
        else:
            llama_cmd.extend(["llama-cpp-python"])
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                subprocess.run(llama_cmd, check=True)
                print_status("llama-cpp-python installed successfully")
                break
            except subprocess.CalledProcessError as e:
                if attempt == max_retries - 1:
                    print_status(f"Failed to install llama-cpp-python after {max_retries} attempts: {e}", False)
                    print("You may need to install llama-cpp-python manually later.")
                    break
                print_status(f"Retrying llama-cpp-python (attempt {attempt + 2}/{max_retries})", False)
                time.sleep(5)
        
        print_status("Python dependencies installation completed")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Dependency installation failed: {e}", False)
        return False

def copy_linux_binaries(source_dir: Path, dest_dir: Path) -> None:
    build_bin_dir = source_dir / "build" / "bin"
    
    if not build_bin_dir.exists():
        raise FileNotFoundError(f"Build directory not found at {build_bin_dir}")
    
    print_status("Copying Linux binaries to destination...")
    
    copied_files = []
    for file in build_bin_dir.iterdir():
        if file.is_file() and file.name.startswith("llama"):
            dest_file = dest_dir / file.name
            shutil.copy2(file, dest_file)
            os.chmod(dest_file, 0o755)
            copied_files.append(file.name)
    
    if copied_files:
        print_status(f"Copied {len(copied_files)} binary files")
    else:
        raise FileNotFoundError("No llama binaries found in build/bin directory")

def download_extract_backend(backend: str) -> bool:
    print_status(f"Downloading llama.cpp ({backend})...")
    backend_info = BACKEND_OPTIONS[backend]
    TEMP_DIR.mkdir(exist_ok=True)
    temp_zip = TEMP_DIR / "llama.zip"
    
    try:
        import zipfile
        
        download_with_progress(backend_info["url"], temp_zip, f"Downloading {backend}")
        
        dest_path = BASE_DIR / backend_info["dest"]
        dest_path.mkdir(parents=True, exist_ok=True)
        
        print_status("Extracting backend files...")
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            members = zip_ref.namelist()
            total_files = len(members)
            
            for i, member in enumerate(members):
                zip_ref.extract(member, dest_path)
                if i % 25 == 0 or i == total_files - 1:
                    progress = simple_progress_bar(i + 1, total_files)
                    print(f"\rExtracting: {progress}", end='', flush=True)
            
            print()
        
        if PLATFORM == "linux":
            copy_linux_binaries(dest_path, dest_path)
        
        cli_path = BASE_DIR / backend_info["cli_path"]
        if not cli_path.exists():
            raise FileNotFoundError(f"llama-cli not found at {cli_path}")
        
        if PLATFORM == "linux":
            os.chmod(cli_path, 0o755)
        
        print_status(f"llama.cpp ({backend}) installed successfully")
        return True
    except Exception as e:
        print_status(f"Backend installation failed: {str(e)}", False)
        return False
    finally:
        temp_zip.unlink(missing_ok=True)

def select_backend_type() -> None:
    global BACKEND_TYPE
    if PLATFORM == "windows":
        options = [
            "Vulkan - GPU/CPU - For AMD/nVidia/Intel GPU with x64 CPU",
            "Hip-Radeon - GPU/CPU - Experimental Vulkan for AMD-ROCM",
            "CUDA 11.7 - GPU/CPU - For CUDA 11.7 GPUs with CPU fallback",
            "CUDA 12.4 - GPU/CPU - For CUDA 12.4 GPUs with CPU fallback"
        ]
        mapping = {
            options[0]: "GPU/CPU - Vulkan",
            options[1]: "GPU/CPU - HIP-Radeon",
            options[2]: "GPU/CPU - CUDA 11.7",
            options[3]: "GPU/CPU - CUDA 12.4"
        }
    else:
        options = [
            "Vulkan - For AMD/Intel/NVIDIA GPUs",
            "CUDA - For NVIDIA GPUs with CUDA"
        ]
        mapping = {
            options[0]: "GPU/CPU - Vulkan",
            options[1]: "GPU/CPU - CUDA"
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
    
    # Verify backend dependencies first
    if not verify_backend_dependencies(BACKEND_TYPE):
        print_status("Required dependencies not satisfied!", False)
        sys.exit(1)
    
    create_directories()
    
    if not create_venv():
        print_status("Virtual environment creation failed!", False)
        sys.exit(1)
    
    if not install_python_deps(BACKEND_TYPE):
        print_status("Python dependency installation failed!", False)
        sys.exit(1)
    
    backend_info = BACKEND_OPTIONS[BACKEND_TYPE]
    if not backend_info["needs_python_bindings"]:
        if not download_extract_backend(BACKEND_TYPE):
            print_status("llama.cpp backend installation failed!", False)
            sys.exit(1)
    
    create_config(BACKEND_TYPE)
    
    print_status(f"{APP_NAME} installation completed successfully!")
    print("\nYou can now run the application from option 1 on the Bash/Batch Menu\n")

if __name__ == "__main__":
    try:
        select_backend_type()
        install()
    except KeyboardInterrupt:
        print("\nInstallation cancelled")
        sys.exit(1)
