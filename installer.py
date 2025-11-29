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

# Backend definitions
if PLATFORM == "windows":
    BACKEND_OPTIONS = {
        "x64 CPU Only": {
            "url": None,
            "dest": None,
            "cli_path": None,
            "needs_python_bindings": True,
            "vulkan_required": False,
            "build_flags": {}
        },
        "Vulkan GPU": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli.exe",
            "needs_python_bindings": True,
            "vulkan_required": True,
            "build_flags": {}
        },
        "Force Vulkan GPU": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-vulkan-x64.zip",  # or ubuntu variant for linux
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli.exe",  # or llama-cli for linux
            "needs_python_bindings": True,
            "vulkan_required": False,  # override check
            "build_flags": {}
        }
    }
else:  # Linux
    BACKEND_OPTIONS = {
        "x64 CPU Only": {
            "url": None,
            "dest": None,
            "cli_path": None,
            "needs_python_bindings": True,
            "vulkan_required": False,
            "build_flags": {}
        },
        "Vulkan GPU": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-ubuntu-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli",  # NO .exe on Linux
            "needs_python_bindings": True,
            "vulkan_required": True,
            "build_flags": {}
        },
        "Force Vulkan GPU": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-ubuntu-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli",  # NO .exe on Linux
            "needs_python_bindings": True,
            "vulkan_required": False,  # Skip detection
            "build_flags": {}
        }
    }


# Python requirements (CPU-only, no torch)
BASE_REQ = [
    "gradio==5.49.1",
    "requests==2.31.0",
    "pyperclip",
    "spacy>=3.7.0",
    "psutil",
    "ddgs",
    "newspaper3k",
    "langchain-community>=0.3.18",
    "faiss-cpu>=1.8.0",
    "langchain>=0.3.18",
    "pygments==2.17.2",
    "lxml[html_clean]",
    "pyttsx3",
    # onnxruntime and fastembed will be installed separately with special handling
    "tokenizers",
]

if PLATFORM == "windows":
    BASE_REQ.extend(["pywin32", "tk"])


# Utility helpers
def print_header(title: str) -> None:
    os.system('clear' if PLATFORM == "linux" else 'cls')
    width = shutil.get_terminal_size().columns - 1
    print("=" * width)
    print(f"    {APP_NAME} - {title}")
    print("=" * width)
    print()

def print_status(message: str, success: bool = True) -> None:
    status = "[✓]" if success else "[✗]"
    print(f"{status} {message}")
    time.sleep(1 if success else 3)

def get_user_choice(prompt: str, options: list) -> str:
    """Display menu exactly as requested"""
    print_header("Gpu Options")
    print("\n\n\n\n")
    for i, option in enumerate(options, 1):
        print(f"    {i}) {option}\n")
    print("\n\n\n\n\n")
    print("-" * (shutil.get_terminal_size().columns - 1))

    while True:
        choice = input(f"Selecton; Menu Options 1-{len(options)}, Abandon Install = A: ").strip().upper()
        if choice == "A":
            print("\nAbandoning installation...")
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
    """Build configuration with CPU-Only or Vulkan settings"""
    info = BACKEND_OPTIONS[backend]
    
    # FIX: Handle all three backend options correctly
    if backend in ["Vulkan GPU", "Force Vulkan GPU"]:
        backend_type = "Vulkan"
        vulkan_available = True
        vram_size = 8192
    else:  # x64 CPU Only
        backend_type = "CPU-Only"
        vulkan_available = False
        vram_size = 0
    
    config = {
        "model_settings": {
            "model_dir": "models",
            "model_name": "",
            "context_size": 32768,
            "temperature": 0.66,
            "repeat_penalty": 1.1,
            "use_python_bindings": True,
            "vram_size": vram_size,
            "selected_gpu": None,
            "mmap": True,
            "mlock": True,
            "n_batch": 1024,
            "dynamic_gpu_layers": True,
            "max_history_slots": 12,
            "max_attach_slots": 6,
            "print_raw_output": False,
            "show_think_phase": False, 
            "show_think_phase": False, 
            "bleep_on_events": False, 
            "session_log_height": 500,
            "cpu_threads": 4,
            "vulkan_available": vulkan_available,
            "backend_type": backend_type
        }
    }
    
    # FIX: Add llama-cli paths for both Vulkan options with relative paths
    if backend in ["Vulkan GPU", "Force Vulkan GPU"] and info["cli_path"]:
        # Use relative paths with platform-specific separators
        if PLATFORM == "windows":
            config["model_settings"]["llama_cli_path"] = ".\\data\\llama-vulkan-bin\\llama-cli.exe"
        else:  # Linux
            config["model_settings"]["llama_cli_path"] = "./data/llama-vulkan-bin/llama-cli"
        
        if info["dest"]:
            config["model_settings"]["llama_bin_path"] = "data/llama-vulkan-bin"
    
    return config

def create_config(backend: str) -> None:
    """Create configuration file with unified format"""
    config_path = BASE_DIR / "data" / "persistent.json"
    config = build_config(backend)
    
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print_status("Configuration file created")
        
        print("\nGenerated configuration:")
        print(f"  Backend: {config['model_settings']['backend_type']}")
        print(f"  Vulkan Available: {config['model_settings']['vulkan_available']}")
        print(f"  VRAM: {config['model_settings']['vram_size']} MB")
        print(f"  Context: {config['model_settings']['context_size']}")
        if "llama_cli_path" in config["model_settings"]:
            print(f"  llama-cli: {config['model_settings']['llama_cli_path']}")
        else:
            print(f"  Mode: Python bindings only (CPU)")
        
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

def check_vcredist_windows() -> bool:
    """Check if Visual C++ Redistributables are installed on Windows"""
    try:
        import winreg
        key_paths = [
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",  # VS 2015+
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        ]
        
        for key_path in key_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path):
                    return True
            except FileNotFoundError:
                continue
        return False
    except Exception as e:
        print(f"Warning: Could not check VC++ Redistributables: {e}")
        return False

def install_vcredist_windows() -> bool:
    """Download and install Visual C++ Redistributable on Windows"""
    print_status("Visual C++ Redistributable not found")
    print_status("Downloading Visual C++ 2015-2022 Redistributable...")
    
    TEMP_DIR.mkdir(exist_ok=True)
    vcredist_url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    vcredist_path = TEMP_DIR / "vc_redist.x64.exe"
    
    try:
        # Download with progress bar (matching your theme)
        download_with_progress(vcredist_url, vcredist_path, "Downloading VC++ Redistributable")
        
        print_status("Installing Visual C++ Redistributable (silent mode)...")
        print("  This may take 1-2 minutes...")
        
        # Silent install: /install /quiet /norestart
        result = subprocess.run(
            [str(vcredist_path), "/install", "/quiet", "/norestart"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Clean up installer
        vcredist_path.unlink(missing_ok=True)
        
        # Check if installation succeeded
        if result.returncode == 0:
            print_status("Visual C++ Redistributable installed successfully")
            return True
        elif result.returncode == 1638:
            # Already installed (edge case)
            print_status("Visual C++ Redistributable already present")
            return True
        elif result.returncode == 3010:
            # Success but reboot required
            print_status("Visual C++ Redistributable installed (reboot recommended)")
            print("\nNote: A system reboot is recommended but not required.")
            return True
        else:
            print_status(f"Installation returned code {result.returncode}", False)
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("Visual C++ installation timed out", False)
        vcredist_path.unlink(missing_ok=True)
        return False
    except Exception as e:
        print_status(f"Failed to install VC++ Redistributable: {e}", False)
        vcredist_path.unlink(missing_ok=True)
        return False

def install_onnxruntime() -> bool:
    """Install onnxruntime with proper error handling"""
    print_status("Installing onnxruntime (required for embeddings)...")
    
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                    ("python.exe" if PLATFORM == "windows" else "python"))
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                 ("pip.exe" if PLATFORM == "windows" else "pip"))
    
    # On Windows, ensure VC++ Redistributables are installed FIRST
    if PLATFORM == "windows":
        if not check_vcredist_windows():
            print("\n" + "=" * 80)
            print("CRITICAL DEPENDENCY: Visual C++ Redistributable Required")
            print("=" * 80)
            
            if not install_vcredist_windows():
                print("\n" + "!" * 80)
                print("CRITICAL ERROR: Visual C++ Redistributable installation failed!")
                print("!" * 80)
                print("\nonnxruntime cannot function without this component.")
                print("Installation cannot continue.\n")
                print("Manual installation required:")
                print("  1. Download: https://aka.ms/vs/17/release/vc_redist.x64.exe")
                print("  2. Run the installer as Administrator")
                print("  3. Re-run this installer\n")
                return False
            
            # Verify installation worked
            time.sleep(2)  # Give registry time to update
            if not check_vcredist_windows():
                print_status("VC++ Redistributable verification failed", False)
                print("Installation reported success but registry not updated.")
                print("Attempting onnxruntime installation anyway...")
    
    try:
        # Install onnxruntime
        subprocess.run(
            [pip_exe, "install", "onnxruntime"],
            check=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        print_status("onnxruntime installed")
        
        # Test if it loads
        result = subprocess.run(
            [python_exe, "-c", "import onnxruntime; print('OK')"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and "OK" in result.stdout:
            print_status("onnxruntime verified")
            return True
        else:
            print_status("onnxruntime installed but failed to load", False)
            print(f"Error: {result.stderr}")
            print("\n" + "!" * 80)
            print("CRITICAL ERROR: onnxruntime verification failed!")
            print("!" * 80)
            return False
            
    except subprocess.CalledProcessError as e:
        print_status(f"onnxruntime installation failed", False)
        print(f"Error: {e.stderr}")
        print("\n" + "!" * 80)
        print("CRITICAL ERROR: Failed to install onnxruntime")
        print("!" * 80)
        return False
    except subprocess.TimeoutExpired:
        print_status("onnxruntime installation timed out", False)
        return False
    except Exception as e:
        print_status(f"onnxruntime installation error: {e}", False)
        return False

def download_fastembed_model() -> bool:
    """Download and cache FastEmbed model during installation."""
    print_status("Downloading embedding model (FastEmbed)...")
    
    try:
        python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                        ("python.exe" if PLATFORM == "windows" else "python"))
        
        # FIX: Ensure cache directory path matches what validator expects
        cache_dir = BASE_DIR / "data" / "fastembed_cache"
        
        download_script = f'''
import os
from pathlib import Path

cache_dir = Path(r"{str(cache_dir.absolute())}")
cache_dir.mkdir(parents=True, exist_ok=True)

os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir.absolute())

try:
    from fastembed import TextEmbedding
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    print(f"Downloading {{model_name}}...")
    embedding = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
    print("Model downloaded successfully!")
except ImportError as e:
    print(f"Import error: {{e}}")
    exit(1)
except Exception as e:
    print(f"Download error: {{e}}")
    exit(1)
'''
        
        download_script_path = TEMP_DIR / "download_fastembed.py"
        with open(download_script_path, 'w') as f:
            f.write(download_script)
        
        result = subprocess.run(
            [python_exe, str(download_script_path)],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print_status("Embedding model downloaded")
            download_script_path.unlink(missing_ok=True)
            return True
        else:
            print_status(f"Model download failed", False)
            print(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("Model download timed out", False)
        return False
    except Exception as e:
        print_status(f"Model download error: {e}", False)
        return False

def download_spacy_model() -> bool:
    """Download spaCy English model during installation."""
    print_status("Downloading spaCy language model...")
    
    try:
        python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                        ("python.exe" if PLATFORM == "windows" else "python"))
        
        result = subprocess.run(
            [python_exe, "-m", "spacy", "download", "en_core_web_sm"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print_status("spaCy model downloaded")
            return True
        else:
            print_status(f"spaCy download failed: {result.stderr}", False)
            return False
            
    except subprocess.TimeoutExpired:
        print_status("spaCy download timed out", False)
        return False
    except Exception as e:
        print_status(f"spaCy download error: {e}", False)
        return False

def download_with_progress(url: str, filepath: Path, description: str = "Downloading") -> None:
    """Download file with progress bar using venv's requests library"""
    import time
    
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                    ("python.exe" if PLATFORM == "windows" else "python"))
    
    # Create a download script that uses the venv's requests
    download_script = f'''
import requests
import time
from pathlib import Path

def format_bytes(b):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024.0:
            return f"{{b:.1f}}{{unit}}"
        b /= 1024.0
    return f"{{b:.1f}}TB"

def simple_progress_bar(current, total, width=25):
    if total == 0:
        return "[" + "=" * width + "] 100%"
    filled_width = int(width * current // total)
    bar = "=" * filled_width + "-" * (width - filled_width)
    percent = 100 * current // total
    return f"[{{bar}}] {{percent}}% ({{format_bytes(current)}}/{{format_bytes(total)}})"

try:
    response = requests.get("{url}", stream=True, timeout=30)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    chunk_size = 8192
    last_update = time.time()
    
    with open(r"{str(filepath)}", 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                current_time = time.time()
                if current_time - last_update > 0.35 or downloaded >= total_size:
                    if total_size > 0:
                        progress = simple_progress_bar(downloaded, total_size)
                        print(f"\\r{description}: {{progress}}", end='', flush=True)
                    else:
                        downloaded_mb = downloaded / 1024 / 1024
                        print(f"\\r{description}: {{downloaded_mb:.1f}} MB downloaded", end='', flush=True)
                    last_update = current_time
    print()  # New line after progress
except Exception as e:
    Path(r"{str(filepath)}").unlink(missing_ok=True)
    raise e
'''
    
    download_script_path = TEMP_DIR / "download_file.py"
    try:
        with open(download_script_path, 'w') as f:
            f.write(download_script)
        
        result = subprocess.run(
            [python_exe, str(download_script_path)],
            check=True,
            timeout=600  # 10 minute timeout for large files
        )
        
    except subprocess.CalledProcessError as e:
        filepath.unlink(missing_ok=True)
        raise Exception(f"Download failed: {e}")
    except subprocess.TimeoutExpired:
        filepath.unlink(missing_ok=True)
        raise Exception("Download timed out")
    finally:
        download_script_path.unlink(missing_ok=True)

# Dependency checks
def is_vulkan_installed() -> bool:
    """Check if Vulkan is installed on the system"""
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

def verify_backend_dependencies(backend: str) -> bool:
    """Only check dependencies for Vulkan backend"""
    if backend == "Vulkan GPU":
        if not is_vulkan_installed():
            print("\n" + "!" * 80)
            print(f"⚠️  WARNING: Vulkan not detected!")
            if PLATFORM == "windows":
                print("  Download from: https://vulkan.lunarg.com/sdk/home")
            else:
                print("  Install with: sudo apt install vulkan-tools libvulkan-dev")
            print("!" * 80 + "\n")
            return False
    if backend == "Force Vulkan GPU":
        return True  # skip all checks
    return True

def install_linux_system_dependencies(backend: str) -> bool:
    if PLATFORM != "linux":
        return True

    print_status("Installing Linux system dependencies...")

    # Core packages including python3-venv which is CRITICAL for venv creation
    packages = [
        "build-essential",
        "python3-venv",      # CRITICAL: Required for virtual environment creation
        "python3-dev",       # Often needed for compiling Python packages
        "portaudio19-dev",
        "libasound2-dev",
        "python3-tk",
        "espeak",
        "libespeak-dev",
        "ffmpeg",
        "xclip"
    ]

    if backend == "Vulkan GPU":
        packages.extend(["vulkan-tools", "libvulkan-dev", "mesa-utils", "libvulkan1"])

    try:
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y"] + list(set(packages)), check=True)
        print_status("Linux dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"System dependencies failed: {e}", False)
        return False


# Python dependencies
def install_python_deps(backend: str) -> bool:
    print_status("Installing Python dependencies...")
    try:
        python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python"))
        pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("pip.exe" if PLATFORM == "windows" else "pip"))

        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print_status("Upgraded pip")

        # Install base packages
        subprocess.run([pip_exe, "install", *BASE_REQ], check=True)
        print_status("Base dependencies installed")

        # Install onnxruntime separately with error handling
        if not install_onnxruntime():
            print("\n" + "!" * 80)
            print("CRITICAL ERROR: onnxruntime installation failed!")
            print("!" * 80)
            print("\nThis is a required dependency for RAG features.")
            print("Installation cannot continue without it.\n")
            return False

        # Install fastembed (only if onnxruntime succeeded)
        try:
            subprocess.run([pip_exe, "install", "fastembed"], check=True, timeout=120)
            print_status("fastembed installed")
        except Exception as e:
            print_status(f"fastembed installation failed: {e}", False)
            print("\n" + "!" * 80)
            print("CRITICAL ERROR: fastembed installation failed!")
            print("!" * 80)
            return False

        # Always install llama-cpp-python
        llama_cmd = [pip_exe, "install", "llama-cpp-python"]
        
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


def install_optional_file_support() -> bool:
    """Install optional file format libraries (PDF, DOCX, etc.)"""
    print_status("Installing optional file format support...")
    
    optional_packages = [
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11", 
        "openpyxl>=3.0.0",
        "python-pptx>=0.6.21"
    ]
    
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                    ("python.exe" if PLATFORM == "windows" else "python"))
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                 ("pip.exe" if PLATFORM == "windows" else "pip"))
    
    failed_packages = []
    for package in optional_packages:
        try:
            subprocess.run([pip_exe, "install", package], 
                          check=True, 
                          capture_output=True)
            print_status(f"Installed {package.split('>=')[0]}")
        except subprocess.CalledProcessError:
            failed_packages.append(package.split('>=')[0])
            print_status(f"Optional package {package.split('>=')[0]} failed", False)
    
    if failed_packages:
        print(f"\nNote: Some file formats may not be supported: {', '.join(failed_packages)}")
        print("The program will work with text files only for these formats.\n")
    else:
        print_status("All file format support installed")
    
    return True


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
    """Download Vulkan backend only if selected"""
    if backend == "x64 CPU Only":
        print_status("CPU-Only mode: No binary download needed")
        return True
        
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
    """Show simplified 2-option menu"""
    opts = [
        "x64 CPU Only (No GPU Option)",
        "Vulkan GPU with x64 CPU Backend", 
        "Force Vulkan GPU (Skip Detection)"
    ]
    choice = get_user_choice("Select backend:", opts)
    
    mapping = {
        "x64 CPU Only (No GPU Option)": "x64 CPU Only",
        "Vulkan GPU with x64 CPU Backend": "Vulkan GPU",
        "Force Vulkan GPU (Skip Detection)": "Force Vulkan GPU"
    }
    return mapping[choice]


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

    # All functions below already handle venv paths internally
    # DO NOT wrap in activate_venv() context manager
    if not install_python_deps(backend):
        print_status("Python dependencies failed", False)
        sys.exit(1)

    install_optional_file_support()

    # Try to download models (now critical - must succeed)
    embedding_ok = download_fastembed_model()
    if not embedding_ok:
        print("\n" + "!" * 80)
        print("CRITICAL ERROR: Embedding model download failed!")
        print("!" * 80)
        print("\nRAG features require this model to function.")
        print("Installation cannot continue.\n")
        sys.exit(1)

    spacy_ok = download_spacy_model()
    if not spacy_ok:
        print_status("WARNING: spaCy model download failed", False)
        print("Session labeling may not work properly")
        # Non-critical, can continue

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