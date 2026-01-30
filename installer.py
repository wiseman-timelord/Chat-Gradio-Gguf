# Script: installer.py - Installation script for Chat-Gradio-Gguf
# Note: All install routes that state download NOT compile, should NOT compile.
# Note: Uses sentence-transformers for embeddings, cross-platform, Win 7-11 and Ubuntu 22-25
# Note: Uses Qt WebEngine for custom browser, Qt5 for Win 7/8/8.1, Qt6 for Win 10/11 and Ubuntu

# Imports
import os
import json
import subprocess
import sys
import contextlib
import time
from pathlib import Path
import shutil
import atexit
import re

# Constants / Variables ...
_PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
APP_NAME = "Chat-Gradio-Gguf"
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
LLAMACPP_GIT_REPO = "https://github.com/ggml-org/llama.cpp.git"
LLAMACPP_PYTHON_GIT_REPO = "https://github.com/abetlen/llama-cpp-python.git"
LLAMACPP_PYTHON_VERSION = "v0.3.16"  # Latest stable release
LLAMACPP_TARGET_VERSION = "b7688"
DOWNLOAD_RELEASE_TAG = "b7688" 
WIN_COMPILE_TEMP = Path("C:/temp_build")      # fixed Windows build folder (short path)
LINUX_COMPILE_TEMP = None                       # Linux keeps using project-local temp
_INSTALL_PROCESSES = set()
_DID_COMPILATION = False 
_PRE_EXISTING_PROCESSES = {} 
PYTHON_VERSION = sys.version_info
WINDOWS_VERSION = None  # Will detect Windows version
_CPU_FEATURES = None  # Will hold the detected features dict
_CPU_DETECTED_EARLY = False
OS_VERSION = None
VS_GENERATOR = None
DETECTED_PYTHON_INFO = {}
SELECTED_GRADIO = ""
SELECTED_QTWEB = ""

# Maps/Lists...
DIRECTORIES = [
    "data", "scripts", "models",
    "data/history", "data/temp", "data/vectors",
    "data/embedding_cache"
]

PROTECTED_DIRECTORIES = [
    "data/embedding_cache",
]

EMBEDDING_MODELS = {
    "1": {
        "name": "BAAI/bge-small-en-v1.5",
        "display": "Bge-Small-En v1.5 (Fastest - 132MB)",
        "size_mb": 132
    },
    "2": {
        "name": "BAAI/bge-base-en-v1.5", 
        "display": "Bge-Base-En v1.5 (Regular - 425MB)",
        "size_mb": 425
    },
    "3": {
        "name": "BAAI/bge-large-en-v1.5",
        "display": "Bge-Large-En v1.5 (Quality - 1.35GB)", 
        "size_mb": 1350
    }
}

# Platform detection windows / linux
PLATFORM = None

def set_platform() -> None:
    global PLATFORM
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ["windows", "linux"]:
        print("ERROR: Platform argument required (windows/linux)")
        sys.exit(1)
    PLATFORM = sys.argv[1].lower()

set_platform()

# Functions...
def print_status(message: str, success: bool = True) -> None:
    status = "[✓]" if success else "[✗]"
    print(f"{status} {message}")
    time.sleep(1 if success else 3)

def detect_cpu_features() -> dict:
    """Detect CPU SIMD features accurately."""
    global _CPU_FEATURES, _CPU_DETECTED_EARLY
    
    if _CPU_FEATURES is not None:
        return _CPU_FEATURES
    
    features = {
        "AVX": False, "AVX2": False, "AVX512": False, "FMA": False,
        "F16C": False, "SSE3": False, "SSSE3": False, "SSE4_1": False, "SSE4_2": False
    }
    
    success = False
    
    if PLATFORM == "windows":
        if VENV_DIR.exists():
            python_exe = VENV_DIR / "Scripts" / "python.exe"
            script_code = """
import cpuinfo
try:
    info = cpuinfo.get_cpu_info()
    flags = [f.lower() for f in info.get('flags', [])]
    print('|'.join(flags))
except Exception as e:
    print(f"ERROR:{e}")
"""
            temp_script = TEMP_DIR / "get_cpu_flags.py"
            try:
                temp_script.write_text(script_code)
                result = subprocess.run([str(python_exe), str(temp_script)], capture_output=True, text=True, timeout=10)
                output = result.stdout.strip()
                if result.returncode == 0 and not output.startswith("ERROR:"):
                    flags = output.split('|')
                    features["AVX"] = 'avx' in flags
                    features["AVX2"] = 'avx2' in flags
                    features["AVX512"] = any('avx512' in f for f in flags)
                    features["FMA"] = 'fma' in flags
                    features["F16C"] = 'f16c' in flags
                    features["SSE3"] = 'sse3' in flags or 'pni' in flags
                    features["SSSE3"] = 'ssse3' in flags
                    features["SSE4_1"] = 'sse4_1' in flags
                    features["SSE4_2"] = 'sse4_2' in flags
                    success = True
                else:
                    print_status(f"venv cpuinfo failed: {output or result.stderr}", False)
            finally:
                temp_script.unlink(missing_ok=True)
        
        if not success:
            features["SSE3"] = True
            print_status("Detected Legacy CPU, Using SSE3")
            success = True
    
    else:  # Linux
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read().lower()
            features["AVX"] = 'avx' in content
            features["AVX2"] = 'avx2' in content
            features["AVX512"] = 'avx512' in content
            features["FMA"] = 'fma' in content
            features["F16C"] = 'f16c' in content
            features["SSE3"] = 'sse3' in content or 'pni' in content
            features["SSSE3"] = 'ssse3' in content
            features["SSE4_1"] = 'sse4_1' in content
            features["SSE4_2"] = 'sse4_2' in content
            success = True
        except Exception:
            features["SSE3"] = True
            print_status("Linux CPU detection fallback - SSE3 only")
            success = True
    
    if success:
        _CPU_FEATURES = features
        _CPU_DETECTED_EARLY = True
    
    return features

# Set TEMP_DIR based on platform
if PLATFORM == "windows":
    TEMP_DIR = WIN_COMPILE_TEMP
else:
    TEMP_DIR = BASE_DIR / "data" / "temp"
    
# Backend definitions
if PLATFORM == "windows":
    BACKEND_OPTIONS = {
        "Download CPU Wheel / Default CPU Wheel": {
            "url": None, "dest": None, "cli_path": None,
            "needs_python_bindings": True, "compile_binary": False,
            "compile_wheel": False, "vulkan_required": False, "build_flags": {}
        },
        "Download Vulkan Bin / Default CPU Wheel": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{DOWNLOAD_RELEASE_TAG}/llama-{DOWNLOAD_RELEASE_TAG}-bin-win-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli.exe",
            "needs_python_bindings": True, "compile_binary": False,
            "compile_wheel": False, "vulkan_required": False, "build_flags": {}
        },
        "Compile CPU Binaries / Compile CPU Wheel": {
            "url": None, "dest": "data/llama-cpu-bin",
            "cli_path": "data/llama-cpu-bin/llama-cli.exe",
            "needs_python_bindings": True, "compile_binary": True,
            "compile_wheel": True, "vulkan_required": False, "build_flags": {}
        },
        "Compile Vulkan Binaries / Compile Vulkan Wheel": {
            "url": None, "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli.exe",
            "needs_python_bindings": True, "compile_binary": True,
            "compile_wheel": True, "vulkan_required": True,
            "build_flags": {"GGML_VULKAN": "1"}
        }
    }
else:  # Linux
    BACKEND_OPTIONS = {
        "Download CPU Wheel / Default CPU Wheel": {
            "url": None, "dest": None, "cli_path": None,
            "needs_python_bindings": True, "compile_binary": False,
            "compile_wheel": False, "vulkan_required": False, "build_flags": {}
        },
        "Download Vulkan Bin / Default CPU Wheel": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{DOWNLOAD_RELEASE_TAG}/llama-{DOWNLOAD_RELEASE_TAG}-bin-ubuntu-vulkan-x64.tar.gz",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli",
            "needs_python_bindings": True, "compile_binary": False,
            "compile_wheel": False, "vulkan_required": False, "build_flags": {}
        },
        "Compile CPU Binaries / Compile CPU Wheel": {
            "url": None, "dest": "data/llama-cpu-bin",
            "cli_path": "data/llama-cpu-bin/llama-cli",
            "needs_python_bindings": True, "compile_binary": True,
            "compile_wheel": True, "vulkan_required": False, "build_flags": {}
        },
        "Compile Vulkan Binaries / Compile Vulkan Wheel": {
            "url": None, "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli",
            "needs_python_bindings": True, "compile_binary": True,
            "compile_wheel": True, "vulkan_required": True,
            "build_flags": {"GGML_VULKAN": "1"}
        }
    }

# Python requirements - gradio/qt-web/torch/sentence_transformers are dynamically added based on OS/Python version
BASE_REQ = [
    "numpy<2",                      # Required: torch 2.2.2 incompatible with numpy 2.x
    "requests==2.31.0",
    "pyperclip==1.11.0",
    "spacy>=3.7.0",
    "psutil==7.2.1",
    "ddgs==9.10.0",
    "langchain-community>=0.3.18", 
    "faiss-cpu>=1.8.0",
    "langchain>=0.3.18",            
    "pygments==2.17.2",
    "lxml[html_clean]==6.0.2",
    "lxml_html_clean==0.4.3",
    "tokenizers==0.22.2",
    "beautifulsoup4>=4.12.0",       # HTML parsing for deep research
    "aiohttp>=3.10.0",              # Async HTTP for parallel page fetches
]

if PLATFORM == "windows":
    BASE_REQ.extend([
        "pywin32==311",
        "tk==0.1.0",
        "pythonnet==3.0.5",
        "pyttsx3>=2.90",  # Windows built-in TTS (SAPI)
    ])
elif PLATFORM == "linux":
    BASE_REQ.extend([
        "pyvirtualdisplay>=3.0",    # Headless Qt WebEngine support
    ])

# newspaper4k requires Python 3.10+, 3.9 → must fallback
if sys.version_info >= (3, 10):
    BASE_REQ.append("newspaper4k==0.9.4.1")     # latest stable as of Jan 2026
else:
    BASE_REQ.append("newspaper3k==0.2.8") 

# Functions...
def backend_requires_compilation(backend: str) -> bool:
    """Check if the selected backend requires compilation"""
    info = BACKEND_OPTIONS.get(backend, {})
    return info.get("compile_binary", False) or info.get("compile_wheel", False)

def detect_windows_version() -> str:
    """Detect Windows version and cache in global"""
    global WINDOWS_VERSION
    if WINDOWS_VERSION is not None:
        return WINDOWS_VERSION
    
    if PLATFORM != "windows":
        return None
    
    try:
        import platform
        version = platform.version()
        build = int(version.split('.')[-1])
        
        if build >= 22000:
            WINDOWS_VERSION = "11"
        elif build >= 10240:
            WINDOWS_VERSION = "10"
        elif build == 9600:
            WINDOWS_VERSION = "8.1"
        elif build == 9200:
            WINDOWS_VERSION = "8"
        elif build == 7601:
            WINDOWS_VERSION = "7"
        else:
            WINDOWS_VERSION = "unknown"
        return WINDOWS_VERSION
    except:
        WINDOWS_VERSION = "unknown"
        return "unknown"

def detect_linux_version() -> str:
    """Detect Ubuntu version and cache in global OS_VERSION"""
    global OS_VERSION
    if OS_VERSION is not None:
        return OS_VERSION
    
    try:
        with open("/etc/os-release") as f:
            content = f.read()
        
        if "ubuntu" not in content.lower():
            OS_VERSION = "unknown"
            return "unknown"
        
        version_match = re.search(r'VERSION_ID="?([0-9\.]+)"?', content)
        if version_match:
            OS_VERSION = version_match.group(1)
            return OS_VERSION
        
        OS_VERSION = "unknown"
        return "unknown"
    except Exception as e:
        print_status(f"OS detection failed: {e}", False)
        OS_VERSION = "unknown"
        return "unknown"

def detect_version_selections() -> None:
    """Populate DETECTED_PYTHON_INFO, SELECTED_GRADIO, SELECTED_QTWEB based on OS."""
    global DETECTED_PYTHON_INFO, SELECTED_GRADIO, SELECTED_QTWEB
    import platform as plat
    
    DETECTED_PYTHON_INFO = detect_all_pythons()
    
    if PLATFORM == "windows":
        try:
            major_ver = int(plat.version().split('.')[0])
            if major_ver < 10:
                SELECTED_GRADIO = "3.50.2"
                SELECTED_QTWEB = "v5"
            else:
                SELECTED_GRADIO = "5.49.1"
                SELECTED_QTWEB = "v6"
        except:
            SELECTED_GRADIO = "5.49.1"
            SELECTED_QTWEB = "v6"
    else:
        try:
            with open("/etc/os-release", "r") as f:
                os_release = f.read()
            match = re.search(r'VERSION_ID="?(\d+)', os_release)
            if match and int(match.group(1)) < 24:
                SELECTED_GRADIO = "3.50.2"
                SELECTED_QTWEB = "v5"
            else:
                SELECTED_GRADIO = "5.49.1"
                SELECTED_QTWEB = "v6"
        except:
            SELECTED_GRADIO = "5.49.1"
            SELECTED_QTWEB = "v6"

def get_dynamic_requirements() -> list:
    """
    Build requirements list with dynamic gradio version based on OS.
    
    Version Matrix:
    - Windows 7/8/8.1 + Ubuntu 22/23: Gradio 3.50.2, Qt5, Python 3.9-3.11
    - Windows 10/11 + Ubuntu 24/25: Gradio 5.49.1, Qt6, Python 3.10-3.14
    """
    global SELECTED_GRADIO, SELECTED_QTWEB
    
    # Ensure version selections are populated
    if not SELECTED_GRADIO:
        detect_version_selections()
    
    requirements = BASE_REQ.copy()
    
    # Add gradio with correct version
    if SELECTED_GRADIO == "3.50.2":
        requirements.append("gradio==3.50.2")
    else:
        requirements.append("gradio>=5.49.1")
    
    return requirements

def get_qt_version_for_os() -> tuple:
    """
    Determine Qt version based on OS.
    
    Returns:
        tuple: (qt_major_version, use_qt5: bool)
        
    Version Matrix:
    - Windows 7/8/8.1: Qt5 (PyQt5 + PyQtWebEngine)
    - Windows 10/11: Qt6 (PyQt6 + PyQt6-WebEngine)
    - Ubuntu 22/23: Qt5 (PyQt5 + PyQtWebEngine)
    - Ubuntu 24/25: Qt6 (PyQt6 + PyQt6-WebEngine)
    """
    if PLATFORM == "windows":
        win_ver = detect_windows_version()
        if win_ver in ["7", "8", "8.1"]:
            return (5, True)
        else:
            return (6, False)
    else:  # Linux
        linux_ver = detect_linux_version()
        try:
            major_ver = int(linux_ver.split('.')[0]) if linux_ver != "unknown" else 24
            if major_ver < 24:
                return (5, True)
            else:
                return (6, False)
        except (ValueError, AttributeError):
            return (6, False)

def get_torch_version_for_python() -> str:
    """
    Get appropriate torch version for the Python version.
    
    Returns:
        str: torch version specifier with index URL
        
    Compatibility:
    - Python 3.9-3.11: torch==2.2.2+cpu (stable, wide compatibility)
    - Python 3.12+: torch>=2.4.0+cpu (required for 3.12+ support)
    """
    py_minor = sys.version_info.minor
    
    if py_minor <= 11:
        return "torch==2.2.2+cpu"
    else:
        # Python 3.12+ requires torch 2.4.0+
        return "torch>=2.4.0"

def detect_all_pythons() -> dict:
    """Detect all Python installations and select optimal version for OS compatibility."""
    import platform as plat
    found_pythons = []
    
    if PLATFORM == "windows":
        possible_commands = ["python", "python3", "py -3.14", "py -3.13", "py -3.12", "py -3.11", "py -3.10", "py -3.9"]
        for cmd in possible_commands:
            try:
                parts = cmd.split()
                result = subprocess.run(parts + ["--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version_str = result.stdout.strip() or result.stderr.strip()
                    match = re.search(r"Python\s+(\d+)\.(\d+)", version_str)
                    if match:
                        major, minor = int(match.group(1)), int(match.group(2))
                        found_pythons.append((major, minor, cmd))
            except:
                pass
    else:
        possible_commands = ["python3", "python3.14", "python3.13", "python3.12", "python3.11", "python3.10", "python3.9", "python"]
        for cmd in possible_commands:
            try:
                result = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version_str = result.stdout.strip() or result.stderr.strip()
                    match = re.search(r"Python\s+(\d+)\.(\d+)", version_str)
                    if match:
                        major, minor = int(match.group(1)), int(match.group(2))
                        found_pythons.append((major, minor, cmd))
            except:
                pass
    
    seen_versions = set()
    unique_pythons = []
    for major, minor, cmd in found_pythons:
        if (major, minor) not in seen_versions:
            seen_versions.add((major, minor))
            unique_pythons.append((major, minor, cmd))
    
    # Determine valid range based on OS
    # Windows 7/8/8.1 + Ubuntu 22-23: Gradio 3.50.2 -> Python 3.9-3.11
    # Windows 10/11 + Ubuntu 24-25: Gradio 5.49.1 -> Python 3.10-3.14
    
    if PLATFORM == "windows":
        try:
            major_ver = int(plat.version().split('.')[0])
            if major_ver < 10:
                min_py, max_py = (3, 9), (3, 11)
            else:
                min_py, max_py = (3, 10), (3, 14)
        except:
            min_py, max_py = (3, 10), (3, 14)
    else:
        try:
            with open("/etc/os-release", "r") as f:
                os_release = f.read()
            match = re.search(r'VERSION_ID="?(\d+)', os_release)
            if match and int(match.group(1)) < 24:
                min_py, max_py = (3, 9), (3, 11)
            else:
                min_py, max_py = (3, 10), (3, 14)
        except:
            min_py, max_py = (3, 10), (3, 14)
    
    valid_pythons = [
        (major, minor, cmd) for major, minor, cmd in unique_pythons
        if (major, minor) >= min_py and (major, minor) <= max_py
    ]
    
    if valid_pythons:
        valid_pythons.sort(key=lambda x: (x[0], x[1]), reverse=True)
        optimal = valid_pythons[0]
        return {
            "all_found": unique_pythons,
            "valid_range": (min_py, max_py),
            "optimal": optimal,
            "optimal_version": f"{optimal[0]}.{optimal[1]}",
            "optimal_command": optimal[2]
        }
    
    return {
        "all_found": unique_pythons,
        "valid_range": (min_py, max_py),
        "optimal": None,
        "optimal_version": "None",
        "optimal_command": None
    }

def check_version_compatibility():
    """Check Python and OS compatibility, set globals."""
    global WINDOWS_VERSION, PYTHON_VERSION, PLATFORM
    
    if sys.version_info < (3, 9):
        print_status("Python ≥3.9 required", False)
        return False
    
    PYTHON_VERSION = sys.version_info
    
    if PYTHON_VERSION.minor == 14:
        print("Warning: Python 3.14 detected - some packages may have compatibility issues")
        time.sleep(2)
    
    if PLATFORM == "windows":
        win_ver = detect_windows_version()
        return True  # Windows 7-11 supported
    else:  # Linux
        try:
            with open("/etc/os-release") as f:
                content = f.read()
                if "UBUNTU_VERSION_ID" in content or "ubuntu" in content.lower():
                    version_match = re.search(r'VERSION_ID="?([0-9\.]+)"?', content)
                    if version_match:
                        ubuntu_version = version_match.group(1)
                        if ubuntu_version.startswith(('22', '23', '24', '25')):
                            return True
                        print_status(f"Ubuntu {ubuntu_version} unsupported - requires 22.04-25.04", False)
                        return False
            print_status("Could not determine Ubuntu version", False)
            return False
        except Exception as e:
            print_status(f"OS detection failed: {e}", False)
            return False

def snapshot_pre_existing_processes() -> None:
    """Snapshot all build-related processes that exist BEFORE compilation starts"""
    global _PRE_EXISTING_PROCESSES
    
    if PLATFORM != "windows":
        return
    
    try:
        import psutil
    except ImportError:
        return
    
    build_process_names = [
        "conhost.exe", "MSBuild.exe", "VBCSCompiler.exe", "node.exe",
        "cmake.exe", "cl.exe", "link.exe", "lib.exe", "cvtres.exe",
        "mt.exe", "rc.exe", "mspdbsrv.exe", "vctip.exe", "tracker.exe",
        "git.exe", "python.exe", "pip.exe",
    ]
    
    _PRE_EXISTING_PROCESSES = {}
    
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] in build_process_names:
                _PRE_EXISTING_PROCESSES[proc.info['pid']] = proc.info['name']
    except:
        pass

def track_process(pid: int) -> None:
    """Track a process and all its current and future descendants."""
    global _INSTALL_PROCESSES
    try:
        import psutil
        try:
            proc = psutil.Process(pid)
            _INSTALL_PROCESSES.add(pid)
            for child in proc.children(recursive=True):
                _INSTALL_PROCESSES.add(child.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    except ImportError:
        pass

def cleanup_build_processes() -> None:
    """Terminate lingering build processes created during installation."""
    global _DID_COMPILATION, _INSTALL_PROCESSES, _PRE_EXISTING_PROCESSES
    
    if not _DID_COMPILATION:
        return
    
    if PLATFORM != "windows":
        return
    
    try:
        import psutil
    except ImportError:
        return
    
    try:
        current_proc = psutil.Process()
        current_pid = current_proc.pid
        current_conhost = None

        try:
            parent = current_proc.parent()
            if parent:
                for child in parent.children():
                    if child.name().lower() == "conhost.exe":
                        current_conhost = child.pid
                        break
        except:
            pass

        protected_pids = {current_pid}
        if current_conhost:
            protected_pids.add(current_conhost)

        to_kill = []

        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['pid'] in protected_pids:
                    continue
                name = proc.info['name'].lower()
                if name in ["msbuild.exe", "conhost.exe"]:
                    to_kill.append(proc.info['pid'])
            except:
                continue

        if not to_kill:
            return

        for pid in to_kill:
            try:
                p = psutil.Process(pid)
                p.terminate()
            except:
                pass

        time.sleep(1)

        for pid in to_kill:
            try:
                p = psutil.Process(pid)
                if p.is_running():
                    p.kill()
            except:
                pass

        print_status(f"Cleaned up {len(to_kill)} build processes")
    except Exception as e:
        print(f"Note: Cleanup had issues: {e}")

atexit.register(cleanup_build_processes)

def print_header(title: str) -> None:
    os.system('clear' if PLATFORM == "linux" else 'cls')
    width = shutil.get_terminal_size().columns - 1
    print("=" * width)
    print(f"    {APP_NAME} - {title}")
    print("=" * width)
    print()

def get_user_choice(prompt: str, options: list) -> str:
    """Display menu with system detections, then options"""
    width = shutil.get_terminal_size().columns - 1
    print_header("Backend/Wheel Menu")
   
    print("System Detections...")
    
    if PLATFORM == "windows":
        win_ver = WINDOWS_VERSION or "unknown"
        print(f"    Operating System: Windows {win_ver}")
    else:
        ubuntu_ver = OS_VERSION or "unknown"
        print(f"    Operating System: Ubuntu {ubuntu_ver}")
    
    cpu_features = detect_cpu_features()
    features_list = [feat for feat, supported in cpu_features.items() if supported]
    if features_list:
        print(f"    CPU Features: {', '.join(features_list)}")
    else:
        print("    CPU Features: Baseline (no advanced features)")
    
    vulkan_present = is_vulkan_installed()
    print(f"    Vulkan Present: {'Yes' if vulkan_present else 'No'}")
    
    print()
    
    print("Backend/Wheel Type...")
    for i, option in enumerate(options, 1):
        print(f"    {i}) {option}")
    print()

    while True:
        choice = input(f"Selection; Menu Options 1-{len(options)}, Abandon Install = A: ").strip().upper()
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

def create_directories(backend: str) -> None:
    global TEMP_DIR
    
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
    
    if PLATFORM == "windows" and backend_requires_compilation(backend):
        TEMP_DIR = WIN_COMPILE_TEMP
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        print_status(f"Using build temp path: {TEMP_DIR}")
    else:
        TEMP_DIR = BASE_DIR / "data" / "temp"
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        print_status(f"Using project temp path: {TEMP_DIR}")

def get_optimal_build_threads() -> int:
    """Calculate optimal thread count for building - 85 percent of available cores"""
    import multiprocessing
    try:
        total_threads = multiprocessing.cpu_count()
    except:
        total_threads = 4
    return max(1, int(total_threads * 0.85))

def build_llama_cpp_python_with_flags(build_flags: dict) -> bool:
    """Build llama-cpp-python from source with optimal CPU flags."""
    global _DID_COMPILATION
    _DID_COMPILATION = True
    
    snapshot_pre_existing_processes()
    
    print_status("Building llama-cpp-python from source (10-20 minutes)")
    
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                    ("python.exe" if PLATFORM == "windows" else "python"))
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                 ("pip.exe" if PLATFORM == "windows" else "pip"))
    
    build_threads = get_optimal_build_threads()
    print(f"  Using {build_threads} parallel build threads")
    
    env = os.environ.copy()
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(build_threads)
    env["FORCE_CMAKE"] = "1"
    
    # Detect CPU features and add to build flags
    cpu_features = detect_cpu_features()
    
    # CPU optimization flags
    if cpu_features.get("AVX2"):
        build_flags["GGML_AVX2"] = "ON"
    if cpu_features.get("AVX"):
        build_flags["GGML_AVX"] = "ON"
    if cpu_features.get("FMA"):
        build_flags["GGML_FMA"] = "ON"
    if cpu_features.get("F16C"):
        build_flags["GGML_F16C"] = "ON"
    
    # Minimal safe flags - do NOT disable LLAMA_BUILD_EXAMPLES or LLAMA_BUILD_SERVER
    # as this causes CMake errors with mtmd/llava targets in v0.3.16
    build_flags["LLAMA_CURL"] = "OFF"
    build_flags["GGML_OPENMP"] = "ON"
    
    # Build CMAKE_ARGS properly - use -D prefix for CMake
    cmake_args = []
    for flag, value in build_flags.items():
        cmake_args.append(f"-D{flag}={value}")
    
    # Set CMAKE_ARGS environment variable
    env["CMAKE_ARGS"] = " ".join(cmake_args)
    
    # Also set individual flags as env vars, some builds check these
    for flag, value in build_flags.items():
        env[flag] = value
    
    if cmake_args:
        print(f"  Build flags: {', '.join(f'{k}={v}' for k, v in build_flags.items())}")
    
    if PLATFORM == "windows":
        env["CL"] = f"/MP{build_threads}"
    else:
        env["MAKEFLAGS"] = f"-j{build_threads}"
    
    repo_dir = TEMP_DIR / "llama-cpp-python"
    
    try:
        # Clean any leftover repo directory
        if repo_dir.exists():
            print_status("Cleaning previous build artifacts...")
            try:
                shutil.rmtree(repo_dir, ignore_errors=False)
            except PermissionError:
                # On Windows, try harder
                if PLATFORM == "windows":
                    subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(repo_dir)], 
                                  capture_output=True, timeout=30)
                time.sleep(1)
                if repo_dir.exists():
                    shutil.rmtree(repo_dir, ignore_errors=True)
        
        print_status(f"Cloning llama-cpp-python {LLAMACPP_PYTHON_VERSION}...")
        
        # Clone specific release version with retry for network issues
        max_retries = 5
        retry_delay = 10
        clone_success = False
        
        for attempt in range(max_retries):
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", LLAMACPP_PYTHON_VERSION,
                 "--recurse-submodules", LLAMACPP_PYTHON_GIT_REPO, str(repo_dir)],
                capture_output=True, text=True, timeout=300, env=env
            )
            
            if result.returncode == 0:
                clone_success = True
                break
            else:
                if repo_dir.exists():
                    shutil.rmtree(repo_dir, ignore_errors=True)
                if attempt < max_retries - 1:
                    print(f"  Clone failed, retry {attempt + 1}/{max_retries} in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
        
        if not clone_success:
            print_status(f"Git clone failed after {max_retries} attempts: {result.stderr}", False)
            return False
        
        print_status("Building wheel (this takes a while)...")
        
        # Capture full output for error diagnosis
        full_output = []
        
        process = subprocess.Popen(
            [pip_exe, "install", str(repo_dir), "-v", "--no-cache-dir"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env
        )
        
        track_process(process.pid)
        
        last_progress_line = ""
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            full_output.append(line)
            if any(x in line.lower() for x in ["building", "compiling", "linking", "cmake", "installing", "error", "fatal"]):
                if last_progress_line:
                    print(f"\r{' ' * len(last_progress_line)}\r", end='', flush=True)
                display_line = line[:80] + "..." if len(line) > 80 else line
                print(f"\r  {display_line}", end='', flush=True)
                last_progress_line = display_line
        
        process.wait(timeout=3600)
        
        if last_progress_line:
            print()
        
        if process.returncode == 0:
            print_status("llama-cpp-python built and installed")
            return True
        else:
            print_status("llama-cpp-python build failed", False)
            # Show last 20 lines of output to help diagnose
            print("  Build output (last 20 lines):")
            for line in full_output[-20:]:
                if "error" in line.lower() or "fatal" in line.lower():
                    print(f"    >>> {line[:100]}")
                else:
                    print(f"    {line[:100]}")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("Build timed out", False)
        return False
    except Exception as e:
        print_status(f"Build error: {e}", False)
        return False
    finally:
        if repo_dir.exists():
            try:
                shutil.rmtree(repo_dir, ignore_errors=True)
            except:
                pass

def create_config(backend: str, embedding_model: str) -> None:
    """Create persistent.json configuration file (variables only)"""
    config_path = BASE_DIR / "data" / "persistent.json"
    
    vulkan_enabled = "vulkan" in backend.lower()
    
    # Set layer_allocation_mode based on install route:
    # - Vulkan routes (2, 4): VRAM_SRAM (enables VRAM dropdown for GPU inference)
    # - CPU-only routes (1, 3): SRAM_ONLY (CPU-only inference)
    layer_mode = "VRAM_SRAM" if vulkan_enabled else "SRAM_ONLY"
    default_vram = 8192 if vulkan_enabled else 0
    
    # Calculate optimal CPU threads (85% of available)
    optimal_threads = get_optimal_build_threads()
    
    # Platform-specific defaults for sound/TTS settings
    # Windows uses "Default Sound Device", Linux uses "default"
    if PLATFORM == "windows":
        default_sound_device = "Default Sound Device"
    else:
        default_sound_device = "default"
    
    # Use model_settings format for compatibility with settings.py
    # Note: Constants like llama_cli_path, llama_bin_path, embedding_model, 
    # embedding_backend, vulkan_enabled, and filter_mode are stored in or determined 
    # from constants.ini, not stored in the user-modifiable persistent.json
    config = {
        "model_settings": {
            "layer_allocation_mode": layer_mode,
            "model_dir": "models",
            "model_name": "Select_a_model...",
            "context_size": 8192,
            "vram_size": default_vram,
            "temperature": 0.66,
            "repeat_penalty": 1.1,
            "selected_gpu": "Auto",
            "selected_cpu": "Auto-Select",
            "mmap": True,
            "mlock": False,
            "n_batch": 1024,
            "dynamic_gpu_layers": True,
            "max_history_slots": 8,
            "max_attach_slots": 6,
            "session_log_height": 650,
            "show_think_phase": False,
            "print_raw_output": False,
            "cpu_threads": optimal_threads,
            "bleep_on_events": True,
            "use_python_bindings": True,
            # Sound and TTS configuration - must match keys used in configuration.py save_config()
            "sound_output_device": default_sound_device,
            "sound_sample_rate": 44100,
            "tts_enabled": False,
            "tts_voice": None,
            "tts_voice_name": None,
            "max_tts_length": 4500,
        }
    }
    
    try:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print_status("Configuration file created")
    except Exception as e:
        print_status(f"Failed to create config: {e}", False)

def create_system_ini(platform: str, os_version: str, python_version: str, 
                     backend_type: str, embedding_model: str,
                     windows_version: str = None, vulkan_available: bool = False,
                     llama_cli_path: str = None, llama_bin_path: str = None):
    """Create constants.ini with platform, version, and compatibility information."""
    global SELECTED_GRADIO, SELECTED_QTWEB
    
    # Ensure version selections are populated
    if not SELECTED_GRADIO:
        detect_version_selections()
    
    qt_version, _ = get_qt_version_for_os()
    
    system_ini_path = BASE_DIR / "data" / "constants.ini"
    try:
        with open(system_ini_path, "w") as f:
            f.write("[system]\n")
            f.write(f"platform = {platform}\n")
            f.write(f"os_version = {os_version}\n")
            f.write(f"python_version = {python_version}\n")
            f.write(f"backend_type = {backend_type}\n")
            f.write(f"embedding_model = {embedding_model}\n")
            f.write(f"embedding_backend = sentence_transformers\n")
            f.write(f"vulkan_available = {str(vulkan_available).lower()}\n")
            # Add version compatibility info for main program
            f.write(f"gradio_version = {SELECTED_GRADIO}\n")
            f.write(f"qt_version = {qt_version}\n")
            if llama_cli_path:
                f.write(f"llama_cli_path = {llama_cli_path}\n")
            if llama_bin_path:
                f.write(f"llama_bin_path = {llama_bin_path}\n")
            if platform == "windows" and windows_version:
                f.write(f"windows_version = {windows_version}\n")
        print_status("System information file created")
        return True
    except Exception as e:
        print_status(f"Failed to create constants.ini: {str(e)}", False)
        return False

def create_venv() -> bool:
    try:
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
            print_status("Removed existing virtual environment")

        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        
        print_header("Installer") # mounting the venv cleared the display
        print_status("Created new virtual environment")

        python_exe = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
        pip_exe = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("pip.exe" if PLATFORM == "windows" else "pip")
        
        if not python_exe.exists():
            raise FileNotFoundError(f"Python executable not found at {python_exe}")

        # Upgrade pip immediately to avoid "new release available" notices
        subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], 
                      capture_output=True, timeout=120)
        print_status("Upgraded pip to latest version")

        print_status("Verified virtual environment setup")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to create venv: {e}", False)
        return False

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

def has_avx_support() -> bool:
    """Check if CPU has AVX support"""
    global _CPU_FEATURES
    if _CPU_FEATURES is None:
        detect_cpu_features()
    return _CPU_FEATURES.get('AVX', False) if _CPU_FEATURES is not None else False

def check_vcredist_windows() -> bool:
    """Check if Visual C++ Redistributables are installed on Windows"""
    try:
        import winreg
        key_paths = [
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        ]
        for key_path in key_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path):
                    return True
            except FileNotFoundError:
                continue
        return False
    except:
        return False

def check_vulkan_sdk_installed() -> bool:
    """Check if Vulkan SDK is installed not just runtime"""
    if PLATFORM == "windows":
        vulkan_sdk = os.environ.get("VULKAN_SDK")
        if vulkan_sdk and Path(vulkan_sdk).is_dir():
            return True

        default_sdk = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "VulkanSDK"
        if default_sdk.exists():
            for child in default_sdk.iterdir():
                if child.is_dir() and (child/"Bin"/"vulkaninfoSDK.exe").exists():
                    os.environ["VULKAN_SDK"] = str(child)
                    return True
        return False
    else:  # Linux
        try:
            result = subprocess.run(["vulkaninfo", "--summary"], 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if result.returncode != 0:
                return False
            result = subprocess.run(["which", "glslc"], capture_output=True)
            return result.returncode == 0
        except:
            return False

def is_vulkan_installed() -> bool:
    """Check if Vulkan runtime is installed on the system"""
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
            print_status("Warning: Vulkan not detected!", False)
            return False
    if backend == "Force Vulkan GPU":
        return True
    return True

def install_linux_system_dependencies(backend: str) -> bool:
    """Install Linux system dependencies including optional CUDA/Vulkan build tools."""
    print_status("Installing Linux system dependencies...")
    
    # Get Qt version for OS first
    qt_version, use_qt5 = get_qt_version_for_os()
    linux_ver = detect_linux_version()
    
    try:
        major_ver = int(linux_ver.split('.')[0]) if linux_ver != "unknown" else 24
    except (ValueError, AttributeError):
        major_ver = 24
    
    # Base packages needed for Python packages
    base_packages = [
        "python3-dev",
        "build-essential",
        "libffi-dev",
        "libssl-dev",
        "portaudio19-dev",
        "libespeak-ng1",
        "espeak-ng",
        "libegl1",
        "libgl1",
        "libxkbcommon0",
        "libxcb-cursor0",
        # For headless Qt WebEngine (virtual display)
        "xvfb",
    ]
    
    # Qt dependencies based on version and Ubuntu version
    if use_qt5:
        # Ubuntu 22/23 - Qt5
        qt_packages = [
            "libxcb-xinerama0",
            "libxkbcommon0",
            "libegl1",
            "libgl1",
        ]
        # qt5-default doesn't exist in Ubuntu 22+, use these instead
        if major_ver >= 22:
            qt_packages.extend([
                "qtbase5-dev",
                "qtchooser",
                "qt5-qmake",
                "qtbase5-dev-tools",
                "libqt5webengine5",
                "libqt5webenginewidgets5",
            ])
        qt_fallback = [
            "qtwebengine5-dev",
        ]
    else:
        # Ubuntu 24/25 - Qt6
        qt_packages = [
            "libxcb-cursor0",
            "libxkbcommon0",
            "libegl1",
            "libgl1",
            "libxcb-xinerama0",
        ]
        if major_ver >= 24:
            qt_packages.extend([
                "qt6-base-dev",
                "libqt6webenginecore6",
                "libqt6webenginewidgets6",
            ])
        qt_fallback = [
            "libqt6webengine6-data",
        ]
    
    info = BACKEND_OPTIONS[backend]
    vulkan_packages = []
    if info.get("build_flags", {}).get("GGML_VULKAN"):
        vulkan_packages = [
            "vulkan-tools", "libvulkan-dev", "mesa-utils",
            "glslang-tools", "spirv-tools"
        ]
    elif backend in ["Download Vulkan Bin / Default CPU Wheel"]:
        vulkan_packages = ["vulkan-tools", "libvulkan1"]
    
    try:
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y"] + list(set(base_packages)), check=True)
        print_status("Base dependencies installed")
        
        # Install Qt dependencies
        print_status(f"Installing Qt{qt_version} dependencies...")
        for package in qt_packages:
            try:
                subprocess.run(["sudo", "apt-get", "install", "-y", package], 
                              check=True, capture_output=True)
                print_status(f"  Installed {package}")
            except subprocess.CalledProcessError:
                print_status(f"  Package {package} not available, trying alternatives...", False)
        
        # Try fallback packages if needed
        for package in qt_fallback:
            try:
                subprocess.run(["sudo", "apt-get", "install", "-y", package], 
                              check=True, capture_output=True)
            except subprocess.CalledProcessError:
                pass  # Fallbacks are optional
        
        if vulkan_packages:
            print_status("Installing Vulkan development packages...")
            for package in vulkan_packages:
                try:
                    subprocess.run(["sudo", "apt-get", "install", "-y", package], check=True)
                    print_status(f"  Installed {package}")
                except subprocess.CalledProcessError:
                    print_status(f"  Package {package} not available, trying alternatives...", False)
            
            if info.get("build_flags", {}).get("GGML_VULKAN"):
                result = subprocess.run(["which", "glslc"], capture_output=True, text=True)
                if result.returncode != 0:
                    print_status("Error: glslc shader compiler not found", False)
                    print("Please install: sudo apt install glslang-tools shaderc")
                    return False
        
        print_status("Linux dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"System dependencies failed: {e}", False)
        return False

def install_embedding_backend() -> bool:
    """
    Install torch + sentence-transformers for all platforms.
    
    Torch Version Matrix:
    - Python 3.9-3.11: torch==2.2.2+cpu (stable, wide compatibility)
    - Python 3.12+: torch>=2.4.0+cpu (required for 3.12+ support)
    """
    print_status("Installing embedding backend (torch + sentence-transformers)...")
    
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("python.exe" if PLATFORM == "windows" else "python"))
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                ("pip.exe" if PLATFORM == "windows" else "pip"))
    
    # Determine torch version based on Python version
    torch_spec = get_torch_version_for_python()
    py_minor = sys.version_info.minor
    
    # Install PyTorch with correct version for Python
    if py_minor <= 11:
        print_status(f"Installing PyTorch 2.2.2 (CPU-only) for Python 3.{py_minor}...")
        if not pip_install_with_retry(pip_exe, "torch==2.2.2+cpu", 
                                       ["--index-url", "https://download.pytorch.org/whl/cpu"],
                                       max_retries=10, initial_delay=5.0):
            print_status("PyTorch installation failed after retries", False)
            return False
        print_status("PyTorch 2.2.2 (CPU) installed")
        transformers_version = "transformers==4.41.2"
        sentence_transformers_version = "sentence-transformers==3.0.1"
    else:
        # Python 3.12+ needs newer torch
        print_status(f"Installing PyTorch 2.4+ (CPU-only) for Python 3.{py_minor}...")
        if not pip_install_with_retry(pip_exe, "torch>=2.4.0", 
                                       ["--index-url", "https://download.pytorch.org/whl/cpu"],
                                       max_retries=10, initial_delay=5.0):
            print_status("PyTorch installation failed after retries", False)
            return False
        print_status("PyTorch 2.4+ (CPU) installed")
        # Newer torch needs newer transformers/sentence-transformers
        transformers_version = "transformers>=4.42.0"
        sentence_transformers_version = "sentence-transformers>=3.0.0"
    
    # Install transformers
    print_status(f"Installing {transformers_version}...")
    if not pip_install_with_retry(pip_exe, transformers_version, 
                                   max_retries=10, initial_delay=5.0):
        print_status("transformers installation failed", False)
        return False
    print_status("transformers installed")
    
    # Install sentence-transformers
    print_status(f"Installing {sentence_transformers_version}...")
    if not pip_install_with_retry(pip_exe, sentence_transformers_version,
                                   max_retries=10, initial_delay=5.0):
        print_status("sentence-transformers installation failed", False)
        return False
    print_status("sentence-transformers installed")
    
    # Verify
    verify_script = '''
import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    import torch
    torch.set_grad_enabled(False)
    print(f"torch: {torch.__version__}")
    from sentence_transformers import SentenceTransformer
    print("sentence_transformers: OK")
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
    sys.exit(1)
'''
    verify_path = TEMP_DIR / "verify_embedding.py"
    try:
        with open(verify_path, 'w') as f:
            f.write(verify_script)
            
        verify_result = subprocess.run(
            [python_exe, str(verify_path)], capture_output=True, text=True, timeout=120
        )
        verify_path.unlink(missing_ok=True)
        
        if verify_result.returncode == 0 and "SUCCESS" in verify_result.stdout:
            print_status("Embedding backend verified")
            return True
        else:
            print_status("Embedding backend verification failed", False)
            print(f"Output: {verify_result.stdout}")
            return False
    except Exception as e:
        print_status(f"Verification error: {e}", False)
        return False

def install_qt_webengine() -> bool:
    """
    Install Qt WebEngine for custom browser.
    
    Version Matrix:
    - Windows 7/8/8.1: PyQt5 + PyQtWebEngine (Qt5)
    - Windows 10/11: PyQt6 + PyQt6-WebEngine (Qt6)
    - Ubuntu 22/23: PyQt5 + PyQtWebEngine (Qt5)
    - Ubuntu 24/25: PyQt6 + PyQt6-WebEngine (Qt6)
    """
    print_status("Installing Qt WebEngine for custom browser...")
    
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                ("pip.exe" if PLATFORM == "windows" else "pip"))
    
    qt_version, use_qt5 = get_qt_version_for_os()
    
    try:
        if use_qt5:
            os_name = f"Windows {detect_windows_version()}" if PLATFORM == "windows" else f"Ubuntu {detect_linux_version()}"
            print_status(f"{os_name} - installing PyQt5 + Qt5 WebEngine...")
            
            if not pip_install_with_retry(pip_exe, "PyQt5>=5.15.0,<5.16.0", max_retries=3, initial_delay=5.0):
                print_status("PyQt5 installation failed - will use system browser", False)
                return False
            if not pip_install_with_retry(pip_exe, "PyQtWebEngine>=5.15.0,<5.16.0", max_retries=3, initial_delay=5.0):
                print_status("PyQtWebEngine installation failed - will use system browser", False)
                return False
        else:
            os_name = f"Windows {detect_windows_version()}" if PLATFORM == "windows" else f"Ubuntu {detect_linux_version()}"
            print_status(f"{os_name} - installing PyQt6 + Qt6 WebEngine...")
            
            if not pip_install_with_retry(pip_exe, "PyQt6>=6.5.0", max_retries=3, initial_delay=5.0):
                print_status("PyQt6 installation failed - will use system browser", False)
                return False
            if not pip_install_with_retry(pip_exe, "PyQt6-WebEngine>=6.5.0", max_retries=3, initial_delay=5.0):
                print_status("PyQt6-WebEngine installation failed - will use system browser", False)
                return False
        
        print_status(f"Qt{qt_version} WebEngine installed")
        return True
            
    except Exception as e:
        print_status(f"Qt WebEngine error: {e} - will use system browser", False)
        return False

def initialize_embedding_cache(embedding_model: str) -> bool:
    """Initialize embedding model cache using sentence-transformers"""
    print_status(f"Initializing embedding cache for {embedding_model}...")
    
    cache_dir = BASE_DIR / "data" / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                    ("python.exe" if PLATFORM == "windows" else "python"))
    
    init_script = f'''
import os, sys
from pathlib import Path

# Force CPU-only mode - no CUDA
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

cache_dir = Path(r"{str(cache_dir)}")
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir.absolute())
os.environ["HF_HOME"] = str(cache_dir.parent.absolute())
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir.absolute())

try:
    print("Importing torch...", flush=True)
    import torch
    torch.set_grad_enabled(False)
    print(f"torch version: {{torch.__version__}}", flush=True)
    print(f"CUDA available: {{torch.cuda.is_available()}} (should be False)", flush=True)
    
    print("Importing sentence_transformers...", flush=True)
    from sentence_transformers import SentenceTransformer
    
    print(f"Loading model: {embedding_model}", flush=True)
    model = SentenceTransformer("{embedding_model}", device="cpu")
    model.eval()
    
    print("Testing embedding...", flush=True)
    # Use convert_to_tensor=True to avoid numpy conversion issues
    test_embedding = model.encode(["test"], batch_size=1, normalize_embeddings=True, convert_to_tensor=True)
    dim = test_embedding.shape[1] if len(test_embedding.shape) > 1 else len(test_embedding)
    print(f"SUCCESS: Model loaded, dimension: {{dim}}", flush=True)
    
except ImportError as e:
    print(f"FATAL: Import failed - {{type(e).__name__}}: {{str(e)}}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"FATAL: {{type(e).__name__}}: {{str(e)}}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''

    script_path = TEMP_DIR / "init_embedding.py"
    try:
        with open(script_path, 'w') as f:
            f.write(init_script)
        
        print("Embedding Initialization Output...")
        
        result = subprocess.run(
            [python_exe, str(script_path)], capture_output=True, text=True, timeout=600
        )
        
        if result.stdout:
            for line in result.stdout.strip().split('\n'):
                print(f"    {line}")
        if result.stderr and "error" in result.stderr.lower():
            print("    STDERR:", result.stderr[:200])
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            print_status("Embedding cache initialized")
            return True
        else:
            print_status("Embedding initialization failed", False)
            print(f"Return code: {result.returncode}")
            return False
        
    except subprocess.TimeoutExpired:
        print_status("Embedding initialization timed out (>600s)", False)
        return False
    except Exception as e:
        print_status(f"Embedding initialization failed: {e}", False)
        return False
    finally:
        script_path.unlink(missing_ok=True)

def download_spacy_model() -> bool:
    """Download spaCy English model during installation."""
    try:
        pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                     ("pip.exe" if PLATFORM == "windows" else "pip"))
        
        print_status("Downloading spaCy language model...")
        
        filename = "en_core_web_sm-3.8.0-py3-none-any.whl"
        url = f"https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/{filename}"
        whl_path = TEMP_DIR / filename
        
        download_with_progress(url, whl_path, "Downloading spaCy model")
        
        print_status("Installing spaCy model...")
        result = subprocess.run(
            [pip_exe, "install", "--no-cache-dir", str(whl_path)], 
            capture_output=True, text=True, timeout=600
        )
        
        whl_path.unlink(missing_ok=True)
        
        if result.returncode == 0:
            print_status("spaCy model installed")
            return True
        else:
            print_status(f"spaCy install failed (code {result.returncode})", False)
            return False
            
    except subprocess.TimeoutExpired:
        print_status("spaCy install timed out", False)
        return False
    except Exception as e:
        print_status(f"spaCy error: {e}", False)
        return False

def download_with_progress(url: str, filepath: Path, description: str = "Downloading",
                          max_retries: int = 10, initial_delay: float = 5.0) -> None:
    """Download file with progress bar, resume capability, and exponential backoff retries"""
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                    ("python.exe" if PLATFORM == "windows" else "python"))
    
    download_script = f'''
import requests, time, sys
from pathlib import Path

def format_bytes(b):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024.0:
            return f"{{b:.1f}}{{unit}}"
        b /= 1024.0
    return f"{{b:.1f}}TB"

def progress_bar(current, total, width=30):
    if total == 0:
        return "[" + "=" * width + "] 100%"
    filled = int(width * current // total)
    bar = "=" * filled + "-" * (width - filled)
    pct = 100 * current // total
    return f"[{{bar}}] {{pct}}% ({{format_bytes(current)}}/{{format_bytes(total)}})"

filepath = Path(r"{str(filepath)}")
max_retries = {max_retries}
delay = {initial_delay}

for attempt in range(max_retries):
    try:
        existing_size = filepath.stat().st_size if filepath.exists() else 0
        headers = {{'Range': f'bytes={{existing_size}}-'}} if existing_size > 0 else {{}}
        
        response = requests.get("{url}", stream=True, headers=headers, timeout=60)
        
        if response.status_code == 416:
            print(f"\\r{description}: Already complete", flush=True)
            break
        elif response.status_code == 206:
            total_size = existing_size + int(response.headers.get('content-length', 0))
        elif response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            existing_size = 0
            filepath.unlink(missing_ok=True)
        else:
            response.raise_for_status()
        
        downloaded = existing_size
        mode = 'ab' if existing_size > 0 else 'wb'
        last_update = 0
        
        with open(filepath, mode) as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    # Update progress every 1%
                    if total_size > 0:
                        current_pct = int(100 * downloaded / total_size)
                        if current_pct > last_update:
                            print(f"\\r{description}: {{progress_bar(downloaded, total_size)}}", end='', flush=True)
                            last_update = current_pct
                    elif downloaded % (1024 * 1024) < 8192:  # Every ~1MB for unknown size
                        print(f"\\r{description}: {{format_bytes(downloaded)}} downloaded", end='', flush=True)
        
        print(f"\\r{description}: {{progress_bar(total_size, total_size)}} - Complete", flush=True)
        break
        
    except Exception as e:
        print(f"\\n{description}: Error - {{e}}", flush=True)
        if attempt < max_retries - 1:
            print(f"{description}: Retry {{attempt + 1}}/{{max_retries}} in {{delay:.0f}}s...", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 300)  # Cap at 5 minutes
        else:
            filepath.unlink(missing_ok=True)
            print(f"{description}: FAILED after {{max_retries}} attempts", flush=True)
            sys.exit(1)
'''
    
    download_script_path = TEMP_DIR / "download_file.py"
    try:
        with open(download_script_path, 'w') as f:
            f.write(download_script)
        
        subprocess.run([python_exe, str(download_script_path)], check=True, timeout=3600)
        
    except subprocess.CalledProcessError as e:
        filepath.unlink(missing_ok=True)
        raise Exception(f"Download failed: {e}")
    except subprocess.TimeoutExpired:
        filepath.unlink(missing_ok=True)
        raise Exception("Download timed out")
    finally:
        download_script_path.unlink(missing_ok=True)

def pip_install_with_retry(pip_exe: str, package: str, extra_args: list = None, 
                           max_retries: int = 10, initial_delay: float = 5.0) -> bool:
    """Install a pip package with retry logic and exponential backoff."""
    if extra_args is None:
        extra_args = []
    
    pkg_name = package.split('>=')[0].split('==')[0].split('[')[0]
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            cmd = [pip_exe, "install", package] + extra_args
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                return True
            
            # Check if already installed
            if "already satisfied" in result.stdout.lower():
                return True
                
            if attempt < max_retries - 1:
                print(f"    Retry {attempt + 1}/{max_retries} for {pkg_name} in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 300)  # Cap at 5 minutes
            
        except subprocess.TimeoutExpired:
            if attempt < max_retries - 1:
                print(f"    Timeout, retry {attempt + 1}/{max_retries} for {pkg_name} in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 300)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Error: {e}, retry {attempt + 1}/{max_retries} in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 300)
    
    return False

def install_python_deps(backend: str) -> bool:
    """Install Python dependencies with dynamic version selection."""
    print_status("Installing Python dependencies...")
    try:
        python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                        ("python.exe" if PLATFORM == "windows" else "python"))
        pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                     ("pip.exe" if PLATFORM == "windows" else "pip"))
        
        # Get dynamic requirements with correct gradio version
        requirements = get_dynamic_requirements()
        
        # Install base requirements
        print_status(f"Installing base packages (Gradio {SELECTED_GRADIO})...")
        total_packages = len(requirements)
        for i, req in enumerate(requirements, 1):
            pkg_name = req.split('>=')[0].split('==')[0].split('[')[0]
            print(f"  [{i}/{total_packages}] Installing {pkg_name}...", end='', flush=True)
            
            if pip_install_with_retry(pip_exe, req, max_retries=10, initial_delay=5.0):
                print(f" OK")
            else:
                print(f" FAILED")
                print_status(f"Failed to install {pkg_name} after 10 retries", False)
                return False
        
        print_status("Base packages installed")
        
        # Install embedding backend (torch + sentence-transformers)
        if not install_embedding_backend():
            return False
        
        # Install Qt WebEngine for custom browser
        install_qt_webengine()  # Non-fatal if fails
        
        # llama-cpp-python installation
        info = BACKEND_OPTIONS[backend]
        
        if not info.get("compile_wheel"):
            print_status("Installing pre-built llama-cpp-python (CPU)...")
            if pip_install_with_retry(pip_exe, "llama-cpp-python", 
                                      ["--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"],
                                      max_retries=10, initial_delay=5.0):
                print_status("Pre-built wheel installed")
            else:
                print_status("Pre-built wheel failed, building from source...")
                if not build_llama_cpp_python_with_flags({}):
                    return False
        else:
            build_flags = info.get("build_flags", {})
            
            if build_flags.get("GGML_VULKAN"):
                print_status("Vulkan wheel build - checking Vulkan SDK...")
                if not check_vulkan_sdk_installed():
                    print_status("Error: Vulkan SDK not found", False)
                    print("Please install Vulkan SDK before selecting Vulkan build options")
                    return False
                else:
                    print_status("Vulkan SDK detected")
            
            if not build_llama_cpp_python_with_flags(build_flags):
                return False

        print_status("Python dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print_status(f"Install failed: {e}", False)
        return False

def install_optional_file_support() -> bool:
    """Install optional file format libraries"""
    print_status("Installing optional file format support...")
    
    optional_packages = [
        "PyPDF2>=3.0.0", "python-docx>=0.8.11", 
        "openpyxl>=3.0.0", "python-pptx>=0.6.21"
    ]
    
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                 ("pip.exe" if PLATFORM == "windows" else "pip"))
    
    for package in optional_packages:
        try:
            subprocess.run([pip_exe, "install", package], check=True, capture_output=True)
            print_status(f"  Installed {package.split('>=')[0]}")
        except subprocess.CalledProcessError:
            print_status(f"  Optional package {package.split('>=')[0]} failed", False)
    
    return True

def compile_llama_cpp_binary(backend: str, info: dict) -> bool:
    """Compile llama.cpp binaries from source"""
    import traceback
    
    global _DID_COMPILATION
    _DID_COMPILATION = True
    
    snapshot_pre_existing_processes()
    
    print_status("Compiling llama.cpp binaries from source (15-30 minutes)...")
    
    dest_path = BASE_DIR / info["dest"]
    dest_path.mkdir(parents=True, exist_ok=True)
    
    llamacpp_src = TEMP_DIR / "llama.cpp"
    
    import multiprocessing
    try:
        total_threads = multiprocessing.cpu_count()
    except:
        total_threads = 4
    
    build_threads = max(1, int(total_threads * 0.85))
    print(f"  Building with {build_threads} of {total_threads} threads (85%)")
    
    env = os.environ.copy()
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(build_threads)
    
    if PLATFORM == "windows":
        env["FORCE_CMAKE"] = "1"
        env["CL"] = f"/MP{build_threads}"
    else:
        env["MAKEFLAGS"] = f"-j{build_threads}"
    
    try:
        subprocess.run(["git", "config", "--global", "http.lowSpeedLimit", "200"], capture_output=True)
        subprocess.run(["git", "config", "--global", "http.lowSpeedTime", "240"], capture_output=True)
        
        # Clean any existing repo directory before cloning
        if llamacpp_src.exists():
            print_status("Cleaning previous build artifacts...")
            try:
                shutil.rmtree(llamacpp_src, ignore_errors=False)
            except PermissionError:
                if PLATFORM == "windows":
                    subprocess.run(["cmd", "/c", "rmdir", "/s", "/q", str(llamacpp_src)], 
                                  capture_output=True, timeout=30)
                time.sleep(1)
                if llamacpp_src.exists():
                    shutil.rmtree(llamacpp_src, ignore_errors=True)
        
        print_status("Cloning llama.cpp repository...")
        
        max_retries = 5
        retry_delay = 15
        
        for attempt in range(max_retries):
            try:
                process = subprocess.Popen([
                    "git", "clone", "--depth", "1", "--progress",
                    LLAMACPP_GIT_REPO, str(llamacpp_src)
                ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                   text=True, bufsize=1, env=env)
                
                for line in process.stdout:
                    line = line.strip()
                    if line and any(x in line for x in ["Receiving", "Resolving", "Counting", "Compressing"]):
                        print(f"\r  {line[:70]}", end='', flush=True)
                
                process.wait(timeout=600)
                print()
                
                if process.returncode == 0:
                    break
                else:
                    raise subprocess.CalledProcessError(process.returncode, "git clone")
                    
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if llamacpp_src.exists():
                    shutil.rmtree(llamacpp_src, ignore_errors=True)
                if attempt < max_retries - 1:
                    print(f"\n  Clone failed, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print_status("Failed to clone llama.cpp after retries", False)
                    return False
        
        print_status("Repository cloned")
        
        # Configure build
        build_dir = llamacpp_src / "build"
        build_dir.mkdir(exist_ok=True)
        
        cmake_args = [
            "cmake", "..",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_BUILD_PARALLEL_LEVEL={build_threads}",
            "-DLLAMA_CURL=OFF",  # Disable CURL dependency
        ]
        
        build_flags = info.get("build_flags", {})
        
        # CPU optimization flags
        cpu_features = detect_cpu_features()
        if cpu_features.get("AVX"):
            cmake_args.append("-DGGML_AVX=ON")
        if cpu_features.get("AVX2"):
            cmake_args.append("-DGGML_AVX2=ON")
        if cpu_features.get("FMA"):
            cmake_args.append("-DGGML_FMA=ON")
        if cpu_features.get("F16C"):
            cmake_args.append("-DGGML_F16C=ON")
        
        # Backend-specific flags (e.g., Vulkan)
        if build_flags.get("GGML_VULKAN"):
            cmake_args.append("-DGGML_VULKAN=ON")
        
        # Platform-specific generator
        if PLATFORM == "windows":
            if VS_GENERATOR:
                cmake_args.extend(["-G", VS_GENERATOR, "-A", "x64"])
            else:
                # Fallback - let CMake auto-detect, just specify architecture
                cmake_args.extend(["-A", "x64"])
        
        print_status("Configuring build...")
        result = subprocess.run(cmake_args, cwd=build_dir, capture_output=True, text=True, timeout=300, env=env)
        
        if result.returncode != 0:
            print_status(f"CMake configure failed: {result.stderr[:200]}", False)
            return False
        
        print_status("Building binaries...")
        
        if PLATFORM == "windows":
            build_cmd = ["cmake", "--build", ".", "--config", "Release", "--parallel", str(build_threads)]
        else:
            build_cmd = ["cmake", "--build", ".", "--parallel", str(build_threads)]
        
        process = subprocess.Popen(
            build_cmd, cwd=build_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env
        )
        
        track_process(process.pid)
        
        for line in process.stdout:
            line = line.strip()
            if line and any(x in line.lower() for x in ["building", "compiling", "linking"]):
                print(f"\r  {line[:70]}", end='', flush=True)
        
        process.wait(timeout=3600)
        print()
        
        if process.returncode != 0:
            print_status("Build failed", False)
            return False
        
        # Copy binaries
        print_status("Copying binaries...")
        
        if PLATFORM == "windows":
            bin_src = build_dir / "bin" / "Release"
        else:
            bin_src = build_dir / "bin"
        
        if not bin_src.exists():
            bin_src = build_dir
        
        for item in bin_src.iterdir():
            if item.is_file():
                shutil.copy2(item, dest_path)
        
        cli_path = BASE_DIR / info["cli_path"]
        if not cli_path.exists():
            print_status(f"llama-cli not found at {cli_path}", False)
            return False
        
        if PLATFORM == "linux":
            os.chmod(cli_path, 0o755)
        
        print_status("llama.cpp binaries compiled successfully")
        return True
        
    except Exception as e:
        print_status(f"Compilation failed: {e}", False)
        traceback.print_exc()
        return False
    finally:
        if llamacpp_src.exists():
            try:
                shutil.rmtree(llamacpp_src, ignore_errors=True)
            except:
                pass

def copy_linux_binaries(src_path: Path, dest_path: Path) -> None:
    """Copy Linux binaries from extracted archive to destination and set permissions.
    
    Handles nested directory structures from tarballs (e.g., llama-b7688-bin-ubuntu-vulkan-x64/bin/)
    """
    # First, check if there's a nested directory (common in GitHub release tarballs)
    subdirs = [d for d in src_path.iterdir() if d.is_dir()]
    
    # If there's exactly one subdirectory and it looks like the release folder, use it
    if len(subdirs) == 1 and subdirs[0].name.startswith("llama-"):
        actual_src = subdirs[0]
        # Check for bin subdirectory
        if (actual_src / "bin").exists():
            actual_src = actual_src / "bin"
    elif (src_path / "bin").exists():
        actual_src = src_path / "bin"
    else:
        actual_src = src_path
    
    # Copy all executable files
    copied_count = 0
    for item in actual_src.iterdir():
        if item.is_file():
            dest = dest_path / item.name
            shutil.copy2(item, dest)
            # Set executable permission for files without extensions (Linux binaries)
            if not item.suffix or item.suffix in ['.so']:
                os.chmod(dest, 0o755)
            copied_count += 1
    
    # Also copy any .so files from lib directory if present
    lib_candidates = [
        src_path / "lib",
        subdirs[0] / "lib" if len(subdirs) == 1 else None,
    ]
    for lib_dir in lib_candidates:
        if lib_dir and lib_dir.exists():
            for item in lib_dir.iterdir():
                if item.is_file() and '.so' in item.name:
                    dest = dest_path / item.name
                    shutil.copy2(item, dest)
                    os.chmod(dest, 0o755)
                    copied_count += 1
    
    if copied_count > 0:
        print_status(f"Copied {copied_count} binary files")

def download_extract_backend(backend: str) -> bool:
    """Download and extract backend binaries"""
    import zipfile
    import tarfile
    
    info = BACKEND_OPTIONS[backend]
    
    if info.get("compile_binary"):
        return compile_llama_cpp_binary(backend, info)
    
    if not info["url"]:
        print_status("No backend download required for this option")
        return True
    
    print_status("Downloading backend binaries...")
    
    # Determine file type from URL
    url = info["url"]
    if url.endswith(".tar.gz"):
        temp_archive = TEMP_DIR / "backend.tar.gz"
        is_tarball = True
    else:
        temp_archive = TEMP_DIR / "backend.zip"
        is_tarball = False
    
    try:
        download_with_progress(url, temp_archive, "Downloading backend")
        
        print_status("Extracting backend...")
        
        dest_path = BASE_DIR / info["dest"]
        dest_path.mkdir(parents=True, exist_ok=True)

        if is_tarball:
            # Handle .tar.gz (Linux)
            with tarfile.open(temp_archive, 'r:gz') as tf:
                members = tf.getmembers()
                total = len(members)
                for i, member in enumerate(members):
                    # Use filter for Python 3.12+ compatibility
                    tf.extract(member, dest_path, filter='data' if sys.version_info >= (3, 12) else None)
                    if i % 25 == 0 or i == total - 1:
                        print(f"\rExtracting: {simple_progress_bar(i + 1, total)}", end='', flush=True)
                print()
        else:
            # Handle .zip (Windows)
            with zipfile.ZipFile(temp_archive, 'r') as zf:
                members = zf.namelist()
                total = len(members)
                for i, m in enumerate(members):
                    zf.extract(m, dest_path)
                    if i % 25 == 0 or i == total - 1:
                        print(f"\rExtracting: {simple_progress_bar(i + 1, total)}", end='', flush=True)
                print()

        if PLATFORM == "linux":
            # Copy binaries from nested structure to dest_path root
            copy_linux_binaries(dest_path, dest_path)

        cli_path = BASE_DIR / info["cli_path"]
        if not cli_path.exists():
            # Debug: list what was actually extracted
            print(f"  Debug: Contents of {dest_path}:")
            for item in dest_path.iterdir():
                print(f"    {item.name}{'/' if item.is_dir() else ''}")
                if item.is_dir():
                    for subitem in item.iterdir():
                        print(f"      {subitem.name}{'/' if subitem.is_dir() else ''}")
            raise FileNotFoundError(f"llama-cli not found: {cli_path}")
        
        if PLATFORM == "linux":
            os.chmod(cli_path, 0o755)

        print_status("Backend ready")
        return True
    except Exception as e:
        print_status(f"Backend install failed: {e}", False)
        return False
    finally:
        temp_archive.unlink(missing_ok=True)
		
def clean_compile_temp() -> None:
    """Clean up Windows compilation temp folder"""
    if PLATFORM == "windows" and WIN_COMPILE_TEMP.exists():
        try:
            shutil.rmtree(WIN_COMPILE_TEMP, ignore_errors=True)
            print_status("Cleaned up compilation temp folder")
        except:
            pass

def detect_build_tools_available() -> dict:
    """Detect build tools availability and add to PATH if found. Returns dict of tool: bool"""
    global VS_GENERATOR
    tools = {"Git": False, "CMake": False, "MSVC": False, "MSBuild": False}
    
    # Check Git (required for all platforms)
    if shutil.which("git"):
        tools["Git"] = True
    
    if PLATFORM == "windows":
        # Check CMake
        if shutil.which("cmake"):
            tools["CMake"] = True
        else:
            vs_base = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Microsoft Visual Studio"
            if vs_base.exists():
                for year in ["2022", "2019", "2017"]:
                    for edition in ["Community", "Professional", "Enterprise", "BuildTools"]:
                        cmake_candidate = vs_base / year / edition / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "CMake" / "bin" / "cmake.exe"
                        if cmake_candidate.exists():
                            tools["CMake"] = True
                            bin_dir = str(cmake_candidate.parent)
                            if bin_dir not in os.environ["PATH"]:
                                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
                            break
                    if tools["CMake"]:
                        break
            if not tools["CMake"]:
                cmake_pf = Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "CMake" / "bin" / "cmake.exe"
                if cmake_pf.exists():
                    tools["CMake"] = True
                    bin_dir = str(cmake_pf.parent)
                    if bin_dir not in os.environ["PATH"]:
                        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
        
        # Check MSVC and detect version - map year to generator
        vs_generators = {
            "2022": "Visual Studio 17 2022",
            "2019": "Visual Studio 16 2019",
            "2017": "Visual Studio 15 2017",
        }
        
        vs_base = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Microsoft Visual Studio"
        if vs_base.exists():
            # Check each year in preference order (newest first)
            for year in ["2022", "2019", "2017"]:
                for edition in ["Community", "Professional", "Enterprise", "BuildTools"]:
                    vc_path = vs_base / year / edition / "VC"
                    if vc_path.exists():
                        tools["MSVC"] = True
                        if VS_GENERATOR is None:
                            VS_GENERATOR = vs_generators[year]
                        break
                if tools["MSVC"]:
                    break
        
        # Fallback: try vswhere
        if not tools["MSVC"]:
            try:
                vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
                if vswhere.exists():
                    result = subprocess.run(
                        [str(vswhere), "-latest", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationVersion"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        tools["MSVC"] = True
                        version = result.stdout.strip()
                        # Map major version to generator
                        if version.startswith("17"):
                            VS_GENERATOR = "Visual Studio 17 2022"
                        elif version.startswith("16"):
                            VS_GENERATOR = "Visual Studio 16 2019"
                        elif version.startswith("15"):
                            VS_GENERATOR = "Visual Studio 15 2017"
            except:
                pass
        
        # Check MSBuild
        if shutil.which("MSBuild"):
            tools["MSBuild"] = True
        else:
            if vs_base.exists():
                for year in ["2022", "2019", "2017"]:
                    for edition in ["Community", "Professional", "Enterprise", "BuildTools"]:
                        msbuild_candidate = vs_base / year / edition / "MSBuild" / "Current" / "Bin" / "MSBuild.exe"
                        if msbuild_candidate.exists():
                            tools["MSBuild"] = True
                            bin_dir = str(msbuild_candidate.parent)
                            if bin_dir not in os.environ["PATH"]:
                                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
                            break
                    if tools["MSBuild"]:
                        break
    
    else:  # Linux
        tools["CMake"] = shutil.which("cmake") is not None
        tools["MSVC"] = shutil.which("gcc") is not None  # GCC on Linux
        tools["MSBuild"] = shutil.which("make") is not None
    
    return tools

def select_backend_and_embedding():
    """Combined selection of backend and embedding model on one page"""
    
    width = shutil.get_terminal_size().columns - 1
    print_header("Configure Installation")
    
    all_backend_opts = list(BACKEND_OPTIONS.keys())
    embed_opts = EMBEDDING_MODELS
    
    # System Detections
    print("System Detections...")
    
    cpu_features = detect_cpu_features()
    features_list = [feat for feat, supported in cpu_features.items() if supported]
    if features_list:
        print(f"    CPU Features: {', '.join(features_list)}")
    else:
        print("    CPU Features: Baseline")
    
    if PLATFORM == "windows":
        win_ver = WINDOWS_VERSION or "unknown"
        print(f"    Operating System: Windows {win_ver}")
    else:
        ubuntu_ver = OS_VERSION or "unknown"
        print(f"    Operating System: Ubuntu {ubuntu_ver}")
    
    optimal_py = DETECTED_PYTHON_INFO.get("optimal_version", "None") if DETECTED_PYTHON_INFO else "None"
    vulkan_present = is_vulkan_installed()
    print(f"    Optimal Python: {optimal_py}; Vulkan Present: {'Yes' if vulkan_present else 'No'}")
    
    build_tools = detect_build_tools_available()
    available_tools = [tool for tool, present in build_tools.items() if present]
    missing_tools = [tool for tool, present in build_tools.items() if not present]
    tools_str = ", ".join(available_tools) if available_tools else "None"
    print(f"    Build Tools: {tools_str}")
    
    print()
    
    # Filter backend options based on build tools availability
    build_possible = len(missing_tools) == 0
    backend_opts = []
    for backend in all_backend_opts:
        info = BACKEND_OPTIONS[backend]
        requires_compile = info.get("compile_binary", False) or info.get("compile_wheel", False)
        if requires_compile and not build_possible:
            # Skip compile options if build tools missing
            continue
        backend_opts.append(backend)
    
    # Backend options
    print("Backend Options...")
    for i, backend in enumerate(backend_opts, 1):
        print(f"   {i}) {backend}")
    
    # Show disabled compile options with missing tools
    if not build_possible:
        print(f"   ---")
        for backend in all_backend_opts:
            if backend not in backend_opts:
                print(f"   -) {backend} (Missing: {', '.join(missing_tools)})")
    
    print()
    
    # Embedding Model
    print("Embedding Model...")
    embed_letters = ['a', 'b', 'c']
    for i, key in enumerate(sorted(embed_opts.keys())):
        letter = embed_letters[i]
        model = embed_opts[key]
        print(f"   {letter}) {model['display']}")
    
    print()
    print("=" * width)
    
    max_backend = len(backend_opts)
    prompt = f"Selection; Backend=1-{max_backend}, Embed=a-c, Abandon=A; (e.g. 2b): "
    
    choice = input(prompt).strip().lower()
    
    if choice == "a":
        print("Abandoning installation...")
        sys.exit(0)
        
    choice = choice.replace(" ", "").replace("-", "")
    
    while True:
        # Parse choice: digit + embed letter
        # Valid formats: "2b", "1a", "3c"
        if len(choice) >= 2 and choice[0].isdigit() and choice[1] in "abc":
            backend_num = int(choice[0])
            embed_letter = choice[1]
            
            if 1 <= backend_num <= len(backend_opts):
                embed_key = str(ord(embed_letter) - 96)
                if embed_key in embed_opts:
                    selected_backend = backend_opts[backend_num - 1]
                    selected_model = embed_opts[embed_key]["name"]
                    
                    time.sleep(1)
                    return selected_backend, selected_model
        
        print("Invalid selection. Please enter a valid combination (e.g. 2b).")
        prompt = f"Selection; Backend 1-{max_backend}, Embed a-c (e.g. 2b), Abandon = A: "
        
        choice = input(prompt).strip().lower()
        if choice == "a":
            print("\nAbandoning installation...")
            sys.exit(0)
        choice = choice.replace(" ", "").replace("-", "")

# Main install flow
def install():
    global WINDOWS_VERSION, OS_VERSION
    
    # Version compatibility check FIRST
    if not check_version_compatibility():
        sys.exit(1)

    # Detect Python/Gradio/Qt versions for display
    detect_version_selections()
    
    # Clean temp directories at start (handles leftover from failed builds)
    if TEMP_DIR.exists():
        try:
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
        except:
            pass
    if PLATFORM == "windows" and WIN_COMPILE_TEMP.exists():
        try:
            shutil.rmtree(WIN_COMPILE_TEMP, ignore_errors=True)
        except:
            pass
    
    # Create temp dir
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create venv and install py-cpuinfo early for CPU detection
    if not create_venv():
        print_status("Virtual environment failed", False)
        sys.exit(1)
    
    print_status("Installing py-cpuinfo for CPU detection...")
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("pip.exe" if PLATFORM == "windows" else "pip"))
    subprocess.run([pip_exe, "install", "py-cpuinfo"], check=True)
    print_status("py-cpuinfo installed")
    
    if PLATFORM == "windows" and WINDOWS_VERSION == "8.1":
        print("Detected Windows 8.1 - Using Qt5 WebEngine")
    
    # Get system info before creating config
    os_version = "unknown"
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    windows_version = None
    vulkan_available = False
    
    if PLATFORM == "windows":
        WINDOWS_VERSION = detect_windows_version() or "unknown"
        os_version = WINDOWS_VERSION
        windows_version = WINDOWS_VERSION
        vulkan_available = is_vulkan_installed()
    elif PLATFORM == "linux":
        OS_VERSION = detect_linux_version() or "unknown"
        os_version = OS_VERSION
        vulkan_available = is_vulkan_installed()

    backend, embedding_model = select_backend_and_embedding()
    
    print_header("Installation")
    if PLATFORM == "windows":
        os_display = f"Windows {WINDOWS_VERSION}" if WINDOWS_VERSION else "Windows"
    else:
        os_display = f"Ubuntu {OS_VERSION}" if OS_VERSION else "Ubuntu"
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    print(f"Installing {APP_NAME} on {os_display} with Python {py_ver}")
    print(f"  Route: {backend}")
    print(f"  Llama.Cpp {LLAMACPP_TARGET_VERSION}, Gradio {SELECTED_GRADIO}, Qt-Web {SELECTED_QTWEB}")

    print(f"Embedding model: {embedding_model}")
    
    # Determine backend_type
    if backend in ["Download CPU Wheel / Default CPU Wheel", "Compile CPU Binaries / Compile CPU Wheel"]:
        backend_type = "CPU_CPU"
        vulkan_available = False
    elif backend in ["Download Vulkan Bin / Default CPU Wheel"]:
        backend_type = "VULKAN_CPU"
        vulkan_available = True
    elif backend in ["Download Vulkan Bin / Compile Vulkan Wheel", "Compile Vulkan Binaries / Compile Vulkan Wheel"]:
        backend_type = "VULKAN_VULKAN"
        vulkan_available = True
    else:
        backend_type = "CPU_CPU"
        vulkan_available = False

    # Create directories
    create_directories(backend)
    
    info = BACKEND_OPTIONS[backend]
    
    # Create constants.ini early
    create_system_ini(
        platform=PLATFORM,
        os_version=os_version,
        python_version=python_version,
        backend_type=backend_type,
        embedding_model=embedding_model,
        windows_version=windows_version,
        vulkan_available=vulkan_available,
        llama_cli_path=info["cli_path"],
        llama_bin_path=info["dest"]
    )
    
    # Install system dependencies BEFORE checking build tools
    if PLATFORM == "linux":
        if not install_linux_system_dependencies(backend):
            print_status("System dependencies installation failed", False)
            sys.exit(1)
    
    # Install Python dependencies
    if not install_python_deps(backend):
        print_status("Python dependencies failed", False)
        sys.exit(1)

    install_optional_file_support()

    # Initialize embedding cache
    if not initialize_embedding_cache(embedding_model):
        print_status("CRITICAL: Embedding model required by RAG", False)
        sys.exit(1)

    spacy_ok = download_spacy_model()
    if not spacy_ok:
        print_status("WARNING: spaCy model download failed - session labeling may not work", False)

    # Download/compile backend
    if not download_extract_backend(backend):
        print_status("Backend download failed", False)
        sys.exit(1)

    create_config(backend, embedding_model)
    
    print_status("Installation complete!")
    print("\nRun the launcher to start Chat-Gradio-Gguf\n")

#  Protected main block
if __name__ == "__main__":
    try:
        if len(sys.argv) < 2 or sys.argv[1].lower() not in ["windows", "linux"]:
            print("ERROR: Platform argument required (windows/linux)")
            sys.exit(1)
        PLATFORM = sys.argv[1].lower()
        install()
    except KeyboardInterrupt:
        print("\nInstallation cancelled")
        sys.exit(1)
    except Exception as e:
        print(f"\nInstallation failed: {e}")
        sys.exit(1)
    finally:
        time.sleep(2)
        cleanup_build_processes()
        if PLATFORM == "windows":
            clean_compile_temp()
        sys.exit(0)