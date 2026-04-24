# Script: installer.py - Installation script for Chat-Gradio-Gguf
# Note: All install routes that state download NOT compile, should NOT compile.
# Note: Uses sentence-transformers for embeddings, cross-platform, Win 7-11 and Ubuntu 22-25
# Note: Uses Qt5 WebEngine for custom browser fake GUI

# Imports
import os
import json
import subprocess
import sys
import contextlib
import time
import threading
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
# llama-cpp-python version strategy:
#   LLAMACPP_PYTHON_PREBUILT_VERSION — last version with CPU prebuilt wheels
#     from eswarthammana (cpu-only, Win/Linux/Mac, cp38-cp313).
#     eswarthammana stopped publishing after v0.3.16 (Aug 2025).
#     v0.3.16 bundles llama.cpp from Aug 2025 → does NOT support qwen35
#     (qwen35 was added Feb 2026, b8076).  Users who need qwen35/Qwen3.5
#     models MUST use a compile route (options 3 or 4).
LLAMACPP_PYTHON_PREBUILT_VERSION = "v0.3.16"
#
#   LLAMACPP_PYTHON_VERSION — resolved at install time to the latest GitHub
#     release tag for abetlen/llama-cpp-python.  Compile routes clone this
#     version so users always get current architecture support without
#     needing to update the installer.  Falls back to a hardcoded minimum
#     if the GitHub API is unreachable.
LLAMACPP_PYTHON_VERSION = None          # resolved dynamically by get_latest_llamacpp_python_version()
LLAMACPP_PYTHON_VERSION_FALLBACK = "v0.3.20"  # used when GitHub API is unreachable
LLAMACPP_TARGET_VERSION = "b8882"
DOWNLOAD_RELEASE_TAG = "b8882" 
WIN_COMPILE_TEMP = Path("C:/temp_build")      # fixed Windows build folder (short path)
LINUX_COMPILE_TEMP = None                       # Linux keeps using project-local temp
_INSTALL_PROCESSES = set()
_DID_COMPILATION = False 
_PRE_EXISTING_PROCESSES = {}
_USER_BUILD_THREADS = None  # Explicitly chosen by user; None = use auto 85% rule
PYTHON_VERSION = sys.version_info
WINDOWS_VERSION = None  # Will detect Windows version
_CPU_FEATURES = None  # Will hold the detected features dict
_CPU_DETECTED_EARLY = False
OS_VERSION = None
VS_GENERATOR = None
DETECTED_PYTHON_INFO = {}

# Display/Browser variables
DX11_CAPABLE = None
DX_FEATURE_LEVEL = None
DX_FEATURE_NAME = None
# Standardised on Gradio 3.50.2 + Qt5 (PyQt5) across all platforms/OS versions.
# These are constants now — no dynamic selection logic required.
SELECTED_GRADIO = "3.50.2"
SELECTED_QTWEB  = "v5"

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
        "display": "Bge-Small-En v1.5 (Legacy/Fastest - 132MB)",
        "size_mb": 132
    },
    "2": {
        "name": "BAAI/bge-base-en-v1.5", 
        "display": "Bge-Base-En v1.5 (Recommended/Regular - 425MB)",
        "size_mb": 425
    },
    "3": {
        "name": "BAAI/bge-large-en-v1.5",
        "display": "Bge-Large-En v1.5 (Huge/Slow/Buggy - 1.35GB)", 
        "size_mb": 1350
    }
}

# Fixed English Male/Female voices for Coqui TTS (VCTK model)
# p225 = English Female, p226 = English Male
COQUI_ENGLISH_VOICE = {
    "id":      "p225,p226",   # female,male — stored together so the app can offer both
    "display": "English (Male/Female)",
    "accent":  "english",
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

def short_path(path, max_len=44):
    """Truncate path for display - installer standalone version"""
    path = str(path)
    if len(path) <= max_len:
        return path
    return "..." + path[-max_len:]

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
        # Primary path: Windows kernel32.IsProcessorFeaturePresent — pure stdlib
        # (ctypes), no external packages, works before the venv exists.
        # Relevant PF_* constants (winnt.h):
        #   13 = PF_SSE3_INSTRUCTIONS_AVAILABLE
        #   36 = PF_SSSE3_INSTRUCTIONS_AVAILABLE
        #   37 = PF_SSE4_1_INSTRUCTIONS_AVAILABLE
        #   38 = PF_SSE4_2_INSTRUCTIONS_AVAILABLE
        #   39 = PF_AVX_INSTRUCTIONS_AVAILABLE
        #   40 = PF_AVX2_INSTRUCTIONS_AVAILABLE
        #   41 = PF_AVX512F_INSTRUCTIONS_AVAILABLE
        # FMA and F16C have no IsProcessorFeaturePresent constant so we
        # leave those to the cpuinfo secondary path below.
        try:
            import ctypes
            _ipfp = ctypes.windll.kernel32.IsProcessorFeaturePresent
            _ipfp.restype = ctypes.c_bool
            _ipfp.argtypes = [ctypes.c_uint]
            features["SSE3"]   = bool(_ipfp(13))
            features["SSSE3"]  = bool(_ipfp(36))
            features["SSE4_1"] = bool(_ipfp(37))
            features["SSE4_2"] = bool(_ipfp(38))
            features["AVX"]    = bool(_ipfp(39))
            features["AVX2"]   = bool(_ipfp(40))
            features["AVX512"] = bool(_ipfp(41))
            success = True
        except Exception:
            features["SSE3"] = True   # safe minimum fallback
            success = True

        # Secondary path: try py-cpuinfo in the current process to fill in
        # FMA and F16C (not exposed by IsProcessorFeaturePresent).
        # Succeeds on a Check/Install where cpuinfo was installed in a prior
        # run; silently skipped otherwise — no error printed at menu time.
        if success:
            try:
                import cpuinfo as _cpuinfo
                _info  = _cpuinfo.get_cpu_info()
                _flags = [f.lower() for f in _info.get('flags', [])]
                features["FMA"]    = 'fma'  in _flags
                features["F16C"]   = 'f16c' in _flags
                # Cross-check SIMD flags from cpuinfo for extra accuracy
                features["AVX"]    = features["AVX"]    or ('avx'    in _flags)
                features["AVX2"]   = features["AVX2"]   or ('avx2'   in _flags)
                features["AVX512"] = features["AVX512"] or any('avx512' in f for f in _flags)
            except ImportError:
                pass   # cpuinfo not yet available — fine at menu time
    
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

def detect_browser_acceleration() -> tuple:
    """
    Silent GPU/DX detection with caching.
    No printing — safe to call multiple times.
    """
    global DX11_CAPABLE, DX_FEATURE_LEVEL, DX_FEATURE_NAME

    if DX11_CAPABLE is not None:
        return (DX11_CAPABLE, DX_FEATURE_LEVEL)

    # Non-Windows → assume OK for Qt6
    if PLATFORM != "windows":
        DX11_CAPABLE = True
        DX_FEATURE_LEVEL = 0xb000
        DX_FEATURE_NAME = "11.0"
        return (DX11_CAPABLE, DX_FEATURE_LEVEL)

    try:
        import ctypes

        d3d11 = ctypes.windll.LoadLibrary("d3d11.dll")

        feature_levels = (ctypes.c_uint * 4)(
            0xb100,  # 11.1
            0xb000,  # 11.0
            0xa100,  # 10.1
            0xa000   # 10.0
        )

        device = ctypes.c_void_p()
        fl_out = ctypes.c_uint()
        ctx = ctypes.c_void_p()

        hr = d3d11.D3D11CreateDevice(
            None,
            1,
            None,
            0,
            feature_levels,
            4,
            7,
            ctypes.byref(device),
            ctypes.byref(fl_out),
            ctypes.byref(ctx),
        )

        DX_FEATURE_LEVEL = fl_out.value

        DX_FEATURE_NAME = {
            0xb100: "11.1",
            0xb000: "11.0",
            0xa100: "10.1",
            0xa000: "10.0"
        }.get(fl_out.value, f"0x{fl_out.value:04x}")

        DX11_CAPABLE = (hr == 0 and fl_out.value >= 0xb000)

        return (DX11_CAPABLE, DX_FEATURE_LEVEL)

    except:
        DX11_CAPABLE = False
        DX_FEATURE_LEVEL = 0
        DX_FEATURE_NAME = "Unknown"
        return (False, 0)

def run_initial_detection():
    print_header("Installer")

    print("[DETECT] Testing GPU/D3D11 acceleration capability...")

    dx11, _ = detect_browser_acceleration()

    if DX_FEATURE_LEVEL == 0:
        print_status(
            "GPU acceleration: Not available",
            False
        )
        print()
        return

    if dx11:
        print_status(
            f"GPU acceleration: DirectX {DX_FEATURE_NAME}"
        )
    else:
        print_status(
            f"GPU acceleration: DirectX {DX_FEATURE_NAME}"
        )

    print()

# Set TEMP_DIR based on platform
if PLATFORM == "windows":
    TEMP_DIR = WIN_COMPILE_TEMP
else:
    TEMP_DIR = BASE_DIR / "data" / "temp"
    
# Backend definitions
if PLATFORM == "windows":
    BACKEND_OPTIONS = {
        "Download CPU Binary / Default CPU Wheel": {
            "url": None, "dest": None, "cli_path": None,
            "needs_python_bindings": True, "compile_binary": False,
            "compile_wheel": False, "vulkan_required": False, "build_flags": {}
        },
        "Download Vulkan Binary / Default CPU Wheel": {
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
        "Download CPU Binary / Default CPU Wheel": {
            "url": None, "dest": None, "cli_path": None,
            "needs_python_bindings": True, "compile_binary": False,
            "compile_wheel": False, "vulkan_required": False, "build_flags": {}
        },
        "Download Vulkan Binary / Default CPU Wheel": {
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
    "langchain-text-splitters>=0.3.0",  # configure.py imports directly; explicit dep of langchain
    "faiss-cpu>=1.8.0",
    "langchain>=0.3.18",            
    "pygments==2.17.2",
    "lxml==6.0.2",
    "lxml_html_clean==0.4.3",
    "tokenizers==0.22.2",
    "beautifulsoup4>=4.12.0",       # HTML parsing for deep research
    "aiohttp>=3.10.0",              # Async HTTP for parallel page fetches
    "matplotlib>=3.7.0",            # Required by Gradio 3.50.2 internally (gradio/utils.py)
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
    """
    Populate DETECTED_PYTHON_INFO.
    Qt5 + Gradio 3.50.2 are used on all platforms — no dynamic selection needed.
    """
    global DETECTED_PYTHON_INFO, SELECTED_GRADIO, SELECTED_QTWEB
    import platform as plat
    DETECTED_PYTHON_INFO = detect_all_pythons()
    # Constants — already set at module level; reassign for clarity
    SELECTED_GRADIO = "3.50.2"
    SELECTED_QTWEB  = "v5"

def get_dynamic_requirements() -> list:
    """
    Build requirements list.
    Standardised on Gradio 3.50.2 + Qt5 (PyQt5) for all platforms.

    Version constraint triangle — all three must be compatible:
    - starlette==0.27.0    : Gradio 3.x uses TemplateResponse positional args;
                             starlette 0.28.0 changed that signature → must stay 0.27.0.
    - fastapi==0.103.2     : Last fastapi release whose routing.py does NOT import
                             starlette._exception_handler (added in starlette 0.31.0).
                             fastapi 0.104.0+ requires starlette>=0.31.0, which conflicts
                             with our starlette pin and causes ModuleNotFoundError at launch.
    - jinja2==3.1.3        : Jinja2 3.1.4+ changed LRU cache key construction,
                             breaking Gradio 3.x template rendering.
    - gradio_client        : NOT listed here — gradio 3.50.2 pins gradio_client~=0.6.1
                             itself; adding 0.17.0 (a gradio 4.x/5.x companion) causes
                             a dependency conflict error during install.
    """
    requirements = BASE_REQ.copy()
    requirements.append("gradio==3.50.2")
    requirements.append("fastapi==0.103.2")      # Must match starlette 0.27.0; see docstring
    requirements.append("websockets==10.4")      # gradio_client may pull a newer version; pin to 10.4
    requirements.append("jinja2==3.1.3")
    requirements.append("starlette==0.27.0")
    requirements.append("pydantic==1.10.21")     # Gradio 3.x requires Pydantic v1 (FieldInfo.in_ etc.);
                                                 # pydantic v2 removed these APIs → fatal crash at launch.
                                                 # 1.10.21 is the final v1 release; supports Python 3.9-3.13.
    return requirements
    
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
    print_header("Backend/Wheel Menu")

    print("System Detections...")

    if PLATFORM == "windows":
        win_ver = WINDOWS_VERSION or "unknown"
        print(f"    Platform: Windows {win_ver}")
        print(f"    DX Display: {DX_FEATURE_NAME if DX_FEATURE_NAME else 'Unknown'}")
    else:
        ubuntu_ver = OS_VERSION or "unknown"
        print(f"    Platform: Ubuntu {ubuntu_ver}")

    cpu_features = detect_cpu_features()
    features_list = [feat for feat, supported in cpu_features.items() if supported]

    if features_list:
        print(f"    CPU Features: {', '.join(features_list)}")
    else:
        print("    CPU Features: Baseline")

    vulkan_present = is_vulkan_installed()
    print(f"    Vulkan Present: {'Yes' if vulkan_present else 'No'}")

    print()

    for i, option in enumerate(options, 1):
        print(f"    {i}) {option}")

    while True:
        choice = input(
            f"Selection; Menu Options 1-{len(options)}, Abandon Install = A: "
        ).strip().upper()

        if choice == "A":
            sys.exit(0)

        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]

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

def create_files_and_directories(backend: str) -> None:
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

    # Ensure scripts/__init__.py exists for proper Python package recognition
    scripts_dir = BASE_DIR / "scripts"
    scripts_init = scripts_dir / "__init__.py"
    if not scripts_init.exists():
        try:
            scripts_init.touch()
            print_status("Created scripts/__init__.py")
        except PermissionError:
            print_status("Permission denied for scripts/__init__.py", False)
            sys.exit(1)
    else:
        print_status("scripts/__init__.py already exists")

    if PLATFORM == "windows" and backend_requires_compilation(backend):
        TEMP_DIR = WIN_COMPILE_TEMP
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        print_status(f"Using build temp path: {TEMP_DIR}")
    else:
        TEMP_DIR = BASE_DIR / "data" / "temp"
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        print_status(f"Using project temp path: {TEMP_DIR}")

def get_optimal_build_threads() -> int:
    """Return the number of threads to use for compilation.

    If the user was prompted (dual-thread CPU on a compile route) and made an
    explicit choice, that choice is returned.  Otherwise the 85% auto-rule
    applies, which leaves a small margin so the machine stays responsive.
    """
    global _USER_BUILD_THREADS
    if _USER_BUILD_THREADS is not None:
        return _USER_BUILD_THREADS
    import multiprocessing
    try:
        total_threads = multiprocessing.cpu_count()
    except:
        total_threads = 4
    return max(1, int(total_threads * 0.85))

def prompt_build_threads() -> None:
    """On a dual-thread CPU, ask the user how many threads to use for compilation.

    A machine with only 2 logical threads may become unusably slow if both are
    saturated by the build.  For any other core count the 85% auto-rule already
    leaves headroom, so no prompt is needed.
    """
    global _USER_BUILD_THREADS
    import multiprocessing
    try:
        total = multiprocessing.cpu_count()
    except:
        total = 4

    if total != 2:
        # Auto-select for all other counts (85% rule already applied inside
        # get_optimal_build_threads, so just leave _USER_BUILD_THREADS as None).
        return

    print()
    print("Compile Thread Selection...")
    print(f"    Your CPU has 2 logical threads.")
    print(f"    1) Use 1 thread  (slower compile, PC stays usable during the build)")
    print(f"    2) Use 2 threads (faster compile, PC may feel sluggish during the build)")
    while True:
        choice = input("    Selection (1/2): ").strip()
        if choice == "1":
            _USER_BUILD_THREADS = 1
            print(f"    Using 1 build thread.")
            break
        elif choice == "2":
            _USER_BUILD_THREADS = 2
            print(f"    Using 2 build threads.")
            break
        else:
            print("    Please enter 1 or 2.")

def _get_prebuilt_wheel_urls() -> list:
    """Return an ordered list of URLs/install-specs to try for llama-cpp-python.

    Strategy (tried in order):
      1. eswarthammana mirror  — community CPU wheel builder; has cp38-cp313 for
                                  all versions it covers (currently up to v0.3.16).
      2. abetlen GitHub release — upstream; only CUDA/Metal builds, so this 404s
                                  for CPU-only wheels but is tried cheaply anyway.
      3. abetlen CPU index      — pip extra-index-url install spec; the index at
                                  abetlen.github.io/llama-cpp-python/whl/cpu
                                  sometimes carries newer CPU wheels.
      4. PyPI --prefer-binary   — plain `pip install llama-cpp-python==VERSION`
                                  with --prefer-binary: installs a binary wheel
                                  from any index if one exists; does NOT fall
                                  through to source compilation (fail fast).

    Returns a list of dicts:
        {"type": "url",     "value": "<direct whl url>"}    — install from URL
        {"type": "index",   "value": "<version spec>",
         "extra_index": "<url>"}                            — pip install + index
        {"type": "pypi",    "value": "<version spec>"}      — pip install PyPI
    """
    import platform as _plat
    version = LLAMACPP_PYTHON_PREBUILT_VERSION.lstrip("v")   # "v0.3.16" → "0.3.16"
    py_tag  = _PY_TAG                                # e.g. "cp312"

    if PLATFORM == "windows":
        platform_tag = "win_amd64"
    else:
        machine = _plat.machine().lower()
        if machine in ("x86_64", "amd64"):
            platform_tag = "linux_x86_64"
        elif machine in ("aarch64", "arm64"):
            platform_tag = "linux_aarch64"
        else:
            platform_tag = None

    sources = []
    filename = (f"llama_cpp_python-{version}-{py_tag}-{py_tag}-{platform_tag}.whl"
                if platform_tag else None)

    # 1 — eswarthammana community mirror (CPU prebuilts; may not have latest version)
    if filename:
        sources.append({
            "type": "url",
            "label": f"eswarthammana mirror (v{version})",
            "value": (f"https://github.com/eswarthammana/llama-cpp-wheels"
                      f"/releases/download/v{version}/{filename}")
        })

    # 2 — abetlen GitHub release (covers CUDA/Metal; 404s for CPU-only, cheap to try)
    if filename:
        sources.append({
            "type": "url",
            "label": f"abetlen GitHub releases (v{version})",
            "value": (f"https://github.com/abetlen/llama-cpp-python"
                      f"/releases/download/v{version}/{filename}")
        })

    # 3 — abetlen CPU wheel index (maintained alongside the project)
    sources.append({
        "type": "index",
        "label": f"abetlen CPU index (v{version})",
        "value": f"llama-cpp-python=={version}",
        "extra_index": "https://abetlen.github.io/llama-cpp-python/whl/cpu"
    })

    # 4 — PyPI --prefer-binary (any binary found on any index; no source compile)
    sources.append({
        "type": "pypi",
        "label": f"PyPI binary (v{version}, --prefer-binary)",
        "value": f"llama-cpp-python=={version}",
    })

    return sources
    return sources


def get_latest_llamacpp_python_version() -> str:
    """Query GitHub API for the latest abetlen/llama-cpp-python release tag.

    Returns the clean base tag string (e.g. "v0.3.20") on success,
    stripping any backend-specific suffix that abetlen appends per CUDA/Metal
    release variant (e.g. "v0.3.20-cu123" → "v0.3.20").

    Falls back silently to LLAMACPP_PYTHON_VERSION_FALLBACK if the API is
    unreachable or returns unexpected data.  No console output — callers
    display the resolved version in context (e.g. in the menu label).

    Uses only stdlib (urllib) so it works before the venv is populated.
    """
    global LLAMACPP_PYTHON_VERSION

    # Already resolved — return cached value
    if LLAMACPP_PYTHON_VERSION is not None:
        return LLAMACPP_PYTHON_VERSION

    api_url = "https://api.github.com/repos/abetlen/llama-cpp-python/releases/latest"
    try:
        import urllib.request, urllib.error, json as _json, re as _re
        req = urllib.request.Request(
            api_url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "Chat-Gradio-Gguf-Installer/1.0",
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        tag = data.get("tag_name", "").strip()
        if tag and tag.startswith("v"):
            # abetlen publishes multiple GitHub release entries per version —
            # one per CUDA variant (v0.3.20-cu121, -cu122, -cu123 …) plus
            # -metal, -rocm, -sycl.  releases/latest returns whichever was
            # pushed last (almost always a CUDA variant).  Strip the suffix:
            # the source code at every v0.3.20-* tag is identical, and we
            # compile ourselves with our own flags (Vulkan/CPU/AVX2/FMA etc.)
            # so the backend suffix is irrelevant to us.
            base_tag = _re.sub(
                r'-(cu[0-9]+|metal|rocm[0-9.]*|sycl)$', '',
                tag, flags=_re.IGNORECASE
            )
            LLAMACPP_PYTHON_VERSION = base_tag
            return base_tag
    except Exception:
        pass  # Network/API failure — fall through to fallback below

    LLAMACPP_PYTHON_VERSION = LLAMACPP_PYTHON_VERSION_FALLBACK
    return LLAMACPP_PYTHON_VERSION_FALLBACK
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
    
    # Minimal safe flags.
    # Note: LLAMA_CURL was renamed to GGML_CURL in llama.cpp ~b7500+.
    # Set both so the build works whether the bundled llama.cpp is old or new.
    build_flags["GGML_CURL"] = "OFF"
    build_flags["LLAMA_CURL"] = "OFF"   # legacy alias — safe to set on newer builds
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
        # Forward the detected VS generator so scikit-build / CMake picks the
        # correct toolchain rather than choosing arbitrarily when multiple VS
        # versions are installed.  CMAKE_GENERATOR + CMAKE_GENERATOR_PLATFORM
        # are the correct env vars; embedding "-G ..." inside CMAKE_ARGS is
        # fragile because the string is later split on spaces.
        if VS_GENERATOR:
            env["CMAKE_GENERATOR"] = VS_GENERATOR
            env["CMAKE_GENERATOR_PLATFORM"] = "x64"
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
        
        # Resolve version at compile time — may already be cached from the menu
        compile_version = get_latest_llamacpp_python_version()
        print_status(f"Cloning llama-cpp-python {compile_version} (Python wrapper — bundles llama.cpp internally)...")
        print(f"  Note: {compile_version} is the Python wrapper version, not a llama.cpp build tag.")
        print(f"  This clones the wrapper + its llama.cpp submodule (~200-400 MB). Please wait...")

        # Clone specific release version with retry for network issues.
        # Uses Popen + --progress so git's progress lines stream live rather
        # than the process appearing frozen for several minutes.
        max_retries = 5
        retry_delay = 10
        clone_success = False
        last_clone_error = ""

        for attempt in range(max_retries):
            if repo_dir.exists():
                shutil.rmtree(repo_dir, ignore_errors=True)

            clone_proc = subprocess.Popen(
                ["git", "clone", "--progress", "--depth", "1",
                 "--branch", compile_version,
                 "--recurse-submodules", "--shallow-submodules",
                 LLAMACPP_PYTHON_GIT_REPO, str(repo_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env
            )

            clone_output_lines = []
            for raw in clone_proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                clone_output_lines.append(line)
                # git --progress writes to stderr (merged to stdout here).
                # Forward lines that describe what is actually happening.
                if any(kw in line.lower() for kw in
                       ["cloning", "receiving", "resolving", "counting",
                        "compressing", "remote:", "submodule", "error", "fatal"]):
                    print(f"  {line}", flush=True)

            clone_proc.wait()

            if clone_proc.returncode == 0:
                clone_success = True
                if attempt > 0:
                    print_status(f"Clone succeeded on attempt {attempt + 1}")
                break
            else:
                last_clone_error = "\n".join(clone_output_lines[-10:])
                reason_short = clone_output_lines[-1][:120] if clone_output_lines else "no output from git"
                if attempt < max_retries - 1:
                    print(f"  Clone failed (attempt {attempt + 1}/{max_retries}): {reason_short}")
                    print(f"  Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2

        if not clone_success:
            print_status(f"Git clone failed after {max_retries} attempts", False)
            print(f"  Last error:\n{last_clone_error[-500:]}")
            return False

        print_status("Clone complete — starting build (this can take 10-30 minutes on a slow CPU)...")
        print(f"  Threads: {build_threads}  |  Repo: {repo_dir}")
        print(f"  Phases: CMake configure → compile → link → pip install")
        print(f"  Output is shown for significant events; silence = normal during compile.")


        # Capture full output for error diagnosis
        full_output = []
        last_print_time = [time.time()]
        HEARTBEAT_INTERVAL = 30   # print a status line if nothing else printed for this long

        # Phase detection — derive a human label from cmake/compiler output
        _PHASE_KEYWORDS = {
            "cmake": "CMake configuring...",
            "-- checking": "CMake configuring...",
            "-- found": "CMake configuring...",
            "-- configuring": "CMake configuring...",
            "compiling": "Compiling...",
            "building cxx": "Compiling...",
            "building c ": "Compiling...",
            "linking": "Linking...",
            "running setup": "Running setup...",
            "installing collected": "Installing package...",
            "successfully installed": "Done.",
        }
        current_phase = [None]

        def _phase_for(line: str) -> str | None:
            ll = line.lower()
            for kw, label in _PHASE_KEYWORDS.items():
                if kw in ll:
                    return label
            return None

        process = subprocess.Popen(
            [pip_exe, "install", str(repo_dir), "-v", "--no-cache-dir"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env
        )
        
        track_process(process.pid)

        build_start = time.time()
        
        for line in process.stdout:
            line_s = line.rstrip()
            if not line_s:
                continue
            full_output.append(line_s)
            now = time.time()

            phase = _phase_for(line_s)
            is_error = any(x in line_s.lower() for x in ["error", "fatal", "failed"])

            if is_error:
                # Always surface errors immediately, on their own line
                print(f"\n  [!] {line_s[:120]}", flush=True)
                last_print_time[0] = now
                current_phase[0] = None

            elif phase:
                if phase != current_phase[0]:
                    # Phase transition — print it
                    elapsed = int(now - build_start)
                    print(f"\n  [{elapsed:>4}s] {phase}", end="", flush=True)
                    current_phase[0] = phase
                    last_print_time[0] = now
                # else: same phase, just a dot heartbeat below if needed

            # Heartbeat: if nothing printed for HEARTBEAT_INTERVAL seconds, show elapsed
            if now - last_print_time[0] >= HEARTBEAT_INTERVAL:
                elapsed = int(now - build_start)
                print(f"\n  [{elapsed:>4}s] Still building... (CPU active, please wait)", end="", flush=True)
                last_print_time[0] = now
        
        # Ensure we're on a fresh line after all the in-place updates
        print(flush=True)

        process.wait()   # no timeout — the stdout loop above already drained the process
        
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
    """Create persistent.json configuration file"""
    config_path = BASE_DIR / "data" / "persistent.json"
    
    vulkan_enabled = "vulkan" in backend.lower()
    layer_mode = "VRAM_SRAM" if vulkan_enabled else "SRAM_ONLY"
    default_vram = 8192 if vulkan_enabled else 0
    
    optimal_threads = get_optimal_build_threads()
    
    if PLATFORM == "windows":
        default_sound_device = "Default Sound Device"
    else:
        default_sound_device = "default"
    
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
            "sound_output_device": default_sound_device,
            "sound_sample_rate": 44100,
            "tts_enabled": False,
            "tts_voice": None,
            "tts_voice_name": None,
            "max_tts_length": 4500,
        }
    }
    
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding='utf-8') as f:
            json.dump(config, f, indent=4)
        print_status("Configuration file created")
    except Exception as e:
        print_status(f"Failed to create config: {e}", False)
        sys.exit(1)  # Fail hard if config can't be created

def create_system_ini(platform: str, os_version: str, python_version: str,
                      backend_type: str, embedding_model: str,
                      windows_version: str = None, vulkan_available: bool = False,
                      llama_cli_path: str = None, llama_bin_path: str = None,
                      tts_engine: str = "builtin", coqui_voice_id: str = None,
                      coqui_voice_accent: str = None,
                      browser_acceleration: bool = True,
                      dx_feature_level: int = 0):
    """Create constants.ini with platform, version, TTS, and compatibility information.
    Qt5 (PyQt5) and Gradio 3.50.2 are written unconditionally — no version branching.
    browser_acceleration and dx_feature_level are recorded for diagnostic reference only.
    """
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
            f.write(f"browser_acceleration = {str(browser_acceleration).lower()}\n")
            f.write(f"qt_version = 5\n")
            f.write(f"dx_feature_level = {dx_feature_level}\n")
            f.write(f"gradio_version = 3.50.2\n")
            
            if llama_cli_path:
                f.write(f"llama_cli_path = {llama_cli_path}\n")
            if llama_bin_path:
                f.write(f"llama_bin_path = {llama_bin_path}\n")
            if platform == "windows" and windows_version:
                f.write(f"windows_version = {windows_version}\n")
            
            # TTS Configuration Section
            f.write("\n[tts]\n")
            f.write(f"tts_type = {tts_engine}\n")
            if tts_engine == "coqui" and coqui_voice_id:
                f.write(f"coqui_voice_id = {coqui_voice_id}\n")
                f.write(f"coqui_voice_accent = {coqui_voice_accent or 'english'}\n")
                f.write(f"coqui_model = tts_models/en/vctk/vits\n")
            
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
    
    # Qt5 (PyQt5) is used on all supported Ubuntu versions (22-25).
    # qt5-default was dropped in Ubuntu 22+; use the explicit package names instead.
    linux_ver = detect_linux_version()

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

    # Qt5 system packages (Ubuntu 22-25)
    qt_packages = [
        "libxcb-xinerama0",
        "libxkbcommon0",
        "libegl1",
        "libgl1",
        "qtbase5-dev",
        "qtchooser",
        "qt5-qmake",
        "qtbase5-dev-tools",
        "libqt5webengine5",
        "libqt5webenginewidgets5",
    ]
    qt_fallback = ["qtwebengine5-dev"]
    
    info = BACKEND_OPTIONS[backend]
    vulkan_packages = []
    if info.get("build_flags", {}).get("GGML_VULKAN"):
        vulkan_packages = [
            "vulkan-tools", "libvulkan-dev", "mesa-utils",
            "glslang-tools", "spirv-tools"
        ]
    elif backend in ["Download Vulkan Binary / Default CPU Wheel"]:
        vulkan_packages = ["vulkan-tools", "libvulkan1"]
    
    try:
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y"] + list(set(base_packages)), check=True)
        print_status("Base dependencies installed")
        
        # Install Qt dependencies
        print_status("Installing Qt5 dependencies...")
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

# This version uses sequential installation of each package while explicitly providing the CPU index URL for all three. This ensures that pip always uses the same index and does not attempt to replace the CPU‑only torch wheel with a default PyPI version.
def install_embedding_backend() -> bool:
    """
    Install torch + sentence-transformers for all platforms.
    Torch Version Matrix:
    - Python 3.9-3.11: torch==2.2.2+cpu (stable, wide compatibility)
    - Python 3.12+: torch>=2.4.0+cpu (required for 3.12+ support)
    """
    print_status("Installing embedding backend (torch + sentence-transformers)... ")

    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("python.exe" if PLATFORM == "windows" else "python"))
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                 ("pip.exe" if PLATFORM == "windows" else "pip"))

    py_minor = sys.version_info.minor

    # Determine version strings
    if py_minor <= 11:
        torch_req = "torch==2.2.2+cpu"
        transformers_version = "transformers==4.41.2"
        sentence_transformers_version = "sentence-transformers==3.0.1"
        # Pre-install compatible dependencies for torch 2.2.2
        pip_install_with_retry(pip_exe, "setuptools>=65.0.0,<70.0.0", max_retries=3)
        pip_install_with_retry(pip_exe, "networkx>=2.6,<3.0", max_retries=3)
        pip_install_with_retry(pip_exe, "sympy>=1.12,<1.13", max_retries=3)
    else:
        torch_req = "torch>=2.4.0"
        transformers_version = "transformers>=4.42.0"
        sentence_transformers_version = "sentence-transformers>=3.0.0"
        # Pre-install compatible dependencies for torch 2.4+
        pip_install_with_retry(pip_exe, "setuptools>=68.0.0,<71.0.0", max_retries=3)
        pip_install_with_retry(pip_exe, "networkx>=3.0,<3.3", max_retries=3)
        pip_install_with_retry(pip_exe, "sympy>=1.12,<1.14", max_retries=3)
        pip_install_with_retry(pip_exe, "mpmath>=1.3.0,<1.4.0", max_retries=3)

    # 1. Install PyTorch using the CPU index
    cpu_index = "https://download.pytorch.org/whl/cpu"
    print_status(f"Installing PyTorch (CPU) - {torch_req}...")
    if not pip_install_with_retry(pip_exe, torch_req,
                                  ["--index-url", cpu_index, "--upgrade-strategy", "only-if-needed"],
                                  max_retries=10, initial_delay=5.0):
        print_status("PyTorch installation failed", False)
        return False
    print_status("PyTorch (CPU) installed")

    # 2. Install transformers from default PyPI
    print_status(f"Installing {transformers_version}...")
    if not pip_install_with_retry(pip_exe, transformers_version,
                                  ["--upgrade-strategy", "only-if-needed"],
                                  max_retries=10, initial_delay=5.0):
        print_status("transformers installation failed", False)
        return False
    print_status("transformers installed")

    # 3. Install sentence-transformers from default PyPI
    print_status(f"Installing {sentence_transformers_version}...")
    if not pip_install_with_retry(pip_exe, sentence_transformers_version,
                                  ["--upgrade-strategy", "only-if-needed"],
                                  max_retries=10, initial_delay=5.0):
        print_status("sentence-transformers installation failed", False)
        return False
    print_status("sentence-transformers installed")

    # Verify installation
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

# select_tts_options() removed — TTS engine and voice are now determined
# automatically by the install-size choice (a/b) inside
# select_backend_and_install_size().  Coqui is selected for option b on
# compatible OS versions; built-in pyttsx3/espeak-ng is used otherwise.

def install_coqui_tts():
    """Install Coqui TTS (Idiap fork) with codec support and download VCTK model.
    
    Fails hard on any error - no fallback. Requires torchcodec for audio IO.
    """
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("pip.exe" if PLATFORM == "windows" else "pip"))
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python"))
    
    if PLATFORM == "windows":
        if not install_espeak_ng_windows():
            print_status("CRITICAL: espeak-ng installation failed", False)
            sys.exit(1)
    
    print_status("Installing Coqui TTS with codec support...")
    
    try:
        # Install torchaudio FIRST with CPU index to match existing CPU-only torch
        # Without --index-url, pip pulls the CUDA build which requires libtorch_cuda.so
        print_status("Installing torchaudio (CPU-only to match torch)...")
        if not pip_install_with_retry(pip_exe, "torchaudio",
                                       ["--index-url", "https://download.pytorch.org/whl/cpu"],
                                       max_retries=10, initial_delay=5.0):
            print_status("torchaudio installation failed", False)
            sys.exit(1)
        print_status("torchaudio (CPU) installed")
        
        # Now install coqui-tts[codec] - torchaudio is already satisfied so pip won't replace it
        result = subprocess.run(
            [pip_exe, "install", "coqui-tts[codec]"],
            capture_output=True, text=True, timeout=600
        )
        
        if result.returncode != 0:
            error_detail = result.stderr[-800:] if len(result.stderr) > 800 else result.stderr
            print_status(f"Coqui TTS pip install failed: {error_detail}", False)
            sys.exit(1)
        
        print_status("Coqui TTS package installed")
        
        tts_model_dir = BASE_DIR / "data" / "tts_models"
        tts_model_dir.mkdir(parents=True, exist_ok=True)
        
        print_status("Downloading Coqui VCTK voice model (~1.4GB)...")
        
        tts_model_dir_safe = str(tts_model_dir).replace("\\", "/")
        temp_wav_safe = str(TEMP_DIR / "tts_test.wav").replace("\\", "/")
        
        if PLATFORM == "windows":
            espeak_local_path = str(BASE_DIR / "data" / "espeak-ng").replace("\\", "/")
            
            # CRITICAL FIX: Add espeak-ng directory to PATH so Windows can find DLL dependencies
            # and set PHONEMIZER_ESPEAK_PATH to directory (not exe)
            download_script = f'''
import os
import sys

local_espeak = r"{espeak_local_path}"

# CRITICAL: Add espeak-ng directory to PATH FIRST
# This allows Windows to find DLL dependencies when ctypes loads libespeak-ng.dll
current_path = os.environ.get("PATH", "")
if local_espeak not in current_path:
    os.environ["PATH"] = local_espeak + os.pathsep + current_path

# Set phonemizer environment variables
# PHONEMIZER_ESPEAK_LIBRARY = path to the DLL file
# PHONEMIZER_ESPEAK_PATH = path to the DIRECTORY (not the exe!)
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(local_espeak, "libespeak-ng.dll")
os.environ["PHONEMIZER_ESPEAK_PATH"] = local_espeak
os.environ["ESPEAK_DATA_PATH"] = os.path.join(local_espeak, "espeak-ng-data")

# Verify files exist
dll_path = os.environ["PHONEMIZER_ESPEAK_LIBRARY"]
exe_path = os.path.join(local_espeak, "espeak-ng.exe")
data_path = os.environ["ESPEAK_DATA_PATH"]

if not os.path.exists(dll_path):
    print("[FATAL] espeak-ng DLL not found at " + dll_path)
    sys.exit(1)
if not os.path.exists(exe_path):
    print("[FATAL] espeak-ng executable not found at " + exe_path)
    sys.exit(1)
if not os.path.isdir(data_path):
    print("[FATAL] espeak-ng data directory not found at " + data_path)
    sys.exit(1)

print("[COQUI] espeak-ng directory added to PATH")
print("[COQUI] espeak-ng DLL: " + dll_path)
print("[COQUI] espeak-ng data: " + data_path)
print("[COQUI] espeak-ng verified")

os.environ["TTS_HOME"] = "{tts_model_dir_safe}"

from TTS.api import TTS
print("[COQUI] Loading model...")
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True)
print("[COQUI] Testing synthesis...")
tts.tts_to_file(text="Installation test successful.", file_path="{temp_wav_safe}", speaker="p243")

if not os.path.exists("{temp_wav_safe}"):
    print("[FATAL] Test file not created")
    sys.exit(1)
print("[COQUI] Model test passed")
'''
        else:
            # Linux: espeak-ng is installed system-wide via apt (libespeak-ng1, espeak-ng)
            # When running as sudo, we must explicitly point phonemizer to the system libraries
            # because the root env may not have the same library search paths as the user
            download_script = f'''
import os
import sys
import subprocess

# Locate system espeak-ng for phonemizer (critical when running as sudo/root)
espeak_lib = None
espeak_exe = None

# Find libespeak-ng.so from ldconfig (works even under sudo)
try:
    result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "libespeak-ng.so.1" in line and "=>" in line:
            espeak_lib = line.split("=>")[-1].strip()
            break
except Exception:
    pass

# Fallback: check common library paths directly
if not espeak_lib:
    for candidate in [
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
        "/usr/lib/libespeak-ng.so.1",
        "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
        "/usr/local/lib/libespeak-ng.so.1",
    ]:
        if os.path.exists(candidate):
            espeak_lib = candidate
            break

# Find espeak-ng executable
for candidate in ["/usr/bin/espeak-ng", "/usr/local/bin/espeak-ng"]:
    if os.path.exists(candidate):
        espeak_exe = candidate
        break

if not espeak_lib:
    print("[FATAL] libespeak-ng.so.1 not found on system")
    print("[FATAL] Install with: sudo apt install libespeak-ng1 espeak-ng")
    sys.exit(1)

if not espeak_exe:
    print("[FATAL] espeak-ng executable not found on system")
    sys.exit(1)

# Determine espeak-ng-data path from the executable location
espeak_data = None
for candidate in ["/usr/lib/x86_64-linux-gnu/espeak-ng-data", "/usr/share/espeak-ng-data", "/usr/local/lib/espeak-ng-data"]:
    if os.path.isdir(candidate):
        espeak_data = candidate
        break

os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = espeak_lib
os.environ["PHONEMIZER_ESPEAK_PATH"] = espeak_exe
if espeak_data:
    os.environ["ESPEAK_DATA_PATH"] = espeak_data

print(f"[COQUI] espeak-ng lib: {{espeak_lib}}")
print(f"[COQUI] espeak-ng exe: {{espeak_exe}}")
print(f"[COQUI] espeak-ng data: {{espeak_data or 'auto'}}")

os.environ["TTS_HOME"] = "{tts_model_dir_safe}"

from TTS.api import TTS
print("[COQUI] Loading model...")
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=True)
print("[COQUI] Testing synthesis...")
tts.tts_to_file(text="Installation test successful.", file_path="{temp_wav_safe}", speaker="p243")
if not os.path.exists("{temp_wav_safe}"):
    print("[FATAL] Test file not created")
    sys.exit(1)
print("[COQUI] Test passed")
'''
        
        temp_script = TEMP_DIR / "download_tts_model.py"
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(download_script)
        
        # Run the download script without capture_output to see progress
        result = subprocess.run([python_exe, str(temp_script)], timeout=1800)
        temp_script.unlink(missing_ok=True)
        
        test_wav = TEMP_DIR / "tts_test.wav"
        if test_wav.exists():
            test_wav.unlink(missing_ok=True)
        
        if result.returncode != 0:
            print_status("Coqui model verification failed (see errors above)", False)
            sys.exit(1)
        
        print_status("Coqui TTS installed and verified")
        return True
        
    except subprocess.TimeoutExpired:
        print_status("Coqui TTS installation timed out", False)
        sys.exit(1)
    except Exception as e:
        print_status(f"Coqui TTS installation failed: {e}", False)
        sys.exit(1)
        
def install_qt_webengine() -> bool:
    """
    Install PyQt5 + PyQtWebEngine for the custom browser window.
    Qt5 is used on all supported platforms (Windows 7-11, Ubuntu 22-25).
    """
    print_status("Installing Qt5 WebEngine for custom browser...")
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                 ("pip.exe" if PLATFORM == "windows" else "pip"))

    try:
        if not pip_install_with_retry(pip_exe, "PyQt5>=5.15.0,<5.16.0", max_retries=3, initial_delay=5.0):
            print_status("PyQt5 installation failed - will use system browser", False)
            return False
        if not pip_install_with_retry(pip_exe, "PyQtWebEngine>=5.15.0,<5.16.0", max_retries=3, initial_delay=5.0):
            print_status("PyQtWebEngine installation failed - will use system browser", False)
            return False

        print_status("Qt5 WebEngine installed successfully")
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
                           max_retries: int = 10, initial_delay: float = 5.0,
                           force_reinstall: bool = False, no_deps: bool = False) -> bool:
    """Install a pip package with retry logic and exponential backoff.

    Uses an *activity-based* timeout rather than a hard wall-clock timeout:
    the install is only interrupted when no output has been received for
    INACTIVITY_TIMEOUT seconds.  A large wheel that is actively downloading
    will therefore never be killed mid-transfer just because it takes a long
    time — the timeout only fires when the connection has genuinely gone silent.

    Progress lines from pip (Downloading …, Installing …, error …) are
    forwarded to stdout in real time so the user can see what is happening.
    Before every retry a plain-English reason for the restart is printed.

    Args:
        force_reinstall: If True, add --force-reinstall flag to override existing installs
        no_deps: If True, add --no-deps flag to skip dependency resolution (use with caution)
    """
    # Seconds of silence before we consider the process stalled.
    INACTIVITY_TIMEOUT = 300

    # pip output keywords worth surfacing to the user.
    _PROGRESS_KEYWORDS = (
        "downloading", "installing", "collected",
        "building", "error", "warning", "failed", "%",
    )

    # Non-fatal pip resolver warning to suppress (cosmetic only, not a failure)
    _SUPPRESS_WARNINGS = (
        "pip's dependency resolver does not currently take into account",
    )

    if extra_args is None:
        extra_args = []

    pkg_name = package.split(">=")[0].split("==")[0].split("[")[0]
    delay = initial_delay

    # Build install flags based on parameters
    install_flags = []
    if force_reinstall:
        install_flags.append("--force-reinstall")
    if no_deps:
        install_flags.append("--no-deps")

    for attempt in range(max_retries):
        cmd = [pip_exe, "install"] + install_flags + [package] + extra_args
        all_output: list[str] = []
        last_activity = [time.time()]   # mutable so the reader thread can update it
        reader_done = [False]
        stall_reason: list[str] = [None]

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            # ── background thread: read pip output line-by-line ──────────────
            def _read_output():
                try:
                    for raw_line in proc.stdout:
                        line = raw_line.rstrip()
                        if not line:
                            continue
                        # Suppress known non-fatal resolver warnings (cosmetic only)
                        if any(kw in line.lower() for kw in _SUPPRESS_WARNINGS):
                            continue
                        last_activity[0] = time.time()
                        all_output.append(line)
                        if any(kw in line.lower() for kw in _PROGRESS_KEYWORDS):
                            print(f"    {line}", flush=True)
                finally:
                    reader_done[0] = True

            reader = threading.Thread(target=_read_output, daemon=True)
            reader.start()

            # ── main thread: watch for inactivity ────────────────────────────
            while not reader_done[0]:
                time.sleep(2)
                idle = time.time() - last_activity[0]
                if idle >= INACTIVITY_TIMEOUT:
                    stall_reason[0] = (
                        f"No output for {idle:.0f}s — connection stalled or server unresponsive"
                    )
                    proc.kill()
                    break

            reader.join(timeout=5)
            proc.wait()

            combined = "\n".join(all_output).lower()

            if proc.returncode == 0 or "already satisfied" in combined:
                return True

            # ── build a human-readable reason before retrying ────────────────
            if stall_reason[0]:
                reason = stall_reason[0]
            else:
                # Surface the last error line from pip if there is one.
                error_lines = [l for l in all_output if "error" in l.lower()]
                if error_lines:
                    reason = f"pip error — {error_lines[-1][:120]}"
                else:
                    reason = f"pip exited with code {proc.returncode} (no explicit error message)"

            if attempt < max_retries - 1:
                print(f"    Reason: {reason}")
                print(f"    Retry {attempt + 1}/{max_retries} for {pkg_name} in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 300)   # cap at 5 minutes

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Unexpected error: {e}")
                print(f"    Retry {attempt + 1}/{max_retries} for {pkg_name} in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 300)

    return False

def install_python_deps(backend: str) -> bool:
    """Install Python dependencies with dynamic version selection and conflict handling."""
    print_status("Installing Python dependencies...")
    try:
        python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                        ("python.exe" if PLATFORM == "windows" else "python"))
        pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("pip.exe" if PLATFORM == "windows" else "pip"))
        # Get dynamic requirements
        all_requirements = get_dynamic_requirements()
        
        # Separate critical pinned packages that must override any transitive dependencies.
        # IMPORTANT: gradio and gradio_client must NOT be in this set — installing them
        # with --no-deps would skip their own dependencies (matplotlib, pandas, pillow,
        # fastapi, uvicorn, pydub, etc.) and cause ModuleNotFoundError at runtime.
        # They are installed normally in Phase 1 so pip resolves their full dep tree.
        # Only packages that need to *override* what gradio's deps pull in go here.
        CRITICAL_PINNED = {
            "fastapi==0.103.2",       # gradio's resolver may pull 0.104.0+ which requires
                                      # starlette._exception_handler (added in starlette 0.31.0),
                                      # conflicting with our starlette==0.27.0 pin.
            "websockets==10.4",       # gradio_client may pull a newer incompatible version
            "jinja2==3.1.3",          # Gradio 3.x template compatibility
            "starlette==0.27.0",      # Gradio 3.x TemplateResponse positional-args signature
            "requests==2.31.0",       # Avoid breaking changes in 2.32+
            "pydantic==1.10.21",      # Gradio 3.x requires Pydantic v1; v2 removes FieldInfo.in_
                                      # → fatal 'FieldInfo object has no attribute in_' at launch.
                                      # Force-reinstall ensures gradio's dep resolver can't pull v2.
        }
        
        base_reqs = [r for r in all_requirements if r not in CRITICAL_PINNED]
        critical_reqs = [r for r in all_requirements if r in CRITICAL_PINNED]
        
        # Phase 1: Install base requirements normally (allow dependency resolution)
        print_status(f"Installing base packages (Gradio {SELECTED_GRADIO})...")
        total_base = len(base_reqs)
        for i, req in enumerate(base_reqs, 1):
            pkg_name = req.split('>=')[0].split('==')[0].split('[')[0]
            print(f"  [{i}/{total_base}] Installing {pkg_name}...  ", end='', flush=True)
            
            if pip_install_with_retry(pip_exe, req, max_retries=10, initial_delay=5.0):
                print(f" OK")
            else:
                print(f" FAILED")
                print_status(f"Failed to install {pkg_name} after 10 retries", False)
                return False
        
        print_status("Base packages installed")
        
        # Phase 2: Force-install critical pinned packages to override conflicts
        # Use --force-reinstall --no-deps to bypass resolver warnings
        if critical_reqs:
            print_status("Applying critical version pins for Gradio 3.50.2 compatibility...")
            for req in critical_reqs:
                pkg_name = req.split('==')[0]
                print(f"  Pinning {pkg_name}...  ", end='', flush=True)
                
                # Use force-reinstall with no-deps to override without dependency checks
                result = subprocess.run(
                    [pip_exe, "install", "--force-reinstall", "--no-deps", req],
                    capture_output=True, text=True, timeout=300
                )
                
                if result.returncode == 0 or "already satisfied" in result.stdout.lower():
                    print(f" OK")
                else:
                    # Fallback to normal install if force fails
                    print(f" (fallback)  ", end='', flush=True)
                    if pip_install_with_retry(pip_exe, req, max_retries=3, initial_delay=5.0):
                        print(f" OK")
                    else:
                        print(f" FAILED")
                        print_status(f"Failed to pin {pkg_name}", False)
                        return False
        
        # Install embedding backend (torch + sentence-transformers)
        if not install_embedding_backend():
            return False
        
        # Install Qt WebEngine for custom browser
        install_qt_webengine()  # Non-fatal if fails
        
        # llama-cpp-python installation
        info = BACKEND_OPTIONS[backend]
        
        if not info.get("compile_wheel"):
            # Download-only routes (options 1 & 2) — try sources in priority order.
            # No source compilation occurs here; if all prebuilt sources fail the
            # user is directed to a compile route (options 3 or 4).
            wheel_version = LLAMACPP_PYTHON_PREBUILT_VERSION.lstrip("v")
            sources = _get_prebuilt_wheel_urls()

            if not sources:
                print_status("No pre-built wheel sources available for this platform.", False)
                print("  Re-run the installer and select option 3 or 4 to compile from source.")
                return False

            print_status(f"Installing llama-cpp-python {wheel_version} (CPU, trying {len(sources)} sources)...")
            installed = False

            for src in sources:
                label = src.get("label", src["value"])
                print(f"  Trying: {label}")

                if src["type"] == "url":
                    # Direct .whl URL — quick 404 check before full retry loop
                    installed = pip_install_with_retry(
                        pip_exe, src["value"],
                        max_retries=2, initial_delay=3.0   # fail fast on 404
                    )

                elif src["type"] == "index":
                    # pip install with an extra index URL
                    installed = pip_install_with_retry(
                        pip_exe, src["value"],
                        extra_args=["--extra-index-url", src["extra_index"],
                                    "--prefer-binary"],
                        max_retries=3, initial_delay=5.0
                    )

                elif src["type"] == "pypi":
                    # PyPI with --prefer-binary: uses any binary wheel available;
                    # does NOT compile from source (pip exits non-zero if no binary found).
                    installed = pip_install_with_retry(
                        pip_exe, src["value"],
                        extra_args=["--prefer-binary"],
                        max_retries=3, initial_delay=5.0
                    )

                if installed:
                    print_status(f"llama-cpp-python {wheel_version} installed via {label}")
                    break
                else:
                    print(f"  Source unavailable: {label}")

            if not installed:
                print_status(f"llama-cpp-python {wheel_version} could not be installed from any prebuilt source.", False)
                print()
                print("  All prebuilt wheel sources were tried and failed.")
                print(f"  The most likely cause: no prebuilt CPU wheel exists yet for v{wheel_version}.")
                print(f"  (eswarthammana last published v0.3.16; abetlen CPU index may lag behind PyPI)")
                print()
                print("  Options:")
                print("    1. Re-run the installer and select option 3 or 4 to compile from source.")
                print("       Compilation takes 10-30 minutes but produces a wheel matched to your CPU.")
                print(f"   2. Manually install an older compatible wheel:")
                print(f"      pip install llama-cpp-python==0.3.16 --extra-index-url")
                print(f"      https://abetlen.github.io/llama-cpp-python/whl/cpu")
                print(f"      (Note: v0.3.16 does NOT support Qwen3.5/qwen35 architecture)")
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
            
            if PLATFORM == "windows" and not check_vcredist_windows():
                print_status("Warning: Visual C++ Redistributable (x64) not detected -  "
                            "compiled binaries may fail to run. Install vc_redist.x64.exe  "
                            "from Microsoft if you encounter DLL errors.", False)
                time.sleep(3)
            
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
                # Explicit generator + architecture: fully deterministic
                cmake_args.extend(["-G", VS_GENERATOR, "-A", "x64"])
            else:
                # No generator detected - let CMake auto-select the VS generator
                # but still pin the architecture so it never falls back to Win32.
                # Note: -A without -G is only valid when CMake auto-selects a
                # Visual Studio generator; if no VS is present at all, cmake
                # will already have failed earlier in detect_build_tools_available.
                cmake_args.extend(["-A", "x64"])
        
        print_status("Configuring build...")
        result = subprocess.run(cmake_args, cwd=build_dir, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print_status(f"CMake configure failed: {result.stderr[:200]}", False)
            return False
        
        print_status("Building binaries (this may take a while on slower hardware)...")
        print(f"  Threads: {build_threads}")
        
        if PLATFORM == "windows":
            build_cmd = ["cmake", "--build", ".", "--config", "Release", "--parallel", str(build_threads)]
        else:
            build_cmd = ["cmake", "--build", ".", "--parallel", str(build_threads)]
        
        process = subprocess.Popen(
            build_cmd, cwd=build_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env
        )
        
        track_process(process.pid)

        bin_build_start = time.time()
        last_bin_print = [time.time()]
        HEARTBEAT_INTERVAL = 30

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            now = time.time()
            is_error = any(x in line.lower() for x in ["error", "fatal", "failed"])
            is_progress = any(x in line.lower() for x in ["building", "compiling", "linking", "["])
            if is_error:
                print(f"\n  [!] {line[:120]}", flush=True)
                last_bin_print[0] = now
            elif is_progress:
                elapsed = int(now - bin_build_start)
                print(f"\r  [{elapsed:>4}s] {line[:100]}", end="", flush=True)
                last_bin_print[0] = now
            if now - last_bin_print[0] >= HEARTBEAT_INTERVAL:
                elapsed = int(now - bin_build_start)
                print(f"\n  [{elapsed:>4}s] Still building... (please wait)", end="", flush=True)
                last_bin_print[0] = now

        process.wait()   # no timeout — stdout loop already drained the process
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

def _path_prepend(bin_dir: str) -> None:
    """Prepend bin_dir to PATH only when it is not already a member (exact path-list check)."""
    current_entries = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in current_entries:
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"

def _vs_base_dirs():
    """
    Return ordered list of (year, base_path) pairs to probe for Visual Studio.

    VS 2022 changed its install root to %ProgramFiles% (64-bit).
    VS 2019 and 2017 install to %ProgramFiles(x86)% by default but can also
    live in %ProgramFiles% when installed on a 64-bit-only host, so we check
    both roots for every year (newest year first, 64-bit root first per year).
    """
    pf   = Path(os.environ.get("ProgramFiles",       r"C:\Program Files"))
    pf86 = Path(os.environ.get("ProgramFiles(x86)",  r"C:\Program Files (x86)"))
    vs_pf   = pf   / "Microsoft Visual Studio"
    vs_pf86 = pf86 / "Microsoft Visual Studio"
    # (year, primary_base, fallback_base)
    return [
        ("2022", vs_pf,   vs_pf86),   # VS 2022: default is ProgramFiles
        ("2019", vs_pf86, vs_pf),     # VS 2019: default is ProgramFiles(x86)
        ("2017", vs_pf86, vs_pf),     # VS 2017: default is ProgramFiles(x86)
    ]

def detect_build_tools_available() -> dict:
    """Detect build tools availability and add to PATH if found. Returns dict of tool: bool"""
    global VS_GENERATOR
    tools = {"Git": False, "CMake": False, "MSVC": False, "MSBuild": False}

    # Git - required for all platforms
    if shutil.which("git"):
        tools["Git"] = True

    if PLATFORM == "windows":
        _EDITIONS = ["Community", "Professional", "Enterprise", "BuildTools"]

        # VS year -> CMake generator string
        _VS_GENERATORS = {
            "2022": "Visual Studio 17 2022",
            "2019": "Visual Studio 16 2019",
            "2017": "Visual Studio 15 2017",
        }

        # VS year -> MSBuild sub-path relative to edition root
        # VS 2017 uses the versioned "15.0" folder; 2019/2022 use "Current"
        _MSBUILD_SUBPATH = {
            "2022": Path("MSBuild") / "Current" / "Bin" / "MSBuild.exe",
            "2019": Path("MSBuild") / "Current" / "Bin" / "MSBuild.exe",
            "2017": Path("MSBuild") / "15.0"    / "Bin" / "MSBuild.exe",
        }

        # ── CMake ─────────────────────────────────────────────────────────────
        if shutil.which("cmake"):
            tools["CMake"] = True
        else:
            # Search bundled cmake inside every VS installation
            for year, primary, fallback in _vs_base_dirs():
                for vs_base in (primary, fallback):
                    if not vs_base.exists():
                        continue
                    for edition in _EDITIONS:
                        cmake_candidate = (
                            vs_base / year / edition
                            / "Common7" / "IDE" / "CommonExtensions"
                            / "Microsoft" / "CMake" / "CMake" / "bin" / "cmake.exe"
                        )
                        if cmake_candidate.exists():
                            tools["CMake"] = True
                            _path_prepend(str(cmake_candidate.parent))
                            break
                    if tools["CMake"]:
                        break
                if tools["CMake"]:
                    break
            # Standalone CMake installation
            if not tools["CMake"]:
                for pf_env in ("ProgramFiles", "ProgramFiles(x86)"):
                    cmake_standalone = (
                        Path(os.environ.get(pf_env, r"C:\Program Files"))
                        / "CMake" / "bin" / "cmake.exe"
                    )
                    if cmake_standalone.exists():
                        tools["CMake"] = True
                        _path_prepend(str(cmake_standalone.parent))
                        break

        # ── MSVC + VS_GENERATOR ───────────────────────────────────────────────
        for year, primary, fallback in _vs_base_dirs():
            if tools["MSVC"]:
                break
            for vs_base in (primary, fallback):
                if not vs_base.exists():
                    continue
                for edition in _EDITIONS:
                    vc_tools = vs_base / year / edition / "VC" / "Tools" / "MSVC"
                    if not vc_tools.exists():
                        continue
                    # Verify at least one cl.exe exists in a Hostx64/x64 toolchain
                    cl_found = any(
                        (ver_dir / "bin" / "Hostx64" / "x64" / "cl.exe").exists()
                        for ver_dir in sorted(vc_tools.iterdir(), reverse=True)
                        if ver_dir.is_dir()
                    )
                    if cl_found:
                        tools["MSVC"] = True
                        if VS_GENERATOR is None:
                            VS_GENERATOR = _VS_GENERATORS[year]
                        break
                if tools["MSVC"]:
                    break

        # Fallback: vswhere (check PATH first, then fixed installer location)
        if not tools["MSVC"]:
            vswhere_exe = shutil.which("vswhere")
            if not vswhere_exe:
                vswhere_path = (
                    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
                    / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
                )
                if vswhere_path.exists():
                    vswhere_exe = str(vswhere_path)
            if vswhere_exe:
                try:
                    result = subprocess.run(
                        [vswhere_exe, "-latest",
                         "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                         "-property", "installationVersion"],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        tools["MSVC"] = True
                        ver = result.stdout.strip()
                        if   ver.startswith("17"): VS_GENERATOR = "Visual Studio 17 2022"
                        elif ver.startswith("16"): VS_GENERATOR = "Visual Studio 16 2019"
                        elif ver.startswith("15"): VS_GENERATOR = "Visual Studio 15 2017"
                except Exception:
                    pass

        # ── MSBuild ───────────────────────────────────────────────────────────
        if shutil.which("MSBuild"):
            tools["MSBuild"] = True
        else:
            for year, primary, fallback in _vs_base_dirs():
                if tools["MSBuild"]:
                    break
                for vs_base in (primary, fallback):
                    if not vs_base.exists():
                        continue
                    for edition in _EDITIONS:
                        msbuild_candidate = vs_base / year / edition / _MSBUILD_SUBPATH[year]
                        if msbuild_candidate.exists():
                            tools["MSBuild"] = True
                            _path_prepend(str(msbuild_candidate.parent))
                            break
                    if tools["MSBuild"]:
                        break

    else:  # Linux
        tools["CMake"]   = shutil.which("cmake") is not None
        tools["MSVC"]    = shutil.which("gcc")   is not None   # GCC on Linux
        tools["MSBuild"] = shutil.which("make")  is not None

    return tools

def select_backend_and_install_size():
    """Combined selection of backend and install size (small/large) on one page.

    Install size determines both the embedding model and TTS engine:
      a) Smaller  ~132MB  - Bge-Small-En v1.5 + pyttsx3/espeak-ng (built-in TTS)
      b) Larger   ~2GB    - Bge-Base-En  v1.5 + Coqui TTS English Male/Female
                            (falls back to built-in TTS on incompatible OS)

    Returns: (backend_str, embedding_model_name, tts_engine, coqui_voice_dict_or_None)
    """

    width = shutil.get_terminal_size().columns - 1
    print_header("Configure Installation")

    all_backend_opts = list(BACKEND_OPTIONS.keys())

    # ── System Detections ────────────────────────────────────────────────────
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
    missing_tools   = [tool for tool, present in build_tools.items() if not present]
    tools_str = ", ".join(available_tools) if available_tools else "None"
    print(f"    Build Tools: {tools_str}")

    print()

    # ── Backend options (filter out compile routes if tools missing) ─────────
    build_possible = len(missing_tools) == 0
    backend_opts = []
    for backend in all_backend_opts:
        info = BACKEND_OPTIONS[backend]
        requires_compile = info.get("compile_binary", False) or info.get("compile_wheel", False)
        if requires_compile and not build_possible:
            continue
        backend_opts.append(backend)

    # Resolve compile-route wheel version (API call; cached after first call)
    compile_ver = get_latest_llamacpp_python_version()
    prebuilt_ver = LLAMACPP_PYTHON_PREBUILT_VERSION

    def _wheel_label(backend_name: str) -> str:
        """Return the wheel version annotation for a backend menu option."""
        info = BACKEND_OPTIONS[backend_name]
        if info.get("compile_wheel"):
            return f"Wheel {compile_ver}"
        else:
            return f"Wheel {prebuilt_ver}"

    print("Backend Options...")
    for i, backend in enumerate(backend_opts, 1):
        print(f"   {i}) {backend} ({_wheel_label(backend)})")

    if not build_possible:
        print(f"   ---")
        for backend in all_backend_opts:
            if backend not in backend_opts:
                print(f"   -) {backend} ({_wheel_label(backend)}) (Missing: {', '.join(missing_tools)})")

    print()

    # ── Install Size ─────────────────────────────────────────────────────────
    coqui_ok = is_coqui_compatible()
    print("Install Size...")
    print(f"   a) Small  ~135MB - Bge-Small-En v1.5 + pyttsx3/espeak-ng (built-in TTS)")
    print(f"   b) Medium ~450MB - Bge-Base-En v1.5 + pyttsx3/espeak-ng (built-in TTS)")
    if coqui_ok:
        print(f"   c) Large   ~2GB  - Bge-Base-En v1.5 + Coqui TTS (high quality voices)")
    else:
        # Inform the user Coqui is unavailable on this OS version
        if PLATFORM == "windows":
            note = f"Windows {WINDOWS_VERSION} - Coqui requires Windows 8.1/10/11"
        else:
            note = f"Ubuntu {OS_VERSION} - Coqui requires Ubuntu 24+"
        print(f"   c) Large   ~2GB  - Bge-Base-En v1.5 + Coqui TTS ({note})")

    print()
    print("=" * width)

    max_backend = len(backend_opts)
    prompt = f"Selection; Backend=1-{max_backend}, Size=a-c, Abandon=A; (e.g. 2b): "

    choice = input(prompt).strip().lower()

    if choice == "a":
        print("Abandoning installation...")
        sys.exit(0)

    choice = choice.replace(" ", "").replace("-", "")

    while True:
        # Valid formats: "1a", "2b", "3c", etc.
        if len(choice) >= 2 and choice[0].isdigit() and choice[1] in "abc":
            backend_num  = int(choice[0])
            size_letter  = choice[1]

            if 1 <= backend_num <= len(backend_opts):
                selected_backend = backend_opts[backend_num - 1]

                if size_letter == "a":
                    # Small: bge-small + built-in TTS
                    embedding_model = EMBEDDING_MODELS["1"]["name"]
                    tts_engine      = "builtin"
                    coqui_voice     = None
                elif size_letter == "b":
                    # Medium: bge-base + built-in TTS
                    embedding_model = EMBEDDING_MODELS["2"]["name"]
                    tts_engine      = "builtin"
                    coqui_voice     = None
                else:
                    # Large: bge-base + Coqui TTS (if compatible)
                    embedding_model = EMBEDDING_MODELS["2"]["name"]
                    if coqui_ok:
                        tts_engine  = "coqui"
                        coqui_voice = COQUI_ENGLISH_VOICE
                    else:
                        # Coqui not compatible - inform user and continue loop
                        print("Coqui TTS is not compatible with this OS version.")
                        print("Please select option 'a' (Small) or 'b' (Medium).")
                        print()
                        # Show the menu again
                        print("Install Size...")
                        print(f"   a) Small  ~135MB - Bge-Small-En v1.5 + pyttsx3/espeak-ng (built-in TTS)")
                        print(f"   b) Medium ~450MB - Bge-Base-En v1.5 + pyttsx3/espeak-ng (built-in TTS)")
                        if PLATFORM == "windows":
                            note = f"Windows {WINDOWS_VERSION} - Coqui requires Windows 8.1/10/11"
                        else:
                            note = f"Ubuntu {OS_VERSION} - Coqui requires Ubuntu 24+"
                        print(f"   c) Large   ~2GB  - Bge-Base-En v1.5 + Coqui TTS ({note})")
                        print()
                        # Get new input
                        choice = input(f"Selection; Backend=1-{max_backend}, Size=a-c, Abandon=A; (e.g. 2b): ").strip().lower()
                        if choice == "a":
                            print("Abandoning installation...")
                            sys.exit(0)
                        choice = choice.replace(" ", "").replace("-", "")
                        continue

                time.sleep(1)
                return selected_backend, embedding_model, tts_engine, coqui_voice

        print("Invalid selection. Please enter a valid combination (e.g. 2b).")
        prompt = f"Selection; Backend=1-{max_backend}, Size=a-c, Abandon=A; (e.g. 2b): "

        choice = input(prompt).strip().lower()
        if choice == "a":
            print("\nAbandoning installation...")
            sys.exit(0)
        choice = choice.replace(" ", "").replace("-", "")

def is_coqui_compatible() -> bool:
    """Check if current OS supports Coqui TTS.
    
    Coqui TTS requires:
    - Windows 8.1, 10, or 11
    - Ubuntu 24 or 25
    
    Returns:
        bool: True if Coqui TTS is supported on this OS
    """
    if PLATFORM == "windows":
        # WINDOWS_VERSION is detected earlier in install()
        # Valid versions: "8.1", "10", "11"
        # Invalid versions: "7", "8"
        if WINDOWS_VERSION in ["8.1", "10", "11"]:
            return True
        return False
    
    elif PLATFORM == "linux":
        # OS_VERSION contains Ubuntu version like "24.04", "24.10", "25.04"
        # We need Ubuntu 24.x or 25.x
        try:
            if OS_VERSION:
                major_version = int(OS_VERSION.split('.')[0])
                if major_version >= 24:
                    return True
        except (ValueError, AttributeError, IndexError):
            pass
        return False
    
    return False

def install_espeak_ng_windows():
    """Extract espeak-ng to project data folder. Fails hard if extraction fails."""
    import platform
    import urllib.request
    import shutil
    
    espeak_dir = BASE_DIR / "data" / "espeak-ng"
    espeak_exe = espeak_dir / "espeak-ng.exe"
    espeak_dll = espeak_dir / "libespeak-ng.dll"
    
    print_status("Installing espeak-ng (Coqui dependency)...")
    
    if espeak_dll.exists() and espeak_exe.exists():
        try:
            result = subprocess.run([str(espeak_exe), "--version"], capture_output=True, timeout=5)
            if result.returncode == 0:
                print_status("espeak-ng verified")
                return True
        except:
            shutil.rmtree(espeak_dir, ignore_errors=True)
    
    espeak_dir.mkdir(parents=True, exist_ok=True)
    
    is_64bit = platform.machine().endswith('64')
    msi_url = "https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi" if is_64bit else "https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X86.msi"
    msi_filename = "espeak-ng-X64.msi" if is_64bit else "espeak-ng-X86.msi"
    
    msi_path = TEMP_DIR / msi_filename
    extract_dir = TEMP_DIR / "espeak_extract"
    
    try:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        
        urllib.request.urlretrieve(msi_url, str(msi_path))
        
        if not msi_path.exists():
            print_status("ERROR: espeak-ng download failed", False)
            sys.exit(1)
        
        # Extract using msiexec /a (administrative extraction, no system install)
        result = subprocess.run(
            ["msiexec", "/a", str(msi_path), "/qn", f"TARGETDIR={str(extract_dir)}"],
            capture_output=True, timeout=120
        )
        
        if result.returncode not in [0, 3010]:
            # Try 7z fallback
            try:
                result = subprocess.run(
                    ["7z", "x", str(msi_path), f"-o{str(extract_dir)}", "-y"],
                    capture_output=True, timeout=60
                )
                if result.returncode != 0:
                    print_status(f"ERROR: Extraction failed: {result.stderr.decode()[:200]}", False)
                    sys.exit(1)
            except FileNotFoundError:
                print_status("ERROR: msiexec failed and 7z not available", False)
                sys.exit(1)
        
        # Find extracted files
        source_dir = None
        for root, dirs, files in os.walk(extract_dir):
            if "espeak-ng.exe" in files:
                source_dir = Path(root)
                break
        
        if not source_dir:
            print_status("ERROR: espeak-ng.exe not found in extracted MSI", False)
            sys.exit(1)
        
        # Copy to project folder
        for item in source_dir.iterdir():
            dest = espeak_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
        
        # Verify critical files
        if not espeak_dll.exists():
            print_status("ERROR: libespeak-ng.dll missing after extraction", False)
            sys.exit(1)
        if not espeak_exe.exists():
            print_status("ERROR: espeak-ng.exe missing after extraction", False)
            sys.exit(1)
        
        msi_path.unlink(missing_ok=True)
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)
        
        print_status(f"espeak-ng installed ({len(list(espeak_dir.glob('**/*')))} files)")
        return True
        
    except subprocess.TimeoutExpired:
        print_status("ERROR: espeak-ng installation timed out", False)
        sys.exit(1)
    except Exception as e:
        print_status(f"ERROR: espeak-ng installation failed: {e}", False)
        sys.exit(1)
    finally:
        if msi_path.exists():
            msi_path.unlink(missing_ok=True)
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)

# =============================================================================
# INSTALL MODE HELPERS
# =============================================================================

def select_install_mode() -> str:
    """
    Show the install-mode menu. Called as the very first step inside install(),
    before any purge, venv creation, or package work.
    Returns one of: 'clean', 'check', 'refresh'
    """
    print_header("Installation")
    print("\n\n\n\n")
    print("   1. Clean Install (Purge First)\n")
    print("   2. Check/Install (Fix Missing Packages/Libraries)\n")
    print("   3. Refresh Configs (Only Remake Ini/Json)\n")
    print("\n\n\n\n")
    print("-" * (shutil.get_terminal_size().columns - 1))
    while True:
        choice = input("Selection; Menu Options = 1-3, Abandon Install = A: ").strip().upper()
        if choice == "A":
            print("\nAbandoning installation...")
            sys.exit(0)
        if choice == "1":
            return "clean"
        if choice == "2":
            return "check"
        if choice == "3":
            return "refresh"
        print("Invalid choice, please try again.")


def ensure_venv() -> bool:
    """Create venv only when it does not already exist (no purge)."""
    if VENV_DIR.exists():
        print_status("Existing virtual environment found - skipping recreation")
        return True
    return create_venv()


def _read_existing_ini() -> dict:
    """
    Read data/constants.ini and return a dict of stored values, or None if
    the file is missing or unreadable.  Used by Check/Install and Refresh modes
    so they can skip re-showing the backend/TTS menus.
    """
    import configparser
    ini_path = BASE_DIR / "data" / "constants.ini"
    if not ini_path.exists():
        return None
    try:
        config = configparser.ConfigParser()
        config.read(ini_path, encoding='utf-8')
        if 'system' not in config:
            return None
        sys_sec = config['system']
        result = {
            'platform':           sys_sec.get('platform',        PLATFORM),
            'os_version':         sys_sec.get('os_version',      'unknown'),
            'python_version':     sys_sec.get('python_version',  f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),
            'backend_type':       sys_sec.get('backend_type',    'CPU_CPU'),
            'embedding_model':    sys_sec.get('embedding_model', 'BAAI/bge-small-en-v1.5'),
            'vulkan_available':   sys_sec.getboolean('vulkan_available', False),
            'windows_version':    sys_sec.get('windows_version', None),
            'llama_cli_path':     sys_sec.get('llama_cli_path',  None),
            'llama_bin_path':     sys_sec.get('llama_bin_path',  None),
            'tts_engine':         'builtin',
            'coqui_voice_id':     None,
            'coqui_voice_accent': None,
        }
        if 'tts' in config:
            tts_sec = config['tts']
            result['tts_engine'] = tts_sec.get('tts_type', 'builtin')
            if result['tts_engine'] == 'coqui':
                result['coqui_voice_id']     = tts_sec.get('coqui_voice_id',    None)
                result['coqui_voice_accent'] = tts_sec.get('coqui_voice_accent', None)
        return result
    except Exception as e:
        print_status(f"Could not read constants.ini: {e}", False)
        return None


def _backend_type_to_string(backend_type: str) -> str:
    """
    Map a backend_type token back to the corresponding BACKEND_OPTIONS key.
    Download variants are preferred so Check/Install never triggers compilation.
    """
    if backend_type == "VULKAN_VULKAN":
        return "Compile Vulkan Binaries / Compile Vulkan Wheel"
    elif backend_type == "VULKAN_CPU":
        return "Download Vulkan Binary / Default CPU Wheel"
    else:
        return "Download CPU Binary / Default CPU Wheel"

def validate_installation(backend: str, embedding_model: str, tts_engine: str) -> bool:
    """
    Comprehensive validation of installation integrity.
    Used by Check/Install mode to verify all components are working.
    Returns True if all critical checks pass.
    """
    print_status("Validating installation integrity...")
    
    python_exe_path = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / \
                      ("python.exe" if PLATFORM == "windows" else "python")
    python_exe = str(python_exe_path)

    if not python_exe_path.exists():
        print_status("Python executable missing from venv", False)
        return False
    
    all_passed = True
    
    # 1. Verify core libraries can be imported
    print("\n=== Core Library Validation ===")
    core_libs = [
        ("gradio", "gradio"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("sentence_transformers", "sentence_transformers"),
        ("llama_cpp", "llama_cpp"),
        ("spacy", "spacy"),
    ]
    
    for pkg_name, import_name in core_libs:
        try:
            result = subprocess.run(
                [python_exe, "-c", f"import {import_name}; print('OK')"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and "OK" in result.stdout:
                print_status(f"  {pkg_name} OK")
            else:
                print_status(f"  {pkg_name} FAILED", False)
                all_passed = False
        except Exception as e:
            print_status(f"  {pkg_name} ERROR: {e}", False)
            all_passed = False
    
    # 2. Verify embedding model loads correctly
    print("\n=== Embedding Model Validation ===")
    cache_dir = BASE_DIR / "data" / "embedding_cache"
    if cache_dir.exists():
        test_code = f'''
import os
os.environ["TRANSFORMERS_CACHE"] = r"{str(cache_dir.absolute())}"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"{str(cache_dir.absolute())}"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("{embedding_model}", device="cpu", cache_folder=r"{str(cache_dir.absolute())}")
    result = model.encode(["validation test"], convert_to_tensor=True)
    print("OK")
except Exception as e:
    print(f"Error: {{e}}")
'''
        try:
            result = subprocess.run(
                [python_exe, "-c", test_code],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0 and "OK" in result.stdout:
                print_status(f"  Embedding model verified ({embedding_model})")
            else:
                print_status("  Embedding model failed to load", False)
                print(f"    Error: {result.stderr.strip()[:200]}")
                all_passed = False
        except subprocess.TimeoutExpired:
            print_status("  Embedding model check timed out", False)
            all_passed = False
        except Exception as e:
            print_status(f"  Embedding validation error: {e}", False)
            all_passed = False
    else:
        print_status("  Embedding cache directory missing", False)
        all_passed = False
    
    # 3. Verify spaCy model
    print("\n=== spaCy Model Validation ===")
    try:
        result = subprocess.run(
            [python_exe, "-c", "import spacy; nlp = spacy.load('en_core_web_sm'); print('OK')"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and "OK" in result.stdout:
            print_status("  en_core_web_sm model available")
        else:
            print_status("  en_core_web_sm model not found", False)
            all_passed = False
    except Exception as e:
        print_status(f"  spaCy model check failed: {e}", False)
        all_passed = False
    
    # 4. Verify TTS engine
    print("\n=== TTS Validation ===")
    if tts_engine == "coqui":
        try:
            result = subprocess.run(
                [python_exe, "-c", "from TTS.api import TTS; print('OK')"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and "OK" in result.stdout:
                print_status("  Coqui TTS package importable")
            else:
                print_status("  Coqui TTS package import failed", False)
                all_passed = False
        except Exception as e:
            print_status(f"  Coqui TTS error: {e}", False)
            all_passed = False
        
        # Check TTS model directory
        tts_model_dir = BASE_DIR / "data" / "tts_models"
        if tts_model_dir.exists():
            model_files = list(tts_model_dir.rglob("*.pth")) + list(tts_model_dir.rglob("*.json"))
            if len(model_files) > 0:
                print_status(f"  TTS model directory present ({len(model_files)} files)")
            else:
                print_status("  TTS model directory empty", False)
                all_passed = False
        else:
            print_status("  TTS model directory missing", False)
            all_passed = False
    else:
        # Built-in TTS
        if PLATFORM == "windows":
            try:
                result = subprocess.run(
                    [python_exe, "-c", "import pyttsx3; print('OK')"],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0 and "OK" in result.stdout:
                    print_status("  pyttsx3 (Built-in TTS) available")
                else:
                    print_status("  pyttsx3 import failed", False)
                    all_passed = False
            except Exception as e:
                print_status(f"  pyttsx3 error: {e}", False)
                all_passed = False
        else:
            try:
                result = subprocess.run(["espeak-ng", "--version"],
                                       capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print_status(f"  espeak-ng available")
                else:
                    print_status("  espeak-ng returned error", False)
                    all_passed = False
            except FileNotFoundError:
                print_status("  espeak-ng not found", False)
                all_passed = False
    
    # 5. Verify backend binaries (if applicable)
    print("\n=== Backend Binary Validation ===")
    info = BACKEND_OPTIONS.get(backend, {})
    cli_path = info.get("cli_path")
    
    if cli_path:
        cli_full_path = BASE_DIR / cli_path
        if cli_full_path.exists():
            print_status(f"  llama-cli found: {cli_full_path.name}")
            if PLATFORM == "linux":
                if os.access(cli_full_path, os.X_OK):
                    print_status("  llama-cli is executable")
                else:
                    print_status("  llama-cli not executable", False)
                    all_passed = False
        else:
            print_status(f"  llama-cli not found: {cli_path}", False)
            all_passed = False
    else:
        print_status("  Python bindings mode: No binary needed")
    
    # 6. Verify configuration files
    print("\n=== Configuration Validation ===")
    constants_ini = BASE_DIR / "data" / "constants.ini"
    persistent_json = BASE_DIR / "data" / "persistent.json"
    
    if constants_ini.exists():
        print_status("  constants.ini exists")
    else:
        print_status("  constants.ini missing", False)
        all_passed = False
    
    if persistent_json.exists():
        try:
            with open(persistent_json, 'r') as f:
                json.load(f)
            print_status("  persistent.json valid")
        except Exception as e:
            print_status(f"  persistent.json corrupted: {e}", False)
            all_passed = False
    else:
        print_status("  persistent.json missing", False)
        all_passed = False
    
    # Summary
    print(f"\n{'=' * 50}")
    if all_passed:
        print_status("All validations passed!")
        print("  Installation is complete and ready to use.")
    else:
        print_status("Some checks failed", False)
        print("  Re-run installer with 'Clean Install' to fix issues.")
    print(f"{'=' * 50}\n")
    
    return all_passed


# =============================================================================
# MAIN INSTALL FLOW
# =============================================================================

def validate_installation(backend: str, embedding_model: str, tts_engine: str) -> bool:
    """
    Comprehensive validation of installation integrity.
    Used by Check/Install mode to verify all components are working.
    Returns True if all critical checks pass.
    """
    print_status("Validating installation integrity...")
    
    python_exe_path = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / \
                      ("python.exe" if PLATFORM == "windows" else "python")
    python_exe = str(python_exe_path)

    if not python_exe_path.exists():
        print_status("Python executable missing from venv", False)
        return False
    
    all_passed = True
    
    # 1. Verify core libraries can be imported
    print("\n=== Core Library Validation ===")
    core_libs = [
        ("gradio", "gradio"),
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("sentence_transformers", "sentence_transformers"),
        ("llama_cpp", "llama_cpp"),
        ("spacy", "spacy"),
    ]
    
    for pkg_name, import_name in core_libs:
        try:
            result = subprocess.run(
                [python_exe, "-c", f"import {import_name}; print('OK')"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and "OK" in result.stdout:
                print_status(f"  {pkg_name} OK")
            else:
                print_status(f"  {pkg_name} FAILED", False)
                all_passed = False
        except Exception as e:
            print_status(f"  {pkg_name} ERROR: {e}", False)
            all_passed = False
    
    # 2. Verify embedding model loads correctly
    print("\n=== Embedding Model Validation ===")
    cache_dir = BASE_DIR / "data" / "embedding_cache"
    if cache_dir.exists():
        test_code = f'''
import os
os.environ["TRANSFORMERS_CACHE"] = r"{str(cache_dir.absolute())}"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"{str(cache_dir.absolute())}"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("{embedding_model}", device="cpu", cache_folder=r"{str(cache_dir.absolute())}")
    result = model.encode(["validation test"], convert_to_tensor=True)
    print("OK")
except Exception as e:
    print(f"Error: {{e}}")
'''
        try:
            result = subprocess.run(
                [python_exe, "-c", test_code],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0 and "OK" in result.stdout:
                print_status(f"  Embedding model verified ({embedding_model})")
            else:
                print_status("  Embedding model failed to load", False)
                print(f"    Error: {result.stderr.strip()[:200]}")
                all_passed = False
        except subprocess.TimeoutExpired:
            print_status("  Embedding model check timed out", False)
            all_passed = False
        except Exception as e:
            print_status(f"  Embedding validation error: {e}", False)
            all_passed = False
    else:
        print_status("  Embedding cache directory missing", False)
        all_passed = False
    
    # 3. Verify spaCy model
    print("\n=== spaCy Model Validation ===")
    try:
        result = subprocess.run(
            [python_exe, "-c", "import spacy; nlp = spacy.load('en_core_web_sm'); print('OK')"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0 and "OK" in result.stdout:
            print_status("  en_core_web_sm model available")
        else:
            print_status("  en_core_web_sm model not found", False)
            all_passed = False
    except Exception as e:
        print_status(f"  spaCy model check failed: {e}", False)
        all_passed = False
    
    # 4. Verify TTS engine
    print("\n=== TTS Validation ===")
    if tts_engine == "coqui":
        try:
            result = subprocess.run(
                [python_exe, "-c", "from TTS.api import TTS; print('OK')"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and "OK" in result.stdout:
                print_status("  Coqui TTS package importable")
            else:
                print_status("  Coqui TTS package import failed", False)
                all_passed = False
        except Exception as e:
            print_status(f"  Coqui TTS error: {e}", False)
            all_passed = False
        
        # Check TTS model directory
        tts_model_dir = BASE_DIR / "data" / "tts_models"
        if tts_model_dir.exists():
            model_files = list(tts_model_dir.rglob("*.pth")) + list(tts_model_dir.rglob("*.json"))
            if len(model_files) > 0:
                print_status(f"  TTS model directory present ({len(model_files)} files)")
            else:
                print_status("  TTS model directory empty", False)
                all_passed = False
        else:
            print_status("  TTS model directory missing", False)
            all_passed = False
    else:
        # Built-in TTS
        if PLATFORM == "windows":
            try:
                result = subprocess.run(
                    [python_exe, "-c", "import pyttsx3; print('OK')"],
                    capture_output=True, text=True, timeout=15
                )
                if result.returncode == 0 and "OK" in result.stdout:
                    print_status("  pyttsx3 (Built-in TTS) available")
                else:
                    print_status("  pyttsx3 import failed", False)
                    all_passed = False
            except Exception as e:
                print_status(f"  pyttsx3 error: {e}", False)
                all_passed = False
        else:
            try:
                result = subprocess.run(["espeak-ng", "--version"],
                                       capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print_status("  espeak-ng available")
                else:
                    print_status("  espeak-ng returned error", False)
                    all_passed = False
            except FileNotFoundError:
                print_status("  espeak-ng not found", False)
                all_passed = False
    
    # 5. Verify backend binaries (if applicable)
    print("\n=== Backend Binary Validation ===")
    info = BACKEND_OPTIONS.get(backend, {})
    cli_path = info.get("cli_path")
    
    if cli_path:
        cli_full_path = BASE_DIR / cli_path
        if cli_full_path.exists():
            print_status(f"  llama-cli found: {cli_full_path.name}")
            if PLATFORM == "linux":
                if os.access(cli_full_path, os.X_OK):
                    print_status("  llama-cli is executable")
                else:
                    print_status("  llama-cli not executable", False)
                    all_passed = False
        else:
            print_status(f"  llama-cli not found: {cli_path}", False)
            all_passed = False
    else:
        print_status("  Python bindings mode: No binary needed")
    
    # 6. Verify configuration files
    print("\n=== Configuration Validation ===")
    constants_ini = BASE_DIR / "data" / "constants.ini"
    persistent_json = BASE_DIR / "data" / "persistent.json"
    
    if constants_ini.exists():
        print_status("  constants.ini exists")
    else:
        print_status("  constants.ini missing", False)
        all_passed = False
    
    if persistent_json.exists():
        try:
            with open(persistent_json, 'r') as f:
                json.load(f)
            print_status("  persistent.json valid")
        except Exception as e:
            print_status(f"  persistent.json corrupted: {e}", False)
            all_passed = False
    else:
        print_status("  persistent.json missing", False)
        all_passed = False
    
    # Summary
    print(f"\n{'=' * 50}")
    if all_passed:
        print_status("All validations passed!")
        print("  Installation is complete and ready to use.")
    else:
        print_status("Some checks failed", False)
        print("  Re-run installer with 'Clean Install' to fix issues.")
    print(f"{'=' * 50}\n")
    
    return all_passed


def install():
    global WINDOWS_VERSION, OS_VERSION

    # ✅ NEW: run detection once at startup
    run_initial_detection()

    if not check_version_compatibility():
        sys.exit(1)

    detect_version_selections()

    install_mode = select_install_mode()

    # ✅ FIX: define python_version properly
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # OS detection
    windows_version = None

    if PLATFORM == "windows":
        WINDOWS_VERSION = detect_windows_version() or "unknown"
        os_version = WINDOWS_VERSION
        windows_version = WINDOWS_VERSION
        vulkan_available = is_vulkan_installed()
    else:
        OS_VERSION = detect_linux_version() or "unknown"
        os_version = OS_VERSION
        vulkan_available = is_vulkan_installed()

    # ✅ SAFE: cached, no prints — retained for diagnostic fields in constants.ini
    dx11_capable, dx_feature_level = detect_browser_acceleration()

    # =========================================================================
    # MODE: Refresh Configs
    # =========================================================================
    if install_mode == "refresh":
        existing = _read_existing_ini()
        if existing is None:
            print_status(
                "No existing configuration found - cannot refresh. "
                "Please run a Clean Install first.", False
            )
            sys.exit(1)

        print_header("Installation")
        print(f"Refreshing configuration files for {APP_NAME}...\n")

        create_system_ini(
            platform             = PLATFORM,
            os_version           = os_version,
            python_version       = python_version,
            backend_type         = existing['backend_type'],
            embedding_model      = existing['embedding_model'],
            windows_version      = windows_version,
            vulkan_available     = existing['vulkan_available'],
            llama_cli_path       = existing['llama_cli_path'],
            llama_bin_path       = existing['llama_bin_path'],
            tts_engine           = existing['tts_engine'],
            coqui_voice_id       = existing['coqui_voice_id'],
            coqui_voice_accent   = existing['coqui_voice_accent'],
            browser_acceleration = dx11_capable,
            dx_feature_level     = dx_feature_level,
        )

        backend_str = _backend_type_to_string(existing['backend_type'])
        create_config(backend_str, existing['embedding_model'])

        print_status("Configuration refresh complete!")
        print("\nRun the launcher to start Chat-Gradio-Gguf\n")
        return

    # =========================================================================
    # MODES: Clean Install / Check/Install
    # =========================================================================

    if install_mode == "check":
        existing = _read_existing_ini()
        if existing is not None:
            backend            = _backend_type_to_string(existing['backend_type'])
            embedding_model    = existing['embedding_model']
            tts_engine         = existing['tts_engine']
            coqui_voice_id     = existing['coqui_voice_id']
            coqui_voice_accent = existing['coqui_voice_accent']
            print_status("Existing configuration read - skipping backend/TTS menus")
        else:
            print_status("No existing configuration found - showing setup menus", False)
            time.sleep(2)
            backend, embedding_model, tts_engine, coqui_voice = select_backend_and_install_size()
            if backend_requires_compilation(backend):
                prompt_build_threads()
            coqui_voice_id     = coqui_voice['id']     if (tts_engine == "coqui" and coqui_voice) else None
            coqui_voice_accent = coqui_voice['accent'] if (tts_engine == "coqui" and coqui_voice) else None
    else:
        backend, embedding_model, tts_engine, coqui_voice = select_backend_and_install_size()
        if backend_requires_compilation(backend):
            prompt_build_threads()
        coqui_voice_id     = coqui_voice['id']     if (tts_engine == "coqui" and coqui_voice) else None
        coqui_voice_accent = coqui_voice['accent'] if (tts_engine == "coqui" and coqui_voice) else None

    # Resolve backend_type token
    if backend in ["Download CPU Binary / Default CPU Wheel",
                   "Compile CPU Binaries / Compile CPU Wheel"]:
        backend_type     = "CPU_CPU"
        vulkan_available = False
    elif backend == "Download Vulkan Binary / Default CPU Wheel":
        backend_type     = "VULKAN_CPU"
        vulkan_available = True
    elif backend in ["Download Vulkan Binary / Compile Vulkan Wheel",
                     "Compile Vulkan Binaries / Compile Vulkan Wheel"]:
        backend_type     = "VULKAN_VULKAN"
        vulkan_available = True
    else:
        backend_type     = "CPU_CPU"
        vulkan_available = False

    # ── Install header ────────────────────────────────────────────────────────
    print_header("Installation")

    if PLATFORM == "windows":
        os_display = f"Windows {WINDOWS_VERSION}" if WINDOWS_VERSION else "Windows"
    else:
        os_display = f"Ubuntu {OS_VERSION}" if OS_VERSION else "Ubuntu"

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    mode_label = "Check/Install" if install_mode == "check" else "Clean Install"

    print(f"Installing {APP_NAME} on {os_display} with Python {py_ver}")
    print(f"  Mode: {mode_label}")
    print(f"  Route: {backend}")
    print(f"  Llama.Cpp {LLAMACPP_TARGET_VERSION}, Gradio {SELECTED_GRADIO}, Qt-Web {SELECTED_QTWEB}")
    print(f"  Embedding: {embedding_model}")

    if PLATFORM == "windows":
        fl_str = f"0x{dx_feature_level:04x}"
        print(f"  GPU: DirectX Feature Level {fl_str}")

    if tts_engine == "coqui" and coqui_voice_id:
        print(f"  TTS: Coqui ({coqui_voice_id} / {coqui_voice_accent or 'english'})")
    else:
        print(f"  TTS: Built-in (pyttsx3/espeak-ng)")

    if install_mode == "clean":
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

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if install_mode == "clean":
        if not create_venv():
            print_status("Virtual environment failed", False)
            sys.exit(1)
    else:
        if not ensure_venv():
            print_status("Virtual environment failed", False)
            sys.exit(1)

    print_status("Installing py-cpuinfo for CPU detection...")
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                 ("pip.exe" if PLATFORM == "windows" else "pip"))
    subprocess.run([pip_exe, "install", "py-cpuinfo"], check=True)
    print_status("py-cpuinfo installed")

    if PLATFORM == "windows" and WINDOWS_VERSION == "8.1":
        print("Detected Windows 8.1")
    elif PLATFORM == "windows" and not dx11_capable:
        print(f"Detected DirectX {dx_feature_level:#06x}")

    create_files_and_directories(backend)

    info = BACKEND_OPTIONS[backend]

    create_system_ini(
        platform             = PLATFORM,
        os_version           = os_version,
        python_version       = python_version,
        backend_type         = backend_type,
        embedding_model      = embedding_model,
        windows_version      = windows_version,
        vulkan_available     = vulkan_available,
        llama_cli_path       = info["cli_path"],
        llama_bin_path       = info["dest"],
        tts_engine           = tts_engine,
        coqui_voice_id       = coqui_voice_id,
        coqui_voice_accent   = coqui_voice_accent,
        browser_acceleration = dx11_capable,
        dx_feature_level     = dx_feature_level,
    )

    if PLATFORM == "linux":
        if not install_linux_system_dependencies(backend):
            print_status("System dependencies installation failed", False)
            sys.exit(1)

    if not install_python_deps(backend):
        print_status("Python dependencies failed", False)
        sys.exit(1)

    install_optional_file_support()

    if not initialize_embedding_cache(embedding_model):
        print_status("CRITICAL: Embedding model required by RAG", False)
        sys.exit(1)

    spacy_ok = download_spacy_model()
    if not spacy_ok:
        print_status("WARNING: spaCy model download failed, session labeling may not work", False)

    if tts_engine == "coqui":
        if not install_coqui_tts():
            sys.exit(1)

    if not download_extract_backend(backend):
        print_status("Backend download failed", False)
        sys.exit(1)

    create_config(backend, embedding_model)

    # =========================================================================
    # ✅ NEW: Run validation after Check/Install mode completes
    # =========================================================================
    if install_mode == "check":
        print("\n")
        if not validate_installation(backend, embedding_model, tts_engine):
            print_status("Validation failed - some components may need reinstallation", False)
            retry = input("\nWould you like to perform a Clean Install to fix issues? (y/n): ").strip().lower()
            if retry == "y":
                print("\nRestarting with Clean Install mode...\n")
                install_mode = "clean"
                # Re-run critical installation steps
                if not create_venv():
                    sys.exit(1)
                if not install_python_deps(backend):
                    sys.exit(1)
                if not initialize_embedding_cache(embedding_model):
                    sys.exit(1)
                if tts_engine == "coqui":
                    if not install_coqui_tts():
                        sys.exit(1)
                if not download_extract_backend(backend):
                    sys.exit(1)
                # Validate again
                if not validate_installation(backend, embedding_model, tts_engine):
                    print_status("Validation still failed after clean install", False)

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