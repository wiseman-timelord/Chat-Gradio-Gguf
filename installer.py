# Script: installer.py - Installation script for Chat-Gradio-Gguf
# v2: Targets Windows 10-11 / Ubuntu 24-25 / Python 3.11-3.13 / Gradio 5.x / PyQt6
# Note: Uses sentence-transformers for embeddings (cross-platform, offline)
# Note: Uses PyQt6 WebEngine for custom browser window

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

# Constants / Variables
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
LLAMACPP_PYTHON_PREBUILT_VERSION = "v0.3.16"
#   LLAMACPP_PYTHON_VERSION — resolved at install time to the latest GitHub
#     release tag. Falls back to LLAMACPP_PYTHON_VERSION_FALLBACK if unreachable.
LLAMACPP_PYTHON_VERSION = None
LLAMACPP_PYTHON_VERSION_FALLBACK = "v0.3.20"
# Set during install_python_deps() once the wheel is confirmed installed.
# Written to constants.ini by update_ini_wheel_version() so the main program
# can display it in the About/Debug tab.
_INSTALLED_LLAMA_WHEEL_VERSION = None
LLAMACPP_TARGET_VERSION = "b8882"
WIN_COMPILE_TEMP = Path("C:/temp_build")
LINUX_COMPILE_TEMP = None
_INSTALL_PROCESSES = set()
_DID_COMPILATION = False
_PRE_EXISTING_PROCESSES = {}
_USER_BUILD_THREADS = None
PYTHON_VERSION = sys.version_info
WINDOWS_VERSION = None
_CPU_FEATURES = None
_CPU_DETECTED_EARLY = False
OS_VERSION = None
VS_GENERATOR = None

# Display/Browser variables
DX11_CAPABLE = None
DX_FEATURE_LEVEL = None
DX_FEATURE_NAME = None

# Maps/Lists
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
COQUI_ENGLISH_VOICE = {
    "id":      "p225,p226",
    "display": "English (Male/Female)",
    "accent":  "english",
}

PLATFORM = None

def set_platform() -> None:
    global PLATFORM
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ["windows", "linux"]:
        print("ERROR: Platform argument required (windows/linux)")
        sys.exit(1)
    PLATFORM = sys.argv[1].lower()

set_platform()

# Functions
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
            features["SSE3"] = True
            success = True

        if success:
            try:
                import cpuinfo as _cpuinfo
                _info  = _cpuinfo.get_cpu_info()
                _flags = [f.lower() for f in _info.get('flags', [])]
                features["FMA"]    = 'fma'  in _flags
                features["F16C"]   = 'f16c' in _flags
                features["AVX"]    = features["AVX"]    or ('avx'    in _flags)
                features["AVX2"]   = features["AVX2"]   or ('avx2'   in _flags)
                features["AVX512"] = features["AVX512"] or any('avx512' in f for f in _flags)
            except ImportError:
                pass

    else:  # Linux
        try:
            with open('/proc/cpuinfo', 'r') as f:
                content = f.read().lower()
            features["AVX"]    = 'avx'    in content
            features["AVX2"]   = 'avx2'   in content
            features["AVX512"] = 'avx512' in content
            features["FMA"]    = 'fma'    in content
            features["F16C"]   = 'f16c'   in content
            features["SSE3"]   = 'sse3'   in content or 'pni' in content
            features["SSSE3"]  = 'ssse3'  in content
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

    if PLATFORM != "windows":
        DX11_CAPABLE = True
        DX_FEATURE_LEVEL = 0xb000
        DX_FEATURE_NAME = "11.0"
        return (DX11_CAPABLE, DX_FEATURE_LEVEL)

    try:
        import ctypes

        d3d11 = ctypes.windll.LoadLibrary("d3d11.dll")
        feature_levels = (ctypes.c_uint * 4)(0xb100, 0xb000, 0xa100, 0xa000)
        device  = ctypes.c_void_p()
        fl_out  = ctypes.c_uint()
        ctx     = ctypes.c_void_p()

        hr = d3d11.D3D11CreateDevice(
            None, 1, None, 0, feature_levels, 4, 7,
            ctypes.byref(device), ctypes.byref(fl_out), ctypes.byref(ctx),
        )

        DX_FEATURE_LEVEL = fl_out.value
        DX_FEATURE_NAME  = {
            0xb100: "11.1", 0xb000: "11.0",
            0xa100: "10.1", 0xa000: "10.0"
        }.get(fl_out.value, f"0x{fl_out.value:04x}")
        DX11_CAPABLE = (hr == 0 and fl_out.value >= 0xb000)
        return (DX11_CAPABLE, DX_FEATURE_LEVEL)

    except:
        DX11_CAPABLE    = False
        DX_FEATURE_LEVEL = 0
        DX_FEATURE_NAME  = "Unknown"
        return (False, 0)


def run_initial_detection():
    run_detections_once()
    if _DETECTED_DX_LEVEL == 0:
        print_status("GPU acceleration: Not available (no DirectX 11)", False)
    else:
        print_status(f"GPU acceleration: DirectX {_DETECTED_DX_NAME}")
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
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-win-vulkan-x64.zip",
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
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_TARGET_VERSION}/llama-{LLAMACPP_TARGET_VERSION}-bin-ubuntu-vulkan-x64.tar.gz",
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

# =============================================================================
# v2 BASE REQUIREMENTS
# Targets: Python 3.11-3.13 / Gradio 5.x / PyQt6
#
# Removed from v1:
#   - numpy<2         (torch 2.2.2 compat pin — not needed for torch 2.5+)
#   - requests==2.31.0 (pin for Gradio 3.x compat — not needed)
#   - tokenizers==0.22.2 (pin for Python 3.9 compat — not needed)
#   - matplotlib>=3.7.0 (was needed by Gradio 3.x internals; Gradio 5 manages it)
#   - pyvirtualdisplay>=3.0 (Linux headless Qt5 — Qt6 runs natively on Ubuntu 24)
#   - newspaper3k (Python 3.9 fallback — min is now 3.11)
# =============================================================================
BASE_REQ = [
    "numpy>=2.0",
    "requests>=2.32.0",
    "pyperclip>=1.8.2",
    "spacy>=3.8.0",
    "psutil>=6.0.0",
    "ddgs>=9.10.0",
    "langchain-community>=0.3.18",
    "langchain-text-splitters>=0.3.0",
    "faiss-cpu>=1.9.0",
    "langchain>=0.3.18",
    "pygments>=2.17.0",
    "lxml>=5.2.0,<5.5.0",      # newspaper4k requires lxml<5.5; pin upfront to avoid downgrade
    "lxml_html_clean>=0.3.0",
    "beautifulsoup4>=4.12.0",
    "aiohttp>=3.10.0",
    "newspaper4k>=0.9.4.1",   # Python 3.11+ always; newspaper3k branch removed
]

if PLATFORM == "windows":
    BASE_REQ.extend([
        "pywin32>=306",
        "tk==0.1.0",
        "pythonnet==3.0.5",
    ])
# Linux: no platform-specific extras needed for v2 (pyvirtualdisplay removed)

def clear_screen():
    os.system('cls' if PLATFORM == "windows" else 'clear')

def backend_requires_compilation(backend: str) -> bool:
    """Check if the selected backend requires compilation"""
    info = BACKEND_OPTIONS.get(backend, {})
    return info.get("compile_binary", False) or info.get("compile_wheel", False)


def detect_windows_version() -> str:
    """Detect Windows version and cache in global.
    v2: Only Windows 10/11 are supported."""
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
        else:
            # Windows 7/8/8.1 — not supported in v2
            WINDOWS_VERSION = "unsupported"
        return WINDOWS_VERSION
    except:
        WINDOWS_VERSION = "unknown"
        return "unknown"


def detect_linux_version() -> str:
    """Detect Ubuntu version and cache in global OS_VERSION.
    v2: Only Ubuntu 24/25 are supported."""
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

def get_dynamic_requirements() -> list:
    """
    Build requirements list for Gradio 5.x.

    v2 vs v1:
    - Gradio 5.x uses Pydantic v2 natively — no shim, no forced pydantic pin.
    - fastapi, starlette, websockets, jinja2 are all managed by Gradio 5.x itself.
    - CRITICAL_PINNED set is gone entirely — no force-reinstall phase needed.
    """
    requirements = BASE_REQ.copy()
    # Gradio 5.x — upper-bound at <6.0 for stability
    requirements.append("gradio>=5.0.0,<6.0.0")
    return requirements


def get_torch_version_for_python() -> str:
    """
    v2: All supported Python versions (3.11-3.13) use torch>=2.5.0 unified.
    No two-tier split needed.
    """
    return "torch>=2.5.0"

def check_version_compatibility():
    """Check Python and OS compatibility.
    v2: Minimum Python 3.11, Windows 10, Ubuntu 24."""
    global WINDOWS_VERSION, PYTHON_VERSION, PLATFORM

    if sys.version_info < (3, 11):
        print_status("Python ≥3.11 required for Chat-Gradio-Gguf v2", False)
        return False

    PYTHON_VERSION = sys.version_info

    if PLATFORM == "windows":
        win_ver = detect_windows_version()
        if win_ver in ("unsupported", "7", "8", "8.1"):
            print_status(
                f"Windows {win_ver} is not supported in v2. Requires Windows 10 or 11.", False
            )
            return False
        return True

    else:  # Linux
        try:
            with open("/etc/os-release") as f:
                content = f.read()
            if "UBUNTU_VERSION_ID" in content or "ubuntu" in content.lower():
                version_match = re.search(r'VERSION_ID="?([0-9\.]+)"?', content)
                if version_match:
                    ubuntu_version = version_match.group(1)
                    major = int(ubuntu_version.split('.')[0])
                    if major >= 24:
                        return True
                    print_status(
                        f"Ubuntu {ubuntu_version} is not supported in v2. Requires Ubuntu 24 or 25.", False
                    )
                    return False
            print_status("Could not determine Ubuntu version", False)
            return False
        except Exception as e:
            print_status(f"OS detection failed: {e}", False)
            return False


def is_coqui_compatible() -> bool:
    """Check if current OS supports Coqui TTS.
    v2: Windows 10/11 and Ubuntu 24/25 both fully support Coqui.
    Windows 7/8/8.1 and Ubuntu 22/23 support removed."""
    if PLATFORM == "windows":
        # v2: only Win10/11 survive check_version_compatibility(),
        # so if we reach here we're on a supported Windows.
        return WINDOWS_VERSION in ["10", "11"]

    elif PLATFORM == "linux":
        try:
            if OS_VERSION:
                major_version = int(OS_VERSION.split('.')[0])
                return major_version >= 24
        except (ValueError, AttributeError, IndexError):
            pass
        return False

    return False


# =============================================================================
# INSTALLATION HELPERS — unchanged from v1 unless noted
# =============================================================================

def snapshot_pre_existing_processes() -> None:
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
    except Exception:
        pass


def track_process(pid: int) -> None:
    _INSTALL_PROCESSES.add(pid)


def cleanup_build_processes() -> None:
    if PLATFORM != "windows":
        return
    try:
        import psutil
    except ImportError:
        return

    build_process_names = [
        "MSBuild.exe", "VBCSCompiler.exe", "cmake.exe", "cl.exe",
        "link.exe", "lib.exe", "cvtres.exe", "mt.exe", "rc.exe",
        "mspdbsrv.exe", "vctip.exe", "tracker.exe",
    ]

    for pid in list(_INSTALL_PROCESSES):
        try:
            proc = psutil.Process(pid)
            if proc.is_running():
                proc.terminate()
        except Exception:
            pass

    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] in build_process_names:
                if proc.info['pid'] not in _PRE_EXISTING_PROCESSES:
                    proc.terminate()
        except Exception:
            pass


def _force_rmtree(path: Path) -> None:
    """Delete a directory tree, forcibly removing read-only files (Windows git repos).
    shutil.rmtree(ignore_errors=True) silently fails on read-only .git files;
    this handler chmod's each file before retrying the delete."""
    import stat

    def _on_error(func, fpath, exc_info):
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception:
            pass

    if path.exists():
        shutil.rmtree(str(path), onerror=_on_error)


def get_optimal_build_threads() -> int:
    """Return 85% of logical CPU cores for parallel builds. Always auto — no user prompt."""
    if _USER_BUILD_THREADS is not None:
        return _USER_BUILD_THREADS
    import multiprocessing
    try:
        total = multiprocessing.cpu_count()
    except:
        total = 4
    return max(1, int(total * 0.85))


# =============================================================================
# MODULE-LEVEL DETECTION CACHE
# Populated once by run_detections_once() at installer start.
# All subsequent code reads these globals instead of re-detecting.
# =============================================================================
_DETECTED_CPU_FEATURES: dict  = {}
_DETECTED_BUILD_TOOLS:  dict  = {}
_DETECTED_VULKAN:       bool  = False
_DETECTED_DX_CAPABLE:   bool  = False
_DETECTED_DX_LEVEL:     int   = 0
_DETECTED_DX_NAME:      str   = "Unknown"
_DETECTIONS_RUN:        bool  = False


def run_detections_once() -> None:
    """Run all hardware/tool detections exactly once and cache results in globals.
    Each detection is wrapped individually — one failure cannot block the others."""
    global _DETECTED_CPU_FEATURES, _DETECTED_BUILD_TOOLS, _DETECTED_VULKAN
    global _DETECTED_DX_CAPABLE, _DETECTED_DX_LEVEL, _DETECTED_DX_NAME, _DETECTIONS_RUN

    if _DETECTIONS_RUN:
        return

    try:
        _DETECTED_CPU_FEATURES = detect_cpu_features()
    except Exception:
        _DETECTED_CPU_FEATURES = {}

    try:
        _DETECTED_BUILD_TOOLS = detect_build_tools_available()
    except Exception:
        _DETECTED_BUILD_TOOLS = {"Git": False, "CMake": False, "MSVC": False, "MSBuild": False}

    try:
        _DETECTED_VULKAN = is_vulkan_installed()
    except Exception:
        _DETECTED_VULKAN = False

    try:
        _DETECTED_DX_CAPABLE, _DETECTED_DX_LEVEL = detect_browser_acceleration()
        _DETECTED_DX_NAME = DX_FEATURE_NAME or "Unknown"
    except Exception:
        _DETECTED_DX_CAPABLE  = False
        _DETECTED_DX_LEVEL    = 0
        _DETECTED_DX_NAME     = "Unknown"

    _DETECTIONS_RUN = True
    print(f"[DETECT] CPU: {', '.join(k for k,v in _DETECTED_CPU_FEATURES.items() if v) or 'baseline'}")
    print(f"[DETECT] Vulkan: {'YES' if _DETECTED_VULKAN else 'NO'} | DX: {_DETECTED_DX_NAME}")
    print(f"[DETECT] Build tools: {', '.join(k for k,v in _DETECTED_BUILD_TOOLS.items() if v) or 'none'}")

def print_header(section: str = "Installer") -> None:
    clear_screen()
    width = shutil.get_terminal_size().columns - 1
    print("=" * width)
    print(f" Chat-Gradio-Gguf v2 — {section}")
    print("=" * width)
    print()


def create_files_and_directories(backend: str) -> None:
    for directory in DIRECTORIES:
        dir_path = BASE_DIR / directory
        if str(dir_path) in [str(BASE_DIR / p) for p in PROTECTED_DIRECTORIES]:
            if dir_path.exists():
                print_status(f"Protected directory preserved: {directory}")
                continue
        dir_path.mkdir(parents=True, exist_ok=True)
    print_status("Directories created/verified")


# =============================================================================
# EMBEDDING BACKEND INSTALLATION
# v2: Unified torch>=2.5.0 for Python 3.11-3.13 (no two-tier split)
# =============================================================================

def install_embedding_backend() -> bool:
    """Install PyTorch CPU and sentence-transformers.
    v2: Unified path — torch>=2.5.0 supports Python 3.11-3.13 equally."""
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("python.exe" if PLATFORM == "windows" else "python"))
    pip_exe    = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("pip.exe"    if PLATFORM == "windows" else "pip"))

    torch_req                   = "torch>=2.5.0"
    transformers_version        = "transformers>=4.44.0"
    sentence_transformers_version = "sentence-transformers>=3.3.0"

    # 1. Install PyTorch using the CPU index
    cpu_index = "https://download.pytorch.org/whl/cpu"
    print_status(f"Installing PyTorch (CPU) — {torch_req}...")
    if not pip_install_with_retry(pip_exe, torch_req,
                                  ["--index-url", cpu_index,
                                   "--upgrade-strategy", "only-if-needed"],
                                  max_retries=10, initial_delay=5.0):
        print_status("PyTorch installation failed", False)
        return False
    print_status("PyTorch (CPU) installed")

    # PyTorch's CPU index ships its own pinned setuptools (e.g. 70.2.0) which
    # downgrades the venv's setuptools. Reinstall the latest to restore it.
    subprocess.run(
        [pip_exe, "install", "setuptools>=80.0", "--upgrade", "--quiet"],
        capture_output=True, timeout=120
    )
    print_status("setuptools restored after torch install")

    # 2. Install transformers
    print_status(f"Installing {transformers_version}...")
    if not pip_install_with_retry(pip_exe, transformers_version,
                                  ["--upgrade-strategy", "only-if-needed"],
                                  max_retries=10, initial_delay=5.0):
        print_status("transformers installation failed", False)
        return False
    print_status("transformers installed")

    # 3. Install sentence-transformers
    print_status(f"Installing {sentence_transformers_version}...")
    if not pip_install_with_retry(pip_exe, sentence_transformers_version,
                                  ["--upgrade-strategy", "only-if-needed"],
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


# =============================================================================
# QT WEBENGINE INSTALLATION
# v2: PyQt6 + PyQt6-WebEngine (replaces PyQt5 + PyQtWebEngine)
# =============================================================================

def install_qt_webengine() -> bool:
    """Install PyQt6 + PyQt6-WebEngine for the custom browser window.
    v2: Qt6 is used on all supported platforms (Windows 10-11, Ubuntu 24-25).
    PyQt6-WebEngine wheels bundle Qt automatically — no system Qt needed."""
    print_status("Installing Qt6 WebEngine for custom browser...")
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                 ("pip.exe" if PLATFORM == "windows" else "pip"))

    try:
        # PyQt6 — binds Python to Qt6
        if not pip_install_with_retry(pip_exe, "PyQt6>=6.6.0", max_retries=3, initial_delay=5.0):
            print_status("PyQt6 installation failed - will use system browser", False)
            return False

        # PyQt6-WebEngine — Chromium-based browser widget
        if not pip_install_with_retry(pip_exe, "PyQt6-WebEngine>=6.6.0", max_retries=3, initial_delay=5.0):
            print_status("PyQt6-WebEngine installation failed - will use system browser", False)
            return False

        print_status("Qt6 WebEngine installed successfully")
        return True

    except Exception as e:
        print_status(f"Qt WebEngine error: {e} - will use system browser", False)
        return False


# =============================================================================
# LINUX SYSTEM DEPENDENCIES
# v2: Qt6 apt packages; xvfb and Qt5 packages removed
# =============================================================================

def install_linux_system_dependencies(backend: str) -> bool:
    """Install Linux system dependencies.
    v2: Ubuntu 24-25 only. Qt6 runtime packages, xvfb removed."""
    print_status("Installing Linux system dependencies...")

    # Base packages needed for Python and audio
    base_packages = [
        "python3-dev",
        "build-essential",
        "libffi-dev",
        "libssl-dev",
        # Qt6 XCB / Wayland runtime dependencies
        # (PyQt6 wheels bundle Qt itself; these are the host-side xcb libs they dlopen)
        "libegl1",
        "libgl1",
        "libxkbcommon0",
        "libxkbcommon-x11-0",
        "libxcb-cursor0",
        "libxcb-icccm4",
        "libxcb-image0",
        "libxcb-keysyms1",
        "libxcb-randr0",
        "libxcb-render-util0",
        "libxcb-shape0",
        "libxcb-xinerama0",
        "libxcb-xkb1",
        "libxcb1",
    ]

    # Optional Vulkan packages
    info = BACKEND_OPTIONS[backend]
    vulkan_packages = []
    if info.get("build_flags", {}).get("GGML_VULKAN"):
        vulkan_packages = [
            "vulkan-tools", "libvulkan-dev", "mesa-utils",
            "glslang-tools", "spirv-tools"
        ]
    elif "Vulkan" in backend:
        vulkan_packages = ["vulkan-tools", "libvulkan1"]

    try:
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(
            ["sudo", "apt-get", "install", "-y"] + list(set(base_packages)), check=True
        )
        print_status("Base dependencies installed")

        if vulkan_packages:
            print_status("Installing Vulkan dependencies...")
            for package in vulkan_packages:
                try:
                    subprocess.run(
                        ["sudo", "apt-get", "install", "-y", package],
                        capture_output=True, check=True
                    )
                    print_status(f"  Installed {package}")
                except subprocess.CalledProcessError:
                    print_status(f"  Optional package {package} failed (continuing)", False)

        print_status("Linux system dependencies installed")
        return True

    except subprocess.CalledProcessError as e:
        print_status(f"System dependency installation failed: {e}", False)
        return False


# =============================================================================
# PYTHON DEPENDENCY INSTALLATION
# v2: CRITICAL_PINNED section and phase 2 force-reinstall removed entirely.
# Gradio 5.x resolves its own deps; no manual overrides needed.
# =============================================================================

def install_python_deps(backend: str) -> bool:
    """Install Python dependencies.
    v2: Single-phase install — no critical pinned packages, no force-reinstall phase."""
    global _INSTALLED_LLAMA_WHEEL_VERSION
    print_status("Installing Python dependencies...")
    try:
        pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("pip.exe" if PLATFORM == "windows" else "pip"))

        all_requirements = get_dynamic_requirements()

        print_status(f"Installing Python packages...")
        total = len(all_requirements)
        for i, req in enumerate(all_requirements, 1):
            pkg_name = req.split('>=')[0].split('==')[0].split('[')[0]
            print(f"  [{i}/{total}] Installing {pkg_name}...  ", end='', flush=True)

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

        # Install Qt WebEngine for custom browser (non-fatal if it fails)
        install_qt_webengine()

        # llama-cpp-python installation
        info = BACKEND_OPTIONS[backend]

        if not info.get("compile_wheel"):
            wheel_version = LLAMACPP_PYTHON_PREBUILT_VERSION.lstrip("v")
            sources = _get_prebuilt_wheel_urls()

            if not sources:
                print_status("No pre-built wheel sources available for this platform.", False)
                return False

            print_status(f"Installing llama-cpp-python {wheel_version} (CPU, trying {len(sources)} sources)...")
            installed = False

            for src in sources:
                label = src.get("label", src["value"])
                print(f"  Trying: {label}")

                if src["type"] == "url":
                    installed = pip_install_with_retry(pip_exe, src["value"], max_retries=2, initial_delay=3.0)
                elif src["type"] == "index":
                    installed = pip_install_with_retry(
                        pip_exe, src["value"],
                        extra_args=["--extra-index-url", src["extra_index"], "--prefer-binary"],
                        max_retries=3, initial_delay=5.0
                    )
                elif src["type"] == "pypi":
                    installed = pip_install_with_retry(
                        pip_exe, src["value"],
                        extra_args=["--prefer-binary"],
                        max_retries=3, initial_delay=5.0
                    )

                if installed:
                    print_status(f"llama-cpp-python {wheel_version} installed via {label}")
                    _INSTALLED_LLAMA_WHEEL_VERSION = f"v{wheel_version}"
                    break
                else:
                    print(f"  Source unavailable: {label}")

            if not installed:
                print_status(f"llama-cpp-python {wheel_version} could not be installed from any prebuilt source.", False)
                return False
        else:
            build_flags = info.get("build_flags", {})

            if build_flags.get("GGML_VULKAN"):
                print_status("Vulkan wheel build - checking Vulkan SDK...")
                if not check_vulkan_sdk_installed():
                    print_status("Error: Vulkan SDK not found", False)
                    return False

            if PLATFORM == "windows" and not check_vcredist_windows():
                print_status("Warning: Visual C++ Redistributable (x64) not detected", False)
                time.sleep(3)

            if not build_llama_cpp_python_with_flags(build_flags):
                return False
            # Record the compiled wheel version
            _INSTALLED_LLAMA_WHEEL_VERSION = get_latest_llamacpp_python_version()

        print_status("Python dependencies installed successfully")
        return True

    except subprocess.CalledProcessError as e:
        print_status(f"Install failed: {e}", False)
        return False


def install_optional_file_support() -> bool:
    """Install optional file format libraries"""
    print_status("Installing optional file format support...")

    # v2: Updated versions
    optional_packages = [
        "PyPDF2>=3.0.0",
        "python-docx>=1.1.0",
        "openpyxl>=3.1.0",
        "python-pptx>=1.0.0",
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


# =============================================================================
# SYSTEM INI CREATION
# v2: qt_version = 6, gradio_version = 5.x
# =============================================================================

def create_system_ini(platform: str, os_version: str, python_version: str,
                      backend_type: str, embedding_model: str,
                      windows_version: str = None, vulkan_available: bool = False,
                      llama_cli_path: str = None, llama_bin_path: str = None,
                      tts_engine: str = "builtin", coqui_voice_id: str = None,
                      coqui_voice_accent: str = None,
                      browser_acceleration: bool = True,
                      dx_feature_level: int = 0):
    """Create constants.ini with platform, version, TTS, and compatibility information.
    v2: qt_version = 6, gradio_version = 5.x written unconditionally."""
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
            f.write(f"qt_version = 6\n")                  # v2: PyQt6
            f.write(f"dx_feature_level = {dx_feature_level}\n")
            f.write(f"gradio_version = 5.x\n")            # v2: Gradio 5.x

            if llama_cli_path:
                f.write(f"llama_cli_path = {llama_cli_path}\n")
            if llama_bin_path:
                f.write(f"llama_bin_path = {llama_bin_path}\n")
            if platform == "windows" and windows_version:
                f.write(f"windows_version = {windows_version}\n")

            f.write("\n[tts]\n")
            if coqui_voice_id:
                f.write(f"coqui_voice_id = {coqui_voice_id}\n")
                f.write(f"coqui_voice_accent = {coqui_voice_accent or 'english'}\n")
                f.write(f"coqui_model = tts_models/en/vctk/vits\n")

        print_status("System information file created")
        return True
    except Exception as e:
        print_status(f"Failed to create constants.ini: {str(e)}", False)
        return False


def update_ini_wheel_version(version: str) -> bool:
    """Patch constants.ini to record the llama-cpp-python wheel version.

    Called after install_python_deps() so the version is confirmed installed.
    Uses configparser to update the [system] section in-place, preserving all
    other keys and the [tts] section.
    """
    import configparser as _cp
    ini_path = BASE_DIR / "data" / "constants.ini"
    if not ini_path.exists():
        print_status("constants.ini not found — cannot record wheel version", False)
        return False
    try:
        cfg_ini = _cp.ConfigParser()
        cfg_ini.read(ini_path, encoding='utf-8')
        if 'system' not in cfg_ini:
            print_status("constants.ini missing [system] — cannot record wheel version", False)
            return False
        cfg_ini['system']['llama_wheel_version'] = version
        with open(ini_path, 'w', encoding='utf-8') as f:
            cfg_ini.write(f)
        print_status(f"Recorded llama-cpp-python wheel version: {version}")
        return True
    except Exception as e:
        print_status(f"Could not update wheel version in constants.ini: {e}", False)
        return False


def create_venv() -> bool:
    try:
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
            print_status("Removed existing virtual environment")

        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)

        print_status("Created new virtual environment")

        python_exe = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / \
                     ("python.exe" if PLATFORM == "windows" else "python")
        pip_exe    = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / \
                     ("pip.exe"    if PLATFORM == "windows" else "pip")

        if not python_exe.exists():
            raise FileNotFoundError(f"Python executable not found at {python_exe}")

        subprocess.run([str(pip_exe), "install", "--upgrade", "pip"],
                      capture_output=True, timeout=120)
        print_status("Upgraded pip to latest version")
        print_status("Verified virtual environment setup")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to create venv: {e}", False)
        return False


def ensure_venv() -> bool:
    if VENV_DIR.exists():
        print_status("Existing virtual environment found - skipping recreation")
        return True
    return create_venv()


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
    if PLATFORM == "windows":
        vulkan_sdk = os.environ.get("VULKAN_SDK")
        if vulkan_sdk and Path(vulkan_sdk).is_dir():
            return True
        default_sdk = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "VulkanSDK"
        if default_sdk.exists():
            for child in default_sdk.iterdir():
                if child.is_dir() and (child / "Bin" / "vulkaninfoSDK.exe").exists():
                    os.environ["VULKAN_SDK"] = str(child)
                    return True
        return False
    else:
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
    """Check if Vulkan runtime is installed on the system.
    Windows: checks registry ICD key, vulkaninfo on PATH, VulkanRT/VulkanSDK
             in Program Files, VULKAN_SDK env var, and vulkan-1.dll in system32.
    Linux:   checks vulkaninfo and ldconfig for libvulkan.
    """
    if PLATFORM == "windows":
        # 1. GPU driver ICD registration (most authoritative — set by NVIDIA/AMD/Intel drivers)
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"SOFTWARE\Khronos\Vulkan\Drivers")
            # Key exists AND has at least one value (registered driver)
            _, num_values, _ = winreg.QueryInfoKey(key)
            winreg.CloseKey(key)
            if num_values > 0:
                return True
        except Exception:
            pass

        # 2. vulkaninfo.exe on PATH (VulkanSDK adds itself to PATH at install)
        if shutil.which("vulkaninfo"):
            try:
                result = subprocess.run(
                    ["vulkaninfo", "--summary"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    timeout=8
                )
                if result.returncode == 0:
                    return True
            except Exception:
                pass

        # 3. VULKAN_SDK environment variable (set by VulkanSDK installer)
        vulkan_sdk_env = os.environ.get("VULKAN_SDK", "")
        if vulkan_sdk_env and Path(vulkan_sdk_env).is_dir():
            return True

        # 4. VulkanRT / VulkanSDK in Program Files (user's reported locations)
        pf = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
        for folder_name in ("VulkanRT", "VulkanSDK"):
            folder = pf / folder_name
            if folder.exists() and any(folder.iterdir()):
                return True

        # 5. vulkan-1.dll in system32 (VulkanRT copies it there)
        sys32 = Path(os.environ.get("SYSTEMROOT", r"C:\Windows")) / "System32"
        if (sys32 / "vulkan-1.dll").exists():
            return True

        return False

    else:
        try:
            result1 = subprocess.run(["vulkaninfo", "--summary"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            result2 = subprocess.run(["ldconfig", "-p"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return result1.returncode == 0 or b"libvulkan" in result2.stdout
        except FileNotFoundError:
            return False


def pip_install_with_retry(pip_exe: str, package: str, extra_args: list = None,
                           max_retries: int = 10, initial_delay: float = 5.0,
                           force_reinstall: bool = False, no_deps: bool = False) -> bool:
    """Install a pip package with retry logic and exponential backoff."""
    INACTIVITY_TIMEOUT = 300
    _PROGRESS_KEYWORDS = ("downloading", "installing", "collected", "building",
                          "error", "warning", "failed", "%")
    _SUPPRESS_WARNINGS = ("pip's dependency resolver does not currently take into account",)

    if extra_args is None:
        extra_args = []

    pkg_name = package.split(">=")[0].split("==")[0].split("[")[0]
    delay = initial_delay

    install_flags = []
    if force_reinstall:
        install_flags.append("--force-reinstall")
    if no_deps:
        install_flags.append("--no-deps")

    for attempt in range(max_retries):
        cmd = [pip_exe, "install"] + install_flags + [package] + extra_args
        all_output: list[str] = []
        last_activity = [time.time()]
        reader_done  = [False]
        stall_reason: list[str] = [None]

        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )

            def _read_output():
                try:
                    for raw_line in proc.stdout:
                        line = raw_line.rstrip()
                        if not line:
                            continue
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

            while not reader_done[0]:
                time.sleep(2)
                idle = time.time() - last_activity[0]
                if idle >= INACTIVITY_TIMEOUT:
                    stall_reason[0] = f"No output for {idle:.0f}s — connection stalled"
                    proc.kill()
                    break

            reader.join(timeout=5)
            proc.wait()

            combined = "\n".join(all_output).lower()

            if proc.returncode == 0 or "already satisfied" in combined:
                return True

            if stall_reason[0]:
                reason = stall_reason[0]
            else:
                error_lines = [l for l in all_output if "error" in l.lower()]
                reason = (f"pip error — {error_lines[-1][:120]}" if error_lines
                         else f"pip exited with code {proc.returncode}")

            if attempt < max_retries - 1:
                print(f"    Reason: {reason}")
                print(f"    Retry {attempt + 1}/{max_retries} for {pkg_name} in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 300)

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Unexpected error: {e}")
                print(f"    Retry {attempt + 1}/{max_retries} for {pkg_name} in {delay:.0f}s...")
                time.sleep(delay)
                delay = min(delay * 2, 300)

    return False


# =============================================================================
# PREBUILT WHEEL URL HELPERS — unchanged from v1
# =============================================================================

def _get_prebuilt_wheel_urls() -> list:
    """Return ordered list of prebuilt wheel sources to try."""
    wheel_version = LLAMACPP_PYTHON_PREBUILT_VERSION.lstrip("v")
    py_tag = _PY_TAG
    sources = []

    if PLATFORM == "windows":
        filename = f"llama_cpp_python-{wheel_version}-{py_tag}-{py_tag}-win_amd64.whl"
        sources.append({
            "type": "url",
            "label": f"eswarthammana/llama-cpp-wheels {wheel_version}",
            # FIXED: Repository name changed from llama-cpp-python-cpu to llama-cpp-wheels
            "value": f"https://github.com/eswarthammana/llama-cpp-wheels/releases/download/v{wheel_version}/{filename}",
        })
        sources.append({
            "type": "index",
            "label": f"abetlen CPU index {wheel_version}",
            "value": f"llama-cpp-python=={wheel_version}",
            "extra_index": "https://abetlen.github.io/llama-cpp-python/whl/cpu",
        })
    else:
        filename = f"llama_cpp_python-{wheel_version}-{py_tag}-{py_tag}-linux_x86_64.whl"
        sources.append({
            "type": "url",
            "label": f"eswarthammana/llama-cpp-wheels {wheel_version}",
            # FIXED: Repository name changed from llama-cpp-python-cpu to llama-cpp-wheels
            "value": f"https://github.com/eswarthammana/llama-cpp-wheels/releases/download/v{wheel_version}/{filename}",
        })
        sources.append({
            "type": "index",
            "label": f"abetlen CPU index {wheel_version}",
            "value": f"llama-cpp-python=={wheel_version}",
            "extra_index": "https://abetlen.github.io/llama-cpp-python/whl/cpu",
        })

    sources.append({
        "type": "pypi",
        "label": f"PyPI llama-cpp-python=={wheel_version} (--prefer-binary)",
        "value": f"llama-cpp-python=={wheel_version}",
    })

    return sources


def get_latest_llamacpp_python_version() -> str:
    global LLAMACPP_PYTHON_VERSION
    if LLAMACPP_PYTHON_VERSION is not None:
        return LLAMACPP_PYTHON_VERSION

    api_url = "https://api.github.com/repos/abetlen/llama-cpp-python/releases/latest"
    try:
        import urllib.request, json as _json
        req = urllib.request.Request(
            api_url,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": "Chat-Gradio-Gguf-Installer/2.0",
            }
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        tag = data.get("tag_name", "").strip()
        if tag and tag.startswith("v"):
            base_tag = re.sub(r'-(cu[0-9]+|metal|rocm[0-9.]*|sycl)$', '',
                              tag, flags=re.IGNORECASE)
            LLAMACPP_PYTHON_VERSION = base_tag
            return base_tag
    except Exception:
        pass

    LLAMACPP_PYTHON_VERSION = LLAMACPP_PYTHON_VERSION_FALLBACK
    return LLAMACPP_PYTHON_VERSION_FALLBACK


# =============================================================================
# LLAMA-CPP-PYTHON COMPILE — unchanged from v1
# =============================================================================

def build_llama_cpp_python_with_flags(build_flags: dict) -> bool:
    """Build llama-cpp-python from source with optimal CPU flags."""
    global _DID_COMPILATION
    _DID_COMPILATION = True

    snapshot_pre_existing_processes()
    print_status("Building llama-cpp-python from source (10-20 minutes)")

    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("python.exe" if PLATFORM == "windows" else "python"))
    pip_exe    = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("pip.exe"    if PLATFORM == "windows" else "pip"))

    build_threads = get_optimal_build_threads()
    print(f"  Using {build_threads} parallel build threads (85% of logical cores)")

    env = os.environ.copy()
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(build_threads)
    env["FORCE_CMAKE"] = "1"

    cpu_features = detect_cpu_features()
    if cpu_features.get("AVX2"):  build_flags["GGML_AVX2"] = "ON"
    if cpu_features.get("AVX"):   build_flags["GGML_AVX"]  = "ON"
    if cpu_features.get("FMA"):   build_flags["GGML_FMA"]  = "ON"
    if cpu_features.get("F16C"):  build_flags["GGML_F16C"] = "ON"

    build_flags["GGML_CURL"] = "OFF"
    build_flags["LLAMA_CURL"] = "OFF"
    build_flags["GGML_OPENMP"] = "ON"

    cmake_args = [f"-D{flag}={value}" for flag, value in build_flags.items()]
    env["CMAKE_ARGS"] = " ".join(cmake_args)
    for flag, value in build_flags.items():
        env[flag] = value

    if cmake_args:
        print(f"  Build flags: {', '.join(f'{k}={v}' for k, v in build_flags.items())}")

    if PLATFORM == "windows":
        env["CL"] = f"/MP{build_threads}"
        if VS_GENERATOR:
            env["CMAKE_GENERATOR"] = VS_GENERATOR
            env["CMAKE_GENERATOR_PLATFORM"] = "x64"
    else:
        env["MAKEFLAGS"] = f"-j{build_threads}"

    # Install build backend deps required by --no-build-isolation
    # scikit-build-core is the build backend declared in llama-cpp-python's pyproject.toml;
    # without it pip cannot locate the backend and raises BackendUnavailable.
    print_status("Installing build backend dependencies (scikit-build-core, cmake, ninja)...")
    build_backend_deps = [
        "scikit-build-core>=0.9.0",
        "cmake>=3.20",
        "ninja",
        "setuptools>=80.0",
        "wheel",
    ]
    for dep in build_backend_deps:
        pip_install_with_retry(pip_exe, dep, max_retries=5, initial_delay=3.0)
    print_status("Build backend dependencies installed")

    compile_version = get_latest_llamacpp_python_version()
    repo_dir = TEMP_DIR / "llama-cpp-python"

    # Progress keyword classification for clone output
    _PROGRESS_KW = ("counting", "compressing", "receiving", "resolving", "deltum")
    _INFO_KW     = ("cloning", "submodule", "registered", "checked out", "remote:", "total")
    _ERROR_KW    = ("error", "fatal", "warning")

    def _print_clone_line(line: str) -> None:
        """Print git clone output: progress lines reuse the same terminal line."""
        ll = line.lower()
        if any(kw in ll for kw in _ERROR_KW):
            print(f"\n  [!] {line[:120]}", flush=True)
        elif any(kw in ll for kw in _PROGRESS_KW):
            # Trim long lines and overwrite the same terminal line
            print(f"\r  {line[:100]:<100}", end="", flush=True)
        elif any(kw in ll for kw in _INFO_KW):
            print(f"\n  {line}", flush=True)

    try:
        # Force-clean any leftover build directory before starting
        if repo_dir.exists():
            _force_rmtree(repo_dir)

        print_status(f"Cloning llama-cpp-python {compile_version}...")

        max_retries = 5
        retry_delay = 10
        clone_success = False

        for attempt in range(max_retries):
            # Force-clean before each attempt (handles partial/failed prior clones)
            if repo_dir.exists():
                _force_rmtree(repo_dir)

            clone_proc = subprocess.Popen(
                ["git", "clone", "--progress", "--depth", "1",
                 "--branch", compile_version,
                 "--recurse-submodules", "--shallow-submodules",
                 LLAMACPP_PYTHON_GIT_REPO, str(repo_dir)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env
            )

            for raw in clone_proc.stdout:
                line = raw.rstrip("\n\r")
                if line:
                    _print_clone_line(line)

            clone_proc.wait()
            print()  # newline after in-place progress
            if clone_proc.returncode == 0:
                clone_success = True
                break
            else:
                if attempt < max_retries - 1:
                    print(f"  Clone failed, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 120)

        if not clone_success:
            print_status("Failed to clone llama-cpp-python after retries", False)
            return False

        print_status("Repository cloned — building wheel...")

        build_start = time.time()
        HEARTBEAT_INTERVAL = 30

        process = subprocess.Popen(
            [pip_exe, "install", "--no-build-isolation", repo_dir],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env
        )
        track_process(process.pid)

        full_output = []
        last_print_time = [time.time()]
        current_phase   = [None]

        def _phase_for(line_s):
            if "building" in line_s:    return "Building..."
            if "compiling" in line_s:   return "Compiling..."
            if "linking" in line_s:     return "Linking..."
            if "installing" in line_s:  return "Installing..."
            return None

        for line in process.stdout:
            line_s = line.rstrip()
            if not line_s:
                continue
            full_output.append(line_s)
            now   = time.time()
            phase = _phase_for(line_s.lower())
            is_error = any(x in line_s.lower() for x in ["error", "fatal", "failed"])

            if is_error:
                print(f"\n  [!] {line_s[:120]}", flush=True)
                last_print_time[0] = now
                current_phase[0]   = None
            elif phase:
                if phase != current_phase[0]:
                    elapsed = int(now - build_start)
                    print(f"\n  [{elapsed:>4}s] {phase}", end="", flush=True)
                    current_phase[0]   = phase
                    last_print_time[0] = now

            if now - last_print_time[0] >= HEARTBEAT_INTERVAL:
                elapsed = int(now - build_start)
                print(f"\n  [{elapsed:>4}s] Still building... (please wait)", end="", flush=True)
                last_print_time[0] = now

        print(flush=True)
        process.wait()

        if process.returncode == 0:
            print_status("llama-cpp-python built and installed")
            return True
        else:
            print_status("llama-cpp-python build failed", False)
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
            _force_rmtree(repo_dir)


# =============================================================================
# COQUI TTS INSTALLATION — unchanged from v1
# =============================================================================

def install_espeak_ng_windows():
    """Extract espeak-ng to project data folder (Windows only)."""
    import platform
    import urllib.request

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

    is_64bit  = platform.machine().endswith('64')
    msi_url   = ("https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi"
                 if is_64bit else
                 "https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X86.msi")
    msi_filename = "espeak-ng-X64.msi" if is_64bit else "espeak-ng-X86.msi"

    msi_path    = TEMP_DIR / msi_filename
    extract_dir = TEMP_DIR / "espeak_extract"

    try:
        if extract_dir.exists():
            shutil.rmtree(extract_dir)

        urllib.request.urlretrieve(msi_url, str(msi_path))

        if not msi_path.exists():
            print_status("ERROR: espeak-ng download failed", False)
            sys.exit(1)

        result = subprocess.run(
            ["msiexec", "/a", str(msi_path), "/qn", f"TARGETDIR={str(extract_dir)}"],
            capture_output=True, timeout=120
        )

        if result.returncode not in [0, 3010]:
            try:
                result = subprocess.run(
                    ["7z", "x", str(msi_path), f"-o{str(extract_dir)}", "-y"],
                    capture_output=True, timeout=60
                )
                if result.returncode != 0:
                    print_status(f"ERROR: Extraction failed", False)
                    sys.exit(1)
            except FileNotFoundError:
                print_status("ERROR: msiexec failed and 7z not available", False)
                sys.exit(1)

        source_dir = None
        for root, dirs, files in os.walk(extract_dir):
            if "espeak-ng.exe" in files:
                source_dir = Path(root)
                break

        if not source_dir:
            print_status("ERROR: espeak-ng.exe not found in extracted MSI", False)
            sys.exit(1)

        for item in source_dir.iterdir():
            dest = espeak_dir / item.name
            if item.is_dir():
                if dest.exists():
                    shutil.rmtree(dest)
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        if not espeak_dll.exists():
            print_status("ERROR: libespeak-ng.dll missing after extraction", False)
            sys.exit(1)
        if not espeak_exe.exists():
            print_status("ERROR: espeak-ng.exe missing after extraction", False)
            sys.exit(1)

        print_status(f"espeak-ng installed ({len(list(espeak_dir.glob('**/*')))} files)")
        return True

    except subprocess.TimeoutExpired:
        print_status("ERROR: espeak-ng installation timed out", False)
        sys.exit(1)
    except Exception as e:
        print_status(f"ERROR: espeak-ng installation failed: {e}", False)
        sys.exit(1)
    finally:
        msi_path.unlink(missing_ok=True)
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)


def install_coqui_tts():
    """Install Coqui TTS (Idiap fork) with codec support and download VCTK model.
    
    Patches the autoregressive module to replace missing 'isin_mps_friendly' import.
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
        print_status("Installing torchaudio (CPU-only to match torch)...")
        if not pip_install_with_retry(pip_exe, "torchaudio",
                                       ["--index-url", "https://download.pytorch.org/whl/cpu"],
                                       max_retries=10, initial_delay=5.0):
            print_status("torchaudio installation failed", False)
            sys.exit(1)
        print_status("torchaudio (CPU) installed")
        
        # Install coqui-tts[codec]
        result = subprocess.run(
            [pip_exe, "install", "coqui-tts[codec]"],
            capture_output=True, text=True, timeout=600
        )
        
        if result.returncode != 0:
            error_detail = result.stderr[-800:] if len(result.stderr) > 800 else result.stderr
            print_status(f"Coqui TTS pip install failed: {error_detail}", False)
            sys.exit(1)
        
        print_status("Coqui TTS package installed")

        # coqui-tts[codec] should pull torchcodec but sometimes fails silently.
        # Install it explicitly so TTS/__init__.py import check always passes.
        print_status("Ensuring torchcodec is installed (required by Coqui TTS)...")
        result_tc = subprocess.run(
            [pip_exe, "install", "torchcodec"],
            capture_output=True, text=True, timeout=300
        )
        if result_tc.returncode != 0:
            print_status(f"torchcodec install failed: {result_tc.stderr[-400:]}", False)
            sys.exit(1)
        print_status("torchcodec installed")

        # ---- PATCH Coqui TTS to fix transformers compatibility ----
        # Determine site-packages directory deterministically
        if PLATFORM == "windows":
            site_packages = VENV_DIR / "Lib" / "site-packages"
        else:
            py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
            site_packages = VENV_DIR / "lib" / py_version / "site-packages"
        
        if not site_packages.exists():
            print_status(f"CRITICAL: site-packages not found at {site_packages}", False)
            sys.exit(1)
        
        autoregressive_path = site_packages / "TTS" / "tts" / "layers" / "tortoise" / "autoregressive.py"
        if not autoregressive_path.exists():
            print_status(f"CRITICAL: autoregressive.py not found at {autoregressive_path}", False)
            sys.exit(1)
        
        with open(autoregressive_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        old_line = "from transformers.pytorch_utils import isin_mps_friendly as isin"
        new_import = (
            "try:\n"
            "    from transformers.pytorch_utils import isin_mps_friendly as isin\n"
            "except ImportError:\n"
            "    import torch\n"
            "    def isin(ar1, ar2):\n"
            "        return torch.isin(ar1, ar2)\n"
        )
        
        if old_line in content:
            new_content = content.replace(old_line, new_import)
            with open(autoregressive_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print_status(f"Patched {autoregressive_path.name} for transformers compatibility")
        else:
            # Maybe already patched? Check for our custom lines
            if "def isin(ar1, ar2)" in content:
                print_status("autoregressive.py already patched")
            else:
                print_status("CRITICAL: Expected line not found in autoregressive.py", False)
                sys.exit(1)
        
        # ---- Download and verify VCTK model ----
        tts_model_dir = BASE_DIR / "data" / "tts_models"
        tts_model_dir.mkdir(parents=True, exist_ok=True)
        
        print_status("Downloading Coqui VCTK voice model (~1.4GB)...")
        
        tts_model_dir_safe = str(tts_model_dir).replace("\\", "/")
        temp_wav_safe = str(TEMP_DIR / "tts_test.wav").replace("\\", "/")
        
        if PLATFORM == "windows":
            espeak_local_path = str(BASE_DIR / "data" / "espeak-ng").replace("\\", "/")
            
            download_script = f'''
import os
import sys

local_espeak = r"{espeak_local_path}"

# Add espeak-ng directory to PATH for DLL dependencies
current_path = os.environ.get("PATH", "")
if local_espeak not in current_path:
    os.environ["PATH"] = local_espeak + os.pathsep + current_path

# Set phonemizer environment variables
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
            # Linux version (unchanged but keep for completeness)
            download_script = f'''
import os
import sys
import subprocess

espeak_lib = None
espeak_exe = None

try:
    result = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "libespeak-ng.so.1" in line and "=>" in line:
            espeak_lib = line.split("=>")[-1].strip()
            break
except Exception:
    pass

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


# =============================================================================
# EMBEDDING CACHE INITIALIZATION — unchanged from v1
# =============================================================================

def initialize_embedding_cache(embedding_model: str) -> bool:
    """Initialize embedding model cache using sentence-transformers"""
    print_status(f"Initializing embedding cache for {embedding_model}...")

    cache_dir = BASE_DIR / "data" / "embedding_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("python.exe" if PLATFORM == "windows" else "python"))

    # Use a more verbose initialization script with progress output
    init_script = f'''
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_DEVICE"] = "cpu"

cache_dir = Path(r"{str(cache_dir)}")
os.environ["TRANSFORMERS_CACHE"] = str(cache_dir.absolute())
os.environ["HF_HOME"] = str(cache_dir.parent.absolute())
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir.absolute())

# Enable progress bars
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

print("Importing torch...", flush=True)
import torch
torch.set_grad_enabled(False)
print(f"torch version: {{torch.__version__}}", flush=True)
print(f"CUDA available: {{torch.cuda.is_available()}} (should be False)", flush=True)

print("Importing sentence_transformers...", flush=True)
from sentence_transformers import SentenceTransformer

print(f"Loading model: {embedding_model}", flush=True)
print("This may take several minutes while downloading model files...", flush=True)
sys.stdout.flush()

# Load model with progress bar enabled
model = SentenceTransformer(
    "{embedding_model}", 
    device="cpu",
    cache_folder=str(cache_dir.absolute())
)
model.eval()

print("Testing embedding...", flush=True)
test_embedding = model.encode(
    ["test"], 
    batch_size=1, 
    normalize_embeddings=True,
    convert_to_tensor=True,
    show_progress_bar=True
)
dim = test_embedding.shape[1] if len(test_embedding.shape) > 1 else len(test_embedding)
print(f"SUCCESS: Model loaded, dimension: {{dim}}", flush=True)
'''

    script_path = TEMP_DIR / "init_embedding.py"
    try:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(init_script)

        print("Embedding Initialization Output...")
        print("(This may take 2-10 minutes depending on download speed)")
        
        # Run with timeout and capture output in real-time
        process = subprocess.Popen(
            [python_exe, str(script_path)],
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            bufsize=1,
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in process.stdout:
            print(f"    {line.rstrip()}")
        
        process.wait(timeout=1200)  # 20 minutes timeout for large models
        
        script_path.unlink(missing_ok=True)

        if process.returncode == 0:
            # Verify cache was populated
            cache_files = list(cache_dir.rglob("*"))
            if len(cache_files) > 10:
                print_status(f"Embedding cache initialized ({len(cache_files)} files)")
                return True
            else:
                print_status("Embedding cache appears incomplete", False)
                return False
        else:
            print_status("Embedding initialization failed", False)
            return False

    except subprocess.TimeoutExpired:
        process.kill()
        print_status("Embedding initialization timed out after 20 minutes", False)
        print_status("Check your internet connection and try again", False)
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
        url      = (f"https://github.com/explosion/spacy-models/releases/download/"
                    f"en_core_web_sm-3.8.0/{filename}")
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

    except Exception as e:
        print_status(f"spaCy error: {e}", False)
        return False


def download_with_progress(url: str, filepath: Path, description: str = "Downloading",
                          max_retries: int = 10, initial_delay: float = 5.0) -> None:
    """Download file with progress bar and retry."""
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                    ("python.exe" if PLATFORM == "windows" else "python"))

    download_script = f'''
import requests, time, sys
from pathlib import Path

def format_bytes(b):
    for unit in ["B","KB","MB","GB"]:
        if b < 1024.0: return f"{{b:.1f}}{{unit}}"
        b /= 1024.0
    return f"{{b:.1f}}TB"

def progress_bar(current, total, width=30):
    if total == 0: return "[" + "=" * width + "] 100%"
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
        headers = {{"Range": f"bytes={{existing_size}}-"}} if existing_size > 0 else {{}}
        response = requests.get("{url}", stream=True, headers=headers, timeout=60)
        if response.status_code == 416:
            print(f"\\r{description}: Already complete", flush=True); break
        elif response.status_code == 206:
            total_size = existing_size + int(response.headers.get("content-length", 0))
        elif response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))
            existing_size = 0; filepath.unlink(missing_ok=True)
        else:
            response.raise_for_status()
        downloaded = existing_size
        last_update = 0
        with open(filepath, "ab" if existing_size > 0 else "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk); downloaded += len(chunk)
                    if total_size > 0:
                        pct = int(100 * downloaded / total_size)
                        if pct > last_update:
                            print(f"\\r{description}: {{progress_bar(downloaded, total_size)}}", end="", flush=True)
                            last_update = pct
        print(f"\\r{description}: {{progress_bar(total_size, total_size)}} - Complete", flush=True)
        break
    except Exception as e:
        print(f"\\n{description}: Error - {{e}}", flush=True)
        if attempt < max_retries - 1:
            print(f"{description}: Retry {{attempt+1}}/{{max_retries}} in {{delay:.0f}}s...", flush=True)
            time.sleep(delay); delay = min(delay * 2, 300)
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


# =============================================================================
# BINARY DOWNLOAD/EXTRACT — unchanged from v1
# =============================================================================

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
        subprocess.run(["git", "config", "--global", "http.lowSpeedLimit", "200"],
                      capture_output=True)
        subprocess.run(["git", "config", "--global", "http.lowSpeedTime",  "240"],
                      capture_output=True)

        if llamacpp_src.exists():
            _force_rmtree(llamacpp_src)

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

                for raw in process.stdout:
                    line = raw.rstrip("\n\r")
                    if not line:
                        continue
                    ll = line.lower()
                    if any(x in ll for x in ["error", "fatal"]):
                        print(f"\n  [!] {line[:120]}", flush=True)
                    elif any(x in ll for x in ["receiving", "resolving", "counting", "compressing"]):
                        print(f"\r  {line[:100]:<100}", end="", flush=True)
                    elif any(x in ll for x in ["cloning", "remote:", "total"]):
                        print(f"\n  {line}", flush=True)

                process.wait(timeout=600)
                print()
                if process.returncode == 0:
                    break
                else:
                    raise subprocess.CalledProcessError(process.returncode, "git clone")

            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                if llamacpp_src.exists():
                    _force_rmtree(llamacpp_src)
                if attempt < max_retries - 1:
                    print(f"\n  Clone failed, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print_status("Failed to clone llama.cpp after retries", False)
                    return False

        print_status("Repository cloned")

        build_dir = llamacpp_src / "build"
        build_dir.mkdir(exist_ok=True)

        cmake_args = [
            "cmake", "..",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_BUILD_PARALLEL_LEVEL={build_threads}",
            "-DLLAMA_CURL=OFF",
        ]

        cpu_features = detect_cpu_features()
        if cpu_features.get("AVX"):   cmake_args.append("-DGGML_AVX=ON")
        if cpu_features.get("AVX2"):  cmake_args.append("-DGGML_AVX2=ON")
        if cpu_features.get("FMA"):   cmake_args.append("-DGGML_FMA=ON")
        if cpu_features.get("F16C"):  cmake_args.append("-DGGML_F16C=ON")

        build_flags = info.get("build_flags", {})
        if build_flags.get("GGML_VULKAN"):
            cmake_args.append("-DGGML_VULKAN=ON")

        if PLATFORM == "windows":
            if VS_GENERATOR:
                cmake_args.extend(["-G", VS_GENERATOR, "-A", "x64"])
            else:
                cmake_args.extend(["-A", "x64"])

        print_status("Configuring build...")
        result = subprocess.run(cmake_args, cwd=build_dir, capture_output=True, text=True, env=env)
        if result.returncode != 0:
            print_status(f"CMake configure failed: {result.stderr[:200]}", False)
            return False

        print_status("Building binaries...")
        if PLATFORM == "windows":
            build_cmd = ["cmake", "--build", ".", "--config", "Release",
                         "--parallel", str(build_threads)]
        else:
            build_cmd = ["cmake", "--build", ".", "--parallel", str(build_threads)]

        process = subprocess.Popen(
            build_cmd, cwd=build_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env
        )
        track_process(process.pid)

        bin_build_start = time.time()
        last_bin_print  = [time.time()]
        HEARTBEAT_INTERVAL = 30

        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            now = time.time()
            is_error    = any(x in line.lower() for x in ["error", "fatal", "failed"])
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
                print(f"\n  [{elapsed:>4}s] Still building...", end="", flush=True)
                last_bin_print[0] = now

        process.wait()
        print()

        if process.returncode != 0:
            print_status("Build failed", False)
            return False

        print_status("Copying binaries...")
        if PLATFORM == "windows":
            bin_src = build_dir / "bin" / "Release"
        else:
            bin_src = build_dir / "bin"

        if not bin_src.exists():
            bin_src = build_dir

        for item in bin_src.iterdir():
            if item.is_file():
                shutil.copy2(item, dest_path / item.name)

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
            _force_rmtree(llamacpp_src)


def copy_linux_binaries(src_path: Path, dest_path: Path) -> None:
    subdirs = [d for d in src_path.iterdir() if d.is_dir()]
    if len(subdirs) == 1 and subdirs[0].name.startswith("llama-"):
        actual_src = subdirs[0]
        if (actual_src / "bin").exists():
            actual_src = actual_src / "bin"
    elif (src_path / "bin").exists():
        actual_src = src_path / "bin"
    else:
        actual_src = src_path

    copied_count = 0
    for item in actual_src.iterdir():
        if item.is_file():
            dest = dest_path / item.name
            shutil.copy2(item, dest)
            if not item.suffix or item.suffix in ['.so']:
                os.chmod(dest, 0o755)
            copied_count += 1

    lib_candidates = [src_path / "lib",
                      subdirs[0] / "lib" if len(subdirs) == 1 else None]
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
    import zipfile, tarfile

    info = BACKEND_OPTIONS[backend]

    if info.get("compile_binary"):
        return compile_llama_cpp_binary(backend, info)

    if not info["url"]:
        print_status("No backend download required for this option")
        return True

    print_status("Downloading backend binaries...")
    url = info["url"]
    if url.endswith(".tar.gz"):
        temp_archive = TEMP_DIR / "backend.tar.gz"
        is_tarball   = True
    else:
        temp_archive = TEMP_DIR / "backend.zip"
        is_tarball   = False

    try:
        download_with_progress(url, temp_archive, "Downloading backend")
        print_status("Extracting backend...")

        dest_path = BASE_DIR / info["dest"]
        dest_path.mkdir(parents=True, exist_ok=True)

        if is_tarball:
            with tarfile.open(temp_archive, 'r:gz') as tf:
                members = tf.getmembers()
                total   = len(members)
                for i, member in enumerate(members):
                    tf.extract(member, dest_path,
                               filter='data' if sys.version_info >= (3, 12) else None)
                    if i % 25 == 0 or i == total - 1:
                        print(f"\rExtracting: {simple_progress_bar(i + 1, total)}", end='', flush=True)
                print()
        else:
            with zipfile.ZipFile(temp_archive, 'r') as zf:
                members = zf.namelist()
                total   = len(members)
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
        temp_archive.unlink(missing_ok=True)


def clean_compile_temp() -> None:
    if PLATFORM == "windows" and WIN_COMPILE_TEMP.exists():
        try:
            _force_rmtree(WIN_COMPILE_TEMP)
            print_status("Cleaned up compilation temp folder")
        except:
            pass


# =============================================================================
# BUILD TOOL DETECTION — unchanged from v1
# =============================================================================

def _path_prepend(bin_dir: str) -> None:
    current_entries = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in current_entries:
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"


def _vs_base_dirs():
    pf   = Path(os.environ.get("ProgramFiles",      r"C:\Program Files"))
    pf86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    vs_pf   = pf   / "Microsoft Visual Studio"
    vs_pf86 = pf86 / "Microsoft Visual Studio"
    return [
        ("2022", vs_pf,   vs_pf86),
        ("2019", vs_pf86, vs_pf),
        ("2017", vs_pf86, vs_pf),
    ]


def detect_build_tools_available() -> dict:
    global VS_GENERATOR
    tools = {"Git": False, "CMake": False, "MSVC": False, "MSBuild": False}

    if shutil.which("git"):
        tools["Git"] = True

    if PLATFORM == "windows":
        _EDITIONS      = ["Community", "Professional", "Enterprise", "BuildTools"]
        _VS_GENERATORS = {"2022": "Visual Studio 17 2022",
                          "2019": "Visual Studio 16 2019",
                          "2017": "Visual Studio 15 2017"}
        _MSBUILD_SUBPATH = {"2022": Path("MSBuild")/"Current"/"Bin"/"MSBuild.exe",
                            "2019": Path("MSBuild")/"Current"/"Bin"/"MSBuild.exe",
                            "2017": Path("MSBuild")/"15.0"/"Bin"/"MSBuild.exe"}
        # CMake bundled inside every VS edition at this relative sub-path
        _CMAKE_VS_SUBPATH = (Path("Common7") / "IDE" / "CommonExtensions" /
                             "Microsoft" / "CMake" / "CMake" / "bin" / "cmake.exe")

        # 1. System PATH (user installed cmake + added to PATH)
        if shutil.which("cmake"):
            tools["CMake"] = True

        # 2. Standalone CMake install locations (not always on PATH)
        if not tools["CMake"]:
            _pf   = os.environ.get("ProgramFiles",      r"C:\Program Files")
            _pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
            for candidate in [
                Path(_pf)   / "CMake" / "bin" / "cmake.exe",
                Path(_pf86) / "CMake" / "bin" / "cmake.exe",
            ]:
                if candidate.exists():
                    tools["CMake"] = True
                    _path_prepend(str(candidate.parent))
                    break

        for year, primary, fallback in _vs_base_dirs():
            for vs_base in (primary, fallback):
                if not vs_base.exists():
                    continue
                for edition in _EDITIONS:
                    edition_path = vs_base / year / edition
                    # MSVC compiler
                    vc_tools = edition_path / "VC" / "Tools" / "MSVC"
                    if vc_tools.exists():
                        tools["MSVC"] = True
                        if not VS_GENERATOR:
                            VS_GENERATOR = _VS_GENERATORS.get(year, "")
                    # MSBuild
                    msbuild_path = edition_path / _MSBUILD_SUBPATH[year]
                    if msbuild_path.exists():
                        tools["MSBuild"] = True
                        _path_prepend(str(msbuild_path.parent))
                    # 3. VS-bundled CMake (not on PATH unless VS dev shell is active)
                    if not tools["CMake"]:
                        cmake_vs = edition_path / _CMAKE_VS_SUBPATH
                        if cmake_vs.exists():
                            tools["CMake"] = True
                            _path_prepend(str(cmake_vs.parent))

    else:
        if shutil.which("cmake"):
            tools["CMake"] = True
        if shutil.which("make") or shutil.which("ninja"):
            tools["MSVC"] = True  # Reused slot for "C compiler available"

    return tools


# =============================================================================
# CONFIG CREATION — unchanged from v1
# =============================================================================

def create_config(backend: str, embedding_model: str) -> None:
    config_path = BASE_DIR / "data" / "persistent.json"

    vulkan_enabled = "vulkan" in backend.lower()
    layer_mode     = "VRAM_SRAM" if vulkan_enabled else "SRAM_ONLY"
    default_vram   = 8192 if vulkan_enabled else 0
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
            "context_size": 32768,
            "vram_size": default_vram,
            "temperature": 0.66,
            "repeat_penalty": 1.1,
            "selected_gpu": "Auto",
            "selected_cpu": "Auto-Select",
            "mmap": True,
            "mlock": False,
            "n_batch": 2048,
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
        sys.exit(1)


# =============================================================================
# INSTALL MODE HELPERS — unchanged from v1
# =============================================================================

def select_install_mode() -> str:
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
            print("Abandoning installation...")
            sys.exit(0)
        if choice == "1":
            return "clean"
        if choice == "2":
            return "check"
        if choice == "3":
            return "refresh"
        print("Invalid selection. Please enter 1, 2, 3, or A.")


def _read_existing_ini() -> dict:
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
            'platform':              sys_sec.get('platform',        PLATFORM),
            'os_version':            sys_sec.get('os_version',      'unknown'),
            'python_version':        sys_sec.get('python_version',
                                                 f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"),
            'backend_type':          sys_sec.get('backend_type',    'CPU_CPU'),
            'embedding_model':       sys_sec.get('embedding_model', 'BAAI/bge-small-en-v1.5'),
            'vulkan_available':      sys_sec.getboolean('vulkan_available', False),
            'windows_version':       sys_sec.get('windows_version', None),
            'llama_cli_path':        sys_sec.get('llama_cli_path',  None),
            'llama_bin_path':        sys_sec.get('llama_bin_path',  None),
            'llama_wheel_version':   sys_sec.get('llama_wheel_version', None),
            'tts_engine':            'coqui',
            'coqui_voice_id':        None,
            'coqui_voice_accent':    None,
        }
        if 'tts' in config:
            tts_sec = config['tts']
            result['tts_engine'] = tts_sec.get('tts_type', 'coqui')
            if result['tts_engine'] == 'coqui':
                result['coqui_voice_id']     = tts_sec.get('coqui_voice_id', None)
                result['coqui_voice_accent'] = tts_sec.get('coqui_voice_accent', None)
        return result
    except Exception as e:
        print_status(f"Could not read constants.ini: {e}", False)
        return None


def _backend_type_to_string(backend_type: str) -> str:
    if backend_type == "VULKAN_VULKAN":
        return "Compile Vulkan Binaries / Compile Vulkan Wheel"
    elif backend_type == "VULKAN_CPU":
        return "Download Vulkan Binary / Default CPU Wheel"
    else:
        return "Download CPU Binary / Default CPU Wheel"


def select_backend_and_install_size():
    width = shutil.get_terminal_size().columns - 1
    print_header("Backend & Install Size")

    # Use cached globals from run_detections_once() — no re-detection
    tools          = _DETECTED_BUILD_TOOLS
    missing_tools  = [t for t, avail in tools.items() if not avail and t in ("Git", "CMake", "MSVC")]
    all_backend_opts = list(BACKEND_OPTIONS.keys())
    build_possible   = len(missing_tools) == 0
    backend_opts     = []

    for backend in all_backend_opts:
        info = BACKEND_OPTIONS[backend]
        requires_compile = info.get("compile_binary", False) or info.get("compile_wheel", False)
        if requires_compile and not build_possible:
            continue
        backend_opts.append(backend)

    compile_ver  = get_latest_llamacpp_python_version()
    prebuilt_ver = LLAMACPP_PYTHON_PREBUILT_VERSION

    def _wheel_label(backend_name: str) -> str:
        info = BACKEND_OPTIONS[backend_name]
        if info.get("compile_wheel"):
            return f"Wheel {compile_ver}"
        return f"Wheel {prebuilt_ver}"

    # ── Compact detection summary (cached globals) ────────────────────────────
    feat_str = " | ".join(k for k, v in _DETECTED_CPU_FEATURES.items()
                          if v and k in ("AVX", "AVX2", "AVX512", "FMA", "F16C"))
    feat_str = feat_str or "SSE3"

    tool_parts = [f"{name} {'OK' if avail else '--'}" for name, avail in tools.items()]
    tool_str   = " | ".join(tool_parts)

    if PLATFORM == "windows":
        os_str = f"Windows {WINDOWS_VERSION or detect_windows_version() or '?'}"
        vk_str = "YES" if is_vulkan_installed() else "NO"
        dx_str = f"DX{_DETECTED_DX_NAME}"
        print(f"System Detections...")
        print(f"   CPU Features : {feat_str}")
        print(f"   Build Tools  : {tool_str}")
        print(f"   Platform     : {os_str} | Python {sys.version_info.major}.{sys.version_info.minor}")
        print(f"   GPU          : {dx_str} | Vulkan: {vk_str}")
    else:
        os_str = f"Ubuntu {OS_VERSION or detect_linux_version() or '?'}"
        vk_str = "YES" if is_vulkan_installed() else "NO"
        print(f"System Detections...")
        print(f"   CPU Features : {feat_str}")
        print(f"   Build Tools  : {tool_str}")
        print(f"   Platform     : {os_str} | Python {sys.version_info.major}.{sys.version_info.minor}")
        print(f"   Vulkan       : {vk_str}")
    print()

    print("Backend Options...")
    for i, backend in enumerate(backend_opts, 1):
        print(f"   {i}) {backend} ({_wheel_label(backend)})")

    if not build_possible:
        print(f"   ---")
        for backend in all_backend_opts:
            if backend not in backend_opts:
                print(f"   -) {backend} ({_wheel_label(backend)}) (Missing: {', '.join(missing_tools)})")

    print()

    print("Install Size...")
    print(f"   a) Small  +450MB  - Bge-Small-En v1.5 + Coqui TTS (faster)")
    print(f"   b) Medium +1.5GB  - Bge-Base-En v1.5  + Coqui TTS (quality)")

    print()
    print("=" * width)

    max_backend = len(backend_opts)
    prompt = f"Selection; Backend=1-{max_backend}, Size=a-b, Abandon=A; (e.g. 2b): "
    choice = input(prompt).strip().lower()

    if choice == "a":
        print("Abandoning installation...")
        sys.exit(0)

    choice = choice.replace(" ", "").replace("-", "")

    while True:
        if len(choice) >= 2 and choice[0].isdigit() and choice[1] in "ab":
            backend_num = int(choice[0])
            size_letter = choice[1]

            if 1 <= backend_num <= len(backend_opts):
                selected_backend = backend_opts[backend_num - 1]

                if size_letter == "a":
                    embedding_model = EMBEDDING_MODELS["1"]["name"]
                else:
                    embedding_model = EMBEDDING_MODELS["2"]["name"]

                tts_engine  = "coqui"
                coqui_voice = COQUI_ENGLISH_VOICE

                time.sleep(1)
                return selected_backend, embedding_model, tts_engine, coqui_voice

        print("Invalid selection. Please enter a valid combination (e.g. 2b).")
        choice = input(prompt).strip().lower()
        if choice == "a":
            sys.exit(0)
        choice = choice.replace(" ", "").replace("-", "")


def validate_installation(backend: str, embedding_model: str, tts_engine: str) -> bool:
    """Comprehensive validation of installation integrity."""
    print_status("Validating installation integrity...")

    python_exe_path = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / \
                      ("python.exe" if PLATFORM == "windows" else "python")
    python_exe = str(python_exe_path)

    if not python_exe_path.exists():
        print_status("Python executable missing from venv", False)
        return False

    all_passed = True

    print("\n=== Core Library Validation ===")
    core_libs = [
        ("gradio",               "gradio"),
        ("numpy",                "numpy"),
        ("torch",                "torch"),
        ("sentence_transformers","sentence_transformers"),
        ("llama_cpp",            "llama_cpp"),
        ("spacy",                "spacy"),
        ("PyQt6",                "PyQt6"),
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
    model = SentenceTransformer("{embedding_model}", device="cpu",
                                cache_folder=r"{str(cache_dir.absolute())}")
    result = model.encode(["validation test"], convert_to_tensor=True)
    print("OK")
except Exception as e:
    print(f"Error: {{e}}")
'''
        try:
            result = subprocess.run([python_exe, "-c", test_code],
                                   capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and "OK" in result.stdout:
                print_status(f"  Embedding model verified ({embedding_model})")
            else:
                print_status("  Embedding model failed to load", False)
                all_passed = False
        except Exception as e:
            print_status(f"  Embedding validation error: {e}", False)
            all_passed = False
    else:
        print_status("  Embedding cache directory missing", False)
        all_passed = False

    print("\n=== spaCy Model Validation ===")
    try:
        result = subprocess.run(
            [python_exe, "-c",
             "import spacy; nlp = spacy.load('en_core_web_sm'); print('OK')"],
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

    print("\n=== TTS Validation ===")
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

    print("\n=== Backend Binary Validation ===")
    info     = BACKEND_OPTIONS.get(backend, {})
    cli_path = info.get("cli_path")
    if cli_path:
        cli_full_path = BASE_DIR / cli_path
        if cli_full_path.exists():
            print_status(f"  llama-cli found: {cli_full_path.name}")
            if PLATFORM == "linux" and not os.access(cli_full_path, os.X_OK):
                print_status("  llama-cli not executable", False)
                all_passed = False
        else:
            print_status(f"  llama-cli not found: {cli_path}", False)
            all_passed = False
    else:
        print_status("  Python bindings mode: No binary needed")

    print("\n=== Configuration Validation ===")
    for path_obj, label in [
        (BASE_DIR / "data" / "constants.ini",  "constants.ini"),
        (BASE_DIR / "data" / "persistent.json", "persistent.json"),
    ]:
        if path_obj.exists():
            if label.endswith(".json"):
                try:
                    with open(path_obj, 'r') as f:
                        json.load(f)
                    print_status(f"  {label} valid")
                except Exception as e:
                    print_status(f"  {label} corrupted: {e}", False)
                    all_passed = False
            else:
                print_status(f"  {label} exists")
        else:
            print_status(f"  {label} missing", False)
            all_passed = False

    print(f"\n{'=' * 50}")
    if all_passed:
        print_status("All validations passed!")
    else:
        print_status("Some checks failed", False)
        print("  Re-run installer with 'Clean Install' to fix issues.")
    print(f"{'=' * 50}\n")

    return all_passed


# =============================================================================
# MAIN INSTALL FLOW
# =============================================================================

def install():
    global WINDOWS_VERSION, OS_VERSION

    run_initial_detection()

    if not check_version_compatibility():
        sys.exit(1)

    install_mode = select_install_mode()

    python_version = (f"{sys.version_info.major}.{sys.version_info.minor}"
                      f".{sys.version_info.micro}")

    windows_version = None
    if PLATFORM == "windows":
        WINDOWS_VERSION  = detect_windows_version() or "unknown"
        os_version       = WINDOWS_VERSION
        windows_version  = WINDOWS_VERSION
        vulkan_available = is_vulkan_installed()
    else:
        OS_VERSION       = detect_linux_version() or "unknown"
        os_version       = OS_VERSION
        vulkan_available = is_vulkan_installed()

    dx11_capable     = _DETECTED_DX_CAPABLE
    dx_feature_level = _DETECTED_DX_LEVEL

    # ── Refresh Configs mode ──────────────────────────────────────────────────
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
        # Re-apply the existing wheel version (create_system_ini doesn't know it)
        if existing.get('llama_wheel_version'):
            update_ini_wheel_version(existing['llama_wheel_version'])
        print_status("Configuration refresh complete!")
        print("\nRun the launcher to start Chat-Gradio-Gguf\n")
        return

    # ── Clean Install / Check+Install modes ──────────────────────────────────
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
            coqui_voice_id     = coqui_voice['id']     if (tts_engine == "coqui" and coqui_voice) else None
            coqui_voice_accent = coqui_voice['accent'] if (tts_engine == "coqui" and coqui_voice) else None
    else:
        backend, embedding_model, tts_engine, coqui_voice = select_backend_and_install_size()
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

    print_header("Installation")

    if PLATFORM == "windows":
        os_display = f"Windows {WINDOWS_VERSION}" if WINDOWS_VERSION else "Windows"
    else:
        os_display = f"Ubuntu {OS_VERSION}" if OS_VERSION else "Ubuntu"

    py_ver     = f"{sys.version_info.major}.{sys.version_info.minor}"
    mode_label = "Check/Install" if install_mode == "check" else "Clean Install"

    print(f"Installing {APP_NAME} v2 on {os_display} with Python {py_ver}")
    print(f"  Mode: {mode_label}")
    print(f"  Route: {backend}")
    print(f"  Llama.Cpp {LLAMACPP_TARGET_VERSION}")
    print(f"  Embedding: {embedding_model}")

    if PLATFORM == "windows":
        fl_str = f"0x{dx_feature_level:04x}"
        print(f"  GPU: DirectX Feature Level {fl_str}")

    if tts_engine == "coqui" and coqui_voice_id:
        print(f"  TTS: Coqui ({coqui_voice_id} / {coqui_voice_accent or 'english'})")
    else:
        print(f"  TTS: Coqui (voice selection not configured — re-run installer)")

    if install_mode == "clean":
        if TEMP_DIR.exists():
            shutil.rmtree(TEMP_DIR, ignore_errors=True)
        if PLATFORM == "windows" and WIN_COMPILE_TEMP.exists():
            shutil.rmtree(WIN_COMPILE_TEMP, ignore_errors=True)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    if install_mode == "clean":
        if not create_venv():
            print_status("Virtual environment failed", False)
            sys.exit(1)
    else:
        if not ensure_venv():
            print_status("Virtual environment failed", False)
            sys.exit(1)

    # py-cpuinfo needed early for CPU detection during build
    print_status("Installing py-cpuinfo for CPU detection...")
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                 ("pip.exe" if PLATFORM == "windows" else "pip"))
    subprocess.run([pip_exe, "install", "py-cpuinfo"], check=True)
    print_status("py-cpuinfo installed")

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

    # Persist the confirmed wheel version into constants.ini now that install succeeded
    if _INSTALLED_LLAMA_WHEEL_VERSION:
        update_ini_wheel_version(_INSTALLED_LLAMA_WHEEL_VERSION)

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

    if install_mode == "check":
        print("\n")
        if not validate_installation(backend, embedding_model, tts_engine):
            print_status("Validation failed - some components may need reinstallation", False)
            retry = input("\nWould you like to perform a Clean Install to fix issues? (y/n): ").strip().lower()
            if retry == "y":
                print("\nRestarting with Clean Install mode...\n")
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
                validate_installation(backend, embedding_model, tts_engine)

    print_status("Installation complete!")
    print("\nRun the launcher to start Chat-Gradio-Gguf v2\n")


# Protected main block
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