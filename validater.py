# Script: validater.py - Validation script for Chat-Gradio-Gguf
# Note: Reads constants.ini to determine install configuration, checks presence of files/libraries

# Imports
import os
import sys
import subprocess
import json
from pathlib import Path

# Platform setup (MUST come before any PLATFORM-dependent code)
if len(sys.argv) < 2 or sys.argv[1].lower() not in ["windows", "linux"]:
    print("ERROR: Platform argument required (windows/linux)")
    sys.exit(1)
PLATFORM = sys.argv[1].lower()

# Platform-specific import
if PLATFORM == "windows":
    import winreg

# Global paths
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
CONFIG_PATH = BASE_DIR / "data" / "persistent.json"
CONSTANTS_INI_PATH = BASE_DIR / "data" / "constants.ini"
EMBEDDING_CACHE = BASE_DIR / "data" / "embedding_cache"

# System info from ini - loaded at start
SYSTEM_INFO = {}

# Required script files
REQUIRED_SCRIPTS = [
    "browser.py", "interface.py", "models.py", 
    "prompts.py", "settings.py", "temporary.py", "utility.py"
]

# Required directories (matching installer DIRECTORIES)
REQUIRED_DIRS = [
    "data", "scripts", "models",
    "data/history", "data/temp", "data/vectors",
    "data/embedding_cache"
]

def print_status(msg: str, success: bool = True) -> None:
    """Simplified status printer"""
    status = "✓" if success else "✗"
    print(f"  {status} {msg}")

def load_constants_ini() -> bool:
    """Load constants.ini to determine what to validate"""
    global SYSTEM_INFO
    
    if not CONSTANTS_INI_PATH.exists():
        print("WARNING: constants.ini not found - installation incomplete")
        return False
    
    try:
        import configparser
        config = configparser.ConfigParser()
        config.read(CONSTANTS_INI_PATH)
        
        SYSTEM_INFO = {
            'platform': config.get('system', 'platform', fallback=PLATFORM),
            'os_version': config.get('system', 'os_version', fallback='unknown'),
            'python_version': config.get('system', 'python_version', fallback='unknown'),
            'backend_type': config.get('system', 'backend_type', fallback='CPU_CPU'),
            'embedding_model': config.get('system', 'embedding_model', fallback='BAAI/bge-small-en-v1.5'),
            'embedding_backend': config.get('system', 'embedding_backend', fallback='sentence_transformers'),
            'vulkan_available': config.getboolean('system', 'vulkan_available', fallback=False),
            'llama_cli_path': config.get('system', 'llama_cli_path', fallback=None),
            'llama_bin_path': config.get('system', 'llama_bin_path', fallback=None),
            'gradio_version': config.get('system', 'gradio_version', fallback='5.49.1'),
            'qt_version': config.get('system', 'qt_version', fallback='6'),
        }
        
        if PLATFORM == 'windows':
            SYSTEM_INFO['windows_version'] = config.get('system', 'windows_version', fallback=None)
        
        return True
    except Exception as e:
        print(f"ERROR reading constants.ini: {e}")
        return False

def get_python_minor_version() -> int:
    """Get Python minor version from constants.ini or current runtime"""
    py_ver = SYSTEM_INFO.get('python_version', 'unknown')
    try:
        parts = py_ver.split('.')
        if len(parts) >= 2:
            return int(parts[1])
    except:
        pass
    return sys.version_info.minor

def test_directories() -> bool:
    """Verify required directories exist"""
    print("=== Directory Validation ===")
    success = True
    
    for dir_name in REQUIRED_DIRS:
        dir_path = BASE_DIR / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print_status(f"{dir_name}")
        else:
            print_status(f"Missing: {dir_name}", False)
            success = False
    
    return success

def test_scripts() -> bool:
    """Verify required script files exist"""
    print("\n=== Script Files Validation ===")
    success = True
    scripts_dir = BASE_DIR / "scripts"
    
    if not scripts_dir.exists():
        print_status("scripts/ directory missing", False)
        return False
    
    for script_name in REQUIRED_SCRIPTS:
        script_path = scripts_dir / script_name
        if script_path.exists():
            print_status(f"{script_name}")
        else:
            print_status(f"Missing: {script_name}", False)
            success = False
    
    return success

def test_config() -> bool:
    """Verify configuration files exist and are valid"""
    print("\n=== Configuration Validation ===")
    
    # constants.ini already loaded in main(), just verify it exists
    if not CONSTANTS_INI_PATH.exists():
        print_status("constants.ini missing!", False)
        return False
    
    print_status(f"constants.ini valid (Backend: {SYSTEM_INFO['backend_type']})")
    
    # Check persistent.json
    if not CONFIG_PATH.exists():
        print_status("persistent.json missing!", False)
        return False
        
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            
        model_settings = config.get("model_settings", {})
        if not model_settings:
            print_status("model_settings section missing", False)
            return False
        
        print_status("persistent.json valid")
        return True
        
    except Exception as e:
        print_status(f"persistent.json corrupted: {str(e)}", False)
        return False

def test_llama_cli() -> bool:
    """Verify llama-cli exists if backend uses binaries"""
    print("\n=== Backend Binary Validation ===")
    
    backend_type = SYSTEM_INFO.get('backend_type', 'CPU_CPU')
    llama_cli_path = SYSTEM_INFO.get('llama_cli_path')
    
    # CPU_CPU without cli_path = Python bindings only mode
    if backend_type == "CPU_CPU" and not llama_cli_path:
        print_status("Python bindings mode: No binary needed")
        return True
    
    # If cli_path is set, verify it exists
    if llama_cli_path:
        if os.path.isabs(llama_cli_path):
            cli_path = Path(llama_cli_path)
        else:
            clean_path = llama_cli_path.lstrip("./").lstrip(".\\")
            cli_path = BASE_DIR / clean_path
        
        if cli_path.exists():
            print_status(f"llama-cli found: {cli_path.name}")
            return True
        else:
            print_status(f"llama-cli not found: {cli_path}", False)
            return False
    
    # VULKAN_CPU or VULKAN_VULKAN should have binaries
    if backend_type in ["VULKAN_CPU", "VULKAN_VULKAN"]:
        print_status(f"Backend {backend_type} missing llama_cli_path in config", False)
        return False
    
    print_status("Python bindings mode")
    return True

def test_venv() -> bool:
    """Verify virtual environment exists"""
    print("\n=== Virtual Environment Validation ===")
    
    if not VENV_DIR.exists():
        print_status("Virtual environment not found", False)
        return False
    
    venv_py = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
    
    if not venv_py.exists():
        print_status("Python executable missing from venv", False)
        return False
    
    print_status("Virtual environment OK")
    return True

def test_core_libs() -> bool:
    """Test if core libraries are installed based on constants.ini"""
    print("\n=== Core Library Validation ===")
    
    venv_py = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
    
    if not venv_py.exists():
        print_status("Python executable missing", False)
        return False
    
    success = True
    py_minor = get_python_minor_version()
    
    # Core packages all installs need (matching installer BASE_REQ + dynamic additions)
    core_checks = [
        ("gradio", "gradio"),
        ("numpy", "numpy"),
        ("requests", "requests"),
        ("pyperclip", "pyperclip"),
        ("spacy", "spacy"),
        ("psutil", "psutil"),
        ("ddgs", "ddgs"),  # DuckDuckGo search
        ("langchain-community", "langchain_community"),
        ("faiss-cpu", "faiss"),
        ("langchain", "langchain"),
        ("pygments", "pygments"),
        ("lxml", "lxml"),
        ("lxml_html_clean", "lxml_html_clean"),
        ("tokenizers", "tokenizers"),
        ("beautifulsoup4", "bs4"),
        ("aiohttp", "aiohttp"),
        ("py-cpuinfo", "cpuinfo"),
        # Embedding backend packages
        ("sentence-transformers", "sentence_transformers"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        # LLM backend
        ("llama-cpp-python", "llama_cpp"),
    ]
    
    # newspaper version depends on Python version
    if py_minor >= 10:
        core_checks.append(("newspaper4k", "newspaper"))
    else:
        core_checks.append(("newspaper3k", "newspaper"))
    
    # Platform-specific packages
    if PLATFORM == "windows":
        core_checks.extend([
            ("pywin32", "win32api"),
            ("tk", "tkinter"),
            ("pythonnet", "clr"),
        ])
    else:
        # Linux-specific
        core_checks.append(("pyvirtualdisplay", "pyvirtualdisplay"))
    
    for pkg_name, import_name in core_checks:
        try:
            result = subprocess.run(
                [str(venv_py), "-c", f"import {import_name}; print('OK')"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0 and "OK" in result.stdout:
                print_status(pkg_name)
            else:
                print_status(f"{pkg_name} missing", False)
                success = False
                
        except subprocess.TimeoutExpired:
            print_status(f"{pkg_name} (timeout)", False)
            success = False
        except Exception as e:
            print_status(f"{pkg_name} error", False)
            success = False
    
    return success

def test_optional_libs() -> None:
    """Test optional file format libraries"""
    print("\n=== Optional Library Validation ===")
    
    venv_py = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
    
    optional_checks = [
        ("PyPDF2", "PyPDF2"),
        ("python-docx", "docx"),
        ("openpyxl", "openpyxl"),
        ("python-pptx", "pptx"),
    ]
    
    optional_ok = 0
    optional_missing = 0
    
    for pkg_name, import_name in optional_checks:
        try:
            result = subprocess.run(
                [str(venv_py), "-c", f"import {import_name}; print('OK')"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and "OK" in result.stdout:
                print_status(f"{pkg_name} (optional)")
                optional_ok += 1
            else:
                print_status(f"{pkg_name} (not installed - optional)", False)
                optional_missing += 1
                
        except Exception:
            print_status(f"{pkg_name} (not installed - optional)", False)
            optional_missing += 1
    
    if optional_missing > 0:
        print(f"\n  Note: {optional_missing} optional packages missing (text-only fallback will be used)")

def test_browser_setup() -> bool:
    """Minimal check: gradio + the Qt WebEngine from constants.ini"""
    print("\n=== Gradio & Custom Browser Check ===")
    
    venv_py = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
    
    # Get Qt version from constants.ini (set by installer)
    qt_version = SYSTEM_INFO.get('qt_version', '6')
    gradio_version = SYSTEM_INFO.get('gradio_version', '5.49.1')
    
    # Determine Qt package name
    if str(qt_version) == "5":
        qt_part = "PyQt5 WebEngine"
        qt_module = "PyQt5"
    else:
        qt_part = "PyQt6 WebEngine"
        qt_module = "PyQt6"
    
    print(f"  Expected: Gradio {gradio_version}, {qt_part}")
    
    # 1. Gradio exists?
    try:
        subprocess.run([str(venv_py), "-c", "import gradio"],
                       capture_output=True, timeout=10, check=True)
        print_status("Gradio import OK")
    except:
        print_status("Gradio import FAILED - interface will not work", False)
        return False
    
    # 2. Expected WebEngine exists?
    try:
        code = f"from {qt_module}.QtWebEngineWidgets import QWebEngineView"
        subprocess.run([str(venv_py), "-c", code],
                       capture_output=True, timeout=12, check=True)
        print_status(f"{qt_part} import OK")
        return True
    except:
        print_status(f"{qt_part} NOT found → will fall back to system browser", False)
        return False  # still non-fatal


def test_spacy_model() -> bool:
    """Verify spaCy English model is downloaded"""
    print("\n=== spaCy Model Validation ===")
    
    venv_py = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
    
    try:
        result = subprocess.run(
            [str(venv_py), "-c", "import spacy; nlp = spacy.load('en_core_web_sm'); print('OK')"],
            capture_output=True,
            text=True,
            timeout=15
        )
        
        if result.returncode == 0 and "OK" in result.stdout:
            print_status("en_core_web_sm model available")
            return True
        else:
            print_status("en_core_web_sm model not found", False)
            return False
            
    except Exception as e:
        print_status(f"spaCy model check failed: {str(e)}", False)
        return False

def test_embedding_model() -> bool:
    """Verify sentence-transformers embedding model is cached"""
    print("\n=== Embedding Model Validation ===")
    
    if not EMBEDDING_CACHE.exists():
        print_status("Embedding cache directory missing", False)
        return False
    
    # Check if cache has model files
    cache_contents = list(EMBEDDING_CACHE.rglob("*"))
    model_files = [f for f in cache_contents if f.is_file()]
    
    if len(model_files) == 0:
        print_status("Embedding cache is empty", False)
        return False
    
    # Try to load the model
    venv_py = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
    
    target_model = SYSTEM_INFO.get('embedding_model', 'BAAI/bge-small-en-v1.5')
    cache_dir_abs = str(EMBEDDING_CACHE.absolute())
    
    test_code = f'''
import os
os.environ["TRANSFORMERS_CACHE"] = r"{cache_dir_abs}"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"{cache_dir_abs}"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("{target_model}", device="cpu")
    result = model.encode(["test"], convert_to_tensor=True)
    print("OK")
except Exception as e:
    print(f"Error: {{e}}")
'''
    
    try:
        result = subprocess.run(
            [str(venv_py), "-c", test_code],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and "OK" in result.stdout:
            print_status(f"Embedding model verified ({target_model})")
            return True
        else:
            print_status("Embedding model failed to load", False)
            if result.stderr:
                print(f"  Error: {result.stderr.strip()[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("Embedding model check timed out", False)
        return False
    except Exception as e:
        print_status(f"Embedding validation error: {str(e)}", False)
        return False

def test_linux_system_packages() -> bool:
    """Check Linux system dependencies for TTS and Qt"""
    if PLATFORM != "linux":
        return True
        
    print("\n=== Linux System Dependencies ===")
    
    qt_version = SYSTEM_INFO.get('qt_version', '6')
    
    # Qt system dependencies
    if str(qt_version) == "5":
        qt_packages = ["libxcb-xinerama0", "libxkbcommon0", "libegl1", "libgl1"]
    else:
        qt_packages = ["libxcb-cursor0", "libxkbcommon0", "libegl1", "libgl1"]
    
    # Headless Qt support
    display_packages = ["xvfb"]
    
    all_packages = qt_packages + display_packages
    missing = []
    
    for pkg in all_packages:
        try:
            r = subprocess.run(["dpkg", "-s", pkg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode != 0:
                missing.append(pkg)
        except:
            missing.append(pkg + " (check failed)")
    
    if missing:
        print_status("Some system packages may be missing: " + ", ".join(missing), False)
        print(f"  → Suggested fix: sudo apt install {' '.join(missing)}")
        return False
    else:
        print_status("Linux system packages present")
        return True

def main():
    """Main validation routine"""
    print(f"{'=' * 50}")
    print(f"  Chat-Gradio-Gguf Validator ({PLATFORM.upper()})")
    print(f"{'=' * 50}\n")
    
    # Check for installation files FIRST
    if not CONSTANTS_INI_PATH.exists() or not CONFIG_PATH.exists():
        missing = []
        if not CONSTANTS_INI_PATH.exists():
            missing.append("constants.ini")
        if not CONFIG_PATH.exists():
            missing.append("persistent.json")
        print_status(f"Missing: {', '.join(missing)}", False)
        print("\nNo installation detected. Run installer first.")
        input("\nPress Enter to exit...")
        return 1
    
    if not load_constants_ini():
        print_status("constants.ini corrupted", False)
        print("\nRe-run installer to fix.")
        input("\nPress Enter to exit...")
        return 1
    
    print(f"System: {SYSTEM_INFO['platform']} {SYSTEM_INFO['os_version']}")
    print(f"Python: {SYSTEM_INFO['python_version']}")
    print(f"Backend: {SYSTEM_INFO['backend_type']}")
    print(f"Embedding: {SYSTEM_INFO['embedding_model']}")
    print(f"Gradio: {SYSTEM_INFO['gradio_version']}, Qt: {SYSTEM_INFO['qt_version']}\n")
    
    results = {
        "directories": test_directories(),
        "scripts": test_scripts(),
        "config": test_config(),
        "venv": test_venv(),
    }
    
    # Only test further if venv exists
    if results["venv"]:
        results["llama_cli"] = test_llama_cli()
        results["core_libs"] = test_core_libs()
        test_optional_libs()  # Non-fatal
        results["spacy"] = test_spacy_model()
        results["embedding"] = test_embedding_model()
        results["browser"] = test_browser_setup()
        
        if PLATFORM == "linux":
            results["linux_packages"] = test_linux_system_packages()
    
    # Summary
    print(f"\n{'=' * 50}")
    print("  Validation Summary")
    print(f"{'=' * 50}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n  Result: {passed}/{total} checks passed")
    
    overall_success = all(results.values())
    
    if overall_success:
        print_status("\nAll validations passed!")
    else:
        print_status("\nSome checks failed", False)
        print("  Re-run installer if needed.")
    
    input("\nPress Enter to exit...")
    return 0 if overall_success else 1

if __name__ == "__main__":
    sys.exit(main())