# Script: validator.py

# Imports
import os
import sys
import subprocess
import json
from pathlib import Path
if PLATFORM == "windows":
    import winreg

# Global paths
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
CONFIG_PATH = BASE_DIR / "data" / "persistent.json"
SYSTEM_INI_PATH = BASE_DIR / "data" / "system.ini"  # NEW
FASTEMBED_CACHE = BASE_DIR / "data" / "fastembed_cache"

# Platform setup
if len(sys.argv) < 2 or sys.argv[1].lower() not in ["windows", "linux"]:
    print("ERROR: Platform argument required (windows/linux)")
    sys.exit(1)
PLATFORM = sys.argv[1].lower()

# System info from ini - loaded at start
SYSTEM_INFO = {}

# Core requirements (matching installer BASE_REQ)
CORE_REQS = [
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
    "onnxruntime",
    "fastembed",
    "tokenizers",
]

# Optional file format support (matching installer optional packages)
OPTIONAL_REQS = [
    "PyPDF2",
    "python-docx",
    "openpyxl",
    "python-pptx"
]

# Platform-specific Code
if PLATFORM == "windows":
    CORE_REQS.extend(["pywin32", "tk"])

def print_status(msg: str, success: bool = True) -> None:
    """Simplified status printer"""
    status = "✓" if success else "✗"
    print(f"  {status} {msg}")

def load_system_ini():
    """Load system.ini to determine what to validate"""
    global SYSTEM_INFO
    
    if not SYSTEM_INI_PATH.exists():
        print("WARNING: system.ini not found - using defaults")
        return False
    
    try:
        import configparser
        config = configparser.ConfigParser()
        config.read(SYSTEM_INI_PATH)
        
        SYSTEM_INFO = {
            'platform': config.get('system', 'platform', fallback=PLATFORM),
            'os_version': config.get('system', 'os_version', fallback='unknown'),
            'python_version': config.get('system', 'python_version', fallback='unknown'),
            'backend_type': config.get('system', 'backend_type', fallback='CPU_CPU'),
            'embedding_model': config.get('system', 'embedding_model', fallback='unknown'),
            'vulkan_available': config.getboolean('system', 'vulkan_available', fallback=False),
            'windows_version': config.get('system', 'windows_version', fallback=None) if PLATFORM == 'windows' else None
        }
        return True
    except Exception as e:
        print(f"ERROR reading system.ini: {e}")
        return False

# Functions...
def test_directories():
    """Verify required directories exist"""
    print("=== Directory Validation ===")
    success = True
    
    required_dirs = [
        BASE_DIR / "data",
        BASE_DIR / "scripts",
        BASE_DIR / "models",
        BASE_DIR / "data/history",
        BASE_DIR / "data/temp",
        BASE_DIR / "data/vectors",
        BASE_DIR / "data/fastembed_cache"
    ]
    
    for dir_path in required_dirs:
        if dir_path.exists() and dir_path.is_dir():
            print_status(f"{dir_path.name}")
        else:
            print_status(f"Missing: {dir_path}", False)
            success = False
            
    return success

def test_config():
    """Verify configuration files exist and are valid"""
    print("\n=== Configuration Validation ===")
    
    # system.ini already loaded in main(), just verify it exists
    if not SYSTEM_INI_PATH.exists():
        print_status("system.ini missing!", False)
        return False
    
    print_status(f"system.ini valid (Backend: {SYSTEM_INFO['backend_type']})")
    
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

def test_llama_cli():
    """Verify llama-cli exists if backend uses binaries"""
    print("\n=== Backend Binary Validation ===")
    
    backend_type = SYSTEM_INFO.get('backend_type', 'CPU_CPU')
    
    # CPU_CPU with Python bindings only (Option 1)
    if backend_type == "CPU_CPU":
        try:
            with open(CONFIG_PATH, 'r') as f:
                config = json.load(f)
            llama_cli_path = config.get("model_settings", {}).get("llama_cli_path")
            
            if not llama_cli_path:
                print_status("Python bindings mode: No binary needed")
                return True
        except:
            print_status("Python bindings mode: No binary needed")
            return True
    
    # Other backends need binaries
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        llama_cli_path = config.get("model_settings", {}).get("llama_cli_path")
        
        if not llama_cli_path:
            print_status(f"Backend {backend_type} missing llama_cli_path", False)
            return False
        
        if os.path.isabs(llama_cli_path):
            cli_path = Path(llama_cli_path)
        else:
            clean_path = llama_cli_path.lstrip("./").lstrip(".\\")
            cli_path = BASE_DIR / clean_path
        
        if not cli_path.exists():
            print_status(f"llama-cli not found: {cli_path.name}", False)
            return False
        
        print_status(f"llama-cli found: {cli_path.name}")
        return True
            
    except Exception as e:
        print_status(f"Backend validation error: {str(e)}", False)
        return False

def test_core_libs():
    """Test if core libraries are installed based on system.ini"""
    print("\n=== Core Library Validation ===")
    
    if not VENV_DIR.exists():
        print_status("Virtual environment not found", False)
        return False
    
    venv_py = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
    
    if not venv_py.exists():
        print_status("Python executable missing", False)
        return False
    
    success = True
    
    # Always check these
    core_checks = [
        ("gradio", "gradio"),
        ("requests", "requests"),
        ("pyperclip", "pyperclip"),
        ("spacy", "spacy"),
        ("psutil", "psutil"),
        ("newspaper3k", "newspaper"),
        ("langchain", "langchain"),
        ("faiss-cpu", "faiss"),
        ("fastembed", "fastembed"),
        ("onnxruntime", "onnxruntime"),
        ("llama-cpp-python", "llama_cpp")  # NEW - always check this
    ]
    
    # Platform-specific
    if PLATFORM == "windows":
        core_checks.extend([
            ("pywin32", "win32api"),
            ("tk", "tkinter")
        ])
        
        # Only check pywebview if Python < 3.13
        py_version = SYSTEM_INFO.get('python_version', '3.11.0')
        py_minor = int(py_version.split('.')[1])
        if py_minor < 13:
            core_checks.append(("pywebview", "webview"))
    
    for pkg_name, import_name in core_checks:
        try:
            result = subprocess.run(
                [str(venv_py), "-c", f"import {import_name}; print('OK')"],
                capture_output=True,
                text=True,
                timeout=15
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

def test_optional_libs():
    """Test optional file format libraries"""
    print("\n=== Optional Library Validation ===")
    
    venv_py = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
    
    import_names = {
        "PyPDF2": "PyPDF2",
        "python-docx": "docx",
        "openpyxl": "openpyxl",
        "python-pptx": "pptx"
    }
    
    optional_ok = 0
    optional_missing = 0
    
    for req in OPTIONAL_REQS:
        pkg_name = req.split('>=')[0].split('==')[0].strip()
        import_name = import_names.get(pkg_name, pkg_name.replace('-', '_'))
        
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
        print(f"\nNote: {optional_missing} optional packages missing (text-only fallback will be used)")
    
    return True  # Optional packages don't fail validation

def test_browser_setup():
    """Check browser components based on OS version"""
    if PLATFORM != "windows":
        print("\n=== Browser Validation (Linux) ===")
        print_status("Using GTK WebView (system)")
        return True
    
    print("\n=== Browser Validation (Windows) ===")
    
    win_ver = SYSTEM_INFO.get('windows_version', 'unknown')
    
    if win_ver in ["7", "8"]:
        print_status(f"Windows {win_ver} - system browser fallback")
        return True
    
    # Windows 8.1+ should have WebView2 runtime
    if win_ver in ["8.1", "10", "11"]:
        try:
            import winreg  # Import here to avoid issues on Linux
            key_path = r"SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path):
                print_status(f"WebView2 runtime installed (Win {win_ver})")
                return True
        except ImportError:
            print_status("winreg module not available", False)
            return False
        except:
            print_status("WebView2 runtime not detected", False)
            print("  Custom browser may not work")
            return False
    
    print_status(f"Unknown Windows version: {win_ver}", False)
    return False

def test_spacy_model():
    """Verify spaCy English model is downloaded"""
    print("\n=== spaCy Model Validation ===")
    
    venv_py = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
    
    try:
        result = subprocess.run(
            [str(venv_py), "-c", "import spacy; nlp = spacy.load('en_core_web_sm'); print('OK')"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0 and "OK" in result.stdout:
            print_status("en_core_web_sm model available")
            return True
        else:
            print_status("en_core_web_sm model not found", False)
            print("  Run: python -m spacy download en_core_web_sm")
            return False
            
    except Exception as e:
        print_status(f"spaCy model check failed: {str(e)}", False)
        return False

def test_fastembed_model():
    """Verify FastEmbed model is cached"""
    print("\n=== FastEmbed Model Validation ===")
    
    if not FASTEMBED_CACHE.exists():
        print_status("FastEmbed cache directory missing", False)
        return False
    
    # Check if cache has model files
    cache_contents = list(FASTEMBED_CACHE.rglob("*"))
    model_files = [f for f in cache_contents if f.is_file()]
    
    if len(model_files) == 0:
        print_status("FastEmbed cache is empty", False)
        print("  Model needs to be downloaded")
        return False
    
    # Try to load the model
    venv_py = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
    
    try:
        cache_dir_abs = FASTEMBED_CACHE.absolute()
        # Match the model name used in installer.py
        target_model = SYSTEM_INFO.get('embedding_model', 'BAAI/bge-small-en-v1.5')

        test_code = f'''
        import os
        import sys
        from pathlib import Path
        try:
            cache_dir = Path(r"{str(cache_dir_abs)}")
            os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir.absolute())

            from fastembed import TextEmbedding
            model = TextEmbedding(
                model_name="{target_model}", 
                cache_dir=str(cache_dir),
                providers=["CPUExecutionProvider"]
            )
            print('OK')
        except Exception as e:
            print(f"Error: {{e}}")
            sys.exit(1)
        '''
        
        result = subprocess.run(
            [str(venv_py), "-c", test_code],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0 and "OK" in result.stdout:
            print_status(f"FastEmbed model verified ({target_model})")
            return True
        else:
            print_status("FastEmbed model failed to load", False)
            if result.stderr:
                print(f"  Error: {result.stderr.strip()[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("FastEmbed model check timed out", False)
        return False
    except Exception as e:
        print_status(f"FastEmbed validation error: {str(e)}", False)
        return False

def test_linux_system_packages():
    """Check Linux system dependencies"""
    if PLATFORM != "linux":
        return True
        
    print("\n=== Linux System Package Validation ===")
    
    required_packages = [
        "build-essential",
        "python3-venv",
        "python3-dev",
        "portaudio19-dev",
        "libasound2-dev",
        "python3-tk",      # installer installs this meta-package
        "espeak",
        "libespeak-dev",
        "ffmpeg",
        "xclip"
    ]
    
    success = True
    
    for pkg in required_packages:
        try:
            result = subprocess.run(
                ["dpkg", "-s", pkg],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            if result.returncode == 0:
                print_status(f"{pkg}")
            else:
                print_status(f"{pkg} not installed", False)
                success = False
                
        except Exception as e:
            print_status(f"{pkg} check failed", False)
            success = False
    
    # Check for python3-tk or python3.X-tk
    tk_found = False
    for tk_pkg in ["python3-tk", f"python3.{sys.version_info.minor}-tk"]:
        try:
            result = subprocess.run(
                ["dpkg", "-s", tk_pkg],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if result.returncode == 0:
                print_status(f"{tk_pkg}")
                tk_found = True
                break
        except:
            pass
    
    if not tk_found:
        print_status("python3-tk not installed", False)
        success = False
    
    return success

def main():
    """Main validation routine"""
    print(f"=== Chat-Gradio-Gguf Validator ({PLATFORM.upper()}) ===\n")
    
    # Load system.ini FIRST
    if not SYSTEM_INI_PATH.exists():
        print_status("system.ini not found!", False)
        print("\nNo installation detected. Run installer first.")
        return 1
    
    if not load_system_ini():
        print_status("system.ini corrupted", False)
        return 1
    
    print(f"System: {SYSTEM_INFO['platform']} {SYSTEM_INFO['os_version']}")
    print(f"Python: {SYSTEM_INFO['python_version']}")
    print(f"Backend: {SYSTEM_INFO['backend_type']}\n")
    
    overall_success = True
    
    if not test_directories():
        overall_success = False
        
    if not test_config():
        overall_success = False
        
    if not test_llama_cli():
        overall_success = False
        
    if not test_core_libs():
        overall_success = False
    
    test_optional_libs()
    
    if not test_spacy_model():
        overall_success = False
        
    if not test_fastembed_model():
        overall_success = False
    
    if not test_browser_setup():  # NEW
        overall_success = False
    
    if PLATFORM == "linux":
        if not test_linux_system_packages():
            overall_success = False
    
    print("\n=== Validation Summary ===")
    if overall_success:
        print_status("All validations passed!")
        return 0
    else:
        print_status("Some checks failed", False)
        print("\nRe-run installer if needed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())