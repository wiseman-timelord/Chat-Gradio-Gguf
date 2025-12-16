# Script: validator.py

# Imports
import os
import sys
import subprocess
import json
from pathlib import Path

# Global paths
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
CONFIG_PATH = BASE_DIR / "data" / "persistent.json"
FASTEMBED_CACHE = BASE_DIR / "data" / "fastembed_cache"

# Platform setup
if len(sys.argv) < 2 or sys.argv[1].lower() not in ["windows", "linux"]:
    print("ERROR: Platform argument required (windows/linux)")
    sys.exit(1)
PLATFORM = sys.argv[1].lower()

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

# Platform-specific
if PLATFORM == "windows":
    CORE_REQS.extend(["pywin32", "tk"])

def print_status(msg: str, success: bool = True) -> None:
    """Simplified status printer"""
    status = "✓" if success else "✗"
    print(f"  {status} {msg}")

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
    """Verify configuration file exists and is valid"""
    print("\n=== Configuration Validation ===")
    
    if not CONFIG_PATH.exists():
        print_status("persistent.json missing!", False)
        return False
        
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            
        # Validate model_settings structure
        model_settings = config.get("model_settings", {})
        if not model_settings:
            print_status("model_settings section missing", False)
            return False
            
        # Check critical fields
        backend_type = model_settings.get("backend_type")
        
        # FIX: Added "Vulkan-Binary" to valid types to match installer.py
        valid_backends = ["CPU-Only", "Vulkan", "Vulkan-Binary"]
        if backend_type not in valid_backends:
            print_status(f"Invalid backend_type: {backend_type}", False)
            print(f"  Expected one of: {valid_backends}")
            return False
            
        print_status(f"Config valid (Backend: {backend_type})")
        
        # Check llama-cli path only if NOT CPU-Only
        if backend_type != "CPU-Only":
            llama_cli_path = model_settings.get("llama_cli_path")
            if llama_cli_path:
                # Handle relative paths from config
                if os.path.isabs(llama_cli_path):
                    cli_path = Path(llama_cli_path)
                else:
                    cli_path = BASE_DIR / llama_cli_path

                if cli_path.exists():
                    print_status(f"llama-cli configured: {cli_path.name}")
                else:
                    print_status(f"llama-cli not found at: {cli_path}", False)
                    return False
            else:
                print_status(f"{backend_type} backend requires llama_cli_path", False)
                return False
        else:
            print_status("CPU-Only mode (no llama-cli needed)")
            
        return True
        
    except Exception as e:
        print_status(f"Config validation failed: {str(e)}", False)
        return False

def test_llama_cli():
    """Verify llama-cli exists and is executable (Vulkan/Vulkan-Binary backends)"""
    print("\n=== Backend Binary Validation ===")
    
    # Load config to check backend type
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        model_settings = config.get("model_settings", {})
        backend_type = model_settings.get("backend_type")
        
        if backend_type == "CPU-Only":
            print_status("CPU-Only mode: Binary validation skipped")
            return True
            
        llama_cli_path = model_settings.get("llama_cli_path")
        
        if not llama_cli_path:
            print_status("llama_cli_path not in config", False)
            return False
            
        # Handle potential relative paths in config
        if os.path.isabs(llama_cli_path):
            cli_path = Path(llama_cli_path)
        else:
            cli_path = BASE_DIR / llama_cli_path
        
        if not cli_path.exists():
            print_status(f"llama-cli not found at: {cli_path}", False)
            return False
            
        # Check executability on Linux
        if PLATFORM == "linux":
            if not os.access(cli_path, os.X_OK):
                print_status("llama-cli is not executable", False)
                return False
            print_status(f"llama-cli verified: {cli_path.name}")
            return True
        else:
            # For Windows, try to run with --help
            try:
                result = subprocess.run(
                    [str(cli_path), "--help"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=5
                )
                if result.returncode == 0 or result.returncode == 1: # 1 is often help return code
                    print_status(f"llama-cli verified: {cli_path.name}")
                    return True
            except Exception:
                pass
                
            print_status("llama-cli failed to execute", False)
            return False
            
    except Exception as e:
        print_status(f"Backend validation error: {str(e)}", False)
        return False

def test_core_libs():
    """Test if all core libraries are properly installed"""
    print("\n=== Core Library Validation ===")
    
    if not VENV_DIR.exists():
        print_status("Virtual environment not found", False)
        return False
    
    venv_py = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
    
    if not venv_py.exists():
        print_status("Python executable missing", False)
        return False
    
    # Package name to import name mapping
    import_names = {
        "gradio": "gradio",
        "requests==2.31.0": "requests",
        "pyperclip": "pyperclip",
        "spacy>=3.7.0": "spacy",
        "psutil": "psutil",
        "ddgs": "ddgs",
        "newspaper3k": "newspaper",
        "langchain-community>=0.3.18": "langchain_community",
        "faiss-cpu>=1.8.0": "faiss",
        "langchain>=0.3.18": "langchain",
        "pygments==2.17.2": "pygments",
        "lxml[html_clean]": "lxml",
        "pyttsx3": "pyttsx3",
        "onnxruntime": "onnxruntime",
        "fastembed": "fastembed",
        "tokenizers": "tokenizers",
        "llama-cpp-python": "llama_cpp",
        "pywin32": "win32api",
        "tk": "tkinter"
    }
    
    success = True
    
    for req in CORE_REQS:
        pkg_name = req.split('>=')[0].split('==')[0].split('<')[0].split('[')[0].strip()
        import_name = import_names.get(pkg_name, pkg_name.replace('-', '_'))
        
        try:
            # Special handling for newspaper3k
            if pkg_name == "newspaper3k":
                cmd = """
try:
    import lxml.html.clean
    import newspaper
    from newspaper import Article
    print('OK')
except ImportError as e:
    print(f'MISSING: {str(e)}')
"""
            else:
                cmd = f"import {import_name}; print('OK')"
            
            result = subprocess.run(
                [str(venv_py), "-c", cmd],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0 and "OK" in result.stdout:
                print_status(pkg_name)
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                print_status(f"{pkg_name} - {error_msg}", False)
                success = False
                
        except subprocess.TimeoutExpired:
            print_status(f"{pkg_name} (timeout)", False)
            success = False
        except Exception as e:
            print_status(f"{pkg_name} (error: {str(e)})", False)
            success = False
    
    return success

def test_optional_libs():
    """Test optional file format libraries"""
    print("\n=== Optional Library Validation ===")
    
    venv_py = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
    
    import_names = {
        "PyPDF2>=3.0.0": "PyPDF2",
        "python-docx>=0.8.11": "docx",
        "openpyxl>=3.0.0": "openpyxl",
        "python-pptx>=0.6.21": "pptx"
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
        target_model = "BAAI/bge-small-en-v1.5"
        
        test_code = f'''
import os
import sys
from pathlib import Path
try:
    cache_dir = Path(r"{str(cache_dir_abs)}")
    os.environ["FASTEMBED_CACHE_PATH"] = str(cache_dir.absolute())

    from fastembed import TextEmbedding
    # Use exact model from installer
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
    
    overall_success = True
    
    # Run all validation steps
    if not test_directories():
        overall_success = False
        
    if not test_config():
        overall_success = False
        
    if not test_llama_cli():
        overall_success = False
        
    if not test_core_libs():
        overall_success = False
    
    test_optional_libs()  # Always succeeds
    
    if not test_spacy_model():
        overall_success = False
        
    if not test_fastembed_model():
        overall_success = False
    
    if PLATFORM == "linux":
        if not test_linux_system_packages():
            overall_success = False
    
    # Final result
    print("\n=== Validation Summary ===")
    if overall_success:
        print_status("All validations passed successfully!")
        print("\nYour installation is ready to use.")
        return 0
    else:
        print_status("Validation failed with errors", False)
        print("\nRecommendations:")
        print("1. Run installer again: python installer.py [windows|linux]")
        print("2. Check directory permissions")
        print("3. Verify internet connection during installation")
        if PLATFORM == "linux":
            print("4. Install system packages: sudo apt install python3-tk python3-venv")
        return 1

if __name__ == "__main__":
    sys.exit(main())