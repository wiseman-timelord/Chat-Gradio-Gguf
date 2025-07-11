# Script: validator.py

# Imports
import os
import sys
import subprocess
import glob
import json
from pathlib import Path

# Global paths
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
CONFIG_PATH = BASE_DIR / "data" / "persistent.json"

# Platform setup
if len(sys.argv) < 2 or sys.argv[1].lower() not in ["windows", "linux"]:
    print("ERROR: Platform argument required (windows/linux)")
    sys.exit(1)
PLATFORM = sys.argv[1].lower()

# Requirements list based on platform
if PLATFORM == "windows":
    REQS = [
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
        "lxml[html_clean]",  # Replaced lxml_html_clean
        "pyttsx3",
        "tk"     # Added for file dialogs
    ]
elif PLATFORM == "linux":
    REQS = [
        "gradio>=4.25.0",
        "requests==2.31.0",
        "pyperclip",
        "yake",
        "psutil",
        "duckduckgo-search",
        "newspaper3k",
        "llama-cpp-python",
        "langchain-community==0.3.18",
        "pygments==2.17.2",
        "lxml",  # Replaced lxml_html_clean
        "pyttsx3"
    ]
    # Note: python3-tk is a system package on Linux

def print_status(msg: str, success: bool = True) -> None:
    """Simplified status printer"""
    status = "✓" if success else "✗"
    print(f"  {status} {msg}")

def find_llama_cli() -> Path:
    """Search for llama-cli in expected locations"""
    # Look in all llama-* directories under data
    search_pattern = str(BASE_DIR / "data" / "llama-*" / "llama-cli")
    candidates = glob.glob(search_pattern)
    
    if candidates:
        return Path(candidates[0])
    
    # Also check the build directory for Linux
    if PLATFORM == "linux":
        linux_build = BASE_DIR / "data" / "llama-bin" / "build" / "bin" / "llama-cli"
        if linux_build.exists():
            return linux_build
            
    return None

def test_directories():
    """Verify required directories exist"""
    print("=== Directory Validation ===")
    success = True
    
    required_dirs = [
        BASE_DIR / "data",
        BASE_DIR / "scripts",
        BASE_DIR / "models",
        BASE_DIR / "data/history",
        BASE_DIR / "data/temp"
    ]
    
    for dir_path in required_dirs:
        if dir_path.exists() and dir_path.is_dir():
            print_status(f"Directory exists: {dir_path}")
        else:
            print_status(f"Missing directory: {dir_path}", False)
            success = False
            
    return success

def test_config():
    """Verify configuration file exists and is valid"""
    print("\n=== Configuration Validation ===")
    
    if not CONFIG_PATH.exists():
        print_status("Configuration file (persistent.json) missing!", False)
        return False
        
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
            
        # Validate backend type
        backend = config.get("backend_config", {}).get("backend_type", "")
        if backend not in ["GPU/CPU - Vulkan", "GPU/CPU - HIP-Radeon", "GPU/CPU - CUDA 11.7", 
                          "GPU/CPU - CUDA 12.4", "GPU/CPU - CUDA"]:
            print_status(f"Invalid backend type: {backend}", False)
            return False
            
        print_status("Configuration file is valid")
        return True
        
    except Exception as e:
        print_status(f"Configuration validation failed: {str(e)}", False)
        return False

def test_llama_cli():
    """Verify llama-cli exists and is executable"""
    print("\n=== Backend Validation ===")
    
    llama_cli_path = find_llama_cli()
    
    if not llama_cli_path:
        print_status("llama-cli binary not found!", False)
        return False
        
    if not llama_cli_path.exists():
        print_status(f"llama-cli not found at: {llama_cli_path}", False)
        return False
        
    # Check executability on Linux
    if PLATFORM == "linux":
        if not os.access(llama_cli_path, os.X_OK):
            print_status("llama-cli is not executable", False)
            return False
        
        # For Linux, just verify the file exists and is executable
        print_status(f"llama-cli verified at: {llama_cli_path}")
        return True
    else:
        # For Windows, try to run with --help
        try:
            result = subprocess.run(
                [str(llama_cli_path), "--help"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5
            )
            if result.returncode == 0:
                print_status(f"llama-cli verified at: {llama_cli_path}")
                return True
        except Exception:
            pass
            
        print_status("llama-cli failed to execute", False)
        return False

def test_libs():
    """Test if all required libraries are properly installed"""
    print("\n=== Library Validation ===")
    
    if not VENV_DIR.exists():
        print_status("Virtual environment not found", False)
        return False
    
    # Set venv_py based on platform
    venv_py = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
    
    if not venv_py.exists():
        print_status("Python executable missing", False)
        return False
    
    # Package name to import name mapping
    import_names = {
        "gradio": "gradio",
        "requests": "requests", 
        "pyperclip": "pyperclip",
        "yake": "yake",
        "psutil": "psutil",
        "pywin32": "win32api",
        "duckduckgo-search": "duckduckgo_search",
        "newspaper3k": "newspaper",
        "llama-cpp-python": "llama_cpp",
        "langchain-community": "langchain_community",
        "pygments": "pygments",
        "lxml[html_clean]": "lxml",
        "pyttsx3": "pyttsx3",
        "tkinter": "tkinter"
    }
    
    # Platform-specific adjustments
    requirements = REQS.copy()
    if PLATFORM == "windows":
        import_names["tk"] = "tkinter"
    else:
        requirements = [req for req in requirements if req != "tk"]
    
    # Test each requirement
    success = True
    print("Checking packages:")
    
    for req in requirements:
        pkg_name = req.split('>=')[0].split('==')[0].split('<')[0].strip()
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
    print(f'MISSING DEPENDENCY: {str(e)}')
except Exception as e:
    print(f'ERROR: {str(e)}')
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
            print_status(f"{pkg_name} (timeout during import)", False)
            success = False
        except Exception as e:
            print_status(f"{pkg_name} (validation error: {str(e)})", False)
            success = False
    
    # Enhanced tkinter validation for Linux
    if PLATFORM == "linux":
        print("\nChecking Linux system packages:")
        try:
            # Check for both python3-tk and python3.13-tk
            tk_packages = ["python3-tk", "python3.13-tk"]
            found = False
            
            for pkg in tk_packages:
                result = subprocess.run(
                    ["dpkg", "-s", pkg],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                if result.returncode == 0:
                    print_status(f"{pkg} (system)")
                    found = True
                    break
            
            if not found:
                print_status("python3-tk/python3.13-tk not found!", False)
                success = False
            
            # Verify tkinter is actually importable
            try:
                import_result = subprocess.run(
                    [str(venv_py), "-c", "import tkinter; tkinter.Tk(); print('OK')"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if import_result.returncode == 0 and "OK" in import_result.stdout:
                    print_status("tkinter (functional)")
                else:
                    error = import_result.stderr.strip()
                    print_status(f"tkinter - {error}", False)
                    success = False
            except subprocess.TimeoutExpired:
                print_status("tkinter (timeout during test)", False)
                success = False
                
        except Exception as e:
            print_status(f"System package check failed: {str(e)}", False)
            success = False
    
    return success

def main():
    """Main validation routine"""
    overall_success = True
    
    # Run all validation steps
    if not test_directories():
        overall_success = False
        
    if not test_config():
        overall_success = False
        
    if not test_llama_cli():
        overall_success = False
        
    if not test_libs():
        overall_success = False
    
    # Final result
    print("\n=== Validation Summary ===")
    if overall_success:
        print_status("All validations passed successfully!")
        return 0
    else:
        print_status("Validation failed with errors", False)
        print("\nRecommendations:")
        print("1. Run the installer again: python installer.py [windows|linux]")
        print("2. Check directory permissions")
        print("3. Verify internet connection during installation")
        if PLATFORM == "linux":
            print("4. Install missing system packages: sudo apt install python3-tk")
        return 1

if __name__ == "__main__":
    sys.exit(main())