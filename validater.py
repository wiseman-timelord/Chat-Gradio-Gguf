# validation.py
import os, sys, subprocess
from pathlib import Path

# Global paths
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"

# Requirements list
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
    "lxml_html_clean",
]

def print_status(msg: str, success: bool = True) -> None:
    """Simplified status printer"""
    status = "OK" if success else "FAIL"
    print(f"{msg}: {status}")

def test_libs():
    """Test if all required libraries are properly installed"""
    print("=== Library Validation ===")
    
    if not VENV_DIR.exists():
        print_status("Venv not found", False)
        return 1
    
    venv_py = VENV_DIR / "Scripts" / "python.exe"
    if not venv_py.exists():
        print_status("Python missing", False)
        return 1
    
    failed_libs = []
    
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
        "lxml_html_clean": "lxml_html_clean"
    }
    
    # Test each requirement
    for req in REQS:
        pkg_name = req.split('>=')[0].split('==')[0].split('<')[0].strip()
        import_name = import_names.get(pkg_name, pkg_name.replace('-', '_'))
        
        try:
            result = subprocess.run([
                str(venv_py), "-c", f"import {import_name}; print('OK')"
            ], capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0 and "OK" in result.stdout:
                print_status(pkg_name)
            else:
                print_status(pkg_name, False)
                failed_libs.append(pkg_name)
                
        except subprocess.TimeoutExpired:
            print_status(f"{pkg_name} timeout", False)
            failed_libs.append(pkg_name)
        except Exception as e:
            print_status(f"{pkg_name} error", False)
            failed_libs.append(pkg_name)
    
    if failed_libs:
        print(f"\n{len(failed_libs)} libraries failed validation:")
        print(", ".join(failed_libs))
        print(f"Fix with: {VENV_DIR / 'Scripts' / 'pip.exe'} install [package]")
        return 1
    
    print("\nAll libraries validated successfully")
    return 0

if __name__ == "__main__":
    sys.exit(test_libs())