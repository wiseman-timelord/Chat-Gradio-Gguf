# Script: validator.py

# Imports
import os
import sys
import subprocess
from pathlib import Path

# Global paths
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"

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
        "lxml_html_clean",
        "pyttsx3"  # Added
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
        "lxml_html_clean",
        "pyttsx3",  # Added
        "pyobjc"    # Added
    ]

def set_platform():
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ["windows", "linux"]:
        print("ERROR: Platform argument required (windows/linux)")
        sys.exit(1)
    temporary.PLATFORM = sys.argv[1].lower()

def print_status(msg: str, success: bool = True) -> None:
    """Simplified status printer"""
    status = "✓" if success else "✗"
    print(f"  {status} {msg}")

def test_libs():
    """Test if all required libraries are properly installed"""
    print("=== Library Validation ===")
    
    if not VENV_DIR.exists():
        print_status("Virtual environment not found", False)
        return 1
    
    # Set venv_py based on platform
    if PLATFORM == "windows":
        venv_py = VENV_DIR / "Scripts" / "python.exe"
    else:
        venv_py = VENV_DIR / "bin" / "python"
    
    if not venv_py.exists():
        print_status("Python executable missing", False)
        return 1
    
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
        "lxml_html_clean": "lxml_html_clean",
        "pyttsx3": "pyttsx3",  # Added
        "pyobjc": "objc"       # Added
    }
    
    # Test each requirement
    failed_libs = []
    print("Checking packages:")
    
    for req in REQS:
        pkg_name = req.split('>=')[0].split('==')[0].split('<')[0].strip()
        import_name = import_names.get(pkg_name, pkg_name.replace('-', '_'))
        
        try:
            cmd = f"import {import_name}; print('OK')"
            result = subprocess.run(
                [str(venv_py), "-c", cmd],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0 and "OK" in result.stdout:
                print_status(pkg_name)
            else:
                print_status(pkg_name, False)
                failed_libs.append(pkg_name)
                
        except subprocess.TimeoutExpired:
            print_status(f"{pkg_name} (timeout)", False)
            failed_libs.append(pkg_name)
        except Exception as e:
            print_status(f"{pkg_name} (error)", False)
            failed_libs.append(pkg_name)
    
    # Check backend-specific features
    if PLATFORM == "linux":
        print("\nChecking backend support:")
        try:
            # Test Vulkan support
            cmd = "from llama_cpp.llama_vulkan import llava_vulkan_init; print('OK')"
            result = subprocess.run(
                [str(venv_py), "-c", cmd],
                capture_output=True,
                text=True,
                timeout=10
            )
            print_status("Vulkan support", result.returncode == 0)
            
            # Test CUDA support
            cmd = "from llama_cpp.llama_cuda import llava_cuda_init; print('OK')"
            result = subprocess.run(
                [str(venv_py), "-c", cmd],
                capture_output=True,
                text=True,
                timeout=10
            )
            print_status("CUDA support", result.returncode == 0)
            
        except Exception:
            print_status("Backend check failed", False)
    
    # Final results
    if failed_libs:
        print(f"\n{len(failed_libs)} libraries failed validation:")
        print(", ".join(failed_libs))
        pip_path = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("pip.exe" if PLATFORM == "windows" else "pip")
        print(f"Fix with: {pip_path} install [package]")
        return 1
    
    print("\nAll libraries validated successfully")
    return 0

if __name__ == "__main__":
    set_platform()
    sys.exit(test_libs())