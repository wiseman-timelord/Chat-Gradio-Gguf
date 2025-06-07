# Script: `.\requisites.py` (Installer/validator)

# Imports
import os, json, platform, subprocess, sys, contextlib, time, copy
import pkg_resources
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import shutil

# Globals
APP_NAME = "Chat-Gradio-Gguf"
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
TEMP_DIR = BASE_DIR / "data/temp"
VULKAN_LEGACY = "1.1.126.0"  # Win7-8
VULKAN_MODERN = "1.4.304.1"  # Win8.1-11
VULKAN_TARGET = None
LLAMACPP_VER = "b5596"
BACKEND_TYPE = None
DIRS = ["data", "scripts", "models", "data/history", "data/temp"]
VULKAN_PATHS = [
    Path("C:/VulkanSDK"),
    Path("C:/Program Files/VulkanSDK"),
    Path("C:/Program Files (x86)/VulkanSDK"),
    Path("C:/programs/VulkanSDK"),
    Path("C:/program_files/VulkanSDK"),
    Path("C:/drivers/VulkanSDK")
]
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
BACKEND_OPTS = {
    "CPU AVX2": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_VER}/llama-{LLAMACPP_VER}-bin-win-avx2-x64.zip",
        "dest": "data/llama-avx2-bin",
        "cli": "data/llama-avx2-bin/llama-cli.exe",
        "needs_py": True
    },
    "CPU AVX512": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_VER}/llama-{LLAMACPP_VER}-bin-win-avx512-x64.zip",
        "dest": "data/llama-avx512-bin",
        "cli": "data/llama-avx512-bin/llama-cli.exe",
        "needs_py": True
    },
    "GPU Vulkan": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_VER}/llama-{LLAMACPP_VER}-bin-win-vulkan-x64.zip",
        "dest": "data/llama-vulkan-bin",
        "cli": "data/llama-vulkan-bin/llama-cli.exe",
        "needs_py": False,
        "vulkan": True
    },
    "GPU HIP-Radeon": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_VER}/llama-{LLAMACPP_VER}-bin-win-hip-radeon-x64.zip",
        "dest": "data/llama-hip-radeon-bin",
        "cli": "data/llama-hip-radeon-bin/llama-cli.exe",
        "needs_py": False,
        "vulkan": True
    },
    "GPU CUDA 11.7": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_VER}/llama-{LLAMACPP_VER}-bin-win-cuda-cu11.7-x64.zip",
        "dest": "data/llama-cuda-11.7-bin",
        "cli": "data/llama-cuda-11.7-bin/llama-cli.exe",
        "needs_py": False,
        "cuda": True
    },
    "GPU CUDA 12.4": {
        "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMACPP_VER}/llama-{LLAMACPP_VER}-bin-win-cuda-cu12.4-x64.zip",
        "dest": "data/llama-cuda-12.4-bin",
        "cli": "data/llama-cuda-12.4-bin/llama-cli.exe",
        "needs_py": False,
        "cuda": True
    }
}
CONFIG_TEMPLATE = """{
	"model_settings": {
	  "model_dir": "models",
	  "model_name": "",
	  "context_size": 8192,
	  "temperature": 0.66,
	  "repeat_penalty": 1.1,
	  "use_python_bindings": false,
	  "llama_cli_path": "data/llama-vulkan-bin/llama-cli.exe",
	  "vram_size": 8192,
	  "selected_gpu": null,
	  "selected_cpu": "null",
	  "mmap": true,
	  "mlock": true,
	  "n_batch": 2048,
	  "dynamic_gpu_layers": true,
	  "max_history_slots": 12,
	  "max_attach_slots": 6,
      "session_log_height": 500
	},
  "backend_config": {
    "backend_type": "GPU Vulkan",
    "llama_bin_path": "data/llama-vulkan-bin"
  }
}"""

# Console utilities
SEP_THICK = "="
SEP_THIN = "-"
CONSOLE_WIDTH = 80

def get_console_width() -> int:
    try:
        return 119 if shutil.get_terminal_size().columns >= 120 else 79
    except:
        return 79

CONSOLE_WIDTH = get_console_width()

def print_header(title: str) -> None:
    clear_screen()
    sep = SEP_THICK * CONSOLE_WIDTH
    print(f"{sep}\n    {APP_NAME}: {title}\n{sep}\n")

def print_separator(thick: bool = True) -> None:
    print((SEP_THICK if thick else SEP_THIN) * CONSOLE_WIDTH)

def print_status(msg: str, success: bool = True) -> None:
    status = "[GOOD]" if success else "[FAIL]"
    msg_width = CONSOLE_WIDTH - len(status) - 3
    trunc = (msg[:msg_width-3] + '...') if len(msg) > msg_width else msg
    print(f"{trunc.ljust(msg_width)} {status}")
    time.sleep(1 if success else 3)

# Utilities
def clear_screen() -> None:
    os.system('cls')

def get_windows_version() -> Tuple[int, int]:
    """Get Windows version with proper error handling"""
    try:
        ver_info = platform.win32_ver()
        if ver_info[1]:
            parts = ver_info[1].split('.')
            return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    except:
        pass
    
    try:
        win_ver = sys.getwindowsversion()
        return (win_ver.major, win_ver.minor)
    except:
        print_status("Warning: Assume Win10+", False)
        return (10, 0)

def determine_vulkan_version() -> str:
    major, minor = get_windows_version()
    if major < 6 or (major == 6 and minor < 3):
        ver = VULKAN_LEGACY
        print_status(f"Win{major}.{minor}: Vulkan {ver}")
    else:
        ver = VULKAN_MODERN
        print_status(f"Win{major}.{minor}: Vulkan {ver}")
    return ver

def init_vulkan() -> None:
    global VULKAN_TARGET
    VULKAN_TARGET = determine_vulkan_version()

# Configuration
def get_user_choice(prompt: str, options: list) -> str:
    print_header(prompt)
    print("\nMenu Options:")
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    print("    X. Exit Menu")
    print_separator()
    
    while True:
        choice = input(f"Selection; Options = 1-{len(options)}, Exit = X: ").strip().upper()
        if choice == "X":
            print("\nExiting...")
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid choice")

# Venv Management
@contextlib.contextmanager
def activate_venv():
    if not VENV_DIR.exists():
        raise FileNotFoundError(f"Venv missing: {VENV_DIR}")
    activate = VENV_DIR / "Scripts" / "activate.bat"
    if not activate.exists():
        raise FileNotFoundError(f"Activate missing: {activate}")
    old_path = os.environ["PATH"]
    old_py = sys.executable
    try:
        os.environ["PATH"] = f"{VENV_DIR / 'Scripts'}{os.pathsep}{old_path}"
        sys.executable = str(VENV_DIR / "Scripts" / "python.exe")
        yield
    finally:
        os.environ["PATH"] = old_path
        sys.executable = old_py

# Checks
def check_llama_conflicts() -> bool:
    """Check if llama-cpp-python works in venv"""
    venv_py = str(VENV_DIR / "Scripts" / "python.exe")
    if not Path(venv_py).exists():
        print_status("Venv python missing", False)
        return False
    
    try:
        result = subprocess.run(
            [venv_py, "-c", "import llama_cpp; print('OK')"],
            capture_output=True, text=True, check=True, timeout=30
        )
        if "OK" in result.stdout:
            print_status("llama-cpp works")
            return True
        else:
            print_status("llama-cpp import fail", False)
            return False
    except subprocess.CalledProcessError as e:
        if "ModuleNotFoundError" in e.stderr:
            print_status("llama-cpp not found", False)
        else:
            print_status("llama-cpp broken", False)
        return False
    except subprocess.TimeoutExpired:
        print_status("llama-cpp timeout", False)
        return False

def find_vulkan_versions() -> Dict[str, Path]:
    vulkan_vers = {}
    env_sdk = os.environ.get("VULKAN_SDK")
    env_base = None
    
    # Check env var first
    if env_sdk:
        env_path = Path(env_sdk)
        env_base = env_path.parent
        lib = env_path / "Lib/vulkan-1.lib"
        if lib.exists():
            ver = env_path.name
            vulkan_vers[ver] = env_path
            print_status(f"Found env: {ver}")
    
    # Check common paths
    for base in VULKAN_PATHS:
        if env_base and base.resolve() == env_base.resolve():
            continue  # Skip env path
        if base.exists():
            for sdk in base.iterdir():
                if sdk.is_dir():
                    ver = sdk.name
                    lib = sdk / "Lib/vulkan-1.lib"
                    if lib.exists() and ver not in vulkan_vers:
                        vulkan_vers[ver] = sdk
                        print_status(f"Found: {ver}")
    
    return vulkan_vers

def check_vulkan_support() -> bool:
    vulkan_vers = find_vulkan_versions()
    
    # Check exact match first
    if VULKAN_TARGET in vulkan_vers:
        print_status(f"Vulkan {VULKAN_TARGET} found")
        return True
    
    if not vulkan_vers:
        print_status("No Vulkan found", False)
        return False
    
    # Determine compatibility
    major, minor = get_windows_version()
    is_legacy = major < 6 or (major == 6 and minor < 3)
    compat_vers = []
    
    for ver in vulkan_vers:
        try:
            ver_parts = ver.split('.')
            if len(ver_parts) >= 2:
                major_ver = int(ver_parts[0])
                minor_ver = int(ver_parts[1])
                
                if is_legacy and major_ver == 1 and minor_ver == 1:
                    compat_vers.append(ver)
                elif not is_legacy and major_ver == 1 and minor_ver >= 2:
                    compat_vers.append(ver)
        except (ValueError, IndexError):
            continue  # Skip malformed versions
    
    # Show compatible options
    if compat_vers:
        print(f"\nCompatible Vulkan versions:")
        for i, ver in enumerate(compat_vers, 1):
            print(f" {i}. {ver} at {vulkan_vers[ver]}")
        
        while True:
            ch = input(f"\nChoose: [1-{len(compat_vers)}] use, [I] install, [Q] quit: ").strip().upper()
            if ch == "Q":
                print_status("Vulkan cancelled", False)
                sys.exit(0)
            elif ch == "I":
                return False
            elif ch.isdigit() and 1 <= int(ch) <= len(compat_vers):
                sel_ver = compat_vers[int(ch)-1]
                print_status(f"Using: {sel_ver}")
                return True
            print(" Invalid choice")
    
    # Show incompatible options
    print("\nIncompatible Vulkan found:")
    vers_list = list(vulkan_vers.items())
    for i, (ver, path) in enumerate(vers_list, 1):
        print(f" {i}. {ver} at {path}")
    
    while True:
        ch = input(f"\nChoose: [1-{len(vers_list)}] use anyway, [I] install, [Q] quit: ").strip().upper()
        if ch == "Q":
            print_status("Vulkan cancelled", False)
            sys.exit(0)
        elif ch == "I":
            return False
        elif ch.isdigit() and 1 <= int(ch) <= len(vers_list):
            sel_ver = vers_list[int(ch)-1][0]
            print_status(f"Using: {sel_ver} (risky)", False)
            time.sleep(2)
            return True
        print(" Invalid choice")

# Installations
def create_dirs() -> None:
    for dir_path in DIRS:
        path = BASE_DIR / dir_path
        path.mkdir(parents=True, exist_ok=True)
        try:
            test = path / "perm_test"
            test.touch()
            test.unlink()
            print_status(f"Verified: {dir_path}")
        except PermissionError:
            print_status(f"Perm denied: {dir_path}", False)
            sys.exit(1)

def create_config(backend: str) -> None:
    config_path = BASE_DIR / "data" / "persistence.json"  # Fixed: was persistent.json
    config = json.loads(CONFIG_TEMPLATE)
    info = BACKEND_OPTS[backend]
    config["backend_config"]["backend_type"] = backend
    config["backend_config"]["llama_bin_path"] = info["dest"]
    config["model_settings"]["llama_cli_path"] = info["cli"]
    config["model_settings"]["use_python_bindings"] = info["needs_py"]
    try:
        config_path.parent.mkdir(exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print_status("Config created")
    except Exception as e:
        print_status(f"Config fail", False)
        raise

def create_venv() -> bool:
    try:
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
            print_status("Removed old venv")
        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print_status("Created venv")
        py_exe = VENV_DIR / "Scripts" / "python.exe"
        if not py_exe.exists():
            raise FileNotFoundError(f"Python missing: {py_exe}")
        print_status("Venv verified")
        return True
    except Exception as e:
        print_status(f"Venv fail: {str(e)}", False)
        return False

def ensure_vulkan_init() -> None:
    """Initialize Vulkan once"""
    global VULKAN_TARGET
    if VULKAN_TARGET is None:
        VULKAN_TARGET = determine_vulkan_version()

def install_vulkan() -> bool:
    if VULKAN_TARGET is None:
        init_vulkan()

    install_paths = [base / VULKAN_TARGET for base in VULKAN_PATHS]
    
    print(f"\nInstall Vulkan {VULKAN_TARGET}:")
    for i, path in enumerate(install_paths, 1):
        print(f" {i}. {path}")
    
    while True:
        ch = input(f"Selection: Options = 1-{len(install_paths)}, Exit = X: ").strip().upper()
        if ch == "X":
            print_status("Vulkan cancelled", False)
            return False
        if ch.isdigit() and 1 <= int(ch) <= len(install_paths):
            path = install_paths[int(ch) - 1]
            break
        print(" Invalid choice")
    
    print_status(f"Installing to: {path}")
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print_status(f"Dir fail: {str(e)}", False)
        return False
    
    TEMP_DIR.mkdir(exist_ok=True)
    
    # Check if legacy version needed
    is_legacy_vulkan = VULKAN_TARGET.startswith("1.1.")
    
    if is_legacy_vulkan:
        return install_vulkan_choco(path)
    else:
        url = f"https://sdk.lunarg.com/sdk/download/{VULKAN_TARGET}/windows/VulkanSDK-{VULKAN_TARGET}-Installer.exe"
        inst_path = TEMP_DIR / "VulkanSDK.exe"
        return download_install_vulkan(url, inst_path, path)

def install_vulkan_choco(path: Path) -> bool:
    try:
        chk = subprocess.run(["choco", "--version"], capture_output=True, text=True)
        if chk.returncode != 0:
            print_status("Installing Chocolatey...")
            subprocess.run(
                ["powershell", "-Command", 
                 "Set-ExecutionPolicy Bypass -Scope Process -Force; "
                 "[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; "
                 "iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))"],
                check=True
            )
            print_status("Choco installed")
        
        print_status(f"Installing Vulkan via Choco...")
        try:
            subprocess.run(
                ["choco", "install", "vulkan-sdk", 
                 "--version", VULKAN_TARGET, 
                 "--force", "--yes"],
                check=True
            )
            print_status("Vulkan installed")
            
            default = Path("C:/Program Files (x86)/VulkanSDK") / VULKAN_TARGET
            if default.exists():
                try:
                    shutil.move(str(default), str(path))
                    print_status(f"Moved to: {path}")
                    return True
                except Exception:
                    print_status(f"Using default: {default}", False)
                    return True
            else:
                print_status(f"Not found: {default}", False)
                return False
        except subprocess.CalledProcessError:
            print_status("Choco install fail", False)
            return False
    except Exception:
        print_status("Choco error", False)
        return False

def download_install_vulkan(url: str, inst_path: Path, path: Path) -> bool:
    """Download and install Vulkan SDK"""
    try:
        import requests
        
        # Download installer
        print_status("Downloading installer...")
        for attempt in range(3):
            try:
                resp = requests.get(url, stream=True, timeout=60)
                resp.raise_for_status()
                
                size = int(resp.headers.get('content-length', 0))
                with open(inst_path, 'wb') as f:
                    downloaded = 0
                    for chunk in resp.iter_content(8192):
                        if chunk: 
                            f.write(chunk)
                            downloaded += len(chunk)
                            if size > 0 and downloaded % (5 * 1024 * 1024) == 0:  # Every 5MB
                                percent = (downloaded / size) * 100
                                print(f"  {percent:.1f}%")
                break
            except requests.exceptions.RequestException:
                print_status(f"Download fail {attempt + 1}", False)
                if attempt == 2: 
                    return False
                time.sleep(5)
        
        print_status("Download complete")
        
        # Validate downloaded file
        if not inst_path.exists() or inst_path.stat().st_size < 1024 * 1024:
            print_status("Download invalid", False)
            return False
        
        # Try installation methods
        methods = [
            {
                "name": "Custom path",
                "cmd": [str(inst_path), "--accept-licenses", "--confirm-command", "install", "--root", str(path)]
            },
            {
                "name": "Default install",
                "cmd": [str(inst_path), "--accept-licenses", "--confirm-command", "install"]
            },
            {
                "name": "Silent install",
                "cmd": [str(inst_path), "/S"]
            }
        ]
        
        for i, method in enumerate(methods, 1):
            print_status(f"Method {i}: {method['name']}")
            
            try:
                result = subprocess.run(
                    method["cmd"],
                    check=False, capture_output=True, text=True, timeout=600
                )
                
                if result.returncode == 0:
                    # Check installation locations
                    check_locations = [
                        path,
                        Path("C:/VulkanSDK") / VULKAN_TARGET,
                        Path("C:/Program Files/VulkanSDK") / VULKAN_TARGET,
                        Path("C:/Program Files (x86)/VulkanSDK") / VULKAN_TARGET
                    ]
                    
                    for loc in check_locations:
                        vulkan_info = loc / "Bin/vulkaninfo.exe"
                        vulkan_lib = loc / "Lib/vulkan-1.lib"
                        if vulkan_info.exists() and vulkan_lib.exists():
                            # Set environment variable
                            os.environ["VULKAN_SDK"] = str(loc)
                            print_status(f"Vulkan ready: {loc}")
                            return True
                
                print_status(f"Method {i} failed", False)
                
            except subprocess.TimeoutExpired:
                print_status(f"Method {i} timeout", False)
            except Exception:
                print_status(f"Method {i} error", False)
        
        # All methods failed
        print_status("Install failed", False)
        print(f"\nManual steps required:")
        print(f"1. Run: {inst_path}")
        print(f"2. Set VULKAN_SDK env var")
        print(f"3. Re-run installer")
        return False
        
    except ImportError:
        print_status("requests missing", False)
        return False
    except PermissionError:
        print_status("Need admin rights", False)
        return False
    except Exception:
        print_status("Vulkan install error", False)
        return False
    finally:
        # Cleanup downloaded file
        if inst_path.exists():
            try:
                inst_path.unlink()
            except:
                pass
        
def download_backend(backend: str) -> bool:
    """Download and extract backend"""
    print_status(f"Downloading {backend}...")
    info = BACKEND_OPTS[backend]
    
    if not info:
        print_status("Invalid backend", False)
        return False
    
    TEMP_DIR.mkdir(exist_ok=True)
    zip_path = TEMP_DIR / "llama.zip"
    
    try:
        import requests
        
        # Download with validation
        for attempt in range(3):
            try:
                print_status(f"Download try {attempt + 1}")
                resp = requests.get(info["url"], stream=True, timeout=120)
                resp.raise_for_status()
                
                size = int(resp.headers.get('content-length', 0))
                if size < 1024 * 1024:  # Less than 1MB suspicious
                    print_status("Download too small", False)
                    continue
                
                with open(zip_path, 'wb') as f:
                    downloaded = 0
                    for chunk in resp.iter_content(8192):
                        if chunk: 
                            f.write(chunk)
                            downloaded += len(chunk)
                            if size > 0 and downloaded % (5 * 1024 * 1024) == 0:  # Every 5MB
                                percent = (downloaded / size) * 100
                                print(f"  {percent:.1f}%")
                
                # Validate download
                if zip_path.stat().st_size < 1024 * 1024:
                    print_status("File too small", False)
                    continue
                    
                break
                
            except requests.exceptions.RequestException:
                print_status(f"Download fail {attempt + 1}", False)
                if attempt == 2: 
                    return False
                time.sleep(5)
        
        print_status("Download complete")
        
        # Extract with cleanup
        import zipfile
        dest_dir = BASE_DIR / info["dest"]
        
        # Remove existing
        if dest_dir.exists():
            try:
                shutil.rmtree(dest_dir)
                print_status("Removed old backend")
            except Exception:
                print_status("Cleanup warning", False)
        
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                # Check for suspicious files
                for member in zipf.namelist():
                    if member.startswith('../') or member.startswith('/'):
                        print_status("Unsafe archive", False)
                        return False
                zipf.extractall(dest_dir)
            print_status("Backend extracted")
        except zipfile.BadZipFile:
            print_status("Corrupted archive", False)
            return False
        
        # Verify installation
        cli_path = BASE_DIR / info["cli"]
        if not cli_path.exists():
            print_status("llama-cli missing", False)
            return False
        
        print_status(f"{backend} ready")
        return True
        
    except ImportError:
        print_status("requests missing", False)
        return False
    except Exception as e:
        print_status("Backend error", False)
        return False
    finally:
        # Always cleanup
        if zip_path.exists():
            try:
                zip_path.unlink()
            except:
                pass

def install_deps(backend: str) -> bool:
    """Install Python dependencies"""
    print_status("Installing deps...")
    try:
        pip_exe = VENV_DIR / "Scripts" / "pip.exe"
        py_exe = VENV_DIR / "Scripts" / "python.exe"
        
        if not pip_exe.exists() or not py_exe.exists():
            print_status("Pip/Python missing", False)
            return False
        
        pip = str(pip_exe)
        py = str(py_exe)
        
        # Upgrade pip first
        print_status("Upgrading pip...")
        subprocess.run([py, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, timeout=180)
        
        # Install requirements with proper index
        print_status("Installing packages...")
        cmd = [pip, "install", "--no-warn-script-location", 
               "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu",
               "--prefer-binary", "--timeout", "300"] + REQS
        
        result = subprocess.run(cmd, check=True, timeout=900, 
                              capture_output=True, text=True)
        
        print_status("Deps installed")
        
        # Quick verification of core packages
        print_status("Verifying install...")
        test_imports = [
            "gradio", "requests", "llama_cpp", "psutil", "yake"
        ]
        
        for pkg in test_imports:
            result = subprocess.run([
                py, "-c", f"import {pkg}; print('OK')"
            ], capture_output=True, text=True, timeout=10)
            
            if "OK" not in result.stdout:
                print_status(f"{pkg} verify fail", False)
                return False
        
        print_status("All deps verified")
        return True
                
    except subprocess.CalledProcessError as e:
        print_status("Install failed", False)
        if e.stderr:
            print(f"Error: {e.stderr[:200]}")  # Show first 200 chars
        return False
    except subprocess.TimeoutExpired:
        print_status("Install timeout", False)
        return False
    except Exception as e:
        print_status("Deps error", False)
        return False

# Main Flow
def install():
    print("Proceeding with Installation...")
    if platform.system() != "Windows":
        print_status("Windows only", False)
        sys.exit(1)
    if sys.version_info < (3, 8):
        print_status("Python 3.8+ required", False)
        sys.exit(1)
    
    try:
        print_status("Creating dirs...")
        create_dirs()
        
        print_status("Backend: " + BACKEND_TYPE)
        info = BACKEND_OPTS[BACKEND_TYPE]
        
        # Check Vulkan requirements
        if info.get("vulkan"):
            ensure_vulkan_init()
            if not check_vulkan_support():
                if not install_vulkan():
                    print_status("Vulkan install fail", False)
                    sys.exit(1)
        
        print_status("Creating venv...")
        if not create_venv():
            sys.exit(1)
        
        # Install dependencies
        with activate_venv():
            if not install_deps(BACKEND_TYPE):
                sys.exit(1)
            if info.get("needs_py") and not check_llama_conflicts():
                sys.exit(1)
        
        print_status("Installing backend...")
        if not download_backend(BACKEND_TYPE):
            sys.exit(1)
        
        print_status("Creating config...")
        create_config(BACKEND_TYPE)
        
        print_status(f"\n{APP_NAME} ready!")
        time.sleep(2)
        
    except KeyboardInterrupt:
        print_status("\nInstall cancelled", False)
        sys.exit(1)
    except Exception as e:
        print_status("Install failed", False)
        sys.exit(1)

# Menus
def select_backend() -> None:
    global BACKEND_TYPE
    print("Backend Options:")
    print("    1. CPU AVX2 - AVX2 compatible")
    print("    2. CPU AVX512 - AVX512 compatible")
    print("    3. GPU Vulkan - AMD/nVidia/Intel")
    print("    4. GPU HIP-Radeon - AMD-ROCM experimental")
    print("    5. GPU CUDA 11.7 - NVIDIA CUDA 11.7")
    print("    6. GPU CUDA 12.4 - NVIDIA CUDA 12.4")
    print("    X. Exit Installation")
    
    mapping = {
        1: "CPU AVX2",
        2: "CPU AVX512",
        3: "GPU Vulkan",
        4: "GPU HIP-Radeon",
        5: "GPU CUDA 11.7",
        6: "GPU CUDA 12.4"
    }
    
    while True:
        choice = input("\nSelection; Options = 1-6, Exit = X: ").strip().upper()
        if choice == "X":
            print("\nExiting installation...")
            sys.exit(0)
        if choice.isdigit() and 1 <= int(choice) <= 6:
            BACKEND_TYPE = mapping[int(choice)]
            return
        print("Invalid choice")

# Entry
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python requisites.py installer")
        sys.exit(1)
    if sys.argv[1] == "installer":
        print_header("Requirements Installation")
        try:
            select_backend()
            install()
        except KeyboardInterrupt:
            print("\nCancelled")
            sys.exit(1)
    else:
        print("Invalid argument. Use 'installer'")
        sys.exit(1)