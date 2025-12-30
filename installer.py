# Script: installer.py (Installation script for Chat-Gradio-Gguf)

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

# Constants / Variables ...
_PY_TAG = f"cp{sys.version_info.major}{sys.version_info.minor}"
APP_NAME = "Chat-Gradio-Gguf"
BASE_DIR = Path(__file__).parent
VENV_DIR = BASE_DIR / ".venv"
LLAMACPP_GIT_REPO = "https://github.com/ggerganov/llama.cpp.git"
LLAMACPP_PYTHON_GIT_REPO = "https://github.com/abetlen/llama-cpp-python.git"
LLAMACPP_TARGET_VERSION = "b7358"
DOWNLOAD_RELEASE_TAG = "b7358" 
WIN_COMPILE_TEMP = Path("C:/temp_compile")      # fixed Windows build folder
LINUX_COMPILE_TEMP = None                       # Linux keeps using project-local temp
_INSTALL_PROCESSES = set()
_DID_COMPILATION = False 
_PRE_EXISTING_PROCESSES = {} 

# Maps/Lists...
DIRECTORIES = [
    "data", "scripts", "models",
    "data/history", "data/temp", "data/vectors",
    "data/fastembed_cache"
]

# Platform detection (windows / linux)
PLATFORM = None

# Functions...
def set_platform() -> None:
    global PLATFORM
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ["windows", "linux"]:
        print("ERROR: Platform argument required (windows/linux)")
        sys.exit(1)
    PLATFORM = sys.argv[1].lower()

set_platform()

# Set TEMP_DIR based on platform (Windows uses short path to avoid 260 char limit)
if PLATFORM == "windows":
    TEMP_DIR = WIN_COMPILE_TEMP                 # always the same folder
else:
    TEMP_DIR = BASE_DIR / "data" / "temp"
# Backend definitions
if PLATFORM == "windows":
    BACKEND_OPTIONS = {
        # Option 1: Download CPU binary + Download CPU wheel
        "Download CPU Binaries / Download CPU Wheel": {
            "url": None,  # CPU binaries not used anymore
            "dest": None,
            "cli_path": None,
            "needs_python_bindings": True,
            "compile_binary": False,
            "compile_wheel": False,
            "vulkan_required": False,
            "build_flags": {}
        },
        
        # Option 2: Compile CPU binary + Compile CPU wheel
        "Compile CPU Binaries / Compile CPU Wheel": {
            "url": None,
            "dest": "data/llama-cpu-bin",
            "cli_path": "data/llama-cpu-bin/llama-cli.exe",
            "needs_python_bindings": True,
            "compile_binary": True,
            "compile_wheel": True,
            "vulkan_required": False,
            "build_flags": {}  # Will auto-detect AVX2, F16C, FMA
        },
        
        # Option 3: Download Vulkan binary + Download CPU wheel
        "Download Vulkan Bin / Download CPU Wheel": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{DOWNLOAD_RELEASE_TAG}/llama-{DOWNLOAD_RELEASE_TAG}-bin-win-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli.exe",
            "needs_python_bindings": True,
            "compile_binary": False,
            "compile_wheel": False,
            "vulkan_required": True,
            "build_flags": {}
        },
        
        # Option 4: Download Vulkan binary + Download CPU wheel (Forced)
        "Download Vulkan Bin / Download CPU Wheel (Forced)": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{DOWNLOAD_RELEASE_TAG}/llama-{DOWNLOAD_RELEASE_TAG}-bin-win-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli.exe",
            "needs_python_bindings": True,
            "compile_binary": False,
            "compile_wheel": False,
            "vulkan_required": False,  # Skip checks
            "build_flags": {}
        },
        
        # Option 5: Download Vulkan binary + Compile Vulkan wheel
        "Download Vulkan Bin / Compile Vulkan Wheel": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{DOWNLOAD_RELEASE_TAG}/llama-{DOWNLOAD_RELEASE_TAG}-bin-win-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli.exe",
            "needs_python_bindings": True,
            "compile_binary": False,
            "compile_wheel": True,
            "vulkan_required": True,
            "build_flags": {"GGML_VULKAN": "1"}
        },
        
        # Option 6: Compile Vulkan binary + Compile Vulkan wheel
        "Compile Vulkan Binaries / Compile Vulkan Wheel": {
            "url": None,
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli.exe",
            "needs_python_bindings": True,
            "compile_binary": True,
            "compile_wheel": True,
            "vulkan_required": True,
            "build_flags": {"GGML_VULKAN": "1"}  # Will also auto-detect AVX2, F16C, FMA
        }
    }
else:  # Linux - mirror structure
    BACKEND_OPTIONS = {
        # Option 1: Download CPU binary + Download CPU wheel
        "Download CPU Binaries / Download CPU Wheel": {
            "url": None,
            "dest": None,
            "cli_path": None,
            "needs_python_bindings": True,
            "compile_binary": False,
            "compile_wheel": False,
            "vulkan_required": False,
            "build_flags": {}
        },
        
        # Option 2: Compile CPU binary + Compile CPU wheel
        "Compile CPU Binaries / Compile CPU Wheel": {
            "url": None,
            "dest": "data/llama-cpu-bin",
            "cli_path": "data/llama-cpu-bin/llama-cli",
            "needs_python_bindings": True,
            "compile_binary": True,
            "compile_wheel": True,
            "vulkan_required": False,
            "build_flags": {}
        },
        
        # Option 3: Download Vulkan binary + Download CPU wheel
        "Download Vulkan Bin / Download CPU Wheel": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{DOWNLOAD_RELEASE_TAG}/llama-{DOWNLOAD_RELEASE_TAG}-bin-ubuntu-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli",
            "needs_python_bindings": True,
            "compile_binary": False,
            "compile_wheel": False,
            "vulkan_required": True,
            "build_flags": {}
        },
        
        # Option 4: Download Vulkan binary + Download CPU wheel (Forced)
        "Download Vulkan Bin / Download CPU Wheel (Forced)": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{DOWNLOAD_RELEASE_TAG}/llama-{DOWNLOAD_RELEASE_TAG}-bin-ubuntu-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli",
            "needs_python_bindings": True,
            "compile_binary": False,
            "compile_wheel": False,
            "vulkan_required": False,
            "build_flags": {}
        },
        
        # Option 5: Download Vulkan binary + Compile Vulkan wheel
        "Download Vulkan Bin / Compile Vulkan Wheel": {
            "url": f"https://github.com/ggml-org/llama.cpp/releases/download/{DOWNLOAD_RELEASE_TAG}/llama-{DOWNLOAD_RELEASE_TAG}-bin-ubuntu-vulkan-x64.zip",
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli",
            "needs_python_bindings": True,
            "compile_binary": False,
            "compile_wheel": True,
            "vulkan_required": True,
            "build_flags": {"GGML_VULKAN": "1"}
        },
        
        # Option 6: Compile Vulkan binary + Compile Vulkan wheel
        "Compile Vulkan Binaries / Compile Vulkan Wheel": {
            "url": None,
            "dest": "data/llama-vulkan-bin",
            "cli_path": "data/llama-vulkan-bin/llama-cli",
            "needs_python_bindings": True,
            "compile_binary": True,
            "compile_wheel": True,
            "vulkan_required": True,
            "build_flags": {"GGML_VULKAN": "1"}
        }
    }


# Python requirements (CPU-only, no torch)
BASE_REQ = [
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
    # onnxruntime and fastembed will be installed separately with special handling
    "tokenizers",
]

if PLATFORM == "windows":
    BASE_REQ.extend(["pywin32", "tk"])

def snapshot_pre_existing_processes() -> None:
    """Snapshot all build-related processes that exist BEFORE compilation starts"""
    global _PRE_EXISTING_PROCESSES
    
    if PLATFORM != "windows":
        return
    
    try:
        import psutil
    except ImportError:
        return  # psutil not available yet, skip snapshot
    
    build_process_names = [
        "conhost.exe",
        "MSBuild.exe", 
        "VBCSCompiler.exe",
        # ... rest unchanged
    ]
    
    _PRE_EXISTING_PROCESSES = {}
    
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] in build_process_names:
                _PRE_EXISTING_PROCESSES[proc.info['pid']] = proc.info['name']
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

def track_process(pid: int) -> None:
    """Track a process and all its current and future descendants."""
    global _INSTALL_PROCESSES
    try:
        import psutil
        parent = psutil.Process(pid)
        # Immediately add the process and all its current children recursively
        def add_descendants(p):
            _INSTALL_PROCESSES.add(p.pid)
            for child in p.children(recursive=True):
                _INSTALL_PROCESSES.add(child.pid)
        add_descendants(parent)
    except (ImportError, Exception):
        _INSTALL_PROCESSES.add(pid)

def print_status(message: str, success: bool = True) -> None:
    status = "[✓]" if success else "[✗]"
    print(f"{status} {message}")
    time.sleep(1 if success else 3)

# Utility helpers
def clean_compile_temp() -> None:
    """Wipe the fixed compile folder (Windows only), handling read-only files."""
    if PLATFORM == "windows":
        if WIN_COMPILE_TEMP.exists():
            try:
                # Handle read-only files (common in .git folders)
                def handle_remove_readonly(func, path, exc):
                    """Error handler for rmtree to handle read-only files"""
                    import stat
                    if not os.access(path, os.W_OK):
                        os.chmod(path, stat.S_IWUSR | stat.S_IREAD)
                        func(path)
                    else:
                        raise
                
                shutil.rmtree(WIN_COMPILE_TEMP, onerror=handle_remove_readonly)
                print_status(f"Removed temp folder {WIN_COMPILE_TEMP}")
            except Exception as e:
                print(f"Warning: Could not fully remove {WIN_COMPILE_TEMP}: {e}")
                print("  Continuing anyway...")

def cleanup_build_processes() -> None:
    """Force-terminate all build-related processes if compilation occurred."""
    global _DID_COMPILATION
    if not _DID_COMPILATION or PLATFORM != "windows":
        return

    try:
        import psutil
    except ImportError:
        return  # psutil not available, skip cleanup
    
    current_pid = os.getpid()
    try:
        current_proc = psutil.Process(current_pid)
        current_conhost = None

        # Find the conhost.exe belonging to current console
        try:
            # Parent is cmd.exe/PowerShell.exe; its child is our conhost
            parent = current_proc.parent()
            if parent:
                for child in parent.children():
                    if child.name().lower() == "conhost.exe":
                        current_conhost = child.pid
                        break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

        protected_pids = {current_pid}
        if current_conhost:
            protected_pids.add(current_conhost)

        to_kill = []

        # Collect all MSBuild.exe and conhost.exe (except current)
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                if proc.info['pid'] in protected_pids:
                    continue
                name = proc.info['name'].lower()
                if name == "msbuild.exe":
                    to_kill.append(proc.info['pid'])
                elif name == "conhost.exe":
                    to_kill.append(proc.info['pid'])
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if not to_kill:
            return

        # Terminate first
        for pid in to_kill:
            try:
                p = psutil.Process(pid)
                p.terminate()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        time.sleep(1)

        # Force kill leftovers
        for pid in to_kill:
            try:
                p = psutil.Process(pid)
                if p.is_running():
                    p.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
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
    """Display menu exactly as requested"""
    print_header("Gpu Options")
    print("\n")
    for i, option in enumerate(options, 1):
        print(f"    {i}) {option}\n")
    print("\n\n")
    print("-" * (shutil.get_terminal_size().columns - 1))

    while True:
        choice = input(f"Selecton; Menu Options 1-{len(options)}, Abandon Install = A: ").strip().upper()
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

def create_directories() -> None:
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
    
    # Create temp dir separately (may be outside project on Windows)
    if PLATFORM == "windows":
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        print_status(f"Using short temp path: {TEMP_DIR}")

def get_optimal_build_threads() -> int:
    """Calculate optimal thread count for building - 85% of available cores"""
    import multiprocessing
    
    try:
        total_threads = multiprocessing.cpu_count()
    except:
        total_threads = 4  # fallback
    
    # Use 85% of threads as requested
    use_threads = max(1, int(total_threads * 0.85))
    
    return use_threads

def build_llama_cpp_python_with_flags(build_flags: dict) -> bool:
    """
    Build llama-cpp-python from latest git master with optimal CPU flags.
    Auto-detects AVX2, FMA, F16C for maximum performance.
    Includes retry logic for git operations with resume capability.
    """
    global _DID_COMPILATION
    _DID_COMPILATION = True
    
    # Snapshot processes before build starts
    snapshot_pre_existing_processes()
    
    print_status("Building llama-cpp-python from source (this takes 10-20 minutes)")
    
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                    ("python.exe" if PLATFORM == "windows" else "python"))
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                 ("pip.exe" if PLATFORM == "windows" else "pip"))
    
    build_threads = get_optimal_build_threads()
    print(f"  Using {build_threads} parallel build threads")
    
    # Set up environment for parallel compilation
    env = os.environ.copy()
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(build_threads)
    env["GIT_PROGRESS"] = "1"
    
    if PLATFORM == "windows":
        env["FORCE_CMAKE"] = "1"
        env["CL"] = f"/MP{build_threads}"
    else:
        env["MAKEFLAGS"] = f"-j{build_threads}"
        env["NINJAFLAGS"] = f"-j{build_threads}"
    
    # Detect CPU features for optimization
    print_status("Detecting CPU features...")
    cpu_features = detect_cpu_features()
    
    print("CPU Features detected:")
    for feature, supported in cpu_features.items():
        status = "✓" if supported else "✗"
        print(f"  {status} {feature}")
    
    # Build CMAKE_ARGS with CPU optimizations
    cmake_args = []
    
    # User-specified flags (Vulkan, etc.)
    if build_flags:
        cmake_args.extend([f"-D{k}={v}" for k, v in build_flags.items()])
    
    # CPU optimization flags (CRITICAL for performance)
    cmake_args.append(f"-DGGML_AVX={'ON' if cpu_features['AVX'] else 'OFF'}")
    cmake_args.append(f"-DGGML_AVX2={'ON' if cpu_features['AVX2'] else 'OFF'}")
    cmake_args.append(f"-DGGML_AVX512={'ON' if cpu_features['AVX512'] else 'OFF'}")
    cmake_args.append(f"-DGGML_FMA={'ON' if cpu_features['FMA'] else 'OFF'}")
    cmake_args.append(f"-DGGML_F16C={'ON' if cpu_features['F16C'] else 'OFF'}")
    cmake_args.append("-DGGML_OPENMP=ON")
    
    # Parallel compilation flags - NOW APPLIED TO LINUX TOO
    if PLATFORM == "windows":
        cmake_args.extend([
            "-DCMAKE_CXX_FLAGS=/MP",
            f"-DCMAKE_BUILD_PARALLEL_LEVEL={build_threads}"
        ])
    else:
        cmake_args.extend([
            f"-DCMAKE_BUILD_PARALLEL_LEVEL={build_threads}",
            "-DCMAKE_CXX_FLAGS=-pthread"
        ])
    
    if cmake_args:
        env["CMAKE_ARGS"] = " ".join(cmake_args)
        if PLATFORM == "windows":
            env["FORCE_CMAKE"] = "1"
        print(f"\nCompilation flags:")
        for arg in cmake_args:
            if arg.startswith("-DGGML_"):
                print(f"  {arg}")
    
    # Configure git globally for unreliable connections
    subprocess.run(["git", "config", "--global", "http.lowSpeedLimit", "50"], 
                   capture_output=True)
    subprocess.run(["git", "config", "--global", "http.lowSpeedTime", "300"], 
                   capture_output=True)
    subprocess.run(["git", "config", "--global", "http.postBuffer", "524288000"], 
                   capture_output=True)
    subprocess.run(["git", "config", "--global", "progress.showProgress", "true"], 
                   capture_output=True)
    
    print("\n" + "=" * 80)
    print("BUILD OUTPUT (building from latest git master):")
    print("  This will:")
    print("    1. Clone llama-cpp-python repository with retry logic")
    print("    2. Clone matching llama.cpp submodule with retry logic")
    print("    3. Build with CPU optimizations for your hardware")
    print("=" * 80 + "\n")
    
    # Create random hash for unique build directory
    import hashlib
    import random
    random_hash = hashlib.md5(str(time.time() + random.random()).encode()).hexdigest()[:12]
    llama_cpp_python_dir = TEMP_DIR / f"build_{random_hash}"
    
    try:
        # Step 1: Clone main repo with retry
        if not llama_cpp_python_dir.exists():
            print_status("Cloning llama-cpp-python repository...")
            print(f"  Build directory: {llama_cpp_python_dir}")
            max_retries = 10
            retry_delay = 30
            
            for attempt in range(max_retries):
                try:
                    subprocess.run([
                        "git", "clone", "--progress",
                        "https://github.com/abetlen/llama-cpp-python.git",
                        str(llama_cpp_python_dir)
                    ], check=True, timeout=600, env=env)
                    print_status("Main repository cloned")
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    if llama_cpp_python_dir.exists():
                        shutil.rmtree(llama_cpp_python_dir)
                    
                    if attempt < max_retries - 1:
                        print(f"\n{'=' * 80}")
                        print(f"Clone attempt {attempt + 1}/{max_retries} failed")
                        print(f"Retrying in {retry_delay} seconds...")
                        print(f"{'=' * 80}")
                        
                        for remaining in range(retry_delay, 0, -1):
                            print(f"\rRetrying in {remaining} seconds...  ", end='', flush=True)
                            time.sleep(1)
                        print("\r" + " " * 40 + "\r", end='', flush=True)
                        retry_delay = min(retry_delay + 15, 60)
                    else:
                        print_status("Failed to clone main repository", False)
                        return False
        else:
            print_status("Using existing llama-cpp-python clone")
        
        # Step 2: Initialize submodules with aggressive retry
        print_status("Initializing submodules (llama.cpp)...")
        
        max_submodule_retries = 10
        retry_delay = 15
        
        for attempt in range(max_submodule_retries):
            try:
                # Use git submodule update with resume capability
                # Don't capture output so progress shows in real-time
                process = subprocess.Popen([
                    "git", "submodule", "update", "--init", "--recursive", "--progress"
                ], cwd=llama_cpp_python_dir, env=env)
                
                # Wait for process with timeout
                try:
                    process.wait(timeout=900)
                    
                    if process.returncode == 0:
                        print_status("Submodules initialized successfully")
                        break
                    else:
                        raise subprocess.CalledProcessError(process.returncode, "git submodule")
                        
                except subprocess.TimeoutExpired:
                    # Kill the process if it times out
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except:
                        pass
                    raise
                
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                # Make sure process is dead before retry
                try:
                    if 'process' in locals():
                        process.kill()
                        process.wait(timeout=5)
                except:
                    pass
                
                if attempt < max_submodule_retries - 1:
                    print(f"\n{'=' * 80}")
                    print(f"Submodule attempt {attempt + 1}/{max_submodule_retries} failed")
                    print(f"Retrying in {retry_delay} seconds (git will resume partial download)...")
                    print(f"{'=' * 80}\n")
                    
                    # Countdown on same line
                    for remaining in range(retry_delay, 0, -1):
                        print(f"\rRetrying in {remaining} seconds...  ", end='', flush=True)
                        time.sleep(1)
                    print("\r" + " " * 40 + "\r", end='', flush=True)
                    retry_delay = min(retry_delay + 15, 90)
                else:
                    print_status("Failed to clone submodules after all retries", False)
                    shutil.rmtree(llama_cpp_python_dir)
                    return False
        
        # Step 3: Install from local directory
        print_status("Building and installing llama-cpp-python...")
        print("  This will take 10-20 minutes depending on your CPU")
        
        cmd = [
            pip_exe, "install",
            str(llama_cpp_python_dir),
            "--no-cache-dir",
            "--force-reinstall",
            "--upgrade",
            "--verbose"
        ]
        
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        track_process(process.pid)
        
        for line in process.stdout:
            print(line, end='')
        
        process.wait()
        
        if process.returncode == 0:
            optimizations = []
            if cpu_features['AVX2']:
                optimizations.append("AVX2")
            if cpu_features['FMA']:
                optimizations.append("FMA (15-25% faster)")
            if cpu_features['F16C']:
                optimizations.append("F16C (50% less RAM, 10-30% faster)")
            if cpu_features['AVX512']:
                optimizations.append("AVX512")
            
            print("\n" + "=" * 80)
            print_status("llama-cpp-python built from source successfully")
            print(f"Optimizations enabled: {', '.join(optimizations) if optimizations else 'baseline'}")
            
            # Clean up on success
            try:
                shutil.rmtree(llama_cpp_python_dir)
                print_status("Cleaned up build directory")
            except:
                pass
            
            return True
        else:
            print_status("Build failed", False)
            return False
            
    except Exception as e:
        print_status(f"Build error: {e}", False)
        if llama_cpp_python_dir.exists():
            try:
                shutil.rmtree(llama_cpp_python_dir)
            except:
                pass
        return False

def build_config(backend: str) -> dict:
    """Build configuration for all 6 backend options"""
    info = BACKEND_OPTIONS[backend]

    # Determine backend_type based on selection
    if backend in ["Download CPU Binaries / Download CPU Wheel", "Compile CPU Binaries / Compile CPU Wheel"]:
        backend_type = "CPU_CPU"
        vulkan_available = False
    elif backend in ["Download Vulkan Bin / Download CPU Wheel", "Download Vulkan Bin / Download CPU Wheel (Forced)"]:
        backend_type = "VULKAN_CPU"
        vulkan_available = True
    elif backend in ["Download Vulkan Bin / Compile Vulkan Wheel", "Compile Vulkan Binaries / Compile Vulkan Wheel"]:
        backend_type = "VULKAN_VULKAN"
        vulkan_available = True
    else:
        backend_type = "CPU_CPU"
        vulkan_available = False
    
    vram_size = 8192 if vulkan_available else 0
    layer_allocation_mode = "SRAM_ONLY"

    config = {
        "model_settings": {
            "model_dir": "models",
            "model_name": "",
            "context_size": 32768,
            "temperature": 0.66,
            "repeat_penalty": 1.1,
            "use_python_bindings": True,
            "vram_size": vram_size,
            "selected_gpu": None,
            "selected_cpu": None,
            "mmap": True,
            "mlock": True,
            "n_batch": 1024,
            "dynamic_gpu_layers": True,
            "max_history_slots": 12,
            "max_attach_slots": 6,
            "print_raw_output": False,
            "show_think_phase": False,
            "bleep_on_events": False,
            "session_log_height": 500,
            "cpu_threads": 4,
            "vulkan_available": vulkan_available,
            "backend_type": backend_type,
            "layer_allocation_mode": layer_allocation_mode
        }
    }

    # Set CLI path for any option with binaries
    if info.get("cli_path"):
        if PLATFORM == "windows":
            if "CPU" in backend and "Compile CPU" in backend:
                config["model_settings"]["llama_cli_path"] = ".\\data\\llama-cpu-bin\\llama-cli.exe"
            else:
                config["model_settings"]["llama_cli_path"] = ".\\data\\llama-vulkan-bin\\llama-cli.exe"
        else:  # Linux
            if "CPU" in backend and "Compile CPU" in backend:
                config["model_settings"]["llama_cli_path"] = "./data/llama-cpu-bin/llama-cli"
            else:
                config["model_settings"]["llama_cli_path"] = "./data/llama-vulkan-bin/llama-cli"
        
        if info.get("dest"):
            config["model_settings"]["llama_bin_path"] = info["dest"]

    return config

def create_config(backend: str) -> None:
    """Create configuration file with unified format"""
    config_path = BASE_DIR / "data" / "persistent.json"
    config = build_config(backend)
    
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print_status("Configuration file created")
        
        print("\nGenerated configuration:")
        print(f"  Backend: {config['model_settings']['backend_type']}")
        print(f"  Vulkan Available: {config['model_settings']['vulkan_available']}")
        print(f"  VRAM: {config['model_settings']['vram_size']} MB")
        print(f"  Context: {config['model_settings']['context_size']}")
        if "llama_cli_path" in config["model_settings"]:
            print(f"  llama-cli: {config['model_settings']['llama_cli_path']}")
        else:
            print(f"  Mode: Python bindings only (CPU)")
        
    except Exception as e:
        print_status(f"Failed to create config: {str(e)}", False)

def create_venv() -> bool:
    try:
        if VENV_DIR.exists():
            shutil.rmtree(VENV_DIR)
            print_status("Removed existing virtual environment")

        subprocess.run([sys.executable, "-m", "venv", str(VENV_DIR)], check=True)
        print_status("Created new virtual environment")

        python_exe = VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / ("python.exe" if PLATFORM == "windows" else "python")
        if not python_exe.exists():
            raise FileNotFoundError(f"Python executable not found at {python_exe}")

        print_status("Verified virtual environment setup")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"Failed to create venv: {e}", False)
        return False


# Progress helpers
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

def detect_cpu_features() -> dict:
    """
    Detect CPU SIMD features for optimal compilation.
    Returns dict with feature flags for CMake.
    """
    features = {
        "AVX": False,
        "AVX2": False,
        "AVX512": False,
        "FMA": False,
        "F16C": False
    }
    
    if PLATFORM == "windows":
        try:
            import subprocess
            # Use WMIC to get CPU info
            result = subprocess.run(
                ["wmic", "cpu", "get", "caption"],
                capture_output=True,
                text=True,
                timeout=5
            )
            cpu_info = result.stdout.lower()
            
            # Try cpuinfo if available
            try:
                import cpuinfo
                info = cpuinfo.get_cpu_info()
                flags = info.get('flags', [])
                
                features["AVX"] = 'avx' in flags
                features["AVX2"] = 'avx2' in flags
                features["AVX512"] = any('avx512' in f for f in flags)
                features["FMA"] = 'fma' in flags
                features["F16C"] = 'f16c' in flags
                
            except ImportError:
                # Fallback: assume modern CPU has these if Intel/AMD
                if any(x in cpu_info for x in ['intel', 'amd']):
                    # Conservative assumption for Intel Haswell+ / AMD Piledriver+
                    features["AVX"] = True
                    features["AVX2"] = True
                    features["FMA"] = True
                    features["F16C"] = True
                    print_status("CPU detection limited - assuming AVX2+FMA+F16C support")
                
        except Exception as e:
            print(f"Warning: CPU detection failed: {e}")
            # Safe fallback
            features["AVX"] = True
            features["AVX2"] = True
            
    else:  # Linux
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read().lower()
            
            features["AVX"] = 'avx' in cpuinfo
            features["AVX2"] = 'avx2' in cpuinfo
            features["AVX512"] = 'avx512' in cpuinfo
            features["FMA"] = 'fma' in cpuinfo
            features["F16C"] = 'f16c' in cpuinfo
            
        except Exception as e:
            print(f"Warning: CPU detection failed: {e}")
            features["AVX"] = True
            features["AVX2"] = True
    
    return features

def check_vcredist_windows() -> bool:
    """Check if Visual C++ Redistributables are installed on Windows"""
    try:
        import winreg
        key_paths = [
            r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",  # VS 2015+
            r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        ]
        
        for key_path in key_paths:
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path):
                    return True
            except FileNotFoundError:
                continue
        return False
    except Exception as e:
        print(f"Warning: Could not check VC++ Redistributables: {e}")
        return False

def check_vulkan_sdk_installed() -> bool:
    """Check if Vulkan SDK (not just runtime) is installed"""
    if PLATFORM == "windows":
        # 1. Environment variable set and points to existing dir
        vulkan_sdk = os.environ.get("VULKAN_SDK")
        if vulkan_sdk and Path(vulkan_sdk).is_dir():
            return True

        # 2. Fallback: default Lunarg install path
        default_sdk = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "VulkanSDK"
        if default_sdk.exists():
            # any version sub-dir is enough
            for child in default_sdk.iterdir():
                if child.is_dir() and (child/"Bin"/"vulkaninfoSDK.exe").exists():
                    # add it to env for the current session so pip build sees it
                    os.environ["VULKAN_SDK"] = str(child)
                    return True
        return False
    else:  # Linux
        try:
            # Check for vulkaninfo command
            result = subprocess.run(["vulkaninfo", "--summary"], 
                                   stdout=subprocess.DEVNULL, 
                                   stderr=subprocess.DEVNULL)
            if result.returncode != 0:
                return False
            
            # Check for vulkan-sdk or development headers
            result = subprocess.run(["pkg-config", "--exists", "vulkan"],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
            return result.returncode == 0
        except FileNotFoundError:
            return False

def install_vulkan_sdk_windows() -> bool:
    """Prompt user to install Vulkan SDK on Windows"""
    print("\n" + "!" * 80)
    print("VULKAN SDK REQUIRED")
    print("!" * 80)
    print("\nBuilding llama-cpp-python with Vulkan requires the Vulkan SDK.")
    print("\nPlease install from: https://vulkan.lunarg.com/sdk/home")
    print("  - Download the Windows SDK installer")
    print("  - Run the installer")
    print("  - Only 'Vulkan SDK Core' is required (uncheck other components)")
    print("  - After installation, re-run this installer")
    print("!" * 80 + "\n")
    
    input("Press Enter after installing Vulkan SDK (or Ctrl+C to cancel)...")
    
    if check_vulkan_sdk_installed():
        print_status("Vulkan SDK detected")
        return True
    else:
        print_status("Vulkan SDK still not detected", False)
        return False

def install_vulkan_sdk_linux() -> bool:
    """Install Vulkan SDK on Linux with shader compiler support"""
    print_status("Installing Vulkan SDK for building llama-cpp-python...")
    packages = [
        "vulkan-tools",
        "libvulkan-dev", 
        "vulkan-headers",
        "vulkan-validationlayers-dev",
        "mesa-utils",
        "shaderc",  # Provides glslc shader compiler
        "glslang-tools"  # Alternative shader compiler tools
    ]
    try:
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y"] + packages, check=True)
        # Verify glslc is available
        result = subprocess.run(["which", "glslc"], capture_output=True, text=True)
        if result.returncode == 0:
            print_status("Vulkan SDK with shader compiler installed")
            return True
        else:
            print_status("Vulkan SDK installed but glslc not found", False)
            print("  Please ensure shaderc package was installed correctly")
            return False
    except subprocess.CalledProcessError as e:
        print_status(f"Vulkan SDK installation failed: {e}", False)
        return False

def install_vcredist_windows() -> bool:
    """Download and install Visual C++ Redistributable on Windows"""
    print_status("Visual C++ Redistributable not found")
    print_status("Downloading Visual C++ 2015-2022 Redistributable...")
    
    TEMP_DIR.mkdir(exist_ok=True)
    vcredist_url = "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    vcredist_path = TEMP_DIR / "vc_redist.x64.exe"
    
    try:
        # Download with progress bar (matching your theme)
        download_with_progress(vcredist_url, vcredist_path, "Downloading VC++ Redistributable")
        
        print_status("Installing Visual C++ Redistributable (silent mode)...")
        print("  This may take 1-2 minutes...")
        
        # Silent install: /install /quiet /norestart
        result = subprocess.run(
            [str(vcredist_path), "/install", "/quiet", "/norestart"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        # Clean up installer
        vcredist_path.unlink(missing_ok=True)
        
        # Check if installation succeeded
        if result.returncode == 0:
            print_status("Visual C++ Redistributable installed successfully")
            return True
        elif result.returncode == 1638:
            # Already installed (edge case)
            print_status("Visual C++ Redistributable already present")
            return True
        elif result.returncode == 3010:
            # Success but reboot required
            print_status("Visual C++ Redistributable installed (reboot recommended)")
            print("\nNote: A system reboot is recommended but not required.")
            return True
        else:
            print_status(f"Installation returned code {result.returncode}", False)
            print(f"Output: {result.stdout}")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("Visual C++ installation timed out", False)
        vcredist_path.unlink(missing_ok=True)
        return False
    except Exception as e:
        print_status(f"Failed to install VC++ Redistributable: {e}", False)
        vcredist_path.unlink(missing_ok=True)
        return False

def install_onnxruntime() -> bool:
    """Install onnxruntime with proper error handling. Do not verify by importing."""
    print_status("Installing onnxruntime (required for embeddings)...")
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                     ("python.exe" if PLATFORM == "windows" else "python"))
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") /
                  ("pip.exe" if PLATFORM == "windows" else "pip"))
    
    # On Windows, ensure VC++ Redistributables are installed FIRST
    if PLATFORM == "windows":
        if not check_vcredist_windows():
            print("\n" + "=" * 80)
            print("CRITICAL DEPENDENCY: Visual C++ Redistributable Required")
            print("=" * 80)
            if not install_vcredist_windows():
                print("\n" + "!" * 80)
                print("CRITICAL WARNING: Visual C++ Redistributable installation failed!")
                print("!" * 80)
                print("\nonnxruntime may fail at runtime, but installation continues.\n")
                # Do NOT abort—proceed with install, let app handle it later

    try:
        # Install onnxruntime
        subprocess.run(
            [pip_exe, "install", "onnxruntime"],
            check=True,
            capture_output=True,
            text=True,
            timeout=300
        )
        print_status("onnxruntime installed")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"onnxruntime installation failed", False)
        print(f"Error output: {e.stderr if e.stderr else 'No error output'}")
        return False
    except subprocess.TimeoutExpired:
        print_status("onnxruntime installation timed out", False)
        return False
    except Exception as e:
        print_status(f"onnxruntime installation error: {e}", False)
        return False

def download_fastembed_model() -> bool:
    """Download BAAI/bge-small-en-v1.5 model files directly via HTTP (no fastembed needed)."""
    model_name = "BAAI/bge-small-en-v1.5"
    cache_dir = BASE_DIR / "data" / "fastembed_cache"
    model_cache_path = cache_dir / f"models--BAAI--bge-small-en-v1.5"
    if model_cache_path.exists():
        print_status("Embedding model already cached")
        return True

    files = [
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "1_Pooling/config.json"
    ]
    base_url = "https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/main"
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_dir = cache_dir / "models--BAAI--bge-small-en-v1.5" / "snapshots" / "main"
    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        for file in files:
            url = f"{base_url}/{file}"
            dest = model_dir / file
            dest.parent.mkdir(parents=True, exist_ok=True)
            print_status(f"Downloading {file}...")
            download_with_progress(url, dest, f"  {file}")
        print_status("Embedding model downloaded and cached")
        return True
    except Exception as e:
        print_status(f"Model download failed: {e}", False)
        # Clean partial download
        if model_dir.exists():
            try:
                shutil.rmtree(model_dir)
            except:
                pass
        return False

def download_spacy_model() -> bool:
    """Download spaCy English model during installation."""
    
    try:
        pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                     ("pip.exe" if PLATFORM == "windows" else "pip"))
        
        print_status("Downloading spaCy language model...")
        
        # Keep original filename so pip recognizes the wheel tags
        filename = "en_core_web_sm-3.8.0-py3-none-any.whl"
        url = f"https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/{filename}"
        whl_path = TEMP_DIR / filename
        
        download_with_progress(url, whl_path, "Downloading spaCy model")
        
        print_status("Installing spaCy model...")
        result = subprocess.run(
            [pip_exe, "install", "--no-cache-dir", str(whl_path)], 
            capture_output=True, 
            text=True, 
            timeout=600
        )
        
        whl_path.unlink(missing_ok=True)
        
        if result.returncode == 0:
            print_status("spaCy model installed")
            return True
        else:
            print_status(f"spaCy install failed (code {result.returncode})", False)
            if result.stderr:
                print(f"Error output: {result.stderr[:200]}")
            if result.stdout:
                print(f"Install output: {result.stdout[:200]}")
            return False
            
    except subprocess.TimeoutExpired:
        print_status("spaCy install timed out", False)
        if 'whl_path' in locals():
            whl_path.unlink(missing_ok=True)
        return False
    except Exception as e:
        print_status(f"spaCy error: {e}", False)
        return False

def download_with_progress(url: str, filepath: Path, description: str = "Downloading") -> None:
    """Download file with progress bar, resume capability, and retries"""
    import time
    
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                    ("python.exe" if PLATFORM == "windows" else "python"))
    
    download_script = f'''
import requests
import time
from pathlib import Path

def format_bytes(b):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if b < 1024.0:
            return f"{{b:.1f}}{{unit}}"
        b /= 1024.0
    return f"{{b:.1f}}TB"

def simple_progress_bar(current, total, width=25):
    if total == 0:
        return "[" + "=" * width + "] 100%"
    filled_width = int(width * current // total)
    bar = "=" * filled_width + "-" * (width - filled_width)
    percent = 100 * current // total
    return f"[{{bar}}] {{percent}}% ({{format_bytes(current)}}/{{format_bytes(total)}})"

filepath = Path(r"{str(filepath)}")
max_retries = 5
retry_delay = 2

for attempt in range(max_retries):
    try:
        existing_size = filepath.stat().st_size if filepath.exists() else 0
        
        headers = {{}}
        if existing_size > 0:
            headers['Range'] = f'bytes={{existing_size}}-'
        
        response = requests.get("{url}", stream=True, headers=headers, timeout=30)
        
        if response.status_code == 416:
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
        chunk_size = 8192
        update_interval = max(total_size // 15, 8192) if total_size > 0 else 1024 * 1024
        last_update_at = downloaded
        
        mode = 'ab' if existing_size > 0 else 'wb'
        with open(filepath, mode) as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if (downloaded - last_update_at >= update_interval) or (downloaded >= total_size):
                        if total_size > 0:
                            progress = simple_progress_bar(downloaded, total_size)
                            print(f"\\r{description}: {{progress}}", end='', flush=True)
                        else:
                            downloaded_mb = downloaded / 1024 / 1024
                            print(f"\\r{description}: {{downloaded_mb:.1f}} MB downloaded", end='', flush=True)
                        last_update_at = downloaded
        
        print()
        break
        
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, 
            requests.exceptions.ChunkedEncodingError, ConnectionResetError) as e:
        if attempt < max_retries - 1:
            print(f"\\n{description}: Connection reset, retrying in {{retry_delay}}s... ({{attempt+1}}/{{max_retries}})")
            time.sleep(retry_delay)
            retry_delay *= 2
        else:
            filepath.unlink(missing_ok=True)
            raise Exception(f"Download failed after {{max_retries}} attempts: {{e}}")
    except Exception as e:
        filepath.unlink(missing_ok=True)
        raise e
'''
    
    download_script_path = TEMP_DIR / "download_file.py"
    try:
        with open(download_script_path, 'w') as f:
            f.write(download_script)
        
        result = subprocess.run(
            [python_exe, str(download_script_path)],
            check=True,
            timeout=1800
        )
        
    except subprocess.CalledProcessError as e:
        filepath.unlink(missing_ok=True)
        raise Exception(f"Download failed: {e}")
    except subprocess.TimeoutExpired:
        filepath.unlink(missing_ok=True)
        raise Exception("Download timed out")
    finally:
        download_script_path.unlink(missing_ok=True)

# Dependency checks
def is_vulkan_installed() -> bool:
    """Check if Vulkan is installed on the system"""
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
            print("\n" + "!" * 80)
            print(f"⚠️  WARNING: Vulkan not detected!")
            if PLATFORM == "windows":
                print("  Download from: https://vulkan.lunarg.com/sdk/home")
            else:
                print("  Install with: sudo apt install vulkan-tools libvulkan-dev")
            print("!" * 80 + "\n")
            return False
    if backend == "Force Vulkan GPU":
        return True  # skip all checks
    return True

def install_linux_system_dependencies(backend: str) -> bool:
    if PLATFORM != "linux":
        return True
    print_status("Installing Linux system dependencies...")
    base_packages = [
        "build-essential",
        "cmake",
        "python3-venv",
        "python3-dev",
        "portaudio19-dev",
        "libasound2-dev",
        "python3-tk",
        "espeak",
        "libespeak-dev",
        "ffmpeg",
        "xclip"
    ]
    info = BACKEND_OPTIONS[backend]
    vulkan_packages = []
    if info.get("build_flags", {}).get("GGML_VULKAN"):
        # Try minimal Vulkan packages first
        vulkan_packages = [
            "vulkan-tools",
            "libvulkan-dev",
            "mesa-utils",
            "glslang-tools",  # Provides glslc
            "spirv-tools"     # SPIR-V tools
        ]
    elif backend in ["Download Vulkan Bin / Download CPU Wheel", "Download Vulkan Bin / Download CPU Wheel (Forced)"]:
        vulkan_packages = ["vulkan-tools", "libvulkan1"]
    
    try:
        # Update package lists
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        
        # Install base packages
        subprocess.run(["sudo", "apt-get", "install", "-y"] + list(set(base_packages)), check=True)
        print_status("Base dependencies installed")
        
        # Install Vulkan packages if needed, one by one with fallbacks
        if vulkan_packages:
            print_status("Installing Vulkan development packages...")
            successful_vulkan_packages = []
            
            # Try modern package names first
            for package in vulkan_packages:
                try:
                    subprocess.run(["sudo", "apt-get", "install", "-y", package], check=True)
                    successful_vulkan_packages.append(package)
                    print_status(f"  ✓ {package}")
                except subprocess.CalledProcessError as e:
                    print_status(f"  ✗ {package} not available, trying alternatives...", False)
                    
                    # Fallback package mappings for Ubuntu 25.04
                    fallbacks = {
                        "glslang-tools": ["glslc"],
                        "spirv-tools": ["spirv-tools"],
                        "vulkan-tools": ["vulkan-utils"],
                        "libvulkan-dev": ["vulkan-headers", "libvulkan1-dev"],
                        "mesa-utils": ["mesa-utils"]
                    }
                    
                    if package in fallbacks:
                        for fallback in fallbacks[package]:
                            try:
                                subprocess.run(["sudo", "apt-get", "install", "-y", fallback], check=True)
                                successful_vulkan_packages.append(fallback)
                                print_status(f"    ✓ Fallback: {fallback}")
                                break
                            except subprocess.CalledProcessError:
                                continue
            
            # Verify glslc is available
            if info.get("build_flags", {}).get("GGML_VULKAN"):
                result = subprocess.run(["which", "glslc"], capture_output=True, text=True)
                if result.returncode == 0:
                    print_status("glslc shader compiler found")
                else:
                    print_status("glslc not found, trying manual installation steps...", False)
                    # Try direct installation of shader tools
                    try:
                        subprocess.run(["sudo", "apt-get", "install", "-y", "glslc"], check=True)
                        print_status("glslc installed directly")
                    except subprocess.CalledProcessError:
                        print_status("glslc still not found", False)
                        print("Please install Vulkan shader compiler manually:")
                        print("  sudo apt update")
                        print("  sudo apt install -y glslang-tools spirv-tools")
                        print("  sudo apt install -y shaderc")  # Alternative shader compiler
                        print("If still failing, try:")
                        print("  sudo apt install -y vulkan-sdk")
                        return False
        
        print_status("Linux dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print_status(f"System dependencies failed: {e}", False)
        return False

# Python dependencies
def install_python_deps(backend: str) -> bool:
    print_status("Installing Python dependencies...")
    try:
        python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                        ("python.exe" if PLATFORM == "windows" else "python"))
        pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                     ("pip.exe" if PLATFORM == "windows" else "pip"))

        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print_status("Upgraded pip")

        # Install base packages
        subprocess.run([pip_exe, "install", *BASE_REQ], check=True)
        print_status("Base dependencies installed")

        if not install_onnxruntime():
            return False

        try:
            subprocess.run([pip_exe, "install", "fastembed"], check=True, timeout=240)
            print_status("fastembed installed")
        except Exception as e:
            print_status(f"fastembed installation failed: {e}", False)
            return False

        # Handle llama-cpp-python installation
        info = BACKEND_OPTIONS[backend]
        
        # Option 1: Download pre-built CPU wheel (fastest)
        if not info.get("compile_wheel"):
            print_status("Installing pre-built llama-cpp-python (CPU)...")
            try:
                subprocess.run([
                    pip_exe, "install", "llama-cpp-python",
                    "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cpu"
                ], check=True, timeout=300)
                print_status("Pre-built wheel installed")
            except:
                print_status("Pre-built wheel failed, building from source...")
                if not build_llama_cpp_python_with_flags({}):
                    return False
        else:
            # Options 2, 5, 6: Compile wheel from source
            build_flags = info.get("build_flags", {})
            
            # Check Vulkan SDK for Vulkan builds
            if build_flags.get("GGML_VULKAN"):
                print_status("Vulkan wheel build requested - checking for Vulkan SDK...")
                
                if not check_vulkan_sdk_installed():
                    print_status("Vulkan SDK not found", False)
                    
                    if PLATFORM == "windows":
                        if not install_vulkan_sdk_windows():
                            print("\n" + "!" * 80)
                            print("Cannot build Vulkan wheel without Vulkan SDK")
                            print("!" * 80)
                            return False
                    else:  # Linux
                        if not install_vulkan_sdk_linux():
                            return False
                else:
                    print_status("Vulkan SDK already installed")
            
            # Build wheel
            if not build_llama_cpp_python_with_flags(build_flags):
                return False

        print_status("Python dependencies installed")
        return True
        
    except subprocess.CalledProcessError as e:
        print_status(f"Install failed: {e}", False)
        return False

def install_optional_file_support() -> bool:
    """Install optional file format libraries (PDF, DOCX, etc.)"""
    print_status("Installing optional file format support...")
    
    optional_packages = [
        "PyPDF2>=3.0.0",
        "python-docx>=0.8.11", 
        "openpyxl>=3.0.0",
        "python-pptx>=0.6.21"
    ]
    
    python_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                    ("python.exe" if PLATFORM == "windows" else "python"))
    pip_exe = str(VENV_DIR / ("Scripts" if PLATFORM == "windows" else "bin") / 
                 ("pip.exe" if PLATFORM == "windows" else "pip"))
    
    failed_packages = []
    for package in optional_packages:
        try:
            subprocess.run([pip_exe, "install", package], 
                          check=True, 
                          capture_output=True)
            print_status(f"Installed {package.split('>=')[0]}")
        except subprocess.CalledProcessError:
            failed_packages.append(package.split('>=')[0])
            print_status(f"Optional package {package.split('>=')[0]} failed", False)
    
    if failed_packages:
        print(f"\nNote: Some file formats may not be supported: {', '.join(failed_packages)}")
        print("The program will work with text files only for these formats.\n")
    else:
        print_status("All file format support installed")
    
    return True


# Backend download & extraction
def copy_linux_binaries(source_dir: Path, dest_dir: Path) -> None:
    build_bin_dir = source_dir / "build" / "bin"
    if not build_bin_dir.exists():
        raise FileNotFoundError(f"Build dir not found: {build_bin_dir}")

    copied = 0
    for file in build_bin_dir.iterdir():
        if file.is_file() and file.name.startswith("llama"):
            dest_file = dest_dir / file.name
            shutil.copy2(file, dest_file)
            os.chmod(dest_file, 0o755)
            copied += 1

    if copied:
        print_status(f"Copied {copied} binaries")
    else:
        raise FileNotFoundError("No llama binaries")

def download_binary_from_url(backend: str, info: dict) -> bool:
    """Download pre-built binaries (options 3, 4, 5)"""
    # Check if already exists
    if info["dest"] and info["cli_path"]:
        cli_path = BASE_DIR / info["cli_path"]
        if cli_path.exists():
            print_status(f"Backend already exists at {info['dest']}")
            return True
    
    print_status(f"Downloading llama.cpp binaries...")
    TEMP_DIR.mkdir(exist_ok=True)
    temp_zip = TEMP_DIR / "llama.zip"

    try:
        import zipfile
        url = info["url"]
        if url is None:
            print_status("No download URL specified", False)
            return False
            
        download_with_progress(url, temp_zip, f"Downloading binary")

        dest_path = BASE_DIR / info["dest"]
        dest_path.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(temp_zip, 'r') as zf:
            members = zf.namelist()
            total = len(members)
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
        temp_zip.unlink(missing_ok=True)

def check_build_tools() -> bool:
    """Check if required build tools are available for compilation"""
    
    # Check for git first (required for all platforms)
    if not shutil.which("git"):
        print_status("Git not found", False)
        print("\n" + "!" * 80)
        print("Git is required to clone repositories for compilation")
        if PLATFORM == "windows":
            print("Download from: https://git-scm.com/download/win")
        else:
            print("Install with: sudo apt install git")
        print("!" * 80 + "\n")
        return False
    
    if PLATFORM == "windows":
        # Check for CMake - multiple methods
        cmake_found = False
        cmake_path = None
        
        # Method 1: Check PATH
        if shutil.which("cmake"):
            cmake_found = True
            cmake_path = "PATH"
        
        # Method 2: Check common Visual Studio installations
        if not cmake_found:
            vs_base = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Microsoft Visual Studio"
            if vs_base.exists():
                # Check for VS 2022, 2019, 2017
                for year in ["2022", "2019", "2017"]:
                    for edition in ["Community", "Professional", "Enterprise"]:
                        cmake_candidate = vs_base / year / edition / "Common7" / "IDE" / "CommonExtensions" / "Microsoft" / "CMake" / "CMake" / "bin" / "cmake.exe"
                        if cmake_candidate.exists():
                            cmake_found = True
                            cmake_path = str(cmake_candidate)
                            # Add to PATH for this session
                            bin_dir = str(cmake_candidate.parent)
                            if bin_dir not in os.environ["PATH"]:
                                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
                            break
                    if cmake_found:
                        break
        
        # Method 3: Check standalone CMake installation
        if not cmake_found:
            cmake_program_files = Path(os.environ.get("ProgramFiles", "C:\\Program Files")) / "CMake" / "bin" / "cmake.exe"
            if cmake_program_files.exists():
                cmake_found = True
                cmake_path = str(cmake_program_files)
                bin_dir = str(cmake_program_files.parent)
                if bin_dir not in os.environ["PATH"]:
                    os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ['PATH']}"
        
        if not cmake_found:
            print_status("CMake not found", False)
            print("\n" + "!" * 80)
            print("CMake is required for building llama.cpp")
            print("Download from: https://cmake.org/download/")
            print("Or install via: winget install Kitware.CMake")
            print("!" * 80 + "\n")
            return False
        
        print_status(f"Found CMake ({cmake_path})")
        
        # Check for Visual Studio or MinGW
        has_mingw = shutil.which("gcc") is not None
        has_msvc = False
        
        if not has_mingw:
            # Check for Visual Studio via vswhere
            try:
                vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
                if vswhere.exists():
                    result = subprocess.run([str(vswhere), "-latest", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"], 
                                          capture_output=True, timeout=5)
                    has_msvc = result.returncode == 0
            except:
                pass
        
        if not (has_msvc or has_mingw):
            print_status("C++ compiler not found", False)
            print("\n" + "!" * 80)
            print("A C++ compiler is required for building llama.cpp")
            print("\nOptions:")
            print("  1. Visual Studio 2019+ with 'Desktop development with C++'")
            print("     Download: https://visualstudio.microsoft.com/downloads/")
            print("  2. MinGW-w64")
            print("     Download: https://winlibs.com/")
            print("!" * 80 + "\n")
            return False
        
        # Report which compiler was found
        if has_msvc:
            print_status("Found Visual Studio C++ compiler")
        elif has_mingw:
            print_status("Found MinGW GCC compiler")
        
    else:  # Linux
        required = ["cmake", "gcc", "g++", "make"]
        missing = [tool for tool in required if not shutil.which(tool)]
        
        if missing:
            print_status(f"Missing build tools: {', '.join(missing)}", False)
            print("\n" + "!" * 80)
            print("Required build tools are missing")
            print("Install with: sudo apt install build-essential cmake")
            print("!" * 80 + "\n")
            return False
    
    print_status("All build tools available")
    return True

def compile_llama_cpp_binary(backend: str, info: dict) -> bool:
    """
    Compile llama.cpp binaries from source with optimal CPU flags and aggressive multi-threading.
    Auto-detects AVX2, FMA, F16C for maximum performance.
    """
    import traceback
    
    global _DID_COMPILATION
    _DID_COMPILATION = True
    
    # Snapshot processes before build starts
    snapshot_pre_existing_processes()
    
    print_status(f"Compiling llama.cpp binaries from source (15-30 minutes)...")
    
    dest_path = BASE_DIR / info["dest"]
    dest_path.mkdir(parents=True, exist_ok=True)
    
    llamacpp_src = TEMP_DIR / "llama.cpp"
    
    # Get optimal build threads - 85% as requested
    import multiprocessing
    try:
        total_threads = multiprocessing.cpu_count()
    except:
        total_threads = 4
    
    build_threads = max(1, int(total_threads * 0.85))
    print(f"  Building with {build_threads} of {total_threads} threads (85%)")
    
    # Set up environment for parallel compilation - FORCE 85% usage
    env = os.environ.copy()
    env["CMAKE_BUILD_PARALLEL_LEVEL"] = str(build_threads)
    
    if PLATFORM == "windows":
        env["FORCE_CMAKE"] = "1"
        # Override any existing CL flags to ensure /MP is used
        existing_cl = env.get("CL", "")
        if f"/MP" not in existing_cl:
            env["CL"] = f"/MP{build_threads}"
        else:
            # Replace existing /MP setting
            import re
            env["CL"] = re.sub(r'/MP\d*', f"/MP{build_threads}", existing_cl)
    else:
        # Force override any existing MAKEFLAGS/NINJAFLAGS
        env["MAKEFLAGS"] = f"-j{build_threads}"
        env["NINJAFLAGS"] = f"-j{build_threads}"
        # Also set for make-based builds
        env["CMAKE_MAKE_PROGRAM"] = f"make -j{build_threads}"
    
    try:
        # Configure git for unreliable connections before cloning
        subprocess.run(["git", "config", "--global", "http.lowSpeedLimit", "200"], 
                       capture_output=True)
        subprocess.run(["git", "config", "--global", "http.lowSpeedTime", "240"], 
                       capture_output=True)
        subprocess.run(["git", "config", "--global", "http.postBuffer", "524288000"], 
                       capture_output=True)
        
        # Clone llama.cpp if not exists
        if not llamacpp_src.exists():
            print_status("Cloning llama.cpp repository...")
            
            max_retries = 5
            retry_delay = 15
            
            for attempt in range(max_retries):
                try:
                    process = subprocess.Popen([
                        "git", "clone", "--depth", "1", "--progress",
                        LLAMACPP_GIT_REPO,
                        str(llamacpp_src)
                    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                       text=True, bufsize=1, universal_newlines=True, env=env)
                    
                    last_progress_line = ""
                    for line in process.stdout:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Progress lines that should overwrite
                        if any(x in line for x in ["Receiving objects:", "Resolving deltas:", 
                                                     "Counting objects:", "Compressing objects:", 
                                                     "Updating files:"]):
                            # Clear previous line and print new one
                            if last_progress_line:
                                print(f"\r{' ' * len(last_progress_line)}\r", end='', flush=True)
                            print(f"\r  {line}", end='', flush=True)
                            last_progress_line = line
                        else:
                            # Non-progress lines - print normally
                            if last_progress_line:
                                print()  # Newline after progress
                                last_progress_line = ""
                            print(f"  {line}")
                    
                    process.wait(timeout=600)
                    
                    if last_progress_line:
                        print()  # Final newline after progress
                    
                    if process.returncode == 0:
                        break
                    else:
                        raise subprocess.CalledProcessError(process.returncode, "git clone")
                        
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    if last_progress_line:
                        print()
                    if llamacpp_src.exists():
                        try:
                            shutil.rmtree(llamacpp_src)
                        except:
                            pass
                    
                    if attempt < max_retries - 1:
                        print(f"\n{'=' * 80}")
                        print(f"Clone attempt {attempt + 1}/{max_retries} failed")
                        print(f"Retrying in {retry_delay} seconds...")
                        print(f"{'=' * 80}")
                        
                        for remaining in range(retry_delay, 0, -1):
                            print(f"\rRetrying in {remaining} seconds...  ", end='', flush=True)
                            time.sleep(1)
                        print("\r" + " " * 40 + "\r", end='', flush=True)
                        
                        retry_delay = min(retry_delay + 15, 60)
                    else:
                        print_status("Clone failed after all retries", False)
                        return False
        
        # Detect CPU features
        print_status("Detecting CPU features...")
        cpu_features = detect_cpu_features()
        
        print("CPU Features detected:")
        for feature, supported in cpu_features.items():
            status = "✓" if supported else "✗"
            print(f"  {status} {feature}")
        
        # Build with CMake
        build_dir = llamacpp_src / "build"
        build_dir.mkdir(exist_ok=True)
        
        # Prepare CMake args with CPU optimizations and parallel build flags
        cmake_args = [
            "-DLLAMA_BUILD_TESTS=OFF",
            "-DLLAMA_BUILD_EXAMPLES=ON",
            "-DBUILD_SHARED_LIBS=OFF",
            "-DLLAMA_CURL=OFF",
            f"-DCMAKE_BUILD_PARALLEL_LEVEL={build_threads}",
        ]
        
        # Detect compiler and set appropriate flags
        has_msvc = False
        has_mingw = False
        has_ninja = False
        
        if PLATFORM == "windows":
            # Check for MSVC (Visual Studio) using vswhere
            try:
                vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
                if vswhere.exists():
                    result = subprocess.run(
                        [str(vswhere), "-latest", "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"],
                        capture_output=True,
                        timeout=5
                    )
                    has_msvc = result.returncode == 0
            except:
                pass
            
            # Check for MinGW
            if not has_msvc:
                has_mingw = shutil.which("gcc") is not None
            
            # Check for Ninja
            has_ninja = shutil.which("ninja") is not None
            
            # Set compiler-specific flags
            if has_msvc:
                cmake_args.extend([
                    f"-DCMAKE_CXX_FLAGS=/MP{build_threads}",
                    f"-DCMAKE_C_FLAGS=/MP{build_threads}",
                ])
            elif has_mingw:
                # MinGW uses Unix-style parallel flags
                env["MAKEFLAGS"] = f"-j{build_threads}"
        else:  # Linux
            # Linux always uses Unix-style flags
            env["MAKEFLAGS"] = f"-j{build_threads}"
        
        # Add CPU optimization flags (CRITICAL for performance)
        cmake_args.append(f"-DGGML_AVX={'ON' if cpu_features['AVX'] else 'OFF'}")
        cmake_args.append(f"-DGGML_AVX2={'ON' if cpu_features['AVX2'] else 'OFF'}")
        cmake_args.append(f"-DGGML_AVX512={'ON' if cpu_features['AVX512'] else 'OFF'}")
        cmake_args.append(f"-DGGML_FMA={'ON' if cpu_features['FMA'] else 'OFF'}")
        cmake_args.append(f"-DGGML_F16C={'ON' if cpu_features['F16C'] else 'OFF'}")
        
        # OpenMP for multi-threading
        cmake_args.append("-DGGML_OPENMP=ON")
        
        # Add Vulkan support if requested
        if info.get("build_flags", {}).get("GGML_VULKAN"):
            if not check_vulkan_sdk_installed():
                print_status("Vulkan SDK required for Vulkan binary compilation", False)
                if PLATFORM == "windows":
                    if not install_vulkan_sdk_windows():
                        return False
                else:
                    if not install_vulkan_sdk_linux():
                        return False
            
            cmake_args.append("-DGGML_VULKAN=ON")
            print("  Compiling with Vulkan GPU support")
        
        print("\nCompilation flags:")
        for arg in cmake_args:
            if arg.startswith("-DGGML_") or arg.startswith("-DCMAKE_") or arg.startswith("-DLLAMA_"):
                print(f"  {arg}")
        
        # CMake configure
        print_status("Configuring CMake...")
        if PLATFORM == "windows":
            if has_msvc:
                # Use vswhere to get the actual Visual Studio version
                try:
                    vswhere = Path(os.environ.get("ProgramFiles(x86)", "C:\\Program Files (x86)")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
                    result = subprocess.run(
                        [str(vswhere), "-latest", "-property", "installationVersion"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0:
                        version_str = result.stdout.strip()
                        major_version = int(version_str.split('.')[0])
                        
                        # Map major version to generator name
                        if major_version >= 17:
                            generator = "Visual Studio 17 2022"
                            print("  Using Visual Studio 2022 generator")
                        elif major_version == 16:
                            generator = "Visual Studio 16 2019"
                            print("  Using Visual Studio 2019 generator")
                        elif major_version == 15:
                            generator = "Visual Studio 15 2017"
                            print("  Using Visual Studio 2017 generator")
                        else:
                            print_status(f"Unsupported Visual Studio version: {major_version}", False)
                            return False
                        
                        cmake_cmd = ["cmake", "..", *cmake_args, "-G", generator, "-A", "x64"]
                    else:
                        print_status("Could not determine Visual Studio version", False)
                        return False
                        
                except Exception as e:
                    print_status(f"Failed to detect VS version: {e}", False)
                    return False
                    
            elif has_ninja:
                print("  Using Ninja generator")
                cmake_cmd = ["cmake", "..", *cmake_args, "-G", "Ninja"]
            elif has_mingw:
                print("  Using MinGW Makefiles generator")
                cmake_cmd = ["cmake", "..", *cmake_args, "-G", "MinGW Makefiles"]
            else:
                print_status("No suitable compiler found (need MSVC, Ninja, or MinGW)", False)
                return False
            
            subprocess.run(cmake_cmd, cwd=build_dir, check=True, timeout=300, env=env)
        else:  # Linux
            subprocess.run([
                "cmake", "..",
                *cmake_args
            ], cwd=build_dir, check=True, timeout=300, env=env)
        
        # CMake build with explicit parallel flags
        print_status(f"Building binaries with {build_threads} threads...")
        
        build_cmd = [
            "cmake", "--build", ".",
            "--config", "Release",
            "--parallel", str(build_threads)
        ]
        
        process = subprocess.Popen(build_cmd, cwd=build_dir, env=env)
        track_process(process.pid)
        returncode = process.wait(timeout=2400)
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, build_cmd)
        
        # Copy binaries to destination
        print_status("Installing binaries...")
        if PLATFORM == "windows":
            bin_dir = build_dir / "bin" / "Release"
            for exe in bin_dir.glob("*.exe"):
                if exe.name.startswith("llama"):
                    shutil.copy2(exe, dest_path / exe.name)
                    print(f"  Installed {exe.name}")
        else:  # Linux
            bin_dir = build_dir / "bin"
            for exe in bin_dir.iterdir():
                if exe.is_file() and exe.name.startswith("llama"):
                    dest_file = dest_path / exe.name
                    shutil.copy2(exe, dest_file)
                    os.chmod(dest_file, 0o755)
                    print(f"  Installed {exe.name}")
        
        # Verify CLI exists
        cli_path = BASE_DIR / info["cli_path"]
        if not cli_path.exists():
            raise FileNotFoundError(f"llama-cli not found after build: {cli_path}")
        
        # Print optimization summary
        optimizations = []
        if cpu_features['AVX2']:
            optimizations.append("AVX2")
        if cpu_features['FMA']:
            optimizations.append("FMA")
        if cpu_features['F16C']:
            optimizations.append("F16C (50% less RAM)")
        if cpu_features['AVX512']:
            optimizations.append("AVX512")
        
        print_status("Binary compilation complete")
        print(f"Optimizations enabled: {', '.join(optimizations) if optimizations else 'baseline'}")
        
        # Clean up source directory after successful build
        if llamacpp_src.exists():
            try:
                shutil.rmtree(llamacpp_src)
                print_status("Cleaned up build directory")
            except Exception as e:
                print(f"Warning: Could not remove build directory: {e}")
        
        return True
        
    except subprocess.TimeoutExpired:
        print_status("Compilation timed out", False)
        if llamacpp_src.exists():
            try:
                shutil.rmtree(llamacpp_src)
            except:
                pass
        return False
    except Exception as e:
        print_status(f"Compilation failed: {e}", False)
        print(traceback.format_exc())
        if llamacpp_src.exists():
            try:
                shutil.rmtree(llamacpp_src)
            except:
                pass
        return False

def download_extract_backend(backend: str) -> bool:
    """Handle backend download OR compilation"""
    info = BACKEND_OPTIONS[backend]
    
    # Option 1: No binaries needed
    if backend == "Download CPU Binaries / Download CPU Wheel":
        print_status("CPU-Only mode: No binary download needed")
        return True
    
    # Options 2 & 6: Compile binaries
    if info.get("compile_binary"):
        return compile_llama_cpp_binary(backend, info)
    
    # Options 3, 4, 5: Download binaries
    if info.get("url"):
        return download_binary_from_url(backend, info)
    
    print_status("No backend action required", False)
    return False
        
# Backend selection
def select_backend_type() -> str:
    """Show 5-option menu"""
    opts = list(BACKEND_OPTIONS.keys())
    choice = get_user_choice("Select backend:", opts)
    return choice


# Main install flow
def install():
    backend = select_backend_type()
    print_header("Installation")
    print(f"Installing {APP_NAME} on {PLATFORM} using {backend}")
    if sys.version_info < (3, 8):
        print_status("Python ≥3.8 required", False)
        sys.exit(1)
    # Clean compile temp (Windows only)
    if PLATFORM == "windows":
        clean_compile_temp()
    # Create directories first (needed for temp files)
    create_directories()
    # Install system dependencies BEFORE checking build tools
    if PLATFORM == "linux":
        if not install_linux_system_dependencies(backend):
            print_status("System dependencies installation failed", False)
            sys.exit(1)
    info = BACKEND_OPTIONS[backend]
    # Now check build tools (dependencies should be installed)
    if info.get("compile_binary") or info.get("compile_wheel"):
        if not check_build_tools():
            print_status("Missing required build tools", False)
            sys.exit(1)
    if not verify_backend_dependencies(backend):
        print_status("Missing system dependencies", False)
        sys.exit(1)
    if not create_venv():
        print_status("Virtual environment failed", False)
        sys.exit(1)

    # All functions below already handle venv paths internally
    # DO NOT wrap in activate_venv() context manager
    if not install_python_deps(backend):
        print_status("Python dependencies failed", False)
        sys.exit(1)

    install_optional_file_support()

    # Try to download models (now critical - must succeed)
    embedding_ok = download_fastembed_model()
    if not embedding_ok:
        print("\n" + "!" * 80)
        print("CRITICAL ERROR: Embedding model download failed!")
        print("!" * 80)
        print("\nRAG features require this model to function.")
        print("Installation cannot continue.\n")
        sys.exit(1)

    spacy_ok = download_spacy_model()
    if not spacy_ok:
        print_status("WARNING: spaCy model download failed", False)
        print("Session labeling may not work properly")
        # Non-critical, can continue

    if not download_extract_backend(backend):
        print_status("Backend download failed", False)
        sys.exit(1)

    create_config(backend)
    
    print_status("Installation complete!")
    print("\nRun the launcher to start Chat-Gradio-Gguf\n")


# ------------------------------------------------------------------
#  Protected main block
# ------------------------------------------------------------------
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
        # Ensure cleanup runs AFTER all file operations (including rmtree)
        time.sleep(2)  # Let file handles release
        cleanup_build_processes()
        if PLATFORM == "windows":
            clean_compile_temp()
        sys.exit(0)