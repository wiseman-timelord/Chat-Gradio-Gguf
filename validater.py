# Script: validator.py

# Imports
import os, sys, subprocess, glob, json, tempfile, wave, math, struct
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

# constants (copied from temporary.py)
AUDIO_SYSTEM_OPTIONS = ["windows", "pipewire", "pulseaudio", "alsa"]
AUDIO_SAMPLE_RATE_OPTIONS = [8000, 11025, 16000, 22050, 32000, 44100, 48000]
AUDIO_VOLUME_OPTIONS = [round(i * 0.1, 1) for i in range(11)]
AUDIO_RATE_OPTIONS   = list(range(50, 301, 25)) 

# Requirements list
REQS = [
    "gradio>=4.25.0",
    "requests==2.31.0",
    "pyperclip",
    "yake",
    "psutil",
    "ddgs",
    "newspaper3k",
    "llama-cpp-python",      # CPU wheel or [cuda]/[rocm] extras are NOT validated here
    "langchain-community==0.3.18",
    "faiss-cpu>=1.8.0",      # CPU-only vector store
    "langchain>=0.3.18",
    "pygments==2.17.2",
    "lxml[html_clean]",
    "pyttsx3",
    "onnxruntime",           # For fastembed runtime
    "fastembed",             # Quantised embeddings
    "tokenizers",            # Hugging Face tokenizers
]

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
        if not backend:
            print_status("Empty backend type", False)
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
        "duckduckgo-search": "duckduckgo_search",
        "newspaper3k": "newspaper",
        "llama-cpp-python": "llama_cpp",
        "langchain-community": "langchain_community",
        "faiss-cpu": "faiss",
        "langchain": "langchain",
        "pygments": "pygments",
        "lxml[html_clean]": "lxml",
        "pyttsx3": "pyttsx3",
        "onnxruntime": "onnxruntime",
        "fastembed": "fastembed",
        "tokenizers": "tokenizers",
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

def print_audio_setup_info(audio_system: str = None, verification_success: bool = True):
    """
    Print detailed audio setup information.
    """
    if PLATFORM == "windows":
        return
        
    print("\n" + "="*70)
    print("LINUX AUDIO SYSTEM SETUP INFORMATION")
    print("="*70)
    
    if audio_system == "pipewire":
        print("✓ PipeWire Audio Stack Configuration:")
        print("  • PipeWire: Modern low-latency audio server")
        print("  • PipeWire-Pulse: PulseAudio compatibility layer") 
        print("  • PipeWire-ALSA: ALSA compatibility layer")
        print("  • WirePlumber: Session & policy manager")
        print("\n  Troubleshooting commands:")
        print("    systemctl --user status pipewire")
        print("    systemctl --user restart pipewire wireplumber")
        print("    pw-top  # Show PipeWire graph")
        
    elif audio_system == "pulseaudio":
        print("✓ PulseAudio Configuration:")
        print("  • Traditional PulseAudio setup")
        print("\n  Troubleshooting commands:")
        print("    pulseaudio --check -v")
        print("    systemctl --user restart pulseaudio")
        
    else:
        print("✓ ALSA Configuration:")
        print("  • Basic ALSA audio system")
        print("\n  Troubleshooting commands:")
        print("    aplay -l  # List audio devices")
        print("    alsamixer  # Audio mixer")
    
    print(f"\n✓ TTS Engines Installed:")
    print("  • espeak-ng: Fast, lightweight synthesis")
    print("  • festival: Higher quality voices")  
    print("  • speech-dispatcher: System TTS coordination")
    print("  • pyttsx3: Python TTS library")
    
    if not verification_success:
        print(f"\n⚠ AUDIO SETUP NOTICES:")
        print("  Some audio components may need manual configuration.")
        print("  Try logging out and back in to refresh audio services.")
        print("  Run the validator to check specific issues.")
    
    print("="*70 + "\n")

# Add to test_directories() section
def test_audio_system() -> bool:
    """Comprehensive TTS and audio system test (Linux only)."""
    if PLATFORM != "linux":
        return True

    print("\n=== Audio System Validation ===")
    success = True

    # 1. Read configured audio system
    try:
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        audio_system = config.get("audio_config", {}).get("audio_system", "alsa")
        print_status(f"Configured audio system: {audio_system}")
    except Exception as e:
        print_status(f"Could not read audio config: {e}", False)
        audio_system = "alsa"
        success = False

    # 2. Test audio system services
    if audio_system == "pipewire":
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "pipewire"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print_status("PipeWire service active")
        else:
            print_status("PipeWire service not active", False)
            success = False
    elif audio_system == "pulseaudio":
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "pulseaudio"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            print_status("PulseAudio service active")
        else:
            print_status("PulseAudio service not active", False)
            success = False
            
    # 3. Test basic audio device detection
    try:
        result = subprocess.run(
            ["aplay", "-l"], 
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and "card" in result.stdout.lower():
            print_status("Audio devices detected via ALSA")
        else:
            print_status("No audio devices found via ALSA", False)
            success = False
    except Exception as e:
        print_status(f"Audio device detection failed: {e}", False)
        success = False

    # 4. Test TTS engines
    test_phrase = "Audio validation test"
    tts_success = False
    env = os.environ.copy()
    env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")

    tts_tests = [
        (["espeak-ng", "-v", "en", "-s", "150", test_phrase], "eSpeak NG"),
        (["spd-say", "--wait", "-t", "female1", "-r", "0", test_phrase], "Speech Dispatcher"),
        (["espeak", "-v", "en-us", "-s", "150", test_phrase], "eSpeak (legacy)"),
    ]

    for cmd, engine_name in tts_tests:
        try:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print_status(f"{engine_name} - SUCCESS")
                tts_success = True
                break
            else:
                print_status(f"{engine_name} - failed", False)
        except Exception:
            print_status(f"{engine_name} - not available", False)

    if not tts_success:
        print_status("No TTS engines are working", False)
        success = False

    # 5. Test Python TTS library (pyttsx3)
    venv_py = VENV_DIR / ("bin" if PLATFORM == "linux" else "Scripts") / ("python" if PLATFORM == "linux" else "python.exe")
    code = """
import pyttsx3, sys
try:
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.stop()
    sys.exit(0)
except Exception as e:
    sys.exit(1)
"""
    try:
        result = subprocess.run([str(venv_py), "-c", code], capture_output=True, text=True, timeout=15, env=env)
        if result.returncode == 0:
            print_status("pyttsx3 initialized successfully")
        else:
            print_status("pyttsx3 initialization failed", False)
            success = False
    except Exception:
        print_status("pyttsx3 test error", False)
        success = False

    return success

def test_audio_devices_detailed() -> bool:
    """
    Additional detailed audio device testing.
    This function provides more granular audio device verification.
    """
    if PLATFORM != "linux":
        return True
        
    print("\n=== Detailed Audio Device Analysis ===")
    
    try:
        # 1. Check user groups for audio permissions
        try:
            import pwd
            import grp
            username = pwd.getpwuid(os.getuid()).pw_name
            user_groups = [g.gr_name for g in grp.getgrall() if username in g.gr_mem]
            
            audio_groups = [g for g in user_groups if g in ['audio', 'pulse', 'pulse-access']]
            if audio_groups:
                print_status(f"User in audio groups: {', '.join(audio_groups)}")
            else:
                print_status("User not in audio groups (may be normal for modern systems)")
        except Exception as e:
            print_status(f"Could not check user groups: {e}")
    
        # 2. Check for common audio issues
        runtime_dir = os.environ.get('XDG_RUNTIME_DIR')
        if runtime_dir:
            pulse_dir = Path(runtime_dir) / "pulse"
            pipewire_dir = Path(runtime_dir) / "pipewire-0"
            
            if pulse_dir.exists():
                print_status(f"PulseAudio runtime directory exists: {pulse_dir}")
            if pipewire_dir.exists():
                print_status(f"PipeWire runtime directory exists: {pipewire_dir}")
        
        # 3. Test basic audio output capability
        try:
            # Try to generate a brief tone to test audio stack
            result = subprocess.run(
                ["timeout", "2", "speaker-test", "-t", "sine", "-f", "440", "-l", "1"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode in [0, 124]:  # 0=success, 124=timeout (expected)
                print_status("Audio output test successful")
                return True
            else:
                print_status(f"Audio output test failed: {result.stderr}", False)
        except Exception as e:
            print_status(f"Audio output test error: {e}", False)
    
        return False
        
    except Exception as e:
        print_status(f"Detailed audio analysis failed: {e}", False)
        return False

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

    if PLATFORM == "linux" and not test_audio_system():
        overall_success = False
        
    # NEW: actually test the audio stack on Linux
    if PLATFORM == "linux" and not test_audio_system():
        overall_success = False
    
    # Final result
    print("\n=== Validation Summary ===")
    if overall_success:
        print_status("All validations passed successfully!")
        return 0
    else:
        print_status("Validation failed with errors", False)
        return 1

if __name__ == "__main__":
    sys.exit(main())