# scripts/sounds.py
"""
Centralized audio module for TTS and system sounds.
- Coqui TTS (neural): Windows 8.1-11 + Ubuntu 22-25 with Python 3.10+
- pyttsx3 + SAPI5: Windows 7 or Windows with Python 3.9
- pyttsx3 + espeak: Linux with Python 3.9
"""

import os
import subprocess
import threading
import queue
from pathlib import Path
from scripts.temporary import SAMPLE_RATES, COQUI_DEFAULT_VOICES

_pyttsx3_engine = None
_tts_initialized = False
_current_engine = None


def _get_temporary():
    import scripts.temporary as temporary
    return temporary


# =============================================================================
# TTS ENGINE DETECTION
# =============================================================================

def get_available_tts_engine() -> str:
    tmp = _get_temporary()
    tts_engine = getattr(tmp, 'TTS_ENGINE', None)
    if tts_engine:
        return tts_engine
    try:
        from TTS.api import TTS
        return "coqui"
    except ImportError:
        pass
    try:
        import pyttsx3
        return "pyttsx3"
    except ImportError:
        pass
    return "none"


def initialize_tts() -> bool:
    global _tts_initialized, _current_engine
    if _tts_initialized:
        return _current_engine is not None
    _current_engine = get_available_tts_engine()
    _tts_initialized = True
    if _current_engine == "coqui":
        print("[TTS] Coqui TTS engine available (neural quality)")
    elif _current_engine == "pyttsx3":
        print("[TTS] pyttsx3 engine available (system voices)")
    else:
        print("[TTS] No TTS engine available")
    return _current_engine is not None


def get_tts_engine_name() -> str:
    if not _tts_initialized:
        initialize_tts()
    return {"coqui": "Coqui TTS (Neural)", "pyttsx3": "pyttsx3 (System)", "none": "Not Available"}.get(_current_engine, "Unknown")


# =============================================================================
# SOUND DEVICE DETECTION
# =============================================================================

def get_available_sound_devices() -> list:
    """
    Get available sound output devices.
    On Windows: Filter to WASAPI devices only (matches system tray sound settings)
    On Linux: Filter to PulseAudio/ALSA primary outputs
    """
    devices = [{"id": -1, "name": "Default", "is_default": True}]
    
    try:
        import sounddevice as sd
        
        tmp = _get_temporary()
        platform = getattr(tmp, 'PLATFORM', 'windows')
        
        # Get host API info to filter by API type
        host_apis = sd.query_hostapis()
        
        # Find the preferred host API index
        preferred_api_idx = None
        if platform == "windows":
            # Prefer WASAPI (Windows Audio Session API) - matches system sound settings
            for idx, api in enumerate(host_apis):
                if "WASAPI" in api.get('name', ''):
                    preferred_api_idx = idx
                    break
        else:
            # Linux: prefer PulseAudio, then ALSA
            for idx, api in enumerate(host_apis):
                api_name = api.get('name', '')
                if "pulse" in api_name.lower():
                    preferred_api_idx = idx
                    break
                elif "alsa" in api_name.lower() and preferred_api_idx is None:
                    preferred_api_idx = idx
        
        seen_names = set()
        all_devices = sd.query_devices()
        
        for i, dev in enumerate(all_devices):
            # Skip input-only devices
            if dev.get('max_output_channels', 0) <= 0:
                continue
            
            # Filter by preferred host API if found
            if preferred_api_idx is not None:
                if dev.get('hostapi') != preferred_api_idx:
                    continue
            
            # Clean up device name - remove API suffix if present
            name = dev['name']
            # Remove common suffixes like "(WASAPI)", "(DirectSound)", etc.
            for suffix in ['(WASAPI)', '(DirectSound)', '(MME)', '(Windows WASAPI)', '(Windows DirectSound)']:
                name = name.replace(suffix, '').strip()
            
            # Truncate long names
            name = name[:45]
            
            # Skip duplicates (same physical device via different APIs)
            if name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            
            # Skip virtual/loopback devices that aren't real outputs
            skip_keywords = ['loopback', 'virtual', 'stereo mix', 'what u hear', 'wave out']
            if any(kw in name.lower() for kw in skip_keywords):
                continue
            
            devices.append({"id": i, "name": name, "is_default": False})
        
    except Exception as e:
        print(f"[SOUND] Device detection error: {e}")
    
    return devices



def get_sound_device_names() -> list:
    return [d["name"] for d in get_available_sound_devices()]


def get_sample_rate_options() -> list:
    return SAMPLE_RATES

# =============================================================================
# VOICE OPTIONS
# =============================================================================

def get_installed_coqui_voices() -> list:
    """Scan for installed Coqui TTS voices and return list of available voices."""
    voices = []
    
    # Check TTS models cache directory (our custom location)
    cache_dir = Path(__file__).parent.parent / "data" / "tts_models"
    
    # Also check default Coqui cache locations
    home = Path.home()
    coqui_paths = [
        cache_dir,
        home / ".local" / "share" / "tts",
        home / "AppData" / "Local" / "tts",
    ]
    
    # Map model folder names to friendly names
    # Coqui uses folder names like: tts_models--en--ljspeech--tacotron2-DDC
    model_names = {
        "tacotron2-DDC": "English Male",
        "glow-tts": "English Female",
        "speedy-speech": "American Male",
        "vits": "American Female",
        "jenny": "English Female (Jenny)",
    }
    
    found_models = set()
    
    for base_path in coqui_paths:
        if not base_path.exists():
            continue
        
        # Look for Coqui's folder naming convention: tts_models--en--ljspeech--<model_name>
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("tts_models--"):
                # Extract model name from folder like "tts_models--en--ljspeech--tacotron2-DDC"
                parts = item.name.split("--")
                if len(parts) >= 4:
                    model_key = parts[-1]  # e.g., "tacotron2-DDC"
                    if model_key in model_names and model_key not in found_models:
                        found_models.add(model_key)
                        voices.append({
                            "id": model_key,
                            "name": model_names[model_key],
                            "model_path": f"tts_models/en/ljspeech/{model_key}"
                        })
        
        # Also check nested structure for .pth files
        for model_file in base_path.rglob("*.pth"):
            # Check parent folder name
            parent = model_file.parent.name
            if parent in model_names and parent not in found_models:
                found_models.add(parent)
                voices.append({
                    "id": parent,
                    "name": model_names[parent],
                    "model_path": f"tts_models/en/ljspeech/{parent}"
                })
            # Also check grandparent for nested structures
            grandparent = model_file.parent.parent.name
            for key in model_names:
                if key in grandparent and key not in found_models:
                    found_models.add(key)
                    voices.append({
                        "id": key,
                        "name": model_names[key],
                        "model_path": f"tts_models/en/ljspeech/{key}"
                    })
    
    # If no voices found, return all 4 defaults (will download on first use)
    if not voices:
        voices = [
            {"id": "tacotron2-DDC", "name": "English Male", "model_path": COQUI_DEFAULT_VOICES["english_male"]},
            {"id": "glow-tts", "name": "English Female", "model_path": COQUI_DEFAULT_VOICES["english_female"]},
            {"id": "speedy-speech", "name": "American Male", "model_path": COQUI_DEFAULT_VOICES["american_male"]},
            {"id": "vits", "name": "American Female", "model_path": COQUI_DEFAULT_VOICES["american_female"]},
        ]
    
    # Sort: American first, then English, Female before Male
    def voice_sort_key(v):
        name = v["name"]
        # Priority: American Female=0, American Male=1, English Female=2, English Male=3
        if "American" in name:
            return (0, "Male" in name, name)
        else:
            return (1, "Male" in name, name)
    
    voices.sort(key=voice_sort_key)
    
    return voices

def get_pyttsx3_voices() -> list:
    voices = []
    try:
        import pyttsx3
        engine = pyttsx3.init()
        for voice in engine.getProperty('voices'):
            voices.append({"id": voice.id, "name": voice.name})
        engine.stop()
    except:
        pass
    if not voices:
        voices = [{"id": "default", "name": "Default"}]
    return voices


def get_voice_options_for_engine() -> list:
    """Get voice display names based on current engine."""
    if not _tts_initialized:
        initialize_tts()
    if _current_engine == "coqui":
        return [v["name"] for v in get_installed_coqui_voices()]
    elif _current_engine == "pyttsx3":
        return [v["name"] for v in get_pyttsx3_voices()]
    return ["Not Available"]


def get_voice_model_path(voice_name: str) -> str:
    """Get model path for a voice name (Coqui only)."""
    if _current_engine == "coqui":
        for v in get_installed_coqui_voices():
            if v["name"] == voice_name:
                return v["model_path"]
        return COQUI_DEFAULT_VOICES["english_female"]
    return None


def get_voice_id(voice_name: str) -> str:
    """Get voice ID for a voice name (pyttsx3 only)."""
    if _current_engine == "pyttsx3":
        for v in get_pyttsx3_voices():
            if v["name"] == voice_name:
                return v["id"]
    return None


# =============================================================================
# TTS CONFIG VISIBILITY
# =============================================================================

def get_tts_config_visibility() -> dict:
    """Return which TTS config elements should be visible."""
    if not _tts_initialized:
        initialize_tts()
    if _current_engine == "coqui":
        return {"sound_device": True, "sample_rate": True, "voice_select": True}
    elif _current_engine == "pyttsx3":
        return {"sound_device": False, "sample_rate": False, "voice_select": True}
    return {"sound_device": False, "sample_rate": False, "voice_select": False}


# =============================================================================
# SPEAK TEXT
# =============================================================================

def speak_text(text: str) -> None:
    if not text or not text.strip():
        return
    if not _tts_initialized:
        initialize_tts()
    if _current_engine == "coqui":
        _speak_coqui(text)
    elif _current_engine == "pyttsx3":
        tmp = _get_temporary()
        if tmp.PLATFORM == "windows":
            _speak_pyttsx3_windows(text)
        else:
            _speak_pyttsx3_linux(text)


def _speak_coqui(text: str) -> None:
    tmp = _get_temporary()
    try:
        import sounddevice as sd
        from TTS.api import TTS
        
        # Get selected voice model path
        voice_name = getattr(tmp, 'TTS_VOICE', None)
        model_name = get_voice_model_path(voice_name) if voice_name else COQUI_DEFAULT_VOICES["english_female"]
        device_id = getattr(tmp, 'TTS_SOUND_DEVICE', -1)
        
        cache_dir = Path(__file__).parent.parent / "data" / "tts_models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["COQUI_TTS_CACHE"] = str(cache_dir)
        
        tts = TTS(model_name=model_name, progress_bar=False)
        tts.to("cpu")
        wav = tts.tts(text=text)
        
        if device_id >= 0:
            sd.play(wav, samplerate=tts.synthesizer.output_sample_rate, device=device_id)
        else:
            sd.play(wav, samplerate=tts.synthesizer.output_sample_rate)
        sd.wait()
        print(f"[TTS-COQUI] Spoke: {text[:50]}...")
    except Exception as e:
        print(f"[TTS-COQUI] Error: {e}")
        _speak_pyttsx3_fallback(text)


def _speak_pyttsx3_windows(text: str) -> None:
    result_queue = queue.Queue()
    tmp = _get_temporary()
    
    def _speak_isolated():
        import pythoncom
        import win32com.client
        try:
            pythoncom.CoInitialize()
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            voice_name = getattr(tmp, 'TTS_VOICE', None)
            if voice_name:
                for voice in speaker.GetVoices():
                    if voice_name in voice.GetDescription():
                        speaker.Voice = voice
                        break
            speaker.Speak(text)
            result_queue.put(("success", None))
        except Exception as e:
            result_queue.put(("error", str(e)))
        finally:
            try:
                pythoncom.CoUninitialize()
            except:
                pass
    
    thread = threading.Thread(target=_speak_isolated, daemon=True)
    thread.start()
    thread.join(timeout=30)
    
    if thread.is_alive():
        print("[TTS-WIN] Speech timed out")
        return
    try:
        status, error = result_queue.get_nowait()
        if status == "error":
            print(f"[TTS-WIN] Error: {error}")
        else:
            print(f"[TTS-WIN] Spoke: {text[:50]}...")
    except queue.Empty:
        pass


def _speak_pyttsx3_linux(text: str) -> None:
    global _pyttsx3_engine
    tmp = _get_temporary()
    try:
        import pyttsx3
        if _pyttsx3_engine is None:
            try:
                _pyttsx3_engine = pyttsx3.init(driverName='espeak')
            except:
                _pyttsx3_engine = pyttsx3.init()
            _pyttsx3_engine.setProperty('rate', 150)
            _pyttsx3_engine.setProperty('volume', 0.9)
        
        voice_name = getattr(tmp, 'TTS_VOICE', None)
        if voice_name:
            voice_id = get_voice_id(voice_name)
            if voice_id:
                _pyttsx3_engine.setProperty('voice', voice_id)
        
        _pyttsx3_engine.say(text)
        _pyttsx3_engine.runAndWait()
        print(f"[TTS-LINUX] Spoke: {text[:50]}...")
    except Exception as e:
        print(f"[TTS-LINUX] pyttsx3 error: {e}")
        _speak_espeak_fallback(text)


def _speak_pyttsx3_fallback(text: str) -> None:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except:
        pass


def _speak_espeak_fallback(text: str) -> None:
    try:
        subprocess.run(['espeak', '-v', 'en', text], timeout=30, check=False,
                      stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    except:
        pass


# =============================================================================
# BEEP
# =============================================================================

def beep() -> None:
    tmp = _get_temporary()
    if not getattr(tmp, "BLEEP_ON_EVENTS", False):
        return
    if tmp.PLATFORM == "windows":
        _beep_windows()
    elif tmp.PLATFORM == "linux":
        _beep_linux()


def _beep_windows() -> None:
    try:
        import winsound
        winsound.Beep(1000, 150)
    except:
        try:
            import winsound
            winsound.MessageBeep(winsound.MB_OK)
        except:
            pass


def _beep_linux() -> None:
    methods = [
        lambda: subprocess.run(['beep', '-f', '1000', '-l', '150'], timeout=2, check=True, 
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL),
        lambda: subprocess.run(['paplay', '/usr/share/sounds/freedesktop/stereo/complete.oga'], 
                              timeout=2, check=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL) 
                if os.path.exists('/usr/share/sounds/freedesktop/stereo/complete.oga') else (_ for _ in ()).throw(Exception()),
        lambda: subprocess.run(['play', '-n', 'synth', '0.15', 'sin', '1000'], timeout=2, check=True,
                              stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL),
        lambda: print("\a", end="", flush=True),
    ]
    for method in methods:
        try:
            method()
            return
        except:
            continue


# =============================================================================
# CLEANUP
# =============================================================================

def cleanup_tts_resources() -> None:
    global _pyttsx3_engine
    tmp = _get_temporary()
    try:
        if tmp.PLATFORM == "windows":
            import pythoncom
            try:
                pythoncom.CoUninitialize()
            except:
                pass
        elif _pyttsx3_engine is not None:
            try:
                _pyttsx3_engine.stop()
            except:
                pass
            _pyttsx3_engine = None
    except:
        pass