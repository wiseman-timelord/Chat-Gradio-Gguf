# scripts/tools.py
"""
Centralized tools module for TTS, web search, and system sounds.

TTS Engines:
- Coqui TTS (neural): Windows 8.1-11 + Ubuntu 22-25 with Python 3.10+
- pyttsx3 + SAPI5: Windows 7 or Windows with Python 3.9
- pyttsx3 + espeak: Linux with Python 3.9

Search Tools:
- DDG Hybrid Search: DDG snippets + full article fetching via newspaper
- Web Search: Comprehensive multi-source web search with parallel page fetching
"""

import os
import subprocess
import threading
import queue
from pathlib import Path
from datetime import datetime
import asyncio
import re
from urllib.parse import urlparse, urljoin, quote_plus
from typing import List, Dict, Tuple, Optional, Any
import time

# Lazy import to avoid circular dependency
def _get_temporary():
    import scripts.temporary as temporary
    return temporary


# =============================================================================
# TTS CONFIGURATION
# =============================================================================

_pyttsx3_engine = None
_tts_initialized = False
_current_engine = None


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
        
        host_apis = sd.query_hostapis()
        
        preferred_api_idx = None
        if platform == "windows":
            for idx, api in enumerate(host_apis):
                if "WASAPI" in api.get('name', ''):
                    preferred_api_idx = idx
                    break
        else:
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
            if dev.get('max_output_channels', 0) <= 0:
                continue
            
            if preferred_api_idx is not None:
                if dev.get('hostapi') != preferred_api_idx:
                    continue
            
            name = dev['name']
            for suffix in ['(WASAPI)', '(DirectSound)', '(MME)', '(Windows WASAPI)', '(Windows DirectSound)']:
                name = name.replace(suffix, '').strip()
            
            name = name[:45]
            
            if name.lower() in seen_names:
                continue
            seen_names.add(name.lower())
            
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
    tmp = _get_temporary()
    return getattr(tmp, 'SAMPLE_RATES', [22050, 44100, 48000])


# =============================================================================
# VOICE OPTIONS
# =============================================================================

def get_installed_coqui_voices() -> list:
    """Scan for installed Coqui TTS voices and return list of available voices."""
    tmp = _get_temporary()
    COQUI_DEFAULT_VOICES = getattr(tmp, 'COQUI_DEFAULT_VOICES', {
        "english_male": "tts_models/en/ljspeech/tacotron2-DDC",
        "english_female": "tts_models/en/ljspeech/glow-tts",
        "american_male": "tts_models/en/ljspeech/speedy-speech",
        "american_female": "tts_models/en/ljspeech/vits"
    })
    
    voices = []
    
    cache_dir = Path(__file__).parent.parent / "data" / "tts_models"
    
    home = Path.home()
    coqui_paths = [
        cache_dir,
        home / ".local" / "share" / "tts",
        home / "AppData" / "Local" / "tts",
    ]
    
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
        
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith("tts_models--"):
                parts = item.name.split("--")
                if len(parts) >= 4:
                    model_key = parts[-1]
                    if model_key in model_names and model_key not in found_models:
                        found_models.add(model_key)
                        voices.append({
                            "id": model_key,
                            "name": model_names[model_key],
                            "model_path": f"tts_models/en/ljspeech/{model_key}"
                        })
        
        for model_file in base_path.rglob("*.pth"):
            parent = model_file.parent.name
            if parent in model_names and parent not in found_models:
                found_models.add(parent)
                voices.append({
                    "id": parent,
                    "name": model_names[parent],
                    "model_path": f"tts_models/en/ljspeech/{parent}"
                })
            grandparent = model_file.parent.parent.name
            for key in model_names:
                if key in grandparent and key not in found_models:
                    found_models.add(key)
                    voices.append({
                        "id": key,
                        "name": model_names[key],
                        "model_path": f"tts_models/en/ljspeech/{key}"
                    })
    
    if not voices:
        voices = [
            {"id": "tacotron2-DDC", "name": "English Male", "model_path": COQUI_DEFAULT_VOICES["english_male"]},
            {"id": "glow-tts", "name": "English Female", "model_path": COQUI_DEFAULT_VOICES["english_female"]},
            {"id": "speedy-speech", "name": "American Male", "model_path": COQUI_DEFAULT_VOICES["american_male"]},
            {"id": "vits", "name": "American Female", "model_path": COQUI_DEFAULT_VOICES["american_female"]},
        ]
    
    def voice_sort_key(v):
        name = v["name"]
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
    tmp = _get_temporary()
    COQUI_DEFAULT_VOICES = getattr(tmp, 'COQUI_DEFAULT_VOICES', {})
    
    if _current_engine == "coqui":
        for v in get_installed_coqui_voices():
            if v["name"] == voice_name:
                return v["model_path"]
        return COQUI_DEFAULT_VOICES.get("english_female", "tts_models/en/ljspeech/glow-tts")
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
    tmp = _get_temporary()
    if _current_engine == "coqui":
        _speak_coqui(text)
    elif _current_engine == "pyttsx3":
        if tmp.PLATFORM == "windows":
            _speak_pyttsx3_windows(text)
        else:
            _speak_pyttsx3_linux(text)


def _speak_coqui(text: str) -> None:
    tmp = _get_temporary()
    COQUI_DEFAULT_VOICES = getattr(tmp, 'COQUI_DEFAULT_VOICES', {})
    
    try:
        import sounddevice as sd
        from TTS.api import TTS
        
        voice_name = getattr(tmp, 'TTS_VOICE', None)
        model_name = get_voice_model_path(voice_name) if voice_name else COQUI_DEFAULT_VOICES.get("english_female", "tts_models/en/ljspeech/glow-tts")
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


# =============================================================================
# WEB SEARCH - COMPREHENSIVE MULTI-SOURCE SEARCH
# =============================================================================

class WebSearchEngine:
    """
    Comprehensive web search engine that:
    1. Searches multiple sources for titles/descriptions
    2. Ranks and selects the most relevant pages
    3. Fetches full content from selected pages in parallel
    4. Extracts and processes content intelligently
    5. Returns a comprehensive summary for the LLM
    """
    
    # Domain quality scores for ranking
    DOMAIN_QUALITY = {
        'high': ['wikipedia.org', 'britannica.com', 'reuters.com', 'bbc.com', 'bbc.co.uk',
                 'nytimes.com', 'theguardian.com', 'washingtonpost.com', 'apnews.com',
                 'npr.org', 'nature.com', 'science.org', 'gov', 'edu', 'arxiv.org',
                 'sciencedirect.com', 'pubmed.ncbi.nlm.nih.gov', 'smithsonianmag.com'],
        'medium': ['medium.com', 'substack.com', 'cnn.com', 'forbes.com', 'wired.com',
                   'techcrunch.com', 'arstechnica.com', 'theatlantic.com', 'economist.com',
                   'ft.com', 'bloomberg.com', 'wsj.com', 'aljazeera.com', 'dw.com'],
        'low': ['reddit.com', 'quora.com', 'twitter.com', 'facebook.com', 'pinterest.com']
    }
    
    # User agent rotation for avoiding blocks
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0'
    ]
    
    def __init__(self):
        self._session = None
        self._user_agent_idx = 0
    
    def _get_user_agent(self) -> str:
        """Rotate user agents to avoid detection."""
        ua = self.USER_AGENTS[self._user_agent_idx % len(self.USER_AGENTS)]
        self._user_agent_idx += 1
        return ua
    
    def _get_domain_score(self, url: str) -> int:
        """Score a domain based on quality/trustworthiness."""
        try:
            domain = urlparse(url).netloc.lower()
            domain = domain.replace('www.', '')
            
            for high_domain in self.DOMAIN_QUALITY['high']:
                if high_domain in domain:
                    return 3
            for med_domain in self.DOMAIN_QUALITY['medium']:
                if med_domain in domain:
                    return 2
            for low_domain in self.DOMAIN_QUALITY['low']:
                if low_domain in domain:
                    return 0
            return 1  # Default score for unknown domains
        except:
            return 1
    
    def _score_search_result(self, result: Dict, query_words: set) -> int:
        """Score a search result based on relevance to query."""
        score = 0
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        url = result.get('url', '')
        
        # Query word matches
        for word in query_words:
            if len(word) > 3:
                if word in title:
                    score += 5
                if word in snippet:
                    score += 2
        
        # Domain quality bonus
        score += self._get_domain_score(url) * 2
        
        # Date bonus (if present)
        if result.get('date'):
            score += 3
        
        # Length bonus for informative snippets
        if len(snippet) > 150:
            score += 2
        
        return score
    
    def _search_duckduckgo_html(self, query: str, max_results: int = 15) -> List[Dict]:
        """
        Search DuckDuckGo via HTML scraping (more comprehensive than API).
        Returns list of {title, url, snippet, date, source}
        """
        import requests
        from bs4 import BeautifulSoup
        
        results = []
        try:
            # Use DuckDuckGo HTML search
            search_url = f"https://html.duckduckgo.com/html/"
            params = {'q': query, 'kl': 'wt-wt'}
            
            headers = {
                'User-Agent': self._get_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Referer': 'https://duckduckgo.com/'
            }
            
            response = requests.post(search_url, data=params, headers=headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Parse results
            for result_div in soup.select('.result')[:max_results]:
                try:
                    title_elem = result_div.select_one('.result__title a')
                    snippet_elem = result_div.select_one('.result__snippet')
                    
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    
                    # Extract actual URL from DDG redirect
                    if 'uddg=' in url:
                        import urllib.parse
                        parsed = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
                        if 'uddg' in parsed:
                            url = parsed['uddg'][0]
                    
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                    
                    if title and url and url.startswith('http'):
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'date': '',
                            'source': 'duckduckgo'
                        })
                except Exception as e:
                    continue
            
            print(f"[WEB-SEARCH] DDG HTML returned {len(results)} results")
            
        except Exception as e:
            print(f"[WEB-SEARCH] DDG HTML error: {e}")
        
        return results
    
    def _search_ddgs_api(self, query: str, max_results: int = 10) -> List[Dict]:
        """
        Search using ddgs library as fallback/supplement.
        """
        results = []
        try:
            from ddgs import DDGS
            
            ddgs = DDGS(timeout=15)
            
            # Detect if news query
            news_keywords = ['news', 'latest', 'current', 'recent', 'today', 'breaking',
                           '2024', '2025', '2026', '2027']
            is_news = any(kw in query.lower() for kw in news_keywords)
            
            if is_news:
                raw_results = list(ddgs.news(query, region="wt-wt", safesearch="off",
                                            timelimit="m", max_results=max_results))
                for r in raw_results:
                    results.append({
                        'title': r.get('title', ''),
                        'url': r.get('url', ''),
                        'snippet': r.get('body', ''),
                        'date': r.get('date', ''),
                        'source': r.get('source', 'ddgs_news')
                    })
            else:
                raw_results = list(ddgs.text(query, region="wt-wt", safesearch="off",
                                            timelimit="m", max_results=max_results))
                for r in raw_results:
                    results.append({
                        'title': r.get('title', ''),
                        'url': r.get('href', ''),
                        'snippet': r.get('body', ''),
                        'date': '',
                        'source': 'ddgs_text'
                    })
            
            print(f"[WEB-SEARCH] DDGS API returned {len(results)} results")
            
        except Exception as e:
            print(f"[WEB-SEARCH] DDGS API error: {e}")
        
        return results
    
    async def _fetch_page_async(self, session, url: str, timeout: int = 10) -> Tuple[str, str, bool]:
        """
        Asynchronously fetch a page and extract its content.
        Returns: (url, content, success)
        """
        import aiohttp
        from bs4 import BeautifulSoup
        
        try:
            headers = {
                'User-Agent': self._get_user_agent(),
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5'
            }
            
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout),
                                  allow_redirects=True, ssl=False) as response:
                if response.status != 200:
                    return (url, f"HTTP {response.status}", False)
                
                html = await response.text()
                
                # Try newspaper first for article extraction
                try:
                    from newspaper import Article
                    article = Article(url)
                    article.set_html(html)
                    article.parse()
                    
                    if article.text and len(article.text) > 200:
                        content = article.text
                        if article.publish_date:
                            content = f"[Published: {article.publish_date.strftime('%Y-%m-%d')}]\n{content}"
                        return (url, content[:5000], True)
                except:
                    pass
                
                # Fallback to BeautifulSoup extraction
                soup = BeautifulSoup(html, 'lxml')
                
                # Remove unwanted elements
                for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 
                               'advertisement', 'iframe', 'noscript']):
                    tag.decompose()
                
                # Try to find main content
                main_content = None
                for selector in ['article', 'main', '[role="main"]', '.article-body', 
                               '.post-content', '.entry-content', '#content', '.content']:
                    main_content = soup.select_one(selector)
                    if main_content:
                        break
                
                if main_content:
                    text = main_content.get_text(separator='\n', strip=True)
                else:
                    text = soup.get_text(separator='\n', strip=True)
                
                # Clean up the text
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                text = '\n'.join(lines)
                
                if len(text) > 200:
                    return (url, text[:5000], True)
                else:
                    return (url, "Content too short", False)
                    
        except asyncio.TimeoutError:
            return (url, "Timeout", False)
        except Exception as e:
            return (url, str(e)[:100], False)
    
    async def _fetch_pages_parallel(self, urls: List[str], max_concurrent: int = 5) -> List[Tuple[str, str, bool]]:
        """
        Fetch multiple pages in parallel using aiohttp.
        """
        import aiohttp
        
        results = []
        connector = aiohttp.TCPConnector(limit=max_concurrent, ssl=False)
        
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [self._fetch_page_async(session, url) for url in urls]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append((urls[i], str(result)[:100], False))
                else:
                    processed_results.append(result)
            
            return processed_results
    
    def search(self, query: str, max_results: int = 12, deep_fetch: int = 6,
              progress_callback=None) -> Dict:
        """
        Perform comprehensive web search.
        
        Phases:
        1. Search Discovery: Get search results from multiple sources
        2. Rank & Select: Score and rank results, select top for deep fetch
        3. Deep Fetch: Fetch full content from top pages in parallel
        4. Process & Merge: Extract, process, and merge all content
        
        Args:
            query: Search query
            max_results: Maximum search results to fetch
            deep_fetch: Number of pages to deep fetch
            progress_callback: Optional callback for progress updates
        
        Returns:
            dict: {
                'content': str - Combined search results for LLM context,
                'metadata': {
                    'type': 'web_search',
                    'query': str,
                    'total_results': int,
                    'pages_fetched': int,
                    'sources': list,
                    'error': str or None
                }
            }
        """
        current_date = datetime.now().strftime("%B %d, %Y")
        current_year = datetime.now().year
        
        # Add current year for time-sensitive queries
        news_keywords = ['news', 'latest', 'current', 'recent', 'today', 'breaking',
                        'protests', 'election', 'war', 'crisis', 'update', 'happening']
        is_news = any(kw in query.lower() for kw in news_keywords)
        has_year = any(str(y) in query for y in range(2020, 2030))
        
        search_query = f"{query} {current_year}" if is_news and not has_year else query
        
        header = (
            f"[Web Search Results - Comprehensive Multi-Source]\n"
            f"[Current Date: {current_date}]\n"
            f"[Query: {search_query}]\n"
            f"[Mode: Multi-source search → Parallel deep fetch]\n\n"
        )
        
        empty_result = {
            'content': header,
            'metadata': {
                'type': 'web_search',
                'query': search_query,
                'total_results': 0,
                'pages_fetched': 0,
                'sources': [],
                'error': None
            }
        }
        
        try:
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 1: Search Discovery - Get results from multiple sources
            # ═══════════════════════════════════════════════════════════════════
            print(f"[WEB-SEARCH] ══════════════════════════════════════════════════════")
            print(f"[WEB-SEARCH] Phase 1: Search Discovery")
            print(f"[WEB-SEARCH] Query: '{search_query}'")
            
            if progress_callback:
                progress_callback(1, "Searching multiple sources...")
            
            all_results = []
            
            # Try DDG HTML scraping first (more comprehensive)
            html_results = self._search_duckduckgo_html(search_query, max_results)
            all_results.extend(html_results)
            
            # Supplement with DDGS API if needed
            if len(all_results) < max_results // 2:
                api_results = self._search_ddgs_api(search_query, max_results)
                # Deduplicate by URL
                existing_urls = {r['url'] for r in all_results}
                for r in api_results:
                    if r['url'] not in existing_urls:
                        all_results.append(r)
                        existing_urls.add(r['url'])
            
            if not all_results:
                print("[WEB-SEARCH] No results found")
                empty_result['content'] = header + "[No Results] Web search returned no results."
                empty_result['metadata']['error'] = "No results found"
                return empty_result
            
            print(f"[WEB-SEARCH] Total results: {len(all_results)}")
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 2: Rank & Select - Score results and select top for deep fetch
            # ═══════════════════════════════════════════════════════════════════
            print(f"[WEB-SEARCH] Phase 2: Ranking {len(all_results)} results")
            
            if progress_callback:
                progress_callback(2, "Ranking and selecting best sources...")
            
            query_words = set(word.lower() for word in search_query.split() if len(word) > 3)
            
            scored_results = []
            for result in all_results:
                score = self._score_search_result(result, query_words)
                scored_results.append((score, result))
            
            # Sort by score descending
            scored_results.sort(key=lambda x: x[0], reverse=True)
            
            # Select top for deep fetch
            top_results = [r for _, r in scored_results[:deep_fetch]]
            remaining_results = [r for _, r in scored_results[deep_fetch:max_results]]
            
            print(f"[WEB-SEARCH] Selected {len(top_results)} for deep fetch:")
            for i, r in enumerate(top_results, 1):
                print(f"[WEB-SEARCH]   {i}. {r['title'][:50]}...")
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 3: Deep Fetch - Fetch full content from top pages in parallel
            # ═══════════════════════════════════════════════════════════════════
            print(f"[WEB-SEARCH] Phase 3: Deep fetching {len(top_results)} pages")
            
            if progress_callback:
                progress_callback(3, f"Fetching {len(top_results)} pages in parallel...")
            
            urls_to_fetch = [r['url'] for r in top_results]
            
            # Run async fetch in a new event loop (compatible with sync context)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                fetch_results = loop.run_until_complete(
                    self._fetch_pages_parallel(urls_to_fetch, max_concurrent=5)
                )
            finally:
                loop.close()
            
            # ═══════════════════════════════════════════════════════════════════
            # PHASE 4: Process & Merge - Combine all content
            # ═══════════════════════════════════════════════════════════════════
            print(f"[WEB-SEARCH] Phase 4: Processing and merging results")
            
            if progress_callback:
                progress_callback(4, "Processing and merging content...")
            
            deep_content = []
            sources = []
            fetched_count = 0
            
            for i, (url, content, success) in enumerate(fetch_results):
                result = top_results[i]
                title = result['title']
                snippet = result['snippet']
                
                source_info = {
                    'title': title,
                    'url': url,
                    'fetched': success,
                    'type': 'deep'
                }
                sources.append(source_info)
                
                if success:
                    fetched_count += 1
                    # Truncate content if too long
                    if len(content) > 4000:
                        content = content[:4000] + "\n[...truncated...]"
                    
                    deep_content.append(f"══════════════════════════════════════════════════════")
                    deep_content.append(f"📰 PAGE {i+1}: {title}")
                    deep_content.append(f"══════════════════════════════════════════════════════")
                    deep_content.append(f"URL: {url}")
                    deep_content.append(f"\n{content}\n")
                    print(f"[WEB-SEARCH]   ✓ [{i+1}] {title[:40]}... ({len(content)} chars)")
                else:
                    # Use snippet as fallback
                    deep_content.append(f"──────────────────────────────────────────────────────")
                    deep_content.append(f"📋 PAGE {i+1} (snippet only): {title}")
                    deep_content.append(f"──────────────────────────────────────────────────────")
                    deep_content.append(f"URL: {url}")
                    deep_content.append(f"\n{snippet}\n")
                    print(f"[WEB-SEARCH]   ○ [{i+1}] {title[:40]}... (snippet only: {content})")
            
            # Add remaining results as snippets
            snippet_content = []
            if remaining_results:
                snippet_content.append("\n──────────────────────────────────────────────────────")
                snippet_content.append("📋 ADDITIONAL SOURCES (Search Snippets)")
                snippet_content.append("──────────────────────────────────────────────────────")
                
                for i, result in enumerate(remaining_results, 1):
                    title = result['title']
                    snippet = result['snippet']
                    url = result['url']
                    date = result.get('date', '')
                    
                    sources.append({
                        'title': title,
                        'url': url,
                        'fetched': False,
                        'type': 'snippet'
                    })
                    
                    date_str = f" ({date})" if date else ""
                    snippet_content.append(f"\n[{i}] {title}{date_str}")
                    snippet_content.append(f"    {snippet[:250]}..." if len(snippet) > 250 else f"    {snippet}")
                    snippet_content.append(f"    <{url}>")
            
            # Combine all content
            final_content = header
            final_content += "\n".join(deep_content)
            final_content += "\n".join(snippet_content)
            
            print(f"[WEB-SEARCH] ══════════════════════════════════════════════════════")
            print(f"[WEB-SEARCH] Complete: {fetched_count} pages fetched, {len(remaining_results)} snippets")
            print(f"[WEB-SEARCH] Total content: {len(final_content)} chars")
            
            return {
                'content': final_content,
                'metadata': {
                    'type': 'web_search',
                    'query': search_query,
                    'total_results': len(all_results),
                    'pages_fetched': fetched_count,
                    'sources': sources,
                    'error': None
                }
            }
            
        except Exception as e:
            print(f"[WEB-SEARCH] Error: {e}")
            import traceback
            traceback.print_exc()
            empty_result['content'] = header + f"[Search Error] {str(e)}"
            empty_result['metadata']['error'] = str(e)
            return empty_result


# Global web search engine instance
_web_search_engine = None

def get_web_search_engine() -> WebSearchEngine:
    """Get or create the global web search engine instance."""
    global _web_search_engine
    if _web_search_engine is None:
        _web_search_engine = WebSearchEngine()
    return _web_search_engine


def web_search(query: str, max_results: int = 12, deep_fetch: int = 6,
              progress_callback=None) -> dict:
    """
    Perform comprehensive web search (convenience function).
    
    This is the main entry point for web search functionality.
    Uses multiple search sources and parallel page fetching for comprehensive results.
    
    Args:
        query: Search query
        max_results: Maximum search results to analyze
        deep_fetch: Number of pages to fetch full content from
        progress_callback: Optional callback for progress updates
    
    Returns:
        dict: {
            'content': str - Combined search results for LLM context,
            'metadata': {
                'type': 'web_search',
                'query': str,
                'total_results': int,
                'pages_fetched': int,
                'sources': list,
                'error': str or None
            }
        }
    """
    engine = get_web_search_engine()
    return engine.search(query, max_results, deep_fetch, progress_callback)


def format_web_search_status_for_chat(search_metadata: dict) -> str:
    """
    Format web search metadata into a readable status string for the chat display.
    
    Args:
        search_metadata: The 'metadata' dict from web_search
        
    Returns:
        Formatted string showing search activity
    """
    if not search_metadata:
        return ""
    
    lines = []
    search_type = search_metadata.get('type', 'unknown')
    query = search_metadata.get('query', '')
    sources = search_metadata.get('sources', [])
    error = search_metadata.get('error')
    total_results = search_metadata.get('total_results', 0)
    pages_fetched = search_metadata.get('pages_fetched', 0)
    
    # Truncate long queries for display
    display_query = query[:80] + "..." if len(query) > 80 else query
    
    if search_type == 'web_search':
        if error:
            lines.append(f"🌐 Web Search: \"{display_query}\" — ⚠️ {error}")
        else:
            deep_sources = [s for s in sources if s.get('type') == 'deep']
            snippet_sources = [s for s in sources if s.get('type') == 'snippet']
            
            lines.append(f"🌐 Web Search: \"{display_query}\"")
            lines.append(f"   📊 {total_results} results found")
            lines.append(f"   📰 {pages_fetched}/{len(deep_sources)} pages deep-fetched")
            if snippet_sources:
                lines.append(f"   📋 {len(snippet_sources)} additional snippets")
            
            # Show domains of successfully fetched pages
            for source in deep_sources:
                if source.get('fetched'):
                    url = source.get('url', '')
                    try:
                        domain = urlparse(url).netloc
                        if domain.startswith('www.'):
                            domain = domain[4:]
                        lines.append(f"      ✓ {domain}")
                    except:
                        pass
    
    return "\n".join(lines)


# =============================================================================
# WEB SEARCH - DDG HYBRID (EXISTING IMPLEMENTATION)
# =============================================================================

def hybrid_search(query: str, ddg_results: int = 8, deep_fetch: int = 4) -> dict:
    """
    Hybrid search: DDG pre-research followed by targeted deep article fetching.
    
    This provides the best of both worlds:
    - DDG gives quick overview of many sources (breadth)
    - Deep fetch gives full content from top sources (depth)
    
    Workflow:
    1. DDG Pre-Search: Quick DDG search to discover sources with snippets
    2. Analyze Results: Score and rank sources based on relevance
    3. Deep Fetch: Extract full article content from top sources
    4. Merge Results: Combine deep articles + DDG snippets
    
    Args:
        query: Search query
        ddg_results: Number of DDG results to fetch (default 8)
        deep_fetch: Number of top results to deep fetch (default 4)
    
    Returns:
        dict: {
            'content': str - Combined search results for LLM context,
            'metadata': {
                'type': 'hybrid',
                'query': str,
                'ddg_count': int,
                'deep_count': int,
                'sources': list,
                'error': str or None
            }
        }
    """
    from ddgs import DDGS
    from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
    from newspaper import Article
    import requests.exceptions
    
    tmp = _get_temporary()
    
    print(f"[HYBRID] ══════════════════════════════════════════════════════")
    print(f"[HYBRID] Starting hybrid search")
    print(f"[HYBRID] Query: '{query}'")
    print(f"[HYBRID] DDG results: {ddg_results}, Deep fetch: {deep_fetch}")
    print(f"[HYBRID] ══════════════════════════════════════════════════════")
    
    current_date = datetime.now().strftime("%B %d, %Y")
    current_year = datetime.now().year
    
    # Detect news/current events
    news_keywords = [
        'news', 'latest', 'current', 'recent', 'today', 'breaking',
        'protests', 'uprising', 'election', 'war', 'crisis', 'conflict',
        'headlines', 'developments', 'happening', 'update',
        'january', 'february', 'march', 'april', 'may', 'june',
        'july', 'august', 'september', 'october', 'november', 'december',
        '2024', '2025', '2026', '2027'
    ]
    
    query_lower = query.lower()
    is_news_query = any(kw in query_lower for kw in news_keywords)
    has_year = any(str(y) in query_lower for y in range(2020, 2030))
    
    search_query = f"{query} {current_year}" if is_news_query and not has_year else query
    
    header = (
        f"[Hybrid Search Results - DDG Overview + Deep Articles]\n"
        f"[Current Date: {current_date}]\n"
        f"[Query: {search_query}]\n"
        f"[Mode: DDG pre-research ({ddg_results} sources) → Deep fetch (top {deep_fetch} articles)]\n\n"
    )
    
    empty_result = {
        'content': header,
        'metadata': {
            'type': 'hybrid',
            'query': search_query,
            'ddg_count': 0,
            'deep_count': 0,
            'sources': [],
            'error': None
        }
    }
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 1: DDG Pre-Search - Get overview of available sources
    # ═══════════════════════════════════════════════════════════════════════
    print(f"[HYBRID] Phase 1: DDG Pre-Search")
    
    ddg_hits = []
    try:
        ddgs = DDGS(timeout=20)
        
        if is_news_query:
            print(f"[HYBRID]   Using NEWS search")
            raw_hits = list(ddgs.news(
                search_query,
                region="wt-wt",
                safesearch="off",
                timelimit="m",
                max_results=ddg_results
            ))
            # Normalize news results
            for h in raw_hits:
                ddg_hits.append({
                    'title': h.get('title', 'Untitled'),
                    'body': h.get('body', ''),
                    'href': h.get('url', ''),
                    'date': h.get('date', ''),
                    'source': h.get('source', '')
                })
        else:
            print(f"[HYBRID]   Using TEXT search")
            raw_hits = list(ddgs.text(
                search_query,
                region="wt-wt",
                safesearch="off",
                timelimit="m",
                max_results=ddg_results
            ))
            for h in raw_hits:
                ddg_hits.append({
                    'title': h.get('title', 'Untitled'),
                    'body': h.get('body', ''),
                    'href': h.get('href', ''),
                    'date': '',
                    'source': ''
                })
        
        print(f"[HYBRID]   Got {len(ddg_hits)} DDG results")
        
    except (DDGSException, RatelimitException, TimeoutException) as e:
        print(f"[HYBRID]   DDG error: {e}")
        empty_result['content'] = header + f"[Search Error] DDG failed: {e}"
        empty_result['metadata']['error'] = str(e)
        return empty_result
    except Exception as e:
        print(f"[HYBRID]   Unexpected error: {e}")
        empty_result['content'] = header + f"[Search Error] {e}"
        empty_result['metadata']['error'] = str(e)
        return empty_result
    
    if not ddg_hits:
        print("[HYBRID]   No DDG results")
        empty_result['content'] = header + "[No Results] DDG returned no results."
        return empty_result
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 2: Analyze DDG Results - Rank and select top sources for deep fetch
    # ═══════════════════════════════════════════════════════════════════════
    print(f"[HYBRID] Phase 2: Analyzing DDG results for deep fetch candidates")
    
    # Score sources based on relevance indicators
    scored_hits = []
    query_words = set(search_query.lower().split())
    
    for hit in ddg_hits:
        score = 0
        title_lower = hit['title'].lower()
        body_lower = hit['body'].lower()
        
        # Score based on query word matches
        for word in query_words:
            if len(word) > 3:  # Skip short words
                if word in title_lower:
                    score += 3
                if word in body_lower:
                    score += 1
        
        # Bonus for news sources (usually more relevant for current events)
        if hit.get('date'):
            score += 2
        
        # Bonus for reputable domains
        url = hit.get('href', '').lower()
        reputable = ['reuters', 'bbc', 'npr', 'guardian', 'nytimes', 'washingtonpost', 
                     'aljazeera', 'apnews', 'wikipedia', '.gov', '.edu', 'britannica']
        if any(r in url for r in reputable):
            score += 3
        
        scored_hits.append((score, hit))
    
    # Sort by score descending
    scored_hits.sort(key=lambda x: x[0], reverse=True)
    
    # Select top N for deep fetch
    deep_fetch_candidates = [hit for score, hit in scored_hits[:deep_fetch]]
    remaining_ddg = [hit for score, hit in scored_hits[deep_fetch:]]
    
    print(f"[HYBRID]   Selected {len(deep_fetch_candidates)} for deep fetch")
    for i, hit in enumerate(deep_fetch_candidates, 1):
        print(f"[HYBRID]     {i}. {hit['title'][:50]}...")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 3: Deep Fetch - Get full article content from top sources
    # ═══════════════════════════════════════════════════════════════════════
    print(f"[HYBRID] Phase 3: Deep fetching top {len(deep_fetch_candidates)} articles")
    
    deep_results = []
    deep_sources = []
    
    for i, hit in enumerate(deep_fetch_candidates, 1):
        url = hit.get('href', '')
        title = hit.get('title', 'Untitled')
        snippet = hit.get('body', '')
        date = hit.get('date', '')
        source = hit.get('source', '')
        
        print(f"[HYBRID]   [{i}/{len(deep_fetch_candidates)}] Fetching: {title[:40]}...")
        
        source_info = {'title': title, 'url': url, 'fetched': False, 'type': 'deep'}
        article_content = ""
        
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text) > 100:
                article_content = article.text[:3000]  # More content for deep fetch
                if len(article.text) > 3000:
                    article_content += "\n[...truncated...]"
                if article.publish_date:
                    article_content = f"[Published: {article.publish_date.strftime('%Y-%m-%d')}]\n{article_content}"
                elif date:
                    article_content = f"[Date: {date}]\n{article_content}"
                source_info['fetched'] = True
                print(f"[HYBRID]     ✓ Fetched {len(article_content)} chars")
            else:
                print(f"[HYBRID]     ○ Article too short, using snippet")
                article_content = snippet
        except Exception as e:
            print(f"[HYBRID]     ✗ Failed: {e}")
            article_content = snippet
        
        deep_sources.append(source_info)
        
        deep_results.append(f"══════════════════════════════════════════════════════")
        deep_results.append(f"📰 DEEP ARTICLE {i}: {title}")
        deep_results.append(f"══════════════════════════════════════════════════════")
        deep_results.append(f"URL: {url}")
        if source:
            deep_results.append(f"Source: {source}")
        deep_results.append(f"\n{article_content or snippet}\n")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PHASE 4: Merge Results - Combine deep articles + DDG snippets
    # ═══════════════════════════════════════════════════════════════════════
    print(f"[HYBRID] Phase 4: Merging results")
    
    ddg_sources = []
    ddg_summaries = []
    
    # Add remaining DDG results as quick reference
    if remaining_ddg:
        ddg_summaries.append("\n──────────────────────────────────────────────────────")
        ddg_summaries.append("📋 ADDITIONAL SOURCES (DDG Snippets)")
        ddg_summaries.append("──────────────────────────────────────────────────────")
        
        for i, hit in enumerate(remaining_ddg, 1):
            title = hit.get('title', 'Untitled')
            body = hit.get('body', '')
            url = hit.get('href', '')
            date = hit.get('date', '')
            
            ddg_sources.append({'title': title, 'url': url, 'fetched': False, 'type': 'ddg'})
            
            date_str = f" ({date})" if date else ""
            ddg_summaries.append(f"\n[{i}] {title}{date_str}")
            ddg_summaries.append(f"    {body[:200]}..." if len(body) > 200 else f"    {body}")
            ddg_summaries.append(f"    <{url}>")
    
    # Combine all
    all_sources = deep_sources + ddg_sources
    fetched_count = sum(1 for s in all_sources if s.get('fetched', False))
    
    final_content = header
    final_content += "\n".join(deep_results)
    final_content += "\n".join(ddg_summaries)
    
    print(f"[HYBRID] Complete: {fetched_count} deep fetched, {len(ddg_sources)} DDG snippets")
    print(f"[HYBRID] Total content: {len(final_content)} chars")
    
    return {
        'content': final_content,
        'metadata': {
            'type': 'hybrid',
            'query': search_query,
            'ddg_count': len(ddg_hits),
            'deep_count': fetched_count,
            'sources': all_sources,
            'error': None
        }
    }


def format_search_status_for_chat(search_metadata: dict) -> str:
    """
    Format search metadata into a readable status string for the chat display.
    This appears before the AI response to show what was searched.
    
    Args:
        search_metadata: The 'metadata' dict from hybrid_search
        
    Returns:
        Formatted string showing search activity
    """
    if not search_metadata:
        return ""
    
    lines = []
    search_type = search_metadata.get('type', 'unknown')
    query = search_metadata.get('query', '')
    sources = search_metadata.get('sources', [])
    error = search_metadata.get('error')
    
    # Truncate long queries for display
    display_query = query[:80] + "..." if len(query) > 80 else query
    
    if search_type == 'hybrid':
        # Hybrid search - show DDG + deep fetch summary
        if error:
            lines.append(f"🔍 Hybrid Search: \"{display_query}\" — ⚠️ {error}")
        else:
            deep_sources = [s for s in sources if s.get('type') == 'deep']
            ddg_sources = [s for s in sources if s.get('type') == 'ddg']
            fetched = sum(1 for s in deep_sources if s.get('fetched', False))
            
            lines.append(f"🔍 Hybrid Search: \"{display_query}\"")
            lines.append(f"   📰 {fetched}/{len(deep_sources)} articles deep-fetched")
            if ddg_sources:
                lines.append(f"   📋 {len(ddg_sources)} additional DDG snippets")
            
            # Show domains of deep-fetched articles
            for source in deep_sources:
                if source.get('fetched'):
                    url = source.get('url', '')
                    try:
                        domain = urlparse(url).netloc
                        if domain.startswith('www.'):
                            domain = domain[4:]
                        lines.append(f"      ✓ {domain}")
                    except:
                        pass
    
    elif search_type == 'web_search':
        # Web search - delegate to specific formatter
        return format_web_search_status_for_chat(search_metadata)
    
    return "\n".join(lines)