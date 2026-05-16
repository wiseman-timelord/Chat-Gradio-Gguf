# scripts/tools.py
# v2: Windows 10-11 / Ubuntu 24-25 / Python 3.11-3.13 / Gradio 5.x
"""
Centralized tools module for web search and TTS.

Search Tools:
- Web Search: Comprehensive multi-source web search with parallel page fetching

TTS Tools:
- Text-to-Speech using Coqui TTS (VCTK model) on all supported platforms
- Audio playback via winsound (Windows) or PipeWire/PulseAudio/ALSA (Linux)
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
import tempfile

# Lazy import to avoid circular dependency
import scripts.configure as cfg

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
            return 1
        except:
            return 1

    def _score_search_result(self, result: Dict, query_words: set) -> int:
        """Score a search result based on relevance to query."""
        score = 0
        title = result.get('title', '').lower()
        snippet = result.get('snippet', '').lower()
        url = result.get('url', '')

        for word in query_words:
            if len(word) > 3:
                if word in title:
                    score += 5
                if word in snippet:
                    score += 2

        score += self._get_domain_score(url) * 2

        if result.get('date'):
            score += 3

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
            search_url = "https://html.duckduckgo.com/html/"
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

            for result_div in soup.select('.result')[:max_results]:
                try:
                    title_elem = result_div.select_one('.result__title a')
                    snippet_elem = result_div.select_one('.result__snippet')

                    if not title_elem:
                        continue

                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')

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
                except Exception:
                    continue

            print(f"[WEB-SEARCH] DDG HTML returned {len(results)} results")

        except Exception as e:
            print(f"[WEB-SEARCH] DDG HTML error: {e}")

        return results

    def _search_ddgs_api(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search using ddgs library as fallback/supplement."""
        results = []
        try:
            from ddgs import DDGS

            ddgs = DDGS(timeout=15)

            news_keywords = ['news', 'latest', 'current', 'recent', 'today', 'breaking',
                             '2024', '2025', '2026', '2027']
            is_news = any(kw in query.lower() for kw in news_keywords)

            if is_news:
                try:
                    ddg_results = list(ddgs.news(query, max_results=max_results))
                    for r in ddg_results:
                        results.append({
                            'title': r.get('title', ''),
                            'url': r.get('url', ''),
                            'snippet': r.get('body', ''),
                            'date': r.get('date', ''),
                            'source': 'ddgs_news'
                        })
                except:
                    pass

            try:
                ddg_results = list(ddgs.text(query, max_results=max_results))
                for r in ddg_results:
                    results.append({
                        'title': r.get('title', ''),
                        'url': r.get('href', ''),
                        'snippet': r.get('body', ''),
                        'date': '',
                        'source': 'ddgs_text'
                    })
            except:
                pass

            print(f"[WEB-SEARCH] DDGS API returned {len(results)} results")

        except Exception as e:
            print(f"[WEB-SEARCH] DDGS API error: {e}")

        return results

    def _fetch_page_content(self, url: str, timeout: int = 10) -> Optional[str]:
        """Fetch and extract main content from a web page."""
        try:
            from newspaper import Article

            article = Article(url)
            article.download()
            article.parse()

            if article.text and len(article.text) > 100:
                content = article.text[:4000]
                if len(article.text) > 4000:
                    content += "\n[...content truncated...]"

                if article.publish_date:
                    content = f"[Published: {article.publish_date.strftime('%Y-%m-%d')}]\n{content}"

                return content

        except Exception as e:
            print(f"[WEB-SEARCH] Failed to fetch {url}: {e}")

        return None

    def _fetch_pages_parallel(self, urls: List[str], max_workers: int = 4) -> Dict[str, str]:
        """Fetch multiple pages in parallel. Returns partial results on timeout."""
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout

        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self._fetch_page_content, url): url for url in urls}

            try:
                for future in as_completed(future_to_url, timeout=25):
                    url = future_to_url[future]
                    try:
                        content = future.result()
                        if content:
                            results[url] = content
                    except Exception as e:
                        print(f"[WEB-SEARCH] Parallel fetch error for {url}: {e}")
            except FuturesTimeout:
                finished = sum(1 for f in future_to_url if f.done())
                pending  = len(future_to_url) - finished
                print(f"[WEB-SEARCH] Fetch timeout: {finished} done, {pending} still running — using partial results")
                for future, url in future_to_url.items():
                    if future.done() and not future.cancelled():
                        try:
                            content = future.result()
                            if content:
                                results[url] = content
                        except Exception:
                            pass

        return results

    def search(self, query: str, max_results: int = 12, deep_fetch: int = 6) -> Dict:
        """Perform comprehensive web search with dependency checking."""
        missing_deps = []
        try:
            import requests
        except ImportError:
            missing_deps.append("requests")
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            missing_deps.append("beautifulsoup4")
        try:
            from newspaper import Article
        except ImportError:
            missing_deps.append("newspaper4k")
        try:
            import lxml
        except ImportError:
            missing_deps.append("lxml")

        if missing_deps:
            error_msg = (f"Web Search requires missing packages: {', '.join(missing_deps)}. "
                         f"Install with: pip install {' '.join(missing_deps)}")
            print(f"[WEB-SEARCH] {error_msg}")
            return {
                'content': f"Web search unavailable: {error_msg}",
                'metadata': {'type': 'web_search', 'query': query, 'error': error_msg, 'sources': []}
            }

        # Phase 1: Gather search results — DDGS API (primary) + DDG HTML scrape (supplemental).
        print(f"[WEB-SEARCH] Searching for: {query}")

        all_results = []

        api_results = self._search_ddgs_api(query, max_results)
        all_results.extend(api_results)

        if len(all_results) < max_results:
            html_results = self._search_duckduckgo_html(query, max_results)
            all_results.extend(html_results)

        if not all_results:
            return {
                'content': f"No search results found for: {query}\n\nCheck your internet connection or try a different query.",
                'metadata': {'type': 'web_search', 'query': query, 'error': 'No results', 'sources': []}
            }

        # Deduplicate by URL
        seen_urls = set()
        unique_results = []
        for r in all_results:
            url = r.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(r)

        # Phase 2: Score and rank results
        query_words = set(query.lower().split())
        scored_results = [(self._score_search_result(r, query_words), r) for r in unique_results]
        scored_results.sort(key=lambda x: x[0], reverse=True)

        top_results = [r for _, r in scored_results[:deep_fetch]]
        remaining_results = [r for _, r in scored_results[deep_fetch:max_results]]

        # Phase 3: Fetch full content from top results
        urls_to_fetch = [r['url'] for r in top_results if r.get('url')]
        fetched_content = self._fetch_pages_parallel(urls_to_fetch)

        # Phase 4: Build final content
        content_parts = []
        sources = []

        content_parts.append(f"═══════════════════════════════════════════════════════")
        content_parts.append(f"WEB SEARCH RESULTS: {query}")
        content_parts.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        content_parts.append(f"═══════════════════════════════════════════════════════\n")

        for i, result in enumerate(top_results, 1):
            url = result.get('url', '')
            title = result.get('title', 'Untitled')

            sources.append({
                'title': title,
                'url': url,
                'fetched': url in fetched_content,
                'type': 'deep'
            })

            content_parts.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            content_parts.append(f"📰 ARTICLE {i}: {title}")
            content_parts.append(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            content_parts.append(f"URL: {url}")

            if url in fetched_content:
                content_parts.append(f"\n{fetched_content[url]}\n")
            else:
                content_parts.append(f"\n{result.get('snippet', 'No content available')}\n")

        if remaining_results:
            content_parts.append("\n───────────────────────────────────────────────────────")
            content_parts.append("📋 ADDITIONAL SOURCES")
            content_parts.append("───────────────────────────────────────────────────────")

            for i, result in enumerate(remaining_results, 1):
                title = result.get('title', 'Untitled')
                snippet = result.get('snippet', '')
                url = result.get('url', '')

                sources.append({
                    'title': title,
                    'url': url,
                    'fetched': False,
                    'type': 'snippet'
                })

                content_parts.append(f"\n[{i}] {title}")
                content_parts.append(f"    {snippet[:200]}..." if len(snippet) > 200 else f"    {snippet}")
                content_parts.append(f"    <{url}>")

        final_content = "\n".join(content_parts)
        fetched_count = sum(1 for s in sources if s.get('fetched'))

        print(f"[WEB-SEARCH] Complete: {fetched_count} deep fetched, {len(remaining_results)} snippets")

        return {
            'content': final_content,
            'metadata': {
                'type': 'web_search',
                'query': query,
                'total_results': len(sources),
                'deep_fetched': fetched_count,
                'sources': sources,
                'error': None
            }
        }


# Global web search engine instance
_web_search_engine = None

def get_web_search_engine() -> WebSearchEngine:
    """Get or create web search engine instance."""
    global _web_search_engine
    if _web_search_engine is None:
        _web_search_engine = WebSearchEngine()
    return _web_search_engine


def web_search(query: str, max_results: int = 12, deep_fetch: int = 6) -> Dict:
    """
    Perform comprehensive web search.

    Args:
        query:       Search query string
        max_results: Maximum results to consider
        deep_fetch:  Number of pages to fetch full content from

    Returns:
        Dict with 'content' (str) and 'metadata' (dict)
    """
    engine = get_web_search_engine()
    return engine.search(query, max_results, deep_fetch)


def format_web_search_status_for_chat(search_metadata: dict) -> str:
    """Format web search metadata into a readable status string for the chat display."""
    if not search_metadata:
        return ""

    lines = []
    query = search_metadata.get('query', '')
    sources = search_metadata.get('sources', [])
    error = search_metadata.get('error')

    display_query = query[:80] + "..." if len(query) > 80 else query

    if error:
        lines.append(f"🌐 Web Search: \"{display_query}\" — ⚠️ {error}")
    else:
        deep_sources = [s for s in sources if s.get('type') == 'deep']
        snippet_sources = [s for s in sources if s.get('type') == 'snippet']
        fetched = sum(1 for s in deep_sources if s.get('fetched'))

        lines.append(f"🌐 Web Search: \"{display_query}\"")
        lines.append(f"   📰 {fetched}/{len(deep_sources)} articles fetched")
        if snippet_sources:
            lines.append(f"   📋 {len(snippet_sources)} additional snippets")

        for source in deep_sources:
            if source.get('fetched'):
                url = source.get('url', '')
                try:
                    domain = urlparse(url).netloc.replace('www.', '')
                    lines.append(f"      ✓ {domain}")
                except:
                    pass

    return "\n".join(lines)


def format_search_status_for_chat(search_metadata: dict) -> str:
    """Format search metadata for chat display. Delegates to the web search formatter."""
    return format_web_search_status_for_chat(search_metadata)


# =============================================================================
# TTS (TEXT-TO-SPEECH) FUNCTIONS
# =============================================================================

# TTS thread management
_tts_lock = threading.Lock()
_tts_thread = None
_tts_stop_flag = threading.Event()


def detect_tts_engine() -> str:
    """
    Detect available TTS engine. v2: Coqui TTS only.

    Returns:
        str: "coqui" if available, "none" if not installed or misconfigured.
    """
    if cfg.PLATFORM == "windows":
        base_dir = Path(__file__).parent.parent
        espeak_dll = base_dir / "data" / "espeak-ng" / "libespeak-ng.dll"
        espeak_exe = base_dir / "data" / "espeak-ng" / "espeak-ng.exe"

        if not espeak_dll.exists():
            print(f"[TTS] espeak-ng DLL not found: {espeak_dll}")
            print("[TTS] Re-run installer to repair Coqui installation.")
            return "none"

        espeak_dir = str(base_dir / "data" / "espeak-ng")
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeak_dll)
        os.environ["PHONEMIZER_ESPEAK_PATH"]    = str(espeak_exe)
        os.environ["ESPEAK_DATA_PATH"]           = str(base_dir / "data" / "espeak-ng" / "espeak-ng-data")
        os.environ["PATH"]                       = espeak_dir + os.pathsep + os.environ.get("PATH", "")

    try:
        from TTS.api import TTS  # noqa: F401
        return "coqui"
    except ImportError as e:
        print(f"[TTS] Coqui TTS import failed: {e}")
        print("[TTS] Re-run installer to repair Coqui installation.")
        return "none"


def detect_audio_backend() -> str:
    """
    Detect audio playback backend with functional testing.

    Returns:
        str: "windows", "pipewire", "pulseaudio", "alsa", or "none"
    """
    if cfg.PLATFORM == "windows":
        return "windows"

    # PipeWire — must be actually running, not just installed
    try:
        result = subprocess.run(["pw-cli", "info", "0"], capture_output=True, timeout=3)
        if result.returncode == 0:
            return "pipewire"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # PulseAudio (or PipeWire-Pulse compatibility layer)
    try:
        result = subprocess.run(["pactl", "info"], capture_output=True, timeout=3)
        if result.returncode == 0:
            if "PipeWire" in result.stdout.decode():
                return "pipewire"
            return "pulseaudio"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    # ALSA
    try:
        result = subprocess.run(["aplay", "--version"], capture_output=True, timeout=3)
        if result.returncode == 0:
            return "alsa"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass

    return "none"


def _get_coqui_voices() -> List[Dict[str, str]]:
    """Get available Coqui TTS voices filtered to the installed accent.

    The VCTK multi-speaker model contains ALL 109 speakers in a single file.
    No per-voice installation is needed — every speaker ID works from the
    same model download. We filter the dropdown to the accent the user
    chose during installation (stored in constants.ini as coqui_voice_accent).

    VCTK model speaker mapping (inverted vs corpus metadata — known bug):
      p229 = British Male      p243 = British Female
      p231 = American Male     p230 = American Female
    """
    accent_voices = {
        "english": [
            {"id": "p226", "name": "English (Male)",   "gender": "male"},
            {"id": "p225", "name": "English (Female)", "gender": "female"},
        ],
        "british": [
            {"id": "p229", "name": "British (Male)",   "gender": "male"},
            {"id": "p243", "name": "British (Female)", "gender": "female"},
        ],
        "american": [
            {"id": "p231", "name": "American (Male)",   "gender": "male"},
            {"id": "p230", "name": "American (Female)", "gender": "female"},
        ],
        "scottish":       [{"id": "p234", "name": "Scottish",        "gender": "male"}],
        "irish":          [{"id": "p245", "name": "Irish",           "gender": "male"}],
        "indian":         [{"id": "p248", "name": "Indian",          "gender": "male"}],
        "canadian":       [{"id": "p302", "name": "Canadian",        "gender": "male"}],
        "south_african":  [{"id": "p323", "name": "South African",   "gender": "male"}],
        "welsh":          [{"id": "p253", "name": "Welsh",           "gender": "male"}],
        "northern_irish": [{"id": "p292", "name": "Northern Irish",  "gender": "male"}],
        "australian":     [{"id": "p303", "name": "Australian",      "gender": "male"}],
        "new_zealand":    [{"id": "p316", "name": "New Zealand",     "gender": "male"}],
    }

    installed_accent = getattr(cfg, 'COQUI_VOICE_ACCENT', 'english')
    raw_voice_id = getattr(cfg, 'COQUI_VOICE_ID', 'p226')
    default_voice_id = raw_voice_id.split(',')[0].strip() if raw_voice_id else 'p226'

    accent_voice_list = accent_voices.get(installed_accent)
    if not accent_voice_list:
        return [{'id': default_voice_id, 'name': f"{installed_accent.title()} (Default)", 'language': 'en'}]

    voices = []
    for v in accent_voice_list:
        entry = {'id': v['id'], 'name': v['name'], 'language': 'en'}
        if v['id'] == default_voice_id:
            voices.insert(0, entry)
        else:
            voices.append(entry)

    if not any(v['id'] == default_voice_id for v in voices):
        voices.insert(0, {'id': default_voice_id, 'name': f"{installed_accent.title()} (Default)", 'language': 'en'})

    return voices


def get_voice_choices() -> List[str]:
    """Get voice names for UI dropdown."""
    voices = _get_coqui_voices()
    if not voices:
        return ["No voices available"]
    return [v['name'] for v in voices]


def get_voice_id_by_name(voice_name: str) -> Optional[str]:
    """Get voice ID from display name."""
    voices = _get_coqui_voices()
    if not voices:
        return None
    for voice in voices:
        if voice['name'] == voice_name:
            return voice['id']
    return None


def get_sample_rate_choices() -> List[int]:
    """Get available sample rate options."""
    return [44100, 48000]


def speak_text(text: str, voice_id: Optional[str] = None,
               output_device: Optional[str] = None,
               sample_rate: Optional[int] = None,
               blocking: bool = False) -> bool:
    """Speak text via TTS in a background thread."""
    global _tts_thread

    if not getattr(cfg, 'TTS_ENABLED', False):
        return False

    if detect_tts_engine() == "none":
        print("[TTS] No TTS engine available")
        return False

    # Apply shared sound settings as fallback for any unspecified parameters
    if not voice_id:
        voice_id = getattr(cfg, 'TTS_VOICE', None)
    if not output_device:
        output_device = getattr(cfg, 'SOUND_OUTPUT_DEVICE', None)
    if not sample_rate:
        sample_rate = getattr(cfg, 'SOUND_SAMPLE_RATE', 44100)

    _tts_thread = threading.Thread(
        target=_speak_thread,
        args=(text, voice_id, output_device, sample_rate),
        daemon=True
    )
    _tts_thread.start()

    if blocking:
        _tts_thread.join()

    return True


def _speak_thread(text: str, voice_id: Optional[str],
                  output_device: Optional[str], sample_rate: int):
    """Background thread for TTS. v2: Coqui only."""
    with _tts_lock:
        try:
            _speak_coqui(text, voice_id, output_device, sample_rate)
        except Exception as e:
            print(f"[TTS] Speech error: {e}")


def _speak_coqui(text: str, voice_id: Optional[str],
                 output_device: Optional[str], sample_rate: int):
    """Speak text using Coqui TTS (VCTK model).

    IMPORTANT: espeak-ng environment variables must be set BEFORE importing TTS
    on Windows, as the phonemizer library checks them at import time.
    Uses local project copy of espeak-ng, NOT Program Files.
    """
    try:
        base_dir = Path(__file__).parent.parent
        espeak_dir = base_dir / "data" / "espeak-ng"

        if cfg.PLATFORM == "windows":
            espeak_dll = espeak_dir / "libespeak-ng.dll"
            espeak_exe = espeak_dir / "espeak-ng.exe"
            espeak_data = espeak_dir / "espeak-ng-data"

            if not espeak_dll.exists():
                print(f"[TTS] ERROR: espeak-ng not found at {espeak_dir}")
                print("[TTS] Please run the installer again and select Coqui TTS")
                return

            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeak_dll)
            os.environ["PHONEMIZER_ESPEAK_PATH"] = str(espeak_exe)
            os.environ["ESPEAK_DATA_PATH"] = str(espeak_data)

        from TTS.api import TTS

        tts_model_dir = base_dir / "data" / "tts_models"
        os.environ["TTS_HOME"] = str(tts_model_dir)

        model_name = getattr(cfg, 'COQUI_MODEL', 'tts_models/en/vctk/vits')

        if not voice_id:
            voice_id = getattr(cfg, 'COQUI_VOICE_ID', 'p243')

        print(f"[TTS] Coqui synthesizing with voice {voice_id}...")

        tts = TTS(model_name=model_name, progress_bar=False)

        temp_dir = Path(cfg.TEMP_DIR) if cfg.TEMP_DIR else base_dir / "data" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_wav = temp_dir / f"coqui_speech_{os.getpid()}.wav"

        if _tts_stop_flag.is_set():
            return

        tts.tts_to_file(text=text, file_path=str(temp_wav), speaker=voice_id)

        if not temp_wav.exists():
            print("[TTS] Coqui failed to generate audio file")
            return

        if _tts_stop_flag.is_set():
            temp_wav.unlink(missing_ok=True)
            return

        _play_audio_file(str(temp_wav), output_device)

    except ImportError as e:
        print(f"[TTS] Coqui TTS not installed: {e}")
        print("[TTS] Install with: pip install coqui-tts")
    except Exception as e:
        print(f"[TTS] Coqui error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if 'temp_wav' in locals() and temp_wav.exists():
                temp_wav.unlink()
        except:
            pass


def _play_audio_file(file_path: str, output_device: Optional[str] = None):
    """Play an audio file using the best available backend."""
    if cfg.PLATFORM == "windows":
        try:
            import winsound
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
            return
        except:
            pass
        try:
            from playsound import playsound
            playsound(file_path)
            return
        except:
            pass
        print("[TTS] No Windows audio playback available")
        return

    # Linux — build environment for user audio session access
    env = os.environ.copy()
    original_uid = os.getuid() if hasattr(os, 'getuid') else None
    sudo_user = os.environ.get('SUDO_USER')

    if original_uid == 0 and sudo_user:
        try:
            import pwd
            user_info = pwd.getpwnam(sudo_user)
            user_uid = user_info.pw_uid
            user_home = user_info.pw_dir

            env['HOME'] = user_home
            env['USER'] = sudo_user

            runtime_dir = f"/run/user/{user_uid}"
            if Path(runtime_dir).exists():
                env['XDG_RUNTIME_DIR'] = runtime_dir

                pulse_path = f"{runtime_dir}/pulse"
                if Path(pulse_path).exists():
                    env['PULSE_RUNTIME_PATH'] = pulse_path

                pipewire_path = f"{runtime_dir}/pipewire-0"
                if Path(pipewire_path).exists():
                    env['PIPEWIRE_RUNTIME_DIR'] = runtime_dir
        except Exception as e:
            print(f"[TTS] Could not setup user audio env: {e}")

    played = False

    # Try PipeWire first
    if not played:
        try:
            subprocess.run(["pw-play", file_path], timeout=120, check=True,
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            played = True
            print("[TTS] Playback via PipeWire")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Try PulseAudio
    if not played:
        try:
            subprocess.run(["paplay", file_path], timeout=120, check=True,
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            played = True
            print("[TTS] Playback via PulseAudio")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    # Try ALSA
    if not played:
        try:
            subprocess.run(["aplay", "-q", file_path], timeout=120, check=True,
                           env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            played = True
            print("[TTS] Playback via ALSA")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass

    if not played:
        print("[TTS] ERROR: All audio backends failed")


def synthesize_text_to_file(text, voice_id=None):
    """Synthesize text to a WAV file without playing. Returns the file path or None."""
    if cfg.TTS_ENGINE == "none":
        print("[TTS] No TTS engine available")
        return None

    if not text or not text.strip():
        print("[TTS] Empty text, skipping")
        return None

    text = _clean_text_for_tts(text)
    if not text:
        print("[TTS] Text empty after cleaning")
        return None

    if len(text) > cfg.MAX_TTS_LENGTH:
        print(f"[TTS] Text truncated from {len(text)} to {cfg.MAX_TTS_LENGTH} chars")
        text = text[:cfg.MAX_TTS_LENGTH]

    if cfg.TTS_ENGINE == "coqui":
        return _synthesize_coqui_to_file(text, voice_id)

    print(f"[TTS] Unsupported engine: {cfg.TTS_ENGINE}")
    return None


def _synthesize_coqui_to_file(text, voice_id=None):
    """Synthesize text to WAV using Coqui TTS. Returns WAV path or None."""
    if not text or not text.strip():
        print("[TTS] Empty text, skipping")
        return None

    effective_voice_id = voice_id or cfg.COQUI_VOICE_ID or "p225"
    if "," in effective_voice_id:
        effective_voice_id = effective_voice_id.split(",")[0].strip()

    wav_path = os.path.join(cfg.TEMP_DIR, f"tts_msg_{int(time.time()*1000)}.wav")

    try:
        base_dir = Path(__file__).parent.parent

        if cfg.PLATFORM == "windows":
            espeak_dir = base_dir / "data" / "espeak-ng"
            espeak_dll = espeak_dir / "libespeak-ng.dll"
            espeak_exe = espeak_dir / "espeak-ng.exe"
            espeak_data = espeak_dir / "espeak-ng-data"

            if not espeak_dll.exists():
                print(f"[TTS] ERROR: espeak-ng not found at {espeak_dir}")
                return None

            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeak_dll)
            os.environ["PHONEMIZER_ESPEAK_PATH"] = str(espeak_exe)
            os.environ["ESPEAK_DATA_PATH"] = str(espeak_data)

        from TTS.api import TTS

        tts_model_dir = base_dir / "data" / "tts_models"
        os.environ["TTS_HOME"] = str(tts_model_dir)

        model_name = getattr(cfg, 'COQUI_MODEL', 'tts_models/en/vctk/vits')

        print(f"[TTS] Coqui synthesizing with voice {effective_voice_id}...")

        tts = TTS(model_name=model_name, progress_bar=False)

        if _tts_stop_flag.is_set():
            return None

        tts.tts_to_file(text=text, file_path=wav_path, speaker=effective_voice_id)

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            print("[TTS] Synthesis produced empty file")
            return None

        print(f"[TTS] Synthesized -> {wav_path}")
        return wav_path

    except ImportError as e:
        print(f"[TTS] Coqui TTS not installed: {e}")
        return None
    except Exception as e:
        print(f"[TTS] Synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return None


def synthesize_last_response(session_messages: list) -> Optional[str]:
    """Synthesize TTS audio from the last AI response (blocking).
    Returns the path to the generated WAV file, or None on failure.
    Does NOT play the audio — call play_tts_audio() separately.
    """
    if not getattr(cfg, 'TTS_ENABLED', False):
        return None

    if not session_messages:
        return None

    last_response = None
    for msg in reversed(session_messages):
        if msg.get('role') == 'assistant':
            last_response = msg.get('content', '')
            break
    if not last_response:
        return None

    text = _clean_text_for_tts(last_response)
    if not text:
        return None

    max_len = getattr(cfg, 'MAX_TTS_LENGTH', 4500)
    if len(text) > max_len:
        text = text[:max_len] + "... Response truncated for speech."

    voice_id = getattr(cfg, 'TTS_VOICE', None)

    try:
        base_dir = Path(__file__).parent.parent

        if cfg.PLATFORM == "windows":
            espeak_dir = base_dir / "data" / "espeak-ng"
            espeak_dll = espeak_dir / "libespeak-ng.dll"
            espeak_exe = espeak_dir / "espeak-ng.exe"
            espeak_data = espeak_dir / "espeak-ng-data"

            if not espeak_dll.exists():
                print(f"[TTS] ERROR: espeak-ng not found at {espeak_dir}")
                return None

            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeak_dll)
            os.environ["PHONEMIZER_ESPEAK_PATH"] = str(espeak_exe)
            os.environ["ESPEAK_DATA_PATH"] = str(espeak_data)

        from TTS.api import TTS

        tts_model_dir = base_dir / "data" / "tts_models"
        os.environ["TTS_HOME"] = str(tts_model_dir)

        model_name = getattr(cfg, 'COQUI_MODEL', 'tts_models/en/vctk/vits')

        if not voice_id:
            voice_id = getattr(cfg, 'COQUI_VOICE_ID', 'p225,p226')

        print(f"[TTS] Coqui synthesizing with voice {voice_id}...")

        tts = TTS(model_name=model_name, progress_bar=False)

        temp_dir = Path(cfg.TEMP_DIR) if cfg.TEMP_DIR else base_dir / "data" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_wav = temp_dir / f"coqui_speech_{os.getpid()}.wav"

        if _tts_stop_flag.is_set():
            return None

        tts.tts_to_file(text=text, file_path=str(temp_wav), speaker=voice_id)

        if not temp_wav.exists():
            print("[TTS] Coqui failed to generate audio file")
            return None

        print(f"[TTS] Synthesis complete: {temp_wav}")
        return str(temp_wav)

    except ImportError as e:
        print(f"[TTS] Coqui TTS not installed: {e}")
        return None
    except Exception as e:
        print(f"[TTS] Synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return None


def play_tts_audio(wav_path: str, output_device: Optional[str] = None):
    """Play a synthesized TTS audio file (blocking) and clean up.

    Args:
        wav_path:      Path to the WAV file to play
        output_device: Audio output device (None = system default)
    """
    if not wav_path or wav_path == "__played__":
        return

    try:
        if not Path(wav_path).exists():
            print(f"[TTS] Audio file not found: {wav_path}")
            return

        if _tts_stop_flag.is_set():
            return

        if output_device is None:
            output_device = getattr(cfg, 'SOUND_OUTPUT_DEVICE', 'Default Sound Device')
        if output_device == "Default Sound Device":
            output_device = "default"

        print(f"[TTS] Playing audio: {wav_path}")
        _play_audio_file(wav_path, output_device)
        print("[TTS] Playback complete")

    except Exception as e:
        print(f"[TTS] Playback error: {e}")
    finally:
        try:
            if Path(wav_path).exists():
                Path(wav_path).unlink()
        except:
            pass


def stop_speaking():
    """Stop any ongoing TTS playback."""
    global _tts_thread
    _tts_stop_flag.set()
    if _tts_thread and _tts_thread.is_alive():
        _tts_thread.join(timeout=1.0)


def clear_tts_stop():
    """Clear the TTS stop flag so a new playback can start."""
    _tts_stop_flag.clear()


def is_speaking() -> bool:
    """Check if TTS is currently active."""
    global _tts_thread
    return _tts_thread is not None and _tts_thread.is_alive()


def initialize_tts():
    """Initialize TTS system. Called during startup AFTER load_config."""
    engine = detect_tts_engine()
    backend = detect_audio_backend()

    cfg.TTS_ENGINE = engine
    cfg.TTS_AUDIO_BACKEND = backend

    if engine == "coqui":
        coqui_voice = getattr(cfg, 'COQUI_VOICE_ID', 'p243')
        coqui_accent = getattr(cfg, 'COQUI_VOICE_ACCENT', 'british')
        print(f"[TTS] Engine: Coqui TTS (voice: {coqui_voice}, accent: {coqui_accent})")
    else:
        print(f"[TTS] Engine: {engine}")
    print(f"[TTS] Audio Backend: {backend}")

    voices = _get_coqui_voices()
    voice_ids = [v['id'] for v in voices]

    if voices:
        saved_id   = getattr(cfg, 'TTS_VOICE', None)
        saved_name = getattr(cfg, 'TTS_VOICE_NAME', None)
        voice_names = [v['name'] for v in voices]

        if saved_id and saved_id in voice_ids:
            for v in voices:
                if v['id'] == saved_id:
                    cfg.TTS_VOICE = v['id']
                    cfg.TTS_VOICE_NAME = v['name']
                    break
            print(f"[TTS] Voice from config: {cfg.TTS_VOICE_NAME} ({cfg.TTS_VOICE})")
        elif saved_name and saved_name in voice_names:
            for v in voices:
                if v['name'] == saved_name:
                    cfg.TTS_VOICE = v['id']
                    cfg.TTS_VOICE_NAME = v['name']
                    break
            print(f"[TTS] Voice from config: {cfg.TTS_VOICE_NAME} ({cfg.TTS_VOICE})")
        else:
            cfg.TTS_VOICE = voices[0]['id']
            cfg.TTS_VOICE_NAME = voices[0]['name']
            print(f"[TTS] Default voice: {voices[0]['name']}")
    else:
        cfg.TTS_VOICE = None
        cfg.TTS_VOICE_NAME = "No voices available"
        print("[TTS] No voices detected")

    return engine != "none"


def get_tts_status() -> str:
    """Get TTS status string for display."""
    enabled = getattr(cfg, 'TTS_ENABLED', False)
    if enabled:
        voice = getattr(cfg, 'TTS_VOICE_NAME', 'Default')
        return f"TTS: ON (Coqui - {voice})"
    return "TTS: OFF (Coqui)"


def _clean_text_for_tts(text: str) -> str:
    """Shared text cleaning pipeline applied before any TTS synthesis.

    Strips all markdown formatting, HTML, code blocks, and symbols that a
    speech engine would either mispronounce or vocalise as literal character
    names (e.g. "asterisk", "hash", "underscore").

    NOTE: asterisks are intentionally stripped here — the session log retains
    the original markdown so bullet-point rendering in the chat is unaffected.
    """
    text = re.sub(r'^AI-Chat:\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'^Thinking[.\s]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'!\[.*?\]\([^)]+\)', '', text)
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'\*',   '', text)
    text = re.sub(r'~~',   '', text)
    text = re.sub(r'(?<!\w)_|_(?!\w)', '', text)
    text = re.sub(r'[#•→⇒★☆]|[-=]{2,}', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\'"()-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def speak_last_response(session_messages: list) -> str:
    """Speak the last AI response from session messages.

    Args:
        session_messages: List of message dicts

    Returns:
        Status message string
    """
    if not getattr(cfg, 'TTS_ENABLED', False):
        return "TTS is disabled"

    if not session_messages:
        return "No messages to speak"

    last_response = None
    for msg in reversed(session_messages):
        if msg.get('role') == 'assistant':
            last_response = msg.get('content', '')
            break

    if not last_response:
        return "No AI response to speak"

    text = _clean_text_for_tts(last_response)

    if not text:
        return "Response has no speakable content after cleaning"

    max_len = getattr(cfg, 'MAX_TTS_LENGTH', 4500)
    if len(text) > max_len:
        text = text[:max_len] + "... Response truncated for speech."

    voice_id      = getattr(cfg, 'TTS_VOICE', None)
    output_device = getattr(cfg, 'SOUND_OUTPUT_DEVICE', None)
    sample_rate   = getattr(cfg, 'SOUND_SAMPLE_RATE', 44100)

    if speak_text(text, voice_id, output_device, sample_rate):
        return "Speaking response..."
    return "Failed to start speech"