# scripts/tools.py
"""
Centralized tools module for web search, system sounds, and TTS.

Search Tools:
- DDG Hybrid Search: DDG snippets + full article fetching via newspaper
- Web Search: Comprehensive multi-source web search with parallel page fetching

TTS Tools:
- Text-to-Speech using pyttsx3 (Windows) or espeak-ng (Linux)
- Audio playback via built-in (Windows), paplay (PulseAudio), or pw-play (PipeWire)
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
import scripts.configuration as cfg

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
                # Use news search
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
            
            # Always also do text search
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
                
                # Add publish date if available
                if article.publish_date:
                    content = f"[Published: {article.publish_date.strftime('%Y-%m-%d')}]\n{content}"
                
                return content
                
        except Exception as e:
            print(f"[WEB-SEARCH] Failed to fetch {url}: {e}")
        
        return None
    
    def _fetch_pages_parallel(self, urls: List[str], max_workers: int = 4) -> Dict[str, str]:
        """Fetch multiple pages in parallel."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = {}
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {executor.submit(self._fetch_page_content, url): url for url in urls}
            
            for future in as_completed(future_to_url, timeout=30):
                url = future_to_url[future]
                try:
                    content = future.result()
                    if content:
                        results[url] = content
                except Exception as e:
                    print(f"[WEB-SEARCH] Parallel fetch error for {url}: {e}")
        
        return results
    
    def search(self, query: str, max_results: int = 12, deep_fetch: int = 6) -> Dict:
        """
        Perform comprehensive web search with dependency checking.
        """
        # Check dependencies
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
            missing_deps.append("newspaper3k")
        try:
            import lxml
        except ImportError:
            missing_deps.append("lxml")
            
        if missing_deps:
            error_msg = f"Web Search requires missing packages: {', '.join(missing_deps)}. Install with: pip install {' '.join(missing_deps)}"
            print(f"[WEB-SEARCH] {error_msg}")
            return {
                'content': f"Web search unavailable: {error_msg}",
                'metadata': {'type': 'web_search', 'query': query, 'error': error_msg, 'sources': []}
            }
        
        # Phase 1: Gather search results from multiple sources
        print(f"[WEB-SEARCH] Searching for: {query}")
        
        all_results = []
        
        # DDG HTML search
        html_results = self._search_duckduckgo_html(query, max_results)
        all_results.extend(html_results)
        
        # DDGS API search (supplement)
        api_results = self._search_ddgs_api(query, max_results // 2)
        all_results.extend(api_results)
        
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
        
        # Select top results for deep fetch
        top_results = [r for _, r in scored_results[:deep_fetch]]
        remaining_results = [r for _, r in scored_results[deep_fetch:max_results]]
        
        # Phase 3: Fetch full content from top results
        urls_to_fetch = [r['url'] for r in top_results if r.get('url')]
        fetched_content = self._fetch_pages_parallel(urls_to_fetch)
        
        # Phase 4: Build final content
        content_parts = []
        sources = []
        
        # Header
        content_parts.append(f"═══════════════════════════════════════════════════════")
        content_parts.append(f"WEB SEARCH RESULTS: {query}")
        content_parts.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        content_parts.append(f"═══════════════════════════════════════════════════════\n")
        
        # Deep-fetched articles
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
        
        # Additional snippets
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
        query: Search query
        max_results: Maximum results to consider
        deep_fetch: Number of pages to fetch full content
        
    Returns:
        Dict with 'content' and 'metadata'
    """
    engine = get_web_search_engine()
    return engine.search(query, max_results, deep_fetch)


def format_web_search_status_for_chat(search_metadata: dict) -> str:
    """Format web search metadata for chat display."""
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


# =============================================================================
# HYBRID SEARCH (DDG + NEWSPAPER)
# =============================================================================

def hybrid_search(search_query: str, ddg_results: int = 8, deep_fetch: int = 4) -> Dict:
    """
    Hybrid search: DDG snippets + newspaper deep fetch.
    Checks for required dependencies before executing.
    """
    # Check dependencies first
    missing_deps = []
    try:
        from newspaper import Article
    except ImportError:
        missing_deps.append("newspaper3k")
    try:
        from ddgs import DDGS
    except ImportError:
        missing_deps.append("duckduckgo-search")
    
    if missing_deps:
        error_msg = f"Search requires missing packages: {', '.join(missing_deps)}. Install with: pip install {' '.join(missing_deps)}"
        print(f"[HYBRID] {error_msg}")
        return {
            'content': f"Search unavailable: {error_msg}",
            'metadata': {'type': 'hybrid', 'query': search_query, 'error': error_msg, 'sources': []}
        }
    
    # Header
    header = f"""═══════════════════════════════════════════════════════
HYBRID SEARCH RESULTS: {search_query}
Search Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
═══════════════════════════════════════════════════════

"""
    
    # Phase 1: DDG search
    print(f"[HYBRID] Phase 1: DDG search for '{search_query}'")
    ddg_hits = []
    
    try:
        ddgs = DDGS(timeout=15)
        
        # Check if news query
        news_keywords = ['news', 'latest', 'current', 'recent', 'today', 'breaking']
        is_news = any(kw in search_query.lower() for kw in news_keywords)
        
        if is_news:
            try:
                news_results = list(ddgs.news(search_query, max_results=ddg_results))
                for r in news_results:
                    ddg_hits.append({
                        'title': r.get('title', ''),
                        'href': r.get('url', ''),
                        'body': r.get('body', ''),
                        'date': r.get('date', ''),
                        'source': r.get('source', '')
                    })
            except:
                pass
        
        # Text search
        text_results = list(ddgs.text(search_query, max_results=ddg_results))
        for r in text_results:
            ddg_hits.append({
                'title': r.get('title', ''),
                'href': r.get('href', ''),
                'body': r.get('body', ''),
                'date': '',
                'source': ''
            })
        
        print(f"[HYBRID]   Found {len(ddg_hits)} DDG results")
        
    except Exception as e:
        print(f"[HYBRID]   DDG error: {e}")
        return {
            'content': f"Search failed: DuckDuckGo search error ({str(e)}). Check your internet connection.",
            'metadata': {'type': 'hybrid', 'query': search_query, 'error': str(e), 'sources': []}
        }
    
    if not ddg_hits:
        return {
            'content': f"No search results found for: {search_query}\n\nThe search engine returned no results. The query may be too specific or there may be connectivity issues.",
            'metadata': {'type': 'hybrid', 'query': search_query, 'error': 'No results', 'sources': []}
        }
    
    # Phase 2: Score and rank
    print(f"[HYBRID] Phase 2: Ranking {len(ddg_hits)} results")
    
    scored_hits = []
    query_words = set(search_query.lower().split())
    
    for hit in ddg_hits:
        score = 0
        title_lower = hit['title'].lower()
        body_lower = hit['body'].lower()
        
        for word in query_words:
            if len(word) > 3:
                if word in title_lower:
                    score += 3
                if word in body_lower:
                    score += 1
        
        if hit.get('date'):
            score += 2
        
        url = hit.get('href', '').lower()
        reputable = ['reuters', 'bbc', 'npr', 'guardian', 'nytimes', 'washingtonpost', 
                     'aljazeera', 'apnews', 'wikipedia', '.gov', '.edu', 'britannica']
        if any(r in url for r in reputable):
            score += 3
        
        scored_hits.append((score, hit))
    
    scored_hits.sort(key=lambda x: x[0], reverse=True)
    
    deep_fetch_candidates = [hit for score, hit in scored_hits[:deep_fetch]]
    remaining_ddg = [hit for score, hit in scored_hits[deep_fetch:]]
    
    # Phase 3: Deep fetch
    print(f"[HYBRID] Phase 3: Deep fetching {len(deep_fetch_candidates)} articles")
    
    deep_results = []
    deep_sources = []
    
    for i, hit in enumerate(deep_fetch_candidates, 1):
        url = hit.get('href', '')
        title = hit.get('title', 'Untitled')
        snippet = hit.get('body', '')
        date = hit.get('date', '')
        source = hit.get('source', '')
        
        source_info = {'title': title, 'url': url, 'fetched': False, 'type': 'deep'}
        article_content = ""
        
        try:
            article = Article(url)
            article.download()
            article.parse()
            
            if article.text and len(article.text) > 100:
                article_content = article.text[:3000]
                if len(article.text) > 3000:
                    article_content += "\n[...truncated...]"
                if article.publish_date:
                    article_content = f"[Published: {article.publish_date.strftime('%Y-%m-%d')}]\n{article_content}"
                elif date:
                    article_content = f"[Date: {date}]\n{article_content}"
                source_info['fetched'] = True
            else:
                article_content = snippet
        except Exception as e:
            print(f"[HYBRID] Failed to fetch {url}: {e}")
            article_content = snippet
        
        deep_sources.append(source_info)
        
        deep_results.append(f"══════════════════════════════════════════════════════")
        deep_results.append(f"📰 DEEP ARTICLE {i}: {title}")
        deep_results.append(f"══════════════════════════════════════════════════════")
        deep_results.append(f"URL: {url}")
        if source:
            deep_results.append(f"Source: {source}")
        deep_results.append(f"\n{article_content or snippet}\n")
    
    # Phase 4: Merge
    ddg_sources = []
    ddg_summaries = []
    
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
    
    all_sources = deep_sources + ddg_sources
    fetched_count = sum(1 for s in all_sources if s.get('fetched', False))
    
    final_content = header
    final_content += "\n".join(deep_results)
    final_content += "\n".join(ddg_summaries)
    
    print(f"[HYBRID] Complete: {fetched_count} deep fetched, {len(ddg_sources)} DDG snippets")
    
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


# =============================================================================
# TTS (TEXT-TO-SPEECH) FUNCTIONS
# =============================================================================

# TTS thread management
_tts_lock = threading.Lock()
_tts_thread = None
_tts_stop_flag = threading.Event()


def detect_tts_engine() -> str:
    """
    Detect available TTS engine based on platform and configuration.
    
    Returns:
        str: "coqui", "pyttsx3" for Windows, "espeak-ng" for Linux, "none" if unavailable
    """
    import os
    from pathlib import Path
    
    # Check if Coqui TTS is configured
    tts_type = getattr(cfg, 'TTS_TYPE', 'builtin')
    
    if tts_type == "coqui":
        # On Windows, verify espeak-ng is installed in project folder (required for phonemizer)
        if cfg.PLATFORM == "windows":
            base_dir = Path(__file__).parent.parent
            espeak_dll = base_dir / "data" / "espeak-ng" / "libespeak-ng.dll"
            espeak_exe = base_dir / "data" / "espeak-ng" / "espeak-ng.exe"
            
            if not espeak_dll.exists():
                print(f"[TTS] Coqui configured but espeak-ng not found at {espeak_dll}")
                print("[TTS] Falling back to builtin would occur here, but failing as requested.")
                # Since you want no fallback and guaranteed install, this should not happen
                # if installer worked correctly. We return "none" to trigger failure later.
                return "none"
            else:
                # Set environment variables before trying to import TTS
                espeak_dir = str(base_dir / "data" / "espeak-ng")
                os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeak_dll)
                os.environ["PHONEMIZER_ESPEAK_PATH"] = str(espeak_exe)
                os.environ["ESPEAK_DATA_PATH"] = str(base_dir / "data" / "espeak-ng" / "espeak-ng-data")
                
                # Now verify Coqui TTS is actually available
                try:
                    from TTS.api import TTS
                    return "coqui"
                except ImportError as e:
                    print(f"[TTS] Coqui configured but import failed: {e}")
                    return "none"
        else:
            # Linux - just verify TTS is importable
            try:
                from TTS.api import TTS
                return "coqui"
            except ImportError:
                print("[TTS] Coqui configured but not installed")
                return "none"
    
    # Built-in TTS detection
    if cfg.PLATFORM == "windows":
        try:
            import pyttsx3
            return "pyttsx3"
        except ImportError:
            return "none"
    
    elif cfg.PLATFORM == "linux":
        try:
            result = subprocess.run(
                ["espeak-ng", "--version"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                return "espeak-ng"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
        return "none"
    
    return "none"

def detect_audio_backend() -> str:
    """
    Detect audio backend for playback with functional testing.
    
    Returns:
        str: "pipewire", "pulseaudio", "alsa", or "none"
    """
    
    if cfg.PLATFORM == "windows":
        return "windows"
    
    # Check PipeWire - must be actually running, not just installed
    try:
        # Test if pipewire daemon is accessible
        result = subprocess.run(
            ["pw-cli", "info", "0"], 
            capture_output=True, 
            timeout=3
        )
        if result.returncode == 0:
            return "pipewire"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Check PulseAudio (or PipeWire-Pulse compatibility)
    try:
        result = subprocess.run(
            ["pactl", "info"], 
            capture_output=True, 
            timeout=3
        )
        if result.returncode == 0:
            # Check if it's actually PipeWire pretending to be Pulse
            output = result.stdout.decode()
            if "PipeWire" in output:
                return "pipewire"
            return "pulseaudio"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    # Check ALSA
    try:
        result = subprocess.run(
            ["aplay", "--version"], 
            capture_output=True, 
            timeout=3
        )
        if result.returncode == 0:
            return "alsa"
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    
    return "none"


def get_tts_voices() -> List[Dict[str, str]]:
    """
    Get available TTS voices based on configured engine.
    
    Returns:
        List of dicts with 'id', 'name', 'language' keys
    """
    
    engine = getattr(cfg, 'TTS_ENGINE', None) or detect_tts_engine()
    
    # Coqui TTS voices
    if engine == "coqui":
        return _get_coqui_voices()
    
    # pyttsx3 voices (Windows)
    elif engine == "pyttsx3":
        try:
            import pyttsx3
            tts = pyttsx3.init()
            voices = tts.getProperty('voices')
            voice_list = []
            for v in voices:
                # Safely extract attributes
                lang = "unknown"
                if hasattr(v, 'languages') and v.languages:
                    try:
                        lang = v.languages[0] if isinstance(v.languages, (list, tuple)) else str(v.languages)
                    except (IndexError, TypeError):
                        lang = "unknown"
                
                name = "Unknown Voice"
                if hasattr(v, 'name') and v.name:
                    name = str(v.name).strip()
                    if "Microsoft" in name and "Desktop" in name:
                        name = name.replace("Desktop", "").strip()
                
                voice_id = name
                if hasattr(v, 'id') and v.id:
                    voice_id = str(v.id)
                
                display = f"{name} ({lang})"
                voice_list.append({
                    'id': voice_id,
                    'name': display,
                    'language': lang
                })
            
            tts.stop()
            return voice_list
            
        except Exception as e:
            print(f"[TTS] pyttsx3 voice enumeration failed: {e}")
            import traceback
            traceback.print_exc()
            
    elif engine == "espeak-ng":
        return _get_espeak_voices()
    
    return []

def _get_coqui_voices() -> List[Dict[str, str]]:
    """Get available Coqui TTS voices filtered to the installed accent.
    
    The VCTK multi-speaker model contains ALL 109 speakers in a single file.
    No per-voice installation is needed - every speaker ID works from the
    same model download. We filter the dropdown to the accent the user
    chose during installation (stored in constants.ini as coqui_voice_accent).
    
    For accents with male/female variants (British, American), both are shown.
    For single-speaker accents, only that speaker is shown.
    
    VCTK model speaker mapping (inverted vs corpus metadata - known bug):
      p229 = British Male      p243 = British Female
      p231 = American Male     p230 = American Female
    """
    
    # Full voice map keyed by accent
    # Each accent lists its available speakers with gender info
    accent_voices = {
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
    
    installed_accent = getattr(cfg, 'COQUI_VOICE_ACCENT', 'british')
    default_voice_id = getattr(cfg, 'COQUI_VOICE_ID', 'p243')
    
    accent_voice_list = accent_voices.get(installed_accent)
    if not accent_voice_list:
        # Unknown accent - show just the configured voice ID
        return [{'id': default_voice_id, 'name': f"{installed_accent.title()} (Default)", 'language': 'en'}]
    
    voices = []
    for v in accent_voice_list:
        entry = {'id': v['id'], 'name': v['name'], 'language': 'en'}
        # Put the installer-selected default voice first
        if v['id'] == default_voice_id:
            voices.insert(0, entry)
        else:
            voices.append(entry)
    
    # Safety: if default_voice_id wasn't in the accent list, add it
    if not any(v['id'] == default_voice_id for v in voices):
        voices.insert(0, {'id': default_voice_id, 'name': f"{installed_accent.title()} (Default)", 'language': 'en'})
    
    return voices

def _get_pyttsx3_voices() -> List[Dict[str, str]]:
    """Get voices from pyttsx3 (Windows SAPI)."""
    voices = []
    try:
        import pyttsx3
        engine = pyttsx3.init()
        for voice in engine.getProperty('voices'):
            # Extract language properly
            lang = voice.languages[0] if hasattr(voice, 'languages') and voice.languages else "en_US"
            name = voice.name.strip()
            # Clean up Microsoft Desktop voices for better display
            if "Microsoft" in name and "Desktop" in name:
                name = name.replace("Desktop", "").strip()
            display = f"{name} ({lang})"
            voices.append({
                'id': voice.id,
                'name': display,
                'language': lang
            })
        engine.stop()
    except Exception as e:
        print(f"[TTS] Error getting pyttsx3 voices: {e}")
    return voices


def _get_espeak_voices() -> List[Dict[str, str]]:
    """Get voices from espeak-ng (Linux) - only returns actually installed voices."""
    import os
    
    voices = []
    
    # Locate espeak-ng voices directory (varies by distro/install method)
    voice_search_paths = [
        "/usr/lib/x86_64-linux-gnu/espeak-ng-data/voices",  # Ubuntu 22.04+
        "/usr/lib/espeak-ng-data/voices",                   # Older Debian/Ubuntu
        "/usr/share/espeak-ng-data/voices",                 # Some distros
        "/usr/local/lib/espeak-ng-data/voices",             # Manual install
        "/opt/local/share/espeak-ng-data/voices",           # MacPorts
    ]
    
    voices_path = None
    for path in voice_search_paths:
        if os.path.isdir(path):
            voices_path = path
            break
    
    try:
        result = subprocess.run(
            ["espeak-ng", "--voices"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            
            for line in lines[1:]:  # Skip header
                if not line.strip():
                    continue
                
                # Parse space-delimited columns
                # Format: Pty Language Age/Gender VoiceName File Other...
                parts = [p.strip() for p in line.split() if p.strip()]
                
                if len(parts) >= 4:
                    lang_code = parts[1]      # e.g., "en", "en-gb", "en-westmidlands"
                    voice_name = parts[3]     # e.g., "default", "british", "west_midlands"
                    # priority = parts[0]     # Priority number
                    # age_gender = parts[2]   # M/F/-
                    # voice_file = parts[4] if len(parts) > 4 else lang_code
                    
                    # Validate voice actually exists on filesystem
                    voice_exists = False
                    if voices_path:
                        # Check direct file: /voices/en-gb
                        direct_path = os.path.join(voices_path, lang_code)
                        if os.path.exists(direct_path):
                            voice_exists = True
                        else:
                            # Check subdirectory variant: /voices/en/westmidlands
                            if '-' in lang_code:
                                base_lang = lang_code.split('-')[0]
                                variant = lang_code[len(base_lang)+1:]
                                variant_path = os.path.join(voices_path, base_lang, variant)
                                if os.path.exists(variant_path):
                                    voice_exists = True
                        
                        # Also check if it's a known special case (like 'en' is always valid)
                        if not voice_exists and lang_code in ('en', 'en-gb', 'en-us', 'default'):
                            voice_exists = True
                    else:
                        # Can't verify, assume it exists
                        voice_exists = True
                    
                    if voice_exists:
                        # Format display name nicely
                        if voice_name.lower() == 'default':
                            display_name = f"Default {lang_code.upper()}"
                        else:
                            formatted = voice_name.replace('_', ' ').title()
                            display_name = f"{formatted} ({lang_code})"
                        
                        voices.append({
                            'id': lang_code,        # Use language code for -v parameter
                            'name': display_name,
                            'language': lang_code
                        })
                        
    except Exception as e:
        print(f"[TTS] Error getting espeak-ng voices: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback if no voices found or error occurred
    if not voices:
        voices.append({
            'id': 'en', 
            'name': 'English (default)', 
            'language': 'en'
        })
        voices.append({
            'id': 'en-gb', 
            'name': 'British English (en-gb)', 
            'language': 'en-gb'
        })
    
    return voices


def get_voice_choices() -> List[str]:
    """Get voice names for UI dropdown."""
    voices = get_tts_voices()
    if not voices:
        return ["No voices available"]
    return [v['name'] for v in voices]


def get_voice_id_by_name(voice_name: str) -> Optional[str]:
    """Get voice ID from display name."""
    voices = get_tts_voices()
    for voice in voices:
        if voice['name'] == voice_name:
            return voice['id']
    return None


def get_output_devices():
    """Get available audio output devices.
    
    Returns only the system default device for both Windows and Linux.
    This ensures compatibility with USB audio devices when set as system default.
    """
    
    if cfg.PLATFORM == "windows":
        return [{'id': 'default', 'name': 'Default Sound Device'}]
    
    # Linux - return only default, detection done at runtime by backend
    return [{'id': 'default', 'name': 'Default Sound Device'}]

def get_output_device_choices():
    """Get output device choices as (display_name, device_id) tuples for UI dropdown."""
    devices = get_output_devices()
    # Return list of tuples: (display_name, device_id)
    # This allows Gradio to show the description but return the technical ID
    return [(d['name'], d['id']) for d in devices]


def get_sample_rate_choices() -> List[int]:
    """Get available sample rate options."""
    return [44100, 48000]


def speak_text(text: str, voice_id: Optional[str] = None,
               output_device: Optional[str] = None,
               sample_rate: Optional[int] = None,
               blocking: bool = False) -> bool:
    """
    Speak text using TTS.
    
    Args:
        blocking: If True, wait for speech to complete before returning.
                  Used by non-coqui engines in synthesize_last_response.
    Uses shared sound settings if output_device/sample_rate not provided.
    """
    global _tts_thread
    

    if not getattr(cfg, 'TTS_ENABLED', False):
        return False

    engine = getattr(cfg, 'TTS_ENGINE', None) or detect_tts_engine()
    if engine == "none":
        print("[TTS] No TTS engine available")
        return False

    # Use shared sound settings as fallback
    if output_device is None:
        output_device = cfg.SOUND_OUTPUT_DEVICE
    if sample_rate is None:
        sample_rate = cfg.SOUND_SAMPLE_RATE

    # Normalize: "Default Sound Device" means use system default (None for backend)
    if output_device == "Default Sound Device":
        output_device = "default"

    # Stop any ongoing speech
    stop_speaking()

    # Start in background
    _tts_stop_flag.clear()
    _tts_thread = threading.Thread(
        target=_speak_thread,
        args=(text, engine, voice_id, output_device, sample_rate),
        daemon=True
    )
    _tts_thread.start()

    # If blocking mode, wait for speech to complete
    if blocking:
        _tts_thread.join()

    return True


def _speak_thread(text: str, engine: str, voice_id: Optional[str],
                  output_device: Optional[str], sample_rate: int):
    """Background thread for TTS."""
    with _tts_lock:
        try:
            if engine == "coqui":
                _speak_coqui(text, voice_id, output_device, sample_rate)
            elif engine == "pyttsx3":
                _speak_pyttsx3(text, voice_id)
            elif engine == "espeak-ng":
                _speak_espeak(text, voice_id, output_device, sample_rate)
        except Exception as e:
            print(f"[TTS] Speech error: {e}")

def _speak_coqui(text: str, voice_id: Optional[str],
                 output_device: Optional[str], sample_rate: int):
    """Speak text using Coqui TTS (VCTK model).
    
    IMPORTANT: espeak-ng environment variables must be set BEFORE importing TTS
    on Windows, as the phonemizer library checks them at import time.
    Uses local project copy of espeak-ng, NOT Program Files.
    """
    import os
    from pathlib import Path
    
    try:
        # CRITICAL: Set espeak-ng environment variables BEFORE importing TTS
        # The phonemizer library (used by Coqui) checks these at import time
        # Use local project folder, NOT Program Files
        base_dir = Path(__file__).parent.parent
        espeak_dir = base_dir / "data" / "espeak-ng"
        
        if cfg.PLATFORM == "windows":
            espeak_dll = espeak_dir / "libespeak-ng.dll"
            espeak_exe = espeak_dir / "espeak-ng.exe"
            espeak_data = espeak_dir / "espeak-ng-data"
            
            # Verify local espeak-ng is present
            if not espeak_dll.exists():
                print(f"[TTS] ERROR: espeak-ng not found at {espeak_dir}")
                print("[TTS] Please run the installer again and select Coqui TTS")
                return
            
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(espeak_dll)
            os.environ["PHONEMIZER_ESPEAK_PATH"] = str(espeak_exe)
            os.environ["ESPEAK_DATA_PATH"] = str(espeak_data)
        
        # Now it's safe to import TTS
        from TTS.api import TTS
        
        # Set TTS home to our model directory
        tts_model_dir = base_dir / "data" / "tts_models"
        os.environ["TTS_HOME"] = str(tts_model_dir)
        
        # Get model name from config
        model_name = getattr(cfg, 'COQUI_MODEL', 'tts_models/en/vctk/vits')
        
        # Use configured voice or default
        if not voice_id:
            voice_id = getattr(cfg, 'COQUI_VOICE_ID', 'p243')
        
        print(f"[TTS] Coqui synthesizing with voice {voice_id}...")
        
        # Initialize TTS (uses cached model)
        tts = TTS(model_name=model_name, progress_bar=False)
        
        # Create temp file for audio - ensure temp dir exists
        temp_dir = Path(cfg.TEMP_DIR) if cfg.TEMP_DIR else base_dir / "data" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_wav = temp_dir / f"coqui_speech_{os.getpid()}.wav"
        
        # Check for stop
        if _tts_stop_flag.is_set():
            return
        
        # Synthesize speech
        tts.tts_to_file(
            text=text,
            file_path=str(temp_wav),
            speaker=voice_id
        )
        
        if not temp_wav.exists():
            print("[TTS] Coqui failed to generate audio file")
            return
        
        # Check for stop before playback
        if _tts_stop_flag.is_set():
            temp_wav.unlink(missing_ok=True)
            return
        
        # Play the audio file
        _play_audio_file(str(temp_wav), output_device)
        
    except ImportError as e:
        print(f"[TTS] Coqui TTS not installed: {e}")
        print("[TTS] Install with: pip install coqui-tts")
    except Exception as e:
        print(f"[TTS] Coqui error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up temp file
        try:
            if 'temp_wav' in locals() and temp_wav.exists():
                temp_wav.unlink()
        except:
            pass


def _speak_pyttsx3(text: str, voice_id: Optional[str]):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        
        if voice_id:
            engine.setProperty('voice', voice_id)
        
        engine.setProperty('rate', 175)
        
        
        if cfg.PLATFORM == "windows" and cfg.SOUND_OUTPUT_DEVICE != 'default':
            print("[TTS] Note: On Windows, pyttsx3 ALWAYS uses the system default audio device — selected output ignored.")
        
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[TTS] pyttsx3 error: {e}")

def _play_audio_file(file_path: str, output_device: Optional[str] = None):
    """Play an audio file using the appropriate backend."""
    
    if cfg.PLATFORM == "windows":
        # Windows playback using winsound or playsound
        try:
            import winsound
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
            return
        except:
            pass
        
        # Fallback to playsound
        try:
            from playsound import playsound
            playsound(file_path)
            return
        except:
            pass
        
        print("[TTS] No Windows audio playback available")
        return
    
    # Linux playback
    env = os.environ.copy()
    
    # Try to get the original user's environment for audio access
    original_uid = os.getuid() if hasattr(os, 'getuid') else None
    sudo_user = os.environ.get('SUDO_USER')
    
    if original_uid == 0 and sudo_user:
        # Running as root via sudo - try to access user's audio session
        try:
            import pwd
            user_info = pwd.getpwnam(sudo_user)
            user_uid = user_info.pw_uid
            user_home = user_info.pw_dir
            
            env['HOME'] = user_home
            env['USER'] = sudo_user
            
            # Try PipeWire/PulseAudio runtime paths
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
            subprocess.run(
                ["pw-play", file_path],
                timeout=120,
                check=True,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            played = True
            print("[TTS] Playback via PipeWire")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    # Try PulseAudio
    if not played:
        try:
            subprocess.run(
                ["paplay", file_path],
                timeout=120,
                check=True,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            played = True
            print("[TTS] Playback via PulseAudio")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    # Try ALSA
    if not played:
        try:
            subprocess.run(
                ["aplay", "-q", file_path],
                timeout=120,
                check=True,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE
            )
            played = True
            print("[TTS] Playback via ALSA")
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    
    if not played:
        print("[TTS] ERROR: All audio backends failed")



def _speak_espeak(text: str, voice_id: Optional[str],
                  output_device: Optional[str], sample_rate: int):
    """Speak using espeak-ng (Linux) with fallback chain for audio backends.
    Handles running as root by connecting to the user's PipeWire/PulseAudio session."""
    
    temp_dir = Path(cfg.TEMP_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_wav = temp_dir / "tts_output.wav"

    # Detect if running as root and prepare environment
    env = os.environ.copy()
    original_uid = os.getuid()
    
    if original_uid == 0:  # Running as root
        # Try to get the sudo user's UID (who invoked sudo)
        sudo_uid = os.environ.get('SUDO_UID')
        sudo_user = os.environ.get('SUDO_USER')
        
        if sudo_uid:
            target_uid = int(sudo_uid)
            target_runtime = f"/run/user/{target_uid}"
            
            if os.path.exists(target_runtime):
                print(f"[TTS] Running as root, switching to user {sudo_user} (UID {target_uid}) audio session")
                env['XDG_RUNTIME_DIR'] = target_runtime
                
                # Also set PULSE_RUNTIME_PATH for PulseAudio compatibility
                env['PULSE_RUNTIME_PATH'] = f"{target_runtime}/pulse"
                env['PIPEWIRE_RUNTIME_DIR'] = target_runtime
                
                # For PipeWire socket
                if os.path.exists(f"{target_runtime}/pipewire-0"):
                    env['PIPEWIRE_RUNTIME_DIR'] = target_runtime
            else:
                print(f"[TTS] Warning: Running as root but cannot access user {target_uid} runtime dir")
                print("[TTS] Hint: Run without sudo, or use 'sudo -E' to preserve environment")
        else:
            # Not run via sudo, try common user UID 1000
            common_runtime = "/run/user/1000"
            if os.path.exists(common_runtime) and os.path.exists(f"{common_runtime}/pulse"):
                print(f"[TTS] Warning: Running as root, attempting to use UID 1000 audio session")
                env['XDG_RUNTIME_DIR'] = common_runtime
                env['PULSE_RUNTIME_PATH'] = f"{common_runtime}/pulse"

    try:
        # Build espeak-ng command
        cmd = ["espeak-ng"]
        
        voice_to_use = voice_id if voice_id else 'en'
        cmd.extend(["-v", voice_to_use])
        
        cmd.extend(["-w", str(temp_wav), text])
        
        if _tts_stop_flag.is_set():
            return
        
        result = subprocess.run(cmd, capture_output=True, timeout=60, env=env)
        
        # If voice doesn't exist, fallback to 'en'
        if result.returncode != 0:
            stderr_str = result.stderr.decode() if result.stderr else ""
            if "voice does not exist" in stderr_str.lower() or "does not exist" in stderr_str.lower():
                print(f"[TTS] Voice '{voice_to_use}' not installed, falling back to 'en'")
                cmd = ["espeak-ng", "-v", "en", "-w", str(temp_wav), text]
                result = subprocess.run(cmd, capture_output=True, timeout=60, env=env)
                
                if result.returncode != 0:
                    print(f"[TTS] Fallback voice also failed: {result.stderr.decode()}")
                    return
            else:
                print(f"[TTS] espeak-ng error: {stderr_str}")
                return
        
        if _tts_stop_flag.is_set():
            return
        
        # Try playback backends in order of preference with fallback
        temp_wav_str = str(temp_wav)
        played = False
        
        # Normalize device name - "Default Sound Device" should use system default
        actual_device = None
        if output_device and output_device != "default" and output_device != "Default Sound Device":
            actual_device = output_device
        # If it's "default" or "Default Sound Device", leave as None to use system default
        
        # Attempt 1: PipeWire (native) - uses XDG_RUNTIME_DIR from env
        try:
            result = subprocess.run(
                ["pw-play", temp_wav_str], 
                capture_output=True, 
                timeout=120,
                check=True,
                env=env  # Pass modified environment
            )
            played = True
            print("[TTS] Playback via PipeWire")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            if isinstance(e, subprocess.CalledProcessError):
                err_out = e.stderr.decode() if e.stderr else ""
                if "Host is down" in err_out or "Connection refused" in err_out:
                    print("[TTS] PipeWire not accessible (root audio session issue)")
                else:
                    print(f"[TTS] PipeWire error: {err_out}")
            # Continue to fallback
        
        # Attempt 2: PulseAudio (paplay) - uses PULSE_RUNTIME_PATH from env
        if not played:
            try:
                play_cmd = ["paplay"]
                if actual_device and actual_device not in ["default", "Default Sound Device"]:
                    play_cmd.extend(["--device=" + actual_device])
                
                play_cmd.append(temp_wav_str)
                
                result = subprocess.run(
                    play_cmd, 
                    timeout=120, 
                    check=True, 
                    env=env,  # Pass modified environment
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.PIPE
                )
                played = True
                print("[TTS] Playback via PulseAudio (paplay)")
            except subprocess.CalledProcessError as e:
                err_out = e.stderr.decode() if e.stderr else "Unknown error"
                if "Connection refused" in err_out:
                    print("[TTS] PulseAudio not accessible (connection refused)")
                else:
                    print(f"[TTS] PulseAudio error: {err_out}")
            except FileNotFoundError:
                pass  # paplay not installed
        
        # Attempt 3: ALSA (aplay) - may not work if PipeWire is blocking the device
        if not played:
            try:
                play_cmd = ["aplay", "-q"]
                if actual_device:
                    play_cmd.extend(["-D", actual_device])
                else:
                    play_cmd.extend(["-D", "default"])
                
                play_cmd.append(temp_wav_str)
                
                result = subprocess.run(
                    play_cmd, 
                    timeout=120, 
                    check=True,
                    env=env,
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.PIPE
                )
                played = True
                print("[TTS] Playback via ALSA (aplay)")
            except subprocess.CalledProcessError as e:
                err_out = e.stderr.decode() if e.stderr else "Unknown error"
                if "Host is down" in err_out:
                    print("[TTS] ALSA blocked by PipeWire (root cannot access ALSA when PipeWire is active)")
                else:
                    print(f"[TTS] ALSA error: {err_out}")
        
        if not played:
            print("[TTS] ERROR: All audio backends failed")
            if original_uid == 0:
                print("[TTS] CAUSE: Running as root without audio session access")
                print("[TTS] FIX: Run without sudo, or use: sudo -E ./Chat-Gradio-Gguf.sh")
            else:
                print("[TTS] Please check your audio system is running")
                
    except subprocess.TimeoutExpired:
        print("[TTS] Speech timed out")
    except Exception as e:
        print(f"[TTS] espeak-ng playback error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            if temp_wav.exists():
                temp_wav.unlink()
        except:
            pass

def synthesize_last_response(session_messages: list) -> Optional[str]:
    """Synthesize TTS audio from the last AI response (blocking).
    
    Returns the path to the generated WAV file, or None on failure.
    Does NOT play the audio - call play_tts_audio() separately.
    This split allows the UI to update progress between synthesis and playback.
    """
    import os
    from pathlib import Path
    
    if not getattr(cfg, 'TTS_ENABLED', False):
        return None
    
    if not session_messages:
        return None
    
    # Find last assistant message
    last_response = None
    for msg in reversed(session_messages):
        if msg.get('role') == 'assistant':
            last_response = msg.get('content', '')
            break
    
    if not last_response:
        return None
    
    # Aggressive cleaning optimized for TTS
    text = last_response
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`[^`]+`', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    text = re.sub(r'!\[.*?\]\([^)]+\)', '', text)
    text = re.sub(r'(\*\*|\*|_|\~\~|`)', '', text)
    text = re.sub(r'([#*\u2022\u2192\u21d2\u2605\u2606]|[-=]{2,})', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\'\"()-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    if not text:
        return None
    
    # Truncate very long responses
    max_len = getattr(cfg, 'MAX_TTS_LENGTH', 4500)
    if len(text) > max_len:
        text = text[:max_len] + "... Response truncated for speech."
    
    voice_id = getattr(cfg, 'TTS_VOICE', None)
    engine = getattr(cfg, 'TTS_ENGINE', None) or detect_tts_engine()
    
    if engine != "coqui":
        # For non-coqui engines, use the old blocking path
        output_device = cfg.SOUND_OUTPUT_DEVICE
        sample_rate = cfg.SOUND_SAMPLE_RATE
        speak_text(text, voice_id, output_device, sample_rate, blocking=True)
        return "__played__"  # Signal that playback already happened
    
    # Coqui synthesis only (no playback)
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
            voice_id = getattr(cfg, 'COQUI_VOICE_ID', 'p243')
        
        print(f"[TTS] Coqui synthesizing with voice {voice_id}...")
        
        tts = TTS(model_name=model_name, progress_bar=False)
        
        temp_dir = Path(cfg.TEMP_DIR) if cfg.TEMP_DIR else base_dir / "data" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_wav = temp_dir / f"coqui_speech_{os.getpid()}.wav"
        
        if _tts_stop_flag.is_set():
            return None
        
        tts.tts_to_file(
            text=text,
            file_path=str(temp_wav),
            speaker=voice_id
        )
        
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
        wav_path: Path to the WAV file to play
        output_device: Audio output device (None for system default)
    """
    from pathlib import Path
    
    if not wav_path or wav_path == "__played__":
        return  # Already played by non-coqui engine, or nothing to play
    
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
        # Clean up temp file
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
    
    # Display engine info
    if engine == "coqui":
        coqui_voice = getattr(cfg, 'COQUI_VOICE_ID', 'p243')
        coqui_accent = getattr(cfg, 'COQUI_VOICE_ACCENT', 'british')
        print(f"[TTS] Engine: Coqui TTS (voice: {coqui_voice}, accent: {coqui_accent})")
    else:
        print(f"[TTS] Engine: {engine}")
    print(f"[TTS] Audio Backend: {backend}")
    
    # Set default voice - match by ID first, then name
    voices = get_tts_voices()
    voice_ids = [v['id'] for v in voices]
    voice_names = [v['name'] for v in voices]
    
    if voices:
        saved_id = getattr(cfg, 'TTS_VOICE', None)
        saved_name = getattr(cfg, 'TTS_VOICE_NAME', None)
        
        if saved_id and saved_id in voice_ids:
            # Voice ID is valid - sync the name to match
            for v in voices:
                if v['id'] == saved_id:
                    cfg.TTS_VOICE = v['id']
                    cfg.TTS_VOICE_NAME = v['name']
                    break
            print(f"[TTS] Voice from config: {cfg.TTS_VOICE_NAME} ({cfg.TTS_VOICE})")
        elif saved_name and saved_name in voice_names:
            # Name match - sync the ID
            for v in voices:
                if v['name'] == saved_name:
                    cfg.TTS_VOICE = v['id']
                    cfg.TTS_VOICE_NAME = v['name']
                    break
            print(f"[TTS] Voice from config: {cfg.TTS_VOICE_NAME} ({cfg.TTS_VOICE})")
        else:
            # No valid saved voice - reset to first available
            cfg.TTS_VOICE = voices[0]['id']
            cfg.TTS_VOICE_NAME = voices[0]['name']
            print(f"[TTS] Default Voice set to: {voices[0]['name']}")
    else:
        cfg.TTS_VOICE = None
        cfg.TTS_VOICE_NAME = "No voices available"
        print("[TTS] No voices detected")
    
    return engine != "none"


def get_tts_status() -> str:
    """Get TTS status string for display."""
    
    engine = getattr(cfg, 'TTS_ENGINE', 'none')
    enabled = getattr(cfg, 'TTS_ENABLED', False)
    tts_type = getattr(cfg, 'TTS_TYPE', 'builtin')
    
    if engine == "none":
        return "TTS: Not Available"
    elif enabled:
        voice = getattr(cfg, 'TTS_VOICE_NAME', 'Default')
        if engine == "coqui":
            return f"TTS: ON (Coqui - {voice})"
        return f"TTS: ON ({voice})"
    else:
        if engine == "coqui":
            return "TTS: OFF (Coqui)"
        return f"TTS: OFF ({engine})"


def speak_last_response(session_messages: list) -> str:
    """
    Speak the last AI response from session.
    
    Args:
        session_messages: List of message dicts
        
    Returns:
        Status message
    """
    
    
    if not getattr(cfg, 'TTS_ENABLED', False):
        return "TTS is disabled"
    
    if not session_messages:
        return "No messages to speak"
    
    # Find last assistant message
    last_response = None
    for msg in reversed(session_messages):
        if msg.get('role') == 'assistant':
            last_response = msg.get('content', '')
            break
    
    if not last_response:
        return "No AI response to speak"
    
    # Aggressive cleaning optimized for TTS
    text = last_response
    
    # Remove markdown code blocks completely
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # Remove inline code
    text = re.sub(r'`[^`]+`', '', text)
    
    # Remove HTML-like tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove markdown links but keep text: [text](url) → text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove images and other markdown junk
    text = re.sub(r'!\[.*?\]\([^)]+\)', '', text)
    
    # Remove emphasis markers but keep the words
    text = re.sub(r'(\*\*|\*|_|\~\~|`)', '', text)
    
    # Replace multiple symbols with space or nothing
    text = re.sub(r'([#*•→⇒★☆]|[-=]{2,})', ' ', text)
    
    # Normalize punctuation (keep , . ! ? ; : and quotes)
    text = re.sub(r'[^\w\s.,!?;:\'\"()-]', ' ', text)
    
    # Collapse multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    if not text:
        return "Response has no speakable content after cleaning"
    
    # Truncate very long responses (TTS engines often struggle above ~4000–5000 chars)
    if len(text) > 4500:
        text = text[:4500] + "... Response truncated for speech."
    
    voice_id = getattr(cfg, 'TTS_VOICE', None)
    output_device = getattr(cfg, 'TTS_OUTPUT_DEVICE', None)
    sample_rate = getattr(cfg, 'TTS_SAMPLE_RATE', 44100)
    
    if speak_text(text, voice_id, output_device, sample_rate):
        return "Speaking response..."
    else:
        return "Failed to start speech"