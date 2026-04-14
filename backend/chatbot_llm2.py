"""
FastAPI RAG Web Crawler with Ollama Integration
- Asynchronous crawling of internal pages (up to 5 pages, depth 2)
- Static HTTP fallback for non‑JS sites
- Playwright with resource blocking for speed
- Caching for scraped content and LLM answers
- Robust text extraction with boilerplate removal
"""

import asyncio
import time
from collections import deque
from urllib.parse import urljoin, urlparse
from typing import List, Optional

import requests
from bs4 import BeautifulSoup
from cachetools import TTLCache
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from playwright.async_api import async_playwright, Browser, Page, Route
from pydantic import BaseModel

# ============== Configuration ==============
MAX_PAGES = 5                # Total pages to crawl
MAX_DEPTH = 2                # Maximum link depth
CONTEXT_CHAR_LIMIT = 8000    # Characters sent to LLM
OLLAMA_MODEL = "llama3"      # Ollama model name
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT = 60         # seconds for LLM call
PAGE_LOAD_TIMEOUT = 15000    # ms for Playwright
CACHE_TTL_SCRAPE = 3600      # 1 hour for scraped pages
CACHE_TTL_QA = 1800          # 30 minutes for Q&A pairs
# ===========================================

app = FastAPI(title="RAG Web Crawler API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caches
scrape_cache = TTLCache(maxsize=200, ttl=CACHE_TTL_SCRAPE)
qa_cache = TTLCache(maxsize=1000, ttl=CACHE_TTL_QA)

# Global Playwright browser instance (reused across requests)
_browser: Optional[Browser] = None
_playwright_instance = None


# ============== Request Model ==============
class QuestionRequest(BaseModel):
    url: str
    question: str


# ============== Playwright Browser Lifecycle ==============
async def get_browser() -> Browser:
    """Return a shared browser instance, creating it if necessary."""
    global _browser, _playwright_instance
    if _browser is None or not _browser.is_connected():
        _playwright_instance = await async_playwright().start()
        _browser = await _playwright_instance.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-blink-features=AutomationControlled",
            ],
        )
    return _browser


async def close_browser():
    """Cleanly shut down the browser (call on app shutdown)."""
    global _browser, _playwright_instance
    if _browser:
        await _browser.close()
        _browser = None
    if _playwright_instance:
        await _playwright_instance.stop()
        _playwright_instance = None


# ============== Content Extraction ==============
def extract_visible_text(html: str) -> str:
    """
    Extract clean, readable text from HTML.
    Removes scripts, styles, nav, footers, and short junk lines.
    """
    soup = BeautifulSoup(html, "lxml")  # lxml is faster than html.parser

    # Remove non-content tags
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]
    ):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # Collapse whitespace and filter short/boilerplate lines
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 30]
    cleaned = " ".join(lines)

    # Remove common cookie/privacy noise
    noise = ["cookie", "privacy policy", "accept all", "advertisement"]
    for phrase in noise:
        cleaned = cleaned.replace(phrase, "")

    return cleaned


async def block_unnecessary_resources(route: Route):
    """Block images, stylesheets, fonts, and media to speed up page loads."""
    if route.request.resource_type in {"image", "stylesheet", "font", "media"}:
        await route.abort()
    else:
        await route.continue_()
from requests_html import AsyncHTMLSession

async def fetch_page_content(url: str) -> str:
    """Alternative using requests-html instead of Playwright"""
    session = AsyncHTMLSession()
    try:
        response = await session.get(url)
        await response.html.arender(timeout=10, sleep=1)
        return extract_visible_text(response.html.html)
    except Exception as e:
        print(f"⚠️ Error fetching {url}: {e}")
        return ""
    finally:
        await session.close()


def fetch_static(url: str) -> Optional[str]:
    """
    Attempt to fetch page using plain HTTP (fast fallback for non‑JS sites).
    Returns text if successful and content length > 200 chars.
    """
    try:
        resp = requests.get(
            url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"},
        )
        if resp.status_code == 200:
            text = extract_visible_text(resp.text)
            if len(text) > 200:
                return text
    except Exception:
        pass
    return None


async def get_page_content(url: str) -> str:
    """Cached wrapper that tries static fetch first, then Playwright."""
    if url in scrape_cache:
        return scrape_cache[url]

    # Try static fetch
    static_content = fetch_static(url)
    if static_content:
        scrape_cache[url] = static_content
        return static_content

    # Fallback to Playwright
    browser = await get_browser()
    content = await fetch_page_content(browser, url)
    scrape_cache[url] = content
    return content


# ============== Crawler: Follow Internal Links ==============
def is_same_domain(url1: str, url2: str) -> bool:
    return urlparse(url1).netloc == urlparse(url2).netloc


def is_valid_link(href: str) -> bool:
    """Skip binary files and anchors."""
    skip_ext = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".mp4", ".exe")
    return not any(href.lower().endswith(ext) for ext in skip_ext)


async def discover_links(html: str, base_url: str) -> List[str]:
    """Extract internal links from HTML."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        absolute = urljoin(base_url, a["href"])
        if is_same_domain(absolute, base_url) and is_valid_link(absolute):
            links.append(absolute)
    return links


async def crawl_site(start_url: str) -> str:
    """
    Crawl internal pages up to MAX_PAGES and MAX_DEPTH.
    Returns concatenated text content (limited to CONTEXT_CHAR_LIMIT).
    """
    visited = set()
    queue = deque([(start_url, 0)])  # (url, depth)
    content_pieces = []
    pages_processed = 0

    # We'll do a limited BFS with concurrent page fetching per depth level.
    browser = await get_browser()

    while queue and pages_processed < MAX_PAGES:
        # Collect URLs for current depth level
        level_urls = []
        current_depth = queue[0][1]
        while queue and queue[0][1] == current_depth and len(level_urls) < (MAX_PAGES - pages_processed):
            url, depth = queue.popleft()
            if url not in visited:
                visited.add(url)
                level_urls.append(url)

        if not level_urls:
            continue

        # Fetch all pages at this depth concurrently
        tasks = [fetch_page_content(browser, url) for url in level_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for url, content in zip(level_urls, results):
            if isinstance(content, Exception):
                print(f"❌ Failed {url}: {content}")
                continue

            # Cache the content
            scrape_cache[url] = content
            if content:
                content_pieces.append(content[:3000])  # Limit per page
                pages_processed += 1

            # Discover new links from this page (only if depth allows)
            if current_depth < MAX_DEPTH and pages_processed < MAX_PAGES:
                # Use the raw HTML we already fetched? We'd need to store it.
                # Simpler: re‑fetch HTML for link discovery (minor overhead).
                try:
                    page = await browser.new_page()
                    await page.goto(url, timeout=PAGE_LOAD_TIMEOUT)
                    html = await page.content()
                    await page.close()
                    new_links = await discover_links(html, url)
                    for link in new_links:
                        if link not in visited:
                            queue.append((link, current_depth + 1))
                except Exception as e:
                    print(f"⚠️ Link discovery failed for {url}: {e}")

    # Combine all content and truncate
    full_text = "\n\n".join(content_pieces)
    return full_text[:CONTEXT_CHAR_LIMIT]


# ============== LLM Integration ==============
def ask_ollama(context: str, question: str) -> str:
    """Send prompt to Ollama and return answer."""
    prompt = f"""You are a helpful assistant that answers questions based on the provided website content.
If the content is insufficient, you may use general knowledge but clearly indicate when you are doing so.

WEBSITE CONTENT:
{context}

QUESTION: {question}

ANSWER:"""
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "No response generated.")
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return f"Error communicating with LLM: {str(e)}"


# ============== FastAPI Endpoints ==============
@app.post("/ask")
async def ask_question(req: QuestionRequest):
    start_time = time.time()

    # Check Q&A cache
    cache_key = (req.url, req.question)
    if cache_key in qa_cache:
        return {"answer": qa_cache[cache_key], "cached": True}

    print(f"🌐 Crawling: {req.url}")

    # Crawl and gather context
    context = await crawl_site(req.url)
    crawl_time = time.time() - start_time
    print(f"⏱️ Crawl took {crawl_time:.2f}s, context length: {len(context)} chars")

    # Ask LLM
    answer = ask_ollama(context, req.question)
    qa_cache[cache_key] = answer

    return {
        "answer": answer,
        "crawl_time_seconds": round(crawl_time, 2),
        "pages_crawled": len(scrape_cache),  # approximate count for this domain
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.on_event("shutdown")
async def shutdown_event():
    await close_browser()


# ============== Entry Point ==============
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)