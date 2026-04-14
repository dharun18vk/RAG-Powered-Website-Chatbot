"""
FastAPI RAG Web Crawler with Ollama Integration (Playwright-free version)
- Static HTTP crawling only (no JavaScript rendering)
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
from pydantic import BaseModel

# ============== Configuration ==============
MAX_PAGES = 5
MAX_DEPTH = 2
CONTEXT_CHAR_LIMIT = 8000
OLLAMA_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT = 60
CACHE_TTL_SCRAPE = 3600
CACHE_TTL_QA = 1800
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

class QuestionRequest(BaseModel):
    url: str
    question: str

def extract_visible_text(html: str) -> str:
    """Extract clean, readable text from HTML."""
    soup = BeautifulSoup(html, "lxml")
    
    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    
    text = soup.get_text(separator=" ", strip=True)
    
    # Collapse whitespace and filter short lines
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 30]
    cleaned = " ".join(lines)
    
    # Remove common noise
    noise = ["cookie", "privacy policy", "accept all", "advertisement"]
    for phrase in noise:
        cleaned = cleaned.replace(phrase, "")
    
    return cleaned

def fetch_static(url: str) -> Optional[str]:
    """Fetch page using plain HTTP."""
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
    except Exception as e:
        print(f"⚠️ Static fetch error for {url}: {e}")
    return None

async def get_page_content(url: str) -> str:
    """Cached wrapper for static fetch."""
    if url in scrape_cache:
        return scrape_cache[url]
    
    content = fetch_static(url) or ""
    scrape_cache[url] = content
    return content

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
    """Crawl internal pages using static HTTP only."""
    visited = set()
    queue = deque([(start_url, 0)])
    content_pieces = []
    pages_processed = 0

    while queue and pages_processed < MAX_PAGES:
        url, depth = queue.popleft()
        
        if url in visited:
            continue
            
        visited.add(url)
        print(f"📄 Crawling: {url} (depth {depth})")
        
        content = await get_page_content(url)
        
        if content:
            content_pieces.append(content[:3000])
            pages_processed += 1
            
            # Discover links if depth allows
            if depth < MAX_DEPTH and pages_processed < MAX_PAGES:
                try:
                    resp = requests.get(
                        url, 
                        timeout=8,
                        headers={"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
                    )
                    if resp.status_code == 200:
                        new_links = await discover_links(resp.text, url)
                        for link in new_links:
                            if link not in visited:
                                queue.append((link, depth + 1))
                except Exception as e:
                    print(f"⚠️ Link discovery failed for {url}: {e}")

    full_text = "\n\n".join(content_pieces)
    print(f"✅ Crawled {pages_processed} pages, {len(full_text)} chars")
    return full_text[:CONTEXT_CHAR_LIMIT]

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

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    start_time = time.time()
    
    cache_key = (req.url, req.question)
    if cache_key in qa_cache:
        return {"answer": qa_cache[cache_key], "cached": True}
    
    print(f"🌐 Crawling: {req.url}")
    context = await crawl_site(req.url)
    crawl_time = time.time() - start_time
    
    answer = ask_ollama(context, req.question)
    qa_cache[cache_key] = answer
    
    return {
        "answer": answer,
        "crawl_time_seconds": round(crawl_time, 2),
        "pages_crawled": len(scrape_cache),
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)