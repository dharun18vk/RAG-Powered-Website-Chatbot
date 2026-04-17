"""
FastAPI RAG Web Crawler with Ollama Integration + LLM Fallback
- Attempts to crawl website content first
- Falls back to LLM's general knowledge if crawling fails
- Returns answer with context about which method was used
"""

import asyncio
import time
import re
from collections import deque
from urllib.parse import urljoin, urlparse
from typing import List, Optional, Tuple

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
MIN_QUALITY_CONTENT_LENGTH = 200  
# ===========================================

app = FastAPI(title="RAG Web Crawler API with LLM Fallback")

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

class AnswerResponse(BaseModel):
    answer: str
    method: str  # "crawled_content" or "llm_knowledge"
    crawl_time_seconds: float = 0
    pages_crawled: int = 0
    content_length: int = 0
    cached: bool = False

def extract_visible_text(html: str) -> str:
    """Extract clean, readable text from HTML."""
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    
    text = soup.get_text(separator=" ", strip=True)

    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 30]
    cleaned = " ".join(lines)
    
    noise = ["cookie", "privacy policy", "accept all", "advertisement", "subscribe", "newsletter"]
    for phrase in noise:
        cleaned = cleaned.replace(phrase, "")
    
    return cleaned

def is_quality_content(text: str) -> bool:
    """Check if scraped content is actually useful (not a login/blocked page)"""
    if len(text) < MIN_QUALITY_CONTENT_LENGTH:
        return False
    
    blocked_patterns = [
        r"enable javascript",
        r"enable cookies",
        r"login",
        r"sign in",
        r"access denied",
        r"blocked",
        r"verify you are human",
        r"captcha",
        r"please wait",
        r"unsupported browser",
        r"your browser is not supported",
        r"cloudflare",
        r"ddos protection",
    ]
    
    text_lower = text.lower()
    for pattern in blocked_patterns:
        if re.search(pattern, text_lower):
            return False
    
    
    words = text.split()
    if len(words) < 50:  
        return False
    

    unique_words = len(set(words))
    if unique_words / len(words) < 0.3:  # Less than 30% unique words
        return False
    
    return True

def fetch_static(url: str) -> Optional[Tuple[str, str]]:
    """Fetch page using plain HTTP. Returns (content, raw_html) or None."""
    try:
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ]
        
        for user_agent in user_agents:
            try:
                resp = requests.get(
                    url,
                    timeout=10,
                    headers={"User-Agent": user_agent},
                    allow_redirects=True
                )
                if resp.status_code == 200:
                    text = extract_visible_text(resp.text)
                    if is_quality_content(text):
                        return (text, resp.text)
            except:
                continue
                
    except Exception as e:
        print(f"⚠️ Static fetch error for {url}: {e}")
    
    return None

async def get_page_content(url: str) -> Tuple[str, Optional[str]]:
    """Returns (content, raw_html) - content may be empty if fetch failed."""
    if url in scrape_cache:
        cached = scrape_cache[url]
        if isinstance(cached, tuple):
            return cached
        return (cached, None)
    
    result = fetch_static(url)
    if result:
        content, raw_html = result
        scrape_cache[url] = (content, raw_html)
        return (content, raw_html)
    
    scrape_cache[url] = ("", None)
    return ("", None)

def is_same_domain(url1: str, url2: str) -> bool:
    return urlparse(url1).netloc == urlparse(url2).netloc

def is_valid_link(href: str) -> bool:
    """Skip binary files and anchors."""
    skip_ext = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".zip", ".mp4", ".exe", ".xml", ".rss")
    return not any(href.lower().endswith(ext) for ext in skip_ext)

async def discover_links(html: str, base_url: str) -> List[str]:
    """Extract internal links from HTML."""
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        absolute = urljoin(base_url, a["href"])
        if is_same_domain(absolute, base_url) and is_valid_link(absolute):
            # Avoid duplicates and common non-content pages
            if not any(skip in absolute.lower() for skip in ['/tag/', '/category/', '/feed', '/wp-']):
                links.append(absolute)
    return list(set(links))  

async def crawl_site(start_url: str) -> Tuple[str, int]:
    """
    Crawl internal pages using static HTTP only.
    Returns (combined_content, pages_crawled)
    """
    visited = set()
    queue = deque([(start_url, 0)])
    content_pieces = []
    pages_processed = 0
    failed_urls = []

    print(f"🕷️ Starting crawl of {start_url}")
    
    while queue and pages_processed < MAX_PAGES:
        url, depth = queue.popleft()
        
        if url in visited:
            continue
            
        visited.add(url)
        print(f"📄 Crawling: {url} (depth {depth})")
        
        content, raw_html = await get_page_content(url)
        
        if content and is_quality_content(content):
            content_pieces.append(f"Source: {url}\n{content[:3000]}")
            pages_processed += 1
  
            if depth < MAX_DEPTH and pages_processed < MAX_PAGES and raw_html:
                try:
                    new_links = await discover_links(raw_html, url)
                    for link in new_links[:10]: 
                        if link not in visited:
                            queue.append((link, depth + 1))
                except Exception as e:
                    print(f"⚠️ Link discovery failed for {url}: {e}")
        else:
            failed_urls.append(url)
            print(f"❌ No quality content from {url}")

    if pages_processed == 0:
        print(f"⚠️ No pages successfully crawled. Failed URLs: {failed_urls[:5]}")
        return "", 0
    
    full_text = "\n\n---\n\n".join(content_pieces)
    print(f"✅ Crawled {pages_processed} pages, total content length: {len(full_text)} chars")
    return full_text[:CONTEXT_CHAR_LIMIT], pages_processed

def ask_ollama_with_url_fallback(url: str, question: str, crawled_content: str = "", has_content: bool = False) -> Tuple[str, str]:
    """
    Send prompt to Ollama. If no content was crawled, ask LLM to use its knowledge about the URL.
    Returns (answer, method_used)
    """
    
    if has_content and crawled_content and len(crawled_content) > 100:
        # Use crawled content
        prompt = f"""You are a helpful assistant that answers questions based on the provided website content.

WEBSITE URL: {url}
WEBSITE CONTENT:
{crawled_content}

QUESTION: {question}

Instructions:
- Answer based ONLY on the website content provided above
- If the answer cannot be found in the content, say "I couldn't find this information on the website"
- Be specific and cite relevant parts from the content

ANSWER:"""
        method = "crawled_content"
    else:
        # Fallback: Use LLM's general knowledge about the URL
        prompt = f"""You are a helpful assistant. The user is asking about a website, but the crawler couldn't access its content (possibly due to blocking, JavaScript requirements, or authentication).

WEBSITE URL: {url}
USER'S QUESTION: {question}

Instructions:
- Use your general knowledge about this website/domain to answer the question
- Be clear that you're answering based on general knowledge, NOT from crawling the actual page
- If you don't know something about this specific website, say so honestly
- Provide helpful information about what this website typically contains or is known for

ANSWER:"""
        method = "llm_knowledge"
    
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return (resp.json().get("response", "No response generated."), method)
    except Exception as e:
        print(f"❌ Ollama error: {e}")
        return (f"Error communicating with LLM: {str(e)}", "error")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(req: QuestionRequest):
    start_time = time.time()
    
    # Check Q&A cache
    cache_key = (req.url, req.question)
    if cache_key in qa_cache:
        cached_result = qa_cache[cache_key]
        return AnswerResponse(
            answer=cached_result['answer'],
            method=cached_result['method'],
            crawl_time_seconds=cached_result.get('crawl_time', 0),
            pages_crawled=cached_result.get('pages', 0),
            content_length=cached_result.get('content_length', 0),
            cached=True
        )
    
    print(f"🌐 Processing request for URL: {req.url}")
    print(f"❓ Question: {req.question}")
    
    # Attempt to crawl the site
    crawled_content, pages_crawled = await crawl_site(req.url)
    crawl_time = time.time() - start_time
    
    has_quality_content = len(crawled_content) > 100
    
    if has_quality_content:
        print(f"✅ Successfully crawled {pages_crawled} pages, using content for answer")
    else:
        print(f"⚠️ Failed to get quality content, falling back to LLM's knowledge")
    
    # Get answer from LLM (with appropriate context)
    answer, method = ask_ollama_with_url_fallback(
        req.url, 
        req.question, 
        crawled_content, 
        has_quality_content
    )
    
    # Cache the result
    qa_cache[cache_key] = {
        'answer': answer,
        'method': method,
        'crawl_time': crawl_time,
        'pages': pages_crawled,
        'content_length': len(crawled_content)
    }
    
    return AnswerResponse(
        answer=answer,
        method=method,
        crawl_time_seconds=round(crawl_time, 2),
        pages_crawled=pages_crawled,
        content_length=len(crawled_content),
        cached=False
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return {
        "scrape_cache_size": len(scrape_cache),
        "qa_cache_size": len(qa_cache),
        "scrape_cache_maxsize": scrape_cache.maxsize,
        "qa_cache_maxsize": qa_cache.maxsize
    }

@app.delete("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    scrape_cache.clear()
    qa_cache.clear()
    return {"message": "Caches cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)