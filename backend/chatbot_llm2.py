"""
RAG Web Chatbot - Full Implementation
- Smart recursive crawling (priority pages)
- Hybrid extraction: newspaper3k → Playwright → BS4
- Block detection (Cloudflare, login walls)
- Content cleaning & chunking (500 chars, 50 overlap)
- Embeddings: sentence-transformers (all-MiniLM-L6-v2)
- Vector store: FAISS (cosine similarity)
- LLM: Ollama (llama3 / mistral)
- FastAPI backend + simple HTML frontend
- Caching & failure handling
"""

import asyncio
import hashlib
import re
import time
from collections import deque
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
import numpy as np
from bs4 import BeautifulSoup
from cachetools import TTLCache
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from newspaper import Article
from sentence_transformers import SentenceTransformer
import faiss
from playwright.async_api import async_playwright, Browser

# ============== CONFIGURATION ==============
MAX_PAGES = 7                # total pages to crawl
MAX_DEPTH = 2                # link depth
PRIORITY_PATHS = ["about", "about-us", "services", "product", "solutions", "contact", "help"]
CHUNK_SIZE = 500             # characters
CHUNK_OVERLAP = 50
TOP_K = 3                    # number of chunks retrieved
OLLAMA_MODEL = "llama3"      # change to "mistral" if preferred
OLLAMA_URL = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT = 30
PLAYWRIGHT_TIMEOUT = 15000   # ms
CACHE_TTL = 3600             # 1 hour
# ===========================================

app = FastAPI(title="RAG Web Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Caches
page_cache = TTLCache(maxsize=200, ttl=CACHE_TTL)      # raw extracted text per URL
index_cache = TTLCache(maxsize=10, ttl=CACHE_TTL)      # FAISS index + chunks per domain

# Global Playwright browser (lazy)
_browser: Optional[Browser] = None

# Embedding model (loaded once)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# ============== HELPER FUNCTIONS ==============
def is_blocked_page(text: str) -> bool:
    """Detect Cloudflare, login, or bot-protected pages."""
    patterns = [
        r"checking your browser",
        r"verify you are human",
        r"access denied",
        r"please enable cookies",
        r"captcha",
        r"login",
        r"sign in",
        r"cloudflare",
    ]
    lower = text.lower()
    return any(re.search(p, lower) for p in patterns)

def clean_html(html: str) -> str:
    """Remove navigation, footers, scripts; keep headings, paragraphs, lists."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    # Keep headings, paragraphs, lists, divs with main content
    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 20]
    cleaned = "\n".join(lines)
    return cleaned

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks, respecting sentence boundaries."""
    if not text:
        return []
    # Simple recursive split on sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = []
    current_len = 0
    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len <= chunk_size:
            current.append(sent)
            current_len += sent_len
        else:
            if current:
                chunks.append(" ".join(current))
            # start new chunk with overlap (last 1-2 sentences)
            overlap_text = " ".join(current[-2:]) if len(current) >= 2 else " ".join(current)
            current = [overlap_text, sent] if overlap_text else [sent]
            current_len = len(overlap_text) + sent_len + 1
    if current:
        chunks.append(" ".join(current))
    return chunks

def get_link_priority(url: str) -> int:
    """Higher score = higher priority."""
    path = urlparse(url).path.lower()
    for i, p in enumerate(PRIORITY_PATHS):
        if p in path:
            return len(PRIORITY_PATHS) - i
    return 0

async def get_browser() -> Browser:
    global _browser
    if _browser is None or not _browser.is_connected():
        p = await async_playwright().start()
        _browser = await p.chromium.launch(headless=True, args=["--disable-gpu"])
    return _browser

# ============== HYBRID EXTRACTION ==============
async def extract_with_playwright(url: str) -> Optional[str]:
    """Use Playwright for JS-heavy sites."""
    browser = await get_browser()
    page = None
    try:
        page = await browser.new_page()
        await page.goto(url, timeout=PLAYWRIGHT_TIMEOUT)
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(0.5)
        html = await page.content()
        text = clean_html(html)
        if is_blocked_page(text):
            return None
        return text
    except Exception as e:
        print(f"Playwright error for {url}: {e}")
        return None
    finally:
        if page:
            await page.close()

def extract_with_newspaper(url: str) -> Optional[str]:
    """Best for articles."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        if text and len(text) > 200 and not is_blocked_page(text):
            return text
    except:
        pass
    return None

def extract_with_bs4(url: str) -> Optional[str]:
    """Fallback static extraction."""
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code == 200:
            text = clean_html(resp.text)
            if len(text) > 200 and not is_blocked_page(text):
                return text
    except:
        pass
    return None

async def extract_content(url: str) -> str:
    """Multi‑strategy extraction with fallback."""
    if url in page_cache:
        return page_cache[url]
    
    # 1. Try newspaper3k
    content = extract_with_newspaper(url)
    if content:
        page_cache[url] = content
        return content
    
    # 2. Try Playwright (async)
    content = await extract_with_playwright(url)
    if content:
        page_cache[url] = content
        return content
    
    # 3. Fallback to BS4
    content = extract_with_bs4(url)
    if content:
        page_cache[url] = content
        return content
    
    return ""

# ============== SMART CRAWLER ==============
async def discover_links(html: str, base_url: str) -> List[str]:
    """Extract internal links, filter by domain and priority."""
    soup = BeautifulSoup(html, "lxml")
    domain = urlparse(base_url).netloc
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        absolute = urljoin(base_url, href)
        # same domain, ignore non‑html extensions
        if urlparse(absolute).netloc == domain and not absolute.endswith(('.pdf', '.jpg', '.png', '.zip')):
            links.add(absolute)
    # Sort by priority (higher score first)
    sorted_links = sorted(links, key=get_link_priority, reverse=True)
    return sorted_links

async def crawl_site(seed_url: str) -> Dict[str, str]:
    """
    Crawl up to MAX_PAGES pages, prioritising important paths.
    Returns {url: extracted_text}
    """
    visited = set()
    queue = deque([(seed_url, 0)])
    results = {}
    
    while queue and len(results) < MAX_PAGES:
        url, depth = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        
        print(f"Crawling: {url} (depth {depth})")
        content = await extract_content(url)
        if content:
            results[url] = content
        else:
            continue  # skip failed pages
        
        if depth >= MAX_DEPTH:
            continue
        
        # Discover new links (use raw HTML – re‑fetch for simplicity)
        try:
            resp = requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200:
                new_links = await discover_links(resp.text, url)
                # Prioritise and limit
                for link in new_links[:10]:  # avoid explosion
                    if link not in visited and len(results) < MAX_PAGES:
                        queue.append((link, depth + 1))
        except Exception as e:
            print(f"Link discovery failed for {url}: {e}")
    return results

# ============== BUILD INDEX ==============
def build_index(page_texts: Dict[str, str]) -> Tuple[faiss.IndexFlatIP, List[Dict]]:
    """
    Chunk all pages, embed, build FAISS index (cosine similarity).
    Returns (index, list of metadata dicts)
    """
    all_chunks = []
    metadata = []
    for url, text in page_texts.items():
        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            metadata.append({"url": url, "chunk": chunk})
    
    if not all_chunks:
        return None, []
    
    # Compute embeddings
    embeddings = embedder.encode(all_chunks, convert_to_numpy=True)
    # Normalize for cosine similarity (Inner Product = cosine if normalized)
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # Inner Product (cosine after normalization)
    index.add(embeddings)
    
    return index, metadata

# ============== RETRIEVAL ==============
def retrieve(query: str, index: faiss.IndexFlatIP, metadata: List[Dict], k: int = TOP_K) -> List[Dict]:
    """Return top‑k relevant chunks with similarity scores."""
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                "chunk": metadata[idx]["chunk"],
                "url": metadata[idx]["url"],
                "score": float(scores[0][i])
            })
    return results

# ============== OLLAMA GENERATION ==============
async def ask_llm(context: str, question: str, use_general_knowledge: bool = False) -> str:
    """Generate answer with optional fallback to general knowledge."""
    if use_general_knowledge:
        prompt = f"""You are a helpful assistant. The website could not be crawled (blocked or no content). 
        Use your general knowledge to answer the following question about the website URL (if possible). 
        If you don't know, say so.
        
        Website URL: {context}   (this is the URL, not content)
        Question: {question}
        
        Answer:"""
    else:
        prompt = f"""You are a RAG assistant. Answer the question based ONLY on the provided context.
        If the answer is not in the context answer one line using your general knowledge.
        
        CONTEXT:
        {context}
        
        QUESTION: {question}
        
        ANSWER:"""
    
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        return resp.json().get("response", "No answer generated.")
    except Exception as e:
        return f"LLM error: {str(e)}"

# ============== API ENDPOINTS ==============
class BuildRequest(BaseModel):
    url: str

class AskRequest(BaseModel):
    url: str
    question: str

class BuildResponse(BaseModel):
    status: str
    pages_crawled: int
    chunks_created: int
    message: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]
    used_fallback: bool
    retrieval_scores: List[float]

# In‑memory index storage per domain (for demo simplicity)
# Key = domain, value = (faiss_index, metadata)
stored_indices: Dict[str, Tuple[faiss.IndexFlatIP, List[Dict]]] = {}

@app.post("/build", response_model=BuildResponse)
async def build_index_endpoint(req: BuildRequest):
    """Crawl and index the website."""
    domain = urlparse(req.url).netloc
    if domain in stored_indices:
        return BuildResponse(status="already_built", pages_crawled=0, chunks_created=0, message="Index already exists. Use /ask or /rebuild.")
    
    print(f"Building index for {req.url}")
    page_texts = await crawl_site(req.url)
    if not page_texts:
        raise HTTPException(status_code=400, detail="No content extracted. Site may be blocked or empty.")
    
    index, metadata = build_index(page_texts)
    if index is None or not metadata:
        raise HTTPException(status_code=400, detail="No chunks created. Content too short.")
    
    stored_indices[domain] = (index, metadata)
    return BuildResponse(
        status="success",
        pages_crawled=len(page_texts),
        chunks_created=len(metadata),
        message=f"Indexed {len(page_texts)} pages into {len(metadata)} chunks."
    )

@app.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """Retrieve and generate answer."""
    domain = urlparse(req.url).netloc
    if domain not in stored_indices:
        # Option: auto‑build if missing
        raise HTTPException(status_code=404, detail="No index found for this domain. Call /build first.")
    
    index, metadata = stored_indices[domain]
    # Retrieve relevant chunks
    results = retrieve(req.question, index, metadata)
    if not results:
        # Fallback to LLM general knowledge
        answer = await ask_llm(req.url, req.question, use_general_knowledge=True)
        return AskResponse(
            answer=answer,
            sources=[],
            used_fallback=True,
            retrieval_scores=[]
        )
    
    # Build context from top chunks
    context = "\n\n---\n\n".join([r["chunk"] for r in results])
    answer = await ask_llm(context, req.question, use_general_knowledge=False)
    sources = list(set([r["url"] for r in results]))
    scores = [r["score"] for r in results]
    
    return AskResponse(
        answer=answer,
        sources=sources,
        used_fallback=False,
        retrieval_scores=scores
    )

@app.get("/", response_class=HTMLResponse)
async def frontend():
    """Simple HTML interface."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Web Chatbot</title>
        <style>
            body { font-family: Arial; max-width: 800px; margin: 40px auto; padding: 20px; }
            input, button { padding: 10px; margin: 5px; width: 100%; }
            .status { color: gray; font-size: 0.9em; }
            .answer { background: #f4f4f4; padding: 15px; border-radius: 8px; margin-top: 20px; }
            .sources { font-size: 0.8em; color: #0066cc; }
            hr { margin: 20px 0; }
        </style>
    </head>
    <body>
        <h1>🤖 RAG Web Chatbot</h1>
        <div>
            <h3>Step 1: Build Index</h3>
            <input type="text" id="buildUrl" placeholder="https://example.com" />
            <button id="buildBtn">🔨 Build / Crawl</button>
            <div id="buildStatus" class="status"></div>
        </div>
        <hr/>
        <div>
            <h3>Step 2: Ask a Question</h3>
            <input type="text" id="askUrl" placeholder="Same domain URL" />
            <input type="text" id="question" placeholder="What is this website about?" />
            <button id="askBtn">💬 Ask</button>
            <div id="answerArea" class="answer" style="display:none;">
                <div id="answerText"></div>
                <div id="sources" class="sources"></div>
                <div id="fallbackNote" class="status"></div>
            </div>
        </div>
        <script>
            document.getElementById('buildBtn').onclick = async () => {
                const url = document.getElementById('buildUrl').value;
                const statusDiv = document.getElementById('buildStatus');
                statusDiv.innerText = '🕷️ Crawling and indexing... (may take 10-30 sec)';
                try {
                    const res = await fetch('/build', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({url})
                    });
                    const data = await res.json();
                    if (res.ok) {
                        statusDiv.innerText = `✅ Success: ${data.pages_crawled} pages, ${data.chunks_created} chunks.`;
                    } else {
                        statusDiv.innerText = `❌ Error: ${data.detail || 'Unknown'}`;
                    }
                } catch(e) {
                    statusDiv.innerText = `❌ Network error: ${e.message}`;
                }
            };
            document.getElementById('askBtn').onclick = async () => {
                const url = document.getElementById('askUrl').value;
                const question = document.getElementById('question').value;
                const answerDiv = document.getElementById('answerArea');
                const answerText = document.getElementById('answerText');
                const sourcesDiv = document.getElementById('sources');
                const fallbackNote = document.getElementById('fallbackNote');
                answerDiv.style.display = 'block';
                answerText.innerText = '🤔 Thinking...';
                sourcesDiv.innerText = '';
                fallbackNote.innerText = '';
                try {
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({url, question})
                    });
                    const data = await res.json();
                    if (res.ok) {
                        answerText.innerText = data.answer;
                        if (data.sources.length) {
                            sourcesDiv.innerText = '📄 Sources: ' + data.sources.join(', ');
                        }
                        if (data.used_fallback) {
                            fallbackNote.innerText = '⚠️ Answer based on general knowledge (no content could be extracted).';
                        } else if (data.retrieval_scores) {
                            fallbackNote.innerText = `🔍 Retrieval confidence: ${data.retrieval_scores.map(s=>s.toFixed(2)).join(', ')}`;
                        }
                    } else {
                        answerText.innerText = `Error: ${data.detail || 'Unknown'}`;
                    }
                } catch(e) {
                    answerText.innerText = `Network error: ${e.message}`;
                }
            };
        </script>
    </body>
    </html>
    """

@app.on_event("shutdown")
async def shutdown():
    global _browser
    if _browser:
        await _browser.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)