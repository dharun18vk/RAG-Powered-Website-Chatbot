from fastapi import FastAPI
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urljoin, urlparse
import time
from collections import deque

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model
class RequestData(BaseModel):
    url: str
    question: str

# ---------------------------
# Enhanced Crawler & Scraper
# ---------------------------

def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs belong to the same domain."""
    return urlparse(url1).netloc == urlparse(url2).netloc

def extract_visible_text(html: str) -> str:
    """Extract clean visible text from HTML, removing boilerplate."""
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    # Get text and clean it
    text = soup.get_text(separator=" ", strip=True)

    # Basic cleaning: collapse whitespace, remove very short lines, filter common junk
    lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 30]
    cleaned = " ".join(lines)

    # Remove cookie/privacy notice fragments (case-insensitive)
    junk_phrases = ["cookie", "privacy policy", "accept all", "advertisement"]
    for phrase in junk_phrases:
        cleaned = cleaned.replace(phrase, "")

    return cleaned

def crawl_and_extract(start_url: str, max_pages: int = 5, max_depth: int = 2) -> str:
    """
    Crawl internal pages starting from start_url, extract visible text,
    and return combined content (limited to ~10k characters).
    """
    from playwright.sync_api import sync_playwright

    visited = set()
    queue = deque()
    queue.append((start_url, 0))  # (url, depth)
    all_content = []
    total_pages = 0

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)

        while queue and total_pages < max_pages:
            url, depth = queue.popleft()
            if url in visited or depth > max_depth:
                continue

            try:
                page = browser.new_page()
                page.goto(url, timeout=30000)
                # Wait for initial content to load
                page.wait_for_load_state("networkidle", timeout=10000)
                # Additional time for dynamic content
                page.wait_for_timeout(2000)

                html = page.content()
                page.close()

                visited.add(url)
                total_pages += 1

                # Extract text from current page
                text = extract_visible_text(html)
                if text:
                    all_content.append(text[:3000])  # Limit per page to keep context balanced

                # If we haven't reached max_pages, find internal links for next level
                if total_pages < max_pages and depth < max_depth:
                    soup = BeautifulSoup(html, "html.parser")
                    for link in soup.find_all("a", href=True):
                        absolute = urljoin(url, link["href"])
                        if (is_same_domain(absolute, start_url) and
                            absolute not in visited and
                            not absolute.endswith((".pdf", ".jpg", ".png", ".zip", ".mp4"))):
                            queue.append((absolute, depth + 1))

            except Exception as e:
                print(f"⚠️ Failed to process {url}: {e}")
                continue

        browser.close()

    # Combine and truncate to fit LLM context (adjust based on model)
    full_content = "\n\n".join(all_content)
    return full_content[:10000]  # Keep ~10k chars for Llama3's context window

# ---------------------------
# Ollama LLM Call
# ---------------------------
def ask_ollama(context: str, question: str) -> str:
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": f"""
You are a helpful assistant that answers questions based on the provided website content.
If the content is insufficient, you may use general knowledge but clearly indicate when you are doing so.

WEBSITE CONTENT:
{context}

QUESTION: {question}

ANSWER:""",
                "stream": False
            },
            timeout=90
        )
        data = response.json()
        return data.get("response", "No response generated.")
    except Exception as e:
        print("❌ Ollama error:", e)
        return f"Error calling LLM: {str(e)}"

@app.post("/ask")
def ask(req: RequestData):
    print(f"🌐 Crawling from: {req.url}")
    start = time.time()
    context = crawl_and_extract(req.url)
    print(f"⏱️ Crawl took {time.time() - start:.2f}s, content length: {len(context)} chars")
    answer = ask_ollama(context, req.question)
    return {"answer": answer}
@app.get("/")
def root():
    return {"status": "RAG Web Crawler API is running"}