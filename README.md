````markdown
# 🚀 RAG-Powered Website Chatbot

## 📌 Overview

This project is a Retrieval-Augmented Generation (RAG) based chatbot that can:

- 🌐 Ingest any website URL
- 🔁 Recursively crawl linked pages
- 🧠 Extract structured & unstructured content
- 🔍 Perform semantic search using embeddings
- 🤖 Generate accurate answers using a local LLM (Ollama)

## 🎯 Features

- ✔ Smart recursive crawling (multi-page)
- ✔ Hybrid scraping (Newspaper + Playwright + BeautifulSoup)
- ✔ Cloudflare / bot detection handling
- ✔ Chunking + semantic embeddings
- ✔ FAISS vector search (fast retrieval)
- ✔ Local LLM (Ollama – llama3/mistral)
- ✔ Caching for performance
- ✔ Modern chat UI

## 🧠 How It Works

### 🔄 Pipeline Flow

```
User URL
   ↓
Crawler (multi-page)
   ↓
Content Extraction (Hybrid)
   ↓
Cleaning + Chunking
   ↓
Embeddings (MiniLM)
   ↓
FAISS Vector Index
   ↓
User Question
   ↓
Relevant Chunks Retrieved
   ↓
Ollama LLM
   ↓
Answer
```

## ⚙️ Tech Stack

| Component          | Technology                     |
|--------------------|--------------------------------|
| Backend            | FastAPI                        |
| Frontend           | HTML + JS                      |
| Scraping           | Playwright, BeautifulSoup, Newspaper3k |
| Embeddings         | sentence-transformers          |
| Vector DB          | FAISS                          |
| LLM                | Ollama (llama3 / mistral)      |

## 📁 Project Structure

```
project/
│
├── backend/
│   └── chatbot_llm2.py   # main backend
│
├── frontend/
│   └── main_ll2.html     # UI
│
└── README.md
```

## ⚡ Installation & Setup

### 🔧 1. Clone Project

```bash
git clone <your-repo-url>
cd project
```

### 🔧 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 🔧 3. Install Dependencies

```bash
pip install fastapi uvicorn requests beautifulsoup4
pip install playwright newspaper3k
pip install faiss-cpu sentence-transformers
pip install cachetools lxml_html_clean
```

### 🔧 4. Install Playwright Browsers

```bash
python -m playwright install
```

### 🔧 5. Setup Ollama

#### Install Ollama

- Download and install Ollama from the official website: [https://ollama.com](https://ollama.com)
- Follow the installation instructions for your operating system (Windows, macOS, or Linux).

#### Run Ollama Server

- Open a terminal or command prompt.
- Start the Ollama server:

```bash
ollama serve
```

#### Pull and Run llama3 Model

- In a separate terminal or command prompt, pull the llama3 model:

```bash
ollama pull llama3
```

- To run the model interactively (optional, for testing):

```bash
ollama run llama3
```

- Note: You can also use `mistral` for faster performance by replacing `llama3` with `mistral` in the commands above.

## ▶️ Running the Project

### 🚀 Start Backend

```bash
cd backend
python -m uvicorn chatbot_llm2:app --reload
```

👉 Server runs at: http://127.0.0.1:8000

### 🌐 Open Frontend

Open this file in browser: `frontend/main_ll2.html`

## 🧪 Usage

### Step 1: Index Website

- Enter URL in sidebar
- Click "Index Website"

👉 Backend will:

- Crawl multiple pages
- Extract content
- Build vector index

### Step 2: Ask Questions

Example questions:

- What does this company do?
- What services are offered?
- Summarize this website

## 🧠 Key Components Explained

### 🔹 1. Smart Crawler

- Extracts internal links
- Prioritizes important pages: `/about`, `/services`
- Limits pages to avoid overload

### 🔹 2. Hybrid Extraction

Uses 3 methods:

- Newspaper3k → articles
- Playwright → JS-heavy sites
- BeautifulSoup → fallback

### 🔹 3. Block Detection

Detects:

- Cloudflare
- CAPTCHA
- Login walls

👉 Skips blocked pages automatically

### 🔹 4. Chunking

- Splits content into 500-character chunks
- Uses overlap for better context

### 🔹 5. Embeddings + FAISS

- Converts text → vectors
- Stores in FAISS
- Enables fast similarity search

### 🔹 6. Retrieval

- Finds top 3 relevant chunks
- Sends only relevant context to LLM

### 🔹 7. LLM (Ollama)

- Generates final answer
- Uses: Context (RAG) OR fallback (general knowledge)

## ⚡ Performance Optimization

- ✔ Caching (TTLCache)
- ✔ Limited pages (max 7)
- ✔ Limited chunk size
- ✔ Async Playwright

## 🚨 Known Limitations

| Issue                  | Reason          | Solution                  |
|------------------------|-----------------|---------------------------|
| Some sites blocked     | Cloudflare      | Use subpages              |
| Login-required pages   | Auth needed     | Not supported             |
| Dynamic apps           | JS-heavy        | Playwright helps          |
| Slow scraping          | Heavy pages     | Reduce depth              |

## 🔥 Example

**Input:** https://www.claysys.com/about-us/

**Output:** ClaySys is a technology company specializing in software development, automation, and digital transformation solutions.

## 🎯 Future Improvements

- 🔥 Google Search fallback
- ⚡ Async multi-thread crawling
- 🧠 Chat memory
- 🎨 UI improvements
- 📄 Source highlighting

## 🧠 Interview Explanation

This project implements a RAG-based chatbot that crawls and extracts multi-page website content using a hybrid scraping approach. The data is chunked, embedded, and stored in a FAISS vector database for semantic retrieval. A local LLM (Ollama) then generates context-aware answers, with fallback handling for blocked or dynamic websites.

## 🔚 Conclusion

This project demonstrates:

- ✔ Real-world RAG system
- ✔ Web crawling + extraction
- ✔ Vector search
- ✔ Local LLM integration
````