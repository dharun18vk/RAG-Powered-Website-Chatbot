from fastapi import FastAPI
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup

app = FastAPI()

class RequestData(BaseModel):
    url: str
    question: str

def scrape(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    for tag in soup(["script", "style"]):
        tag.extract()

    text = " ".join([p.get_text() for p in soup.find_all("p")])
    return text[:4000]  # limit size

def ask_ollama(context, question):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": f"""
Answer based ONLY on the content below.

Content:
{context[:2000]}

Question: {question}
""",
                "stream": False   # ✅ THIS FIXES YOUR ERROR
            },
            timeout=60
        )

        data = response.json()

        return data.get("response", "⚠️ No response")

    except Exception as e:
        print("❌ ERROR:", str(e))
        return f"Error: {str(e)}"

@app.post("/ask")
def ask(req: RequestData):
    content = scrape(req.url)
    answer = ask_ollama(content, req.question)

    return {"answer": answer}