from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import os
import requests
from dotenv import load_dotenv

# ===== Load ENV =====
load_dotenv()

# ===== App =====
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ===== ENV Variables =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TOP_K = int(os.getenv("TOP_K", 5))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.9))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 800))

# ===== Simple RAG (خفيف) =====
import faiss
import pickle
import numpy as np

class RAGEngine:
    def __init__(self):
        if not os.path.exists("db/faiss.index"):
            print("⚠️ Building RAG index...")
            self.build_index()

        self.index = faiss.read_index("db/faiss.index")

        with open("db/texts.pkl", "rb") as f:
            self.texts = pickle.load(f)

    def fake_embed(self, text):
        return np.array([hash(text) % 1000], dtype='float32')

    def build_index(self):
        os.makedirs("db", exist_ok=True)

        with open("data/texts.txt", "r", encoding="utf-8") as f:
            content = f.read()

        texts = [t.strip() for t in content.split("\n\n") if t.strip()]

        embeddings = np.array([self.fake_embed(t) for t in texts])

        index = faiss.IndexFlatL2(1)
        index.add(embeddings)

        faiss.write_index(index, "db/faiss.index")

        with open("db/texts.pkl", "wb") as f:
            pickle.dump(texts, f)

        print("✅ RAG ready")

    def retrieve(self, topic, k=5):
        vec = self.fake_embed(topic).reshape(1, -1)
        _, idx = self.index.search(vec, k)

        return [self.texts[i] for i in idx[0]]

# ===== Initialize RAG =====
rag = RAGEngine()

# ===== Schema =====
class GenerateRequest(BaseModel):
    topic: str

# ===== Prompt =====
def build_prompt(topic, examples):
    examples_text = "\n\n".join(examples)

    return f"""
أنت كاتب بأسلوب قوي جدًا.

ممنوع:
- الشرح
- المقدمات
- الحشو

الأسلوب:
- مباشر
- صادم
- جمل قصيرة
- تأثير نفسي قوي

أمثلة:
{examples_text}

اكتب نصًا عن:
{topic}
"""

# ===== Gemini API =====
def generate_text(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "topP": 0.95,
            "maxOutputTokens": MAX_TOKENS
        }
    }

    try:
        res = requests.post(url, json=payload)
        data = res.json()

        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return f"خطأ: {str(e)}"

# ===== Pages =====

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        name="index.html",
        context={"request": request},
        request=request
    )

@app.get("/ingest", response_class=HTMLResponse)
def ingest_page(request: Request):
    return templates.TemplateResponse(
        name="ingest.html",
        context={"request": request},
        request=request
    )

# ===== API =====

@app.post("/generate")
def generate(req: GenerateRequest):
    examples = rag.retrieve(req.topic, k=TOP_K)
    prompt = build_prompt(req.topic, examples)
    output = generate_text(prompt)

    return JSONResponse({"generated": output})

# ===== Ingest =====

@app.post("/ingest")
def ingest_data(content: str = Form(...)):
    with open("data/texts.txt", "w", encoding="utf-8") as f:
        f.write(content)

    # إعادة بناء RAG
    rag.build_index()

    return JSONResponse({"status": "updated"})
