from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import os
import requests
from dotenv import load_dotenv

import faiss
import pickle
import numpy as np

import pandas as pd
from docx import Document
import PyPDF2

# ===== ENV =====
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TOP_K = int(os.getenv("TOP_K", 5))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.9))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 800))

# ===== APP =====
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ===== RAG =====
class RAGEngine:
    def __init__(self):
        if not os.path.exists("db/faiss.index"):
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

    def retrieve(self, topic, k=5):
        vec = self.fake_embed(topic).reshape(1, -1)
        _, idx = self.index.search(vec, k)
        return [self.texts[i] for i in idx[0]]

rag = RAGEngine()

# ===== Schema =====
class GenerateRequest(BaseModel):
    topic: str
    length: str

# ===== Prompt =====
def length_instruction(length):
    if length == "short":
        return "اكتب 3-5 جمل قصيرة"
    elif length == "medium":
        return "اكتب منشور متوسط 8-12 جملة"
    else:
        return "اكتب مقال طويل"

def build_prompt(topic, examples, length):
    return f"""
أنت كاتب بأسلوب قوي.

ممنوع:
- الشرح
- المقدمات

الأسلوب:
- مباشر
- صادم
- جمل قصيرة

{length_instruction(length)}

أمثلة:
{chr(10).join(examples)}

اكتب عن:
{topic}
"""

# ===== Gemini =====
def generate_text(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_TOKENS
        }
    }

    res = requests.post(url, json=payload)
    data = res.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return str(data)

# ===== Pages =====
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        name="index.html",
        context={"request": request},
        request=request
    )

# ===== Generate =====
@app.post("/generate")
def generate(req: GenerateRequest):
    examples = rag.retrieve(req.topic, k=TOP_K)
    prompt = build_prompt(req.topic, examples, req.length)
    result = generate_text(prompt)

    return JSONResponse({"generated": result})

# ===== Upload =====
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    content = ""

    if file.filename.endswith(".txt"):
        content = (await file.read()).decode("utf-8")

    elif file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
        content = "\n".join(df.astype(str).values.flatten())

    elif file.filename.endswith(".xlsx"):
        df = pd.read_excel(file.file)
        content = "\n".join(df.astype(str).values.flatten())

    elif file.filename.endswith(".docx"):
        doc = Document(file.file)
        content = "\n".join([p.text for p in doc.paragraphs])

    elif file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file.file)
        content = "\n".join([p.extract_text() or "" for p in reader.pages])

    elif file.filename.endswith(".json"):
        content = (await file.read()).decode("utf-8")

    else:
        return {"error": "نوع غير مدعوم"}

    with open("data/texts.txt", "a", encoding="utf-8") as f:
        f.write("\n\n" + content)

    rag.build_index()

    return {"status": "uploaded"}
