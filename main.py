from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import requests
import subprocess
import os
from dotenv import load_dotenv

from rag import RAGEngine

# ===== Load env =====
load_dotenv()

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

rag = RAGEngine()

# ===== ENV =====
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TOP_K = int(os.getenv("TOP_K", 5))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.9))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", 800))

# ===== Schema =====
class GenerateRequest(BaseModel):
    topic: str

# ===== Prompt =====
def build_prompt(topic, examples):
    examples_text = "\n\n".join(examples)

    return f"""
أنت كاتب محترف بأسلوب قوي جدًا.

ممنوع:
- الشرح
- المقدمات
- الحشو

الهدف:
تقليد الأسلوب فقط بدقة

الأسلوب:
- مباشر
- صادم
- جمل قصيرة
- تأثير نفسي عالي

أمثلة:
{examples_text}

اكتب نصًا عن:
{topic}
"""

# ===== Gemini (requests) =====
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
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ingest", response_class=HTMLResponse)
def ingest_page(request: Request):
    return templates.TemplateResponse("ingest.html", {"request": request})

# ===== API =====
@app.post("/generate")
def generate(req: GenerateRequest):
    examples = rag.retrieve(req.topic, k=TOP_K)
    prompt = build_prompt(req.topic, examples)
    output = generate_text(prompt)

    return {"generated": output}

# ===== ingest =====
@app.post("/ingest")
def ingest_data(content: str = Form(...)):
    with open("data/texts.txt", "w", encoding="utf-8") as f:
        f.write(content)

    subprocess.run(["python", "ingest.py"])

    return {"status": "updated"}
