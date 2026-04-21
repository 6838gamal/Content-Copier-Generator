from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL")

model = SentenceTransformer(MODEL_NAME)

def load_texts():
    with open("data/texts.txt", "r", encoding="utf-8") as f:
        content = f.read()

    return [t.strip() for t in content.split("\n\n") if t.strip()]

texts = load_texts()

embeddings = model.encode(texts)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

os.makedirs("db", exist_ok=True)

faiss.write_index(index, "db/faiss.index")

with open("db/texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("✅ تم بناء قاعدة الأسلوب")
