import faiss
import pickle
import numpy as np
import os

def fake_embed(text):
    return np.array([hash(text) % 1000], dtype='float32')

def load_texts():
    with open("data/texts.txt", "r", encoding="utf-8") as f:
        content = f.read()

    return [t.strip() for t in content.split("\n\n") if t.strip()]

texts = load_texts()

embeddings = np.array([fake_embed(t) for t in texts])

index = faiss.IndexFlatL2(1)
index.add(embeddings)

os.makedirs("db", exist_ok=True)

faiss.write_index(index, "db/faiss.index")

with open("db/texts.pkl", "wb") as f:
    pickle.dump(texts, f)

print("OK")
