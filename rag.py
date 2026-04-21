import faiss
import pickle
import os
import numpy as np

class RAGEngine:
    def __init__(self):
        if not os.path.exists("db/faiss.index"):
            print("⚠️ بناء قاعدة البيانات...")
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

        print("✅ تم بناء RAG تلقائيًا")

    def retrieve(self, topic, k=5):
        query_vec = self.fake_embed(topic).reshape(1, -1)
        _, idx = self.index.search(query_vec, k)

        return [self.texts[i] for i in idx[0]]
