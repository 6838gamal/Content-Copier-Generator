import faiss
import pickle
import numpy as np

class RAGEngine:
    def __init__(self):
        self.index = faiss.read_index("db/faiss.index")

        with open("db/texts.pkl", "rb") as f:
            self.texts = pickle.load(f)

    def fake_embed(self, text):
        # embedding بسيط جدًا لتجنب torch
        return np.array([hash(text) % 1000], dtype='float32')

    def retrieve(self, topic, k=5):
        query_vec = self.fake_embed(topic).reshape(1, -1)

        _, idx = self.index.search(query_vec, k)

        return [self.texts[i] for i in idx[0]]
