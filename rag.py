import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL")

class RAGEngine:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index("db/faiss.index")

        with open("db/texts.pkl", "rb") as f:
            self.texts = pickle.load(f)

    def retrieve(self, topic, k=5):
        query = f"نص تحفيزي مباشر بأسلوب قوي عن {topic}"
        emb = self.model.encode([query])

        _, idx = self.index.search(emb, k)

        return [self.texts[i] for i in idx[0]]
