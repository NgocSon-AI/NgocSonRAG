from openai import OpenAI
import numpy as np

class EmbeddingService:
    def __init__(self):
        self.client = OpenAI()

    def get_embeddings(self, texts: list) -> np.ndarray:
        embeddings = []
        for t in texts:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=t
            )
            embeddings.append(response.data[0].embedding)
        return np.array(embeddings, dtype=np.float32)
