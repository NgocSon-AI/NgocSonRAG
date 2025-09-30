import faiss
import numpy as np
import os

class VectorStore:
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self.metadata = np.load(index_path + ".meta.npy", allow_pickle=True).tolist()
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.metadata = []

    def add_vectors(self, vectors: np.ndarray, metadatas: list):
        self.index.add(vectors)
        self.metadata.extend(metadatas)
        self.save()

    def search(self, vector: np.ndarray, top_k: int = 5):
        D, I = self.index.search(vector, top_k)
        results = []
        for ids in I:
            results.append([self.metadata[i] for i in ids])
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        np.save(self.index_path + ".meta.npy", np.array(self.metadata, dtype=object))
