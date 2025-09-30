from groq import Groq
from src.models.vector_store import VectorStore
from src.services.embedding import EmbeddingService
from src.config import get_settings


settings = get_settings()

class RAGService:
    def __init__(self):
        self.client = Groq(api_key=settings.GROQ_API_KEY)
        self.embedding_service = EmbeddingService()
        # FAISS 1536 dim, tương ứng với text-embedding-3-small
        self.vector_store = VectorStore(dim=1536, index_path="data/faiss.index")

    def add_document(self, document):
        """
        Thêm document mới: tạo embedding và lưu vào FAISS
        """
        embeddings = self.embedding_service.get_embeddings(document.chunks)
        metadata = [{"text": c} for c in document.chunks]
        self.vector_store.add_vectors(embeddings, metadata)

    def query(self, question: str, top_k=5) -> str:
        """
        Truy vấn RAG: tìm context trong vector DB rồi gọi Groq API
        """
        q_emb = self.embedding_service.get_embeddings([question])
        results = self.vector_store.search(q_emb, top_k=top_k)
        context_text = "\n".join([item["text"] for item in results[0]])

        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Bạn là trợ lý thông minh."},
                {"role": "user", "content": question}
            ],
            model=settings.GROQ_MODEL,
            documents=[{"content": context_text}],
            enable_citations=True
        )
        return response.choices[0].message.content
