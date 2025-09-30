from fastapi import APIRouter, UploadFile, File
from src.models.documents import Document
from src.services.rag import RAGService


router = APIRouter()
rag_service = RAGService()

@router.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    path = f"data/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    doc = Document(path)
    rag_service.add_document(doc)

    return {"message": f"File {file.filename} đã được thêm vào RAG database."}


@router.post("/query/")
async def query_rag(question: str):
    answer = rag_service.query(question)
    return {"answer": answer}
