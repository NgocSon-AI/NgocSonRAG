from fastapi import FastAPI
from src.app.app import router

app = FastAPI(title="RAG with Groq API")
app.include_router(router)
