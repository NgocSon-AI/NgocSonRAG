import os
from typing import List
from PyPDF2 import PdfReader
from docx import Document as DocxDocument

class Document:
    def __init__(self, file_path: str, chunk_size: int, chunk_overlap: int):
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.content = self._load_file()
        self.chunks = self._chunk_content()

    def _load_file(self) -> str:
        ext = os.path.splitext(self.file_path)[1].lower()
        if ext == ".pdf":
            return self._read_pdf()
        elif ext == ".docx":
            return self._read_docx()
        elif ext == ".txt":
            return self._read_txt()
        else:
            raise ValueError("Unsupported file type")

    def _read_pdf(self) -> str:
        reader = PdfReader(self.file_path)
        return "\n".join([page.extract_text() for page in reader.pages])

    def _read_docx(self) -> str:
        doc = DocxDocument(self.file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    def _read_txt(self) -> str:
        with open(self.file_path, "r", encoding="utf-8") as f:
            return f.read()

    def _chunk_content(self) -> List[str]:
        """
        Chia content thanh cac chunk voi overlap tuong ung
        """
        text = self.content.replace("\n", " ")
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks
