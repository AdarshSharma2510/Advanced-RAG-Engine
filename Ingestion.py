import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings

class DocumentIngestor:
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len,
            separators = ["\n\n", "\n", " ", ""]
        )
        
    def load_pdf(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at {file_path}")
    
        try:
            loader = PyPDFLoader(file_path)
            return loader.load()
        except Exception as e:
            raise RuntimeError(f"Failed to parse PDF file due to: {str(e)}")
    
    def split_documents(self, documents: List[Document]) ->List[Document]:
        if not documents:
            return []
        return self.splitter.split_documents(documents)
    
    def process_file(self, file_path: str)->List[Document]:
        raw_docs = self.load_pdf(file_path)
        return self.split_documents(raw_docs)
    