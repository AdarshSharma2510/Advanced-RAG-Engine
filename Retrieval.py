import os
from typing import List
from pathlib import Path
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, ChatHuggingFace, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_core.retrievers import ContextualCompressionRetriever
from dotenv import load_dotenv
load_dotenv()

embedding_model = HuggingFaceEmbeddings(model_name = 'BAAI/bge-base-en-v1.5')
llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.2-3B-Instruct',
    task = 'text-generation'
)
model = ChatHuggingFace(llm = llm)

class VectorStorageManager:
    def __init__(self):
        self.embeddings = embedding_model
        self.vector_store = None
    
    def get_vector_store(self) -> Chroma:
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name="advanced_rag_collection",
                embedding_function = self.embeddings,
                persist_directory= str(Path(__file__).parent.parent / "data" / "chroma") 
            )
        return self._vector_store
    
    def index_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        store = self.get_vector_store()
        store.add_documents(documents)
        
    def reset_storage(self)->None:
        store = self.get_vector_store()
        store.delete_collection()
        self._vector_store = None

class QueryTransformer:
    def __init__(self):
        self.llm = model
        self.prompt = ChatPromptTemplate([
            ("system", "You are an AI assistant tasked with optimizing user questions for Vector Database retrieval. Rewrite the original query to include technical synonyms, clearer phrasing, and crucial domain-specific keywords. Output ONLY the optimized rewritten query text, with absolutely no introduction or explanation."),
            ("human", "Original Query: {query}\nOptimized Query:")
        ])
        self.parser = StrOutputParser()
        self.rewriter_chain = self.prompt | self.llm | self.parser
        
    def rewrite_query(self, query: str) -> str:
        try:
            return self.rewriter_chain.invoke({"query": query})
        except Exception:
            return query
        
class ContextReranker:
    def __init__(self, vector_manager : VectorStorageManager):
        self.vector_manager = vector_manager
        self.compressor = FlashrankRerank(
            model = 'ms-marco-MiniLM-L-12-v2',
            top_n = 4
        )
    
    def get_managed_retriever(self) -> ContextualCompressionRetriever:
        base_retriever = self.vector_manager.get_vector_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}
        )
        return ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=base_retriever
        )