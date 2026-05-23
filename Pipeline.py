from typing import Dict, Any, List
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from Ingestion import DocumentIngestor
from Retrieval import VectorStorageManager, QueryTransformer, ContextReranker
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = 'meta-llama/Llama-3.2-3B-Instruct',
    task = 'text-generation'
)

class AdvancedRAGEngine:
    def __init__(self):
        self.ingestor = DocumentIngestor()
        self.vector_manager = VectorStorageManager()
        self.transformer = QueryTransformer()
        self.reranker = ContextReranker(self.vector_manager)
        
        self.llm = ChatHuggingFace(llm = llm)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert system answering questions based strictly on the provided context fragments. If the context does not contain the answer, state clearly that you cannot answer based on the current data. Do not make up information.\n\nContext:\n{context}"),
            ("human", "Question: {question}")
        ])
        
        self._build_core_chain()

    def _format_docs(self, docs: List[Document]) -> str:
        formatted = []
        for doc in docs:
            page = doc.metadata.get("page", "N/A")
            formatted.append(f"[Page {page}]: {doc.page_content}")
        return "\n\n".join(formatted)

    def _build_core_chain(self) -> None:
        retriever = self.reranker.get_managed_retriever()
        
        generation_chain = (
            RunnableParallel({
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough()
            })
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        self.full_chain = generation_chain

    def ingest_document(self, file_path: str) -> None:
        processed_chunks = self.ingestor.process_file(file_path)
        self.vector_manager.index_documents(processed_chunks)

    def clear_knowledge_base(self) -> None:
        self.vector_manager.reset_storage()

    def execute(self, user_query: str) -> Dict[str, Any]:
        optimized_query = self.transformer.rewrite_query(user_query)
        
        retriever = self.reranker.get_managed_retriever()
        retrieved_docs = retriever.invoke(optimized_query)
        formatted_context = self._format_docs(retrieved_docs)
        
        response_text = self.full_chain.invoke(optimized_query)
        
        return {
            "answer": response_text,
            "transformed_query": optimized_query,
            "source_documents": retrieved_docs
        }