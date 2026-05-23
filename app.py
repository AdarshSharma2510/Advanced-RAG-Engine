import os
import shutil
import streamlit as st
from Pipeline import AdvancedRAGEngine

st.set_page_config(
    page_title="Advanced Enterprise RAG Engine",
    page_icon="🤖",
    layout="wide"
)

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = AdvancedRAGEngine()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

engine = st.session_state.rag_engine

st.title("🤖 Advanced Enterprise RAG Engine")
st.markdown("Production-grade pipeline featuring query transformation, vector storage indexing, and CPU-first context reranking.")

with st.sidebar:
    st.header("Document Control Center")
    uploaded_file = st.file_uploader("Upload Document (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Process & Index Document", use_container_width=True):
            with st.spinner("Executing advanced ingestion and chunking..."):
                try:
                    temp_dir = "data/tmp"
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_path = os.path.join(temp_dir, uploaded_file.name)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    engine.ingest_document(temp_path)
                    st.success("Document indexed successfully into vector storage!")
                    
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except Exception as e:
                    st.error(f"Ingestion lifecycle failed: {str(e)}")
                    
    st.markdown("---")
    if st.button("Clear Knowledge Base", type="primary", use_container_width=True):
        try:
            engine.clear_knowledge_base()
            st.session_state.chat_history = []
            st.warning("Database collections dropped and session state flushed.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to clear database: {str(e)}")

st.subheader("Interactive Inquiry Interface")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "transformation" in message:
            st.caption(f"**Optimized Retrieval Query:** *{message['transformation']}*")
        if "sources" in message and message["sources"]:
            with st.expander("Inspected Context Blocks"):
                for idx, doc in enumerate(message["sources"]):
                    st.markdown(f"**Fragment {idx + 1}** — *Page {doc.metadata.get('page', 'N/A')}*")
                    st.text(doc.page_content)

if user_input := st.chat_input("Ask a question relative to your indexed document corpus:"):
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Orchestrating transformation, retrieval, and synthesis..."):
            try:
                pipeline_output = engine.execute(user_input)
                
                st.write(pipeline_output["answer"])
                st.caption(f"**Optimized Retrieval Query:** *{pipeline_output['transformed_query']}*")
                
                if pipeline_output["source_documents"]:
                    with st.expander("Inspected Context Blocks"):
                        for idx, doc in enumerate(pipeline_output["source_documents"]):
                            st.markdown(f"**Fragment {idx + 1}** — *Page {doc.metadata.get('page', 'N/A')}*")
                            st.text(doc.page_content)
                
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": pipeline_output["answer"],
                    "transformation": pipeline_output["transformed_query"],
                    "sources": pipeline_output["source_documents"]
                })
            except Exception as e:
                st.error(f"Pipeline execution encountered an error: {str(e)}")