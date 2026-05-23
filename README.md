# Advanced Enterprise RAG Engine

A production-grade Retrieval-Augmented Generation (RAG) engine built with pure LangChain (LCEL) and Streamlit. This repository completely moves away from naive RAG workflows by implementing a modular, fully decoupled architecture featuring query transformation, persistent vector indexing, and a two-stage retrieval funnel using a local CPU cross-encoder reranker.

## Architecture & System Features

* **Robust Document Ingestion**: Clean PDF structural parsing using `pypdf` paired with `RecursiveCharacterTextSplitter`. The system splits text along logical boundaries (paragraphs and sentences) using a fallback sequence of structural separators rather than slicing blindly at arbitrary token bounds.
* **Two-Stage Retrieval Funnel**:
    * *Stage 1 (Bi-Encoder)*: High-throughput semantic similarity search leveraging OpenAI's optimized `text-embedding-3-small` model to pull a wide candidate context window (k=15).
    * *Stage 2 (Cross-Encoder)*: Deep token-to-query attention scoring via a local CPU-first Cross-Encoder (`Flashrank`), filtering the candidate pool down to the top 4 highest-scoring fragments to combat context pollution and the "lost in the middle" phenomenon.
* **Query Transformation**: An LCEL-driven optimization layer powered by `gpt-4o-mini` at zero temperature. This layer standardizes and enriches colloquial questions with technical synonyms and domain keywords prior to vector lookup.
* **Declarative Orchestration**: Built entirely using LangChain Expression Language (LCEL) pipelines with explicit pipe (`|`) syntax, `RunnableParallel`, and `RunnablePassthrough` primitives to achieve optimal streaming and data routing.
* **Zero-Overhead Observability**: Complete plug-and-play integration with LangSmith for tracking execution trees, step-by-step token costs, and component latency breakdowns.

---

## Repository Structure

```text
advanced_rag_engine/
│
├── .env                  # Local environment configuration keys (git-ignored)
├── requirements.txt      # Production dependencies
├── README.md             # Project documentation and engineering blueprint
│
├── app.py                # Streamlit user interface and session state manager
├── config.py             # Type-safe configuration management via Pydantic
├── ingestion.py          # PDF parsing and hierarchical text splitting logic
├── retrieval.py          # Vector stores, query expansion, and Flashrank compression
└── pipeline.py           # End-to-end LCEL execution graph orchestration
```

---

## Technical Dependencies

The following requirements are specified inside `requirements.txt`:

```text
langchain>=0.3.0
langchain-community>=0.3.0
langchain-openai>=0.3.0
langchain-chroma>=0.2.0
flashrank>=0.2.0
pydantic-settings>=2.2.0
streamlit>=1.35.0
pypdf>=4.2.0
python-dotenv>=1.0.1
```

---

## Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/advanced-rag-engine.git](https://github.com/yourusername/advanced-rag-engine.git)
cd advanced-rag-engine
```

### 2. Configure Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Package Dependencies
```bash
pip install -r requirements.txt
```

### 4. Populate Local Environment Variables
Create a `.env` file in the root directory of the workspace and include your credentials:
```text
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_api_key_here
LANGCHAIN_PROJECT=Advanced-RAG-Engine
```

---

## Running the Application

Execute the Streamlit application from your terminal workspace root:

```bash
streamlit run app.py
```

---

## Core Component Architectures

### Ingestion (`Ingestion.py`)
Encapsulates PDF scanning and structural file chunking. The utility avoids character-clipping artifacts by keeping interrelated contextual definitions grouped within a single window.

### Retrieval & Compression (`Retrieval.py`)
Maintains a singleton persistence handle over `Chroma` collections. Houses the `QueryTransformer` block and wires the `ContextReranker` to an expanded k-value to let the cross-encoder sift text context effectively.

### Pipeline Orchestration (`Pipeline.py`)
Compiles the retrieval and generation phases into a clean functional execution stream. Utilizes `RunnableParallel` to parse raw user prompts and context documents simultaneously before presenting the combined payload to the final LLM synthesis layer.

---

## Production Design Patterns & Guardrails

1. **Hallucination Mitigation**: The generation system prompt applies strict rules forcing the synthesis engine to declare if an answer cannot be surfaced within the provided text blocks, containing fabricated inferences.
2. **State Stability**: The RAG engine components are tied to Streamlit's `st.session_state` cache. This pattern isolates core engine instantiation from daily interface reactive re-renders, preventing redundant embedding downloads and cross-encoder reallocations.
3. **Defensive Error Isolation**: The query optimization chain is enclosed inside dedicated try-except code blocks. If upstream network exceptions or API rate limits trigger, the module silently passes the original input forward to avoid total pipeline execution collapse.
