import os
import logging
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from typing import List, Any
from typing_extensions import TypedDict
from langchain.docstore.document import Document
from langgraph.checkpoint.memory import MemorySaver

# -----------------------------
# Setup
# -----------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
model = ChatOpenAI(model="gpt-4o-mini")

class GraphState(TypedDict, total=False):
    query: str
    db: Any
    retrieved_docs: List[Document]
    answer: str
    filter_doc: str


# PDF Loading

@traceable(name='load_pdfs')
def load_pdfs(uploaded_files, save_dir="./uploaded_pdfs") -> List[Document]:
    os.makedirs(save_dir, exist_ok=True)
    docs = []

    if not uploaded_files:
        logging.warning("No files provided for upload.")
        return docs

    for file in uploaded_files:
        try:
            if not file.name.lower().endswith('.pdf'):
                logging.warning(f"Skipped non-PDF file: {file.name}")
                continue

            file_path = os.path.join(save_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            if not loaded_docs:
                logging.warning(f"No content extracted from {file.name}")
            docs.extend(loaded_docs)
            logging.info(f"Loaded {len(loaded_docs)} pages from {file.name}")
        except Exception as e:
            logging.error(f"Failed to load {file.name}: {e}")

    return docs


# Build Vector Index

@traceable(name='build_index')
def build_index(docs: List[Document], persist_dir="./chroma_store") -> Chroma:
    if not docs:
        raise ValueError("No documents provided to build the index.")

    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
        db.persist()
        logging.info(f"Vector store built with {len(chunks)} chunks.")
        return db
    except Exception as e:
        logging.error(f"Failed to build vector store: {e}")
        raise e


# Retriever Node

@traceable(name='make_retriever_node')
def make_retriever_node(db: Chroma):
    def retriever_node(state: GraphState) -> GraphState:
        query = state.get("query", "")
        filter_doc = state.get("filter_doc", "")

        if not query:
            logging.warning("Empty query received.")
            return {"retrieved_docs": []}

        try:
            if filter_doc:
                docs = db.similarity_search(query, k=3, filter={"source": filter_doc})
            else:
                docs = db.similarity_search(query, k=4)
            logging.info(f"Retrieved {len(docs)} documents for query: {query}")
            return {"retrieved_docs": docs}
        except Exception as e:
            logging.error(f"Retriever failed: {e}")
            return {"retrieved_docs": []}

    return retriever_node


# Answer Node

@traceable(name='answer_node')
def answer_node(state: GraphState) -> GraphState:
    retrieved_docs = state.get('retrieved_docs', [])
    query = state.get('query', '')

    if not retrieved_docs:
        logging.warning("No documents retrieved; returning fallback answer.")
        return {"answer": "No relevant documents found."}

    try:
        context = "\n\n".join([
            f"Document: {d.metadata.get('source', 'unknown')}, Page {d.metadata.get('page', '?')}\n{d.page_content}"
            for d in retrieved_docs
        ])
        prompt = f"""
You are a helpful research assistant.
Answer the question using only the context below.
If the user asks about a specific paper, prioritize content from that document.

Context:
{context}

Question: {query}
"""
        response = model.invoke(prompt)
        return {"answer": response.content}
    except Exception as e:
        logging.error(f"Answer generation failed: {e}")
        return {"answer": "An error occurred while generating the answer."}


# Build LangGraph

@traceable(name='build_graph')
def build_graph(db: Chroma) -> StateGraph:
    try:
        graph = StateGraph(GraphState)
        graph.add_node('retriever', make_retriever_node(db))
        graph.add_node('answer', answer_node)

        graph.add_edge(START, 'retriever')
        graph.add_edge('retriever', 'answer')
        graph.add_edge('answer', END)

        checkpointer = MemorySaver()
        compiled_graph = graph.compile(checkpointer=checkpointer)
        logging.info("Graph compiled successfully.")
        return compiled_graph
    except Exception as e:
        logging.error(f"Failed to build graph: {e}")
        raise e


# Example Usage

if __name__ == "__main__":
    pass
