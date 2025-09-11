import os
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


load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

class GraphState(TypedDict, total=False):
    query: str
    db: Any
    retrieved_docs: List[Document]
    answer: str

@traceable(name='load_pdfs')
def load_pdfs(uploaded_files, save_dir = "./uploaded_pdfs"):
    os.makedirs(save_dir, exist_ok=True)
    docs = []
    for file in uploaded_files:
        file_path = os.path.join(save_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())
    return docs


@traceable(name='build_index')
def build_index(docs, persist_dir = "./chroma_store"):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    chunks = splitter.split_documents(docs)
    embeddings =OpenAIEmbeddings()
    db = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
    return db

@traceable(name='make_retriever_node')
def make_retriever_node(db):
    def retriever_node(state: GraphState) -> GraphState:
        query = state["query"]
        docs = db.similarity_search(query, k=4)
        return {"retrieved_docs": docs}
    return retriever_node

@traceable(name='answer_node')
def answer_node(state: GraphState) -> GraphState:
    context = "\n".join([d.page_content for d in state['retrieved_docs']])
    prompt = f"Answer based only on: \n{context}\n\nQuestion: {state['query']}"
    response = model.invoke(prompt)
    return {"answer": response.content}


@traceable(name='build_graph')
def build_graph(db):
    graph = StateGraph(GraphState)
    graph.add_node('retriever', make_retriever_node(db))
    graph.add_node('answer', answer_node)

    graph.add_edge(START, 'retriever')
    graph.add_edge('retriever', 'answer')
    graph.add_edge('answer', END)

    checkpointer = MemorySaver()

    return graph.compile(
    checkpointer=checkpointer)
