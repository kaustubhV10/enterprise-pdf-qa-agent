# 📄 Enterprise PDF Q&A Agent

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-v1.30-orange.svg)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-purple.svg)](https://platform.openai.com/)

An **AI-powered document Q&A system** that lets users upload multiple PDF documents, query their content, and get accurate answers using GPT-4o-mini. Includes filtering by specific PDF and shows source citations.

---

## 🚀 Features  
- Upload and query **multiple PDFs** at once  
- **Semantic vector index** with OpenAI embeddings + Chroma  
- Accurate answers on **retrieved passages**  
- **Dropdown filter** to query specific PDFs  
- **Session-based memory** with `langgraph`  

---

## 🛠️ Tech Stack
- Python 3.10+  
- [Streamlit](https://streamlit.io) (Frontend)  
- [LangChain](https://www.langchain.com) (Embeddings & Retrieval)  
- [Chroma](https://www.trychroma.com/) (Vector Store)  
- [OpenAI GPT-4o-mini](https://platform.openai.com/docs/models/gpt-4o) (LLM)  
- [LangGraph](https://github.com/langgraph/langgraph) (Graph Orchestration)  

---

## ⚙️ Setup & Run 

```
# Clone repo
git clone https://github.com/your-username/pdf-qa-agent.git
cd pdf-qa-agent

# Create virtual environment (optional but recommended)
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=your_api_key" > .env

# Run Streamlit app
streamlit run app.py
```

---

## 📂 Project Structure 
pdf-qa-agent/  
├─ app.py              # Streamlit frontend  
├─ backend.py          # PDF loader, index builder, retriever & answer nodes  
├─ requirements.txt    # Dependencies  
├─ README.md           # Project documentation  
├─ uploaded_pdfs/      # Uploaded PDFs saved at runtime  
└─ chroma_store/       # Persistent vector DB (Chroma)
