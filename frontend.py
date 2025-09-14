import streamlit as st
from backend import load_pdfs, build_index, build_graph
from uuid import uuid4
import logging


# Page setup

st.set_page_config(page_title="PDF Q&A Agent", layout='wide')
st.title("Enterprise PDF Q&A Agent")
logging.basicConfig(level=logging.INFO)


# PDF Upload

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    accept_multiple_files=True,
    type='pdf'
)

if uploaded_files:
    try:
        with st.spinner("Loading PDFs..."):
            docs = load_pdfs(uploaded_files)
        if not docs:
            st.warning("No valid PDF content was loaded. Please check your files.")
            st.stop()

        with st.spinner("Building vector index..."):
            db = build_index(docs)

        with st.spinner("Initializing Q&A agent..."):
            agent = build_graph(db)

        pdf_names = sorted(list({d.metadata.get("source", "unknown") for d in docs}))
        selected_pdf = st.selectbox("Filter by document (optional)", ["All"] + pdf_names)

    except Exception as e:
        logging.error(f"Error during setup: {e}")
        st.error(f"An error occurred: {e}")
        st.stop()
else:
    st.info("Please upload PDF files to continue.")
    st.stop()

# Session state

if "thread_id" not in st.session_state:
    st.session_state['thread_id'] = str(uuid4())


# Query input

query = st.text_input("Ask a question about your PDFs")

if st.button("Get Answer") and query:
    try:
        with st.spinner("Retrieving answer..."):
            result = agent.invoke(
                {
                    'query': query,
                    "filter_doc": None if selected_pdf == "All" else selected_pdf
                },
                config={'configurable': {"thread_id": st.session_state["thread_id"]}}
            )

        # Show retrieved documents
        retrieved_docs = result.get('retrieved_docs', [])
        
        # Show final answer
        st.subheader("Answer")
        st.write(result.get("answer", "No answer returned."))

    except Exception as e:
        logging.error(f"Error during Q&A: {e}")
        st.error(f"Failed to get answer: {e}")
elif query:
    st.info("Click 'Get Answer' to retrieve the answer from your PDFs.")