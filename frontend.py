import streamlit as st
from backend import load_pdfs, build_index, build_graph
from uuid import uuid4

st.set_page_config(page_title="PDF Q&A Agent", layout='wide')
st.title("Enterprise PDF Q&A agent")

uploaded_files = st.file_uploader("Upload PDF documents", accept_multiple_files=True, type='pdf')

if uploaded_files:
    docs = load_pdfs(uploaded_files)
    db = build_index(docs)
    agent = build_graph(db)

    pdf_names = sorted(list({d.metadata.get("source","unknown") for d in docs}))
    selected_pdf = st.selectbox("filter by document (optional)", ["All"] + pdf_names)


    if "thread_id" not in st.session_state:
        st.session_state['thread_id'] = str(uuid4())

    query = st.text_input("Ask a question about your PDFs")

    if st.button("Get Answer") and query:
        result = agent.invoke(
            {'query':query, "filter_doc": None if selected_pdf == "All" else selected_pdf},
            config={'configurable':{"thread_id": st.session_state["thread_id"]}})
        
        #show retrieved docs (if any)
        if 'retrieved_docs' in result and result['retrieved_docs']:

            # show final answer
            st.subheader("Answer")
            st.write(result["answer"])