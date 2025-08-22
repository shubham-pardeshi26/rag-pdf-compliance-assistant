import streamlit as st
import requests

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="RAG PDF Compliance Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 RAG PDF Compliance Assistant")
st.write(
    "Upload compliance-related PDF documents (e.g., RBI Guidelines, HR Policies) "
    "and ask questions. This assistant uses **RAG (Retrieval-Augmented Generation)** "
    "to provide context-aware answers."
)

# -------------------------
# API Base URL
# -------------------------
API_URL = "http://127.0.0.1:8000"  # change if FastAPI runs elsewhere

# -------------------------
# Sidebar: Upload PDF
# -------------------------
st.sidebar.header("📂 Document Upload")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("📤 Uploading and processing PDF..."):
        files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
        response = requests.post(f"{API_URL}/upload_pdf", files=files)

    if response.status_code == 200:
        st.sidebar.success(response.json()["message"])
    else:
        st.sidebar.error("❌ Failed to upload file")

# -------------------------
# Main Area: Ask Questions
# -------------------------
st.header("💬 Ask Compliance Questions")

question = st.text_input("Enter your question:")

if st.button("Get Answer") and question:
    with st.spinner("🤔 Searching and generating answer..."):
        response = requests.post(
            f"{API_URL}/query",
            json={"question": question}
        )

    if response.status_code == 200:
        result = response.json()
        st.subheader("✅ Answer")
        st.write(result["answer"])

        with st.expander("📄 Context Used"):
            for idx, chunk in enumerate(result["context_used"], 1):
                st.markdown(f"**Chunk {idx}:** {chunk}")
    else:
        st.error("❌ Something went wrong while querying.")
