import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI

# --- Initialize Core Components ---
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
load_dotenv()
collection = chroma_client.create_collection("compliance_docs")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------------------------
# 📌 STEP 1: Document Ingestion
# ------------------------------
def process_pdf_and_store(file_bytes: bytes, filename: str):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    all_text = ""
    for page in doc:
        all_text += page.get_text("text")

    # Chunk text for embeddings
    chunks = [all_text[i:i+500] for i in range(0, len(all_text), 500)]

    # Embed chunks
    embeddings = embedding_model.encode(chunks).tolist()

    # Store chunks + embeddings in ChromaDB
    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[emb],
            ids=[f"{filename}_{idx}"]
        )

    return len(chunks)


# ------------------------------
# 📌 STEP 2: Query with RAG
# ------------------------------
def query_with_rag(question: str):
    # --- R (Retrieval) ---
    q_embedding = embedding_model.encode([question]).tolist()[0]
    results = collection.query(query_embeddings=[q_embedding], n_results=3)
    retrieved_chunks = results.get("documents", [[]])[0]

    # --- A (Augmentation) ---
    context = "\n\n".join(retrieved_chunks)
    prompt = f"""
    You are a compliance assistant. 
    Use ONLY the following document text to answer the question.

    Context:
    {context}

    Question:
    {question}
    """

    # --- G (Generation) ---
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content

    return {"answer": answer, "context_used": retrieved_chunks}
