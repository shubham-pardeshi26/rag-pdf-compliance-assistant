from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI(title="RAG PDF Compliance Assistant", version="0.0.1")

# initialize embedding model + chroma client
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="vectorstore")
collection = chroma_client.get_or_create_collection(name="pdf_docs")

@app.get("/")
def root():
    return {"message": "Backend is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    temp_path = Path("uploads") / file.filename
    temp_path.parent.mkdir(exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Extract text from PDF
    reader = PdfReader(str(temp_path))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    # Chunk text (basic split)
    chunks = [full_text[i:i+500] for i in range(0, len(full_text), 500)]

    # Embed and store in Chroma
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{file.filename}_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    return {
        "filename": file.filename,
        "chunks_stored": len(chunks)
    }


@app.post("/query")
async def query(question: str):
    # Embed the question
    q_embedding = embedding_model.encode([question]).tolist()[0]

    # Search in vector DB
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=3  
    )

    # Extract documents
    retrieved_chunks = results.get("documents", [[]])[0]

    return {
        "question": question,
        "top_chunks": retrieved_chunks
    }
