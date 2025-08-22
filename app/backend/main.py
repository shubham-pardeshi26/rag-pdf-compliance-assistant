from fastapi import FastAPI, UploadFile, File,HTTPException
from pydantic import BaseModel
from .rag_helpers import process_pdf_and_store, query_with_rag


# Initialize FastAPI app
app = FastAPI(title="PDF Compliance Assistant (RAG)")

# ------------------------------
# 📌 Upload PDF Endpoint
# ------------------------------


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    # Ensure it's a PDF
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_bytes = await file.read()
    try:
        num_chunks = process_pdf_and_store(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

    return {"message": f"Stored {num_chunks} chunks from {file.filename}"}


# ------------------------------
# 📌 Query Endpoint
# ------------------------------
class QueryRequest(BaseModel):
    question: str

@app.post("/query")
async def query_docs(req: QueryRequest):
    result = query_with_rag(req.question)
    return result
