from fastapi import FastAPI, UploadFile, File
from pathlib import Path
from pypdf import PdfReader

app = FastAPI(title="RAG PDF Compliance Assistant", version="0.0.1")

@app.get("/")
def root():
    return {"message": "Backend is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # Save file temporarily (later we’ll push to vector store)
    temp_path = Path("uploads") / file.filename
    temp_path.parent.mkdir(exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Extract text from PDF
    reader = PdfReader(str(temp_path))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() or ""

    # Save extracted text locally (later we’ll vectorize)
    text_path = Path("uploads") / f"{file.filename}.txt"
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    return {
        "filename": file.filename,
        "saved_to": str(temp_path),
        "text_saved_to": str(text_path),
        "chars_extracted": len(full_text)
    }