from fastapi import FastAPI, UploadFile, File
from pathlib import Path

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

    return {
        "filename": file.filename,
        "saved_to": str(temp_path),
        "status": "received"
    }