from fastapi import FastAPI

app = FastAPI(title="RAG PDF Compliance Assistant", version="0.0.1")

@app.get("/")
def root():
    return {"message": "Backend is running 🚀"}

@app.get("/health")
def health():
    return {"status": "ok"}
