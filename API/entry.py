from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router as models_router

app = FastAPI(title="Trustworthy Model Registry", version="1.0.0")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

app.include_router(models_router)


@app.get("/")
def root():
    return {"message": "Trustworthy Model Registry API is running."}


@app.get("/health")
def health():
    return {"status": "good"}
