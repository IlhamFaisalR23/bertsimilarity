from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI()

model = None

class RequestModel(BaseModel):
    text: str

@app.get("/")
def root():
    return {"status": "ok"}

@app.post("/embedding")
def get_embedding(req: RequestModel):
    global model

    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    embedding = model.encode(req.text).tolist()
    return {"embedding": embedding}