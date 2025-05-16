import os
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Mini-MLOps Sentiment API")

sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,
)

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str
    score: float

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    pred = sentiment(payload.text)[0]
    return PredictionOut(label=pred["label"], score=pred["score"])

# ⬇︎ only addition ⬇︎
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))   # Render sets $PORT
    uvicorn.run("app:app", host="0.0.0.0", port=port)
