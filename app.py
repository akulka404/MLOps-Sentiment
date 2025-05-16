# app.py  –  35 clean lines
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Mini-MLOps Sentiment API")

# 1. load the pretrained model once (CPU-only)
sentiment_clf = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=-1,          # -1 = CPU
)

# 2. input / output schemas
class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str
    score: float

# 3. prediction endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    pred = sentiment_clf(payload.text)[0]     # returns dict
    return PredictionOut(label=pred["label"], score=pred["score"])
