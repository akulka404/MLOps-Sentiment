# app.py
import mlflow.pyfunc
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import mlflow

MODEL_NAME = "distilbert-sst2"

app = FastAPI(title="Mini-MLOps Sentiment API")

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str
    score: float

def load_or_register():
    """Load HF model and push to local MLflow registry once."""
    if not mlflow.registered_model.get_registered_model(MODEL_NAME):
        clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        mlflow.pyfunc.log_model(
            artifact_path=MODEL_NAME,
            python_model=clf,
            registered_model_name=MODEL_NAME
        )
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/latest")
    return model

model = load_or_register()

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    pred = model.predict([payload.text])[0]     # returns dict
    return PredictionOut(label=pred["label"], score=pred["score"])

