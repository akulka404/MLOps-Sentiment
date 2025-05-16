# app.py  (only the top part changes)
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import mlflow
from mlflow.tracking import MlflowClient
import pathlib, tempfile

MODEL_NAME = "distilbert-sst2"
TRACKING_DIR = pathlib.Path("/tmp/mlruns")          # local, works on Render

app = FastAPI(title="Mini-MLOps Sentiment API")

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str
    score: float

def load_or_register():
    """Try to load a registered model; if missing, create & register once."""
    mlflow.set_tracking_uri(f"file://{TRACKING_DIR}")
    client = MlflowClient()

    try:                       # already registered?
        mv = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
        return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{mv.version}")
    except Exception:
        # First boot: load HF pipeline and register
        clf = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1,          # CPU
        )

        # save temporary pyfunc
        with tempfile.TemporaryDirectory() as tmp:
            mlflow.pyfunc.save_model(
                path=tmp,
                python_model=mlflow.pyfunc.PythonModel.from_function(
                    predict=lambda self, ctx, inp: clf(inp),
                ),
                conda_env=None,
            )
            mv = client.create_model_version(
                name=MODEL_NAME,
                source=tmp,
                run_id=None,
            )
            client.transition_model_version_stage(
                name=MODEL_NAME, version=mv.version, stage="Production"
            )
        return clf              # return pipeline this time

model = load_or_register()

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    pred = model([payload.text])[0]        # works for both pyfunc & pipeline
    return PredictionOut(label=pred["label"], score=pred["score"])
