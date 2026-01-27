from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse
from app.inference import SentimentInference
from contextlib import asynccontextmanager

model_inference = SentimentInference()

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_inference.load_model()
    yield

app = FastAPI(
    title="Insightly.ai API",
    lifespan=lifespan
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(payload: PredictRequest):
    if not payload.text.strip():
        return {"label": "Neutral", "confidence": 0.0, "probabilities": {}}
    
    return model_inference.predict(payload.text)