from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse
from app.inference import SentimentInference

app = FastAPI(
    title="Insightly.ai API",
    version="1.0",
    description="Customer Feedback Sentiment Analysis API"
)

model = None

@app.on_event("startup")
def load_model():
    global model
    model = SentimentInference()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(payload: PredictRequest):
    if not payload.text.strip():
        return {"label": "neutral", "confidence": 0.0}

    return model.predict(payload.text)
