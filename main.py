from fastapi import FastAPI, Request
from app.schemas import PredictRequest, PredictResponse
from app.inference import SentimentInference
from contextlib import asynccontextmanager
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

model_inference = SentimentInference()
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_inference.load_model()
    yield

app = FastAPI(
    title="Insightly.ai API",
    version="1.0.0",
    description="Professional Sentiment Analysis API for Customer Feedback",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
@limiter.limit("60/minute")
def predict_sentiment(payload: PredictRequest, request: Request):
    if not payload.text.strip():
        return {
            "label": "Neutral", 
            "confidence": 0.0, 
            "probabilities": {"Positive": 0.0, "Neutral": 1.0, "Negative": 0.0}
        }
    
    return model_inference.predict(payload.text)