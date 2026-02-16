import io
import pandas as pd
from fastapi import UploadFile, File, HTTPException
from fastapi import FastAPI, Request
from app.schemas import PredictRequest, PredictResponse
from app.inference import SentimentInference
from app.prompt_service import GeminiService
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

gemini_service = GeminiService()

@app.post("/analyze-batch")
async def analyze_batch(texts: list[str]):
    results = []
    neg_samples = []
    
    for t in texts:
        res = model_inference.predict(t)
        results.append(res['label'])
        if res['label'] == "Negative":
            neg_samples.append(t)

    total = len(results)
    stats = {
        "Positive": f"{(results.count('Positive')/total)*100:.1f}%",
        "Negative": f"{(results.count('Negative')/total)*100:.1f}%",
        "Neutral": f"{(results.count('Neutral')/total)*100:.1f}%"
    }

    insights = await gemini_service.generate_business_report(stats, neg_samples[:5])

    return {
        "summary_stats": stats,
        "ai_insights": insights,
        "total_processed": total
    }

@app.post("/analyze-upload")
async def analyze_upload(file: UploadFile = File(...)):
    
    filename = file.filename
    ext = filename.split('.')[-1].lower()

    if ext not in ['csv', 'xlsx', 'xls']:
        raise HTTPException(
            status_code=400, 
            detail="Format file tidak didukung. Harap unggah file dengan ekstensi .csv atau .xlsx."
        )

    try:
        contents = await file.read()
        if ext == 'csv':
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses file {filename}: {str(e)}")

    possible_cols = ['content', 'review', 'text', 'feedback', 'komentar', 'ulasan', 'comment']
    target_col = next((col for col in df.columns if col.lower() in possible_cols), df.columns[0])

    texts = df[target_col].dropna().astype(str).tolist()
    if not texts:
        raise HTTPException(status_code=400, detail="Data tidak ditemukan.")

    results = model_inference.predict_batch(texts, batch_size=16)
    neg_samples = [texts[i] for i, label in enumerate(results) if label == "Negative"]
    
    total = len(results)
    stats = {
        "positive_count": results.count('Positive'),
        "negative_count": results.count('Negative'),
        "neutral_count": results.count('Neutral'),
        "distribution": {
            "Positive": f"{(results.count('Positive')/total)*100:.2f}%",
            "Negative": f"{(results.count('Negative')/total)*100:.2f}%",
            "Neutral": f"{(results.count('Neutral')/total)*100:.2f}%"
        }
    }

    strategic_insights = await gemini_service.generate_business_report(
        stats['distribution'], 
        neg_samples[:10]
    )

    return {
        "status": "success",
        "results": {
            "statistics": stats,
            "strategic_insights": strategic_insights
        }
    }