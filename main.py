import io
import os
import asyncio
import pandas as pd
from collections import Counter
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
    return {
        "status": "ok",
        "model_loaded": model_inference.model is not None
    }

@app.post("/predict", response_model=PredictResponse)
@limiter.limit("120/minute")
def predict_sentiment(request: Request, payload: PredictRequest):
    if not payload.text.strip():
        return {
            "label": "Neutral", 
            "confidence": 0.0, 
            "probabilities": {"Positive": 0.0, "Neutral": 1.0, "Negative": 0.0}
        }
    
    return model_inference.predict(payload.text)

gemini_service = GeminiService()

@app.post("/analyze-batch")
@limiter.limit("30/minute")
async def analyze_batch(request: Request, texts: list[str]):
    
    if len(texts) > 200:
        raise HTTPException(400, "Max 200 texts per request")
    
    if any(len(t) > 2000 for t in texts):
        raise HTTPException(400, "Text too long (max 2000 chars each)")

    if not texts:
        raise HTTPException(400, "Empty input")

    results = []
    neg_samples = []
    
    tasks = [
        asyncio.to_thread(model_inference.predict, t)
        for t in texts
    ]
    responses = await asyncio.gather(*tasks)

    for res, t in zip(responses, texts):
        results.append(res['label'])
        if res['label'] == "Negative":
            neg_samples.append(t)
    
    total = len(results)
    if total == 0:
        return {"summary_stats": {}, "ai_insights": "", "total_processed": 0}
    
    c = Counter(results)
    stats = {
        "Positive": f"{(c['Positive']/total)*100:.1f}%",
        "Negative": f"{(c['Negative']/total)*100:.1f}%",
        "Neutral": f"{(c['Neutral']/total)*100:.1f}%"
    }

    try:
        insights = await asyncio.wait_for(
            gemini_service.generate_business_report(stats, neg_samples[:5]),
            timeout=20
        )
    except Exception:
        insights = "AI insights unavailable at the moment."

    return {
        "summary_stats": stats,
        "ai_insights": insights,
        "total_processed": total
    }

@app.post("/analyze-upload")
@limiter.limit("10/minute")
async def analyze_upload(request: Request, file: UploadFile = File(...)):
    
    filename = file.filename
    ext = filename.split('.')[-1].lower()

    if ext not in ['csv', 'xlsx', 'xls']:
        raise HTTPException(
            status_code=400, 
            detail="Format file tidak didukung. Harap unggah file dengan ekstensi .csv atau .xlsx."
        )

    if not file.filename:
        raise HTTPException(400, "Invalid file")

    try:
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:  # 5MB
            raise HTTPException(400, "File too large (max 5MB)")
        
        if ext == 'csv':
            try:
                df = pd.read_csv(io.BytesIO(contents), encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding='latin-1')
        else:
            df = pd.read_excel(io.BytesIO(contents))
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses file {filename}: {str(e)}")

    possible_cols = ['content', 'review', 'text', 'feedback', 'komentar', 'ulasan', 'comment']
    target_col = next((col for col in df.columns if col.lower() in possible_cols), df.columns[0])

    texts = df[target_col].dropna().astype(str).tolist()
    if not texts:
        raise HTTPException(status_code=400, detail="Data tidak ditemukan.")
    
    if len(texts) > 2000:
        raise HTTPException(400, "Max 2000 rows")

    results = model_inference.predict_batch(texts, batch_size=16)
    neg_samples = [texts[i] for i, label in enumerate(results) if label == "Negative"]
    
    total = len(results)
    if total == 0:
        return {"summary_stats": {}, "ai_insights": "", "total_processed": 0}
    
    c = Counter(results)
    stats = {
        "positive_count": c['Positive'],
        "negative_count": c['Negative'],
        "neutral_count": c['Neutral'],
        "distribution": {
            "Positive": f"{(c['Positive']/total)*100:.2f}%",
            "Negative": f"{(c['Negative']/total)*100:.2f}%",
            "Neutral": f"{(c['Neutral']/total)*100:.2f}%"
        }
    }
    
    try:
        strategic_insights = await asyncio.wait_for(
            gemini_service.generate_business_report(
                stats['distribution'], 
                neg_samples[:10]
            ),
            timeout=30
        )
    except asyncio.TimeoutError:
        strategic_insights = "AI insights generation timed out."

    return {
        "status": "success",
        "results": {
            "statistics": stats,
            "strategic_insights": strategic_insights
        }
    }