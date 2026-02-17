import io
import os
import asyncio
import pandas as pd
from datetime import datetime
from collections import Counter
from contextlib import asynccontextmanager
from fastapi import UploadFile, File, HTTPException, FastAPI, Request
from app.schemas import PredictRequest, PredictResponse
from app.inference import SentimentInference
from app.prompt_service import GeminiService
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
    if not texts:
        raise HTTPException(400, "Input teks kosong, harap masukkan data.")
        
    if len(texts) > 200:
        raise HTTPException(400, "Maksimal 200 teks per permintaan.")
    
    if any(len(t) > 2000 for t in texts):
        raise HTTPException(400, "Teks terlalu panjang (maksimal 2000 karakter per teks).")

    try:
        results = model_inference.predict_batch(texts, batch_size=16)
    except Exception as e:
        raise HTTPException(500, f"Gagal memproses model: {str(e)}")

    neg_samples = [texts[i] for i, label in enumerate(results) if label == "Negative"]
    
    total = len(results)
    c = Counter(results)
    
    stats = {
        "Positive": f"{(c['Positive']/total)*100:.1f}%",
        "Negative": f"{(c['Negative']/total)*100:.1f}%",
        "Neutral": f"{(c['Neutral']/total)*100:.1f}%"
    }

    try:
        batch_insights = await asyncio.wait_for(
            gemini_service.generate_quick_summary(stats, texts[:15]),
            timeout=30
        )
    except asyncio.TimeoutError:
        batch_insights = {"error": "AI analysis timed out", "summary": "Layanan AI sedang sibuk."}
    except Exception:
        batch_insights = {"error": "AI unavailable", "summary": "Analisis AI tidak tersedia saat ini."}

    return {
        "status": "success",
        "total_processed": total,
        "sentiment_stats": stats,
        "quick_analysis": batch_insights
    }

@app.post("/analyze-upload")
@limiter.limit("10/minute")
async def analyze_upload(request: Request, file: UploadFile = File(...)):
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="File tidak valid atau nama file kosong.")

    ext = filename.split('.')[-1].lower()
    if ext not in ['csv', 'xlsx', 'xls']:
        raise HTTPException(
            status_code=400, 
            detail="Format file tidak didukung. Harap unggah file .csv atau .xlsx."
        )

    try:
        contents = await file.read()
        if len(contents) > 5 * 1024 * 1024:  # 5MB
            raise HTTPException(status_code=400, detail="File terlalu besar. Maksimal ukuran file adalah 5MB.")
        
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
        raise HTTPException(status_code=500, detail=f"Gagal membaca file {filename}: {str(e)}")

    possible_cols = ['content', 'review', 'text', 'feedback', 'komentar', 'ulasan', 'comment']
    target_col = next((col for col in df.columns if col.lower() in possible_cols), df.columns[0])

    texts = df[target_col].dropna().astype(str).tolist()
    if not texts:
        raise HTTPException(status_code=400, detail="Tidak ada data teks yang ditemukan.")
    
    if len(texts) > 2000:
        raise HTTPException(status_code=400, detail="Maksimal 2000 baris teks per file.")

    results = model_inference.predict_batch(texts, batch_size=16)
    
    neg_samples = [texts[i] for i, label in enumerate(results) if label == "Negative"]
    total = len(results)
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
            timeout=45
        )
    except asyncio.TimeoutError:
        strategic_insights = {"error": "Analisis AI memakan waktu terlalu lama. Silakan coba lagi."}
    except Exception as e:
        strategic_insights = {"error": f"Layanan AI tidak tersedia: {str(e)}"}

    return {
        "status": "success",
        "metadata": {
            "filename": filename,
            "processed_at": datetime.now().isoformat(),
            "detected_column": target_col,
            "total_records": total
        },
        "results": {
            "statistics": stats,
            "strategic_insights": strategic_insights
        }
    }