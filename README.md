# Insightly.ai – Customer Feedback Intelligence API

Insightly.ai is a backend service for analyzing customer feedback using **Natural Language Processing (NLP)**. The API performs **sentiment classification** (Negative, Neutral, Positive) on textual reviews, such as Google Play Store reviews, to help businesses gain actionable insights from user feedback.

This service is built with **FastAPI** and powered by a **Transformer-based model (DistilBERT)**, integrated directly with the **Hugging Face Hub** for efficient model management.

---

## Key Features

- **Sentiment Analysis**: Classifies text into Negative, Neutral, and Positive.
- **Transformer-based**: Powered DistilBERT.
- **Cloud Optimized**: Decoupled model storage (Hugging Face) from application logic for faster builds and deployments.
- **Rate Limited**: Built-in protection against API abuse (60 req/min).
- **Auto-generated Docs**: Fully documented via Swagger UI.

---

## Model Overview

- **Task**: Sentiment Classification
- **Labels**: Negative, Neutral, Positive
- **Architecture**: DistilBERT
- **Framework**: Hugging Face Transformers
- **Training Source**: Google Play Store Reviews. You can accsess [here](https://www.kaggle.com/datasets/prakharrathi25/google-play-store-reviews).

### Evaluation Summary

| Metric | Value |
|------|------|
| Accuracy | 75% |
| Macro F1-score | 0.66 |
| Weighted F1-score | 0.75 |

> Note: The Neutral class remains challenging due to class imbalance, which can be improved in future iterations.

---

## Getting Started (Local)

### 1. Clone Repository

```bash
git clone https://github.com/warizmy/insightly-api.git
cd insightly-api
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Server

```bash
uvicorn main:app --reload
```

API will be available at:
```
http://127.0.0.1:8000
```

Swagger UI:
```
http://127.0.0.1:8000/docs
```

---

## API Usage

### Health Check

**GET** ```/health```

Response:
```json
{ "status": "ok" }
```

---

### Sentiment Prediction

**POST** ```/predict```

Request Body:
```json
{
  "text": "This app is very useful and easy to use"
}
```

Response:
```json
{
  "label": "Positive",
  "confidence": 0.8942,
  "probabilities": {
    "Positive": 0.8942,
    "Neutral": 0.0821,
    "Negative": 0.0237
  }
}

```

## Deployment

This API is Docker-ready and optimized for **Railway** or **Render**.

**Pro Tip**: The model is automatically fetched from the Hugging Face Hub during the application's lifespan. **No need to include heavy model weights in your Git repository**.

### Start Command (Railway)

```bash
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
```

---

## API Policy
To ensure high availability, this API implements a Rate Limiting policy:
- **Limit**: 60 requests per minute per IP.
- **Exceeding limit**: Returns a ```429 Too Many Requests``` status code.
