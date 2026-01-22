# Insightly.ai – Customer Feedback Intelligence API

Insightly.ai is a backend service for analyzing customer feedback using **Natural Language Processing (NLP)**. The API performs **sentiment classification** (Negative, Neutral, Positive) on textual reviews, such as Google Play Store reviews, to help businesses gain actionable insights from user feedback.

This service is built with **FastAPI** and powered by a **Transformer-based model (DistilBERT)** fine-tuned on real-world review data.

---

## ✨ Key Features

- 🔍 Sentiment analysis for customer reviews
- 🤖 Transformer-based NLP model (DistilBERT)
- ⚡ Fast and lightweight FastAPI backend
- 📦 Ready-to-deploy for cloud platforms (Render, Railway, etc.)
- 🌐 RESTful API, easy to integrate with any frontend

---

## 🧠 Model Overview

- **Task**: Sentiment Classification
- **Labels**: Negative, Neutral, Positive
- **Architecture**: DistilBERT
- **Framework**: Hugging Face Transformers
- **Training Environment**: Google Colab

### Evaluation Summary

| Metric | Value |
|------|------|
| Accuracy | 75% |
| Macro F1-score | 0.66 |
| Weighted F1-score | 0.75 |

> Note: The Neutral class remains challenging due to class imbalance, which can be improved in future iterations.

---

## 🗂 Project Structure

```
backend/
│
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── inference.py     # Model loading & prediction logic
│   └── schemas.py       # Request & response schemas
│
├── models/              # Trained model & tokenizer files
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   ├── vocab.txt
│   ├── tokenizer_config.json
│   └── special_tokens_map.json
│
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started (Local)

### 1. Clone Repository

```bash
git clone https://github.com/your-username/insightly-api.git
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
uvicorn app.main:app --reload
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

## 📡 API Usage

### Health Check

**GET /**

Response:
```json
{ "status": "ok" }
```

---

### Sentiment Prediction

**POST /predict**

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
  "confidence": 0.89
}
```

---

## 🧪 Testing with Postman

1. Set method to **POST**
2. URL: `/predict`
3. Headers:
   - `Content-Type: application/json`
4. Body → raw → JSON

---

## ☁️ Deployment

This API is optimized for deployment on platforms like **Render**.

### Recommended Start Command

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Ensure that the `models/` directory is included in the deployment.

---

## 🔮 Future Improvements

- Improve Neutral class performance
- Add batch prediction endpoint
- Aspect-based sentiment analysis
- Topic modeling for feedback clustering
- Authentication & rate limiting

---

## 👨‍💻 Author

**Abid Rizmi**  
Computer Science Graduate  
Focused on AI, Machine Learning, and Data-Driven Systems

---

## 📄 License

This project is intended for educational and portfolio purposes.

