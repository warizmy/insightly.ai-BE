# Insightly.ai - API
### Professional Sentiment Analysis & Strategic Business Insights API

**Insightly.ai** is a high-performance backend service designed to transform raw customer feedback into actionable business strategies. By combining a fine-tuned **IndoBERT** model for precision sentiment classification and **Gemini 2.5 Flash** for strategic intelligence, Insightly.ai provides deep-dive analytics that go beyond simple "Positive/Negative" labels.

---

## Key Features

- **High-Precision Sentiment Engine**: Fine-tuned IndoBERT-base-p2 with **0.79 Macro F1-Score**.
- **Strategic AI Insights**: Automated business recommendations and issue categorization powered by Gemini 2.5 Flash.
- **Efficient Batch Processing**: Optimized batch inference handling 1,500+ records in under 45 seconds.
- **Multi-Format Support**: Seamlessly process `.csv`, `.xlsx`, and `.xls` files.
- **Smart Column Detection**: Automatically identifies feedback columns (e.g., 'review', 'content', 'text').
- **Production Ready**: Equipped with rate limiting (SlowAPI), professional logging, and Docker containerization.

---

## Model Performance

The core classification engine was trained on a curated dataset of Indonesian customer reviews, achieving industry-standard benchmarks:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **0.7730** |
| **Macro F1-Score** | **0.7726** |
| **Precision (Neutral)** | **0.8325** |

> **Strategic Edge**: Unlike standard models, Insightly.ai excels at identifying "Neutral" sentiments and "Authentication Issues" with high precision, which are critical for fintech and service-based applications.

---

## Tech Stack

- **Framework**: FastAPI (Python)
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **LLM Integration**: Google Generative AI (Gemini 2.5 Flash)
- **Data Handling**: Pandas, OpenPyXL
- **Rate Limiting**: SlowAPI
- **Deployment**: Docker, Hugging Face Spaces

---

## Getting Started

### Prerequisites
- Python 3.12+
- Gemini API Key (Google AI Studio)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/abidrizmii/insightly.ai.git
   cd insightly.ai
   ```
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\\Scripts\\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Setup environment variables:
   Create a `.env` file and add:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

### Running the API
```bash
uvicorn main:app --reload
```

---

## API Endpoints

### 1. Single Prediction
`POST /predict`
Predict sentiment for a single string.

### 2. Batch Analysis (JSON)
`POST /analyze-batch`
Analyze a list of strings and generate strategic insights.

### 3. File Upload Analysis
`POST /analyze-upload` (Form-Data)
Upload a CSV or Excel file for mass analysis and full strategic reporting.

---

## Example Output
```json
{
    "status": "success",
    "results": {
        "statistics": { ... },
        "strategic_insights": [
            {
                "topic": "Authentication Issues",
                "urgency": "Critical",
                "evidence": "...",
                "recommendation": "..."
            }
        ]
    }
}
```
