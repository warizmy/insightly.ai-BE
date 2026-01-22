import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config import MODEL_PATH, LABEL_MAP
from app.preprocess import clean_text

class SentimentInference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text: str):
        text = clean_text(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)[0]

        probs = probs.cpu().numpy()
        pred_id = probs.argmax()

        return {
            "label": LABEL_MAP[pred_id],
            "confidence": float(probs[pred_id]),
            "probabilities": {
                LABEL_MAP[i]: float(probs[i]) for i in range(len(probs))
            }
        }
