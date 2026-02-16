import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.preprocess import clean_text 
from app.config import LABEL_MAP

class SentimentInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = "abidrizmii/insightly.ai" 

    def load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()

    @torch.no_grad()
    def predict(self, text: str):
        self.load_model()
        cleaned = clean_text(text)
        inputs = self.tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()

        return {
            "label": LABEL_MAP[pred_id],
            "confidence": round(probs[0][pred_id].item(), 4),
            "probabilities": {LABEL_MAP[i]: round(p.item(), 4) for i, p in enumerate(probs[0])}
        }

    @torch.no_grad()
    def predict_batch(self, texts: list, batch_size: int = 16):
        self.load_model()
        all_labels = []
        for i in range(0, len(texts), batch_size):
            batch_texts = [clean_text(t) for t in texts[i : i + batch_size]]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(self.device)
            outputs = self.model(**inputs)
            pred_ids = torch.argmax(outputs.logits, dim=1).tolist()
            all_labels.extend([LABEL_MAP[pid] for pid in pred_ids])
        return all_labels