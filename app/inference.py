import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.preprocess import clean_text 

class SentimentInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cpu") 
        self.model_name = "abidrizmii/insightly-sentiment" 

    def load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully from Hugging Face!")

    @torch.no_grad()
    def predict(self, text: str):
        self.load_model()
        
        text = clean_text(text)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()

        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

        return {
            "label": label_map[pred_id],
            "confidence": round(probs[0][pred_id].item(), 4),
            "probabilities": {label_map[i]: round(p.item(), 4) for i, p in enumerate(probs[0])}
        }