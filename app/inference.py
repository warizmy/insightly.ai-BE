import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained("models/")
            self.model = AutoModelForSequenceClassification.from_pretrained("models/")
            self.model.to(self.device)
            self.model.eval()

    @torch.no_grad()
    def predict(self, text: str):
        self.load_model()

        text = clean_text(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        ).to(self.device)

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()

        label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

        return {
            "label": label_map[pred_id],
            "confidence": round(probs[0][pred_id].item(), 4)
        }
