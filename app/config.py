import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_PATH = "./models"
LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}