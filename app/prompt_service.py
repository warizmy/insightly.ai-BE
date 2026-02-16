import google.generativeai as genai
import json
import re
from app.config import GEMINI_API_KEY

class GeminiService:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is missing!")
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    async def generate_business_report(self, stats: dict, samples: list):
        formatted_samples = "\n".join([f"- {s}" for s in samples])
        
        prompt = f"""
        Role: Senior Strategic Data Analyst.
        Analisis data sentimen berikut dari 1.500 feedback:
        - Statistik: Positive ({stats['Positive']}), Negative ({stats['Negative']}), Neutral ({stats['Neutral']})
        - Sampel Keluhan: {formatted_samples}
        Tugas: Identifikasi 5 masalah spesifik (Topik) yang paling mendesak.
        Output harus dalam format JSON ARRAY yang valid.
        Setiap objek harus memiliki kunci:
        1. "topic": Nama masalah spesifik (Misal: "Error QRIS", bukan "Gangguan Aplikasi").
        2. "urgency": (Critical, High, atau Medium).
        3. "evidence": Satu kutipan review asli paling relevan dari sampel.
        4. "percentage_estimate": Estimasi persentase kemunculan masalah ini dari total komplain (angka saja).
        5. "recommendation": Langkah teknis spesifik untuk tim IT.
        Pastikan output hanya JSON.
        """
        
        response = await self.model.generate_content_async(prompt)
        
        raw_text = response.text
        json_match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                return {"error": "Gagal parsing insight", "raw": raw_text}
        return {"error": "Format insight tidak sesuai", "raw": raw_text}