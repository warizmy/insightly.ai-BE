import re

SLANG_DICT = {
    'gak': 'tidak', 'ga': 'tidak', 'gk': 'tidak', 'tdk': 'tidak',
    'yg': 'yang', 'bgt': 'banget', 'apk': 'aplikasi', 'nha': 'nya',
    'udh': 'sudah', 'udah': 'sudah', 'sdh': 'sudah',
    'krn': 'karena', 'knp': 'kenapa', 'cmn': 'cuma'
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    # Slang Normalization
    words = text.split()
    fixed_words = [SLANG_DICT.get(w, w) for w in words]
    text = " ".join(fixed_words)
    
    text = re.sub(r"\s+", " ", text).strip()
    return text