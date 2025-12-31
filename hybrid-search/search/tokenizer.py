import re

def tokenize(text: str):
    """
    Simple tokenizer:
    - lowercase
    - split on non-letters
    """
    text = text.lower()
    tokens = re.split(r"[^a-z]+", text)
    return [t for t in tokens if t]
