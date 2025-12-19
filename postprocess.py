import re

def clean_text(text):
    if not text:
        return ""

    cleaned = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if len(line) > 1:
            cleaned.append(line)
