import json
import re
import random
import os
import unicodedata


def load_jsonl(filepath):
    texts = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                if 'text' in data:
                    texts.append(data['text'])
                else:
                    texts.append(str(data))
            except:
                continue
    return texts


def clean_text(text):
    if not text:
        return ""

    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    text = text.replace('\t', ' ')
    lines = [line.strip() for line in text.split('\n')]
    lines = [line for line in lines if line]
    return ' '.join(lines).strip()


def clean_corpus(texts):
    return [clean_text(t) for t in texts if clean_text(t)]


def split_data(texts, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    random.seed(seed)
    texts_copy = texts.copy()
    random.shuffle(texts_copy)

    n = len(texts_copy)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    return {
        'train': texts_copy[:train_end],
        'val': texts_copy[train_end:val_end],
        'test': texts_copy[val_end:]
    }


def save_splits(splits, output_dir, language):
    os.makedirs(output_dir, exist_ok=True)

    for split_name, texts in splits.items():
        filepath = os.path.join(output_dir, f"{language}_{split_name}.txt")
        with open(filepath, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')


def load_text_file(path, limit=None):
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                texts.append(line)
            if limit and i + 1 >= limit:
                break
    return texts