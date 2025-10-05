import torch
from datasets import load_dataset

def load_data(num_samples=50000):
    print("Loading dataset...")
    try:
        dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        texts = []
        for i, item in enumerate(dataset):
            if i >= num_samples:
                break
            text = item["text"]
            if len(text.strip()) > 100:
                texts.append(text)
        print(f"Loaded {len(texts)} text samples from OpenWebText")
    except Exception as e:
        print(f"OpenWebText failed: {e}")
        print("Falling back to TinyStories dataset...")
        dataset = load_dataset("roneneldan/TinyStories", split="train")
        texts = [item["text"] for item in dataset if len(item["text"].strip()) > 100]
        texts = texts[:num_samples]
        print(f"Loaded {len(texts)} text samples from TinyStories")
    
    return texts


def split_data(texts, train_ratio=0.9):
    train_size = int(train_ratio * len(texts))
    train_texts = texts[:train_size]
    val_texts = texts[train_size:]
    return train_texts, val_texts
