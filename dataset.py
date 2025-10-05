import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = []
        self.max_length = max_length
        
        for text in tqdm(texts, desc="Tokenizing"):
            if len(text.strip()) < 10:
                continue
                
            encoded = tokenizer.encode(text)
            tokens = encoded.ids
            
            if len(tokens) < 5:
                continue
            
            if len(tokens) > max_length - 2:
                tokens = tokens[:max_length - 2]
            
            tokens = [2] + tokens + [3]
            
            if len(tokens) < max_length:
                tokens.extend([1] * (max_length - len(tokens)))
            else:
                tokens = tokens[:max_length]
            
            self.encodings.append(tokens)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        tokens = self.encodings[idx]
        
        input_ids = tokens[:-1]
        targets = tokens[1:]
        
        targets = [t if t != 1 else -100 for t in targets]
        
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def collate_fn(batch):
    input_ids, targets = zip(*batch)
    input_ids = torch.stack(input_ids)
    targets = torch.stack(targets)
    return input_ids, targets
