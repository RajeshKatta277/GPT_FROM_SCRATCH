from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer(texts, vocab_size=50257):
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<UNK>", "<PAD>", "<BOS>", "<EOS>"],
        min_frequency=2
    )

    tokenizer.train_from_iterator(texts, trainer)
    tokenizer.enable_padding(pad_id=1, pad_token="<PAD>")
    tokenizer.enable_truncation(max_length=512)
    
    return tokenizer


def load_tokenizer(path):
    return Tokenizer.from_file(path)

