# GPT from Scratch
A clean PyTorch implementation of GPT (Generative Pre-trained Transformer) built from the ground up. This project focuses on understanding transformer architecture fundamentals rather than achieving state-of-the-art results.
## Why This 
I wanted to really understand how GPT works under the hood. Reading papers is one thing, but implementing multi-head attention, positional encodings, and the training loop yourself hits different. This repo is the result of that learning process.
## What's Inside
- **Clean transformer implementation** - No shortcuts, everything built from scratch
- **BPE tokenization** - Proper subword tokenization with 50k vocab
- **Proper training loop** - Learning rate warmup, gradient clipping, validation
- **Text generation** - Top-k and nucleus sampling for decent outputs


## Model Architecture

- 6 transformer layers
- 8 attention heads
- 512 embedding dimensions
- 2048 feedforward dimensions
- Max sequence length: 512 tokens

The architecture is intentionally kept small so you can actually train it without needing a GPU cluster.
## Training Details
The training script handles everything:

- Loads 50k text samples (configurable)
- Trains a BPE tokenizer from scratch
- Splits data into train/val
- Trains with AdamW optimizer
- Uses cosine learning rate schedule with warmup
- Saves best checkpoint based on validation loss

You can watch the loss drop and see the model start generating coherent text after a few epochs.

## What I Learned
Building this taught me way more than reading papers:

- How attention actually computes context
- Why layer norm placement matters
- The importance of proper weight initialization
- How tokenization affects model quality
- The entire working mechanism of a GPT model.


## Limitations
This is a learning project, not production code:

- Small model size means limited capacity
- Training data is limited
- No fancy optimizations like Flash Attention
- Generation can be repetitive

But it works, it's clean, and you can understand every line.

## Acknowledgments
Inspired by Andrej Karpathy's nanoGPT and the original "Attention is All You Need" paper. Built this while reading through both multiple times.
