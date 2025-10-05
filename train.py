import torch
import json
import math
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import GPTModel
from dataset import TextDataset, collate_fn
from tokenizer_utils import train_tokenizer
from data_loader import load_data, split_data
from inference import generate_text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def get_lr(step, warmup_steps, total_steps, learning_rate):
    if step < warmup_steps:
        return learning_rate * step / warmup_steps
    return learning_rate * 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))


def train_epoch(model, dataloader, optimizer, scheduler, device, grad_clip):
    model.train()
    total_loss = 0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch_idx, (input_ids, targets) in enumerate(progress_bar):
        input_ids, targets = input_ids.to(device), targets.to(device)

        optimizer.zero_grad()

        logits, loss = model(input_ids, targets)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}',
            'lr': f'{scheduler.get_last_lr()[0]:.6f}'
        })

    return total_loss / num_batches


def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for input_ids, targets in tqdm(dataloader, desc="Validating"):
            input_ids, targets = input_ids.to(device), targets.to(device)
            logits, loss = model(input_ids, targets)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    config = {
        "vocab_size": 50257,
        "d_model": 512,
        "num_heads": 8,
        "num_layers": 6,
        "d_ff": 2048,
        "max_seq_len": 512,
        "batch_size": 8,
        "learning_rate": 5e-4,
        "num_epochs": 20,
        "dropout": 0.1,
        "warmup_steps": 1000,
        "grad_clip": 1.0,
        "num_samples": 50000
    }

    texts = load_data(num_samples=config["num_samples"])
    
    train_texts, val_texts = split_data(texts, train_ratio=0.9)
    
    print("Training tokenizer...")
    tokenizer = train_tokenizer(train_texts, vocab_size=config["vocab_size"])
    config["vocab_size"] = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {config['vocab_size']}")
    
    tokenizer.save('gpt_tokenizer.json')
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Creating model...")
    model = GPTModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=config["dropout"]
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("Creating datasets...")
    train_dataset = TextDataset(train_texts, tokenizer, config["max_seq_len"])
    val_dataset = TextDataset(val_texts, tokenizer, config["max_seq_len"])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True,
        collate_fn=collate_fn, 
        drop_last=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config["learning_rate"],
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = config["warmup_steps"]
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: get_lr(step, warmup_steps, total_steps, config["learning_rate"]) / config["learning_rate"]
    )
    
    print("\nStarting training...")
    best_val_loss = float('inf')
    
    for epoch in range(config["num_epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        print(f"{'='*60}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, config["grad_clip"])
        val_loss = validate(model, val_loader, device)
        
        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config
            }, 'best_gpt_model.pth')
            print("✓ Saved best model")
    
    torch.save({
        'epoch': config["num_epochs"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config
    }, 'final_gpt_model.pth')
    
    print("\n" + "="*60)
    print("TESTING MODEL - GENERATING SAMPLES")
    print("="*60)
    
    prompts = [
        "The future of artificial intelligence",
        "In the world of technology,",
        "The most important thing to remember is",
        "Scientists have recently discovered",
        "Once upon a time in a distant land,"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        print("-" * 60)
        generated = generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8, device=device)
        print(f"Generated: {generated}")
        print()
    
    print("="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Files saved:")
    print("  - best_gpt_model.pth")
    print("  - final_gpt_model.pth")
    print("  - gpt_tokenizer.json")
    print("  - config.json")


if __name__ == "__main__":
    main()
