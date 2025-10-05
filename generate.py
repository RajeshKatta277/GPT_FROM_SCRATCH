import torch
from model import GPTModel
from tokenizer_utils import load_tokenizer
from inference import generate_text

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model_and_tokenizer(checkpoint_path='best_gpt_model.pth'):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    tokenizer = load_tokenizer('gpt_tokenizer.json')
    
    model = GPTModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        d_ff=config["d_ff"],
        max_seq_len=config["max_seq_len"],
        dropout=0.0
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer


def interactive_generation():
    model, tokenizer = load_model_and_tokenizer()
    
    print("\n" + "="*60)
    print("INTERACTIVE TEXT GENERATION")
    print("="*60)
    print("Type your prompt and press Enter. Type 'quit' to exit.\n")
    
    while True:
        prompt = input("Prompt: ").strip()
        
        if prompt.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not prompt:
            continue
        
        print("\nGenerating...")
        generated = generate_text(model, tokenizer, prompt, max_length=150, temperature=0.8, device=device)
        print(f"\n{generated}\n")
        print("-" * 60)


if __name__ == "__main__":
    interactive_generation()
