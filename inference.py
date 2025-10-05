import torch
import torch.nn.functional as F

def generate_text(model, tokenizer, prompt, max_length=200, temperature=0.8, top_k=50, top_p=0.95, device='cuda'):
    model.eval()
    
    tokens = tokenizer.encode(prompt).ids
    tokens = [2] + tokens
    
    generated = tokens.copy()
    
    with torch.no_grad():
        for _ in range(max_length):
            input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
            
            if input_ids.size(1) > model.max_seq_len:
                input_ids = input_ids[:, -model.max_seq_len:]
            
            logits, _ = model(input_ids)
            logits = logits[0, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            
            if next_token == 3:
                break
            
            tokens.append(next_token)
            generated.append(next_token)
    
    generated_text = tokenizer.decode(generated)
    return generated_text
