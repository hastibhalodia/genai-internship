import torch
import torch.nn.functional as F

def calculate_perplexity(model, text, stoi, device='cpu'):
    model.eval()
    with torch.no_grad():
        # Encode the text
        encoded = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
        x = encoded[:-1].unsqueeze(0)  # (1, T)
        y = encoded[1:].unsqueeze(0)   # (1, T)

        logits, _ = model(x)
        B, T, C = logits.shape

        loss = F.cross_entropy(logits.view(B * T, C), y.view(B * T))
        perplexity = torch.exp(loss).item()
        return perplexity
