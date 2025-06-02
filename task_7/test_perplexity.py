import torch
import torch.nn.functional as F
from torchmetrics.text import Perplexity
import sys
import os
import random

# Ensure task_6 directory is accessible for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'task_6')))
from bigram_lm import BigramLanguageModel, get_batch, vocab_size, encode, decode

# Set random seeds for reproducibility
torch.manual_seed(42)
random.seed(42)

# Load dataset
with open("task_6/tiny-shakespeare.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Train/Val split
n = int(0.9 * len(text))
train_text = text[:n]
val_text = text[n:]

# Encode the text
train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 500
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model
model = BigramLanguageModel(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for step in range(max_iters):
    if step % eval_interval == 0:
        model.eval()
        with torch.no_grad():
            xb_train, yb_train = get_batch(train_data, batch_size, block_size, device)
            _, train_loss = model(xb_train, yb_train)

            xb_val, yb_val = get_batch(val_data, batch_size, block_size, device)
            _, val_loss = model(xb_val, yb_val)

        print(f"step {step}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
        model.train()

    xb, yb = get_batch(train_data, batch_size, block_size, device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Final perplexity calculation
model.eval()
with torch.no_grad():
    xb, yb = get_batch(val_data, batch_size, block_size, device)
    logits, _ = model(xb)  # Do NOT pass yb to avoid flattening logits
    log_probs = F.log_softmax(logits, dim=-1)  # Shape: [B, T, V]

    metric = Perplexity()
    metric.update(log_probs.cpu(), yb.cpu())  # yb shape: [B, T]
    perplexity = metric.compute().item()

print(f"Perplexity (torchmetrics): {perplexity:.4f}")
