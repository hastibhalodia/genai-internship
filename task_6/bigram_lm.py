import torch
import torch.nn as nn
import torch.nn.functional as F

# Load and prepare text
with open("tiny-shakespeare.txt", "r") as f:
    text = f.read()

# Character-level vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in enumerate(chars)}


# Encoding and decoding
def encode(s):
    return [stoi[c] for c in s]


def decode(indices):
    return "".join([itos[i] for i in indices])


# Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T = idx.shape
            logits = logits.view(B * T, vocab_size)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]  # Focus on last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
