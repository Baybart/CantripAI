import torch
import torch.nn as nn

class Config:
    """
    n_emdb: The dimension of the embeddings
    n_positions: Maximum number of input tokens
    """
    def __init__(self, vocab_size, n_embd, n_layer, n_head, n_positions):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_positions = n_positions

class Embedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos = nn.Embedding(config.n_positions, config.n_embd)
    """
    idx: a 2D vector, of size B*T, where B is the batch size
    and T the lenght of the input sequences
    """
    def forward(self, idx):
        B, T = idx.shape
        positions = torch.arange(T, device=idx.device).unsqueeze(0)
        return self.token(idx) + self.pos(positions)


# Critical, needs to be written
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

# Critical, needs to be written
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

    
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 1st layer normalizer
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_emdb)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x