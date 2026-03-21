import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelConfig:
    vocab_size:  int = 256
    context_len: int = 256
    embed_dim:   int = 300
    num_heads:   int = 5
    num_layers:  int = 5
    dropout:     float = 0.12

class AttentionHead(nn.Module):
    def __init__(self, cfg: ModelConfig, head_dim: int):
        super().__init__()
        self.head_dim = head_dim

        self.query  = nn.Linear(cfg.embed_dim, head_dim, bias=False)
        self.key    = nn.Linear(cfg.embed_dim, head_dim, bias=False)
        self.value  = nn.Linear(cfg.embed_dim, head_dim, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.context_len, cfg.context_len))
        )

    def forward(self, x):
        B, T, C = x.shape

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        scale   = self.head_dim ** -0.5
        scores  = (q @ k.transpose(-2, -1)) * scale           # (B, T, T)
        scores  = scores.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)

        return weights @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        head_dim = cfg.embed_dim // cfg.num_heads
        self.heads   = nn.ModuleList([AttentionHead(cfg, head_dim) for _ in range(cfg.num_heads)])
        self.project = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.project(out))

class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim * 4),
            nn.GELU(),
            nn.Linear(cfg.embed_dim * 4, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = MultiHeadAttention(cfg)
        self.ff   = FeedForward(cfg)
        self.ln1  = nn.LayerNorm(cfg.embed_dim)
        self.ln2  = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class TinyLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_embed   = nn.Embedding(cfg.context_len, cfg.embed_dim)
        self.blocks      = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.ln_final    = nn.LayerNorm(cfg.embed_dim)
        self.head        = nn.Linear(cfg.embed_dim, cfg.vocab_size)

        self.head.weight = self.token_embed.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        tok = self.token_embed(idx)
        pos = self.pos_embed(torch.arange(T, device=device))
        x   = tok + pos
        x   = self.blocks(x)
        x   = self.ln_final(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8, top_k: int = 40):

        for _ in range(max_new_tokens):
            idx_ctx = idx[:, -self.cfg.context_len:]
            logits, _ = self(idx_ctx)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            idx    = torch.cat([idx, next_t], dim=1)

        return idx


def count_params(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    if total >= 1_000_000:
        return f"{total / 1_000_000:.1f}M"
    return f"{total / 1_000:.1f}K"
