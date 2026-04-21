import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRecBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(
            x,
            x,
            x,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = residual + self.dropout(x)

        # Feed-forward 
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x) + residual

        return x


# SASRec Model
class SASRec(nn.Module):
    def __init__(
        self,
        num_items: int,
        hidden_size: int = 64,
        max_len: int = 50,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads.")

        self.max_len = max_len
        self.hidden_size = hidden_size
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [SASRecBlock(hidden_size, num_heads, dropout) for _ in range(num_blocks)]
        )
        self.norm = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.max_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds max_len={self.max_len}.")

        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.item_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)
        x = x * input_ids.ne(0).unsqueeze(-1)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device),
            diagonal=1,
        )
        key_padding_mask = input_ids.eq(0)

        for block in self.blocks:
            x = block(x, causal_mask=causal_mask, key_padding_mask=key_padding_mask)
            x = x * input_ids.ne(0).unsqueeze(-1)

        x = self.norm(x)
        x = x * input_ids.ne(0).unsqueeze(-1)
        return x

    @staticmethod
    def get_last_hidden(seq_out: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        lengths = (input_ids != 0).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        return seq_out[batch_indices, lengths]

    def predict(self, input_ids: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        seq_out = self.forward(input_ids)
        last_hidden = self.get_last_hidden(seq_out, input_ids)
        candidate_emb = self.item_emb(candidate_ids)
        return torch.bmm(candidate_emb, last_hidden.unsqueeze(-1)).squeeze(-1)
