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
        x, _ = self.attention(x, x, x, attn_mask=causal_mask, is_causal=True)
        x = self.dropout(x) + residual

        # Feed-forward 
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x) + residual

        return x


# SASRec Model
class SASRec(nn.Module):
    def __init__(self, num_items, hidden_size=64, max_len=50, num_blocks=2, num_heads=2, dropout=0.2):
        super().__init__()
        self.max_len = max_len
        self.hidden_size = hidden_size
        self.item_emb = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_size)

        self.blocks = nn.ModuleList([
            SASRecBlock(hidden_size, num_heads, dropout)
            for _ in range(num_blocks)
        ])

        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

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

    def forward(self, input_ids):
        B, L = input_ids.shape
        device = input_ids.device
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        x = self.dropout(self.item_emb(input_ids) + self.pos_emb(positions))
        causal_mask = torch.triu(
            torch.full((L, L), float("-inf"), device=device), diagonal=1
        )

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.norm(x)
        return x  

    def predict(self, input_ids, candidate_ids):
        seq_out = self.forward(input_ids)          
        last = seq_out[:, -1, :]                   
        cand_emb = self.item_emb(candidate_ids)    
        scores = torch.bmm(cand_emb, last.unsqueeze(-1)).squeeze(-1)  
        return scores
