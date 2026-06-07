from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TOP_K = 10
PAD = 0  

# Reproducibility
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Data loading / preprocessing 
@dataclass
class Dataset:
    user_sequences: Dict[int, List[int]]   
    train_seq: Dict[int, List[int]]        
    val_targets: Dict[int, set]            
    item_id_to_idx: Dict[int, int]         
    idx_to_item_id: np.ndarray             
    n_items: int                           
    popularity: List[int]                  
    pop_scores: np.ndarray                 


def clean_train(train: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "item_id", "timestamp"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"train.csv missing columns: {sorted(missing)}")

    df = train.copy()
    for col in ["user_id", "item_id", "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "timestamp"])
    df = df.astype({"user_id": np.int64, "item_id": np.int64, "timestamp": np.int64})

    df = df.sort_values(["user_id", "item_id", "timestamp"])
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def build_dataset(train_clean: pd.DataFrame, val_quantile: float = 0.90,
                  final: bool = False) -> Dataset:
    unique_items = np.sort(train_clean["item_id"].unique())
    item_id_to_idx = {int(it): i + 1 for i, it in enumerate(unique_items)}
    idx_to_item_id = np.zeros(len(unique_items) + 1, dtype=np.int64)
    for it, idx in item_id_to_idx.items():
        idx_to_item_id[idx] = it
    n_items = len(unique_items)
    user_seq_ts: Dict[int, List[Tuple[int, int]]] = {}
    for uid, grp in train_clean.groupby("user_id", sort=False):
        idxs = [item_id_to_idx[int(i)] for i in grp["item_id"].to_numpy()]
        tss = grp["timestamp"].to_numpy()
        user_seq_ts[int(uid)] = list(zip(idxs, tss))
    user_sequences: Dict[int, List[int]] = {
        u: [i for i, _ in s] for u, s in user_seq_ts.items()
    }

    train_seq: Dict[int, List[int]] = {}
    val_targets: Dict[int, set] = {}
    cutoff = None
    if final:
        train_seq = {u: list(seq) for u, seq in user_sequences.items()}
    else:
        cutoff = float(np.quantile(train_clean["timestamp"].to_numpy(), val_quantile))
        for uid, seq_ts in user_seq_ts.items():
            hist = [i for i, ts in seq_ts if ts < cutoff]
            future = {i for i, ts in seq_ts if ts >= cutoff}
            if hist:                       
                train_seq[uid] = hist
                if future:                 
                    val_targets[uid] = future
            

    pop_rows = train_clean if cutoff is None else train_clean[train_clean["timestamp"] < cutoff]
    counts = pop_rows["item_id"].value_counts()
    popularity = [item_id_to_idx[int(i)] for i in counts.index]
    pop_scores = np.zeros(n_items + 1, dtype=np.float32)
    for raw_id, c in counts.items():
        pop_scores[item_id_to_idx[int(raw_id)]] = np.log1p(c)
    real = pop_scores[1:]
    pop_scores[1:] = (real - real.mean()) / (real.std() + 1e-6)  

    return Dataset(
        user_sequences=user_sequences,
        train_seq=train_seq,
        val_targets=val_targets,
        item_id_to_idx=item_id_to_idx,
        idx_to_item_id=idx_to_item_id,
        n_items=n_items,
        popularity=popularity,
        pop_scores=pop_scores,
    )


# Torch dataset
class SeqDataset(torch.utils.data.Dataset):

    def __init__(self, sequences: Sequence[List[int]], max_len: int):
        self.samples = [s for s in sequences if len(s) >= 2]
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        seq = self.samples[i][-(self.max_len + 1):]  
        inp = seq[:-1]
        tgt = seq[1:]
        pad = self.max_len - len(inp)
        inp = [PAD] * pad + inp
        tgt = [PAD] * pad + tgt
        return torch.tensor(inp, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)


# SASRec model
class SASRec(nn.Module):
    def __init__(self, n_items: int, max_len: int, hidden: int = 64,
                 n_blocks: int = 2, n_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.n_items = n_items
        self.max_len = max_len
        self.hidden = hidden
        self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_len, hidden)
        self.emb_dropout = nn.Dropout(dropout)

        self.attn_layernorms = nn.ModuleList()
        self.attn_layers = nn.ModuleList()
        self.ffn_layernorms = nn.ModuleList()
        self.ffns = nn.ModuleList()
        for _ in range(n_blocks):
            self.attn_layernorms.append(nn.LayerNorm(hidden, eps=1e-8))
            self.attn_layers.append(
                nn.MultiheadAttention(hidden, n_heads, dropout=dropout, batch_first=True)
            )
            self.ffn_layernorms.append(nn.LayerNorm(hidden, eps=1e-8))
            self.ffns.append(nn.Sequential(
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.Dropout(dropout),
            ))
        self.last_layernorm = nn.LayerNorm(hidden, eps=1e-8)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.item_emb.weight[PAD].fill_(0)

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) item indices -> (B, L, H) sequence representations."""
        B, L = seq.shape
        x = self.item_emb(seq) * (self.hidden ** 0.5)
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = x + self.pos_emb(positions)
        x = self.emb_dropout(x)

        pad_mask = (seq == PAD)                                  
        causal = torch.triu(                                     
            torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1
        )

        for ln_a, attn, ln_f, ffn in zip(
            self.attn_layernorms, self.attn_layers, self.ffn_layernorms, self.ffns
        ):
            q = ln_a(x)
            a, _ = attn(q, x, x, attn_mask=causal, key_padding_mask=pad_mask, need_weights=False)
            x = x + a
            x = x + ffn(ln_f(x))
            x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)       

        return self.last_layernorm(x)

    def logits_all(self, seq: torch.Tensor) -> torch.Tensor:
        """Full-vocabulary logits at every position: (B, L, n_items+1)."""
        reps = self.encode(seq)
        return reps @ self.item_emb.weight.t()

    def last_logits(self, seq: torch.Tensor) -> torch.Tensor:
        """Logits at the final position only: (B, n_items+1). Used for inference."""
        reps = self.encode(seq)
        return reps[:, -1, :] @ self.item_emb.weight.t()


# Training
def train_one_epoch(model, loader, optimizer, device, neg_mode, num_neg, n_items,
                    scaler=None) -> float:
    model.train()
    total, n_batches = 0.0, 0
    for inp, tgt in loader:
        inp, tgt = inp.to(device), tgt.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            if neg_mode == "full":
                
                logits = model.logits_all(inp)                       
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    ignore_index=PAD,
                )
            else:
                
                reps = model.encode(inp)                             
                pos_emb = model.item_emb(tgt)                        
                mask = (tgt != PAD).float()                          
                neg = torch.randint(1, n_items + 1, (inp.size(0), inp.size(1), num_neg),
                                    device=device)
                neg_emb = model.item_emb(neg)                        
                pos_score = (reps * pos_emb).sum(-1)                 
                neg_score = (reps.unsqueeze(2) * neg_emb).sum(-1)    
                pos_loss = F.binary_cross_entropy_with_logits(
                    pos_score, torch.ones_like(pos_score), reduction="none")
                neg_loss = F.binary_cross_entropy_with_logits(
                    neg_score, torch.zeros_like(neg_score), reduction="none").mean(-1)
                loss = ((pos_loss + neg_loss) * mask).sum() / mask.sum().clamp_min(1.0)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total += float(loss.detach())
        n_batches += 1
    return total / max(n_batches, 1)


def fit(model, loader, data, optimizer, device, *, epochs, patience, eval_every,
        neg_mode, num_neg, max_len, max_val_users=None, scaler=None, verbose=True):
    best_recall, best_state, since, history = -1.0, None, 0, []
    has_val = len(data.val_targets) > 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, device,
                               neg_mode, num_neg, data.n_items, scaler)
        if not has_val:
            if verbose and (epoch % eval_every == 0 or epoch == epochs):
                print(f"epoch {epoch:3d} | loss {loss:.4f} | (no val: --final) "
                      f"| {time.time() - t0:.0f}s")
            continue
        if epoch % eval_every == 0 or epoch == epochs:
            recall, ndcg = evaluate(model, data, device, max_len,
                                    max_val_users=max_val_users)
            history.append((epoch, loss, recall, ndcg))
            if verbose:
                print(f"epoch {epoch:3d} | loss {loss:.4f} | Recall@10 {recall:.5f} "
                      f"| NDCG@10 {ndcg:.5f} | {time.time() - t0:.0f}s")
            if recall > best_recall:
                best_recall = recall
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                since = 0
            else:
                since += eval_every
                if since >= patience:
                    if verbose:
                        print(f"early stop at epoch {epoch} (best Recall@10 {best_recall:.5f})")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_recall, history

# Evaluation: local Recall@10 / NDCG@10 over the held-out SET 
def blend_scores(logits: torch.Tensor, pop_t: torch.Tensor, alpha: float) -> torch.Tensor:
    if alpha <= 0:
        return logits
    real = logits[:, 1:]                                  # exclude PAD 
    mean = real.mean(dim=1, keepdim=True)
    std = real.std(dim=1, keepdim=True).clamp_min(1e-6)
    z = (logits - mean) / std
    return (1.0 - alpha) * z + alpha * pop_t.unsqueeze(0)


@torch.no_grad()
def evaluate(model, data: Dataset, device, max_len: int,
             batch_size: int = 512, max_val_users: int | None = None,
             pop_blend: float = 0.0) -> Tuple[float, float]:
    model.eval()
    users = list(data.val_targets)
    if max_val_users is not None and len(users) > max_val_users:
        rng = np.random.default_rng(0)
        users = list(rng.choice(users, size=max_val_users, replace=False))
    pop_t = torch.tensor(data.pop_scores, dtype=torch.float32, device=device)

    recalls, ndcgs = [], []
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        seqs, seens, targets = [], [], []
        for u in batch:
            hist = data.train_seq[u][-max_len:]            
            seqs.append([PAD] * (max_len - len(hist)) + hist)
            seens.append(set(data.train_seq[u]))
            targets.append(data.val_targets[u])
        seq_t = torch.tensor(seqs, dtype=torch.long, device=device)
        scores = blend_scores(model.last_logits(seq_t), pop_t, pop_blend)
        scores[:, PAD] = float("-inf")
        for row, seen, tgt in zip(scores, seens, targets):
            row[list(seen)] = float("-inf")               
            topk = torch.topk(row, TOP_K).indices.tolist()
            n_hit = sum(1 for t in topk if t in tgt)
            recalls.append(n_hit / len(tgt))
            dcg = sum(1.0 / np.log2(rank + 2) for rank, t in enumerate(topk) if t in tgt)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(tgt), TOP_K)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return (float(np.mean(recalls)) if recalls else 0.0,
            float(np.mean(ndcgs)) if ndcgs else 0.0)

# Inference 
@torch.no_grad()
def recommend(model, data: Dataset, user_ids: Sequence[int], device, max_len: int,
              batch_size: int = 512, pop_blend: float = 0.0,
              top_k: int = TOP_K) -> Dict[int, List[int]]:
    model.eval()
    recs: Dict[int, List[int]] = {}
    pop_ids = [int(data.idx_to_item_id[i]) for i in data.popularity]
    pop_t = torch.tensor(data.pop_scores, dtype=torch.float32, device=device)

    known = [u for u in user_ids if u in data.user_sequences]
    for start in range(0, len(known), batch_size):
        batch = known[start:start + batch_size]
        seqs, seens = [], []
        for u in batch:
            hist = data.user_sequences[u][-max_len:]            
            seqs.append([PAD] * (max_len - len(hist)) + hist)
            seens.append(set(data.user_sequences[u]))
        seq_t = torch.tensor(seqs, dtype=torch.long, device=device)
        scores = blend_scores(model.last_logits(seq_t), pop_t, pop_blend)
        scores[:, PAD] = float("-inf")
        for u, row, seen in zip(batch, scores, seens):
            row[list(seen)] = float("-inf")
            idxs = torch.topk(row, top_k).indices.tolist()
            recs[u] = [int(data.idx_to_item_id[i]) for i in idxs]
    for u in user_ids:
        if u not in recs:
            recs[u] = pop_ids[:top_k]
    return recs


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    if list(submission.columns) != list(sample.columns):
        raise ValueError(f"columns {list(submission.columns)} != sample {list(sample.columns)}")
    if len(submission) != len(sample):
        raise ValueError("row count differs from sample_submission.csv")
    lengths = submission["item_id"].astype(str).str.split(",").apply(len)
    if not (lengths == TOP_K).all():
        raise ValueError(f"{int((lengths != TOP_K).sum())} rows do not have exactly {TOP_K} items")


def write_submission(sample: pd.DataFrame, recs: Dict[int, List[int]], out: Path) -> pd.DataFrame:
    rows = [{"ID": int(r.ID), "user_id": int(r.user_id),
             "item_id": ",".join(map(str, recs[int(r.user_id)]))}
            for r in sample.itertuples(index=False)]
    sub = pd.DataFrame(rows, columns=list(sample.columns))
    validate_submission(sub, sample)
    sub.to_csv(out, index=False)
    print(f"Saved submission -> {out}")
    return sub

# Main
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SASRec sequential recommender (Group 30, Task 2)")
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--max-len", type=int, default=50, help="max sequence length")
    p.add_argument("--hidden", type=int, default=64, help="embedding / hidden dim")
    p.add_argument("--blocks", type=int, default=2, help="self-attention blocks")
    p.add_argument("--heads", type=int, default=2, help="attention heads")
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--l2", type=float, default=0.0, help="weight decay")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=20, help="early-stop patience (epochs)")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--neg-mode", choices=["full", "sampled"], default="full",
                   help="full = CE over all items; sampled = original SASRec BCE")
    p.add_argument("--num-neg", type=int, default=1, help="negatives per position (sampled mode)")
    p.add_argument("--max-val-users", type=int, default=-1,
                   help="cap validation users for speed; -1 = all")
    p.add_argument("--val-quantile", type=float, default=0.90,
                   help="global timestamp quantile for the temporal val cutoff")
    p.add_argument("--final", action="store_true",
                   help="train on ALL data for the submission run (no holdout)")
    p.add_argument("--pop-blend", type=float, default=0.0,
                   help="fixed popularity blend weight at inference (0=pure SASRec)")
    p.add_argument("--scan-blend", action="store_true",
                   help="scan pop-blend weight on validation and use the best")
    p.add_argument("--device", default=None, help="cuda / cpu (auto if unset)")
    p.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    out = args.output or (args.data_dir / "submission_sasrec.csv")
    max_val_users = None if args.max_val_users == -1 else args.max_val_users
    print(f"Device: {device}")

    train = pd.read_csv(args.data_dir / "train.csv")
    sample = pd.read_csv(args.data_dir / "sample_submission.csv")
    train_clean = clean_train(train)
    print(f"train rows: {len(train)} -> cleaned: {len(train_clean)} "
          f"(removed {len(train) - len(train_clean)})")
    data = build_dataset(train_clean, val_quantile=args.val_quantile, final=args.final)
    if args.final:
        print(f"users: {len(data.user_sequences)} | items: {data.n_items} | "
              f"FINAL run: training on all data (no holdout)")
    else:
        print(f"users: {len(data.user_sequences)} | items: {data.n_items} | "
              f"val users (temporal q={args.val_quantile}): {len(data.val_targets)}")

    train_sequences = list(data.train_seq.values())
    ds = SeqDataset(train_sequences, args.max_len)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=False,
        num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    print(f"training sequences (len>=2): {len(ds)}")

    model = SASRec(data.n_items, args.max_len, args.hidden,
                   args.blocks, args.heads, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 betas=(0.9, 0.98), weight_decay=args.l2)
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SASRec params: {n_params:,} | neg-mode: {args.neg_mode}")

    best_recall, _ = fit(model, loader, data, optimizer, device,
                         epochs=args.epochs, patience=args.patience,
                         eval_every=args.eval_every, neg_mode=args.neg_mode,
                         num_neg=args.num_neg, max_len=args.max_len,
                         max_val_users=max_val_users, scaler=scaler)
    if args.final:
        print(f"Trained {args.epochs} epochs on all data (final submission run).")
    else:
        print(f"Best local Recall@10 (pure SASRec): {best_recall:.5f}")

    blend = args.pop_blend
    if args.scan_blend:
        print("Scanning popularity-blend weight on validation:")
        best_a, best_r = 0.0, best_recall
        for a in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
            r, _ = evaluate(model, data, device, args.max_len,
                            max_val_users=max_val_users, pop_blend=a)
            print(f"  alpha={a:.2f} -> Recall@10 {r:.5f}")
            if r > best_r:
                best_a, best_r = a, r
        blend = best_a
        print(f"Best blend alpha={blend:.2f} (Recall@10 {best_r:.5f})")

    recs = recommend(model, data, sample["user_id"].astype(int).tolist(),
                     device, args.max_len, pop_blend=blend)
    sub = write_submission(sample, recs, out)
    print(sub.head())


if __name__ == "__main__":
    main()
