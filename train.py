import argparse
import json
import os
import random
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import SASRec


class SASRecTrainDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        user_ids: np.ndarray,
        user_seen_items: Dict[int, Sequence[int]],
        num_items: int,
        num_neg: int = 1,
    ):
        self.X = torch.as_tensor(X, dtype=torch.long)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.user_ids = torch.as_tensor(user_ids, dtype=torch.long)
        self.user_seen_items = {int(k): set(v) for k, v in user_seen_items.items()}
        self.num_items = int(num_items)
        self.num_neg = int(num_neg)

        if len(self.X) != len(self.y) or len(self.X) != len(self.user_ids):
            raise ValueError("X, y, and user_ids must have the same length.")
        if self.num_neg < 1:
            raise ValueError("num_neg must be at least 1.")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        input_ids = self.X[idx]
        pos_item = self.y[idx]
        user_id = int(self.user_ids[idx].item())

        seen = set(self.user_seen_items.get(user_id, set()))
        seen.add(int(pos_item.item()))

        negs = []
        while len(negs) < self.num_neg:
            neg = random.randint(1, self.num_items)
            if neg not in seen:
                negs.append(neg)
                seen.add(neg)

        neg_items = torch.tensor(negs, dtype=torch.long)
        return input_ids, pos_item, neg_items


def bce_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    pos_labels = torch.ones_like(pos_scores)
    neg_labels = torch.zeros_like(neg_scores)
    pos_loss = nn.functional.binary_cross_entropy_with_logits(pos_scores, pos_labels)
    neg_loss = nn.functional.binary_cross_entropy_with_logits(neg_scores, neg_labels)
    return pos_loss + neg_loss


def _pad_sequence(seq: Sequence[int], max_len: int) -> List[int]:
    seq = list(seq)[-max_len:]
    return [0] * (max_len - len(seq)) + seq


@torch.no_grad()
def evaluate_split(
    model: SASRec,
    sequences: Dict[int, Sequence[int]],
    targets: Dict[int, int],
    num_items: int,
    max_len: int,
    device: torch.device,
    k_list: Sequence[int] = (10, 20),
    batch_size: int = 256,
    sampled_eval: bool = False,
    num_neg_eval: int = 99,
    eval_seed: int = 42,
) -> Dict[str, float]:
    model.eval()
    k_list = tuple(sorted(set(int(k) for k in k_list)))
    results = {f"recall@{k}": [] for k in k_list}
    results.update({f"ndcg@{k}": [] for k in k_list})

    users = list(sequences.keys())

    for start in range(0, len(users), batch_size):
        batch_users = users[start:start + batch_size]
        batch_prefixes = [sequences[u] for u in batch_users]
        batch_targets = [int(targets[u][0] if isinstance(targets[u], list) else targets[u]) for u in batch_users]

        input_batch = torch.tensor(
            [_pad_sequence(prefix, max_len) for prefix in batch_prefixes],
            dtype=torch.long,
            device=device,
        )

        seq_out = model(input_batch)
        last_hidden = model.get_last_hidden(seq_out, input_batch)

        if sampled_eval:
            for row_idx, user in enumerate(batch_users):
                target = batch_targets[row_idx]
                full_seq_set = set(batch_prefixes[row_idx]) | {target}

                # Deterministic per-user sampled evaluation so metrics are stable
                # across validation/test calls and configuration comparisons.
                rng = random.Random(eval_seed + int(user) * 9973 + int(target) * 37)
                neg_pool = set()
                while len(neg_pool) < num_neg_eval:
                    neg = rng.randint(1, num_items)
                    if neg not in full_seq_set and neg not in neg_pool:
                        neg_pool.add(neg)

                candidates = [target] + list(neg_pool)
                rng.shuffle(candidates)
                pos_rank_in_candidates = candidates.index(target)

                cand_ids = torch.tensor([candidates], dtype=torch.long, device=device)
                cand_emb = model.item_emb(cand_ids)
                scores = torch.bmm(cand_emb, last_hidden[row_idx:row_idx + 1].unsqueeze(-1)).squeeze(-1).squeeze(0)
                order = torch.argsort(scores, descending=True).cpu().tolist()
                rank = order.index(pos_rank_in_candidates) + 1

                for k in k_list:
                    results[f"recall@{k}"].append(1.0 if rank <= k else 0.0)
                    results[f"ndcg@{k}"].append(1.0 / np.log2(rank + 1) if rank <= k else 0.0)
            continue

        all_item_ids = torch.arange(1, num_items + 1, device=device)
        all_item_emb = model.item_emb(all_item_ids)
        scores = torch.matmul(last_hidden, all_item_emb.transpose(0, 1))

        for row_idx, user in enumerate(batch_users):
            seen_items = set(batch_prefixes[row_idx])
            target = batch_targets[row_idx]
            seen_without_target = [item for item in seen_items if item != target]
            if seen_without_target:
                seen_idx = torch.tensor([item - 1 for item in seen_without_target], dtype=torch.long, device=device)
                scores[row_idx, seen_idx] = float("-inf")

        topk_max = max(k_list)
        topk_indices = torch.topk(scores, k=topk_max, dim=1).indices + 1

        for row_idx, target in enumerate(batch_targets):
            ranked_items = topk_indices[row_idx].tolist()
            for k in k_list:
                topk = ranked_items[:k]
                if target in topk:
                    rank = topk.index(target) + 1
                    results[f"recall@{k}"].append(1.0)
                    results[f"ndcg@{k}"].append(1.0 / np.log2(rank + 1))
                else:
                    results[f"recall@{k}"].append(0.0)
                    results[f"ndcg@{k}"].append(0.0)

    return {metric: float(np.mean(values)) for metric, values in results.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X_train = np.load(os.path.join(args.data_dir, "X_train.npy"))
    y_train = np.load(os.path.join(args.data_dir, "y_train.npy"))
    train_user_ids = np.load(os.path.join(args.data_dir, "train_user_ids.npy"))

    with open(os.path.join(args.data_dir, "train_sequences.json"), "r", encoding="utf-8") as f:
        train_seqs = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(args.data_dir, "valid_targets.json"), "r", encoding="utf-8") as f:
        valid_targets = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(args.data_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    num_items = int(metadata["num_items"])
    print(f"num_items={num_items} | training pairs={len(X_train)}")

    dataset = SASRecTrainDataset(
        X=X_train,
        y=y_train,
        user_ids=train_user_ids,
        user_seen_items=train_seqs,
        num_items=num_items,
        num_neg=args.num_neg,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = SASRec(
        num_items=num_items,
        hidden_size=args.hidden_size,
        max_len=args.max_len,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.lr * 0.01,
        )

    best_ndcg10 = -1.0
    best_epoch = 0
    patience_cnt = 0

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        start_time = time.time()
        total_loss = 0.0
        num_batches = 0

        for input_ids, pos_items, neg_items in loader:
            input_ids = input_ids.to(device)
            pos_items = pos_items.to(device)
            neg_items = neg_items.to(device)

            seq_out = model(input_ids)
            last_hidden = model.get_last_hidden(seq_out, input_ids)

            pos_emb = model.item_emb(pos_items)
            pos_scores = (last_hidden * pos_emb).sum(dim=-1)

            neg_emb = model.item_emb(neg_items)
            neg_scores = torch.bmm(neg_emb, last_hidden.unsqueeze(-1)).squeeze(-1)

            loss = bce_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - start_time

        if epoch % args.eval_every == 0:
            val_metrics = evaluate_split(
                model=model,
                sequences=train_seqs,
                targets=valid_targets,
                num_items=num_items,
                max_len=args.max_len,
                device=device,
                k_list=(10, 20),
                batch_size=args.eval_batch_size,
                sampled_eval=args.sampled_eval,
                num_neg_eval=args.num_neg_eval,
                eval_seed=args.seed,
            )
            ndcg10 = val_metrics["ndcg@10"]
            print(
                f"Epoch {epoch:4d}/{args.epochs} | loss={avg_loss:.4f} | "
                f"Val NDCG@10={ndcg10:.4f} NDCG@20={val_metrics['ndcg@20']:.4f} "
                f"Recall@10={val_metrics['recall@10']:.4f} Recall@20={val_metrics['recall@20']:.4f} "
                f"| {elapsed:.1f}s"
            )

            if ndcg10 > best_ndcg10:
                best_ndcg10 = ndcg10
                best_epoch = epoch
                patience_cnt = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "val_metrics": val_metrics,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"  ✓ New best - checkpoint saved to {ckpt_path}")
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(
                        f"\nEarly stopping at epoch {epoch} "
                        f"(best epoch={best_epoch}, best NDCG@10={best_ndcg10:.4f})"
                    )
                    break
        else:
            print(f"Epoch {epoch:4d}/{args.epochs} | loss={avg_loss:.4f} | {elapsed:.1f}s")

    print(f"\nTraining complete. Best validation NDCG@10 = {best_ndcg10:.4f} at epoch {best_epoch}.")
    return ckpt_path


def parse_args():
    parser = argparse.ArgumentParser(description="Train SASRec on MovieLens-1M")
    parser.add_argument("--data_dir", type=str, default="./processed_data")
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_neg", type=int, default=1, help="Negative samples per positive training example")
    parser.add_argument("--use_scheduler", action="store_true", help="Enable cosine annealing learning-rate schedule")
    parser.add_argument("--patience", type=int, default=20, help="Stop after this many non-improving validation checks")
    parser.add_argument("--eval_every", type=int, default=5, help="Run validation every N epochs")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--sampled_eval", action="store_true", help="Use sampled ranking with random negatives instead of full ranking")
    parser.add_argument("--num_neg_eval", type=int, default=99, help="Number of negatives for sampled evaluation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
