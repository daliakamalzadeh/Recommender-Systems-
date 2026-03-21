"""
NCF Experiments — full-ranking evaluation (Recall@10, NDCG@10).
Author: Karthik Sundaram

Run from the repo root:
    python experiments.py
"""

import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from model import NCF


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MovieLensDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.user_ids = torch.tensor(df["userId"].values, dtype=torch.long)
        self.item_ids = torch.tensor(df["movieId"].values, dtype=torch.long)
        self.labels   = torch.tensor(df["label"].values,   dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def build_train_positive_set(train_df: pd.DataFrame) -> dict:
    return train_df[train_df["label"] == 1].groupby("userId")["movieId"].apply(set).to_dict()


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    hits = len(set(recommended[:k]) & relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(recommended: list, relevant: set, k: int) -> float:
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant:
            dcg += 1.0 / np.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(r + 1) for r in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_full_ranking(
    model: nn.Module,
    test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    num_items: int,
    k: int = 10,
    batch_size: int = 512,
    device: torch.device = torch.device("cpu"),
) -> dict:
    model.eval()
    train_positives = build_train_positive_set(train_df)
    test_positives = (
        test_df[test_df["label"] == 1]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    all_items = torch.arange(num_items, device=device)
    recall_scores, ndcg_scores = [], []

    with torch.no_grad():
        for user_id, relevant_items in test_positives.items():
            seen = train_positives.get(user_id, set())
            candidate_items = all_items[
                ~torch.isin(all_items, torch.tensor(list(seen), device=device))
            ]

            user_tensor = torch.full(
                (len(candidate_items),), user_id, dtype=torch.long, device=device
            )
            scores = []
            for start in range(0, len(candidate_items), batch_size):
                u_batch = user_tensor[start: start + batch_size]
                i_batch = candidate_items[start: start + batch_size]
                scores.append(model(u_batch, i_batch).cpu())
            scores = torch.cat(scores)

            top_k_indices = torch.topk(scores, k=k).indices
            top_k_items = candidate_items[top_k_indices].cpu().tolist()

            recall_scores.append(recall_at_k(top_k_items, relevant_items, k))
            ndcg_scores.append(ndcg_at_k(top_k_items, relevant_items, k))

    return {
        f"Recall@{k}": np.mean(recall_scores),
        f"NDCG@{k}":   np.mean(ndcg_scores),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for users, items, labels in loader:
        users, items, labels = users.to(device), items.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(users, items)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate_loss(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for users, items, labels in loader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            preds = model(users, items)
            total_loss += criterion(preds, labels).item() * len(labels)
    return total_loss / len(loader.dataset)


def run_experiment(
    config: dict,
    train_loader, val_loader, test_df, train_df,
    num_users, num_items, device,
    max_epochs=20, patience=3,
):
    model = NCF(
        num_users=num_users,
        num_items=num_items,
        gmf_embed_dim=config["gmf_embed_dim"],
        mlp_embed_dim=config["mlp_embed_dim"],
        mlp_layer_sizes=config["mlp_layer_sizes"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    epochs_no_improve = 0

    print(f"\n--- Config: {config} ---")
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate_loss(model, val_loader, criterion, device)
        elapsed    = time.time() - t0
        print(f"  Epoch {epoch:02d} | train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    metrics = evaluate_full_ranking(
        model=model,
        test_df=test_df,
        train_df=train_df,
        num_items=num_items,
        k=10,
        device=device,
    )
    metrics["config"] = str(config)
    metrics["best_val_loss"] = round(best_val_loss, 4)
    print(f"  >> Recall@10: {metrics['Recall@10']:.4f} | NDCG@10: {metrics['NDCG@10']:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cpu")

    # Load data
    train_df = pd.read_csv("ncf_preprocessed/train.csv")
    val_df   = pd.read_csv("ncf_preprocessed/val.csv")
    test_df  = pd.read_csv("ncf_preprocessed/test.csv")

    with open("ncf_preprocessed/metadata.json") as f:
        meta = json.load(f)

    train_loader = DataLoader(MovieLensDataset(train_df), batch_size=1024, shuffle=True)
    val_loader   = DataLoader(MovieLensDataset(val_df),   batch_size=1024, shuffle=False)

    experiments = [
        {"gmf_embed_dim": 32, "mlp_embed_dim": 32, "mlp_layer_sizes": [64],          "dropout": 0.2},
        {"gmf_embed_dim": 32, "mlp_embed_dim": 32, "mlp_layer_sizes": [128, 64],      "dropout": 0.2},
        {"gmf_embed_dim": 32, "mlp_embed_dim": 32, "mlp_layer_sizes": [256, 128, 64], "dropout": 0.2},
        {"gmf_embed_dim": 16, "mlp_embed_dim": 16, "mlp_layer_sizes": [256, 128, 64], "dropout": 0.2},
        {"gmf_embed_dim": 64, "mlp_embed_dim": 64, "mlp_layer_sizes": [256, 128, 64], "dropout": 0.2},
        {"gmf_embed_dim": 32, "mlp_embed_dim": 32, "mlp_layer_sizes": [256, 128, 64], "dropout": 0.0},
        {"gmf_embed_dim": 32, "mlp_embed_dim": 32, "mlp_layer_sizes": [256, 128, 64], "dropout": 0.4},
    ]

    all_results = []
    for config in experiments:
        result = run_experiment(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            test_df=test_df,
            train_df=train_df,
            num_users=meta["num_users"],
            num_items=meta["num_items"],
            device=device,
            max_epochs=20,
            patience=3,
        )
        all_results.append(result)

    results_df = pd.DataFrame(all_results)[["config", "best_val_loss", "Recall@10", "NDCG@10"]]
    print("\n===== RESULTS SUMMARY =====")
    print(results_df.to_string(index=False))
