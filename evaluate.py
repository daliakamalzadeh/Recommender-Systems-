import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SASRec
from preprocess import SASRecDataProcessor
from train import SASRecTrainDataset, bce_loss, evaluate_split, set_seed

EXPERIMENTS = [
    # Baseline
    {"hidden_size": 64,  "num_blocks": 2, "num_heads": 2, "max_len": 50,  "dropout": 0.2},
    # Vary number of self-attention blocks
    {"hidden_size": 64,  "num_blocks": 1, "num_heads": 2, "max_len": 50,  "dropout": 0.2},
    {"hidden_size": 64,  "num_blocks": 4, "num_heads": 2, "max_len": 50,  "dropout": 0.2},
    # Vary hidden size
    {"hidden_size": 32,  "num_blocks": 2, "num_heads": 2, "max_len": 50,  "dropout": 0.2},
    {"hidden_size": 128, "num_blocks": 2, "num_heads": 4, "max_len": 50,  "dropout": 0.2},
    # Vary max sequence length — X_train is rebuilt per config so no shape mismatch
    {"hidden_size": 64,  "num_blocks": 2, "num_heads": 2, "max_len": 20,  "dropout": 0.2},
    {"hidden_size": 64,  "num_blocks": 2, "num_heads": 2, "max_len": 100, "dropout": 0.2},
    # Vary dropout
    {"hidden_size": 64,  "num_blocks": 2, "num_heads": 2, "max_len": 50,  "dropout": 0.0},
    {"hidden_size": 64,  "num_blocks": 2, "num_heads": 2, "max_len": 50,  "dropout": 0.5},
]

def build_training_arrays(train_seqs, max_len):
    """Re-slice training sequences for the given max_len.

    This must be called per config because the padded input width must match
    the model's max_len. Reusing arrays built with a different max_len causes
    a shape mismatch inside model.forward().
    """
    processor = SASRecDataProcessor.__new__(SASRecDataProcessor)  # no file path needed
    return processor.generate_training_instances(train_seqs, max_len=max_len)

def run_experiment(config, train_seqs, valid_targets, test_targets,
                   num_items, device, args):
    set_seed(args.seed)

    max_len = config["max_len"]
    train_user_ids, X_train, y_train = build_training_arrays(train_seqs, max_len)

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
        hidden_size=config["hidden_size"],
        max_len=max_len,
        num_blocks=config["num_blocks"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_ndcg10 = -1.0
    best_state = None
    patience_cnt = 0

    print(f"\n{'#' * 60}")
    print(f"  Config: {config}")
    print(f"{'#' * 60}")
    print(f"  Parameters : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"  Train pairs: {len(X_train):,}  (X shape={X_train.shape})")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, num_batches = 0.0, 0

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

        avg_loss = total_loss / max(num_batches, 1)
        elapsed = time.time() - t0

        if epoch % args.eval_every == 0:
            val_metrics = evaluate_split(
                model=model,
                sequences=train_seqs,
                targets=valid_targets,
                num_items=num_items,
                max_len=max_len,
                device=device,
                k_list=(10, 20),
                batch_size=args.eval_batch_size,
                sampled_eval=args.sampled_eval,
                num_neg_eval=args.num_neg_eval,
                eval_seed=args.seed,
            )
            ndcg10 = val_metrics["ndcg@10"]
            print(
                f"  Epoch {epoch:4d}/{args.epochs} | loss={avg_loss:.4f} | "
                f"Val NDCG@10={ndcg10:.4f} NDCG@20={val_metrics['ndcg@20']:.4f} "
                f"Recall@10={val_metrics['recall@10']:.4f} Recall@20={val_metrics['recall@20']:.4f} "
                f"| {elapsed:.1f}s"
            )

            if ndcg10 > best_ndcg10:
                best_ndcg10 = ndcg10
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_cnt = 0
                print(f"    ✓ New best")
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f"  Early stopping at epoch {epoch} (best Val NDCG@10={best_ndcg10:.4f})")
                    break
        else:
            print(f"  Epoch {epoch:4d}/{args.epochs} | loss={avg_loss:.4f} | {elapsed:.1f}s")

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Test sequences = training prefix + validation item appended
    test_seqs = {u: train_seqs[u] + [int(valid_targets[u])] for u in train_seqs}

    test_metrics = evaluate_split(
        model=model,
        sequences=test_seqs,
        targets=test_targets,
        num_items=num_items,
        max_len=max_len,
        device=device,
        k_list=(10, 20),
        batch_size=args.eval_batch_size,
        sampled_eval=args.sampled_eval,
        num_neg_eval=args.num_neg_eval,
        eval_seed=args.seed,
    )

    print(
        f"\n  >> TEST  NDCG@10={test_metrics['ndcg@10']:.4f}  NDCG@20={test_metrics['ndcg@20']:.4f}  "
        f"Recall@10={test_metrics['recall@10']:.4f}  Recall@20={test_metrics['recall@20']:.4f}"
    )

    return {
        "hidden_size":      config["hidden_size"],
        "num_blocks":       config["num_blocks"],
        "num_heads":        config["num_heads"],
        "max_len":          config["max_len"],
        "dropout":          config["dropout"],
        "best_val_ndcg@10": round(best_ndcg10, 4),
        "ndcg@10":          round(test_metrics["ndcg@10"], 4),
        "ndcg@20":          round(test_metrics["ndcg@20"], 4),
        "recall@10":        round(test_metrics["recall@10"], 4),
        "recall@20":        round(test_metrics["recall@20"], 4),
    }

def parse_args():
    parser = argparse.ArgumentParser(description="SASRec experiments on MovieLens-1M")
    parser.add_argument("--data_dir",        type=str,   default="./processed_data")
    parser.add_argument("--epochs",          type=int,   default=200)
    parser.add_argument("--batch_size",      type=int,   default=256)
    parser.add_argument("--eval_batch_size", type=int,   default=256)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--weight_decay",    type=float, default=0.0)
    parser.add_argument("--num_neg",         type=int,   default=1)
    parser.add_argument("--eval_every",      type=int,   default=5)
    parser.add_argument("--patience",        type=int,   default=20,
                        help="Early stopping in units of eval checks (not epochs)")
    parser.add_argument("--sampled_eval",    action="store_true",
                        help="Use sampled ranking (faster) instead of full ranking")
    parser.add_argument("--num_neg_eval",    type=int,   default=99)
    parser.add_argument("--num_workers",     type=int,   default=0)
    parser.add_argument("--seed",            type=int,   default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    with open(os.path.join(args.data_dir, "train_sequences.json"), "r", encoding="utf-8") as f:
        train_seqs = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(args.data_dir, "valid_targets.json"), "r", encoding="utf-8") as f:
        valid_targets = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(args.data_dir, "test_targets.json"), "r", encoding="utf-8") as f:
        test_targets = {int(k): v for k, v in json.load(f).items()}
    with open(os.path.join(args.data_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    num_items = int(metadata["num_items"])
    print(f"num_items={num_items} | num_users={metadata['num_users']}")
    print(f"Running {len(EXPERIMENTS)} experiments\n")

    all_results = []
    for config in EXPERIMENTS:
        result = run_experiment(
            config=config,
            train_seqs=train_seqs,
            valid_targets=valid_targets,
            test_targets=test_targets,
            num_items=num_items,
            device=device,
            args=args,
        )
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    col_order = [
        "hidden_size", "num_blocks", "num_heads", "max_len", "dropout",
        "best_val_ndcg@10", "ndcg@10", "ndcg@20", "recall@10", "recall@20",
    ]
    results_df = results_df[col_order].sort_values("ndcg@10", ascending=False)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY (sorted by test NDCG@10)")
    print("=" * 80)
    print(results_df.to_string(index=False))
