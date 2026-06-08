from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import pandas as pd
import torch

from sasrec import (PAD, SASRec, SeqDataset, build_dataset, clean_train,
                    evaluate, fit, set_seed)

FULL_GRID = {
    "hidden": [32, 64, 128],
    "lr": [1e-3, 5e-4],
    "dropout": [0.2, 0.5],
    "blocks": [2],
}
QUICK_GRID = {
    "hidden": [32, 64],
    "lr": [1e-3],
    "dropout": [0.2],
    "blocks": [2],
}

BLEND_ALPHAS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


def grid_configs(grid: dict):
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SASRec hyperparameter sweep")
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--log", type=Path, default=Path("sweep_results.csv"))
    p.add_argument("--quick", action="store_true", help="tiny grid for a fast smoke check")
    p.add_argument("--max-len", type=int, default=50)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--l2", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--neg-mode", choices=["full", "sampled"], default="full")
    p.add_argument("--num-neg", type=int, default=1)
    p.add_argument("--max-val-users", type=int, default=-1)
    p.add_argument("--val-quantile", type=float, default=0.90,
                   help="global timestamp quantile for the temporal val cutoff")
    p.add_argument("--device", default=None)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    max_val_users = None if args.max_val_users == -1 else args.max_val_users
    grid = QUICK_GRID if args.quick else FULL_GRID
    configs = list(grid_configs(grid))
    print(f"Device: {device} | configs: {len(configs)} | log: {args.log}")
    train_clean = clean_train(pd.read_csv(args.data_dir / "train.csv"))
    data = build_dataset(train_clean, val_quantile=args.val_quantile)
    print(f"users: {len(data.user_sequences)} | items: {data.n_items} | "
          f"val users (temporal q={args.val_quantile}): {len(data.val_targets)}")

    fieldnames = ["hidden", "lr", "dropout", "blocks", "heads", "max_len",
                  "neg_mode", "best_recall@10", "best_ndcg@10",
                  "best_blend_alpha", "blended_recall@10",
                  "epochs_to_best", "params", "seconds"]
    write_header = not args.log.exists()
    log_f = open(args.log, "a", newline="")
    writer = csv.DictWriter(log_f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    results = []
    for i, cfg in enumerate(configs, 1):
        set_seed(args.seed)  
        ds = SeqDataset(list(data.train_seq.values()), args.max_len)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed))

        model = SASRec(data.n_items, args.max_len, cfg["hidden"],
                       cfg["blocks"], args.heads, cfg["dropout"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                                     betas=(0.9, 0.98), weight_decay=args.l2)
        scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None
        params = sum(p.numel() for p in model.parameters())

        print(f"\n[{i}/{len(configs)}] {cfg}")
        t0 = time.time()
        best_recall, history = fit(
            model, loader, data, optimizer, device,
            epochs=args.epochs, patience=args.patience, eval_every=args.eval_every,
            neg_mode=args.neg_mode, num_neg=args.num_neg, max_len=args.max_len,
            max_val_users=max_val_users, scaler=scaler, verbose=False)
        seconds = time.time() - t0

        best_ndcg = max((n for _, _, r, n in history), default=0.0)
        epochs_to_best = max(
            ((e for e, _, r, _ in history if r == best_recall)), default=0)
        best_alpha, best_blended = 0.0, best_recall
        for a in BLEND_ALPHAS:
            r, _ = evaluate(model, data, device, args.max_len,
                            max_val_users=max_val_users, pop_blend=a)
            if r > best_blended:
                best_alpha, best_blended = a, r

        row = {"hidden": cfg["hidden"], "lr": cfg["lr"], "dropout": cfg["dropout"],
               "blocks": cfg["blocks"], "heads": args.heads, "max_len": args.max_len,
               "neg_mode": args.neg_mode, "best_recall@10": round(best_recall, 5),
               "best_ndcg@10": round(best_ndcg, 5),
               "best_blend_alpha": best_alpha, "blended_recall@10": round(best_blended, 5),
               "epochs_to_best": epochs_to_best,
               "params": params, "seconds": round(seconds, 1)}
        writer.writerow(row)
        log_f.flush()
        results.append(row)
        print(f"    -> Recall@10 {best_recall:.5f} (pure) | "
              f"{best_blended:.5f} @ blend {best_alpha:.2f} | "
              f"NDCG@10 {best_ndcg:.5f} | {seconds:.0f}s | {params:,} params")

    log_f.close()

    results.sort(key=lambda r: r["blended_recall@10"], reverse=True)
    print("\n===== sweep summary (best blended first) =====")
    print(f"{'hidden':>6} {'lr':>7} {'drop':>5} {'blk':>3} "
          f"{'pure@10':>9} {'blend@10':>9} {'alpha':>6} {'sec':>6}")
    for r in results:
        print(f"{r['hidden']:>6} {r['lr']:>7} {r['dropout']:>5} {r['blocks']:>3} "
              f"{r['best_recall@10']:>9} {r['blended_recall@10']:>9} "
              f"{r['best_blend_alpha']:>6} {r['seconds']:>6.0f}")

    if results:
        b = results[0]
        print(f"\nBest config: hidden={b['hidden']} lr={b['lr']} "
              f"dropout={b['dropout']} blocks={b['blocks']} blend={b['best_blend_alpha']}  "
              f"(blended Recall@10 {b['blended_recall@10']})")
        print("Retrain on ALL data for submission (epochs ~= epochs_to_best of the "
              f"validation run = {b['epochs_to_best']}):")
        print(f"  python sasrec.py --data-dir . --output submission_sasrec.csv "
              f"--hidden {b['hidden']} --lr {b['lr']} --dropout {b['dropout']} "
              f"--blocks {b['blocks']} --final --pop-blend {b['best_blend_alpha']} "
              f"--epochs {max(b['epochs_to_best'], 1)} --amp")


if __name__ == "__main__":
    main()
