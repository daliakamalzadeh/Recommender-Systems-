from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

TOP_K = 10


def clean_train(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    for col in ["user_id", "item_id", "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "timestamp"])
    df = df.astype({"user_id": np.int64, "item_id": np.int64, "timestamp": np.int64})
    df = df.sort_values(["user_id", "item_id", "timestamp"])
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def temporal_holdout(train_clean: pd.DataFrame,
                     quantile: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = train_clean.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    cutoff = float(np.quantile(df["timestamp"].to_numpy(), quantile))
    train_split = df[df["timestamp"] < cutoff].reset_index(drop=True)
    val = df[df["timestamp"] >= cutoff]
    val = val[val["user_id"].isin(set(train_split["user_id"]))].reset_index(drop=True)
    return train_split, val


def leave_last_out_split(train_clean: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = train_clean.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    counts = df.groupby("user_id")["item_id"].transform("size")
    is_last = df.groupby("user_id").cumcount(ascending=False) == 0
    is_target = is_last & (counts >= 2)
    val = df[is_target].reset_index(drop=True)
    train_split = df[~is_target].reset_index(drop=True)
    return train_split, val


def make_split(train_clean: pd.DataFrame, protocol: str,
               quantile: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if protocol == "temporal":
        return temporal_holdout(train_clean, quantile=quantile)
    if protocol == "loo":
        return leave_last_out_split(train_clean)
    raise ValueError(f"unknown protocol: {protocol}")


def build_targets(val: pd.DataFrame) -> Dict[int, Set[int]]:
    targets: Dict[int, Set[int]] = {}
    for row in val.itertuples(index=False):
        targets.setdefault(int(row.user_id), set()).add(int(row.item_id))
    return targets


def parse_candidates(df: pd.DataFrame) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for row in df.itertuples(index=False):
        out[int(row.user_id)] = [int(x) for x in str(row.item_id).split(",") if x != ""]
    return out


def recall_ndcg_at_k(preds: Dict[int, List[int]],
                     targets: Dict[int, Set[int]],
                     k: int = TOP_K) -> Tuple[float, float]:
    recalls, ndcgs = [], []
    for uid, tgt in targets.items():
        top = preds.get(uid, [])[:k]
        n_hit = sum(1 for i in top if i in tgt)
        recalls.append(n_hit / len(tgt))
        dcg = sum(1.0 / np.log2(rank + 2) for rank, i in enumerate(top) if i in tgt)
        idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(len(tgt), k)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    recall = float(np.mean(recalls)) if recalls else 0.0
    ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    return recall, ndcg


def coverage(preds: Dict[int, List[int]], targets: Dict[int, Set[int]]) -> float:
    if not targets:
        return 0.0
    present = sum(1 for uid in targets if preds.get(uid))
    return present / len(targets)


def main() -> None:
    p = argparse.ArgumentParser(description="Kaggle-faithful offline evaluation")
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--protocol", choices=["temporal", "loo"], default="temporal")
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument("--candidates", nargs="+", default=["val_candidates_bpr.csv",
                                                       "val_candidates_sasrec.csv"],
                   help="candidate CSVs, generated on the SAME split as --protocol")
    p.add_argument("--output", type=Path, default=Path("validation_scores.txt"))
    args = p.parse_args()

    train_clean = clean_train(pd.read_csv(args.data_dir / "train.csv"))
    _, val = make_split(train_clean, args.protocol, args.quantile)
    targets = build_targets(val)

    header = (args.protocol if args.protocol == "loo"
              else f"temporal q={args.quantile}")
    lines = [f"OFFLINE EVALUATION (macro Recall@10 / NDCG@10) — protocol: {header}",
             f"  val users: {len(targets)} | "
             f"mean targets/user: {np.mean([len(t) for t in targets.values()]):.2f}", ""]

    for path in args.candidates:
        fp = args.data_dir / path
        if not fp.exists():
            lines.append(f"{path:<28} MISSING")
            continue
        preds = parse_candidates(pd.read_csv(fp))
        cov = coverage(preds, targets)
        recall, ndcg = recall_ndcg_at_k(preds, targets)
        flag = "" if cov > 0.95 else f"  [coverage {cov:.0%} — regenerate on this split]"
        lines.append(f"{path:<28} Recall@10={recall:.5f}  NDCG@10={ndcg:.5f}{flag}")

    text = "\n".join(lines)
    args.output.write_text(text, encoding="utf-8")
    print(text)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
