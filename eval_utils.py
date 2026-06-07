"""
Shared evaluation utilities for validation scoring.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from ensemble_pipeline import build_hybrid_submission, rerank_with_meta
from meta_scoring import MetaScorer

TOP_K = 10


def load_all(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    item_meta = pd.read_csv(data_dir / "item_meta.csv")
    sample_sub = pd.read_csv(data_dir / "sample_submission.csv")

    def _try(path: Path):
        return pd.read_csv(path) if path.exists() else None

    bpr_sub = _try(data_dir / "submission_bpr.csv")
    sasrec_sub = _try(data_dir / "submission_sasrec.csv")
    return train, test, item_meta, sample_sub, bpr_sub, sasrec_sub


def clean_train(train: pd.DataFrame) -> pd.DataFrame:
    df = train.copy()
    for col in ["user_id", "item_id", "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "timestamp"])
    df = df.astype({"user_id": np.int64, "item_id": np.int64, "timestamp": np.int64})
    df = df.sort_values(["user_id", "item_id", "timestamp"])
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)


def parse_submission(df: pd.DataFrame) -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = {}
    for _, row in df.iterrows():
        uid = int(row["user_id"])
        items = [int(x) for x in str(row["item_id"]).split(",")]
        result[uid] = items
    return result


def user_histories(train_clean: pd.DataFrame) -> Dict[int, List[int]]:
    return (
        train_clean.sort_values(["user_id", "timestamp"])
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )


def recall_at_k(recs: List[int], true_item: int) -> int:
    return int(int(true_item) in recs[:TOP_K])


def ndcg_at_k(recs: List[int], true_item: int) -> float:
    true_item = int(true_item)
    if true_item not in recs[:TOP_K]:
        return 0.0
    rank = recs[:TOP_K].index(true_item) + 1
    return 1.0 / np.log2(rank + 1)


def evaluate_predictions(
    preds: Dict[int, List[int]],
    targets: pd.DataFrame,
) -> Tuple[float, float]:
    hits: List[int] = []
    ndcgs: List[float] = []
    for row in targets.itertuples(index=False):
        uid = int(row.user_id)
        true_item = int(row.item_id)
        recs = preds.get(uid, [])
        hits.append(recall_at_k(recs, true_item))
        ndcgs.append(ndcg_at_k(recs, true_item))
    recall = float(np.mean(hits)) if hits else 0.0
    ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    return recall, ndcg


def build_meta_only_submission(
    sample_sub: pd.DataFrame,
    src_preds: Dict[int, List[int]],
    scorer: MetaScorer,
    histories: Dict[int, List[int]],
    user_seen: Dict[int, Set[int]],
    top_k: int = TOP_K,
) -> pd.DataFrame:
    rows = []
    for _, row in sample_sub.iterrows():
        uid = int(row["user_id"])
        candidates = [(iid, 1.0 / (1 + rank))
                      for rank, iid in enumerate(src_preds.get(uid, []))]
        history = histories.get(uid, [])
        seen = user_seen.get(uid, set())
        recs = rerank_with_meta(
            candidates,
            scorer,
            history,
            meta_weight=1.0,
            seen=seen,
            k=top_k,
        )
        rows.append({
            "ID": int(row["ID"]),
            "user_id": uid,
            "item_id": ",".join(map(str, recs[:top_k])),
        })
    return pd.DataFrame(rows, columns=list(sample_sub.columns))


def build_predictions(
    data_dir: Path,
    rrf_k: int = 60,
    sasrec_weight: float = 2.0,
    meta_weight: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[int, List[int]]]]:
    train, test, item_meta, sample_sub, bpr_sub, sasrec_sub = load_all(data_dir)
    train_clean = clean_train(train)
    histories = user_histories(train_clean)
    user_seen: Dict[int, Set[int]] = {uid: set(items) for uid, items in histories.items()}

    scorer = MetaScorer.build(item_meta)

    bpr_preds = parse_submission(bpr_sub) if bpr_sub is not None else {}
    sasrec_preds = parse_submission(sasrec_sub) if sasrec_sub is not None else {}

    preds: Dict[str, Dict[int, List[int]]] = {
        "BPR": bpr_preds,
        "SASRec": sasrec_preds,
    }

    if bpr_preds or sasrec_preds:
        ensemble_only_df = build_hybrid_submission(
            sample_sub=sample_sub,
            bpr_preds=bpr_preds,
            sasrec_preds=sasrec_preds,
            scorer=scorer,
            histories=histories,
            user_seen=user_seen,
            rrf_k=rrf_k,
            meta_weight=0.0,
            sasrec_weight=sasrec_weight,
            top_k=TOP_K,
        )
        preds["Ensemble-RRF"] = parse_submission(ensemble_only_df)

        full_df = build_hybrid_submission(
            sample_sub=sample_sub,
            bpr_preds=bpr_preds,
            sasrec_preds=sasrec_preds,
            scorer=scorer,
            histories=histories,
            user_seen=user_seen,
            rrf_k=rrf_k,
            meta_weight=meta_weight,
            sasrec_weight=sasrec_weight,
            top_k=TOP_K,
        )
        preds["Full-Pipeline"] = parse_submission(full_df)

    if sasrec_preds:
        meta_only_df = build_meta_only_submission(
            sample_sub=sample_sub,
            src_preds=sasrec_preds,
            scorer=scorer,
            histories=histories,
            user_seen=user_seen,
            top_k=TOP_K,
        )
        preds["Meta-Only"] = parse_submission(meta_only_df)
    elif bpr_preds:
        meta_only_df = build_meta_only_submission(
            sample_sub=sample_sub,
            src_preds=bpr_preds,
            scorer=scorer,
            histories=histories,
            user_seen=user_seen,
            top_k=TOP_K,
        )
        preds["Meta-Only"] = parse_submission(meta_only_df)

    return train_clean, test, preds


def cold_start_users(train_clean: pd.DataFrame, test: pd.DataFrame) -> Set[int]:
    train_users = set(train_clean["user_id"].unique())
    test_users = set(test["user_id"].unique())
    return set(int(u) for u in (test_users - train_users))
