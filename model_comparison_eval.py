"""
Model comparison evaluation.
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pandas as pd

from baseline_pipeline import (
    load_data, clean_train, chronological_split,
    make_user_seen_items, make_bm25_popularity,
    train_bpr, recommend_factor_model, recommend_from_ranked,
)
from ensemble_pipeline import build_hybrid_submission
from eval_utils import evaluate_predictions, parse_submission, user_histories
from meta_scoring import MetaScorer

OUTPUT      = Path("model_comparison_scores.txt")
CANDIDATE_K = 30  # candidates per model for ensemble methods
TOP_K       = 10  # final recommendation list length

VAL_BPR_PATH    = Path("val_candidates_bpr.csv")
VAL_SASREC_PATH = Path("val_candidates_sasrec.csv")


def main() -> None:
    train, _, item_meta, _ = load_data(Path("."))
    train_clean = clean_train(train)
    train_split, val = chronological_split(train_clean)

    histories  = user_histories(train_split)
    user_seen  = make_user_seen_items(train_split)
    bm25_items = make_bm25_popularity(train_split)
    scorer     = MetaScorer.build(item_meta)

    val_sample = pd.DataFrame({
        "ID": range(len(val)),
        "user_id": val["user_id"].values,
        "item_id": 0,
    })

    # BPR (top-10)
    print("[model_comparison] Training BPR …")
    with open(os.devnull, "w") as _null, contextlib.redirect_stdout(_null):
        bpr_enc, bpr_uf, bpr_if = train_bpr(train_split)
    bpr_preds = {
        int(row.user_id): recommend_factor_model(
            int(row.user_id), bpr_enc, bpr_uf, bpr_if, bm25_items, k=TOP_K
        )
        for row in val.itertuples()
    }

    # SASRec (top-10)
    sasrec_sub = pd.read_csv(Path("submission_sasrec.csv")) if Path("submission_sasrec.csv").exists() else None
    sasrec_preds_10: dict = parse_submission(sasrec_sub) if sasrec_sub is not None else {}
    for row in val.itertuples():
        uid = int(row.user_id)
        if uid not in sasrec_preds_10:
            sasrec_preds_10[uid] = recommend_from_ranked(uid, user_seen, bm25_items, k=TOP_K)

    # Ensemble candidates (top-30 per model) 
    if VAL_BPR_PATH.exists() and VAL_SASREC_PATH.exists():
        print("[model_comparison] Loading val candidate files …")
        bpr_preds_30    = parse_submission(pd.read_csv(VAL_BPR_PATH))
        sasrec_preds_30 = parse_submission(pd.read_csv(VAL_SASREC_PATH))
    else:
        print("[model_comparison] Val candidate files not found — using BPR k=30 + BM25 fallback.")
        bpr_preds_30 = {
            int(row.user_id): recommend_factor_model(
                int(row.user_id), bpr_enc, bpr_uf, bpr_if, bm25_items, k=CANDIDATE_K
            )
            for row in val.itertuples()
        }
        sasrec_preds_30 = {
            uid: recommend_from_ranked(uid, user_seen, bm25_items, k=CANDIDATE_K)
            for uid in bpr_preds_30
        }

    # Ensemble-RRF (meta_weight=0, 30 candidates -> top-10)
    ensemble_df = build_hybrid_submission(
        val_sample, bpr_preds_30, sasrec_preds_30, scorer,
        histories, user_seen, rrf_k=60, meta_weight=0.0, sasrec_weight=2.0, top_k=TOP_K,
    )
    ensemble_preds = parse_submission(ensemble_df)

    # Full pipeline (meta_weight=0.15, 30 candidates -> top-10)
    full_df = build_hybrid_submission(
        val_sample, bpr_preds_30, sasrec_preds_30, scorer,
        histories, user_seen, rrf_k=60, meta_weight=0.15, sasrec_weight=2.0, top_k=TOP_K,
    )
    full_preds = parse_submission(full_df)

    preds = {
        "BPR":           bpr_preds,
        "SASRec":        sasrec_preds_10,
        "Ensemble-RRF":  ensemble_preds,
        "Full-Pipeline": full_preds,
    }

    lines = ["MODEL COMPARISON (Recall@10, NDCG@10)", "",
             f"  BPR and SASRec use top-{TOP_K} candidates.",
             f"  Ensemble methods use top-{CANDIDATE_K} candidates per model before reranking to {TOP_K}.", ""]
    for name, pred in preds.items():
        recall, ndcg = evaluate_predictions(pred, val)
        lines.append(f"{name:<16} Recall@10={recall:.5f}  NDCG@10={ndcg:.5f}")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote model comparison scores to {OUTPUT}")


if __name__ == "__main__":
    main()