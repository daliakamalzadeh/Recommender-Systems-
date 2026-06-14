from __future__ import annotations

from pathlib import Path

import pandas as pd

from baseline_pipeline import (
    load_data, clean_train, chronological_split,
    make_user_seen_items, make_bm25_popularity, recommend_from_ranked,
)
from ensemble_pipeline import build_hybrid_submission
from eval_utils import evaluate_predictions, parse_submission, user_histories
from meta_scoring import MetaScorer

OUTPUT      = Path("model_comparison_scores.txt")
CANDIDATE_K = 30  
TOP_K       = 10  

VAL_BPR_PATH    = Path("val_candidates_bpr.csv")
VAL_SASREC_PATH = Path("val_candidates_sasrec.csv")


def _topk(preds: dict, k: int) -> dict:
    """Truncate each user's candidate list to its first k items."""
    return {uid: items[:k] for uid, items in preds.items()}


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


    if not (VAL_BPR_PATH.exists() and VAL_SASREC_PATH.exists()):
        raise FileNotFoundError(
            "Missing val_candidates_bpr.csv / val_candidates_sasrec.csv. "
            "Run generate_candidates.py first (it writes held-out candidates "
            "trained on the temporal train_split)."
        )
    print("[model_comparison] Loading held-out val candidate files …")
    bpr_preds_30    = parse_submission(pd.read_csv(VAL_BPR_PATH))
    sasrec_preds_30 = parse_submission(pd.read_csv(VAL_SASREC_PATH))

    
    bpr_preds_10    = _topk(bpr_preds_30, TOP_K)
    sasrec_preds_10 = _topk(sasrec_preds_30, TOP_K)

    pop_preds = {
        uid: recommend_from_ranked(uid, user_seen, bm25_items, k=TOP_K)
        for uid in val["user_id"].unique()
    }

    ensemble_df = build_hybrid_submission(
        val_sample, bpr_preds_30, sasrec_preds_30, scorer,
        histories, user_seen, rrf_k=60, meta_weight=0.0, sasrec_weight=2.0, top_k=TOP_K,
    )
    ensemble_preds = parse_submission(ensemble_df)


    full_df = build_hybrid_submission(
        val_sample, bpr_preds_30, sasrec_preds_30, scorer,
        histories, user_seen, rrf_k=60, meta_weight=0.15, sasrec_weight=2.0, top_k=TOP_K,
    )
    full_preds = parse_submission(full_df)

    preds = {
        "Popularity":    pop_preds,
        "BPR":           bpr_preds_10,
        "SASRec":        sasrec_preds_10,
        "Ensemble-RRF":  ensemble_preds,
        "Full-Pipeline": full_preds,
    }

    lines = ["MODEL COMPARISON (Recall@10, NDCG@10)", "",
             "  All rows scored on a GLOBAL temporal hold-out (no leakage).",
             "  Standalone and ensemble models share the same held-out candidates.",
             f"  Standalone rows use top-{TOP_K}; ensembles fuse top-{CANDIDATE_K} per model.", ""]
    for name, pred in preds.items():
        recall, ndcg = evaluate_predictions(pred, val)
        lines.append(f"{name:<16} Recall@10={recall:.5f}  NDCG@10={ndcg:.5f}")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote model comparison scores to {OUTPUT}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()