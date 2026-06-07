"""
Hybrid Reranking & Ensemble Pipeline
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

from baseline_pipeline import (
    TOP_K,
    load_data,
    clean_train,
    create_submission,
)
from meta_scoring import MetaScorer
from ensemble_pipeline import build_hybrid_submission
from eval_utils import parse_submission, user_histories

RRF_K         = 60
META_WEIGHT   = 0.15
SASREC_WEIGHT = 2.0

DATA_DIR    = Path(".")
OUTPUT_PATH = DATA_DIR / "submission_hybrid.csv"


def main() -> None:
    print("[final_pipeline] Loading data …")
    train, _, item_meta, sample_sub = load_data(DATA_DIR)
    train_clean = clean_train(train)
    histories: Dict[int, List[int]] = user_histories(train_clean)
    user_seen: Dict[int, Set[int]] = {uid: set(items) for uid, items in histories.items()}

    print("[final_pipeline] Building MetaScorer …")
    scorer = MetaScorer.build(item_meta)

    bpr_path    = next((p for p in [DATA_DIR / "candidates_bpr.csv",    DATA_DIR / "submission_bpr.csv"]    if p.exists()), None)
    sasrec_path = next((p for p in [DATA_DIR / "candidates_sasrec.csv", DATA_DIR / "submission_sasrec.csv"] if p.exists()), None)
    bpr_sub    = pd.read_csv(bpr_path)    if bpr_path    else None
    sasrec_sub = pd.read_csv(sasrec_path) if sasrec_path else None

    if bpr_path:
        print(f"[final_pipeline] Using BPR candidates: {bpr_path.name}")
    if sasrec_path:
        print(f"[final_pipeline] Using SASRec candidates: {sasrec_path.name}")
    if bpr_sub is None:
        print("[final_pipeline] WARNING: no BPR submission found — BPR list will be empty.")
    if sasrec_sub is None:
        print("[final_pipeline] WARNING: no SASRec submission found — SASRec list will be empty.")

    bpr_preds    = parse_submission(bpr_sub)    if bpr_sub    is not None else {}
    sasrec_preds = parse_submission(sasrec_sub) if sasrec_sub is not None else {}

    print("[final_pipeline] Fusing submissions and reranking …")
    hybrid_df = build_hybrid_submission(
        sample_sub=sample_sub,
        bpr_preds=bpr_preds,
        sasrec_preds=sasrec_preds,
        scorer=scorer,
        histories=histories,
        user_seen=user_seen,
        rrf_k=RRF_K,
        meta_weight=META_WEIGHT,
        sasrec_weight=SASREC_WEIGHT,
        top_k=TOP_K,
    )
    hybrid_preds = parse_submission(hybrid_df)

    def recommend(user_id: int) -> List[int]:
        return hybrid_preds.get(user_id, [])[:TOP_K]

    create_submission(sample_sub, recommend, OUTPUT_PATH)


if __name__ == "__main__":
    main()
    META_WEIGHT   = 0.0
    OUTPUT_PATH = DATA_DIR / "submission_hybrid_no_meta.csv"
    main()