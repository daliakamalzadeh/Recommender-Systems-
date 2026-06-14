from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baseline_pipeline import load_data, clean_train, encode_interactions
from eval_utils import parse_submission, user_histories
from meta_scoring import MetaScorer
from ensemble_pipeline import rrf_merge, rerank_with_meta

DATA = Path(".")
TOP_K = 10
CAND_K = 30                 

# EASE 
EASE_LAMBDA = 500         
USE_FLOAT64 = False       

# RRF fusion weights 
RRF_K        = 60
SAS_WEIGHT   = 2.0
BPR_WEIGHT   = 1.0
EASE_WEIGHT  = 1.0   
META_WEIGHT  = 0.15         

def build_ease_weights(matrix, lam: float) -> np.ndarray:
    dtype = np.float64 if USE_FLOAT64 else np.float32
    G = (matrix.T @ matrix).toarray().astype(dtype)          
    diag_idx = np.diag_indices_from(G)
    G[diag_idx] += lam
    P = np.linalg.inv(G)
    B = P / (-np.diag(P))[np.newaxis, :]
    B[diag_idx] = 0.0
    return B


def ease_candidates(enc, B, sub_uids, k: int):
    known = [u for u in sub_uids if u in enc.known_users]
    if not known:
        return {}
    uidxs = enc.user_encoder.transform(known)
    scores = (enc.matrix[uidxs] @ B)                          
    scores = np.asarray(scores)
    preds = {}
    for row_i, uid in enumerate(known):
        s = scores[row_i].copy()
        seen = enc.user_seen_idx.get(int(uidxs[row_i]), set())
        if seen:
            s[list(seen)] = -np.inf
        n = min(k, s.size)
        top = np.argpartition(-s, n - 1)[:n]
        top = top[np.argsort(-s[top])]
        preds[int(uid)] = [int(enc.idx_to_item[j]) for j in top if np.isfinite(s[j])]
    return preds


def main() -> None:
    train, _, item_meta, sample_sub = load_data(DATA)
    train_clean = clean_train(train)
    enc = encode_interactions(train_clean)

    histories = user_histories(train_clean)
    user_seen = {u: set(v) for u, v in histories.items()}
    scorer = MetaScorer.build(item_meta)

    sas = parse_submission(pd.read_csv(DATA / "candidates_sasrec.csv"))
    bpr = parse_submission(pd.read_csv(DATA / "candidates_bpr.csv"))

    print(f"[ease] building EASE weights (lambda={EASE_LAMBDA}, "
          f"{'float64' if USE_FLOAT64 else 'float32'}) …")
    B = build_ease_weights(enc.matrix, EASE_LAMBDA)
    sub_uids = [int(u) for u in sample_sub["user_id"].unique()]
    ease = ease_candidates(enc, B, sub_uids, CAND_K)
    print(f"[ease] EASE candidates for {len(ease)}/{len(sub_uids)} users")

    rows = []
    for row in sample_sub.itertuples(index=False):
        uid = int(row.user_id)
        sas_list  = sas.get(uid, [])
        bpr_list  = bpr.get(uid, [])
        ease_list = ease.get(uid, [])

        merged = rrf_merge(
            [sas_list, bpr_list, ease_list],
            weights=[SAS_WEIGHT, BPR_WEIGHT, EASE_WEIGHT],
            k=RRF_K,
        )
        seen = user_seen.get(uid, set())
        recs = rerank_with_meta(
            merged, scorer, histories.get(uid, []),
            meta_weight=META_WEIGHT, seen=seen, k=TOP_K,
        )

        fallback = [i for i in (sas_list + bpr_list + ease_list)
                    if i not in seen and i not in recs]
        while len(recs) < TOP_K and fallback:
            recs.append(fallback.pop(0))

        rows.append({"ID": int(row.ID), "user_id": uid,
                     "item_id": ",".join(map(str, recs[:TOP_K]))})

    out = pd.DataFrame(rows, columns=list(sample_sub.columns))
    assert (out["item_id"].str.split(",").apply(len) == TOP_K).all(), "not all rows have 10 items"
    out_path = DATA / "submission_ease_ensemble.csv"
    out.to_csv(out_path, index=False)
    print(f"[ease] wrote {out_path}  shape={out.shape}")


if __name__ == "__main__":
    main()
