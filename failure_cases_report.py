"""
Failure case report for the full pipeline.
Reflects the actual deployed pipeline (30 candidates per model -> rerank to 10).
Writes failure_cases.txt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from baseline_pipeline import clean_train, chronological_split, make_user_seen_items
from ensemble_pipeline import rrf_merge, build_hybrid_submission
from eval_utils import ndcg_at_k, parse_submission, user_histories, load_all
from meta_scoring import MetaScorer

TOP_K         = 10
CANDIDATE_K   = 30
RRF_K         = 60
SASREC_WEIGHT = 2.0
META_WEIGHT   = 0.15
OUTPUT        = Path("failure_cases.txt")

VAL_BPR_PATH    = Path("val_candidates_bpr.csv")
VAL_SASREC_PATH = Path("val_candidates_sasrec.csv")


def _score_item(
    item_id: int,
    rrf_score: float,
    rrf_rank: int,
    bpr_rank: Optional[int],
    sasrec_rank: Optional[int],
    scorer: MetaScorer,
    category_affinity: Dict[str, float],
    store_affinity: Dict[str, float],
    min_base: float,
    rng: float,
) -> Dict:
    norm_base  = (rrf_score - min_base) / rng
    cat        = scorer.item_categories.get(item_id, "Unknown")
    subcats    = scorer.item_subcats.get(item_id, [])
    cat_boost  = max((category_affinity.get(c, 0.0) for c in subcats), default=0.0)
    store      = scorer.item_stores.get(item_id, "Unknown")
    store_boost = store_affinity.get(store, 0.0)
    quality    = scorer.item_quality.get(item_id, 0.0)
    popularity = scorer.item_popularity.get(item_id, 0.0)
    bsr        = scorer.item_bsr.get(item_id, 0.0)
    meta       = scorer.item_scores.get(item_id, 0.0) + cat_boost * 0.20 + store_boost * 0.10
    final      = (1.0 - META_WEIGHT) * norm_base + META_WEIGHT * meta
    return {
        "rrf_score":   float(rrf_score),
        "bpr_rank":    bpr_rank,
        "sasrec_rank": sasrec_rank,
        "quality":     float(quality),
        "popularity":  float(popularity),
        "bsr":         float(bsr),
        "cat_affinity": float(cat_boost),
        "store_affinity": float(store_boost),
        "final_score": float(final),
        "category":    cat,
    }


def _scored_recs(
    bpr_list: List[int],
    sas_list: List[int],
    scorer: MetaScorer,
    history: List[int],
    seen: Set[int],
) -> List[Dict]:
    merged = rrf_merge([sas_list, bpr_list], weights=[SASREC_WEIGHT, 1.0], k=RRF_K)

    bpr_ranks    = {iid: r + 1 for r, iid in enumerate(bpr_list)}
    sasrec_ranks = {iid: r + 1 for r, iid in enumerate(sas_list)}

    if merged:
        max_base = max(s for _, s in merged)
        min_base = min(s for _, s in merged)
        rng = max_base - min_base if max_base != min_base else 1.0
    else:
        min_base = rng = 1.0

    category_affinity = scorer.user_category_affinity(history)
    store_affinity    = scorer.user_store_affinity(history)

    all_scored: List[Tuple[int, Dict]] = []
    for rank, (item_id, base_score) in enumerate(merged):
        if item_id in seen:
            continue
        info = _score_item(
            item_id, base_score, rank + 1,
            bpr_ranks.get(item_id), sasrec_ranks.get(item_id),
            scorer, category_affinity, store_affinity, min_base, rng,
        )
        all_scored.append((item_id, info))

    all_scored.sort(key=lambda x: x[1]["final_score"], reverse=True)
    return [{"item_id": int(iid), **info} for iid, info in all_scored[:TOP_K]]


def main() -> None:
    if not VAL_BPR_PATH.exists() or not VAL_SASREC_PATH.exists():
        print("[failure_cases] val candidate files not found — run generate_candidates.py first.")
        return

    train, _, item_meta, _, _, _ = load_all(Path("."))
    train_clean = clean_train(train)
    train_split, val = chronological_split(train_clean)

    histories = user_histories(train_split)
    user_seen = make_user_seen_items(train_split)
    scorer    = MetaScorer.build(item_meta)

    bpr_preds    = parse_submission(pd.read_csv(VAL_BPR_PATH))
    sasrec_preds = parse_submission(pd.read_csv(VAL_SASREC_PATH))

    val_uids   = list(val["user_id"].unique())
    val_sample = pd.DataFrame({"ID": range(len(val_uids)), "user_id": val_uids, "item_id": 0})

    full_df = build_hybrid_submission(
        val_sample, bpr_preds, sasrec_preds, scorer,
        histories, user_seen, rrf_k=RRF_K, meta_weight=META_WEIGHT,
        sasrec_weight=SASREC_WEIGHT, top_k=TOP_K,
    )
    full_preds = parse_submission(full_df)

    scored_users: List[Tuple[float, int, int]] = []
    for row in val.itertuples(index=False):
        uid = int(row.user_id)
        true_item = int(row.item_id)
        score = ndcg_at_k(full_preds.get(uid, []), true_item)
        scored_users.append((score, uid, true_item))

    scored_users.sort(key=lambda x: (x[0], x[1]))

    lines = ["FAILURE CASES (3 worst by NDCG@10)", ""]
    for score, uid, true_item in scored_users[:3]:
        history = histories.get(uid, [])
        seen    = user_seen.get(uid, set())
        lines.append(f"User {uid} | true_item={true_item} | ndcg@10={score:.5f} | history_len={len(history)}")
        top = _scored_recs(bpr_preds.get(uid, []), sasrec_preds.get(uid, []), scorer, history, seen)
        header = f"  {'rank':<5} {'item_id':<9} {'category':<28} {'rrf_score':<11} {'bpr_rank':<10} {'sasrec_rank':<12} {'quality':<9} {'popularity':<11} {'bsr':<7} {'cat_aff':<9} {'store_aff':<11} {'final_score'}"
        lines.append(header)
        lines.append("  " + "-" * 140)
        for final_rank, r in enumerate(top, 1):
            bpr_r    = str(r["bpr_rank"])    if r["bpr_rank"]    is not None else "-"
            sasrec_r = str(r["sasrec_rank"]) if r["sasrec_rank"] is not None else "-"
            lines.append(
                f"  {final_rank:<5} {r['item_id']:<9} {r['category']:<28} "
                f"{r['rrf_score']:<11.4f} {bpr_r:<10} {sasrec_r:<12} "
                f"{r['quality']:<9.4f} {r['popularity']:<11.4f} {r['bsr']:<7.4f} "
                f"{r['cat_affinity']:<9.4f} {r['store_affinity']:<11.4f} {r['final_score']:.4f}"
            )
        lines.append("")

    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote failure cases to {OUTPUT}")


if __name__ == "__main__":
    main()