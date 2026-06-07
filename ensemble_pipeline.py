"""
Ensemble and reranking utilities (RRF + meta boost).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from meta_scoring import MetaScorer
import pandas as pd


def rrf_merge(
    ranked_lists: List[List[int]],
    weights: Optional[List[float]] = None,
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Merge multiple ranked item lists using Reciprocal Rank Fusion.

    RRF score for item i  = sum_r  w_r / (k + rank_r(i))
    
    Parameters:
    ---
    ranked_lists : list of rank-ordered item-ID lists
    weights      : per-list multipliers (default: equal weights)
    k            : RRF smoothing constant, larger k lowers influence of top ranks

    Returns
    ---
    Sorted list of (item_id, score) tuples
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    scores: Dict[int, float] = defaultdict(float)
    for lst, w in zip(ranked_lists, weights):
        for rank, item_id in enumerate(lst, start=1):
            scores[item_id] += w / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def rerank_with_meta(
    candidates: List[Tuple[int, float]],
    scorer: MetaScorer,
    user_history: List[int],
    meta_weight: float = 0.15,
    seen: Optional[Set[int]] = None,
    k: int = 10,
) -> List[int]:
    """
    linear blend (item_id, base_score) candidate list from RRF with the meta score.

    final_score = (1 - meta_weight) * base_score
                   + meta_weight       * meta_score

    Items already seen by the user are excluded.

    Parameters
    ---
    candidates   : ranked list from rrf_merge
    scorer       : MetaScorer
    user_history : user's interaction history
    meta_weight  : blend coefficient in [0, 1]
    seen         : set of item_ids to exclude (already interacted)
    k            : number of recommendations to return
    """
    seen = seen or set()
    category_affinity = scorer.user_category_affinity(user_history)
    store_affinity    = scorer.user_store_affinity(user_history)

    # Normalise base scores to [0, 1] over this candidate set
    if candidates:
        max_base = max(s for _, s in candidates)
        min_base = min(s for _, s in candidates)
        rng = max_base - min_base if max_base != min_base else 1.0
    else:
        rng = 1.0

    results: List[Tuple[int, float]] = []
    for item_id, base_score in candidates:
        if item_id in seen: # skip seen items
            continue
        norm_base = (base_score - min_base) / rng if candidates else 0.0
        meta = scorer.score(item_id, category_affinity, store_affinity)
        #combine score
        final = (1.0 - meta_weight) * norm_base + meta_weight * meta
        results.append((item_id, final))

    results.sort(key=lambda x: x[1], reverse=True)
    return [iid for iid, _ in results[:k]]


def build_hybrid_submission(
    sample_sub,
    bpr_preds: Dict[int, List[int]],
    sasrec_preds: Dict[int, List[int]],
    scorer: MetaScorer,
    histories: Dict[int, List[int]],
    user_seen: Dict[int, Set[int]],
    rrf_k: int = 60,
    meta_weight: float = 0.15,
    sasrec_weight: float = 2.0,
    top_k: int = 10,
) -> "pd.DataFrame":
    
    """
    For every user in sample_submission:
      1. Gather BPR and SASRec ranked lists
      2. Fuse them with weighted RRF (SASRec gets higher weight)
      3. Rerank the fused candidate pool with content-aware blending
      4. Return top-k
    """
    rows = []
    for _, row in sample_sub.iterrows():
        uid = int(row["user_id"])
        bpr_list = bpr_preds.get(uid, [])
        sas_list = sasrec_preds.get(uid, [])

        merged = rrf_merge(
            [sas_list, bpr_list],
            weights=[sasrec_weight, 1.0],
            k=rrf_k,
        )

        history = histories.get(uid, [])
        seen = user_seen.get(uid, set())

        recs = rerank_with_meta(
            merged, scorer, history,
            meta_weight=meta_weight,
            seen=seen,
            k=top_k,
        )

        # Safety: fall back to SASRec then BPR if we still have too few recs
        fallback = [i for i in (sas_list + bpr_list) if i not in seen and i not in recs]
        while len(recs) < top_k and fallback:
            recs.append(fallback.pop(0))

        rows.append({
            "ID": int(row["ID"]),
            "user_id": uid,
            "item_id": ",".join(map(str, recs[:top_k])),
        })

    return pd.DataFrame(rows, columns=list(sample_sub.columns))