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
    seen = seen or set()
    category_affinity = scorer.user_category_affinity(user_history)
    store_affinity    = scorer.user_store_affinity(user_history)

    if candidates:
        max_base = max(s for _, s in candidates)
        min_base = min(s for _, s in candidates)
        rng = max_base - min_base if max_base != min_base else 1.0
    else:
        rng = 1.0

    results: List[Tuple[int, float]] = []
    for item_id, base_score in candidates:
        if item_id in seen: 
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

        fallback = [i for i in (sas_list + bpr_list) if i not in seen and i not in recs]
        while len(recs) < top_k and fallback:
            recs.append(fallback.pop(0))

        rows.append({
            "ID": int(row["ID"]),
            "user_id": uid,
            "item_id": ",".join(map(str, recs[:top_k])),
        })

    return pd.DataFrame(rows, columns=list(sample_sub.columns))