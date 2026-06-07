"""
Metadata-based item scoring utilities.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd


@dataclass
class MetaScorer:
    """
    Builds a content score for each item_id using info from item_meta.csv.

    Base score components (all normalised to [0, 1]):
      - quality:    avg_rating * log(rating_count + 1) — confidence-weighted quality
      - popularity: log(rating_count + 1) — social proof
      - bsr:        inverse log(Best Sellers Rank) — sales rank within category (33% coverage)

    Final base score = 0.45 * quality + 0.35 * popularity + 0.20 * bsr

    Per-user affinity signals (applied at query time):
      - subcategory affinity: full category path matched at deepest level possible
      - store affinity:       boost items from stores the user has bought from before
    """

    item_scores:      Dict[int, float]   # combined base score
    item_quality:     Dict[int, float]
    item_popularity:  Dict[int, float]
    item_bsr:         Dict[int, float]   # normalised BSR score (higher = better rank)
    item_categories:  Dict[int, str]     # main_category
    item_subcats:     Dict[int, List[str]]  # full category path list
    item_stores:      Dict[int, str]     # store name

    @classmethod
    def build(cls, item_meta: pd.DataFrame) -> "MetaScorer":
        meta = item_meta.copy()
        meta["item_id"] = meta["item_id"].astype(int)
        meta = meta.set_index("item_id")

        # --- Quality ---
        rating    = meta["average_rating"].fillna(meta["average_rating"].median())
        n_ratings = meta["rating_number"].fillna(0).clip(lower=0)
        confidence = np.log1p(n_ratings)
        norm_quality = _min_max((rating / 5.0) * confidence)

        # --- Popularity ---
        norm_pop = _min_max(np.log1p(n_ratings))

        # --- Best Sellers Rank (lower rank = better, invert) ---
        def _extract_bsr(details_str):
            try:
                d = ast.literal_eval(details_str)
                bsr = d.get("Best Sellers Rank", {})
                if isinstance(bsr, dict) and bsr:
                    return float(min(bsr.values()))
            except Exception:
                pass
            return np.nan

        raw_bsr  = meta["details"].apply(_extract_bsr)
        # Invert: low rank number = popular = high score
        inv_bsr  = 1.0 / np.log1p(raw_bsr.fillna(raw_bsr.max()))
        norm_bsr = _min_max(inv_bsr)

        # --- Combined base score ---
        score = (
            0.45 * norm_quality
            + 0.35 * norm_pop
            + 0.20 * norm_bsr
        ).fillna(0.0)

        # --- Subcategory paths ---
        def _parse_cats(cats_str):
            try:
                return ast.literal_eval(cats_str)
            except Exception:
                return []

        item_subcats: Dict[int, List[str]] = {
            int(iid): _parse_cats(row)
            for iid, row in meta["categories"].items()
        }

        # --- Stores ---
        item_stores: Dict[int, str] = meta["store"].fillna("Unknown").to_dict()

        return cls(
            item_scores     = score.to_dict(),
            item_quality    = norm_quality.to_dict(),
            item_popularity = norm_pop.to_dict(),
            item_bsr        = norm_bsr.to_dict(),
            item_categories = meta["main_category"].fillna("Unknown").to_dict(),
            item_subcats    = item_subcats,
            item_stores     = item_stores,
        )

    # ------------------------------------------------------------------
    # Affinity helpers
    # ------------------------------------------------------------------

    def user_category_affinity(self, history: List[int]) -> Dict[str, float]:
        """
        Subcategory-aware affinity: match at the deepest category level possible.
        Each item in history contributes its full path; counts are normalised.
        Unknown / missing categories are ignored.
        """
        if not history:
            return {}
        counts: Dict[str, int] = defaultdict(int)
        for iid in history:
            for cat in self.item_subcats.get(iid, []):
                if cat and cat != "Unknown":
                    counts[cat] += 1
        if not counts:
            return {}
        total = sum(counts.values())
        return {cat: cnt / total for cat, cnt in counts.items()}

    def user_store_affinity(self, history: List[int]) -> Dict[str, float]:
        """Normalised store engagement from interaction history."""
        if not history:
            return {}
        counts: Dict[str, int] = defaultdict(int)
        for iid in history:
            store = self.item_stores.get(iid, "Unknown")
            if store and store != "Unknown":
                counts[store] += 1
        if not counts:
            return {}
        total = sum(counts.values())
        return {store: cnt / total for store, cnt in counts.items()}

    def score(
        self,
        item_id: int,
        category_affinity: Optional[Dict[str, float]] = None,
        store_affinity:    Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Combined content score for a single item.
        Affinity boosts are additive on top of the base score.
        """
        base = self.item_scores.get(item_id, 0.0)

        if category_affinity:
            # Match against full subcategory path — deepest match wins
            subcats = self.item_subcats.get(item_id, [])
            cat_boost = max(
                (category_affinity.get(cat, 0.0) for cat in subcats),
                default=0.0,
            )
            base += cat_boost * 0.20

        if store_affinity:
            store = self.item_stores.get(item_id, "Unknown")
            base += store_affinity.get(store, 0.0) * 0.10

        return base


def _min_max(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)