from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.preprocessing import LabelEncoder


TOP_K = 10
RANDOM_STATE = 42
DEFAULT_COMPONENTS = 32
DEFAULT_MAX_VAL_USERS = 3000


@dataclass
class EncodedData:
    matrix: csr_matrix
    user_encoder: LabelEncoder
    item_encoder: LabelEncoder
    user_idx: np.ndarray
    item_idx: np.ndarray
    user_seen_idx: Dict[int, Set[int]]
    user_seen_items: Dict[int, Set[int]]
    known_users: Set[int]
    idx_to_item: np.ndarray


# Data loading and preprocessing
def load_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = ["train.csv", "test.csv", "item_meta.csv", "sample_submission.csv"]
    missing = [name for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {data_dir}: {missing}")

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    item_meta = pd.read_csv(data_dir / "item_meta.csv")
    sample_submission = pd.read_csv(data_dir / "sample_submission.csv")
    return train, test, item_meta, sample_submission

# Clean implicit-feedback interactions
def clean_train(train: pd.DataFrame) -> pd.DataFrame:

    required = {"user_id", "item_id", "timestamp"}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing)}")

    df = train.copy()
    for col in ["user_id", "item_id", "timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["user_id", "item_id", "timestamp"])
    df["user_id"] = df["user_id"].astype(np.int64)
    df["item_id"] = df["item_id"].astype(np.int64)
    df["timestamp"] = df["timestamp"].astype(np.int64)

    # Keep the most recent event for repeated user-item interactions.
    df = df.sort_values(["user_id", "item_id", "timestamp"])
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last")

    return df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

# Global temporal validation split (mirrors Kaggle's future-prediction test).
def chronological_split(
    train: pd.DataFrame,
    max_val_users: int | None = DEFAULT_MAX_VAL_USERS,
    val_quantile: float = 0.90,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
   
    df = train.sort_values(["user_id", "timestamp"])
    cutoff = float(np.quantile(df["timestamp"].to_numpy(), val_quantile))

    train_split = df[df["timestamp"] < cutoff]
    val = df[df["timestamp"] >= cutoff]

    # Only score users the model has seen before the cutoff.
    seen_users = set(train_split["user_id"].unique())
    val = val[val["user_id"].isin(seen_users)]

    if max_val_users is not None and val["user_id"].nunique() > max_val_users:
        keep = (val["user_id"].drop_duplicates()
                .sample(max_val_users, random_state=RANDOM_STATE))
        val = val[val["user_id"].isin(set(keep))]

    return train_split.reset_index(drop=True), val.reset_index(drop=True)


# Shared utilities
def make_user_seen_items(train_data: pd.DataFrame) -> Dict[int, Set[int]]:
    return train_data.groupby("user_id")["item_id"].apply(lambda s: set(map(int, s))).to_dict()


def make_popularity(train_data: pd.DataFrame) -> np.ndarray:
    return train_data["item_id"].value_counts().index.astype(int).to_numpy()


def encode_interactions(train_data: pd.DataFrame) -> EncodedData:
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    user_idx = user_encoder.fit_transform(train_data["user_id"])
    item_idx = item_encoder.fit_transform(train_data["item_id"])

    matrix = csr_matrix(
        (np.ones(len(train_data), dtype=np.float32), (user_idx, item_idx)),
        shape=(len(user_encoder.classes_), len(item_encoder.classes_)),
    )

    user_seen_idx: Dict[int, Set[int]] = {}
    for u, i in zip(user_idx, item_idx):
        user_seen_idx.setdefault(int(u), set()).add(int(i))

    return EncodedData(
        matrix=matrix,
        user_encoder=user_encoder,
        item_encoder=item_encoder,
        user_idx=user_idx,
        item_idx=item_idx,
        user_seen_idx=user_seen_idx,
        user_seen_items=make_user_seen_items(train_data),
        known_users=set(map(int, user_encoder.classes_)),
        idx_to_item=item_encoder.classes_.astype(int),
    )


def recommend_from_ranked(user_id: int, user_seen: Dict[int, Set[int]], ranked_items: Sequence[int], k: int = TOP_K) -> List[int]:
    seen = user_seen.get(int(user_id), set())
    recs: List[int] = []
    for item in ranked_items:
        item = int(item)
        if item not in seen and item not in recs:
            recs.append(item)
            if len(recs) == k:
                break
    return recs


def hit_rate_at_k(recs: Sequence[int], true_item: int) -> int:
    return int(int(true_item) in recs)


def ndcg_at_k(recs: Sequence[int], true_item: int) -> float:
    true_item = int(true_item)
    if true_item not in recs:
        return 0.0
    rank = recs.index(true_item) + 1
    return float(1.0 / np.log2(rank + 1))


def evaluate_recommender(val: pd.DataFrame, recommend_fn, name: str) -> Tuple[float, float]:
    hits, ndcgs = [], []
    for row in val.itertuples(index=False):
        recs = recommend_fn(int(row.user_id))
        hits.append(hit_rate_at_k(recs, int(row.item_id)))
        ndcgs.append(ndcg_at_k(recs, int(row.item_id)))
    hr = float(np.mean(hits)) if hits else 0.0
    ndcg = float(np.mean(ndcgs)) if ndcgs else 0.0
    print(f"{name:<18} HitRate@{TOP_K}: {hr:.5f} | NDCG@{TOP_K}: {ndcg:.5f}")
    return hr, ndcg


# BM25-style popularity 
def bm25_weight_matrix(matrix: csr_matrix, k1: float = 1.2, b: float = 0.75) -> csr_matrix:
  
    X = matrix.astype(np.float32).tocsr(copy=True)
    n_users, _ = X.shape
    user_lengths = np.asarray(X.sum(axis=1)).ravel()
    avg_len = float(user_lengths.mean()) if user_lengths.size else 1.0
    avg_len = max(avg_len, 1e-6)

    
    df = np.diff(X.tocsc().indptr).astype(np.float32)
    idf = np.log((n_users - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)

    X = X.tocsr()
    for u in range(n_users):
        start, end = X.indptr[u], X.indptr[u + 1]
        if start == end:
            continue
        denom = X.data[start:end] + k1 * (1.0 - b + b * user_lengths[u] / avg_len)
        X.data[start:end] = X.data[start:end] * (k1 + 1.0) / denom
    X = X.multiply(idf).tocsr()
    return X


def make_bm25_popularity(train_data: pd.DataFrame) -> np.ndarray:
    enc = encode_interactions(train_data)
    weighted = bm25_weight_matrix(enc.matrix)
    scores = np.asarray(weighted.sum(axis=0)).ravel()
    order = np.argsort(-scores)
    return enc.item_encoder.inverse_transform(order).astype(int)


# SVD and NMF matrix-factorization 
def fit_svd(train_data: pd.DataFrame, n_components: int) -> Tuple[EncodedData, np.ndarray, np.ndarray]:
    enc = encode_interactions(train_data)
    n_components = min(n_components, max(1, min(enc.matrix.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(enc.matrix).astype(np.float32)
    item_factors = svd.components_.T.astype(np.float32)
    return enc, user_factors, item_factors


def fit_nmf(train_data: pd.DataFrame, n_components: int, max_iter: int = 100) -> Tuple[EncodedData, np.ndarray, np.ndarray]:
    enc = encode_interactions(train_data)
    n_components = min(n_components, max(1, min(enc.matrix.shape) - 1))
    model = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=RANDOM_STATE,
        max_iter=max_iter,
        alpha_W=0.0,
        alpha_H=0.0,
        l1_ratio=0.0,
    )
    user_factors = model.fit_transform(enc.matrix).astype(np.float32)
    item_factors = model.components_.T.astype(np.float32)
    return enc, user_factors, item_factors


def recommend_factor_model(
    user_id: int,
    enc: EncodedData,
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    fallback_items: Sequence[int],
    k: int = TOP_K,
) -> List[int]:
    user_id = int(user_id)
    if user_id not in enc.known_users:
        return recommend_from_ranked(user_id, enc.user_seen_items, fallback_items, k)

    uidx = int(enc.user_encoder.transform([user_id])[0])
    scores = user_factors[uidx] @ item_factors.T
    seen_idx = list(enc.user_seen_idx.get(uidx, set()))
    if seen_idx:
        scores[seen_idx] = -np.inf

    candidate_count = min(k + len(seen_idx), len(scores))
    if candidate_count <= 0:
        return recommend_from_ranked(user_id, enc.user_seen_items, fallback_items, k)

    top_idx = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
    top_idx = top_idx[np.argsort(-scores[top_idx])]

    recs: List[int] = []
    for idx in top_idx:
        if np.isfinite(scores[idx]):
            recs.append(int(enc.idx_to_item[idx]))
            if len(recs) == k:
                break

    if len(recs) < k:
        for item in fallback_items:
            item = int(item)
            if item not in enc.user_seen_items.get(user_id, set()) and item not in recs:
                recs.append(item)
                if len(recs) == k:
                    break
    return recs[:k]


# BPR matrix factorization 
def sample_negative_item(rng: np.random.Generator, n_items: int, seen: Set[int]) -> int:
    while True:
        j = int(rng.integers(0, n_items))
        if j not in seen:
            return j


def train_bpr(
    train_data: pd.DataFrame,
    n_factors: int = DEFAULT_COMPONENTS,
    epochs: int = 5,
    learning_rate: float = 0.05,
    reg: float = 0.002,
) -> Tuple[EncodedData, np.ndarray, np.ndarray]:
    enc = encode_interactions(train_data)
    n_users, n_items = enc.matrix.shape
    rng = np.random.default_rng(RANDOM_STATE)

    user_factors = 0.01 * rng.standard_normal((n_users, n_factors)).astype(np.float32)
    item_factors = 0.01 * rng.standard_normal((n_items, n_factors)).astype(np.float32)

    interactions = np.column_stack([enc.user_idx, enc.item_idx]).astype(np.int64)

    for epoch in range(epochs):
        rng.shuffle(interactions)
        total_loss = 0.0
        for u, i in interactions:
            u = int(u)
            i = int(i)
            j = sample_negative_item(rng, n_items, enc.user_seen_idx[u])

            u_vec = user_factors[u].copy()
            i_vec = item_factors[i].copy()
            j_vec = item_factors[j].copy()

            x_uij = float(u_vec @ (i_vec - j_vec))
            grad = 1.0 / (1.0 + np.exp(np.clip(x_uij, -35, 35)))

            user_factors[u] += learning_rate * (grad * (i_vec - j_vec) - reg * u_vec)
            item_factors[i] += learning_rate * (grad * u_vec - reg * i_vec)
            item_factors[j] += learning_rate * (-grad * u_vec - reg * j_vec)

            total_loss += np.log1p(np.exp(-np.clip(x_uij, -35, 35)))

        print(f"BPR epoch {epoch + 1}/{epochs} - avg pairwise loss: {total_loss / len(interactions):.5f}")

    return enc, user_factors.astype(np.float32), item_factors.astype(np.float32)

# Submission 
def validate_submission(submission: pd.DataFrame, sample_submission: pd.DataFrame) -> None:
    if list(submission.columns) != list(sample_submission.columns):
        raise ValueError(f"Submission columns {list(submission.columns)} do not match sample {list(sample_submission.columns)}")
    if len(submission) != len(sample_submission):
        raise ValueError("Submission row count does not match sample_submission.csv")
    lengths = submission["item_id"].astype(str).str.split(",").apply(len)
    if not (lengths == TOP_K).all():
        bad = int((lengths != TOP_K).sum())
        raise ValueError(f"{bad} rows do not contain exactly {TOP_K} recommendations")


def create_submission(
    sample_submission: pd.DataFrame,
    recommend_fn,
    output_path: Path,
) -> pd.DataFrame:
    rows = []
    for row in sample_submission.itertuples(index=False):
        recs = recommend_fn(int(row.user_id))
        rows.append({"ID": int(row.ID), "user_id": int(row.user_id), "item_id": ",".join(map(str, recs))})
    submission = pd.DataFrame(rows, columns=list(sample_submission.columns))
    validate_submission(submission, sample_submission)
    submission.to_csv(output_path, index=False)
    print(f"Saved submission to: {output_path}")
    return submission

# Main pipeline
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teammate 1 recommender baseline pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("."), help="Folder containing train/test/item_meta/sample_submission CSVs")
    parser.add_argument("--output", type=Path, default=None, help="Output submission CSV path")
    parser.add_argument("--components", type=int, default=DEFAULT_COMPONENTS, help="Latent dimensions for SVD/NMF/BPR")
    parser.add_argument("--max-val-users", type=int, default=DEFAULT_MAX_VAL_USERS, help="Validation user sample size; use -1 for full validation")
    parser.add_argument("--bpr-epochs", type=int, default=5, help="Number of BPR training epochs")
    parser.add_argument("--skip-nmf", action="store_true", help="Skip NMF if runtime is too high")
    parser.add_argument("--final-model", choices=["svd", "nmf", "bpr", "bm25"], default="bpr", help="Model used for final Kaggle submission")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_val_users = None if args.max_val_users == -1 else args.max_val_users
    output_path = args.output or (args.data_dir / f"submission_{args.final_model}.csv")

    train, test, item_meta, sample_submission = load_data(args.data_dir)

    print("Raw shapes")
    print("train:", train.shape, "test:", test.shape, "item_meta:", item_meta.shape, "sample:", sample_submission.shape)
    print("train columns:", list(train.columns))
    print("sample submission columns:", list(sample_submission.columns))

    train_clean = clean_train(train)
    print("\nPreprocessing")
    print("cleaned train shape:", train_clean.shape)
    print("removed duplicate/invalid rows:", len(train) - len(train_clean))
    print("timestamp range:", int(train_clean["timestamp"].min()), "to", int(train_clean["timestamp"].max()))

    train_split, val = chronological_split(train_clean, max_val_users=max_val_users)
    print("\nChronological split")
    print("train split:", train_split.shape, "validation:", val.shape)

    user_seen_split = make_user_seen_items(train_split)
    pop_items = make_popularity(train_split)
    bm25_items = make_bm25_popularity(train_split)

    print("\nValidation metrics")
    evaluate_recommender(val, lambda u: recommend_from_ranked(u, user_seen_split, pop_items), "Popularity")
    evaluate_recommender(val, lambda u: recommend_from_ranked(u, user_seen_split, bm25_items), "BM25Popularity")

    svd_enc, svd_uf, svd_if = fit_svd(train_split, args.components)
    evaluate_recommender(
        val,
        lambda u: recommend_factor_model(u, svd_enc, svd_uf, svd_if, bm25_items),
        "SVD",
    )

    nmf_pack = None
    if not args.skip_nmf:
        nmf_pack = fit_nmf(train_split, args.components)
        nmf_enc, nmf_uf, nmf_if = nmf_pack
        evaluate_recommender(
            val,
            lambda u: recommend_factor_model(u, nmf_enc, nmf_uf, nmf_if, bm25_items),
            "NMF",
        )

    bpr_enc, bpr_uf, bpr_if = train_bpr(train_split, n_factors=args.components, epochs=args.bpr_epochs)
    evaluate_recommender(
        val,
        lambda u: recommend_factor_model(u, bpr_enc, bpr_uf, bpr_if, bm25_items),
        "BPR-MF",
    )

    print("\nTraining final model on full train.csv only")
    final_bm25_items = make_bm25_popularity(train_clean)

    if args.final_model == "bm25":
        final_user_seen = make_user_seen_items(train_clean)
        final_recommend = lambda u: recommend_from_ranked(u, final_user_seen, final_bm25_items)
    elif args.final_model == "svd":
        enc, uf, itf = fit_svd(train_clean, args.components)
        final_recommend = lambda u: recommend_factor_model(u, enc, uf, itf, final_bm25_items)
    elif args.final_model == "nmf":
        enc, uf, itf = fit_nmf(train_clean, args.components)
        final_recommend = lambda u: recommend_factor_model(u, enc, uf, itf, final_bm25_items)
    else:
        enc, uf, itf = train_bpr(train_clean, n_factors=args.components, epochs=args.bpr_epochs)
        final_recommend = lambda u: recommend_factor_model(u, enc, uf, itf, final_bm25_items)

    submission = create_submission(sample_submission, final_recommend, output_path)
    print(submission.head())


if __name__ == "__main__":
    main()
