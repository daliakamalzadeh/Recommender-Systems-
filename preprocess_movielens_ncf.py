from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class PreprocessConfig:
    ratings_path: str
    output_dir: str
    positive_threshold: int = 4
    negatives_per_positive: int = 1
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description="Preprocess MovieLens 1M for NCF.")
    parser.add_argument("--ratings_path", type=str, default="ml-1m/ratings.dat")
    parser.add_argument("--output_dir", type=str, default="ncf_preprocessed")
    parser.add_argument("--positive_threshold", type=int, default=4)
    parser.add_argument("--negatives_per_positive", type=int, default=1)
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    return PreprocessConfig(
        ratings_path=args.ratings_path,
        output_dir=args.output_dir,
        positive_threshold=args.positive_threshold,
        negatives_per_positive=args.negatives_per_positive,
        random_seed=args.random_seed,
    )


def load_ratings(ratings_path: str) -> pd.DataFrame:
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"ratings.dat not found at: {ratings_path}")

    return pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=["userId", "movieId", "rating", "timestamp"],
    )


def remap_ids(ratings_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, int], Dict[int, int]]:
    df = ratings_df.copy()
    unique_users = sorted(df["userId"].unique())
    unique_items = sorted(df["movieId"].unique())

    user2id = {original_id: new_id for new_id, original_id in enumerate(unique_users)}
    item2id = {original_id: new_id for new_id, original_id in enumerate(unique_items)}

    df["userId"] = df["userId"].map(user2id)
    df["movieId"] = df["movieId"].map(item2id)

    return df, user2id, item2id


def convert_to_implicit_feedback(ratings_df: pd.DataFrame, positive_threshold: int) -> pd.DataFrame:
    positive_df = ratings_df[ratings_df["rating"] >= positive_threshold].copy()
    positive_df["label"] = 1
    return positive_df[["userId", "movieId", "label"]]


def build_user_positive_sets(positive_df: pd.DataFrame) -> Dict[int, Set[int]]:
    return positive_df.groupby("userId")["movieId"].apply(set).to_dict()


def generate_negative_samples(
    user_positive_items: Dict[int, Set[int]],
    num_items: int,
    negatives_per_positive: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    all_items = np.arange(num_items)
    negative_rows: List[List[int]] = []

    for user_id, positive_items in user_positive_items.items():
        num_negatives = negatives_per_positive * len(positive_items)
        candidate_negatives = np.array(list(set(all_items) - positive_items))
        replace_flag = len(candidate_negatives) < num_negatives

        sampled_negatives = rng.choice(
            candidate_negatives,
            size=num_negatives,
            replace=replace_flag,
        )

        for item_id in sampled_negatives:
            negative_rows.append([user_id, int(item_id), 0])

    return pd.DataFrame(negative_rows, columns=["userId", "movieId", "label"])


def random_split(
    full_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")

    train_df, temp_df = train_test_split(
        full_df,
        test_size=(1.0 - train_ratio),
        random_state=random_seed,
        shuffle=True,
        stratify=full_df["label"],
    )

    relative_test_size = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_seed,
        shuffle=True,
        stratify=temp_df["label"],
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_outputs(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user2id: Dict[int, int],
    item2id: Dict[int, int],
    config: PreprocessConfig,
    original_ratings_count: int,
    positive_count: int,
    negative_count: int,
) -> None:
    os.makedirs(config.output_dir, exist_ok=True)

    train_path = os.path.join(config.output_dir, "train.csv")
    val_path = os.path.join(config.output_dir, "val.csv")
    test_path = os.path.join(config.output_dir, "test.csv")
    users_map_path = os.path.join(config.output_dir, "user2id.json")
    items_map_path = os.path.join(config.output_dir, "item2id.json")
    meta_path = os.path.join(config.output_dir, "metadata.json")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    with open(users_map_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in user2id.items()}, f, indent=2)

    with open(items_map_path, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in item2id.items()}, f, indent=2)

    metadata = {
        "ratings_path": config.ratings_path,
        "positive_threshold": config.positive_threshold,
        "negatives_per_positive": config.negatives_per_positive,
        "random_seed": config.random_seed,
        "split_ratios": {
            "train": config.train_ratio,
            "validation": config.val_ratio,
            "test": config.test_ratio,
        },
        "num_original_ratings": original_ratings_count,
        "num_positive_interactions": positive_count,
        "num_negative_interactions": negative_count,
        "num_total_interactions": positive_count + negative_count,
        "num_users": len(user2id),
        "num_items": len(item2id),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    config = parse_args()
    rng = np.random.default_rng(config.random_seed)

    ratings_df = load_ratings(config.ratings_path)
    original_ratings_count = len(ratings_df)

    ratings_df, user2id, item2id = remap_ids(ratings_df)
    num_items = len(item2id)

    positive_df = convert_to_implicit_feedback(ratings_df, config.positive_threshold)
    positive_count = len(positive_df)

    user_positive_items = build_user_positive_sets(positive_df)
    negative_df = generate_negative_samples(
        user_positive_items=user_positive_items,
        num_items=num_items,
        negatives_per_positive=config.negatives_per_positive,
        rng=rng,
    )
    negative_count = len(negative_df)

    full_df = pd.concat([positive_df, negative_df], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=config.random_seed).reset_index(drop=True)

    train_df, val_df, test_df = random_split(
        full_df=full_df,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        random_seed=config.random_seed,
    )

    save_outputs(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        user2id=user2id,
        item2id=item2id,
        config=config,
        original_ratings_count=original_ratings_count,
        positive_count=positive_count,
        negative_count=negative_count,
    )

    print(f"Saved to: {config.output_dir}")


if __name__ == "__main__":
    main()