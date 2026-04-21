import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import argparse

class SASRecDataProcessor:
    def __init__(self, filepath, min_interactions=5):
        self.filepath = filepath
        self.min_interactions = int(min_interactions)
        self.num_users = 0
        self.num_items = 0

    def load_data(self):
        df = pd.read_csv(
            self.filepath,
            sep="::",
            engine="python",
            names=["user", "item", "rating", "timestamp"]
        )
        return df

    def preprocess(self):
        df = self.load_data()
        
        # Implicit feedback (rating >= 4): convert to binary target
        df = df[df["rating"] >= 4].copy()
        
        # Sort by time
        df = df.sort_values(by=["user", "timestamp", "item"]).reset_index(drop=True)
        
        # Build user sequences
        user_sequences = defaultdict(list)
        for row in df.itertuples(index=False):
            user_sequences[int(row.user)].append(int(row.item))
            
        # Filter users
        user_sequences = {
            user: seq for user, seq in user_sequences.items() if len(seq) >= self.min_interactions
        }
        
        # Re-index users and items
        sorted_users = sorted(user_sequences.keys())
        item_set = sorted({item for seq in user_sequences.values() for item in seq})

        user_map = {user_id: idx + 1 for idx, user_id in enumerate(sorted_users)}
        item_map = {item_id: idx + 1 for idx, item_id in enumerate(item_set)}

        mapped_sequences = {
            user_map[user_id]: [item_map[item] for item in seq]
            for user_id, seq in user_sequences.items()
        }

        self.num_users = len(user_map)
        self.num_items = len(item_map)
        return mapped_sequences

    def leave_one_out_split(self,sequences: Dict[int, List[int]],
        ) -> Tuple[Dict[int, List[int]], Dict[int, int], Dict[int, int]]:
        train_data, valid_data, test_data = {}, {}, {}
                               
        for user, seq in sequences.items():
            if len(seq) < 3:
                raise ValueError("Each filtered user must have at least 3 interactions for leave-one-out splitting.")
            train_data[user] = seq[:-2]
            valid_data[user] = seq[-2]
            test_data[user] = seq[-1]
            
        return train_data, valid_data, test_data

    def generate_training_instances(self, train_data: Dict[int, List[int]], max_len: int = 50,
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        user_ids, user_input, user_target = [], [], []

        for user, seq in train_data.items():
            for i in range(1, len(seq)):
                input_seq = seq[:i][-max_len:]
                padded_input = [0] * (max_len - len(input_seq)) + input_seq

                user_ids.append(user)
                user_input.append(padded_input)
                user_target.append(seq[i])

        return (
            np.asarray(user_ids, dtype=np.int64),
            np.asarray(user_input, dtype=np.int64),
            np.asarray(user_target, dtype=np.int64),
        )

    def save_data(
        self,
        save_dir: str,
        train: Dict[int, List[int]],
        valid: Dict[int, int],
        test: Dict[int, int],
        user_ids: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        os.makedirs(save_dir, exist_ok=True)

        # Save numpy arrays
        np.save(os.path.join(save_dir, "train_user_ids.npy"), user_ids)
        np.save(os.path.join(save_dir, "X_train.npy"), X_train)
        np.save(os.path.join(save_dir, "y_train.npy"), y_train)

        # Save sequences
        with open(os.path.join(save_dir, "train_sequences.json"), "w", encoding="utf-8") as f:
            json.dump(train, f)

        with open(os.path.join(save_dir, "valid_targets.json"), "w", encoding="utf-8") as f:
            json.dump(valid, f)
            
        with open(os.path.join(save_dir, "test_targets.json"), "w", encoding="utf-8") as f:
            json.dump(test, f)

        metadata = {
            "num_users": int(self.num_users),
            "num_items": int(self.num_items),
            "min_interactions": int(self.min_interactions),
        }
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"Data saved to {save_dir}")
        print(f"num_users={self.num_users}, num_items={self.num_items}, training_examples={len(X_train)}")

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess MovieLens-1M for SASRec")
    parser.add_argument("--data_path", type=str, default="ratings.dat")
    parser.add_argument("--save_dir", type=str, default="./processed_data")
    parser.add_argument("--min_interactions", type=int, default=5)
    parser.add_argument("--max_len", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    processor = SASRecDataProcessor(args.data_path, min_interactions=args.min_interactions)

    #Preprocess
    sequences = processor.preprocess()

    #Split
    train, valid, test = processor.leave_one_out_split(sequences)

    #Training pairs
    user_ids, X_train, y_train = processor.generate_training_instances(train, max_len=args.max_len)

    #Save everything
    processor.save_data(args.save_dir, train, valid, test, user_ids, X_train, y_train)
