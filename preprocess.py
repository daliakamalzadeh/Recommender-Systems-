import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict


class SASRecDataProcessor:
    def __init__(self, filepath, min_interactions=5):
        self.filepath = filepath
        self.min_interactions = min_interactions

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
        df = df[df["rating"] >= 4]

        # Sort by time
        df = df.sort_values(by=["user", "timestamp"])

        # Build user sequences
        user_sequences = defaultdict(list)
        for row in df.itertuples():
            user_sequences[row.user].append(row.item)

        # Filter users
        user_sequences = {
            u: seq for u, seq in user_sequences.items()
            if len(seq) >= self.min_interactions
        }

        # Re-index users and items
        user_map = {u: i + 1 for i, u in enumerate(user_sequences.keys())}

        item_set = set()
        for seq in user_sequences.values():
            item_set.update(seq)

        item_map = {i: idx + 1 for idx, i in enumerate(item_set)}

        mapped_sequences = {}
        for u, seq in user_sequences.items():
            mapped_sequences[user_map[u]] = [item_map[i] for i in seq]

        self.num_users = len(user_map)
        self.num_items = len(item_map)

        return mapped_sequences

    def leave_one_out_split(self, sequences):
        train_data, valid_data, test_data = {}, {}, {}

        for user, seq in sequences.items():
            train_data[user] = seq[:-2]
            valid_data[user] = seq[-2]
            test_data[user] = seq[-1]

        return train_data, valid_data, test_data

    def generate_training_instances(self, train_data, max_len=50):
        user_input = []
        user_target = []

        for user, seq in train_data.items():
            for i in range(1, len(seq)):
                input_seq = seq[:i]
                target = seq[i]

                # truncate and pad sequences
                input_seq = input_seq[-max_len:]
                pad_len = max_len - len(input_seq)
                input_seq = [0] * pad_len + input_seq

                user_input.append(input_seq)
                user_target.append(target)

        return np.array(user_input), np.array(user_target)

    def save_data(self, save_dir, train, valid, test, X_train, y_train):
        os.makedirs(save_dir, exist_ok=True)

        # Save numpy arrays
        np.save(os.path.join(save_dir, "X_train.npy"), X_train)
        np.save(os.path.join(save_dir, "y_train.npy"), y_train)

        # Save sequences
        with open(os.path.join(save_dir, "train_sequences.json"), "w") as f:
            json.dump(train, f)

        with open(os.path.join(save_dir, "valid_targets.json"), "w") as f:
            json.dump(valid, f)

        with open(os.path.join(save_dir, "test_targets.json"), "w") as f:
            json.dump(test, f)

        print(f"Data saved to {save_dir}")


if __name__ == "__main__":
    data_path = "ratings.dat"
    save_path = "./processed_data"

    processor = SASRecDataProcessor(data_path)

    # Preprocess
    sequences = processor.preprocess()

    # Split
    train, valid, test = processor.leave_one_out_split(sequences)

    # Training pairs
    X_train, y_train = processor.generate_training_instances(train, max_len=50)

    # Save everything
    processor.save_data(save_path, train, valid, test, X_train, y_train)