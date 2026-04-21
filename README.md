# SASRec — Sequential Recommendation on MovieLens 1M

Implementation of SASRec (Kang & McAuley, 2018) using self-attentive transformer blocks for sequential movie recommendations.

## Setup

```bash
pip install -r requirements.txt
```

Download the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) and place `ratings.dat` in a `ml-1m/` directory.

## Usage

**1. Preprocess the data**
```bash
python preprocess.py --data_path ml-1m/ratings.dat --save_dir processed_data
```

Optional flags:
- `--min_interactions` — minimum interactions per user to keep (default: 5)
- `--max_len` — maximum sequence length for training (default: 50)

Output: `processed_data/` with `train_sequences.json`, `valid_targets.json`, `test_targets.json`, `X_train.npy`, `y_train.npy`, `train_user_ids.npy`, and `metadata.json`.

**2. Train the model**
```bash
python train.py --data_dir processed_data --ckpt_dir checkpoints
```

Key flags:
- `--hidden_size` — embedding/hidden dimension (default: 64)
- `--num_blocks` — number of self-attention blocks (default: 2)
- `--num_heads` — number of attention heads (default: 2)
- `--max_len` — max sequence length (default: 50)
- `--dropout` — dropout rate (default: 0.2)
- `--epochs` — maximum training epochs (default: 200)
- `--lr` — learning rate (default: 1e-3)
- `--num_neg` — negative samples per positive (default: 1)
- `--patience` — early stopping patience in validation checks (default: 20)
- `--eval_every` — validate every N epochs (default: 5)
- `--sampled_eval` — use sampled ranking (99 negatives) instead of full ranking
- `--use_scheduler` — enable cosine annealing learning rate schedule

Saves best checkpoint (by validation NDCG@10) to `checkpoints/best_model.pt`.

**3. Run ablation experiments**
```bash
python evaluate.py --data_dir processed_data
```

Trains and evaluates 9 configurations varying number of blocks, hidden size, number of heads, max sequence length, and dropout. Results are printed sorted by test NDCG@10.

Optional flags: same as `train.py` (controls shared training hyperparameters such as `--lr`, `--epochs`, `--batch_size`, `--patience`).

## Project Structure

```
├── model.py          # SASRec model and attention block
├── train.py          # Training loop, dataset, evaluation
├── evaluate.py       # Ablation experiment runner
├── preprocess.py     # Data preprocessing and splitting
└── processed_data/   # Generated after preprocessing
    ├── train_sequences.json
    ├── valid_targets.json
    ├── test_targets.json
    ├── X_train.npy
    ├── y_train.npy
    ├── train_user_ids.npy
    └── metadata.json
```

## Evaluation

Uses leave-one-out splits: the last item is held out for test, the second-to-last for validation. Metrics reported: **Recall@10**, **Recall@20**, **NDCG@10**, **NDCG@20**. Supports both full ranking (all items) and sampled ranking (99 random negatives + target).
