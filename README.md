# Neural Collaborative Filtering — MovieLens 1M

Implementation of NCF (He et al., 2017) combining GMF and MLP branches for movie recommendations.

## Setup

```bash
pip install -r requirements.txt
```

Download the [MovieLens 1M dataset](https://grouplens.org/datasets/movielens/1m/) and place `ratings.dat` in a `ml-1m/` directory.

## Usage

**1. Preprocess the data**
```bash
python preprocess_movielens_ncf.py --ratings_path ml-1m/ratings.dat --output_dir ncf_preprocessed
```

Optional flags:
- `--positive_threshold` — minimum rating to count as a positive (default: 4)
- `--negatives_per_positive` — negative samples per positive interaction (default: 1)
- `--random_seed` (default: 42)

Output: `ncf_preprocessed/` with `train.csv`, `val.csv`, `test.csv`, and `metadata.json`.

**2. Run experiments**
```bash
python experiments.py
```

Trains 7 configurations varying embedding size, MLP depth, and dropout. Uses early stopping (patience=3) and evaluates with **Recall@10** and **NDCG@10** on a full-ranking protocol.

## Project Structure

```
├── model.py                      # NCF model definition
├── experiments.py                # Training loop and evaluation
├── preprocess_movielens_ncf.py   # Data preprocessing
└── ncf_preprocessed/             # Generated after preprocessing
    ├── train.csv
    ├── val.csv
    ├── test.csv
    └── metadata.json
```
