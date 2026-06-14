# Recommender System — Group 30

Top-10 item recommendation for the course Kaggle competition, scored by
**Recall@10**. The final submission is an **RRF ensemble** that fuses SASRec
(with a popularity blend) and BPR candidates and reranks them with an
item-metadata content score. SASRec alone and BPR alone are included as
documented comparison points.

**Data policy:** all models train **only** on `train.csv`. No external data, no
pretrained embeddings, no use of test labels. `test.csv` is not used during
training or validation.

---

## Setup

```bash
pip install -r requirements.txt
```

Place `train.csv`, `test.csv`, `item_meta.csv`, and `sample_submission.csv` in
the working directory (or pass `--data-dir`). A CUDA GPU is used automatically
for the SASRec steps when available; add `--amp` for mixed precision.

---

## Quick start — one command

```bash
python run_all.py --data-dir . --amp
```

This runs the full pipeline in order and aborts on any failure:

1. `baseline_pipeline.py` → `submission_bpr.csv` (BPR baseline)
2. `sasrec.py --final` → `submission_sasrec.csv` (SASRec-only comparison point)
3. `generate_candidates.py` → top-30 BPR and SASRec candidate pools
4. `model_comparison_eval.py` → `model_comparison_scores.txt`
5. `make_ensemble_submission.py` → `submission_ensemble.csv`
   (**official Kaggle submission**: RRF fusion + metadata rerank)

Add `--sweep` to also run the SASRec hyperparameter sweep (analysis only, slow),
or `--skip-baseline` to skip the standalone BPR submission.

---

## Files

| File | Role |
|---|---|
| `sasrec.py` | SASRec model — training, temporal validation, popularity blend, inference, submission. The main model (Task 2). |
| `baseline_pipeline.py` | Data cleaning, temporal split, popularity/BM25/SVD/NMF/BPR baselines (Task 1). |
| `sweep.py` | SASRec hyperparameter grid search; logs blended Recall@10 to `sweep_results.csv` (evidence for Hyperparameter Analysis). |
| `generate_candidates.py` | Writes top-30 BPR and SASRec candidate pools (submission + held-out) for the ensemble (Task 3). |
| `ensemble_pipeline.py` | Reciprocal Rank Fusion of model rankings + content-metadata reranking. |
| `make_ensemble_submission.py` | Builds the **official** `submission_ensemble.csv` from the full-train candidate pools (RRF + metadata rerank, `meta_weight=0.15`). |
| `meta_scoring.py` | Content scorer built from `item_meta.csv` (quality, popularity, BSR, category/store affinity). |
| `model_comparison_eval.py` | Scores all models on one held-out temporal split → `model_comparison_scores.txt`. |
| `eval_utils.py` | Shared Recall@10 / NDCG@10 scoring and prediction assembly. |
| `run_all.py` | One-command end-to-end runner. |
| `validation.py` | Standalone offline evaluator; provides `temporal_holdout` (the 0.90 split) used by `generate_candidates.py`. |

---

## Validation

All scripts use a single **global temporal split**: one timestamp cutoff at the
`--val-quantile` (default 0.90) of all interactions. Everything before the
cutoff is training history; everything at/after is the validation target. This
mirrors the Kaggle task (predict each user's *future* interactions) and avoids
the cross-user leakage of a per-user leave-last-N split — which previously made
local scores disagree with the leaderboard. Model selection uses local Recall@10
only; the public leaderboard is never probed for tuning.

`sasrec.py --final` shifts the cutoff to the 98th percentile of timestamps
(`--final-val-quantile 0.98`), training on nearly all of `train.csv` while
keeping a shallow holdout so early stopping still works for the submission run.

---

## Reproducing the official submission

The official submission is the RRF + metadata ensemble. It is built in two steps
from the full-train candidate pools:

```bash
# 1. Train BPR + SASRec and write the top-30 candidate pools
python generate_candidates.py

# 2. Fuse (RRF) + metadata-rerank into the final top-10 submission
python make_ensemble_submission.py        # writes submission_ensemble.csv
```

`generate_candidates.py` trains the SASRec candidates with `--epochs 12` and a
popularity blend `alpha = 0.20`, and BPR with 30 epochs / 64 factors; these were
selected on the temporal validation split (see `sweep.py` and the report's
Hyperparameter Analysis). `make_ensemble_submission.py` fuses the two pools with
RRF (`w_SASRec = 2.0`, `w_BPR = 1.0`, `k = 60`) and applies the metadata rerank
at `meta_weight = 0.15`. Fixed seeds, deterministic cuDNN, a deterministic
split, and a seeded dataloader make the run reproducible: same data + same seed →
same submission.

The SASRec-only submission (`submission_sasrec.csv`, a comparison point) is
reproduced with:

```bash
python sasrec.py --data-dir . --final --pop-blend 0.20 --epochs 12 \
    --output submission_sasrec.csv --amp
```

---

## Results summary

On the held-out temporal comparison (`model_comparison_scores.txt`), SASRec with
a popularity blend is the strongest *single* model, ahead of BPR and a pure
popularity baseline. Fusing SASRec and BPR with RRF and then reranking with the
metadata content score beats every single model, both offline and on the public
leaderboard: the RRF + metadata ensemble scored **0.01611** on Kaggle versus
**0.01456** for SASRec alone. The ensemble is therefore the official submission.

---

## Outputs produced

`submission_ensemble.csv` (**official Kaggle submission**), `submission_sasrec.csv`
and `submission_bpr.csv` (comparison points), `candidates_*.csv`,
`val_candidates_*.csv`, `model_comparison_scores.txt`, `sweep_results.csv` (if
`--sweep`). All are regenerated from `train.csv`; none need to be committed.
