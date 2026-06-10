# Recommender System — Group 30

Top-10 item recommendation for the course Kaggle competition, scored by
**Recall@10**. The final submission is a **SASRec** sequential model with a
popularity blend; a BPR baseline and an RRF + content-metadata ensemble are
included as documented comparison points.

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
2. `sasrec.py --final` → `submission_sasrec.csv` (**official Kaggle submission**)
3. `generate_candidates.py` → candidate pools for the comparison/ensemble
4. `model_comparison_eval.py` → `model_comparison_scores.txt`

Add `--sweep` to also run the SASRec hyperparameter sweep (analysis only, slow),
or `--skip-baseline` to produce just the SASRec submission.

---

## Files

| File | Role |
|---|---|
| `sasrec.py` | SASRec model — training, temporal validation, popularity blend, inference, submission. The main model (Task 2). |
| `baseline_pipeline.py` | Data cleaning, temporal split, popularity/BM25/SVD/NMF/BPR baselines (Task 1). |
| `sweep.py` | SASRec hyperparameter grid search; logs blended Recall@10 to `sweep_results.csv` (evidence for Hyperparameter Analysis). |
| `generate_candidates.py` | Writes top-30 BPR and SASRec candidate pools (submission + held-out) for the ensemble (Task 3). |
| `ensemble_pipeline.py` | Reciprocal Rank Fusion of model rankings + content-metadata reranking. |
| `meta_scoring.py` | Content scorer built from `item_meta.csv` (quality, popularity, BSR, category/store affinity). |
| `model_comparison_eval.py` | Scores all models on one held-out temporal split → `model_comparison_scores.txt`. |
| `eval_utils.py` | Shared Recall@10 / NDCG@10 scoring and prediction assembly. |
| `failure_cases_report.py`, `final_pipeline.py` | Error analysis / hybrid submission assembly. |
| `run_all.py` | One-command end-to-end runner. |

---

## Validation

All scripts use a single **global temporal split**: one timestamp cutoff at the
`--val-quantile` (default 0.90) of all interactions. Everything before the
cutoff is training history; everything at/after is the validation target. This
mirrors the Kaggle task (predict each user's *future* interactions) and avoids
the cross-user leakage of a per-user leave-last-N split — which previously made
local scores disagree with the leaderboard. Model selection uses local Recall@10
only; the public leaderboard is never probed for tuning.

`sasrec.py --final` disables the holdout and trains on all of `train.csv` for the
actual submission.

---

## Reproducing the official submission

```bash
python sasrec.py --data-dir . --final --pop-blend 0.20 --epochs 12 \
    --output submission_sasrec.csv --amp
```

The settings (`--epochs 12`, `--pop-blend 0.20`) were selected on the temporal
validation split (see `sweep.py` and the report's Hyperparameter Analysis). The popularity blend improved local validation performance and provides a robust global prior alongside the SASRec scores. Fixed seeds, deterministic cuDNN, a deterministic split, and a seeded dataloader make the run reproducible: same data + same seed → same submission.

---

## Results summary

The held-out comparison (`model_comparison_scores.txt`) shows SASRec + blend as
the strongest single model. A BPR + SASRec RRF ensemble and content-metadata
reranking were tested but did **not** beat SASRec alone, so the official
submission is SASRec with a popularity blend; the ensemble is reported as a
negative result.

---

## Outputs produced

`submission_sasrec.csv` (Kaggle submission), `submission_bpr.csv`,
`candidates_*.csv`, `val_candidates_*.csv`, `model_comparison_scores.txt`,
`sweep_results.csv` (if `--sweep`). All are regenerated from `train.csv`; none
need to be committed.
