"""
run_all.py — Group 30 end-to-end pipeline (raw data -> submission + analysis).

Runs the full reproducible pipeline in order with a single command:

    python run_all.py --data-dir . --amp

Steps (each is a separate, independently-runnable script):
  1. baseline_pipeline.py   -> submission_bpr.csv          (Task 1: BPR baseline)
  2. sasrec.py --final      -> submission_sasrec.csv        (Task 2: SASRec, OFFICIAL submission)
  3. generate_candidates.py -> candidates_*.csv,
                               val_candidates_*.csv         (Task 3: candidate pools)
  4. model_comparison_eval.py -> model_comparison_scores.txt (held-out model comparison)

Final hyperparameters (selected on the GLOBAL TEMPORAL validation split; see the
report's Hyperparameter Analysis): SASRec 12 epochs, popularity blend alpha=0.20.

Requirements: see requirements.txt (torch, pandas, numpy, scipy, scikit-learn).
Trains ONLY on train.csv. No external data, no pretrained embeddings, no test labels.

Flags:
  --data-dir   folder with train/test/item_meta/sample_submission CSVs (default: .)
  --amp        enable CUDA mixed precision for the SASRec steps (GPU only)
  --skip-baseline   skip step 1 (BPR submission) if you only need the SASRec submission
  --sweep      additionally run the SASRec hyperparameter sweep (analysis only; slow)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# SASRec final-submission settings, chosen on the temporal split.
SAS_EPOCHS  = 12
POP_BLEND   = 0.20


def run(cmd: list[str], title: str) -> None:
    """Run one pipeline step, streaming its output; abort the run on failure."""
    print("\n" + "=" * 72)
    print(f">>> {title}")
    print("    " + " ".join(cmd))
    print("=" * 72, flush=True)
    t0 = time.time()
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\n[run_all] STEP FAILED ({title}); aborting. "
                 f"See the traceback above.")
    print(f"[run_all] done in {time.time() - t0:.0f}s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Group 30 end-to-end pipeline")
    p.add_argument("--data-dir", default=".", help="folder with the CSVs")
    p.add_argument("--amp", action="store_true", help="CUDA mixed precision (SASRec)")
    p.add_argument("--skip-baseline", action="store_true",
                   help="skip the BPR baseline submission (step 1)")
    p.add_argument("--sweep", action="store_true",
                   help="also run the SASRec hyperparameter sweep (analysis only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable                      # current interpreter (venv-safe)
    dd = args.data_dir
    amp = ["--amp"] if args.amp else []

    if not (Path(dd) / "train.csv").exists():
        sys.exit(f"[run_all] train.csv not found in {dd!r}. Pass --data-dir.")

    total0 = time.time()

    # 1. BPR baseline submission (Task 1).
    if not args.skip_baseline:
        run([py, "baseline_pipeline.py", "--data-dir", dd,
             "--final-model", "bpr", "--output", "submission_bpr.csv"],
            "Step 1/4: BPR baseline -> submission_bpr.csv")

    # 2. SASRec OFFICIAL submission: train on ALL data at the chosen settings.
    run([py, "sasrec.py", "--data-dir", dd, "--final",
         "--pop-blend", str(POP_BLEND), "--epochs", str(SAS_EPOCHS),
         "--output", "submission_sasrec.csv", *amp],
        f"Step 2/4: SASRec final (blend={POP_BLEND}, epochs={SAS_EPOCHS}) "
        f"-> submission_sasrec.csv  [OFFICIAL SUBMISSION]")

    # 3. Candidate pools for the held-out comparison + hybrid experiments.
    run([py, "generate_candidates.py"],
        "Step 3/4: candidate pools -> candidates_*.csv / val_candidates_*.csv")

    # 4. Held-out model comparison table.
    run([py, "model_comparison_eval.py"],
        "Step 4/4: model comparison -> model_comparison_scores.txt")

    # Optional: hyperparameter sweep (evidence table for the report; slow).
    if args.sweep:
        run([py, "sweep.py", "--data-dir", dd, "--epochs", "100",
             "--patience", "20", *amp],
            "Optional: SASRec hyperparameter sweep -> sweep_results.csv")

    print("\n" + "=" * 72)
    print(f"[run_all] PIPELINE COMPLETE in {time.time() - total0:.0f}s")
    print("  Kaggle submission : submission_sasrec.csv  (Recall@10 ~ 0.0142)")
    print("  Comparison table  : model_comparison_scores.txt")
    print("=" * 72)


if __name__ == "__main__":
    main()
