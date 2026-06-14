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
            "Step 1/5: BPR baseline -> submission_bpr.csv")

    # 2. SASRec-only submission (comparison point): train on ALL data at the chosen settings.
    run([py, "sasrec.py", "--data-dir", dd, "--final",
         "--pop-blend", str(POP_BLEND), "--epochs", str(SAS_EPOCHS),
         "--output", "submission_sasrec.csv", *amp],
        f"Step 2/5: SASRec final (blend={POP_BLEND}, epochs={SAS_EPOCHS}) "
        f"-> submission_sasrec.csv  [comparison point]")

    # 3. Candidate pools for the held-out comparison + the ensemble submission.
    run([py, "generate_candidates.py"],
        "Step 3/5: candidate pools -> candidates_*.csv / val_candidates_*.csv")

    # 4. Held-out model comparison table.
    run([py, "model_comparison_eval.py"],
        "Step 4/5: model comparison -> model_comparison_scores.txt")

    # 5. OFFICIAL submission: RRF fusion of the candidate pools + metadata rerank.
    run([py, "make_ensemble_submission.py"],
        "Step 5/5: RRF + metadata ensemble -> submission_ensemble.csv  "
        "[OFFICIAL SUBMISSION]")

    # Optional: hyperparameter sweep (evidence table for the report; slow).
    if args.sweep:
        run([py, "sweep.py", "--data-dir", dd, "--epochs", "100",
             "--patience", "20", *amp],
            "Optional: SASRec hyperparameter sweep -> sweep_results.csv")

    print("\n" + "=" * 72)
    print(f"[run_all] PIPELINE COMPLETE in {time.time() - total0:.0f}s")
    print("  Kaggle submission : submission_ensemble.csv  (Recall@10 ~ 0.0161)")
    print("  SASRec-only       : submission_sasrec.csv     (Recall@10 ~ 0.0146)")
    print("  Comparison table  : model_comparison_scores.txt")
    print("=" * 72)


if __name__ == "__main__":
    main()
