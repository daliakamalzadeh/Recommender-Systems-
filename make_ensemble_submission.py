from pathlib import Path
import pandas as pd
from eval_utils import load_all, clean_train, user_histories, parse_submission
from meta_scoring import MetaScorer
from ensemble_pipeline import build_hybrid_submission

DATA = Path(".")
train, test, item_meta, sample_sub, _, _ = load_all(DATA)
train_clean = clean_train(train)
histories = user_histories(train_clean)
user_seen = {u: set(v) for u, v in histories.items()}
scorer = MetaScorer.build(item_meta)

bpr = parse_submission(pd.read_csv(DATA / "candidates_bpr.csv"))
sas = parse_submission(pd.read_csv(DATA / "candidates_sasrec.csv"))

sub = build_hybrid_submission(
    sample_sub, bpr, sas, scorer, histories, user_seen,
    rrf_k=60, meta_weight=0.15, sasrec_weight=2.0, top_k=10,   # 0.15 = Full-Pipeline (offline best); set 0.0 for plain Ensemble-RRF
)
sub.to_csv(DATA / "submission_ensemble.csv", index=False)
print("wrote submission_ensemble.csv", sub.shape)