"""
Generate extended candidate submissions (30 items per user) for BPR and SASRec.
These are used by hybrid_pipeline.py for a richer RRF pool before reranking to 10.
"""
from __future__ import annotations
 
import contextlib
import os
from pathlib import Path
 
import pandas as pd
import torch
 
from baseline_pipeline import (
    load_data, clean_train, chronological_split,
    make_user_seen_items, make_bm25_popularity,
    train_bpr, recommend_factor_model,
)
from sasrec import (
    SASRec, SeqDataset, build_dataset,
    clean_train as sas_clean, fit,
    recommend as sasrec_recommend, set_seed,
)
 
CANDIDATE_K = 30
DATA_DIR    = Path(".")
 
 
def train_and_save_bpr(train_df, user_ids, bm25_items, out_path: Path, label: str) -> None:
    print(f"[generate_candidates] Training BPR ({label}) …")
    with open(os.devnull, "w") as _null, contextlib.redirect_stdout(_null):
        bpr_enc, bpr_uf, bpr_if = train_bpr(train_df)
    rows = [
        {"user_id": uid, "item_id": ",".join(map(str, recommend_factor_model(uid, bpr_enc, bpr_uf, bpr_if, bm25_items, k=CANDIDATE_K)))}
        for uid in user_ids
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[generate_candidates] Saved {out_path}")
 
 
def train_and_save_sasrec(train_df, user_ids, out_path: Path, label: str) -> None:
    print(f"[generate_candidates] Training SASRec ({label}) …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    data      = build_dataset(sas_clean(train_df))
    ds        = SeqDataset(list(data.train_seq.values()), max_len=50)
    loader    = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    model     = SASRec(data.n_items, max_len=50, hidden=64, n_blocks=2, n_heads=2, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98))
    fit(model, loader, data, optimizer, device,
        epochs=300, patience=30, eval_every=5,
        neg_mode="full", num_neg=1, max_len=50, verbose=True)
    preds = sasrec_recommend(model, data, user_ids, device, max_len=50, top_k=CANDIDATE_K)
    rows  = [{"user_id": uid, "item_id": ",".join(map(str, preds.get(uid, [])))} for uid in user_ids]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[generate_candidates] Saved {out_path}")
 
 
def main() -> None:
    train, _, _, sample_sub = load_data(DATA_DIR)
    train_clean  = clean_train(train)
    train_split, val = chronological_split(train_clean)
 
    submission_uids = list(sample_sub["user_id"].unique())
    val_uids        = list(val["user_id"].unique())
 
    # Full train — for submission
    bm25_full = make_bm25_popularity(train_clean)
    train_and_save_bpr(train_clean, submission_uids, bm25_full, DATA_DIR / "candidates_bpr.csv", "full train")
    train_and_save_sasrec(train_clean, submission_uids, DATA_DIR / "candidates_sasrec.csv", "full train")
 
    # train_split — for val / failure analysis
    bm25_split = make_bm25_popularity(train_split)
    train_and_save_bpr(train_split, val_uids, bm25_split, DATA_DIR / "val_candidates_bpr.csv", "train_split")
    train_and_save_sasrec(train_split, val_uids, DATA_DIR / "val_candidates_sasrec.csv", "train_split")
 
 
if __name__ == "__main__":
    main()