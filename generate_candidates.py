from __future__ import annotations
from validation import temporal_holdout
 
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
BLEND_ALPHA = 0.20
BPR_EPOCHS    = 30
BPR_FACTORS   = 64
SAS = dict(max_len=50, hidden=64, blocks=2, heads=2, dropout=0.2,
           lr=1e-3, epochs=12, patience=30, eval_every=5)
 
 
def train_and_save_bpr(train_df, user_ids, bm25_items, out_path: Path, label: str) -> None:
    print(f"[generate_candidates] Training BPR ({label}, {BPR_EPOCHS} epochs, "
          f"{BPR_FACTORS} factors) …")
    bpr_enc, bpr_uf, bpr_if = train_bpr(train_df, n_factors=BPR_FACTORS, epochs=BPR_EPOCHS)
    rows = [
        {"user_id": uid, "item_id": ",".join(map(str, recommend_factor_model(uid, bpr_enc, bpr_uf, bpr_if, bm25_items, k=CANDIDATE_K)))}
        for uid in user_ids
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[generate_candidates] Saved {out_path}")
 
 
def train_and_save_sasrec(train_df, user_ids, out_path: Path, label: str,
                          final: bool) -> None:

    print(f"[generate_candidates] Training SASRec ({label}) …")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    data      = build_dataset(sas_clean(train_df), final=final)
    ds        = SeqDataset(list(data.train_seq.values()), max_len=SAS["max_len"])
    loader    = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    model     = SASRec(data.n_items, max_len=SAS["max_len"], hidden=SAS["hidden"],
                       n_blocks=SAS["blocks"], n_heads=SAS["heads"],
                       dropout=SAS["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=SAS["lr"], betas=(0.9, 0.98))
    fit(model, loader, data, optimizer, device,
        epochs=SAS["epochs"], patience=SAS["patience"], eval_every=SAS["eval_every"],
        neg_mode="full", num_neg=1, max_len=SAS["max_len"], verbose=True)
    preds = sasrec_recommend(model, data, user_ids, device, max_len=SAS["max_len"],
                             pop_blend=BLEND_ALPHA, top_k=CANDIDATE_K)
    rows  = [{"user_id": uid, "item_id": ",".join(map(str, preds.get(uid, [])))}
             for uid in user_ids]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[generate_candidates] Saved {out_path} (blend alpha={BLEND_ALPHA})")
 
 
def main() -> None:
    train, _, _, sample_sub = load_data(DATA_DIR)
    train_clean  = clean_train(train)
    train_split, val = temporal_holdout(train_clean, quantile=0.95)
 
    submission_uids = list(sample_sub["user_id"].unique())
    val_uids        = list(val["user_id"].unique())
 
    # Full train 
    bm25_full = make_bm25_popularity(train_clean)
    train_and_save_bpr(train_clean, submission_uids, bm25_full, DATA_DIR / "candidates_bpr.csv", "full train")
    train_and_save_sasrec(train_clean, submission_uids, DATA_DIR / "candidates_sasrec.csv", "full train", final=True)
 
    # train_split 
    bm25_split = make_bm25_popularity(train_split)
    train_and_save_bpr(train_split, val_uids, bm25_split, DATA_DIR / "val_candidates_bpr.csv", "train_split")
    train_and_save_sasrec(train_split, val_uids, DATA_DIR / "val_candidates_sasrec.csv", "train_split", final=False)
 
 
if __name__ == "__main__":
    main()