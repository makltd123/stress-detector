"""Train the stress-detection model and save the artifact.

Reads data/stress_detector.db, builds rolling-window features, trains a
RandomForest, prints ROC-AUC + lead-time metrics, saves artifacts/model.joblib.

Usage:
    python train_model.py [--db data/stress_detector.db] --seed 42
"""
import argparse
import os
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve

import features as F


def load_db(db_path):
    con = sqlite3.connect(db_path)
    try:
        clients = pd.read_sql(
            "SELECT client_id, income_monthly, credit_limit, label, default_date FROM clients",
            con).set_index("client_id")
        tx = pd.read_sql(
            "SELECT tx_id, client_id, date, amount, category, balance_after FROM transactions",
            con)
    finally:
        con.close()
    clients["default_date"] = pd.to_datetime(clients["default_date"])
    tx["date"] = pd.to_datetime(tx["date"])
    return clients, tx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/stress_detector.db")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clients, tx = load_db(args.db)
    windows = F.add_labels(F.build_windows(clients, tx), clients)

    train_mask = windows["window_end"] < windows["window_end"].quantile(0.75)
    tr, te = windows[train_mask], windows[~train_mask]
    X_tr, y_tr = tr[F.FEATURES], tr["label"]
    X_te, y_te = te[F.FEATURES], te["label"]
    print(f"[train] windows: total={len(windows)}  train={len(tr)} (pos={y_tr.sum()})  "
          f"test={len(te)} (pos={y_te.sum()})")

    rf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=10, max_depth=8,
        class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1,
    )
    rf.fit(X_tr, y_tr)

    p_tr, p_te = rf.predict_proba(X_tr)[:, 1], rf.predict_proba(X_te)[:, 1]
    prec, rec, thr_grid = precision_recall_curve(y_tr, p_tr)
    f1 = np.where(prec[1:] + rec[1:] > 0, 2 * prec[1:] * rec[1:] / (prec[1:] + rec[1:] + 1e-9), 0)
    threshold = float(thr_grid[int(np.argmax(f1))])

    auc_tr = roc_auc_score(y_tr, p_tr)
    auc_te = roc_auc_score(y_te, p_te)
    print(f"[train] ROC-AUC: train={auc_tr:.3f}  test={auc_te:.3f}  threshold(best-F1 on train)={threshold:.2f}")
    if auc_tr - auc_te > 0.05:
        print(f"[train] WARNING: possible overfitting (train-test gap={auc_tr - auc_te:.3f})")
    print(f"[train] test @thr: recall={(y_te & (p_te >= threshold)).sum()}/{y_te.sum()}")

    # ---- early-warning lead time over all defaulter windows -----------------
    leads = []
    for cid in clients.index[clients["label"] == 1]:
        cw = windows[(windows["client_id"] == cid)].sort_values("window_end")
        if cw.empty or pd.isna(cw["default_date"].iloc[0]):
            continue
        dd = cw["default_date"].iloc[0]
        probs = rf.predict_proba(cw[F.FEATURES])[:, 1]
        alerted = cw.loc[probs >= threshold, "window_end"]
        if len(alerted):
            leads.append((dd - alerted.iloc[0]).days)
    if leads:
        s = pd.Series(leads)
        print(f"[train] first-alert lead time before default: "
              f"median={s.median():.0f}d  min={s.min()}d  max={s.max()}d  "
              f"alerted {len(leads)}/{int((clients['label'] == 1).sum())} defaulters")

    imp = sorted(zip(F.FEATURES, rf.feature_importances_), key=lambda kv: -kv[1])
    print("[train] feature importances:")
    for name, v in imp:
        print(f"    {name:<14} {v:.3f}")

    os.makedirs("artifacts", exist_ok=True)
    means = X_tr.mean().to_dict()
    stds = X_tr.std().to_dict()
    meta = {
        "features": F.FEATURES,
        "threshold": threshold,
        "metrics": {"roc_auc_train": round(auc_tr, 4), "roc_auc_test": round(auc_te, 4),
                    "median_lead_days": int(s.median()) if leads else None},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "feature_means": means,
        "feature_stds": stds,
    }
    joblib.dump({"model": rf, **meta}, "artifacts/model.joblib")
    print("[train] saved -> artifacts/model.joblib")


if __name__ == "__main__":
    main()
