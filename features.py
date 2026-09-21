"""Rolling-window feature engineering for the stress detector.

Features are computed in a 28-day sliding window (7-day step) and are
expressed relative to the client's own baseline (expanding stats over
the client's previous windows), not relative to the population.
"""
from datetime import timedelta

import numpy as np
import pandas as pd

WINDOW_DAYS = 28
STEP_DAYS = 7
MIN_PRIOR_WINDOWS = 3          # baseline needs at least this many prior windows
GAP_CAP_DAYS = 25.0            # "no large debit found" -> healthy long gap
LARGE_DEBIT_SHARE = 0.10       # debit >= 10% of monthly income counts as "large"
DISCRETIONARY = ("entertainment", "subscription")

FEATURES = [
    "ca_share", "ca_share_z",
    "gap_ratio",
    "bal_std_norm", "bal_std_z",
    "disc_share", "disc_trend",
    "overdraft_cnt",
    "net_flow",
]


def _prepare(tx: pd.DataFrame, income: float):
    """Sort txs, build a daily balance series."""
    tx = tx.sort_values("date").reset_index(drop=True)
    daily = (
        tx.set_index("date")["balance_after"]
        .resample("D").last().ffill()
    )
    return tx, daily


def compute_window(tx, daily, income, w_start, w_end):
    """Raw window-level metrics without baseline."""
    win = tx[(tx["date"] > w_start) & (tx["date"] <= w_end)]
    n = len(win)

    ca_cnt = int((win["category"] == "cash_advance").sum())
    ca_share = ca_cnt / n if n else 0.0

    debits = win[win["amount"] < 0]
    spend = float(-debits["amount"].sum())
    disc = float(-debits.loc[debits["category"].isin(DISCRETIONARY), "amount"].sum())
    disc_share = disc / spend if spend > 0 else np.nan

    bal_slice = daily.reindex(pd.date_range(w_start + pd.Timedelta(days=1), w_end))
    bal_std_norm = float(bal_slice.std(ddof=1)) / income

    overdraft_cnt = int((win["balance_after"] < 0).sum())

    net_flow = float(win["amount"].sum()) / income if n else np.nan

    sal = win[win["category"] == "salary"]
    if len(sal) == 0:
        gap_days = np.nan
    else:
        s0 = sal["date"].iloc[0]
        after = tx[
            (tx["date"] > s0)
            & (tx["date"] <= s0 + pd.Timedelta(days=20))
            & (tx["amount"] <= -income * LARGE_DEBIT_SHARE)
        ]
        gap_days = float((after["date"].iloc[0] - s0).days) if len(after) else GAP_CAP_DAYS

    return {
        "window_start": w_start, "window_end": w_end,
        "ca_share": ca_share,
        "gap_days": gap_days,
        "bal_std_norm": bal_std_norm,
        "disc_share": disc_share,
        "overdraft_cnt": overdraft_cnt,
        "net_flow": net_flow,
    }


def add_baseline_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append own-baseline z-scores / ratios using expanding stats over prior windows."""
    g = df.groupby("client_id", sort=False)
    for col in ("ca_share", "gap_days", "bal_std_norm", "disc_share"):
        base_mean = g[col].transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        base_std = g[col].transform(lambda s: s.shift(1).expanding(min_periods=2).std())
        prior_n = g[col].transform(lambda s: s.shift(1).expanding(min_periods=1).count())
        enough = prior_n >= MIN_PRIOR_WINDOWS
        df[f"{col}_bmean"] = base_mean.where(enough)
        df[f"{col}_bz"] = ((df[col] - base_mean) / (base_std + 1e-6)).where(enough)

    df["ca_share_z"] = df["ca_share_bz"].fillna(0.0)
    df["bal_std_z"] = df["bal_std_norm_bz"].fillna(0.0)
    df["gap_ratio"] = (
        df["gap_days"] / df["gap_days_bmean"]
    ).clip(0, 3)
    df.loc[df["gap_days"].isna(), "gap_ratio"] = 0.0
    df["gap_ratio"] = df["gap_ratio"].fillna(1.0)
    df["disc_share"] = df["disc_share"].fillna(df["disc_share"].mean() if df["disc_share"].notna().any() else 0.2)
    df["disc_trend"] = (df["disc_share"] - df["disc_share_bmean"]).fillna(0.0)
    df["net_flow"] = df["net_flow"].fillna(0.0)
    return df


def build_windows(clients: pd.DataFrame, tx: pd.DataFrame,
                  step_days=STEP_DAYS) -> pd.DataFrame:
    """One row per (client, window_end) with raw metrics + own-baseline features."""
    out = []
    for cid, ctx in tx.groupby("client_id"):
        income = float(clients.loc[cid, "income_monthly"])
        ctx, daily = _prepare(ctx, income)
        start, end = ctx["date"].min(), ctx["date"].max()
        w_end = start + pd.Timedelta(days=WINDOW_DAYS)
        while w_end <= end:
            out.append({"client_id": cid,
                        **compute_window(ctx, daily, income,
                                         w_end - pd.Timedelta(days=WINDOW_DAYS), w_end)})
            w_end += pd.Timedelta(days=step_days)

    df = pd.DataFrame(out)
    if df.empty:
        return df
    return add_baseline_features(df)


def client_snapshot(tx_client: pd.DataFrame, income: float) -> dict:
    """Feature vector + human-readable facts for the client's latest state."""
    ctx, daily = _prepare(tx_client.copy(), income)
    end = ctx["date"].max()

    windows = []
    w_end = end
    while w_end >= ctx["date"].min() + pd.Timedelta(days=WINDOW_DAYS):
        w_start = w_end - pd.Timedelta(days=WINDOW_DAYS)
        windows.append({"client_id": 0,
                        **compute_window(ctx, daily, income, w_start, w_end)})
        w_end -= pd.Timedelta(days=STEP_DAYS)
    windows.reverse()
    if not windows:
        row = compute_window(ctx, daily, income,
                             end - pd.Timedelta(days=WINDOW_DAYS), end)
        hist = pd.DataFrame()
    else:
        row = windows[-1]
        hist = pd.DataFrame(windows[:-1])

    df = pd.DataFrame(windows)
    if not df.empty:
        df = add_baseline_features(df)
        row = df.iloc[-1].to_dict()

    facts = {}
    feats = {
        "ca_share": row.get("ca_share", 0.0),
        "ca_share_z": row.get("ca_share_z", 0.0),
        "gap_ratio": float(np.clip(row.get("gap_ratio", 1.0), 0, 3)),
        "bal_std_norm": row.get("bal_std_norm", 0.0),
        "bal_std_z": row.get("bal_std_z", 0.0),
        "disc_share": row.get("disc_share", 0.2),
        "disc_trend": row.get("disc_trend", 0.0),
        "overdraft_cnt": row.get("overdraft_cnt", 0),
        "net_flow": row.get("net_flow", 0.0),
    }

    if not hist.empty:
        bgap = hist["gap_days"].dropna()
        facts["gap_base"] = float(bgap.mean()) if len(bgap) else GAP_CAP_DAYS

        b_bal = hist["bal_std_norm"].mean()
        facts["bal_base"] = float(b_bal)

        b_disc = hist["disc_share"].dropna()
        facts["disc_base"] = float(b_disc.mean()) if len(b_disc) else None
    else:
        facts["gap_base"] = GAP_CAP_DAYS
        facts["bal_base"] = None
        facts["disc_base"] = None

    facts.update({
        "ca_share": row.get("ca_share", 0.0),
        "gap_days": row.get("gap_days", np.nan),
        "disc_share": row.get("disc_share", np.nan),
        "overdraft_cnt": row.get("overdraft_cnt", 0),
        "net_flow": row.get("net_flow", np.nan),
        "bal_std_norm": row.get("bal_std_norm", np.nan),
        "as_of": str(end.date()),
    })
    return {"features": feats, "facts": facts}


def add_labels(windows: pd.DataFrame, clients: pd.DataFrame,
               horizon_days=WINDOW_DAYS) -> pd.DataFrame:
    """label=1 if the client's default_date falls within (window_end, +horizon]."""
    dd = clients["default_date"]
    w = windows.copy()
    w["default_date"] = w["client_id"].map(dd)
    w["label"] = (
        w["default_date"].notna()
        & (w["window_end"] < w["default_date"])
        & (w["window_end"] + pd.Timedelta(days=horizon_days) >= w["default_date"])
    ).astype(int)
    # drop windows at/after the default moment (ground-truth leakage guard)
    w = w[w["default_date"].isna() | (w["window_end"] < w["default_date"])]
    return w
