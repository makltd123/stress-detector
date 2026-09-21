"""Synthetic transaction data generator for Early Financial Stress Detector.

Creates N clients (2 archetypes: stable / decline->default) with 6-12 months
of transactions and writes them straight into an SQLite database.

Usage:
    python generate_data.py --clients 80 --seed 42 --db data/stress_detector.db --plot
"""
import argparse
import os
import sqlite3
from datetime import date, timedelta

import numpy as np

CATEGORIES = (
    "groceries", "utilities", "entertainment", "cash_advance",
    "rent", "salary", "subscription",
)
LARGE_DEBIT_SHARE = 0.10  # share of income considered a "large debit" (gap feature)


def _add_month(y: int, m: int, k: int):
    t = (y * 12 + m - 1) + k
    return t // 12, t % 12 + 1


def _month_last_day(y: int, m: int) -> int:
    ny, nm = _add_month(y, m, 1)
    return (date(ny, nm, 1) - timedelta(days=1)).day


def simulate_client(client_id, rng, archetype, start, months):
    """Return (client_row, tx_rows). Events are planned per calendar month."""
    income = float(round(rng.uniform(50_000, 300_000), -3))
    credit_limit = float(round(income * rng.uniform(0.4, 1.4), -3))
    payday = int(rng.integers(1, 29))

    horizon_end = start + timedelta(days=int(months * 30.4))
    if archetype == "decline":
        lead = int(rng.integers(3, 15))            # data ends N days before default
        default_date = horizon_end - timedelta(days=lead)
        drift_len = int(rng.integers(28, 43))      # 4-6 weeks of degradation
        drift_start = default_date - timedelta(days=drift_len)
    else:
        default_date = None
        drift_start = None

    balance = income * rng.uniform(0.9, 1.7)
    plan = []  # (date, category, amount | None); None -> sized vs live balance

    n_subs = int(rng.integers(1, 3))
    sub_days = [int(rng.integers(1, 29)) for _ in range(n_subs)]
    sub_amounts = [float(rng.uniform(300, 900)) for _ in range(n_subs)]

    def progress_at(dt):
        if drift_start is None:
            return None
        span = max((default_date - drift_start).days, 1)
        return float(np.clip(((dt - drift_start).days) / span, 0.0, 1.0))

    tx_rows = []
    y0, m0 = start.year, start.month
    for mi in range(months):
        yy, mm = _add_month(y0, m0, mi)
        last = _month_last_day(yy, mm)
        if date(yy, mm, 1) > horizon_end:
            break
        month_prog = progress_at(date(yy, mm, min(payday, last)))

        def rnd_day(offset_lo, offset_hi):
            return min(payday + int(rng.integers(offset_lo, offset_hi + 1)), last)

        # salary
        plan.append((date(yy, mm, min(payday, last)), "salary",
                     income * rng.uniform(0.98, 1.02)))

        # rent: mid-month normally, pulled right after payday during drift
        rent_day = rnd_day(1, 3) if month_prog else rnd_day(8, 15)
        plan.append((date(yy, mm, rent_day), "rent",
                     -income * rng.uniform(0.25, 0.33)))

        # utilities
        plan.append((date(yy, mm, rnd_day(4, 10)), "utilities",
                     -income * rng.uniform(0.06, 0.11)))

        # groceries (~weekly)
        for gday in rng.choice(np.arange(1, last + 1), size=int(rng.integers(7, 10)), replace=False):
            plan.append((date(yy, mm, int(gday)), "groceries",
                         -income * rng.uniform(0.020, 0.050)))

        # entertainment: cut first during stress
        lam_ent = 3.0 if month_prog is None else 3.0 * (1.0 - 0.87 * month_prog)
        for _ in range(int(rng.poisson(lam_ent))):
            plan.append((date(yy, mm, int(rng.integers(1, last + 1))), "entertainment",
                         -income * rng.uniform(0.015, 0.050)))

        # subscriptions (second one cancelled mid-drift)
        for si in range(n_subs):
            if month_prog is not None and si == 1 and month_prog > 0.5:
                continue
            plan.append((date(yy, mm, min(sub_days[si], last)), "subscription",
                         -sub_amounts[si]))

        # cash advances: rare for stable, ramping hard during drift;
        # irregular debits add volatility and eventually overdrafts
        if month_prog is None:
            if rng.random() < 0.12:
                plan.append((date(yy, mm, int(rng.integers(1, last + 1))), "cash_advance",
                             -income * rng.uniform(0.03, 0.08)))
        else:
            lam_ca = 0.2 + 4.3 * month_prog
            for _ in range(int(rng.poisson(lam_ca))):
                dt = date(yy, mm, int(rng.integers(1, last + 1)))
                plan.append((dt, "cash_advance", None))
            lam_extra = 0.5 + 2.5 * month_prog
            for _ in range(int(rng.poisson(lam_extra))):
                dt = date(yy, mm, int(rng.integers(1, last + 1)))
                cat = "groceries" if rng.random() < 0.6 else "utilities"
                plan.append((dt, cat, None))

        cutoff = default_date if default_date is not None else horizon_end + timedelta(days=1)
        plan.sort(key=lambda e: (e[0], 0 if e[1] == "salary" else 1))
        # stressed clients burn a growing share of their free savings within
        # the month; the drain is spread over the unplanned events
        uncapped = [e for e in plan if e[2] is None]
        drain_per_event = 0.0
        if uncapped:
            free_savings = max(balance - income * 0.4, 0.0)
            month_drain = (0.15 + 0.75 * month_prog) * free_savings
            drain_per_event = month_drain / len(uncapped)
        for dt, cat, amount in plan:
            if dt >= cutoff:
                continue
            if amount is None:
                s = progress_at(dt)
                base = income * rng.uniform(0.06, 0.16) * (0.7 + 0.8 * s) \
                    if cat == "cash_advance" \
                    else income * rng.uniform(0.04, 0.12) * (0.8 + 0.6 * s)
                # thin buffer: stop scaling down so the balance breaks zero
                if balance >= income * 0.4:
                    amount = -(base + drain_per_event)
                else:
                    amount = -base * 1.5
            balance += amount
            tx_rows.append((client_id, dt.isoformat(), float(amount), cat, float(balance)))
        plan = []

    client_row = {
        "client_id": client_id,
        "income_monthly": income,
        "credit_limit": credit_limit,
        "label": 1 if archetype == "decline" else 0,
        "default_date": default_date.isoformat() if default_date else None,
    }
    return client_row, tx_rows, archetype


SCHEMA = """
DROP TABLE IF EXISTS transactions;
DROP TABLE IF EXISTS clients;
DROP TABLE IF EXISTS risk_scores;

CREATE TABLE clients (
    client_id      INTEGER PRIMARY KEY,
    income_monthly REAL NOT NULL,
    credit_limit   REAL NOT NULL,
    label          INTEGER NOT NULL,
    default_date   TEXT
);

CREATE TABLE transactions (
    tx_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    client_id     INTEGER NOT NULL REFERENCES clients(client_id),
    date          TEXT NOT NULL,
    amount        REAL NOT NULL,
    category      TEXT NOT NULL,
    balance_after REAL NOT NULL
);
CREATE INDEX idx_tx_client_date ON transactions(client_id, date);

CREATE TABLE risk_scores (
    client_id    INTEGER PRIMARY KEY REFERENCES clients(client_id),
    computed_at  TEXT NOT NULL,
    risk_score   REAL NOT NULL,
    top_factors  TEXT
);
"""


def generate_population(n_clients, seed, start=date(2025, 1, 1)):
    rng = np.random.default_rng(seed)
    clients, all_tx, meta = [], [], []
    n_decline = max(1, int(round(n_clients * 0.25)))
    archetypes = ["decline"] * n_decline + ["stable"] * (n_clients - n_decline)
    rng.shuffle(archetypes)
    for cid, arch in enumerate(archetypes, start=1):
        row, txs, _arch = simulate_client(cid, rng, arch, start, int(rng.integers(6, 13)))
        clients.append(row)
        all_tx.extend(txs)
        meta.append({"client_id": cid, "archetype": _arch, "balance_series": txs})
    return clients, all_tx, meta


def run(n_clients=80, seed=42, db_path="data/stress_detector.db", plot=False):
    clients, txs, meta = generate_population(n_clients, seed)
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)
        con.executemany(
            "INSERT INTO clients VALUES (:client_id,:income_monthly,:credit_limit,:label,:default_date)",
            clients)
        con.executemany("INSERT INTO transactions(client_id,date,amount,category,balance_after) "
                        "VALUES (?,?,?,?,?)", txs)
        con.commit()
    finally:
        con.close()

    n_defaulters = sum(c["label"] for c in clients)
    print(f"[generate_data] {len(clients)} clients ({n_defaulters} decline/default, "
          f"{len(clients) - n_defaulters} stable), {len(txs)} transactions -> {db_path}")

    od_counts = {}
    for cid, _d, _a, _c, bal in txs:
        if bal < 0:
            od_counts[cid] = od_counts.get(cid, 0) + 1
    decline_ids = {c["client_id"] for c in clients if c["label"] == 1}
    od_decline = sum(1 for cid in od_counts if cid in decline_ids)
    od_stable = len(od_counts) - od_decline
    print(f"[generate_data] overdraft clients: decline {od_decline}/{len(decline_ids)}, "
          f"stable {od_stable}/{len(clients) - len(decline_ids)}")

    if plot:
        _sanity_plot(meta, seed)


def _sanity_plot(meta, seed):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter

    decline = [m for m in meta if m["archetype"] == "decline"][:2]
    stable = [m for m in meta if m["archetype"] == "stable"][:2]
    sample = stable[:1] + decline[:2] + stable[1:]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, m in zip(axes.ravel(), sample):
        xs = [date.fromisoformat(t[1]) for t in m["balance_series"]]
        ys = [t[4] for t in m["balance_series"]]
        ax.plot(xs, ys, lw=0.9)
        cid = m["client_id"]
        arch = m["archetype"]
        ax.set_title(f"id={cid} {arch}")
        ax.xaxis.set_major_locator(AutoDateLocator())
        ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator()))
    fig.suptitle(f"Sanity check (seed={seed})")
    fig.tight_layout()
    fig.savefig("sanity_check.png", dpi=110)
    print("[generate_data] sanity plot -> sanity_check.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clients", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--db", type=str, default="data/stress_detector.db")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    run(args.clients, args.seed, args.db, args.plot)
