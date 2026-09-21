import sqlite3
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import features as F
import generate_data as G


def test_build_windows_returns_expected_columns():
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    tx = pd.DataFrame({
        "tx_id": range(len(dates)),
        "client_id": 1,
        "date": dates,
        "amount": np.where(dates.day % 7 == 0, 10_000.0, np.where(dates.day % 3 == 0, -2_000.0, -500.0)),
        "category": np.where(dates.day % 7 == 0, "salary", np.where(dates.day % 3 == 0, "groceries", "utilities")),
        "balance_after": 50_000.0 + np.cumsum(np.where(dates.day % 7 == 0, 10_000.0, np.where(dates.day % 3 == 0, -2_000.0, -500.0))),
    })
    clients = pd.DataFrame({
        "client_id": [1],
        "income_monthly": [100_000.0],
        "credit_limit": [100_000.0],
        "label": [0],
        "default_date": [None],
    }).set_index("client_id")
    df = F.build_windows(clients, tx)
    assert not df.empty
    expected = {"client_id", "window_start", "window_end", *F.FEATURES}
    assert expected.issubset(df.columns)


def test_add_labels_marks_windows_before_default():
    dates = pd.date_range("2025-01-01", periods=60, freq="D")
    tx = pd.DataFrame({
        "tx_id": range(len(dates)),
        "client_id": 1,
        "date": dates,
        "amount": np.where(dates.day % 7 == 0, 10_000.0, -2_000.0),
        "category": np.where(dates.day % 7 == 0, "salary", "groceries"),
        "balance_after": 50_000.0 + np.cumsum(np.where(dates.day % 7 == 0, 10_000.0, -2_000.0)),
    })
    clients = pd.DataFrame({
        "client_id": [1],
        "income_monthly": [100_000.0],
        "credit_limit": [100_000.0],
        "label": [1],
        "default_date": [pd.Timestamp("2025-03-01")],
    }).set_index("client_id")
    windows = F.build_windows(clients, tx)
    assert not windows.empty
    labeled = F.add_labels(windows, clients)
    assert "label" in labeled.columns
    assert labeled["label"].sum() >= 1


def test_client_snapshot_returns_features_and_facts():
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    tx = pd.DataFrame({
        "tx_id": range(len(dates)),
        "client_id": 1,
        "date": dates,
        "amount": np.where(dates.day % 7 == 0, 10_000.0, -2_000.0),
        "category": np.where(dates.day % 7 == 0, "salary", "groceries"),
        "balance_after": 50_000.0 + np.cumsum(np.where(dates.day % 7 == 0, 10_000.0, -2_000.0)),
    })
    snap = F.client_snapshot(tx, 100_000.0)
    assert "features" in snap
    assert "facts" in snap
    assert set(F.FEATURES).issubset(snap["features"].keys())
    assert "as_of" in snap["facts"]


def test_generate_population_deterministic_with_seed():
    c1, t1, _ = G.generate_population(20, seed=7)
    c2, t2, _ = G.generate_population(20, seed=7)
    assert len(c1) == len(c2) == 20
    assert len(t1) == len(t2)
    assert [c["client_id"] for c in c1] == [c["client_id"] for c in c2]
    assert all(c1[i]["income_monthly"] == c2[i]["income_monthly"] for i in range(20))


def test_stable_clients_have_no_default():
    clients, _, _ = G.generate_population(40, seed=1)
    stable = [c for c in clients if c["label"] == 0]
    assert all(c["default_date"] is None for c in stable)


def test_decline_clients_have_default_date():
    clients, _, _ = G.generate_population(40, seed=1)
    decline = [c for c in clients if c["label"] == 1]
    assert all(c["default_date"] is not None for c in decline)


def test_write_to_sqlite_creates_tables(tmp_path):
    db = tmp_path / "test.db"
    G.run(n_clients=10, seed=3, db_path=str(db), plot=False)
    con = sqlite3.connect(str(db))
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert {"clients", "transactions", "risk_scores"}.issubset(tables)
        n_clients = con.execute("SELECT COUNT(*) FROM clients").fetchone()[0]
        n_tx = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        assert n_clients == 10
        assert n_tx > 0
    finally:
        con.close()
