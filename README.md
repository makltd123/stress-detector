# Early Financial Stress Detector (prototype)

**English** | [Русский](README.ru.md)

A service for a bank or lender: from a client's transaction patterns it predicts default risk **2–4 weeks before a missed payment** — earlier than classic credit scoring. A prototype on synthetic data: the goal is to show a working idea, not a production system.

**User:** a risk manager, looking at one screen with a list of clients and their risk.

The UI and the human-readable risk factors are in Russian.

---

## 1. Results (seed = 42)

| Metric | Value |
|---|---|
| Population | 80 clients (20 decline → default, 60 stable) |
| Transactions | 11,780 over 6–12 months |
| ROC-AUC (time-based test) | **0.890** |
| Alert threshold (best F1 on train) | 0.58 |
| Median lead time of the first alert | **24 days before default** (min 6, max 180) |
| Defaulters that got an alert | 18 of 20 |
| False alarms on stable clients | 0 (on each client's latest snapshot, as in `/score`) |

**The model overfits.** ROC-AUC is 1.000 on the training windows and 0.890 on the test windows, and `train_model.py` warns about it itself. Part of this comes from the time-based split: only 20 pre-default windows land in train, against 57 in test.

Top features by `feature_importances_`: `net_flow` (0.35), `gap_ratio` (0.14), `ca_share_z` (0.13), `bal_std_z` (0.10), `disc_trend` (0.09). `overdraft_cnt` is not used by the model (importance 0).

---

## 2. Stack

- **Backend:** Python 3.14, FastAPI, uvicorn, SQLite (file `data/stress_detector.db`)
- **ML:** pandas, numpy, scikit-learn (RandomForest), joblib
- **Frontend:** React 18 (Vite), recharts, axios
- **Tests:** pytest

Nothing to set up: SQLite is just a file, the backend is `uvicorn`, the frontend is `npm run dev`.

---

## 3. Project structure

```
├── generate_data.py               # synthetic data generator -> SQLite
├── features.py                    # rolling windows + features relative to the client's own baseline
├── train_model.py                 # RandomForest training, metrics, artifact
├── main.py                        # FastAPI backend (5 endpoints)
├── requirements.txt
├── tests/test_features.py         # tests for features and the generator
├── sanity_check.png               # visual check of the generator (--plot)
├── data/stress_detector.db        # created by generate_data.py (not in the repo)
├── artifacts/model.joblib         # created by train_model.py (not in the repo)
└── frontend/
    ├── src/App.jsx                # one screen: table + expandable client card
    ├── src/api.js                 # axios -> http://localhost:8000
    └── src/index.css
```

---

## 4. Data generator (`generate_data.py`)

Creates clients with a profile `client_id, income_monthly, credit_limit` and transactions `tx_id, client_id, date, amount, category, balance_after` (signed amounts: salary "+", spending "−"). Categories: `groceries / utilities / entertainment / cash_advance / rent / salary / subscription`.

Two archetypes:

1. **Stable (75%)** — an even monthly sawtooth: salary, rent, utilities, groceries about twice a week, entertainment, subscriptions; spending below income, balance slowly growing, no default.
2. **Decline → default (25%)** — a drift starts 4–6 weeks before `default_date`:
   - frequency and size of `cash_advance` grow linearly with the drift;
   - rent "moves" from mid-month to 1–3 days after salary (the "salary → large debit" gap shrinks);
   - entertainment and the second subscription are cut first;
   - the client "burns" a growing share of free savings (debits proportional to the current balance), volatility grows, and with a thin buffer the balance goes below zero → overdrafts;
   - transactions stop 3–14 days before `default_date`.

Ground truth (`label`, `default_date`) is written to the `clients` table — for training and validation only. A fixed seed gives full reproducibility.

```bash
python generate_data.py --clients 80 --seed 42 --plot   # + sanity_check.png
```

### SQLite schema

```sql
clients      (client_id PK, income_monthly, credit_limit, label, default_date)
transactions (tx_id PK, client_id FK, date, amount, category, balance_after)
risk_scores  (client_id PK, computed_at, risk_score, top_factors JSON)
```

---

## 5. Features (`features.py`)

Rolling window of **28 days with a 7-day step**. All key metrics are computed relative to **the client's own baseline** (expanding mean/std over their previous windows), not relative to the population:

| Feature | Meaning |
|---|---|
| `ca_share` | share of cash_advance among the window's transactions |
| `ca_share_z` | z-score of the CA share against the client's own norm |
| `gap_ratio` | days from salary to the first large debit (≥10% of income), relative to the client's average; no salary → 0 |
| `bal_std_norm` | std of the daily balance over the window, normalized by income |
| `bal_std_z` | z-score of volatility |
| `disc_share` / `disc_trend` | share of entertainment+subscription and its shift from baseline (a drop = cutting non-essentials) |
| `overdraft_cnt` | number of transactions with `balance_after < 0` in the window |
| `net_flow` | (inflow − outflow) / income over the window |

Window labels: `label=1` if `default_date` falls within the 28 days after the window ends (the prediction horizon); windows after default are dropped.

---

## 6. Model (`train_model.py`)

- **RandomForest** (300 trees, `min_samples_leaf=10`, `max_depth=8`, `class_weight="balanced_subsample"`) — a single model, no two-stage setup.
- Validation is a **time-based split**: train = windows before the 75th percentile of calendar time, test = the rest.
- The alert threshold is the best F1 on the training probabilities.
- Lead-time analysis: for each defaulter, find the first window with `score ≥ threshold` before default.
- Artifact `artifacts/model.joblib` = `{model, threshold, metrics, features, ...}`.

```bash
python train_model.py          # prints ROC-AUC, lead time, importances
```

Explainability: global `feature_importances_` plus how far the client's feature deviates from the population; top 3 factors = `importance × |population z|`.

---

## 7. Backend (`main.py`)

```bash
python -m uvicorn main:app --reload     # http://localhost:8000, docs: /docs
```

| Endpoint | What it does |
|---|---|
| `POST /generate` | recreates the SQLite database via the generator (demo reset) |
| `POST /score` | computes each client's features as of their latest transaction, predicts, writes `risk_score` and the top 3 factors to `risk_scores` |
| `GET /clients` | list with risk_score, sorted by descending risk; status: `high` (≥ threshold) / `medium` / `low` |
| `GET /clients/{id}` | client card: transactions, factors, demo ground truth (expected default date) |
| `GET /config` | the alert threshold and the model's feature list |

The model is loaded once at startup, so restart the backend after retraining.

Human-readable factors are built from the window's "facts", for example: "Cash advance — 18% of all transactions over 4 weeks (usually ~2%)".

---

## 8. Frontend (`frontend/`)

One screen:

- **"Generate data"** and **"Recalculate risks"** buttons;
- chips: total clients / scored / high risk;
- a table (ID, income, limit, risk score with a colored bar and status, top factor), sorted by risk, with a "high risk only" filter;
- **clicking a row** expands a card right in the table: a daily balance chart (recharts) with a red mark at the expected default (demo ground truth) and the top 3 factors as text.

---

## 9. Full demo cycle

```bash
pip install -r requirements.txt
python generate_data.py --clients 80 --seed 42   # 1. generate data
python train_model.py                            # 2. train the model
python -m uvicorn main:app --reload              # 3. start the API
cd frontend && npm install && npm run dev        # 4. start the UI (http://localhost:5173)
```

In the UI: "Recalculate risks" → the table sorts by risk → click a client with a high score → the balance chart shows the drift in the weeks before the red default line, and the factors explain **why** the model is worried.

Tests:

```bash
pip install pytest
pytest -q
```

## 10. Deliberately left out of the prototype

Postgres, docker-compose, a third archetype (sudden shock), two-stage ML (anomaly detection + gradient boosting + SHAP), separate dashboard/alerts screens, alerts with review/dismiss statuses.

## License

[MIT](LICENSE)
