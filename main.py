"""FastAPI backend for the Early Financial Stress Detector prototype.

Run:  uvicorn main:app --reload
"""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import features as F

DB_PATH = "data/stress_detector.db"
MODEL_PATH = "artifacts/model.joblib"

app = FastAPI(title="Early Financial Stress Detector (prototype)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)


class GenerateResponse(BaseModel):
    status: str
    clients: int
    transactions: int


class ScoreResponse(BaseModel):
    scored: int
    at_risk: int
    threshold: float


class ConfigResponse(BaseModel):
    threshold: float
    features: list[str]


class ClientFactor(BaseModel):
    factor: str
    detail: str


class ClientOut(BaseModel):
    client_id: int
    income_monthly: float
    credit_limit: float
    risk_score: float | None
    status: str | None
    computed_at: str | None
    top_factor: str | None


class ClientDetailOut(ClientOut):
    top_factors: list[ClientFactor]
    transactions: list[dict]
    demo_truth: dict | None


_model_artifact = None


def _load_model_once():
    global _model_artifact
    if _model_artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(404, "Модель не обучена — запустите train_model.py")
        _model_artifact = joblib.load(MODEL_PATH)
    return _model_artifact


@app.on_event("startup")
def _preload_model():
    try:
        _load_model_once()
    except HTTPException:
        pass


def _factor_text(name, feats, facts):
    ca, gap = facts.get("ca_share"), facts.get("gap_days")
    if name == "ca_share":
        base = facts.get("ca_base")
        b = f" (обычно ~{base:.0%})" if base is not None else ""
        return f"Cash advance — {ca:.0%} всех транзакций за 4 нед{b}"
    if name == "gap_ratio":
        d = "нет зарплатного зачисления в окне" if gap is None or (
            isinstance(gap, float) and gap != gap) else f"{gap:.0f} дн против обычных {facts.get('gap_base', 25):.0f} дн"
        return f"Сжался разрыв «зарплата → крупное списание»: {d}"
    if name == "bal_std_z":
        return f"Волатильность дневного баланса выросла (z={feats['bal_std_z']:+.1f})"
    if name == "disc_trend":
        b = facts.get("disc_base")
        cur = facts.get("disc_share")
        extra = f", сейчас {cur:.0%}" if cur is not None and cur == cur else ""
        bb = f" против обычных {b:.0%}" if b else ""
        return f"Урезаны необязательные траты{bb}{extra}"
    if name == "overdraft_cnt":
        return f"Овердрафтов за последние 4 недели: {facts.get('overdraft_cnt', 0)}"
    if name == "net_flow":
        nf = facts.get("net_flow") or 0.0
        return f"Чистый отток средств {-nf:.0%} месячного дохода за окно"
    if name == "bal_std_norm":
        return f"Высокая волатильность баланса ({feats['bal_std_norm']:.0%} дохода)"
    if name == "disc_share":
        return f"Низкая доля дискреционных трат ({feats['disc_share']:.0%})"
    if name == "ca_share_z":
        return f"Cash advance аномально выше собственной нормы (z={feats['ca_share_z']:+.1f})"
    return name


@contextmanager
def db():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
    finally:
        con.close()


@app.post("/generate", response_model=GenerateResponse)
def generate():
    try:
        import generate_data
        generate_data.run(n_clients=80, seed=42, db_path=DB_PATH)
    except Exception as exc:
        raise HTTPException(500, f"Ошибка генерации данных: {exc}") from exc
    try:
        with db() as con:
            n_clients = con.execute("SELECT COUNT(*) c FROM clients").fetchone()["c"]
            n_tx = con.execute("SELECT COUNT(*) c FROM transactions").fetchone()["c"]
    except sqlite3.Error as exc:
        raise HTTPException(500, f"Ошибка чтения БД: {exc}") from exc
    return GenerateResponse(status="ok", clients=n_clients, transactions=n_tx)


@app.post("/score", response_model=ScoreResponse)
def score():
    try:
        art = _load_model_once()
        model, thr = art["model"], art["threshold"]

        with db() as con:
            clients = pd.read_sql(
                "SELECT client_id, income_monthly FROM clients", con).set_index("client_id")
            tx = pd.read_sql(
                "SELECT tx_id, client_id, date, amount, category, balance_after "
                "FROM transactions ORDER BY client_id, date", con)
    except sqlite3.Error as exc:
        raise HTTPException(500, f"Ошибка чтения БД: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"Ошибка подготовки данных: {exc}") from exc

    if clients.empty:
        raise HTTPException(400, "В базе нет клиентов — сгенерируйте данные")

    tx["date"] = pd.to_datetime(tx["date"])

    try:
        snapshots = {}
        for cid, ctx in tx.groupby("client_id"):
            snapshots[cid] = F.client_snapshot(ctx, clients.loc[cid, "income_monthly"])

        X = pd.DataFrame([s["features"] for s in snapshots.values()],
                         columns=F.FEATURES) if snapshots else pd.DataFrame(columns=F.FEATURES)
        probs = model.predict_proba(X[F.FEATURES])[:, 1] if len(X) else []

        imp = dict(zip(F.FEATURES, model.feature_importances_))
        means = art.get("feature_means") or X.mean().to_dict()
        stds = art.get("feature_stds") or X.std().to_dict()
        zs = (X - pd.Series(means)) / (pd.Series(stds) + 1e-9)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = []
        for i, (cid, snap) in enumerate(snapshots.items()):
            dev = {f: abs(float(zs.iloc[i][f])) for f in F.FEATURES}
            top = sorted(F.FEATURES, key=lambda f: -(imp[f] * dev[f]))[:3]
            facts = dict(snap["facts"])
            factors = [{"factor": f, "detail": _factor_text(f, snap["features"], facts)}
                       for f in top]
            rows.append((int(cid), now, round(float(probs[i]), 4),
                         json.dumps(factors, ensure_ascii=False)))
    except Exception as exc:
        raise HTTPException(500, f"Ошибка скоринга: {exc}") from exc

    try:
        with db() as con:
            con.executemany(
                "INSERT INTO risk_scores(client_id, computed_at, risk_score, top_factors) "
                "VALUES (?,?,?,?) ON CONFLICT(client_id) DO UPDATE SET "
                "computed_at=excluded.computed_at, risk_score=excluded.risk_score, "
                "top_factors=excluded.top_factors", rows)
            con.commit()
    except sqlite3.Error as exc:
        raise HTTPException(500, f"Ошибка записи результатов: {exc}") from exc

    at_risk = sum(1 for r in rows if r[2] >= thr)
    return ScoreResponse(scored=len(rows), at_risk=at_risk, threshold=thr)


def _status(p, thr):
    if p is None:
        return None
    if p >= thr:
        return "high"
    if p >= thr * 0.6:
        return "medium"
    return "low"


@app.get("/config", response_model=ConfigResponse)
def get_config():
    art = _load_model_once()
    return ConfigResponse(threshold=art["threshold"], features=art["features"])


@app.get("/clients", response_model=list[ClientOut])
def list_clients():
    art = _load_model_once()
    thr = art["threshold"]
    try:
        with db() as con:
            rows = [dict(r) for r in con.execute(
                "SELECT c.client_id, c.income_monthly, c.credit_limit, "
                "       r.risk_score, r.top_factors, r.computed_at "
                "FROM clients c LEFT JOIN risk_scores r ON r.client_id=c.client_id "
                "ORDER BY (r.risk_score IS NULL), r.risk_score DESC, c.client_id")]
    except sqlite3.Error as exc:
        raise HTTPException(500, f"Ошибка чтения БД: {exc}") from exc
    out = []
    for r in rows:
        factors = json.loads(r["top_factors"]) if r["top_factors"] else []
        out.append(ClientOut(
            client_id=r["client_id"],
            income_monthly=r["income_monthly"],
            credit_limit=r["credit_limit"],
            risk_score=r["risk_score"],
            status=_status(r["risk_score"], thr),
            computed_at=r["computed_at"],
            top_factor=(factors[0]["detail"] if factors else None),
        ))
    return out


@app.get("/clients/{client_id}", response_model=ClientDetailOut)
def client_detail(client_id: int):
    art = _load_model_once()
    thr = art["threshold"]
    try:
        with db() as con:
            row = con.execute(
                "SELECT c.*, r.risk_score, r.top_factors, r.computed_at "
                "FROM clients c LEFT JOIN risk_scores r ON r.client_id=c.client_id "
                "WHERE c.client_id=?", (client_id,)).fetchone()
            if row is None:
                raise HTTPException(404, "Клиент не найден")
            txs = [dict(t) for t in con.execute(
                "SELECT tx_id, date, amount, category, balance_after FROM transactions "
                "WHERE client_id=? ORDER BY date", (client_id,))]
    except sqlite3.Error as exc:
        raise HTTPException(500, f"Ошибка чтения БД: {exc}") from exc
    factors_raw = json.loads(row["top_factors"]) if row["top_factors"] else []

    demo_truth = None
    if row["label"] == 1 and row["default_date"]:
        dd = date.fromisoformat(row["default_date"])
        demo_truth = {"expected_default_date": row["default_date"],
                      "days_from_today": (dd - date.today()).days}

    return ClientDetailOut(
        client_id=client_id,
        income_monthly=row["income_monthly"],
        credit_limit=row["credit_limit"],
        risk_score=row["risk_score"],
        status=_status(row["risk_score"], thr),
        computed_at=row["computed_at"],
        top_factors=[ClientFactor(**f) for f in factors_raw],
        transactions=txs,
        demo_truth=demo_truth,
    )
