# Early Financial Stress Detector (прототип)

[English](README.md) | **Русский**

Сервис для банка или кредитора: по транзакционным паттернам клиента предсказывает риск дефолта за **2–4 недели до просрочки** — раньше, чем классический скоринг. Прототип на синтетических данных: цель — показать работающую идею, а не production-систему.

**Пользователь:** риск-менеджер. Он смотрит один экран со списком клиентов и их риском.

---

## 1. Результаты прогона (seed = 42)

| Метрика | Значение |
|---|---|
| Популяция | 80 клиентов (20 decline → default, 60 stable) |
| Транзакции | 11 780 за 6–12 месяцев |
| ROC-AUC (time-based test) | **0.890** |
| Порог алерта (best-F1 на train) | 0.58 |
| Медианный lead-time первого алерта | **24 дня до дефолта** (min 6, max 180) |
| Дефолтники с алертом | 18 из 20 |
| Ложные срабатывания на stable | 0 (по последнему снимку каждого клиента, как в `/score`) |

**Модель переобучается.** На обучающих окнах ROC-AUC = 1.000, на тестовых — 0.890, и `train_model.py` сам об этом предупреждает. Отчасти это следствие разбиения по времени: в train попадает всего 20 предефолтных окон, а в test — 57.

Важнейшие фичи по `feature_importances_`: `net_flow` (0.35), `gap_ratio` (0.14), `ca_share_z` (0.13), `bal_std_z` (0.10), `disc_trend` (0.09). `overdraft_cnt` модель не использует (важность 0).

---

## 2. Стек

- **Backend:** Python 3.14, FastAPI, uvicorn, SQLite (файл `data/stress_detector.db`)
- **ML:** pandas, numpy, scikit-learn (RandomForest), joblib
- **Frontend:** React 18 (Vite), recharts, axios
- **Тесты:** pytest

Ничего поднимать не надо: SQLite — просто файл, backend — `uvicorn`, frontend — `npm run dev`.

---

## 3. Структура проекта

```
├── generate_data.py               # генератор синтетики -> SQLite
├── features.py                    # скользящие окна + фичи относительно baseline клиента
├── train_model.py                 # обучение RandomForest, метрики, артефакт
├── main.py                        # FastAPI backend (5 эндпоинтов)
├── requirements.txt
├── tests/test_features.py         # тесты фич и генератора
├── sanity_check.png               # визуальная проверка генератора (--plot)
├── data/stress_detector.db        # создаётся generate_data.py (в репозиторий не входит)
├── artifacts/model.joblib         # создаётся train_model.py (в репозиторий не входит)
└── frontend/
    ├── src/App.jsx                # один экран: таблица + разворачивающаяся карточка
    ├── src/api.js                 # axios -> http://localhost:8000
    └── src/index.css
```

---

## 4. Генератор данных (`generate_data.py`)

Создаёт клиентов с профилем `client_id, income_monthly, credit_limit` и транзакции `tx_id, client_id, date, amount, category, balance_after` (суммы со знаком: зарплата «+», траты «−»). Категории: `groceries / utilities / entertainment / cash_advance / rent / salary / subscription`.

Два архетипа:

1. **Stable (75%)** — ровная месячная пила: зарплата, аренда, коммуналка, продукты ~2 раза в неделю, развлечения, подписки; траты меньше дохода, баланс медленно растёт, дефолта нет.
2. **Decline → default (25%)** — за 4–6 недель до `default_date` начинается дрейф:
   - частота и размер `cash_advance` растут линейно по прогрессу дрейфа;
   - аренда «переезжает» с середины месяца на 1–3 дня после зарплаты (сжимается промежуток «зарплата → крупное списание»);
   - развлечения и вторая подписка урезаются первыми;
   - клиент «сжигает» растущую долю свободных сбережений (списания пропорциональны текущему балансу), волатильность растёт, при тонком буфере баланс пробивает ноль → овердрафты;
   - транзакции обрываются за 3–14 дней до `default_date`.

Ground truth (`label`, `default_date`) пишется в таблицу `clients` — только для обучения и валидации. Фиксированный сид даёт полную воспроизводимость.

```bash
python generate_data.py --clients 80 --seed 42 --plot   # + sanity_check.png
```

### Схема SQLite

```sql
clients      (client_id PK, income_monthly, credit_limit, label, default_date)
transactions (tx_id PK, client_id FK, date, amount, category, balance_after)
risk_scores  (client_id PK, computed_at, risk_score, top_factors JSON)
```

---

## 5. Фичи (`features.py`)

Скользящее окно **28 дней, шаг 7 дней**. Все ключевые метрики считаются относительно **собственного baseline клиента** (expanding mean/std по его предыдущим окнам), а не относительно популяции:

| Фича | Смысл |
|---|---|
| `ca_share` | доля cash_advance от числа транзакций за окно |
| `ca_share_z` | z-score доли CA против собственной нормы |
| `gap_ratio` | дни «зарплата → первое крупное списание (≥10% дохода)» относительно своего среднего; нет зарплаты → 0 |
| `bal_std_norm` | std дневного баланса за окно, нормированное на доход |
| `bal_std_z` | z-score волатильности |
| `disc_share` / `disc_trend` | доля entertainment+subscription и её сдвиг от baseline (падение = клиент режет необязательное) |
| `overdraft_cnt` | число транзакций с `balance_after < 0` за окно |
| `net_flow` | (приток − отток) / доход за окно |

Разметка окон: `label=1`, если `default_date` попадает в следующие 28 дней после конца окна (горизонт предсказания); окна после дефолта отбрасываются.

---

## 6. Модель (`train_model.py`)

- **RandomForest** (300 деревьев, `min_samples_leaf=10`, `max_depth=8`, `class_weight="balanced_subsample"`) — одна модель, без two-stage.
- Валидация — **разбиение по времени**: train = окна до 75-го перцентиля календарного времени, test = остальные.
- Порог алерта — лучший F1 на train-вероятностях.
- Анализ lead-time: для каждого дефолтника ищется первое окно с `score ≥ threshold` до дефолта.
- Артефакт `artifacts/model.joblib` = `{model, threshold, metrics, features, ...}`.

```bash
python train_model.py          # печатает ROC-AUC, lead-time, importances
```

Объяснимость — глобальные `feature_importances_` плюс отклонение фичи клиента от популяции: топ-3 фактора = `importance × |z_популяции|`.

---

## 7. Backend (`main.py`)

```bash
python -m uvicorn main:app --reload     # http://localhost:8000, docs: /docs
```

| Эндпоинт | Что делает |
|---|---|
| `POST /generate` | пересоздаёт SQLite через генератор (демо-сброс) |
| `POST /score` | считает фичи каждого клиента на дату последней транзакции, предсказывает, пишет `risk_score` и топ-3 фактора в `risk_scores` |
| `GET /clients` | список с risk_score, сортировка по убыванию риска; статус: `high` (≥ порога) / `medium` / `low` |
| `GET /clients/{id}` | карточка: транзакции, факторы, demo ground-truth (ожидаемая дата дефолта) |
| `GET /config` | порог алерта и список фич модели |

Модель загружается один раз при старте, поэтому после переобучения backend нужно перезапустить.

Человекочитаемые факторы формируются из «фактов» окна, например: «Cash advance — 18% всех транзакций за 4 нед (обычно ~2%)».

---

## 8. Frontend (`frontend/`)

Один экран:

- кнопки **«Сгенерировать данные»** и **«Пересчитать риски»**;
- чипы: всего клиентов / проскорено / высокий риск;
- таблица (ID, доход, лимит, risk score с цветной полосой и статусом, топ-фактор), сортировка по риску, фильтр «только высокий риск»;
- **клик по строке** разворачивает карточку прямо в таблице: график баланса по дням (recharts) с красной отметкой ожидаемого дефолта (demo ground-truth) и список топ-3 факторов текстом.

---

## 9. Полный цикл демо

```bash
pip install -r requirements.txt
python generate_data.py --clients 80 --seed 42   # 1. сгенерировать данные
python train_model.py                            # 2. обучить модель
python -m uvicorn main:app --reload              # 3. поднять API
cd frontend && npm install && npm run dev        # 4. поднять UI (http://localhost:5173)
```

В UI: «Пересчитать риски» → таблица отсортируется по риску → клик по клиенту с высоким score → график баланса показывает дрейф за недели до красной линии дефолта, а факторы объясняют, **почему** модель тревожится.

Тесты:

```bash
pip install pytest
pytest -q
```

## 10. Что сознательно не вошло в прототип

Postgres, docker-compose, третий архетип (sudden shock), two-stage ML (anomaly detection + gradient boosting + SHAP), отдельные экраны dashboard/alerts, алерты со статусами review/dismiss.

## Лицензия

[MIT](LICENSE)

---

Автор: [Федор Маковский](https://fedor.alqira.ru), ИИ-разработчик. Услуги и проекты с цифрами качества: [fedor.alqira.ru](https://fedor.alqira.ru).
