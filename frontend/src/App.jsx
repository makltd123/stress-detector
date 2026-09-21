import { Fragment, useEffect, useState } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { api } from './api'

const STATUS_LABEL = { high: 'Высокий', medium: 'Средний', low: 'Низкий' }

const fmtMoney = (v) => new Intl.NumberFormat('ru-RU').format(Math.round(v ?? 0))
const fmtDay = (ts) =>
  new Date(ts).toLocaleDateString('ru-RU', { day: '2-digit', month: 'short' })

function ScoreCell({ score, status, threshold }) {
  if (score == null) return <span className="badge none">нет оценки</span>
  const color = score >= threshold ? 'var(--high)' : score >= threshold * 0.6 ? 'var(--medium)' : 'var(--low)'
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
      <span className="num-cell" style={{ minWidth: 38 }}>{score.toFixed(2)}</span>
      <div className="score-bar"><i style={{ width: `${Math.min(score * 100, 100)}%`, background: color }} /></div>
      <span className={`badge ${status ?? 'none'}`}>{STATUS_LABEL[status] ?? '—'}</span>
    </div>
  )
}

function BalanceChart({ txs, demoTruth }) {
  const data = txs.map((t) => ({
    ts: new Date(t.date).getTime(),
    balance: Math.round(t.balance_after),
  }))
  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
        <CartesianGrid stroke="#26304a" strokeDasharray="3 3" />
        <XAxis
          dataKey="ts" type="number" scale="time" domain={['dataMin', 'dataMax']}
          tick={{ fill: '#8b96ad', fontSize: 11 }} tickFormatter={fmtDay} minTickGap={48}
        />
        <YAxis
          tick={{ fill: '#8b96ad', fontSize: 11 }} width={58}
          tickFormatter={(v) => `${Math.round(v / 1000)}k`} domain={['auto', 'auto']}
        />
        <Tooltip
          contentStyle={{ background: '#1c2438', border: '1px solid #26304a', borderRadius: 8 }}
          labelFormatter={fmtDay}
          formatter={(v) => [fmtMoney(v) + ' ₽', 'Баланс']}
        />
        {demoTruth && (
          <ReferenceLine
            x={new Date(demoTruth.expected_default_date).getTime()}
            stroke="#ff6b6b" strokeDasharray="5 4"
            label={{ value: 'ожидаемый дефолт', fill: '#ff6b6b', fontSize: 11, position: 'insideTopRight' }}
          />
        )}
        <Line type="monotone" dataKey="balance" stroke="#5aa9ff" strokeWidth={1.6} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  )
}

function ClientDetail({ clientId }) {
  const [data, setData] = useState(null)
  const [err, setErr] = useState(null)
  useEffect(() => {
    let live = true
    setData(null); setErr(null)
    api.get(`/clients/${clientId}`)
      .then((r) => live && setData(r.data))
      .catch((e) => live && setErr(e?.response?.data?.detail ?? e.message))
    return () => { live = false }
  }, [clientId])

  if (err) return <td colSpan={6} className="detail"><div className="err">Ошибка: {err}</div></td>
  if (!data) return <td colSpan={6} className="detail"><div className="empty">Загрузка…</div></td>

  const risk = data.risk_score
  const truth = data.demo_truth
  return (
    <td colSpan={6} className="detail">
      <div className="detail-card">
        <div>
          <h3>Баланс по дням{risk != null && ` · риск ${(risk * 100).toFixed(0)}%`}</h3>
          <div className="chart-box">
            <BalanceChart txs={data.transactions} demoTruth={truth} />
          </div>
          <div className="meta-line">
            Транзакций: {data.transactions.length} · Доход {fmtMoney(data.income_monthly)} ₽/мес ·
            Лимит {fmtMoney(data.credit_limit)} ₽
          </div>
        </div>
        <div>
          <h3>Топ-факторы риска</h3>
          {data.top_factors.length === 0 && <div className="meta-line">Нет скоринга — запустите «Пересчитать риски».</div>}
          <ul className="factors">
            {data.top_factors.map((f, i) => <li key={i}>{f.detail}</li>)}
          </ul>
          {truth && (
            <div className="truth">
              Демо ground-truth: ожидаемый дефолт {truth.expected_default_date}
              {truth.days_from_today > 0
                ? (` — через ${truth.days_from_today} дн.`)
                : ' (дата в прошлом)'}
            </div>
          )}
        </div>
      </div>
    </td>
  )
}

export default function App() {
  const [clients, setClients] = useState([])
  const [expanded, setExpanded] = useState(null)
  const [filter, setFilter] = useState('all')
  const [busy, setBusy] = useState('')
  const [error, setError] = useState(null)
  const [threshold, setThreshold] = useState(0.5)

  useEffect(() => {
    api.get('/config').then((r) => setThreshold(r.data.threshold)).catch(() => {})
  }, [])

  const loadClients = () =>
    api.get('/clients').then((r) => setClients(r.data)).catch((e) => setError(errText(e)))

  useEffect(() => { loadClients().catch(() => {}) }, [])

  const run = async (action, label) => {
    setBusy(label); setError(null)
    try {
      const r = await action()
      await loadClients()
      return r
    } catch (e) {
      setError(errText(e))
    } finally {
      setBusy('')
    }
  }

  const visible = filter === 'all' ? clients : clients.filter((c) => c.status === 'high')
  const highCount = clients.filter((c) => c.status === 'high').length
  const scored = clients.filter((c) => c.risk_score != null).length

  return (
    <div className="container">
      <header className="topbar">
        <h1>Early Financial Stress Detector <span>прототип · синтетические данные</span></h1>
        <div className="actions">
          <button className="btn secondary" disabled={!!busy}
                  onClick={() => run(() => api.post('/generate'), 'gen')}>
            {busy === 'gen' ? 'Генерация…' : 'Сгенерировать данные'}
          </button>
          <button className="btn" disabled={!!busy}
                  onClick={() => run(() => api.post('/score'), 'score')}>
            {busy === 'score' ? 'Скоринг…' : 'Пересчитать риски'}
          </button>
        </div>
      </header>

      {error && <div className="err">⚠ {error}</div>}

      <div className="chips">
        <div className="chip"><div className="num">{clients.length}</div><div className="lbl">клиентов в базе</div></div>
        <div className="chip"><div className="num">{scored}</div><div className="lbl">проскорено</div></div>
        <div className="chip">
          <div className="num" style={{ color: highCount ? 'var(--high)' : undefined }}>{highCount}</div>
          <div className="lbl">высокий риск дефолта</div>
        </div>
      </div>

      <div className="toolbar">
        <select value={filter} onChange={(e) => setFilter(e.target.value)}>
          <option value="all">Все клиенты</option>
          <option value="high">Только высокий риск</option>
        </select>
        <span className="meta-line">клик по строке — карточка клиента</span>
      </div>

      <table className="tbl">
        <thead>
          <tr>
            <th style={{ width: 70 }}>ID</th>
            <th>Доход, ₽/мес</th>
            <th>Кредитный лимит</th>
            <th>Risk score</th>
            <th style={{ width: '30%' }}>Топ-фактор</th>
          </tr>
        </thead>
        <tbody>
          {visible.length === 0 && (
            <tr><td colSpan={5}><div className="empty">
              Нет данных. Нажмите «Сгенерировать данные», затем «Пересчитать риски».
            </div></td></tr>
          )}
          {visible.map((c) => (
            <Fragment key={c.client_id}>
              <tr className="client-row"
                  onClick={() => setExpanded(expanded === c.client_id ? null : c.client_id)}>
                <td className="num-cell">{c.client_id}</td>
                <td className="num-cell">{fmtMoney(c.income_monthly)}</td>
                <td className="num-cell">{fmtMoney(c.credit_limit)}</td>
                <td><ScoreCell score={c.risk_score} status={c.status} threshold={threshold} /></td>
                <td>{c.top_factor ?? '—'}</td>
              </tr>
              {expanded === c.client_id && (
                <tr className="expanded">
                  <ClientDetail clientId={c.client_id} />
                </tr>
              )}
            </Fragment>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function errText(e) {
  return e?.response?.data?.detail ?? e.message ?? String(e)
}
