import { useEffect, useRef, useState } from 'react'
import { api } from '../api.js'
import JourneyTimeline from './JourneyTimeline.jsx'

export default function PlateSearch({ journeys, onSelect, selected, onOpenCase }) {
  const [query, setQuery] = useState('')
  // Single click traces; double click opens the case. A lone click therefore
  // has to wait long enough to be sure a second one is not coming, otherwise a
  // double click fires the trace twice on its way to opening the case.
  const clickTimer = useRef(null)
  useEffect(() => () => clearTimeout(clickTimer.current), [])

  const handleClick = (j) => {
    clearTimeout(clickTimer.current)
    clickTimer.current = setTimeout(() => onSelect(j), 220)
  }
  const handleDoubleClick = (j) => {
    clearTimeout(clickTimer.current)          // cancel the pending trace
    onOpenCase?.(j.plate || j.identity_key)
  }
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    const plate = query.trim()
    if (!plate) return
    setBusy(true); setError('')
    try {
      const rows = await api.journeyByPlate(plate)
      onSelect(rows[0])
    } catch {
      setError(`No reconstructed journey for ${plate.toUpperCase()}.`)
      onSelect(null)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="search">
      <form onSubmit={submit} className="search-form">
        <input value={query} onChange={(e) => setQuery(e.target.value)}
               placeholder="Plate, e.g. KA01AB1234" aria-label="Search by plate"
               spellCheck={false} autoComplete="off" />
        <button type="submit" disabled={busy}>{busy ? '…' : 'Trace'}</button>
      </form>
      {error && <p className="search-error">{error}</p>}

      <div className="panel-title">
        Reconstructed journeys<span className="count">{journeys.length}</span>
      </div>
      <div className="journey-list">
        {journeys.length === 0 && (
          <div className="empty small">
            No journeys yet. Run <code>make resolve</code> after processing cameras.
          </div>
        )}
        {journeys.map((j) => (
          <div className="jrow-wrap" key={j.identity_key}>
            <button type="button"
                    className={`jrow${selected?.identity_key === j.identity_key ? ' on' : ''}`}
                    onClick={() => handleClick(j)}
                    onDoubleClick={() => handleDoubleClick(j)}
                    title="Click to replay the route · double-click to open the case">
              <span className="jrow-plate">{j.plate || j.identity_key}</span>
              <span className="jrow-route">{j.route.join(' → ')}</span>
              <span className="jrow-meta">
                {j.total_distance_km.toFixed(1)} km · conf {j.confidence.toFixed(2)}
              </span>
            </button>
            {/* Keyboard- and screen-reader-reachable equivalent of the
                double click, which is not an accessible interaction alone. */}
            <button type="button" className="jrow-case"
                    onClick={() => handleDoubleClick(j)}
                    aria-label={`Open vehicle case for ${j.plate || j.identity_key}`}>
              case
            </button>
          </div>
        ))}
      </div>

      {selected && <JourneyTimeline journey={selected} />}
    </div>
  )
}
