import { useCallback, useEffect, useState } from 'react'
import { api } from '../api.js'
import { fmtDuration, fmtTime } from '../theme.js'

/**
 * Unclear Possible Leads.
 *
 * These are pairs of sightings the resolver SCORED but REFUSED to assert. They
 * are not identities and are never presented as recognised vehicles.
 *
 * Why this is not simply "identities below 45%": a link must clear
 * accept_threshold to join an identity at all, and an identity's confidence is
 * the minimum of its accepted links -- so no multi-camera identity can exist
 * below that threshold. The sub-threshold evidence is here, in the pairs that
 * were never promoted, which is also where an investigator's interest lies.
 */

const PAGE = 25

export default function LeadsView() {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')
  const [page, setPage] = useState(0)
  const [f, setF] = useState({ camera: '', plate: '', vehicle_class: '', status: '',
                               min_score: 0, max_score: 1 })

  const load = useCallback(async (offset, filters) => {
    setError('')
    try {
      setData(await api.leads({ ...filters, limit: PAGE, offset }))
    } catch {
      setError('Could not load leads. Is the API running?')
    }
  }, [])

  useEffect(() => { load(page * PAGE, f) }, [page, f, load])

  const setFilter = (k, v) => { setPage(0); setF((prev) => ({ ...prev, [k]: v })) }

  const mark = async (id, status) => {
    try {
      await api.setLeadStatus(id, status)
      load(page * PAGE, f)
    } catch { /* leave the row as it was; the reload would show any change */ }
  }

  const total = data?.total ?? 0
  const pages = Math.max(1, Math.ceil(total / PAGE))

  return (
    <div className="scroll">
      <section className="card">
        <div className="card-head">
          <h2>Unclear Possible Leads</h2>
          <span className="card-sub">
            pairs the system scored but refused — evidence for review, not identities
          </span>
        </div>

        <div className="lead-filters">
          <input placeholder="Plate contains…" value={f.plate}
                 onChange={(e) => setFilter('plate', e.target.value)}
                 aria-label="Filter by plate" spellCheck={false} />
          <input placeholder="Camera" value={f.camera}
                 onChange={(e) => setFilter('camera', e.target.value.toUpperCase())}
                 aria-label="Filter by camera" spellCheck={false} />
          <select value={f.vehicle_class} aria-label="Filter by vehicle class"
                  onChange={(e) => setFilter('vehicle_class', e.target.value)}>
            <option value="">Any class</option>
            {['car', 'bus', 'truck', 'motorcycle', 'bicycle'].map((c) =>
              <option key={c} value={c}>{c}</option>)}
          </select>
          <select value={f.status} aria-label="Filter by review status"
                  onChange={(e) => setFilter('status', e.target.value)}>
            <option value="">Any status</option>
            {(data?.statuses || []).map((s) =>
              <option key={s} value={s}>{s.replace('_', ' ')}</option>)}
          </select>
          <label className="lead-range">
            score ≥ {Number(f.min_score).toFixed(2)}
            <input type="range" min="0" max="1" step="0.05" value={f.min_score}
                   onChange={(e) => setFilter('min_score', Number(e.target.value))} />
          </label>
        </div>

        {error && <div className="empty">{error}</div>}
        {!data && !error && <div className="empty">Loading…</div>}

        {data && total === 0 && (
          <div className="empty">
            No unclear leads.
            <span>
              Every cross-camera pair the resolver scored was either accepted outright
              or fell below the near-miss recording floor
              (<code>record_near_miss_above</code>). On a clean dataset this is the
              expected result, not a failure — there was nothing ambiguous to refuse.
            </span>
          </div>
        )}

        {data && total > 0 && (
          <>
            <div className="leads">
              {data.leads.map((l) => (
                <article className="lead" key={l.id}>
                  <div className="lead-head">
                    <span className="chip critical">possible lead</span>
                    <b>{l.from_camera} → {l.to_camera}</b>
                    <span className="lead-score">{l.score.toFixed(2)}</span>
                    <span className={`chip st-${l.status}`}>{l.status.replace('_', ' ')}</span>
                  </div>

                  <div className="lead-pair">
                    {[l.from, l.to].map((s, i) => (
                      <div className="lead-side" key={i}>
                        <div className="lead-side-head">
                          <span className="tag">{s.camera_id}</span>
                          <span className="tag">#{s.track_id}</span>
                          <span className={`ev-plate${s.plate ? '' : ' none'}`}>
                            {s.plate || 'no readable plate'}
                          </span>
                        </div>
                        <div className="lead-side-meta">
                          {fmtTime(s.first_seen)} · {s.vehicle_class}
                          {s.plate_confidence != null &&
                            ` · plate ${(s.plate_confidence * 100).toFixed(0)}%`}
                        </div>
                      </div>
                    ))}
                  </div>

                  <dl className="clink-signals">
                    <div><dt>Plate</dt><dd>{l.plate_similarity.toFixed(2)}</dd></div>
                    <div><dt>Appearance</dt><dd>{l.appearance_similarity.toFixed(2)}</dd></div>
                    <div><dt>Physics</dt><dd>{l.topology_score.toFixed(2)}</dd></div>
                    <div><dt>Gap</dt><dd>{fmtDuration(l.gap_s)}</dd></div>
                    <div><dt>Distance</dt><dd>{l.distance_km.toFixed(1)} km</dd></div>
                    <div><dt>Implied</dt><dd>{Math.round(l.implied_speed_kmph)} km/h</dd></div>
                  </dl>

                  <ul className="lead-why">
                    {l.why_uncertain.map((w, i) => <li key={i}>{w}</li>)}
                  </ul>

                  <div className="lead-actions">
                    {['reviewed', 'potential_match', 'escalated', 'rejected'].map((s) => (
                      <button key={s} type="button"
                              className={l.status === s ? 'on' : undefined}
                              onClick={() => mark(l.id, s)}>
                        {s.replace('_', ' ')}
                      </button>
                    ))}
                  </div>
                </article>
              ))}
            </div>

            <div className="lead-pager">
              <button disabled={page === 0} onClick={() => setPage((p) => p - 1)}>
                ← Newer
              </button>
              <span>
                {page * PAGE + 1}–{Math.min(total, (page + 1) * PAGE)} of {total}
              </span>
              <button disabled={page + 1 >= pages} onClick={() => setPage((p) => p + 1)}>
                Older →
              </button>
            </div>
          </>
        )}
      </section>
    </div>
  )
}
