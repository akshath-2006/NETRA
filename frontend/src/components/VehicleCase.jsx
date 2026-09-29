import { useEffect, useState } from 'react'
import { api } from '../api.js'
import { CONGESTION, fmtDuration, fmtTime } from '../theme.js'

/**
 * The investigation view for one vehicle.
 *
 * Every value here is read from stored evidence. Where a piece of evidence was
 * never retained the UI says so in words -- it never renders a placeholder that
 * could be mistaken for a real observation.
 */

function Stat({ label, value, tone }) {
  return (
    <div className="case-stat">
      <dt>{label}</dt>
      <dd className={tone ? `t-${tone}` : undefined}>{value}</dd>
    </div>
  )
}

function Snapshot({ s }) {
  const [failed, setFailed] = useState(false)
  if (!s.snapshot || failed) {
    return (
      <div className="snap snap-none">
        <span>No snapshot retained</span>
        <small>
          {s.snapshot
            ? 'the image could not be loaded'
            : 'this run predates evidence retention, or it was disabled'}
        </small>
      </div>
    )
  }
  return (
    // Lazy: the browser fetches these only as the case view scrolls them in,
    // so opening a case does not pull every image at once.
    <img className="snap" src={s.snapshot} alt={`${s.camera_id} track ${s.track_id}`}
         loading="lazy" decoding="async" onError={() => setFailed(true)} />
  )
}

function LinkRow({ l }) {
  return (
    <div className={`clink${l.accepted ? ' ok' : ' no'}`}>
      <div className="clink-head">
        <b>{l.from_camera} → {l.to_camera}</b>
        <span className={`chip ${l.accepted ? 'ok' : 'critical'}`}>
          {l.accepted ? 'linked' : 'refused'}
        </span>
        <span className="clink-score">{l.score.toFixed(2)}</span>
      </div>
      <dl className="clink-signals">
        <div><dt>Plate</dt><dd>{l.plate_similarity.toFixed(2)}</dd></div>
        <div><dt>Appearance</dt><dd>{l.appearance_similarity.toFixed(2)}</dd></div>
        <div><dt>Physics</dt><dd>{l.topology_score.toFixed(2)}</dd></div>
        <div><dt>Gap</dt><dd>{fmtDuration(l.gap_s)}</dd></div>
        <div><dt>Distance</dt><dd>{l.distance_km.toFixed(1)} km</dd></div>
        <div><dt>Implied</dt>
          <dd style={{ color: l.implied_speed_kmph > 150 ? CONGESTION.severe : undefined }}>
            {Math.round(l.implied_speed_kmph)} km/h
          </dd>
        </div>
      </dl>
      {l.reason && <p className="clink-reason">{l.reason}</p>}
    </div>
  )
}

export default function VehicleCase({ plate, onClose, onFocusCamera }) {
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let cancelled = false
    setData(null); setError('')
    api.vehicleCase(plate)
      .then((d) => { if (!cancelled) setData(d) })
      .catch(() => { if (!cancelled) setError(`No stored evidence for ${plate}.`) })
    return () => { cancelled = true }
  }, [plate])

  useEffect(() => {
    const esc = (e) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', esc)
    return () => window.removeEventListener('keydown', esc)
  }, [onClose])

  const lead = data && data.status !== 'confirmed'

  return (
    <div className="case-scrim" onClick={onClose}>
      <aside className="case" onClick={(e) => e.stopPropagation()}
             role="dialog" aria-modal="true" aria-label={`Vehicle case ${plate}`}>
        <header className="case-head">
          <div>
            <div className="eyebrow">Vehicle case</div>
            <h2>{plate}</h2>
          </div>
          <button className="case-close" onClick={onClose} aria-label="Close">✕</button>
        </header>

        {error && <div className="empty">{error}</div>}
        {!data && !error && <div className="empty">Loading evidence…</div>}

        {data && (
          <div className="case-body">
            {lead && (
              <div className="case-banner">
                <b>LOW-CONFIDENCE POSSIBLE LEAD</b>
                <span>
                  Identity confidence {(data.identity_confidence * 100).toFixed(0)}% is below
                  the {(data.display_threshold * 100).toFixed(0)}% display threshold.
                  This is <b>not</b> a confirmed identity. Requires human review.
                </span>
              </div>
            )}

            <section className="case-sec">
              <div className="eyebrow">Identity</div>
              <dl className="case-stats">
                <Stat label="Class" value={data.vehicle_class} />
                <Stat label="Plate confidence"
                      value={`${(data.plate_confidence * 100).toFixed(0)}%`}
                      tone={data.plate_status === 'confirmed' ? 'ok' : 'warn'} />
                <Stat label="Identity confidence"
                      value={`${(data.identity_confidence * 100).toFixed(0)}%`}
                      tone={lead ? 'warn' : 'ok'} />
                <Stat label="Cameras" value={data.cameras.length} />
                <Stat label="Sightings" value={data.sighting_count} />
                <Stat label="Links" value={`${data.accepted_links} linked · ${data.refused_links} refused`} />
                <Stat label="First seen" value={fmtTime(data.first_seen)} />
                <Stat label="Last seen" value={fmtTime(data.last_seen)} />
              </dl>
              <p className="case-note">
                Identity confidence is the <b>{data.confidence_basis}</b> — not the
                detector's confidence that a vehicle is present.
              </p>
            </section>

            {data.journey ? (
              <section className="case-sec">
                <div className="eyebrow">Journey</div>
                <p className="case-route">{data.journey.route.join('  →  ')}</p>
                <dl className="case-stats">
                  <Stat label="Distance" value={`${data.journey.total_distance_km.toFixed(1)} km`} />
                  <Stat label="Duration" value={fmtDuration(data.journey.total_duration_s)} />
                  <Stat label="Average" value={`${Math.round(data.journey.average_speed_kmph)} km/h`} />
                  <Stat label="Confidence" value={data.journey.confidence.toFixed(2)} />
                </dl>
                {data.journey.coverage_gaps.length > 0 && (
                  <p className="case-gap">
                    <b>Coverage gap.</b> {data.journey.coverage_gaps.join(', ')} sits on this
                    route but recorded nothing — inferred, not observed.
                  </p>
                )}
              </section>
            ) : (
              <section className="case-sec">
                <div className="eyebrow">Journey</div>
                <div className="empty small">
                  No reconstructed journey. This vehicle was linked across fewer than
                  two cameras, or <code>make resolve</code> has not run since it was seen.
                </div>
              </section>
            )}

            <section className="case-sec">
              <div className="eyebrow">Camera evidence · {data.sightings.length} observations</div>
              <div className="case-obs">
                {data.sightings.map((s) => (
                  <article className="obs" key={s.id}>
                    <Snapshot s={s} />
                    <div className="obs-body">
                      <div className="obs-head">
                        <button type="button" className="tag link"
                                onClick={() => onFocusCamera?.(s.camera_id)}>
                          {s.camera_id}
                        </button>
                        <span className="tag">#{s.track_id}</span>
                        <span className={`plate-status st-${s.plate_status}`}>
                          {s.plate_status}
                        </span>
                      </div>
                      <div className="obs-time">
                        {fmtTime(s.first_seen)} → {fmtTime(s.last_seen)}
                      </div>
                      <dl className="obs-stats">
                        <div><dt>Plate</dt><dd>{s.plate || '—'}</dd></div>
                        <div><dt>Plate conf</dt>
                          <dd>{s.plate_confidence != null
                            ? `${(s.plate_confidence * 100).toFixed(0)}%` : '—'}</dd></div>
                        <div><dt>Detection</dt>
                          <dd>{(s.detection_confidence * 100).toFixed(0)}%</dd></div>
                        <div><dt>Class</dt><dd>{s.vehicle_class}</dd></div>
                        <div><dt>Crossing</dt><dd>{s.crossing || '—'}</dd></div>
                        <div><dt>Frames</dt><dd>{s.frames}</dd></div>
                      </dl>
                      {s.plate_reads.length > 0 && (
                        <details className="obs-reads">
                          <summary>{s.plate_reads.length} strongest OCR reads</summary>
                          <ul>
                            {s.plate_reads.map((r, i) => (
                              <li key={i}>
                                <code>{r.text}</code>
                                <span>{r.confidence.toFixed(2)}</span>
                                <small>{r.plate_width_px}px</small>
                              </li>
                            ))}
                          </ul>
                        </details>
                      )}
                    </div>
                  </article>
                ))}
              </div>
            </section>

            <section className="case-sec">
              <div className="eyebrow">
                Matching evidence · {data.links.length} judged pairs
              </div>
              {data.links.length === 0 ? (
                <div className="empty small">
                  No cross-camera pair was judged for this vehicle.
                </div>
              ) : (
                <div className="clinks">
                  {data.links.map((l) => <LinkRow l={l} key={l.id} />)}
                </div>
              )}
              <p className="case-note">
                Refused pairs are kept deliberately. What the system declined to
                assert, and why, is the evidence that the matching is reasoned.
              </p>
            </section>

            <section className="case-sec">
              <div className="eyebrow">Description</div>
              <p className="case-desc">
                {data.vehicle_class}
                {data.colour_hex && <> with dominant colour <span className="swatch"
                  style={{ background: data.colour_hex }} /> {data.colour_hex}</>},
                {' '}observed at {data.cameras.join(', ')} between {fmtTime(data.first_seen)} and
                {' '}{fmtTime(data.last_seen)}. Plate read as <code>{data.plate}</code> at
                {' '}{(data.plate_confidence * 100).toFixed(0)}% consensus confidence.
              </p>
              <p className="case-note">
                Derived only from stored measurements. NETRA holds no information about
                any person, owner or driver, and infers none.
              </p>
            </section>
          </div>
        )}
      </aside>
    </div>
  )
}
