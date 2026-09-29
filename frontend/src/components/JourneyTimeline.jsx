import { CONGESTION, fmtDuration, fmtTime } from '../theme.js'

export default function JourneyTimeline({ journey }) {
  if (!journey) return null
  const hops = journey.hops || []

  return (
    <div className="journey">
      <div className="journey-head">
        <span className="journey-plate">{journey.plate || journey.identity_key}</span>
        <span className="journey-conf">
          confidence {journey.confidence.toFixed(2)}
          <small> (weakest hop)</small>
        </span>
      </div>

      <ol className="hops">
        {hops.map((h) => (
          <li className="hop" key={h.sequence}>
            <div className="hop-rail"><span className="hop-dot" /></div>
            <div className="hop-body">
              <div className="hop-title">
                <b>{h.from}</b> → <b>{h.to}</b>
                {h.via.length > 0 && (
                  <span className="hop-via">via {h.via.join(', ')} — never saw it</span>
                )}
              </div>
              <div className="hop-times">
                {fmtTime(h.departed_at)} → {fmtTime(h.arrived_at)}
              </div>
              <div className="hop-stats">
                <span>{fmtDuration(h.gap_s)}</span>
                <span>{h.distance_km.toFixed(1)} km</span>
                <span style={{ color: h.congested ? CONGESTION.heavy : undefined }}>
                  {Math.round(h.implied_speed_kmph)} km/h
                  <small> usual {Math.round(h.typical_speed_kmph)}</small>
                </span>
                <span className="hop-score">score {h.confidence.toFixed(2)}</span>
              </div>
            </div>
          </li>
        ))}
      </ol>

      <dl className="journey-totals">
        <div><dt>Distance</dt><dd>{journey.total_distance_km.toFixed(1)} km</dd></div>
        <div><dt>Duration</dt><dd>{fmtDuration(journey.total_duration_s)}</dd></div>
        <div><dt>Average</dt><dd>{Math.round(journey.average_speed_kmph)} km/h</dd></div>
        <div><dt>Cameras</dt><dd>{journey.route.length}</dd></div>
      </dl>

      {journey.coverage_gaps.length > 0 && (
        <p className="journey-gap">
          <b>Coverage gap.</b> {journey.coverage_gaps.join(', ')} sits on this route
          but recorded nothing — a camera problem, not a traffic one.
        </p>
      )}
    </div>
  )
}
