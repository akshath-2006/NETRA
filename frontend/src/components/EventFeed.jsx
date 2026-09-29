function confClass(v) {
  if (v == null) return ''
  return v >= 0.75 ? 'hi' : v >= 0.5 ? 'mid' : 'lo'
}

function time(iso) {
  return iso ? iso.slice(11, 23) : ''
}

export default function EventFeed({ events }) {
  if (!events.length) {
    return (
      <div className="empty">
        No detections yet.
        <span>Start the camera workers with <code>make run</code>. Models take
        15&ndash;20 seconds to load before the first frame arrives.</span>
      </div>
    )
  }
  return (
    <div className="feed-list">
      {events.map((e) => {
        // The backend decides this, not the UI. A tentative plate must never
        // look like a confirmed one -- that is the whole point of carrying the
        // status through instead of just printing the string.
        const status = e.plate_status || (e.plate ? 'tentative' : 'unreadable')
        return (
          <div className="ev" key={e.id}>
            <span className={`ev-plate${e.plate ? '' : ' none'} st-${status}`}>
              {e.plate || 'no readable plate'}
            </span>
            <span className="ev-time">{time(e.first_seen)}</span>
            <span className="ev-meta">
              <span className={`plate-status st-${status}`}>
                {status === 'confirmed' ? 'confirmed'
                  : status === 'tentative' ? 'tentative' : 'unreadable'}
              </span>{' '}
              <span className="tag">{e.camera_id}</span>{' '}
              <span className="tag">#{e.track_id}</span>{' '}
              {e.vehicle_class}
              {e.crossing ? ` · ${e.crossing}` : ''}
            </span>
            <span className={`ev-conf ${confClass(e.plate_confidence)}`}>
              {e.plate_confidence != null ? e.plate_confidence.toFixed(2) : '—'}
            </span>
          </div>
        )
      })}
    </div>
  )
}
