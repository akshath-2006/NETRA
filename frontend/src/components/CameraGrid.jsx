export default function CameraGrid({ cameras, byCamera }) {
  if (!cameras.length) {
    return (
      <div className="empty">
        No cameras configured yet.
        <span>Add one to <code>configs/cameras.yaml</code> — no code change needed.</span>
      </div>
    )
  }
  return (
    <div className="cameras">
      {cameras.map((c) => (
        <div className="cam" key={c.id}>
          <div className="cam-head">
            <span className="cam-id">{c.id}</span>
            <span className="cam-name">{c.name}</span>
            <span className={`badge${c.live ? ' live' : ''}`}>
              {c.live && <i className="pulse" />}
              {c.live ? 'live' : 'idle'}
            </span>
          </div>
          {/* MJPEG: the browser holds the connection open and the worker's
              latest annotated frame streams straight in. */}
          <img src={c.stream_url} alt={`${c.id} live view`} />
          <div className="cam-foot">
            <span>seen <b>{byCamera?.[c.id] ?? 0}</b></span>
            <span>{c.lat.toFixed(4)}, {c.lon.toFixed(4)}</span>
            <span>hdg <b>{Math.round(c.heading_deg)}&deg;</b></span>
          </div>
        </div>
      ))}
    </div>
  )
}
