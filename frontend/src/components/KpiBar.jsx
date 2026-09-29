export default function KpiBar({ stats, alerts = [], journeys = 0 }) {
  const s = stats || {}
  const dirs = Object.entries(s.by_direction || {})
  const critical = alerts.filter((a) => a.severity === 'critical').length

  return (
    <dl className="kpis">
      {/* Vehicles seen leads the row — it is the figure that says the system
          is doing something right now. Everything else supports it. */}
      <div className="kpi lead">
        <dt>Vehicles seen</dt>
        <dd>{s.sightings ?? 0}<small>across the network</small></dd>
      </div>
      <div className="kpi">
        <dt>Cameras live</dt>
        <dd>{s.cameras_live ?? 0}<small>of {s.cameras ?? 0} configured</small></dd>
      </div>
      <div className="kpi">
        <dt>Plates read</dt>
        <dd>{s.plates ?? 0}<small>{s.plate_reads ?? 0} OCR attempts</small></dd>
      </div>
      <div className="kpi">
        <dt>Journeys traced</dt>
        <dd>{journeys}<small>across two or more cameras</small></dd>
      </div>
      <div className={`kpi${critical > 0 ? ' alertful' : ''}`}>
        <dt>Alerts</dt>
        <dd>{alerts.length}<small>{critical > 0 ? `${critical} critical` : 'none critical'}</small></dd>
      </div>
      <div className="kpi">
        <dt>Line crossings</dt>
        <dd className="kpi-chips">
          {dirs.length === 0 && <span className="chip none">none yet</span>}
          {dirs.map(([name, n]) => (
            <span className="chip" key={name}>{name}<b>{n}</b></span>
          ))}
        </dd>
      </div>
    </dl>
  )
}
