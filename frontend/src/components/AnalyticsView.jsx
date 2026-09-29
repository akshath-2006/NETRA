import { Area, AreaChart, CartesianGrid, Legend, ResponsiveContainer,
         Tooltip, XAxis, YAxis } from 'recharts'
import { CONGESTION, INK, SURFACE, VEHICLE_ORDER, vehicleColour } from '../theme.js'

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  const total = payload.reduce((s, p) => s + (p.value || 0), 0)
  return (
    <div className="tip">
      <div className="tip-head">{label}</div>
      {payload.filter((p) => p.value > 0).map((p) => (
        <div className="tip-row" key={p.dataKey}>
          <i style={{ background: p.color }} />
          <span>{p.dataKey}</span><b>{p.value}</b>
        </div>
      ))}
      <div className="tip-total">total <b>{total}</b></div>
    </div>
  )
}

export default function AnalyticsView({ analytics }) {
  if (!analytics || !analytics.totals?.sightings) {
    return (
      <div className="empty">
        Nothing to analyse yet. Process some cameras, then run
        {' '}<code>make resolve</code> and <code>make analytics</code>.
      </div>
    )
  }

  const classes = VEHICLE_ORDER.filter((c) => analytics.vehicle_mix?.[c])
  const volume = (analytics.volume || []).map((b) => ({
    time: b.start.slice(11, 16),
    ...Object.fromEntries(classes.map((c) => [c, b.by_class?.[c] || 0])),
  }))

  const corridors = analytics.corridors || []
  const cameras = analytics.cameras || []
  const maxSeen = Math.max(1, ...cameras.map((c) => c.sightings))
  const maxIndex = Math.max(2, ...corridors.map((c) => c.congestion_index))

  return (
    <div className="analytics">
      <section className="card">
        <div className="card-head">
          <h2>Traffic volume</h2>
          <span className="card-sub">
            vehicles per {analytics.window.bucket_minutes} minutes
          </span>
        </div>
        <div className="chart">
          <ResponsiveContainer width="100%" height={260}>
            <AreaChart data={volume} margin={{ top: 8, right: 16, bottom: 4, left: -12 }}>
              <CartesianGrid stroke={INK.grid} vertical={false} />
              {/* Visual only: thins the tick labels so they stop colliding when the
                  window covers many buckets. Same data, fewer labels drawn. */}
              <XAxis dataKey="time" tick={{ fill: INK.muted, fontSize: 11 }}
                     stroke={INK.axis} tickLine={false}
                     minTickGap={44} interval="preserveStartEnd" tickMargin={8} />
              <YAxis allowDecimals={false} tick={{ fill: INK.muted, fontSize: 11 }}
                     stroke={INK.axis} tickLine={false} axisLine={false} />
              <Tooltip content={<ChartTooltip />} cursor={{ stroke: INK.axis }} />
              {classes.length > 1 && (
                <Legend iconType="square"
                        wrapperStyle={{ fontSize: 12, color: INK.secondary }} />
              )}
              {classes.map((c) => (
                // stroke = surface gives the 2px gap between stacked fills
                <Area key={c} type="monotone" dataKey={c} stackId="v"
                      stroke={SURFACE} strokeWidth={2}
                      fill={vehicleColour(c)} fillOpacity={0.92} />
              ))}
            </AreaChart>
          </ResponsiveContainer>
        </div>
        {analytics.peak && (
          <p className="card-note">
            Peak period <b>{analytics.peak.start.slice(11, 16)}</b> with{' '}
            <b>{analytics.peak.count}</b> vehicles.
          </p>
        )}
      </section>

      <section className="card">
        <div className="card-head">
          <h2>Corridor congestion</h2>
          <span className="card-sub">
            median travel time against free flow — measured, not counted
          </span>
        </div>
        <table className="dt">
          <thead>
            <tr><th>Corridor</th><th className="num">Trips</th><th className="num">km</th>
              <th className="num">Median</th><th className="num">Free flow</th>
              <th>Congestion</th></tr>
          </thead>
          <tbody>
            {corridors.map((c) => (
              <tr key={c.label}>
                <td className="k">{c.label}</td>
                <td className="num">{c.trips}</td>
                <td className="num">{c.distance_km.toFixed(1)}</td>
                <td className="num">{(c.median_travel_s / 60).toFixed(1)}m</td>
                <td className="num">
                  {(c.free_flow_travel_s / 60).toFixed(1)}m
                  {!c.baseline_from_data && <small title="too few trips for a measured baseline"> cfg</small>}
                </td>
                <td>
                  <div className="meter-row">
                    <div className="meter">
                      <span className="meter-fill"
                            style={{ width: `${Math.min(100, (c.congestion_index / maxIndex) * 100)}%`,
                                     background: CONGESTION[c.level] }} />
                    </div>
                    <span className="meter-label">
                      {c.congestion_index.toFixed(2)}× · {c.level}
                    </span>
                  </div>
                </td>
              </tr>
            ))}
            {corridors.length === 0 && (
              <tr><td colSpan={6} className="muted">
                No corridor has been travelled yet.</td></tr>
            )}
          </tbody>
        </table>
      </section>

      <section className="card">
        <div className="card-head">
          <h2>Camera utilisation</h2>
          <span className="card-sub">
            plate rate is a camera-quality signal, not a traffic one
          </span>
        </div>
        <table className="dt">
          <thead>
            <tr><th>Camera</th><th>Share of sightings</th><th className="num">Seen</th>
              <th className="num">Plates</th><th className="num">Rate</th></tr>
          </thead>
          <tbody>
            {cameras.map((c) => (
              <tr key={c.camera_id}>
                <td className="k">{c.camera_id}</td>
                <td className="meter-cell">
                  <div className="meter">
                    <span className="meter-fill"
                          style={{ width: `${(c.sightings / maxSeen) * 100}%`,
                                   background: '#3987e5' }} />
                  </div>
                </td>
                <td className="num">{c.sightings}</td>
                <td className="num">{c.plates_read}</td>
                <td className="num">{Math.round(c.plate_rate * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="card">
        <div className="card-head">
          <h2>Origin → destination</h2>
          <span className="card-sub">where reconstructed journeys started and ended</span>
        </div>
        <table className="dt">
          <thead><tr><th>From</th><th>To</th><th className="num">Trips</th></tr></thead>
          <tbody>
            {Object.entries(analytics.od_matrix || {}).flatMap(([from, dests]) =>
              Object.entries(dests).map(([to, n]) => (
                <tr key={`${from}-${to}`}>
                  <td className="k">{from}</td><td className="k">{to}</td>
                  <td className="num">{n}</td>
                </tr>
              )))}
            {Object.keys(analytics.od_matrix || {}).length === 0 && (
              <tr><td colSpan={3} className="muted">
                No multi-camera journeys yet.</td></tr>
            )}
          </tbody>
        </table>
      </section>
    </div>
  )
}
