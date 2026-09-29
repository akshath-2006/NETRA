import { SEVERITY } from '../theme.js'

const ICON = { critical: '!', warning: '▲', info: 'i' }

export default function AlertsPanel({ alerts }) {
  if (!alerts.length) {
    return (
      <div className="empty small">
        No alerts. Nothing has crossed a threshold — which is a result, not a gap.
      </div>
    )
  }
  return (
    <div className="alerts">
      {alerts.map((a, i) => (
        <div className="alert" key={`${a.kind}-${a.subject}-${i}`}>
          {/* Icon + label, never colour alone. */}
          <span className="alert-icon" style={{ color: SEVERITY[a.severity] }}>
            {ICON[a.severity] || 'i'}
          </span>
          <div className="alert-body">
            <div className="alert-head">
              <span className="alert-kind" style={{ color: SEVERITY[a.severity] }}>
                {a.severity}
              </span>
              <span className="alert-subject">{a.subject}</span>
              {a.needs_review && <span className="alert-review">needs review</span>}
            </div>
            <div className="alert-msg">{a.message}</div>
            <div className="alert-detail">{a.detail}</div>
          </div>
        </div>
      ))}
    </div>
  )
}
