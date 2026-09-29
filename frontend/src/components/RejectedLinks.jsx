export default function RejectedLinks({ links }) {
  return (
    <section className="card">
      <div className="card-head">
        <h2>Links the system refused</h2>
        <span className="card-sub">
          plates that matched, journeys that could not have happened
        </span>
      </div>
      {links.length === 0 ? (
        <div className="empty small">
          No near-misses recorded. Nothing had similar plates and impossible timing.
        </div>
      ) : (
        <ul className="rejects">
          {links.map((l, i) => (
            <li key={i}>
              <div className="reject-head">
                <b>{l.from_camera} → {l.to_camera}</b>
                <span className="reject-sim">
                  plate similarity {l.plate_similarity.toFixed(2)}
                </span>
              </div>
              <div className="reject-reason">{l.reason}</div>
            </li>
          ))}
        </ul>
      )}
      <p className="card-note">
        Showing what was <em>not</em> asserted is the clearest evidence that
        association is reasoned rather than pattern-matched.
      </p>
    </section>
  )
}
