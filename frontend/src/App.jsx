import { useCallback, useEffect, useState } from 'react'
import { api, connectEvents } from './api.js'
import AlertsPanel from './components/AlertsPanel.jsx'
import AnalyticsView from './components/AnalyticsView.jsx'
import CameraGrid from './components/CameraGrid.jsx'
import CityMap from './components/CityMap.jsx'
import EventFeed from './components/EventFeed.jsx'
import KpiBar from './components/KpiBar.jsx'
import PlateSearch from './components/PlateSearch.jsx'
import RejectedLinks from './components/RejectedLinks.jsx'
import LeadsView from './components/LeadsView.jsx'
import VehicleCase from './components/VehicleCase.jsx'

const MAX_EVENTS = 120
const VIEWS = [['live', 'Live'], ['trace', 'Trace'], ['analytics', 'Analytics'],
               ['leads', 'Leads']]

export default function App() {
  const [view, setView] = useState('live')
  const [cameras, setCameras] = useState([])
  const [stats, setStats] = useState(null)
  const [events, setEvents] = useState([])
  const [alerts, setAlerts] = useState([])
  const [journeys, setJourneys] = useState([])
  const [analytics, setAnalytics] = useState(null)
  const [rejects, setRejects] = useState([])
  const [selected, setSelected] = useState(null)
  // Bumped on EVERY selection, including re-selecting the vehicle already
  // shown. CityMap restarts playback from this rather than from the route
  // geometry, which is what makes replay work for two vehicles that happen to
  // travel the same corridor.
  const [playKey, setPlayKey] = useState(0)
  const [caseFor, setCaseFor] = useState(null)
  const [link, setLink] = useState('connecting')

  const selectJourney = useCallback((j) => {
    setSelected(j)
    setPlayKey((k) => k + 1)
  }, [])

  const safe = useCallback(async (fn, set) => {
    try { set(await fn()) } catch { /* backend restarting; next tick recovers */ }
  }, [])

  // Fast loop: what changes every few seconds.
  useEffect(() => {
    let stop = false
    const tick = async () => {
      if (stop) return
      await Promise.all([
        safe(api.cameras, setCameras),
        safe(api.stats, setStats),
      ])
    }
    tick()
    const id = setInterval(tick, 2000)
    return () => { stop = true; clearInterval(id) }
  }, [safe])

  // Slow loop: derived data, recomputed by `resolve` and `analytics`.
  useEffect(() => {
    let stop = false
    const tick = async () => {
      if (stop) return
      await Promise.all([
        safe(api.alerts, setAlerts),
        safe(api.journeys, setJourneys),
        safe(api.analytics, setAnalytics),
        safe(api.rejectedLinks, setRejects),
      ])
    }
    tick()
    const id = setInterval(tick, 8000)
    return () => { stop = true; clearInterval(id) }
  }, [safe])

  useEffect(() => connectEvents((msg) => {
    if (msg.type === 'ping' || !msg.events?.length) return
    setEvents((prev) => [...[...msg.events].reverse(), ...prev].slice(0, MAX_EVENTS))
  }, setLink), [])

  const byCamera = stats?.by_camera || {}
  const corridors = analytics?.corridors || []

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">NET<span>RA</span></span>
          <small>City Traffic Intelligence</small>
        </div>
        <nav className="tabs">
          {VIEWS.map(([id, label]) => (
            <button key={id} type="button"
                    className={view === id ? 'on' : ''}
                    onClick={() => setView(id)}>{label}</button>
          ))}
        </nav>
        <div className="status">
          <span className={`dot ${link}`} />
          {link === 'live' ? 'event feed live' : 'event feed offline'}
        </div>
      </header>

      <KpiBar stats={stats} alerts={alerts} journeys={journeys.length} />

      {view === 'live' && (
        <div className="main">
          <section className="stack">
            <div className="panel-title">
              City network <b>{cameras.filter((c) => c.live).length} streaming</b>
              <span className="count">{cameras.length}</span>
            </div>
            <CityMap cameras={cameras} corridors={corridors} byCamera={byCamera} />
            <div className="panel-title">Camera feeds</div>
            <CameraGrid cameras={cameras} byCamera={byCamera} />
          </section>

          <aside className="side">
            <div className="panel-title">
              Alerts<span className="count">{alerts.length}</span>
            </div>
            <AlertsPanel alerts={alerts} />
            <div className="panel-title">
              Live detections<span className="count">{events.length}</span>
            </div>
            <EventFeed events={events} />
          </aside>
        </div>
      )}

      {view === 'trace' && (
        <div className="main">
          <section className="stack trace-stack">
            <div className="panel-title">
              Journey on the network
              {selected && <b>{selected.plate || selected.identity_key}</b>}
            </div>
            <CityMap cameras={cameras} corridors={corridors}
                     journey={selected} byCamera={byCamera} playKey={playKey} />
          </section>
          <aside className="side">
            <div className="panel-title">Vehicle search</div>
            <PlateSearch journeys={journeys} selected={selected}
                         onSelect={selectJourney} onOpenCase={setCaseFor} />
          </aside>
        </div>
      )}

      {view === 'leads' && <LeadsView />}

      {view === 'analytics' && (
        <div className="scroll">
          <AnalyticsView analytics={analytics} />
          <RejectedLinks links={rejects} />
        </div>
      )}

      {caseFor && (
        <VehicleCase plate={caseFor} onClose={() => setCaseFor(null)} />
      )}
    </div>
  )
}
