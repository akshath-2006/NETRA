// One place that knows how to reach the backend.

export async function getJSON(path) {
  const res = await fetch(path)
  if (!res.ok) throw new Error(`${path} -> ${res.status}`)
  return res.json()
}

export const api = {
  health: () => getJSON('/api/health'),
  cameras: () => getJSON('/api/cameras'),
  stats: () => getJSON('/api/stats'),
  sightings: (limit = 50) => getJSON(`/api/sightings?limit=${limit}`),
  plate: (p) => getJSON(`/api/plates/${encodeURIComponent(p)}`),
  journeys: (limit = 50) => getJSON(`/api/journeys?limit=${limit}`),
  journeyByPlate: (p) => getJSON(`/api/journeys/${encodeURIComponent(p)}`),
  analytics: () => getJSON('/api/analytics'),
  alerts: (limit = 30) => getJSON(`/api/alerts?limit=${limit}`),
  rejectedLinks: () => getJSON('/api/rejected-links'),

  // Investigation surfaces. The case view is fetched only when a case is
  // opened, so the Trace page never pays for evidence nobody asked to see.
  vehicleCase: (p) => getJSON(`/api/case/${encodeURIComponent(p)}`),
  leads: (q = {}) => {
    const params = new URLSearchParams()
    Object.entries(q).forEach(([k, v]) => {
      if (v !== '' && v !== null && v !== undefined) params.set(k, v)
    })
    return getJSON(`/api/leads?${params.toString()}`)
  },
  setLeadStatus: async (id, status, note = '') => {
    const res = await fetch(`/api/leads/${id}/status`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status, note }),
    })
    if (!res.ok) throw new Error(`lead status -> ${res.status}`)
    return res.json()
  },
}

// Reconnecting WebSocket. A demo must survive the backend restarting.
export function connectEvents(onMessage, onState) {
  let ws = null
  let closed = false
  let retry = 1000

  const open = () => {
    if (closed) return
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    ws = new WebSocket(`${proto}//${location.host}/ws/events`)
    ws.onopen = () => { retry = 1000; onState?.('live') }
    ws.onmessage = (e) => { try { onMessage(JSON.parse(e.data)) } catch {} }
    ws.onclose = () => {
      onState?.('offline')
      if (!closed) { setTimeout(open, retry); retry = Math.min(retry * 2, 10000) }
    }
    ws.onerror = () => ws?.close()
  }
  open()
  return () => { closed = true; ws?.close() }
}
