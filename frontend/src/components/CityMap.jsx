import { useEffect, useMemo, useRef, useState } from 'react'
import { CircleMarker, MapContainer, Polyline, TileLayer, Tooltip } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'
import { CONGESTION, INK } from '../theme.js'

// Tiles come through our own API, which caches them on disk. Online it fills
// the cache as you pan; after `make cache-tiles` it needs no network at all;
// and a miss with no network returns a plain dark tile, so the map degrades
// instead of breaking. Underneath they are CARTO dark basemaps built from
// OpenStreetMap data -- free, no API key, no account.
const TILES = '/api/tiles/{z}/{x}/{y}.png'
const ATTRIB = '&copy; OpenStreetMap contributors &copy; CARTO'

const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

/**
 * Route playback, restarted by an explicit token rather than by geometry.
 *
 * The previous version keyed this effect on the route's coordinates. That
 * looked reasonable and was wrong in two ways that both showed up in practice:
 *
 *  1. Two different vehicles travelling the SAME route produce identical
 *     points, so React saw no dependency change, the effect never re-ran, and
 *     the marker stayed frozen wherever the previous vehicle left it.
 *  2. Re-selecting the vehicle already shown could never replay, for the same
 *     reason.
 *
 * `playKey` increments on every selection -- including re-selecting the same
 * vehicle -- so "play this route from the start" is stated explicitly instead
 * of being inferred from whether the numbers happened to differ.
 */
function useJourneyPlayback(points, active, playKey) {
  const [progress, setProgress] = useState(0)
  const raf = useRef(null)
  const run = useRef(0)

  useEffect(() => {
    // Whatever was playing belongs to a previous selection. Stop it and reset
    // to the start before anything else, so there is never a frame where the
    // new vehicle is drawn at the old one's position.
    if (raf.current) cancelAnimationFrame(raf.current)
    const mine = ++run.current
    setProgress(0)

    if (!active || points.length < 2) return
    if (prefersReducedMotion()) { setProgress(1); return }

    const started = performance.now()
    const duration = 2400 + points.length * 500
    const step = (now) => {
      // A frame scheduled by a superseded run must not write progress.
      if (mine !== run.current) return
      const t = Math.min(1, (now - started) / duration)
      setProgress(t)
      if (t < 1) raf.current = requestAnimationFrame(step)
    }
    raf.current = requestAnimationFrame(step)

    return () => {
      if (raf.current) cancelAnimationFrame(raf.current)
      raf.current = null
    }
  }, [playKey, active, points.length])

  return progress
}

/** Position along a polyline at 0..1 of its total length. */
function pointAt(points, t) {
  if (points.length < 2) return points[0]
  const legs = []
  let total = 0
  for (let i = 1; i < points.length; i++) {
    const d = Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1])
    legs.push(d); total += d
  }
  if (total === 0) return points[0]
  let travelled = t * total
  for (let i = 0; i < legs.length; i++) {
    if (travelled <= legs[i]) {
      const f = legs[i] === 0 ? 0 : travelled / legs[i]
      return [points[i][0] + (points[i + 1][0] - points[i][0]) * f,
              points[i][1] + (points[i + 1][1] - points[i][1]) * f]
    }
    travelled -= legs[i]
  }
  return points[points.length - 1]
}

export default function CityMap({ cameras, corridors = [], journey = null,
                                 byCamera = {}, playKey = 0 }) {
  const coords = useMemo(() => {
    const m = {}
    cameras.forEach((c) => { m[c.id] = [c.lat, c.lon] })
    return m
  }, [cameras])

  const centre = cameras.length
    ? [cameras.reduce((s, c) => s + c.lat, 0) / cameras.length,
       cameras.reduce((s, c) => s + c.lon, 0) / cameras.length]
    : [12.9716, 77.5946]

  const routePoints = useMemo(() => {
    if (!journey) return []
    return (journey.full_route || []).map((id) => coords[id]).filter(Boolean)
  }, [journey, coords])

  const progress = useJourneyPlayback(routePoints, Boolean(journey), playKey)
  const vehicle = routePoints.length >= 2 ? pointAt(routePoints, progress) : null
  const maxSeen = Math.max(1, ...Object.values(byCamera))

  return (
    <div className="map-wrap">
      <MapContainer center={centre} zoom={13} className="map" scrollWheelZoom
                    attributionControl={false}>
        <TileLayer url={TILES} attribution={ATTRIB} maxZoom={19} />

        {/* Corridors, coloured by measured congestion. Status colour always
            travels with a label in the tooltip -- never hue alone. */}
        {corridors.map((c) => {
          const a = coords[c.from]; const b = coords[c.to]
          if (!a || !b) return null
          return (
            <Polyline key={c.label} positions={[a, b]}
                      pathOptions={{ color: CONGESTION[c.level] || INK.muted,
                                     weight: 4, opacity: 0.75 }}>
              <Tooltip sticky>
                <b>{c.label}</b><br />
                {c.level} &middot; index {c.congestion_index.toFixed(2)}<br />
                median {(c.median_travel_s / 60).toFixed(1)} min over {c.distance_km} km<br />
                {c.trips} trip{c.trips === 1 ? '' : 's'}
              </Tooltip>
            </Polyline>
          )
        })}

        {/* The reconstructed journey, drawn over the network. */}
        {routePoints.length >= 2 && (
          <Polyline positions={routePoints}
                    pathOptions={{ color: '#EDAF46', weight: 3, opacity: 0.95,
                                   dashArray: '1 8', lineCap: 'round' }} />
        )}
        {vehicle && (
          <CircleMarker center={vehicle} radius={7}
                        pathOptions={{ color: '#0C1218', weight: 2,
                                       fillColor: '#EDAF46', fillOpacity: 1 }} />
        )}

        {cameras.map((c) => {
          const seen = byCamera[c.id] || 0
          const onRoute = journey?.full_route?.includes(c.id)
          const unobserved = journey?.coverage_gaps?.includes(c.id)
          return (
            <CircleMarker key={c.id} center={[c.lat, c.lon]}
                          radius={7 + 7 * Math.sqrt(seen / maxSeen)}
                          pathOptions={{
                            color: unobserved ? CONGESTION.severe
                              : onRoute ? '#EDAF46' : (c.live ? '#5CC08C' : INK.muted),
                            weight: 2,
                            fillColor: unobserved ? CONGESTION.severe
                              : (c.live ? '#5CC08C' : INK.muted),
                            fillOpacity: 0.35,
                          }}>
              <Tooltip direction="top" offset={[0, -6]}>
                <b>{c.id}</b> {c.name}<br />
                {seen} vehicle{seen === 1 ? '' : 's'} seen &middot;{' '}
                {c.live ? 'streaming' : 'idle'}
                {unobserved && <><br /><b>on this route but saw nothing</b></>}
              </Tooltip>
            </CircleMarker>
          )
        })}
      </MapContainer>

      <div className="map-legend">
        <span className="lg-title">Corridor load</span>
        {Object.entries(CONGESTION).filter(([k]) => k !== 'unknown').map(([name, colour]) => (
          <span className="lg-item" key={name}>
            <i style={{ background: colour }} />{name}
          </span>
        ))}
      </div>
    </div>
  )
}
