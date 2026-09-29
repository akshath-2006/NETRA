// Single source of truth for every colour a chart or the map uses.
//
// The categorical slots below are NOT chosen by eye. They come from a
// validated dark-mode palette and were re-checked against this dashboard's own
// surface (#141C24) with the palette validator:
//
//   lightness band PASS · chroma floor PASS · CVD separation PASS (worst
//   adjacent dE 8.4 protan) · normal-vision floor PASS (19.8) · contrast PASS
//
// Vehicle classes use the same hues in the video overlay and in the charts, so
// a bus is the same colour wherever you see it.

export const SURFACE = '#141C24'

export const VEHICLE_COLOURS = {
  car: '#3987e5',        // slot 1  blue
  motorcycle: '#d95926', // slot 2  orange
  bus: '#199e70',        // slot 3  aqua
  truck: '#c98500',      // slot 4  yellow
  bicycle: '#d55181',    // slot 5  magenta
}
export const VEHICLE_ORDER = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']

// Status colours are reserved: never reused as a series colour, and always
// shipped with a label so meaning never rests on hue alone.
export const STATUS = {
  good: '#0ca30c',
  warning: '#fab219',
  serious: '#ec835a',
  critical: '#d03b3b',
}

export const CONGESTION = {
  'free flow': STATUS.good,
  moderate: STATUS.warning,
  heavy: STATUS.serious,
  severe: STATUS.critical,
  unknown: '#74858F',
}

export const SEVERITY = {
  info: '#4EB4CB',
  warning: STATUS.warning,
  critical: STATUS.critical,
}

export const INK = {
  primary: '#E5ECF2',
  secondary: '#A8B7C4',
  muted: '#74858F',
  grid: '#26333F',
  axis: '#3A4B59',
}

export const vehicleColour = (name) => VEHICLE_COLOURS[name] || INK.muted

export function fmtDuration(seconds) {
  const s = Math.max(0, Math.round(seconds || 0))
  const m = Math.floor(s / 60)
  return m > 0 ? `${m}m ${String(s % 60).padStart(2, '0')}s` : `${s}s`
}

export const fmtTime = (iso) => (iso ? iso.slice(11, 19) : '')
