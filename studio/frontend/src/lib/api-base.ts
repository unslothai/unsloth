let apiBase = ''

function detectTauri(): boolean {
  if (typeof window === 'undefined') {
    return false
  }
  return (
    '__TAURI__' in window ||
    '__TAURI_INTERNALS__' in window ||
    window.location.protocol === 'tauri:'
  )
}

const isTauri = detectTauri()

// Tauri dev: use the Vite proxy; direct WKWebView->backend requests can wedge forever.
const tauriDevProxy = isTauri && import.meta.env?.DEV

// A fetch against the ':0' placeholder never settles in WKWebView; callers await the real base.
let resolveApiBaseReady: (() => void) | null = null
let apiBaseReadyPromise = Promise.resolve()

// Time out, worded so classifyFetchError reads it as backend-down.
const API_BASE_READY_TIMEOUT_MS = 60_000

function armApiBaseReady(): void {
  apiBaseReadyPromise = new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      resolveApiBaseReady = null
      reject(new Error("The backend isn't running yet."))
    }, API_BASE_READY_TIMEOUT_MS)
    // Keeps node (tests) from waiting on the timer.
    ;(timer as unknown as { unref?: () => void }).unref?.()
    resolveApiBaseReady = () => {
      clearTimeout(timer)
      resolve()
    }
  })
  apiBaseReadyPromise.catch(() => undefined)
}

if (isTauri && !tauriDevProxy) {
  // never connects; real port arrives via server-port
  apiBase = 'http://127.0.0.1:0'
  armApiBaseReady()
}

const initialApiBase = apiBase

const LOOPBACK_BASE_PORT = /^https?:\/\/127\.0\.0\.1:(\d+)$/

export function resetApiBase() {
  apiBase = initialApiBase
}

export function setApiBase(port: number) {
  if (!tauriDevProxy || port !== 8888) {
    apiBase = `http://127.0.0.1:${port}`
  }
  if (resolveApiBaseReady) {
    resolveApiBaseReady()
    resolveApiBaseReady = null
    return
  }
  // The wait timed out; replace the permanently rejected promise.
  if (isTauri && !tauriDevProxy) {
    apiBaseReadyPromise = Promise.resolve()
  }
}

export function apiBaseReady(): Promise<void> {
  return apiBaseReadyPromise
}

export function getApiBase(): string {
  return apiBase
}

// The placeholder base a caller may have baked in before the port arrived.
const PLACEHOLDER_BASE = 'http://127.0.0.1:0'

/**
 * The port the backend is currently expected on, or null when none is known yet. The
 * placeholder base above is port 0, which reads as "no port yet".
 */
export function getApiPort(): number | null {
  const match = LOOPBACK_BASE_PORT.exec(apiBase)
  if (!match) {
    return null
  }
  const port = Number(match[1])
  return Number.isInteger(port) && port > 0 && port <= 65535 ? port : null
}

export function apiUrl(path: string): string {
  if (path.startsWith(PLACEHOLDER_BASE)) {
    return `${apiBase}${path.slice(PLACEHOLDER_BASE.length)}`
  }
  if (path.startsWith('http')) return path
  return `${apiBase}${path}`
}

export { isTauri }
