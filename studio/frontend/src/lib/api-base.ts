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

// In Tauri dev the page is served by the Vite dev server, whose /api and /v1
// proxies reach the backend over Node's HTTP client. Use same-origin relative
// URLs there: direct WKWebView->backend requests can wedge indefinitely (the
// webview holds the sockets open but never dispatches the requests).
const tauriDevProxy = isTauri && import.meta.env?.DEV

// In Tauri the real port arrives asynchronously (preflight/server-port). A
// fetch against the ':0' placeholder never settles in WKWebView (no error,
// no timeout), so callers must be able to await the real base instead.
let resolveApiBaseReady: (() => void) | null = null
let apiBaseReadyPromise = Promise.resolve()

// A backend that never starts would otherwise leave this pending forever, and
// every caller awaiting it parked behind a spinner with no error to show. Fail
// it instead, wording the reason so classifyFetchError reads it as backend-down.
const API_BASE_READY_TIMEOUT_MS = 60_000

function armApiBaseReady(): void {
  apiBaseReadyPromise = new Promise((resolve, reject) => {
    const timer = setTimeout(() => {
      // Cleared so setApiBase can tell a timed-out wait from a live one: this
      // promise is now permanently rejected and cannot be resolved later.
      resolveApiBaseReady = null
      reject(new Error("The backend isn't running yet."))
    }, API_BASE_READY_TIMEOUT_MS)
    // Browsers ignore this; under node it keeps the timer from holding the
    // process open for a full minute after the tests finish.
    ;(timer as unknown as { unref?: () => void }).unref?.()
    resolveApiBaseReady = () => {
      clearTimeout(timer)
      resolve()
    }
  })
  // An unhandled rejection before the first awaiter would surface as a console
  // error; the awaiters below still see it.
  apiBaseReadyPromise.catch(() => undefined)
}

if (isTauri && !tauriDevProxy) {
  // never connects; real port arrives via server-port
  apiBase = 'http://127.0.0.1:0'
  armApiBaseReady()
}

const initialApiBase = apiBase

export function resetApiBase() {
  apiBase = initialApiBase
}

export function setApiBase(port: number) {
  // The dev proxy targets 8888; only a backend on another port needs the
  // absolute base (and with it the direct-connection risk).
  if (!tauriDevProxy || port !== 8888) {
    apiBase = `http://127.0.0.1:${port}`
  }
  if (resolveApiBaseReady) {
    resolveApiBaseReady()
    resolveApiBaseReady = null
    return
  }
  // The wait already timed out, leaving a permanently rejected promise that
  // every later authFetch would keep failing against. A port arriving now means
  // the backend came up late, so hand callers a settled one instead.
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

export function apiUrl(path: string): string {
  // A URL built before the real port is known would otherwise pass straight
  // through the http check below and hit the placeholder, which is exactly the
  // request apiBaseReady exists to prevent.
  if (path.startsWith(PLACEHOLDER_BASE)) {
    return `${apiBase}${path.slice(PLACEHOLDER_BASE.length)}`
  }
  if (path.startsWith('http')) return path
  return `${apiBase}${path}`
}

export { isTauri }
