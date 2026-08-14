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

if (isTauri && !tauriDevProxy) {
  // never connects; real port arrives via server-port
  apiBase = 'http://127.0.0.1:0'
  apiBaseReadyPromise = new Promise((resolve) => {
    resolveApiBaseReady = resolve
  })
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
  resolveApiBaseReady?.()
  resolveApiBaseReady = null
}

export function apiBaseReady(): Promise<void> {
  return apiBaseReadyPromise
}

export function getApiBase(): string {
  return apiBase
}

export function apiUrl(path: string): string {
  if (path.startsWith('http')) return path
  return `${apiBase}${path}`
}

export { isTauri }
