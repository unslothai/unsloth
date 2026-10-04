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

if (isTauri) {
  // never connects; real port arrives via server-port
  apiBase = 'http://127.0.0.1:0'
}

const initialApiBase = apiBase

const LOOPBACK_BASE_PORT = /^https?:\/\/127\.0\.0\.1:(\d+)$/

export function resetApiBase() {
  apiBase = initialApiBase
}

export function setApiBase(port: number) {
  apiBase = `http://127.0.0.1:${port}`
}

export function getApiBase(): string {
  return apiBase
}

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
  if (path.startsWith('http')) return path
  return `${apiBase}${path}`
}

/** Whether `url` is served by this Studio's backend, so a request to it may carry the sign-in. */
export function isStudioUrl(url: string): boolean {
  const base = typeof window === 'undefined' ? 'http://localhost/' : window.location.href
  try {
    return new URL(apiUrl(url), base).origin === new URL(apiUrl('/'), base).origin
  } catch {
    return false
  }
}

export { isTauri }
