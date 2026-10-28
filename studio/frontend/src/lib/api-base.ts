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

/** null when no port is known yet (the placeholder base is port 0). */
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

/** True when url is served by this Studio backend, so a request to it may carry the sign-in. */
export function isStudioUrl(url: string): boolean {
  const page = typeof window === 'undefined' ? 'http://localhost/' : window.location.href
  try {
    // Resolved, not prefixed: "HTTPS://x" and "//x" are other hosts.
    const studio = new URL(apiBase || page)
    return new URL(url, studio).origin === studio.origin
  } catch {
    return false
  }
}

// Media tools link to the backend origin-relative; Desktop's webview origin is not the backend's.
export function backendPathUrl(url: string): string {
  return /^\/(v1|api)\//.test(url) ? apiUrl(url) : url
}

// Signed backend media (the token is the credential): Desktop's img-src blocks the loopback backend, so fetch it.
export function desktopSignedMediaPath(url: string): string | null {
  return apiBase && /^\/(?:api|v1)\/[^?#]+\/file-signed\?token=[^&#]+$/.test(url) ? url : null
}

// A backend link without a signed token: a plain navigation carries no Authorization header, so it 401s.
// Leading segments are a proxy's root_path, which the chat tools keep in the URLs they return.
export function protectedBackendPath(url: string): string | null {
  return /^(?:\/[^/?#]+)*?\/(?:v1|api)\/[^?#]+(?:\?[^#]*)?$/.test(url) && !/\/file-signed\?/.test(url)
    ? url
    : null
}

export { isTauri }
