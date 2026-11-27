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
// isStudioUrl resolves as fetch does: "/\\evil/api/x" and tab or newline spellings are another host.
export function protectedBackendPath(url: string): string | null {
  return /^(?:\/[^/?#]+)*?\/(?:v1|api)\/[^?#]+(?:\?[^#]*)?$/.test(url) &&
    !/\/file-signed\?/.test(url) &&
    isStudioUrl(url)
    ? url
    : null
}

// The media a reply image may load with the session: gallery files and video frames. Other routes can forward
// their query to a third party (the Hub relay) or act on it, so an image naming one is never fetched.
// The first /api or /v1 segment is the backend's, so a relay path that nests one cannot pass.
export function protectedBackendMediaPath(url: string): string | null {
  const path = protectedBackendPath(url)
  if (path === null) return null
  const at = path.search(/\/(?:api|v1)\//)
  return at >= 0 &&
    /^\/(?:api|v1)\/(?:inference\/)?(?:images|video|audio)\/gallery\/[^/?#]+\/(?:file|frame)(?:\?[^#]*)?$/.test(
      path.slice(at),
    )
    ? path
    : null
}

// Signed gallery media (the URL's token is the credential, no session is sent): a plain <img> may load it.
// Image generation with response_format=url returns these, relative or absolute.
export function isSignedGalleryMediaUrl(url: string): boolean {
  if (!isStudioUrl(url)) return false
  const page = typeof window === 'undefined' ? 'http://localhost/' : window.location.href
  try {
    const parsed = new URL(url, apiBase || page)
    const at = parsed.pathname.search(/\/(?:api|v1)\//)
    return (
      at >= 0 &&
      /^\/(?:api|v1)\/(?:inference\/)?(?:images|video|audio)\/gallery\/[^/]+\/file-signed$/.test(
        parsed.pathname.slice(at),
      ) &&
      parsed.searchParams.has('token')
    )
  } catch {
    return false
  }
}

/** A URL naming one of this backend's /api or /v1 routes, however it is spelled. */
export function isBackendRouteUrl(url: string): boolean {
  if (!isStudioUrl(url)) return false
  const page = typeof window === 'undefined' ? 'http://localhost/' : window.location.href
  try {
    return /\/(?:api|v1)(?:\/|$)/.test(new URL(url, apiBase || page).pathname)
  } catch {
    return false
  }
}

export { isTauri }
