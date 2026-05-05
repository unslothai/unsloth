// Central API base URL for Tauri vs browser mode
let apiBase = ''

const isTauri = typeof window !== 'undefined' && '__TAURI__' in window
const isViteDev = import.meta.env.DEV

if (isTauri && !isViteDev) {
  apiBase = 'http://127.0.0.1:8888'
}

const initialApiBase = apiBase

export function resetApiBase() {
  apiBase = initialApiBase
}

export function setApiBase(port: number) {
  apiBase = `http://127.0.0.1:${port}`
}

export function getApiBase(): string {
  return apiBase
}

export function apiUrl(path: string): string {
  if (path.startsWith('http')) return path
  return `${apiBase}${path}`
}

export { isTauri }
