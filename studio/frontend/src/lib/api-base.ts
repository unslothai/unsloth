// Central API base URL for Tauri vs browser mode
export type ApiScheme = "http" | "https";

let apiBase = "";
let apiScheme: ApiScheme = "http";
let apiPort: number | null = null;

const isTauri = typeof window !== "undefined" && "__TAURI__" in window;
const isViteDev = import.meta.env.DEV;

if (isTauri && !isViteDev) {
  apiBase = "http://127.0.0.1:8888";
  apiPort = 8888;
}

const initialApiBase = apiBase;
const initialApiScheme: ApiScheme = apiScheme;
const initialApiPort = apiPort;

export function resetApiBase() {
  apiBase = initialApiBase;
  apiScheme = initialApiScheme;
  apiPort = initialApiPort;
}

export function setApiBase(port: number, scheme: ApiScheme = apiScheme) {
  apiScheme = scheme;
  apiPort = port;
  // Only rewrite apiBase in Tauri mode. In browser mode `apiBase` stays
  // empty so requests resolve relative to the page origin — repointing
  // it at 127.0.0.1 would break self-signed cert exceptions accepted
  // for `localhost` (or any other hostname the user is browsing from).
  if (isTauri) {
    apiBase = `${scheme}://127.0.0.1:${port}`;
  }
}

export function getApiBase(): string {
  return apiBase;
}

export function getApiScheme(): ApiScheme {
  return apiScheme;
}

export function getApiPort(): number | null {
  return apiPort;
}

export function apiUrl(path: string): string {
  if (path.startsWith("http")) {
    return path;
  }
  return `${apiBase}${path}`;
}

export { isTauri };
