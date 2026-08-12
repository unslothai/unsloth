export interface PlatformBackendConfig {
  enabled: boolean;
  baseUrl: string;
  apiPrefix: string;
  proxyTarget: string;
  requestTimeoutMs: number;
}

export type PlatformPathMode = "api" | "root";

interface PlatformBackendEnv {
  VITE_RAG_PLATFORM_BASE_URL?: string;
  VITE_RAG_PLATFORM_API_PREFIX?: string;
  VITE_RAG_PLATFORM_ENABLED?: string;
  VITE_RAG_PLATFORM_PROXY_TARGET?: string;
}

const DEFAULT_API_PREFIX = "/api/v1";
const DEFAULT_REQUEST_TIMEOUT_MS = 15_000;

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function normalizePrefix(value: string | undefined): string {
  const trimmed = value?.trim();
  if (!trimmed) return DEFAULT_API_PREFIX;
  return `/${trimmed.replace(/^\/+|\/+$/g, "")}`;
}

export function getPlatformBackendConfig(
  env: PlatformBackendEnv = import.meta.env as PlatformBackendEnv,
): PlatformBackendConfig {
  return {
    enabled: env.VITE_RAG_PLATFORM_ENABLED?.trim().toLowerCase() !== "false",
    baseUrl: trimTrailingSlash(env.VITE_RAG_PLATFORM_BASE_URL?.trim() ?? ""),
    apiPrefix: normalizePrefix(env.VITE_RAG_PLATFORM_API_PREFIX),
    proxyTarget: trimTrailingSlash(
      env.VITE_RAG_PLATFORM_PROXY_TARGET?.trim() ?? "",
    ),
    requestTimeoutMs: DEFAULT_REQUEST_TIMEOUT_MS,
  };
}

function assertRelativeEndpoint(endpoint: string): string {
  const trimmed = endpoint.trim();
  if (!trimmed.startsWith("/") || trimmed.startsWith("//")) {
    throw new TypeError(
      "Rag Platform endpoint'i tek eğik çizgiyle başlamalıdır.",
    );
  }
  if (/^[a-z][a-z\d+.-]*:/i.test(trimmed)) {
    throw new TypeError("Rag Platform endpoint'i mutlak URL olamaz.");
  }
  return trimmed;
}

export function resolvePlatformUrl(
  endpoint: string,
  pathMode: PlatformPathMode = "api",
  config: PlatformBackendConfig = getPlatformBackendConfig(),
): string {
  const safeEndpoint = assertRelativeEndpoint(endpoint);
  const prefix = pathMode === "api" ? config.apiPrefix : "";
  return `${config.baseUrl}${prefix}${safeEndpoint}`;
}
