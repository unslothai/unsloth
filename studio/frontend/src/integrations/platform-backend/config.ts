export interface PlatformBackendConfig {
  enabled: boolean;
  baseUrl: string;
  apiPrefix: string;
  proxyTarget: string;
  requestTimeoutMs: number;
}

export type PlatformPathMode = "api" | "root";

interface PlatformBackendEnv {
  VITE_BACKEND_MODE?: string;
  VITE_RAG_PLATFORM_BASE_URL?: string;
  VITE_RAG_PLATFORM_API_PREFIX?: string;
  VITE_RAG_PLATFORM_ENABLED?: string;
  VITE_RAG_PLATFORM_PROXY_TARGET?: string;
  VITE_RAG_PLATFORM_AUTH_ENABLED?: string;
  VITE_RAG_PLATFORM_AUTH_PUBLIC_KEY?: string;
  VITE_RAG_PLATFORM_AUTH_PUBLIC_KEY_B64?: string;
  VITE_RAG_PLATFORM_REGISTRATION_ENABLED?: string;
  VITE_RAG_PLATFORM_PASSWORD_RECOVERY_ENABLED?: string;
  VITE_RAG_PLATFORM_OAUTH_ENABLED?: string;
  VITE_RAG_PLATFORM_MODEL_TOOLS_ENABLED?: string;
  VITE_RAG_PLATFORM_CONNECTORS_ENABLED?: string;
  VITE_RAG_PLATFORM_MEMORY_ENABLED?: string;
  VITE_RAG_PLATFORM_SEARCH_ENABLED?: string;
  VITE_RAG_PLATFORM_ADMIN_ENABLED?: string;
  VITE_RAG_PLATFORM_ADMIN_OPERATIONS_ENABLED?: string;
  VITE_RAG_PLATFORM_TENANTS_ENABLED?: string;
  VITE_RAG_PLATFORM_BOTS_ENABLED?: string;
  VITE_RAG_PLATFORM_CHANNELS_ENABLED?: string;
}

export interface PlatformAuthConfig {
  enabled: boolean;
  oauthEnabled: boolean;
  passwordRecoveryEnabled: boolean;
  publicKeyPem: string;
  registrationEnabled: boolean;
}

export interface PlatformManagementConfig {
  adminEnabled: boolean;
  adminOperationsEnabled: boolean;
  botsEnabled: boolean;
  channelsEnabled: boolean;
  tenantsEnabled: boolean;
}

const DEFAULT_API_PREFIX = "/api/v1";
const DEFAULT_REQUEST_TIMEOUT_MS = 15_000;

function enabledUnlessFalse(value: string | undefined): boolean {
  return value?.trim().toLowerCase() !== "false";
}

function decodePublicKey(
  plain: string | undefined,
  encoded: string | undefined,
): string {
  const direct = plain?.trim().replaceAll("\\n", "\n");
  if (direct) return direct;
  const base64 = encoded?.trim();
  if (!base64) return "";
  try {
    return atob(base64);
  } catch {
    return "";
  }
}

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

export function getPlatformAuthConfig(
  env: PlatformBackendEnv = import.meta.env as PlatformBackendEnv,
): PlatformAuthConfig {
  const platformEnabled = enabledUnlessFalse(env.VITE_RAG_PLATFORM_ENABLED);
  return {
    enabled:
      platformEnabled && enabledUnlessFalse(env.VITE_RAG_PLATFORM_AUTH_ENABLED),
    registrationEnabled: enabledUnlessFalse(
      env.VITE_RAG_PLATFORM_REGISTRATION_ENABLED,
    ),
    passwordRecoveryEnabled: enabledUnlessFalse(
      env.VITE_RAG_PLATFORM_PASSWORD_RECOVERY_ENABLED,
    ),
    oauthEnabled: enabledUnlessFalse(env.VITE_RAG_PLATFORM_OAUTH_ENABLED),
    publicKeyPem: decodePublicKey(
      env.VITE_RAG_PLATFORM_AUTH_PUBLIC_KEY,
      env.VITE_RAG_PLATFORM_AUTH_PUBLIC_KEY_B64,
    ),
  };
}

export function isPlatformAuthEnabled(): boolean {
  return getPlatformAuthConfig().enabled;
}

export function isPlatformModelToolsEnabled(
  env: PlatformBackendEnv = import.meta.env as PlatformBackendEnv,
): boolean {
  return (
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_ENABLED) &&
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_AUTH_ENABLED) &&
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_MODEL_TOOLS_ENABLED)
  );
}

export function isPlatformConnectorsEnabled(
  env: PlatformBackendEnv = import.meta.env as PlatformBackendEnv,
): boolean {
  return (
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_ENABLED) &&
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_AUTH_ENABLED) &&
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_CONNECTORS_ENABLED)
  );
}

export function isPlatformMemoryEnabled(
  env: PlatformBackendEnv = import.meta.env as PlatformBackendEnv,
): boolean {
  return (
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_ENABLED) &&
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_AUTH_ENABLED) &&
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_MEMORY_ENABLED)
  );
}

export function isPlatformSearchEnabled(
  env: PlatformBackendEnv = import.meta.env as PlatformBackendEnv,
): boolean {
  return (
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_ENABLED) &&
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_AUTH_ENABLED) &&
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_SEARCH_ENABLED)
  );
}

export function getPlatformManagementConfig(
  env: PlatformBackendEnv = import.meta.env as PlatformBackendEnv,
): PlatformManagementConfig {
  const baseEnabled =
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_ENABLED) &&
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_AUTH_ENABLED);
  const adminEnabled =
    baseEnabled && enabledUnlessFalse(env.VITE_RAG_PLATFORM_ADMIN_ENABLED);
  return {
    adminEnabled,
    adminOperationsEnabled:
      adminEnabled &&
      enabledUnlessFalse(env.VITE_RAG_PLATFORM_ADMIN_OPERATIONS_ENABLED),
    tenantsEnabled:
      baseEnabled && enabledUnlessFalse(env.VITE_RAG_PLATFORM_TENANTS_ENABLED),
    botsEnabled:
      baseEnabled && enabledUnlessFalse(env.VITE_RAG_PLATFORM_BOTS_ENABLED),
    channelsEnabled:
      baseEnabled && enabledUnlessFalse(env.VITE_RAG_PLATFORM_CHANNELS_ENABLED),
  };
}

/**
 * Chat/session persistence is part of the native Rag Platform integration.
 * It deliberately has no independent rollout flag: when the platform and its
 * authenticated API are enabled, Chat and Session are the source of truth.
 */
export function isPlatformChatPersistenceEnabled(
  env: PlatformBackendEnv = import.meta.env as PlatformBackendEnv,
): boolean {
  const backendMode = env.VITE_BACKEND_MODE?.trim().toLowerCase();
  return (
    backendMode !== "legacy" &&
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_ENABLED) &&
    enabledUnlessFalse(env.VITE_RAG_PLATFORM_AUTH_ENABLED)
  );
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
