export type BackendMode = "platform" | "legacy" | "hybrid";

export type ProductCapabilityId =
  | "chat"
  | "projects"
  | "knowledge"
  | "settings"
  | "agents"
  | "files"
  | "memory"
  | "search"
  | "management"
  | "local-model-lifecycle"
  | "training"
  | "recipes"
  | "export"
  | "image-generation"
  | "audio-generation"
  | "video-generation"
  | "api-monitor"
  | "model-cache";

export interface ProductCapability {
  id: ProductCapabilityId;
  available: boolean;
  visibleInNavigation: boolean;
  reason: string | null;
}

interface BackendModeEnv {
  VITE_BACKEND_MODE?: string;
  VITE_RAG_PLATFORM_ENABLED?: string;
}

const PLATFORM_UNAVAILABLE_REASON =
  "Bu özellik Rag Platform backend tarafından desteklenmiyor.";

export function getBackendMode(
  env: BackendModeEnv = import.meta.env as BackendModeEnv,
): BackendMode {
  const configured = env.VITE_BACKEND_MODE?.trim().toLowerCase();
  if (configured === "legacy" || configured === "hybrid") return configured;
  if (configured === "platform") return "platform";
  return env.VITE_RAG_PLATFORM_ENABLED?.trim().toLowerCase() === "false"
    ? "legacy"
    : "platform";
}

export function isPlatformOnlyMode(
  env: BackendModeEnv = import.meta.env as BackendModeEnv,
): boolean {
  return getBackendMode(env) === "platform";
}

export function createCapabilityRegistry(
  env: BackendModeEnv = import.meta.env as BackendModeEnv,
): Readonly<Record<ProductCapabilityId, ProductCapability>> {
  const platformOnly = isPlatformOnlyMode(env);
  const supported = (
    id: ProductCapabilityId,
    visibleInNavigation = false,
  ): ProductCapability => ({
    id,
    available: true,
    visibleInNavigation,
    reason: null,
  });
  const unavailable = (
    id: ProductCapabilityId,
    visibleInNavigation = false,
    reason = PLATFORM_UNAVAILABLE_REASON,
  ): ProductCapability => ({
    id,
    available: false,
    visibleInNavigation,
    reason,
  });
  const legacyOnly = (id: ProductCapabilityId): ProductCapability =>
    platformOnly ? unavailable(id) : supported(id, true);

  return Object.freeze({
    chat: supported("chat", true),
    projects: supported("projects", true),
    knowledge: supported("knowledge", true),
    settings: supported("settings", true),
    agents:
      getBackendMode(env) === "legacy"
        ? unavailable("agents")
        : supported("agents", true),
    files:
      getBackendMode(env) === "legacy"
        ? unavailable("files")
        : supported("files", true),
    memory:
      getBackendMode(env) === "legacy"
        ? unavailable("memory")
        : supported("memory", true),
    search:
      getBackendMode(env) === "legacy"
        ? unavailable("search")
        : supported("search", true),
    management:
      getBackendMode(env) === "legacy"
        ? unavailable("management")
        : supported("management", true),
    "local-model-lifecycle": legacyOnly("local-model-lifecycle"),
    training: legacyOnly("training"),
    recipes: legacyOnly("recipes"),
    export: legacyOnly("export"),
    "image-generation": legacyOnly("image-generation"),
    "audio-generation": legacyOnly("audio-generation"),
    "video-generation": legacyOnly("video-generation"),
    "api-monitor": legacyOnly("api-monitor"),
    "model-cache": legacyOnly("model-cache"),
  });
}

export const platformCapabilities = createCapabilityRegistry();

export function getProductCapability(
  id: ProductCapabilityId,
): ProductCapability {
  return platformCapabilities[id];
}
