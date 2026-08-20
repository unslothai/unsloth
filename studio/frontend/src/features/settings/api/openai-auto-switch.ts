


import { authFetch } from "@/features/auth";
import { readFastApiError } from "@/lib/format-fastapi-error";

export type OpenAIAutoSwitchSettings = {
  enabled: boolean;
  autoUnloadIdleSeconds: number;
  defaultEnabled: boolean;
  // True when the idle-unload loop will actually unload (e.g. enabled via the
  // UNSLOTH_MODEL_IDLE_TTL env var even while the toggle is off).
  idleUnloadActive: boolean;
  // Persist the KV cache to disk on idle unload and restore it on reload.
  autoUnloadKeepKv: boolean;
  // Fetch a GGUF named in an API request; stored independently of `enabled`, gated on it.
  autoDownloadModel: boolean;
  // Spare models loaded from the UI on idle unload; only free API-loaded ones.
  autoUnloadApiOnly: boolean;
  // Idle TTL for the image and video pipelines. Its own setting, off by default:
  // the chat TTL above is about the OpenAI API and never implied these.
  mediaAutoUnloadIdleSeconds: number;
  // True when the media idle unload will actually run, so the UI can say a veto
  // (residency) is holding a saved TTL off.
  mediaIdleUnloadActive: boolean;
  // Load the image or video model a media request names. Its own setting: the chat
  // toggle above says nothing about pipelines loaded on the Images or Video page.
  mediaAutoSwitchModel: boolean;
};

type ApiOpenAIAutoSwitchSettings = {
  enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_unload_idle_seconds: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  default_enabled: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  idle_unload_active?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_unload_keep_kv?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_download_model?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  auto_unload_api_only?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  media_auto_unload_idle_seconds?: number;
  // biome-ignore lint/style/useNamingConvention: API schema
  media_idle_unload_active?: boolean;
  // biome-ignore lint/style/useNamingConvention: API schema
  media_auto_switch_model?: boolean;
};

let cachedSettings: OpenAIAutoSwitchSettings | null = null;
let inFlightSettings: Promise<OpenAIAutoSwitchSettings> | null = null;
// The generation inFlightSettings was issued at. A caller arriving after an
// invalidation must not adopt a request issued before it: that reply describes
// the pre-write world, and the hub poll puts it straight into idleUnloadArmed.
let inFlightGeneration = -1;

function fromApi(
  settings: ApiOpenAIAutoSwitchSettings,
): OpenAIAutoSwitchSettings {
  return {
    enabled: settings.enabled,
    autoUnloadIdleSeconds: settings.auto_unload_idle_seconds,
    defaultEnabled: settings.default_enabled,
    idleUnloadActive: settings.idle_unload_active ?? false,
    autoUnloadKeepKv: settings.auto_unload_keep_kv ?? true,
    autoDownloadModel: settings.auto_download_model ?? false,
    autoUnloadApiOnly: settings.auto_unload_api_only ?? false,
    mediaAutoUnloadIdleSeconds: settings.media_auto_unload_idle_seconds ?? 0,
    mediaIdleUnloadActive: settings.media_idle_unload_active ?? false,
    mediaAutoSwitchModel: settings.media_auto_switch_model ?? false,
  };
}

async function fetchOpenAIAutoSwitchSettings(): Promise<OpenAIAutoSwitchSettings> {
  const res = await authFetch("/api/settings/openai-auto-switch");
  if (!res.ok) {
    throw new Error(
      await readFastApiError(res, "Failed to load model auto-switch settings"),
    );
  }
  return fromApi(await res.json());
}

// Bumped on every invalidation. A response that was already in flight when the
// cache was cleared must not refill it, or the pre-toggle value would be served
// indefinitely.
let cacheGeneration = 0;

function cacheSettings(settings: OpenAIAutoSwitchSettings, generation: number) {
  if (generation === cacheGeneration) {
    cachedSettings = settings;
  }
  return settings;
}

/**
 * Drop the cached response. `idleUnloadActive` depends on the Model Memory
 * residency setting, which is saved through a different endpoint, so that
 * endpoint invalidates this cache rather than letting it go stale.
 */
export function invalidateOpenAIAutoSwitchSettings() {
  cachedSettings = null;
  cacheGeneration += 1;
}

// An invalidation racing a read is rare, so a couple of retries always
// converges. The bound only exists so a write storm cannot spin here.
const MAX_REREADS = 3;

function startRead(generation: number) {
  // cacheGeneration only increases, so a later read always claims the slot under
  // a different generation and this clear cannot drop its request.
  const read = fetchOpenAIAutoSwitchSettings().finally(() => {
    if (inFlightGeneration === generation) {
      inFlightSettings = null;
    }
  });
  inFlightSettings = read;
  inFlightGeneration = generation;
  return read;
}

export async function loadOpenAIAutoSwitchSettings() {
  let settings: OpenAIAutoSwitchSettings | null = null;
  for (let attempt = 0; attempt < MAX_REREADS; attempt += 1) {
    if (cachedSettings) {
      return cachedSettings;
    }
    const generation = cacheGeneration;
    settings = await (inFlightSettings && inFlightGeneration === generation
      ? inFlightSettings
      : startRead(generation));
    if (generation === cacheGeneration) {
      return cacheSettings(settings, generation);
    }
    // The request was already in flight when the cache was dropped, so this
    // response predates the write. Callers put it straight into idleUnloadArmed,
    // so refetch against the new generation rather than returning it.
  }
  return settings as OpenAIAutoSwitchSettings;
}

/** A partial write: `enabled` is always sent, and an omitted field keeps its stored value. */
export type OpenAIAutoSwitchUpdate = {
  enabled: boolean;
  autoUnloadIdleSeconds?: number;
  autoUnloadKeepKv?: boolean;
  autoDownloadModel?: boolean;
  autoUnloadApiOnly?: boolean;
  mediaAutoUnloadIdleSeconds?: number;
  mediaAutoSwitchModel?: boolean;
};

// Camel-cased update field -> the API schema key it is sent as.
const UPDATE_KEYS = {
  autoUnloadIdleSeconds: "auto_unload_idle_seconds",
  autoUnloadKeepKv: "auto_unload_keep_kv",
  autoDownloadModel: "auto_download_model",
  autoUnloadApiOnly: "auto_unload_api_only",
  mediaAutoUnloadIdleSeconds: "media_auto_unload_idle_seconds",
  mediaAutoSwitchModel: "media_auto_switch_model",
} as const;

export async function updateOpenAIAutoSwitchSettings(
  update: OpenAIAutoSwitchUpdate,
): Promise<OpenAIAutoSwitchSettings> {
  const body: Record<string, unknown> = { enabled: update.enabled };
  for (const [field, key] of Object.entries(UPDATE_KEYS)) {
    const value = update[field as keyof typeof UPDATE_KEYS];
    if (value !== undefined) {
      body[key] = value;
    }
  }
  // Read BEFORE the request: idleUnloadActive depends on the Model Memory
  // setting, so a residency write landing mid-flight makes this response stale
  // even though it is our own write's reply.
  const generation = cacheGeneration;
  const res = await authFetch("/api/settings/openai-auto-switch", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(
      await readFastApiError(
        res,
        "Failed to update model auto-switch settings",
      ),
    );
  }
  const settings = fromApi(await res.json());
  if (generation !== cacheGeneration) {
    // A Model Memory write committed while this PUT was in flight, so this
    // reply predates it. Caching it would pin a stale idleUnloadActive, which
    // the Hub reads to decide whether an eviction was a real unload.
    return loadOpenAIAutoSwitchSettings();
  }
  return cacheSettings(settings, generation);
}
