export const CUSTOM_PROVIDER_PRESETS = [
  {
    providerType: "llama_cpp",
    displayName: "llama.cpp",
    baseUrlPlaceholder: "http://localhost:8080/v1",
    modelIdsPlaceholder: "gpt-oss-20b\nqwen3-14b",
  },
  {
    providerType: "vllm",
    displayName: "vLLM",
    baseUrlPlaceholder: "https://my-vllm-server.com/v1",
    modelIdsPlaceholder: "openai/gpt-oss-20b\nQwen/Qwen3-14B",
  },
  {
    providerType: "ollama",
    displayName: "Ollama",
    baseUrlPlaceholder: "http://localhost:11434/v1",
    modelIdsPlaceholder: "gpt-oss:20b\nqwen3:14b",
  },
] as const;

// No vendor with a fixed endpoint belongs in this list. A preset here is a SECOND row for a
// provider the registry already offers, and a worse one: isCustomProviderType() is true for a
// preset, so toExternalBackendProviderType() maps it to CUSTOM_BACKEND_PROVIDER_TYPE ("openai")
// and the saved row stops carrying its own type. Every backend gate keyed on provider_type then
// misses it -- a DeepInfra preset lost service_tier, its reasoning table and its vision gate --
// and the form grows a Base URL field for an endpoint that is already known. Presets are for
// endpoints the USER supplies (llama.cpp, vLLM, Ollama); cloud vendors belong in
// PROVIDER_REGISTRY alone. See list_available_providers() in providers.py.

const CUSTOM_PROVIDER_LABELS: Record<string, string> = {
  [LEGACY_CUSTOM_PROVIDER_TYPE]: CUSTOM_PROVIDER_DISPLAY_NAME,
  ...Object.fromEntries(
    CUSTOM_PROVIDER_PRESETS.map((preset) => [
      preset.providerType,
      preset.displayName,
    ]),
  ),
};

const CUSTOM_PROVIDER_BASE_URL_PLACEHOLDERS: Record<string, string> = {
  [LEGACY_CUSTOM_PROVIDER_TYPE]: "https://my-vllm-server.com/v1",
  ...Object.fromEntries(
    CUSTOM_PROVIDER_PRESETS.map((preset) => [
      preset.providerType,
      preset.baseUrlPlaceholder,
    ]),
  ),
};

const CUSTOM_PROVIDER_MODEL_IDS_PLACEHOLDERS: Record<string, string> = {
  [LEGACY_CUSTOM_PROVIDER_TYPE]: "openai/gpt-oss-20b\nQwen/Qwen3-14B",
  ...Object.fromEntries(
    CUSTOM_PROVIDER_PRESETS.map((preset) => [
      preset.providerType,
      preset.modelIdsPlaceholder,
    ]),
  ),
};

export function isCustomProviderType(
  providerType: string | null | undefined,
): boolean {
  if (!providerType) return false;
  return providerType in CUSTOM_PROVIDER_LABELS;
}

// Types whose /v1/models lives at a base URL the USER supplies. Membership is what makes the form
// demand a Base URL, so a registry provider with a fixed endpoint must stay out: it would ask for
// a URL it already knows, and Load Models would answer "Base URL is required" on a connection the
// user never saw a URL field for. Kept in step with CUSTOM_PROVIDER_PRESETS above.
const REMOTE_MODEL_CATALOG_CUSTOM_PROVIDER_TYPES = new Set([
  LEGACY_CUSTOM_PROVIDER_TYPE,
  "ollama",
  "vllm",
  "llama_cpp",
]);

export function supportsRemoteModelCatalog(
  providerType: string | null | undefined,
): boolean {
  return (
    providerType != null &&
    REMOTE_MODEL_CATALOG_CUSTOM_PROVIDER_TYPES.has(providerType)
  );
}

/** Presets that hide the API-key field. Ollama is not skipped: Ollama cloud requires a key;
 *  local servers leave the optional field empty. */
export function customPresetSkipsApiKeyField(
  providerType: string | null | undefined,
): boolean {
  return providerType === "llama_cpp";
}

// REGISTRY connections that read a remote catalogue but still accept typed-in ids beside it.
// Their /v1/models omits models the account can reach -- OpenRouter's newest slugs, DeepInfra's
// private `deploy_id:` deployments -- so the catalogue not listing an id is not evidence against
// it, and the dialog must keep its "Additional model IDs" box for them. Membership here also
// stops the picker reporting such a model as "dropped" once its catalogue moves on.
const MANUAL_MODEL_IDS_WITH_CATALOG_PROVIDER_TYPES = new Set([
  "openrouter",
  "deepinfra",
]);

/** Catalog load plus optional manual model IDs. */
export function allowsManualModelIdsWithCatalog(
  providerType: string | null | undefined,
): boolean {
  if (!providerType) return false;
  return (
    MANUAL_MODEL_IDS_WITH_CATALOG_PROVIDER_TYPES.has(providerType) ||
    supportsRemoteModelCatalog(providerType)
  );
}
