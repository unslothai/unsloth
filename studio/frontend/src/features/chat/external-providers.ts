// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export interface ExternalProviderConfig {
  id: string;
  /** Built-in preset id (e.g. openai) or `custom` for advanced base URL. */
  presetId: string;
  /** Display name in UI (usually preset label). */
  name: string;
  /** Only for `custom`: user-supplied OpenAI-compatible base URL. Empty for presets. */
  baseUrl: string;
  /** Not persisted; session-only for now. */
  apiKey: string;
  /** Model ids the user enabled after loading from the proxy `/models` list. */
  models: string[];
  createdAt: number;
  updatedAt: number;
}

const EXTERNAL_PROVIDERS_KEY = "unsloth_chat_external_providers";
const EXTERNAL_MODEL_PREFIX = "external::";

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

export function isExternalModelId(
  value: string | null | undefined,
): value is string {
  return typeof value === "string" && value.startsWith(EXTERNAL_MODEL_PREFIX);
}

export function buildExternalModelId(providerId: string, modelId: string): string {
  return `${EXTERNAL_MODEL_PREFIX}${providerId}::${encodeURIComponent(modelId)}`;
}

export function parseExternalModelId(
  value: string | null | undefined,
): { providerId: string; modelId: string } | null {
  if (!isExternalModelId(value)) return null;
  const payload = value.slice(EXTERNAL_MODEL_PREFIX.length);
  const separator = payload.indexOf("::");
  if (separator < 0) return null;
  const providerId = payload.slice(0, separator);
  const encodedModelId = payload.slice(separator + 2);
  if (!providerId || !encodedModelId) return null;
  try {
    return { providerId, modelId: decodeURIComponent(encodedModelId) };
  } catch {
    return null;
  }
}

function isExternalProviderConfig(value: unknown): value is ExternalProviderConfig {
  if (!value || typeof value !== "object") return false;
  const maybe = value as Partial<ExternalProviderConfig>;
  return (
    typeof maybe.id === "string" &&
    typeof maybe.name === "string" &&
    typeof maybe.baseUrl === "string" &&
    typeof maybe.apiKey === "string" &&
    Array.isArray(maybe.models)
  );
}

function normalizeProvider(raw: ExternalProviderConfig): ExternalProviderConfig {
  const presetId =
    typeof raw.presetId === "string" && raw.presetId.length > 0
      ? raw.presetId
      : "custom";
  return {
    ...raw,
    presetId,
    name: raw.name.trim(),
    baseUrl: raw.baseUrl.trim(),
    apiKey: "",
    models: raw.models
      .map((model) => model.trim())
      .filter((model) => model.length > 0),
  };
}

function isCompleteProvider(provider: ExternalProviderConfig): boolean {
  if (!provider.id || !provider.name) return false;
  if (provider.presetId === "custom") return false;
  if (provider.models.length === 0) return false;
  return true;
}

export function loadExternalProviders(): ExternalProviderConfig[] {
  if (!canUseStorage()) return [];
  try {
    const raw = localStorage.getItem(EXTERNAL_PROVIDERS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(isExternalProviderConfig)
      .map(normalizeProvider)
      .filter(isCompleteProvider);
  } catch {
    return [];
  }
}

export function saveExternalProviders(providers: ExternalProviderConfig[]): void {
  if (!canUseStorage()) return;
  try {
    const withoutKeys = providers.map((provider) => ({
      ...provider,
      apiKey: "",
    }));
    localStorage.setItem(EXTERNAL_PROVIDERS_KEY, JSON.stringify(withoutKeys));
  } catch {
    // ignore
  }
}
