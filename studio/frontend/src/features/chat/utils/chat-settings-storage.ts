// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  getChatSettings,
  saveChatSettingsPatch,
  saveChatSettingsPatchIfCurrent,
  type ChatSettingsPath,
  type PersistedChatPreset,
  type PersistedChatSettings,
  type PersistedInferenceParams,
} from "../api/chat-settings-api";
import { normalizePresetLoadConfig } from "../presets/preset-load-config";
import {
  BUILTIN_PRESETS,
  defaultInferenceParams,
  getPresetOwnedConfigKey,
  getUniquePresetName,
  normalizeCustomPresets,
  type ChatPresetSource,
  type Preset,
} from "../presets/preset-policy";
import type { ReasoningEffort } from "../stores/chat-runtime-store";
import { MAX_SAMPLING_SEED } from "../types/runtime";
import {
  sanitizeCompactionHeadroomRatio,
  sanitizeContextPolicy,
} from "./auto-compaction";
import {
  assignSanitizedMirroredSettings,
  hasNoMirroredSettings,
} from "./mirrored-chat-settings";

const AUTO_TITLE_KEY = "unsloth_chat_auto_title";
const AUTO_HEAL_TOOL_CALLS_KEY = "unsloth_auto_heal_tool_calls";
const NUDGE_TOOL_CALLS_KEY = "unsloth_nudge_tool_calls";
const MAX_TOOL_CALLS_KEY = "unsloth_max_tool_calls_per_message";
const TOOL_CALL_TIMEOUT_KEY = "unsloth_tool_call_timeout";
const INFERENCE_PARAMS_KEY = "unsloth_chat_inference_params";
const CHAT_ACTIVE_PRESET_KEY = "unsloth_chat_active_preset";
const CHAT_ACTIVE_PRESET_SOURCE_KEY = "unsloth_chat_active_preset_source";
const REASONING_EFFORT_KEY = "unsloth_reasoning_effort";
const PRESERVE_THINKING_KEY = "unsloth_preserve_thinking";
const COLLAPSE_HTML_ARTIFACTS_KEY = "unsloth_chat_collapse_html_artifacts";
const ALLOW_ARTIFACT_NETWORK_ACCESS_KEY =
  "unsloth_chat_allow_artifact_network_access";
const CHAT_PRESETS_KEY = "unsloth_chat_custom_presets";
const LEGACY_CHAT_SYSTEM_PROMPTS_KEY = "unsloth_chat_system_prompts";
const LEGACY_CHAT_SETTINGS_IMPORT_KEY =
  "unsloth_chat_settings_imported_to_studio_db";

const NUMERIC_INFERENCE_FIELDS = [
  "temperature",
  "topP",
  "topK",
  "minP",
  "repetitionPenalty",
  "presencePenalty",
  "maxSeqLength",
  "maxTokens",
] as const satisfies readonly (keyof PersistedInferenceParams)[];

const CHAT_PRESET_SOURCES = new Set<string>([
  "builtin-default",
  "custom",
  "modified",
]);

const REASONING_EFFORTS = new Set<string>([
  "none",
  "minimal",
  "low",
  "medium",
  "high",
  "max",
  "xhigh",
]);

interface LegacySystemPromptTemplate {
  name: string;
  content: string;
}

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value != null && typeof value === "object" && !Array.isArray(value);
}

function hasKeys(value: object): boolean {
  return Object.keys(value).length > 0;
}

function getStorageItem(key: string): string | null {
  if (!canUseStorage()) return null;
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function isLegacySettingsImportDone(): boolean {
  return getStorageItem(LEGACY_CHAT_SETTINGS_IMPORT_KEY) === "true";
}

function markLegacySettingsImportDone(): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(LEGACY_CHAT_SETTINGS_IMPORT_KEY, "true");
  } catch {
    // ignore
  }
}

function parseJson(value: string | null): unknown {
  if (!value) return undefined;
  try {
    return JSON.parse(value) as unknown;
  } catch {
    return undefined;
  }
}

function loadBool(key: string): boolean | undefined {
  const raw = getStorageItem(key);
  if (raw === "true") return true;
  if (raw === "false") return false;
  return undefined;
}

function loadInt(key: string, min: number): number | undefined {
  const raw = getStorageItem(key);
  if (raw == null || raw.trim() === "") return undefined;
  const value = Number(raw);
  return Number.isInteger(value) && value >= min ? value : undefined;
}

function sanitizeInferenceParams(
  value: unknown,
): PersistedInferenceParams | undefined {
  if (!isRecord(value)) return undefined;

  const params: PersistedInferenceParams = {};
  for (const field of NUMERIC_INFERENCE_FIELDS) {
    const fieldValue = value[field];
    if (typeof fieldValue === "number" && Number.isFinite(fieldValue)) {
      params[field] = fieldValue;
    }
  }
  if (typeof value.systemPrompt === "string") {
    params.systemPrompt = value.systemPrompt;
  }
  if (typeof value.systemVariables === "string") {
    params.systemVariables = value.systemVariables;
  }
  // trustRemoteCode is no longer persisted: custom code is consented per model via the dialog.
  if (typeof value.fastMode === "boolean") {
    params.fastMode = value.fastMode;
  }
  // Bounded here as well as in the panel: this gates the hydration read and the outgoing PUT alike.
  if (value.seed === null) {
    // Kept, not dropped: the server merge overwrites the keys it receives and removes none.
    params.seed = null;
  } else if (
    typeof value.seed === "number" &&
    Number.isInteger(value.seed) &&
    value.seed >= 0 &&
    value.seed <= MAX_SAMPLING_SEED
  ) {
    params.seed = value.seed;
  }
  return hasKeys(params) ? params : undefined;
}

// Not capped. The server merge never removes keys and keeps an existing key in its original
// position, so a load-time trim would permanently hide the oldest entries: editing one of those
// models would write an update the next reload silently drops. Entries are a dozen numbers each.
function sanitizeInferenceParamsByModel(
  value: unknown,
): Record<string, PersistedInferenceParams> | undefined {
  if (!isRecord(value)) return undefined;
  const byModel: Record<string, PersistedInferenceParams> = {};
  for (const [modelId, params] of Object.entries(value)) {
    if (!modelId) continue;
    const sanitized = sanitizeInferenceParams(params);
    if (sanitized) byModel[modelId] = sanitized;
  }
  return hasKeys(byModel) ? byModel : undefined;
}

function toFullPreset(preset: PersistedChatPreset): Preset {
  const loadConfig = normalizePresetLoadConfig(preset.loadConfig);
  return {
    name: preset.name,
    params: {
      ...defaultInferenceParams,
      ...preset.params,
      checkpoint: defaultInferenceParams.checkpoint,
    },
    ...(loadConfig ? { loadConfig } : {}),
  };
}

function sanitizeCustomPresets(
  value: unknown,
): PersistedChatPreset[] | undefined {
  if (!Array.isArray(value)) return undefined;
  if (value.length === 0) return [];

  const presets = value
    .map((item): PersistedChatPreset | null => {
      if (!isRecord(item) || typeof item.name !== "string") return null;
      const name = item.name.trim();
      if (!name) return null;
      const params = sanitizeInferenceParams(item.params);
      const loadConfig = normalizePresetLoadConfig(item.loadConfig);
      return {
        name,
        params: params ?? {},
        ...(loadConfig ? { loadConfig } : {}),
      };
    })
    .filter((preset): preset is PersistedChatPreset => preset !== null);

  if (presets.length === 0) return [];
  return normalizeCustomPresets(presets.map(toFullPreset)).map(
    (preset, index) => ({
      name: preset.name,
      params: presets[index]?.params ?? {},
      ...(preset.loadConfig ? { loadConfig: preset.loadConfig } : {}),
    }),
  );
}

function sanitizePresetSource(value: unknown): ChatPresetSource | undefined {
  return typeof value === "string" && CHAT_PRESET_SOURCES.has(value)
    ? (value as ChatPresetSource)
    : undefined;
}

function sanitizeReasoningEffort(value: unknown): ReasoningEffort | undefined {
  return typeof value === "string" && REASONING_EFFORTS.has(value)
    ? (value as ReasoningEffort)
    : undefined;
}

function sanitizeBool(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function sanitizeInt(value: unknown, min: number): number | undefined {
  return typeof value === "number" && Number.isInteger(value) && value >= min
    ? value
    : undefined;
}

export function sanitizeChatSettings(value: unknown): PersistedChatSettings {
  if (!isRecord(value)) return {};

  const settings: PersistedChatSettings = {};
  const inferenceParams = sanitizeInferenceParams(value.inferenceParams);
  const inferenceParamsByModel = sanitizeInferenceParamsByModel(
    value.inferenceParamsByModel,
  );
  const rememberParamsPerModel = sanitizeBool(value.rememberParamsPerModel);
  const customPresets = sanitizeCustomPresets(value.customPresets);
  const activePresetSource = sanitizePresetSource(value.activePresetSource);
  const reasoningEffort = sanitizeReasoningEffort(value.reasoningEffort);
  const autoTitle = sanitizeBool(value.autoTitle);
  const preserveThinking = sanitizeBool(value.preserveThinking);
  const collapseHtmlArtifacts = sanitizeBool(value.collapseHtmlArtifacts);
  const allowArtifactNetworkAccess = sanitizeBool(
    value.allowArtifactNetworkAccess,
  );
  const autoHealToolCalls = sanitizeBool(value.autoHealToolCalls);
  const nudgeToolCalls = sanitizeBool(value.nudgeToolCalls);
  const autoCompactEnabled = sanitizeBool(value.autoCompactEnabled);
  const contextPolicy = sanitizeContextPolicy(value.contextPolicy);
  const compactionHeadroomRatio = sanitizeCompactionHeadroomRatio(
    value.compactionHeadroomRatio,
  );
  const maxToolCallsPerMessage = sanitizeInt(value.maxToolCallsPerMessage, 1);
  const toolCallTimeout = sanitizeInt(value.toolCallTimeout, 1);

  if (inferenceParams) settings.inferenceParams = inferenceParams;
  if (inferenceParamsByModel) {
    settings.inferenceParamsByModel = inferenceParamsByModel;
  }
  if (rememberParamsPerModel !== undefined) {
    settings.rememberParamsPerModel = rememberParamsPerModel;
  }
  if (customPresets !== undefined) settings.customPresets = customPresets;
  if (typeof value.activePreset === "string" && value.activePreset.trim()) {
    settings.activePreset = value.activePreset.trim();
  }
  if (activePresetSource) settings.activePresetSource = activePresetSource;
  if (autoTitle !== undefined) settings.autoTitle = autoTitle;
  if (reasoningEffort) settings.reasoningEffort = reasoningEffort;
  if (preserveThinking !== undefined)
    settings.preserveThinking = preserveThinking;
  if (collapseHtmlArtifacts !== undefined) {
    settings.collapseHtmlArtifacts = collapseHtmlArtifacts;
  }
  if (allowArtifactNetworkAccess !== undefined) {
    settings.allowArtifactNetworkAccess = allowArtifactNetworkAccess;
  }
  if (autoHealToolCalls !== undefined) {
    settings.autoHealToolCalls = autoHealToolCalls;
  }
  if (nudgeToolCalls !== undefined) {
    settings.nudgeToolCalls = nudgeToolCalls;
  }
  if (autoCompactEnabled !== undefined) {
    settings.autoCompactEnabled = autoCompactEnabled;
  }
  if (contextPolicy) settings.contextPolicy = contextPolicy;
  if (compactionHeadroomRatio !== undefined) {
    settings.compactionHeadroomRatio = compactionHeadroomRatio;
  }
  if (maxToolCallsPerMessage !== undefined) {
    settings.maxToolCallsPerMessage = maxToolCallsPerMessage;
  }
  if (toolCallTimeout !== undefined) settings.toolCallTimeout = toolCallTimeout;
  assignSanitizedMirroredSettings(value, settings);

  return settings;
}

function loadLegacySystemPromptPresets(
  existingPresets: PersistedChatPreset[],
): PersistedChatPreset[] {
  const parsed = parseJson(getStorageItem(LEGACY_CHAT_SYSTEM_PROMPTS_KEY));
  if (!Array.isArray(parsed)) return [];

  const usedNames = new Set([
    ...BUILTIN_PRESETS.map((preset) => preset.name),
    ...existingPresets.map((preset) => preset.name),
  ]);
  const seenConfigKeys = new Set(
    [...BUILTIN_PRESETS, ...existingPresets.map(toFullPreset)].map((preset) =>
      getPresetOwnedConfigKey(preset.params),
    ),
  );

  return parsed
    .filter((item): item is LegacySystemPromptTemplate => {
      if (!isRecord(item)) return false;
      return typeof item.name === "string" && typeof item.content === "string";
    })
    .map((template) => ({
      template,
      params: {
        ...defaultInferenceParams,
        systemPrompt: template.content,
      },
    }))
    .filter(({ params }) => {
      const configKey = getPresetOwnedConfigKey(params);
      if (seenConfigKeys.has(configKey)) return false;
      seenConfigKeys.add(configKey);
      return true;
    })
    .map(({ template, params }) => ({
      name: getUniquePresetName(`${template.name} Prompt`, usedNames),
      params: sanitizeInferenceParams(params) ?? {},
    }));
}

export function isEmptyChatSettings(settings: PersistedChatSettings): boolean {
  return (
    (!settings.inferenceParams || !hasKeys(settings.inferenceParams)) &&
    settings.inferenceParamsByModel === undefined &&
    settings.rememberParamsPerModel === undefined &&
    settings.customPresets === undefined &&
    settings.activePreset === undefined &&
    settings.activePresetSource === undefined &&
    settings.autoTitle === undefined &&
    settings.reasoningEffort === undefined &&
    settings.preserveThinking === undefined &&
    settings.collapseHtmlArtifacts === undefined &&
    settings.allowArtifactNetworkAccess === undefined &&
    settings.autoHealToolCalls === undefined &&
    settings.nudgeToolCalls === undefined &&
    settings.autoCompactEnabled === undefined &&
    settings.contextPolicy === undefined &&
    settings.compactionHeadroomRatio === undefined &&
    settings.maxToolCallsPerMessage === undefined &&
    settings.toolCallTimeout === undefined &&
    hasNoMirroredSettings(settings)
  );
}

export function loadLegacyChatSettings(): PersistedChatSettings {
  const settings: PersistedChatSettings = {};
  const rawCustomPresets = getStorageItem(CHAT_PRESETS_KEY);
  const rawLegacyPromptPresets = getStorageItem(LEGACY_CHAT_SYSTEM_PROMPTS_KEY);
  const hasLegacyPresetStorage =
    rawCustomPresets !== null || rawLegacyPromptPresets !== null;
  const inferenceParams = sanitizeInferenceParams(
    parseJson(getStorageItem(INFERENCE_PARAMS_KEY)),
  );
  const customPresets = sanitizeCustomPresets(parseJson(rawCustomPresets));
  const legacyPromptPresets = loadLegacySystemPromptPresets(
    customPresets ?? [],
  );
  const activePreset = getStorageItem(CHAT_ACTIVE_PRESET_KEY);
  const activePresetSource = sanitizePresetSource(
    getStorageItem(CHAT_ACTIVE_PRESET_SOURCE_KEY),
  );
  const reasoningEffort = sanitizeReasoningEffort(
    getStorageItem(REASONING_EFFORT_KEY),
  );
  const autoTitle = loadBool(AUTO_TITLE_KEY);
  const preserveThinking = loadBool(PRESERVE_THINKING_KEY);
  const collapseHtmlArtifacts = loadBool(COLLAPSE_HTML_ARTIFACTS_KEY);
  const allowArtifactNetworkAccess = loadBool(ALLOW_ARTIFACT_NETWORK_ACCESS_KEY);
  const autoHealToolCalls = loadBool(AUTO_HEAL_TOOL_CALLS_KEY);
  const nudgeToolCalls = loadBool(NUDGE_TOOL_CALLS_KEY);
  const maxToolCallsPerMessage = loadInt(MAX_TOOL_CALLS_KEY, 1);
  const toolCallTimeout = loadInt(TOOL_CALL_TIMEOUT_KEY, 1);
  const allCustomPresets = sanitizeCustomPresets([
    ...(customPresets ?? []),
    ...legacyPromptPresets,
  ]);

  if (inferenceParams) settings.inferenceParams = inferenceParams;
  if (hasLegacyPresetStorage && allCustomPresets !== undefined) {
    settings.customPresets = allCustomPresets;
  }
  if (activePreset?.trim()) settings.activePreset = activePreset.trim();
  if (activePresetSource) settings.activePresetSource = activePresetSource;
  if (autoTitle !== undefined) settings.autoTitle = autoTitle;
  if (reasoningEffort) settings.reasoningEffort = reasoningEffort;
  if (preserveThinking !== undefined)
    settings.preserveThinking = preserveThinking;
  if (collapseHtmlArtifacts !== undefined) {
    settings.collapseHtmlArtifacts = collapseHtmlArtifacts;
  }
  if (allowArtifactNetworkAccess !== undefined) {
    settings.allowArtifactNetworkAccess = allowArtifactNetworkAccess;
  }
  if (autoHealToolCalls !== undefined) {
    settings.autoHealToolCalls = autoHealToolCalls;
  }
  if (nudgeToolCalls !== undefined) {
    settings.nudgeToolCalls = nudgeToolCalls;
  }
  if (maxToolCallsPerMessage !== undefined) {
    settings.maxToolCallsPerMessage = maxToolCallsPerMessage;
  }
  if (toolCallTimeout !== undefined) settings.toolCallTimeout = toolCallTimeout;

  return settings;
}

export interface LoadedChatSettings {
  settings: PersistedChatSettings;
  /** The GET answered, so a mirrored field missing from `settings` is missing on the server too.
   *  False when the read fell back to this browser's legacy storage: nothing is then known about the
   *  server, and treating every field as absent would back this browser's stale values over another's. */
  fromServer: boolean;
  /**
   * Whether these values are what the server holds. False when a legacy import
   * merged local values but failed to save them: the server answered, so
   * absence is still authoritative, but the merge exists only in this session
   * and a later re-read would silently drop it.
   */
  persisted: boolean;
}

export async function loadChatSettingsWithLegacyImport(): Promise<LoadedChatSettings> {
  let dbSettings: PersistedChatSettings;
  try {
    dbSettings = sanitizeChatSettings(await getChatSettings());
  } catch (error) {
    const legacySettings = loadLegacyChatSettings();
    if (isEmptyChatSettings(legacySettings)) {
      throw error;
    }
    return { settings: legacySettings, fromServer: false, persisted: false };
  }

  const legacySettings = loadLegacyChatSettings();
  if (isLegacySettingsImportDone()) {
    if (
      !isEmptyChatSettings(dbSettings) ||
      isEmptyChatSettings(legacySettings)
    ) {
      return { settings: dbSettings, fromServer: true, persisted: true };
    }
    try {
      return {
        settings: sanitizeChatSettings(
          await saveChatSettingsPatch(legacySettings),
        ),
        fromServer: true,
        persisted: true,
      };
    } catch {
      // The GET still answered (empty), so absence remains authoritative.
      return { settings: legacySettings, fromServer: true, persisted: false };
    }
  }

  if (isEmptyChatSettings(legacySettings)) {
    markLegacySettingsImportDone();
    return { settings: dbSettings, fromServer: true, persisted: true };
  }

  const mergedSettings = {
    ...legacySettings,
    ...dbSettings,
    inferenceParams: {
      ...legacySettings.inferenceParams,
      ...dbSettings.inferenceParams,
    },
  };
  try {
    const savedSettings = sanitizeChatSettings(
      await saveChatSettingsPatch(mergedSettings),
    );
    markLegacySettingsImportDone();
    return { settings: savedSettings, fromServer: true, persisted: true };
  } catch {
    return { settings: mergedSettings, fromServer: true, persisted: false };
  }
}

export async function savePersistedChatSettingsPatch(
  patch: PersistedChatSettings,
  options: { keepalive?: boolean } = {},
): Promise<PersistedChatSettings> {
  return sanitizeChatSettings(
    await saveChatSettingsPatch(sanitizeChatSettings(patch), options),
  );
}

export async function savePersistedChatSettingsPatchIfCurrent(
  expected: PersistedChatSettings,
  patch: PersistedChatSettings,
  expectedAbsent: Array<keyof PersistedChatSettings> = [],
  expectedAbsentPaths: ChatSettingsPath[] = [],
): Promise<{ settings: PersistedChatSettings; applied: boolean }> {
  const result = await saveChatSettingsPatchIfCurrent(
    sanitizeChatSettings(expected),
    sanitizeChatSettings(patch),
    expectedAbsent,
    expectedAbsentPaths,
  );
  return {
    settings: sanitizeChatSettings(result.settings),
    applied: result.applied,
  };
}
