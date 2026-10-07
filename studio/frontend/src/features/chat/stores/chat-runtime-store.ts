// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import type { ImageDisclosure } from "../api/mcp-image";
import {
  mirrorHfTokenInto,
  useHfTokenStore,
} from "@/features/hub/stores/hf-token-store";
// eslint-disable-next-line no-restricted-imports -- Leaf module; the model-picker index pulls in chat.
import {
  pinnedReasoningEffort,
  useModelReasoningEffortStore,
} from "@/features/model-picker/components/model-selector/model-reasoning-effort";
import {
  loadedContextFields,
  type MlxKvQuant,
} from "@/features/model-picker/model-config/per-model-config";
import {
  cachedPinnableGpuIndexKind,
  reconcileCachedGpuSelection,
  type ReconciledGpuSelection,
  type GpuIndexKind,
} from "@/hooks/use-gpu-info";
import { toast } from "@/lib/toast";
import { DRAFT_N_MAX_SPEC_TYPES } from "@/lib/speculative-modes";
import { create } from "zustand";
import { getChatSettings } from "../api/chat-settings-api";
import {
  GPU_LAYERS_AUTO,
  recoverDroppedDiffusionSplit,
  shouldHydrateGpuPlacementControls,
} from "../lib/gpu-placement";
import {
  externalModelSupportsStudioTools,
  isExternalModelId,
  parseExternalModelId,
} from "../external-providers";
import { isOllamaManifestRef } from "../utils/qwen-sampling-table";
import {
  type ChatPresetSource,
  type Preset,
  getPresetSource,
} from "../presets/preset-policy";
import { normalizeModelIdentity } from "../../hub/lib/model-identity";
import { normalizePresetLoadConfig } from "../presets/preset-load-config";
import {
  CHAT_PROJECT_ATTACHMENT_TARGET_KEY,
  DEFAULT_PROJECT_ATTACHMENT_TARGET,
  normalizeProjectAttachmentTarget,
  type ProjectAttachmentTarget,
} from "../utils/project-attachment-target";
import {
  type ExternalReasoningCapabilities,
  externalReasoningTakesEffort,
  getExternalMaxOutputTokens,
  resolveExternalReasoningEffort,
} from "../provider-capabilities";
import {
  PERSISTED_INFERENCE_PARAM_KEYS,
  REMEMBERED_INFERENCE_PARAM_KEYS,
  type PersistedInferenceParamKey,
  getRememberedParamsPatch,
  getReplayedParams,
  pickRememberedChanges,
  pickRememberedParams,
  setInferenceParam,
} from "../lib/per-model-params";
import {
  type ChatLoraSummary,
  type ChatModelRow,
  DEFAULT_INFERENCE_PARAMS,
  type InferenceParams,
} from "../types/runtime";
import {
  loadChatSettingsWithLegacyImport,
  normalizeSavedChatSettings,
  sanitizeChatSettings,
  savePersistedChatSettingsPatch,
  savePersistedChatSettingsPatchIfCurrent,
} from "../utils/chat-settings-storage";
import {
  loadShadowOwnsMirroredSetting,
  MAX_RESEARCH_MODEL_TIMEOUT_SECONDS,
  MIN_FINITE_RESEARCH_MODEL_TIMEOUT_SECONDS,
  normalizeStoredPermissionMode,
  normalizeStoredRagAutoInject,
} from "../utils/mirrored-chat-settings";
import { retryablePatchAfterFailure } from "../utils/settings-retry";
import {
  isPresenceBumpQwen,
  migrateLegacyQwenDefaults,
  type QwenDefaultsMigration,
} from "../utils/qwen-defaults-migration";
import { DEFAULT_AUTO_COMPACT_ENABLED } from "../utils/auto-compaction";
import { preserveThinkingDefaultFromLoad } from "../lib/resolve-preserve-thinking-default";
import {
  THREAD_SCOPED_PARAM_KEYS,
  THREAD_SCOPED_SETTING_KEYS,
  type ThreadScopedSettingKey,
  type ThreadScopedSettings,
  hasThreadScopedSettings,
  isThreadOwnedSettingKey,
  isThreadScopedParamKey,
  normalizeSavedThreadScopedSettings,
  sanitizeThreadScopedSettings,
} from "../utils/thread-scoped-settings";
import {
  chatModelLifecycleGate,
  type ModelLifecycleLease,
  type ModelLifecyclePhase,
} from "../utils/model-lifecycle-gate";
import { shouldAdvanceQueuedSettingsEpoch } from "../utils/queued-settings-epoch";
import type { MmprojFallbackReason } from "../types/api";
import type { ResearchWebsitePolicy } from "../types/research";
import {
  CHAT_GPU_MEMORY_MODE_KEY,
  CHAT_SPECULATIVE_TYPE_KEY,
} from "./chat-runtime-keys";
import { useExternalProvidersStore } from "./external-providers-store";

export {
  CHAT_GPU_MEMORY_MODE_KEY,
  CHAT_SPECULATIVE_TYPE_KEY,
} from "./chat-runtime-keys";

export const CHAT_REASONING_ENABLED_KEY = "unsloth_chat_reasoning_enabled";
export const CHAT_TOOLS_ENABLED_KEY = "unsloth_chat_tools_enabled";
export const CHAT_CODE_TOOLS_ENABLED_KEY = "unsloth_chat_code_tools_enabled";
export const CHAT_IMAGE_TOOLS_ENABLED_KEY = "unsloth_chat_image_tools_enabled";
export const CHAT_DEEP_RESEARCH_ENABLED_KEY =
  "unsloth_chat_deep_research_enabled";
export const CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY =
  "unsloth_chat_deep_research_website_policy";
export const CHAT_DEEP_RESEARCH_MODEL_TIMEOUT_KEY =
  "unsloth_chat_deep_research_model_timeout";
export const CHAT_COLLAPSE_HTML_ARTIFACTS_KEY =
  "unsloth_chat_collapse_html_artifacts";
export const CHAT_ALLOW_ARTIFACT_NETWORK_ACCESS_KEY =
  "unsloth_chat_allow_artifact_network_access";
export const CHAT_SEARCH_IMAGES_KEY = "unsloth_chat_search_images";
export const CHAT_MCP_ENABLED_KEY = "unsloth_chat_mcp_enabled";
export const CHAT_CONFIRM_TOOL_CALLS_KEY = "unsloth_chat_confirm_tool_calls";
export const CHAT_EXPAND_QUANTIZATIONS_KEY =
  "unsloth_chat_expand_quantizations";
export const CHAT_SHOW_ALL_QUANTIZATIONS_KEY =
  "unsloth_chat_show_all_quantizations";
export const CHAT_SHOW_MEMORY_BAR_KEY = "unsloth_chat_show_memory_bar";
export const MODELS_FIT_ON_DEVICE_ONLY_KEY =
  "unsloth_models_fit_on_device_only";
export const CHAT_BYPASS_PERMISSIONS_KEY = "unsloth_chat_bypass_permissions";
export const CHAT_PERMISSION_MODE_KEY = "unsloth_chat_permission_mode";
export const CHAT_SANDBOX_LEVEL_KEY = "unsloth_chat_sandbox_level";

/** "ask" every call, "auto" high-risk only, "off" never, "full" no sandbox (session-only). */
export type PermissionMode = "ask" | "auto" | "off" | "full";
/** "high" adds the OS sandbox (bubblewrap, Seatbelt, MXC) to the software safeguards; "low" uses only those. */
export type SandboxLevel = "high" | "low";
export const CHAT_WEB_FETCH_TOOLS_ENABLED_KEY =
  "unsloth_chat_web_fetch_tools_enabled";
export const CHAT_RAG_SOURCE_KEY = "unsloth_chat_rag_source";
export const CHAT_RAG_MODE_KEY = "unsloth_chat_rag_mode";
export const CHAT_RAG_TOP_K_KEY = "unsloth_chat_rag_top_k";
export const CHAT_RAG_AUTOINJECT_KEY = "unsloth_chat_rag_autoinject";
export const CHAT_RAG_AUTOINJECT_MIN_SCORE_KEY =
  "unsloth_chat_rag_autoinject_min_score";
export const CHAT_RAG_OCR_KEY = "unsloth_chat_rag_ocr_scanned";
export const CHAT_RAG_CAPTION_KEY = "unsloth_chat_rag_caption_figures";
// Only model-agnostic modes persist: a saved drafter mode no-ops on models without one.
const PERSISTED_SPEC_MODES = new Set(["auto", "ngram", "off"]);

export type RagSource = { type: "thread" } | { type: "kb"; kbId: string };

export const PENDING_CHAT_ATTACHMENT_KEY = "__pending__";

let pendingAttachmentTargetClaim = 0;

export function readPendingAttachmentTargetClaim(): number {
  return pendingAttachmentTargetClaim;
}

export type RagMode = "hybrid" | "lexical" | "dense";

export const DEFAULT_RAG_SOURCE: RagSource = { type: "thread" };
export const DEFAULT_RAG_MODE: RagMode = "hybrid";
export const DEFAULT_RAG_TOP_K = 5;
export type RagAutoInject = "auto" | "on" | "off";
export const DEFAULT_RAG_AUTOINJECT: RagAutoInject = "auto";
export const DEFAULT_RAG_AUTOINJECT_MIN_SCORE = 0.7;
export const DEFAULT_RAG_OCR = true;
export const DEFAULT_RAG_CAPTION = false;
export const DEFAULT_RESEARCH_WEBSITE_POLICY: ResearchWebsitePolicy = {
  allowedDomains: [],
  blockedDomains: [],
};
export const DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS = 900;

/** The patch and run routes reject anything other than 0 or a finite budget with 400. */
function isSupportedResearchModelTimeout(value: number): boolean {
  if (!Number.isSafeInteger(value) || value < 0) return false;
  if (value > MAX_RESEARCH_MODEL_TIMEOUT_SECONDS) return false;
  return value === 0 || value >= MIN_FINITE_RESEARCH_MODEL_TIMEOUT_SECONDS;
}

function loadResearchModelTimeoutSeconds(): number {
  if (typeof window === "undefined") return DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS;
  try {
    const raw = window.localStorage.getItem(CHAT_DEEP_RESEARCH_MODEL_TIMEOUT_KEY);
    if (raw === null) return DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS;
    const value = Number(raw);
    return isSupportedResearchModelTimeout(value)
      ? value
      : DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS;
  } catch {
    return DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS;
  }
}

function loadResearchWebsitePolicy(): ResearchWebsitePolicy {
  if (typeof window === "undefined") return DEFAULT_RESEARCH_WEBSITE_POLICY;
  try {
    const parsed = JSON.parse(
      window.localStorage.getItem(CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY) || "{}",
    ) as Partial<ResearchWebsitePolicy>;
    return {
      allowedDomains: Array.isArray(parsed.allowedDomains)
        ? parsed.allowedDomains.filter(
            (value): value is string => typeof value === "string",
          )
        : [],
      blockedDomains: Array.isArray(parsed.blockedDomains)
        ? parsed.blockedDomains.filter(
            (value): value is string => typeof value === "string",
          )
        : [],
    };
  } catch {
    return DEFAULT_RESEARCH_WEBSITE_POLICY;
  }
}

function saveResearchWebsitePolicy(policy: ResearchWebsitePolicy): void {
  persistSetting(CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY, JSON.stringify(policy));
}

function loadRagSource(): RagSource {
  if (typeof window === "undefined") return DEFAULT_RAG_SOURCE;
  try {
    const raw = window.localStorage.getItem(CHAT_RAG_SOURCE_KEY);
    if (!raw) return DEFAULT_RAG_SOURCE;
    const parsed = JSON.parse(raw) as RagSource;
    if (parsed?.type === "kb" && typeof parsed.kbId === "string") {
      return { type: "kb", kbId: parsed.kbId };
    }
    if (parsed?.type === "thread") return { type: "thread" };
    return DEFAULT_RAG_SOURCE;
  } catch {
    return DEFAULT_RAG_SOURCE;
  }
}

function saveRagSource(value: RagSource): void {
  persistSetting(CHAT_RAG_SOURCE_KEY, JSON.stringify(value));
}

function loadProjectAttachmentTarget(): ProjectAttachmentTarget {
  return normalizeProjectAttachmentTarget(
    loadString(CHAT_PROJECT_ATTACHMENT_TARGET_KEY, DEFAULT_PROJECT_ATTACHMENT_TARGET),
  );
}

function loadRagMode(): RagMode {
  const raw = loadString(CHAT_RAG_MODE_KEY, DEFAULT_RAG_MODE);
  return raw === "lexical" || raw === "dense" ? raw : "hybrid";
}

function loadRagAutoInject(): RagAutoInject {
  return normalizeStoredRagAutoInject(
    loadString(CHAT_RAG_AUTOINJECT_KEY, DEFAULT_RAG_AUTOINJECT),
  );
}

function loadRagTopK(): number {
  if (typeof window === "undefined") return DEFAULT_RAG_TOP_K;
  try {
    const raw = window.localStorage.getItem(CHAT_RAG_TOP_K_KEY);
    if (raw === null) return DEFAULT_RAG_TOP_K;
    const parsed = Number.parseInt(raw, 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_RAG_TOP_K;
  } catch {
    return DEFAULT_RAG_TOP_K;
  }
}

// Preserves a stored 0 (score floors can legitimately be 0).
function loadRagNumber(
  key: string,
  fallback: number,
  {
    min,
    max,
    integer = false,
  }: { min: number; max: number; integer?: boolean },
): number {
  if (typeof window === "undefined") return fallback;
  try {
    const raw = window.localStorage.getItem(key);
    if (raw === null) return fallback;
    const parsed = integer ? Number.parseInt(raw, 10) : Number.parseFloat(raw);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  } catch {
    return fallback;
  }
}

// External picks are `external::<providerId>::<modelId>`; the backend only mirrors local ones.
const LAST_EXTERNAL_CHECKPOINT_KEY = "unsloth_chat_last_external_checkpoint";

function loadLastExternalCheckpoint(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const value = window.localStorage.getItem(LAST_EXTERNAL_CHECKPOINT_KEY);
    return isExternalModelId(value) ? value : null;
  } catch {
    return null;
  }
}

/** normalizeModelIdentity, not toLowerCase: POSIX paths are case-sensitive. */
function isOpaqueModelRef(modelId: string): boolean {
  return isExternalModelId(modelId) || isOllamaManifestRef(modelId);
}

function sameCheckpointIdentity(
  left: string | null | undefined,
  right: string | null | undefined,
): boolean {
  if (!(left && right)) {
    return false;
  }
  // Opaque ids compare exactly; normalizeModelIdentity would fold them.
  if (isOpaqueModelRef(left) || isOpaqueModelRef(right)) {
    return left === right;
  }
  return normalizeModelIdentity(left) === normalizeModelIdentity(right);
}

// Chat settings are installation-wide, so an unowned checkpoint may not claim a global snapshot.
let unownedCheckpointBeforeHydration: string | null = null;

function saveLastExternalCheckpoint(value: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (value && isExternalModelId(value)) {
      window.localStorage.setItem(LAST_EXTERNAL_CHECKPOINT_KEY, value);
    } else {
      window.localStorage.removeItem(LAST_EXTERNAL_CHECKPOINT_KEY);
    }
  } catch {
    // Storage failures are non-fatal; the selection just will not survive a refresh.
  }
}

export type ReasoningStyle =
  | "enable_thinking"
  | "reasoning_effort"
  | "enable_thinking_effort";
export type DiffusionCanvasFrame = {
  block: number;
  step: number;
  total: number;
  text: string;
};
export type PendingImageEditReference = {
  threadId: string | null;
  openaiImageGenerationCallId: string;
  openaiResponseId?: string;
  openaiReasoningItem?: unknown;
};
export type LoadingModelPick = {
  id: string;
  ggufVariant: string | null;
  nativePathToken: string | null;
};
export type ReasoningEffort =
  | "none"
  | "minimal"
  | "low"
  | "medium"
  | "high"
  | "max"
  | "xhigh";

let hasShownSettingsPersistenceWarning = false;
let customPresetsMutationVersion = 0;
let activePresetMutationVersion = 0;
let activePresetSourceMutationVersion = 0;
let settingsHydrationPromise: Promise<void> | null = null;

function warnSettingsPersistenceFailure(): void {
  if (hasShownSettingsPersistenceWarning) {
    return;
  }
  hasShownSettingsPersistenceWarning = true;
  toast.warning("Chat settings could not be persisted", {
    description: "Your changes apply now, but may reset after refresh.",
  });
}

type SettingsPatch = Parameters<typeof savePersistedChatSettingsPatch>[0];

const SETTINGS_DEBOUNCE_MS = 400;
let pendingPatch: SettingsPatch = {};
let pendingTimer: ReturnType<typeof setTimeout> | null = null;
let inflightFlush: Promise<void> = Promise.resolve();

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Replaced whole, not merged: a merged `kbId` is forbidden by the backend's thread variant.
const ATOMIC_SETTING_KEYS = new Set<string>(["ragSource"]);

// Merged one level deeper so two edits to different fields in one debounce both survive.
const NESTED_MAP_SETTING_KEYS = new Set<string>(["inferenceParamsByModel"]);

function mergePatch(into: SettingsPatch, more: SettingsPatch): void {
  for (const [key, value] of Object.entries(more)) {
    const intoAny = into as Record<string, unknown>;
    const prev = intoAny[key];
    if (ATOMIC_SETTING_KEYS.has(key)) {
      intoAny[key] = value;
      continue;
    }
    if (!isPlainObject(prev) || !isPlainObject(value)) {
      intoAny[key] = value;
      continue;
    }
    if (!NESTED_MAP_SETTING_KEYS.has(key)) {
      intoAny[key] = { ...prev, ...value };
      continue;
    }
    const merged: Record<string, unknown> = { ...prev };
    for (const [id, entry] of Object.entries(value)) {
      const existing = merged[id];
      merged[id] =
        isPlainObject(existing) && isPlainObject(entry)
          ? { ...existing, ...entry }
          : entry;
    }
    intoAny[key] = merged;
  }
}

async function flushSettingsPatch(keepalive = false): Promise<void> {
  if (Object.keys(pendingPatch).length === 0) return;
  const patch = pendingPatch;
  pendingPatch = {};
  try {
    await savePersistedChatSettingsPatch(patch, { keepalive });
  } catch (error) {
    // extra="forbid" refuses the whole body on one bad field; drop the fields the server named.
    const { patch: retryable, progressed } = retryablePatchAfterFailure(
      patch,
      error,
    );
    const retryPatch: SettingsPatch = {};
    mergePatch(retryPatch, retryable);
    mergePatch(retryPatch, pendingPatch);
    pendingPatch = retryPatch;
    warnSettingsPersistenceFailure();
    if (progressed && !keepalive && Object.keys(pendingPatch).length > 0) {
      scheduleSettingsFlush();
    }
  }
}

let unsettledFlushes = 0;

function settingsWritesAreDrained(): boolean {
  return (
    pendingTimer === null &&
    Object.keys(pendingPatch).length === 0 &&
    unsettledFlushes === 0
  );
}

function enqueueSettingsFlush(): Promise<void> {
  unsettledFlushes += 1;
  inflightFlush = inflightFlush
    .catch(() => undefined)
    .then(() => flushSettingsPatch())
    .finally(() => {
      unsettledFlushes -= 1;
    });
  return inflightFlush;
}

function scheduleSettingsFlush(): void {
  if (pendingTimer !== null) clearTimeout(pendingTimer);
  pendingTimer = setTimeout(() => {
    pendingTimer = null;
    void enqueueSettingsFlush();
  }, SETTINGS_DEBOUNCE_MS);
}

function saveSettingsPatch(patch: SettingsPatch): void {
  mergePatch(pendingPatch, patch);
  scheduleSettingsFlush();
}

// A wedged PATCH must not hold a send open past this.
const SETTINGS_FLUSH_TIMEOUT_MS = 2000;

/** The backend reads some settings from SQLite at call time, so flush before sending. */
export async function flushPendingChatSettings(): Promise<void> {
  const queued = pendingTimer !== null || Object.keys(pendingPatch).length > 0;
  // The debounce may have handed its patch to an unanswered request.
  if (!queued && unsettledFlushes === 0) return;
  if (pendingTimer !== null) {
    clearTimeout(pendingTimer);
    pendingTimer = null;
  }
  if (queued) void enqueueSettingsFlush();
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      inflightFlush.catch(() => undefined),
      new Promise<void>((resolve) => {
        timer = setTimeout(resolve, SETTINGS_FLUSH_TIMEOUT_MS);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

// keepalive lets the PUT outlive the unload.
function flushSettingsOnPageHidden(terminal: boolean): void {
  if (pendingTimer !== null) clearTimeout(pendingTimer);
  // Terminal events only: the beacon PATCHes the row directly, and a missing row answers 404.
  const sentNewest = new Set<string>();
  const flushedThreadId = threadSettingsWriteThreadId;
  flushThreadScopedSettingsWrite(terminal);
  if (terminal && flushedThreadId !== null) sentNewest.add(flushedThreadId);
  // Effect cleanup is not guaranteed during unload, so send held edits from here.
  if (terminal) {
    const heldThreadId = pendingPairingThreadId;
    void commitHeldThreadScopedEditsToTheirThread(true);
    if (heldThreadId !== null) sentNewest.add(heldThreadId);
    beaconUnsettledThreadSettingsWrites(sentNewest);
  }
  drainPreHydrationPatch();
  if (Object.keys(pendingPatch).length === 0) return;
  // Counted: a keepalive PUT still in flight could land after the compare-and-set.
  unsettledFlushes += 1;
  inflightFlush = inflightFlush
    .catch(() => undefined)
    .then(() => flushSettingsPatch(true))
    .finally(() => {
      unsettledFlushes -= 1;
    });
}

if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", () => flushSettingsOnPageHidden(true));
  // beforeunload never fires on discarded mobile tabs, so also use pagehide/visibilitychange.
  window.addEventListener("pagehide", () => flushSettingsOnPageHidden(true));
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") flushSettingsOnPageHidden(false);
  });
}

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function readStorageValue(key: string): string | null {
  if (!canUseStorage()) return null;
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writeStorageValue(key: string, raw: string): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(key, raw);
  } catch {
    // Keep the in-memory setting when storage is unavailable.
  }
}

type MirroredSettingCodec = {
  encode: (value: unknown) => string;
  decode: (raw: string) => unknown;
  readForBackfill?: () => unknown;
};

const BOOLEAN_SETTING: MirroredSettingCodec = {
  encode: (value) => (value ? "true" : "false"),
  decode: (raw) => (raw === "true" ? true : raw === "false" ? false : undefined),
};

const STRING_SETTING: MirroredSettingCodec = {
  encode: (value) => String(value),
  decode: (raw) => raw,
};

const NUMBER_SETTING: MirroredSettingCodec = {
  encode: (value) => String(value),
  decode: (raw) => {
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? parsed : undefined;
  },
};

const JSON_SETTING: MirroredSettingCodec = {
  encode: (value) => JSON.stringify(value),
  decode: (raw) => {
    try {
      return JSON.parse(raw) as unknown;
    } catch {
      return undefined;
    }
  },
};

const RAG_AUTOINJECT_SETTING: MirroredSettingCodec = {
  encode: STRING_SETTING.encode,
  decode: normalizeStoredRagAutoInject,
};

const MIRRORED_SETTINGS = {
  reasoningEnabled: {
    storageKey: CHAT_REASONING_ENABLED_KEY,
    ...BOOLEAN_SETTING,
  },
  toolsEnabled: { storageKey: CHAT_TOOLS_ENABLED_KEY, ...BOOLEAN_SETTING },
  codeToolsEnabled: {
    storageKey: CHAT_CODE_TOOLS_ENABLED_KEY,
    ...BOOLEAN_SETTING,
  },
  imageToolsEnabled: {
    storageKey: CHAT_IMAGE_TOOLS_ENABLED_KEY,
    ...BOOLEAN_SETTING,
  },
  webFetchToolsEnabled: {
    storageKey: CHAT_WEB_FETCH_TOOLS_ENABLED_KEY,
    ...BOOLEAN_SETTING,
  },
  deepResearchEnabled: {
    storageKey: CHAT_DEEP_RESEARCH_ENABLED_KEY,
    ...BOOLEAN_SETTING,
  },
  researchWebsitePolicy: {
    storageKey: CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY,
    ...JSON_SETTING,
  },
  researchModelTimeoutSeconds: {
    storageKey: CHAT_DEEP_RESEARCH_MODEL_TIMEOUT_KEY,
    ...NUMBER_SETTING,
  },
  collapseHtmlArtifacts: {
    storageKey: CHAT_COLLAPSE_HTML_ARTIFACTS_KEY,
    ...BOOLEAN_SETTING,
  },
  allowArtifactNetworkAccess: {
    storageKey: CHAT_ALLOW_ARTIFACT_NETWORK_ACCESS_KEY,
    ...BOOLEAN_SETTING,
  },
  searchImages: { storageKey: CHAT_SEARCH_IMAGES_KEY, ...BOOLEAN_SETTING },
  mcpEnabledForChat: { storageKey: CHAT_MCP_ENABLED_KEY, ...BOOLEAN_SETTING },
  confirmToolCalls: {
    storageKey: CHAT_CONFIRM_TOOL_CALLS_KEY,
    ...BOOLEAN_SETTING,
  },
  permissionMode: {
    storageKey: CHAT_PERMISSION_MODE_KEY,
    ...STRING_SETTING,
    readForBackfill: () =>
      readStorageValue(CHAT_PERMISSION_MODE_KEY) !== null ||
      loadOptionalBool(CHAT_CONFIRM_TOOL_CALLS_KEY) !== null
        ? loadPermissionMode()
        : undefined,
  },
  sandboxLevel: { storageKey: CHAT_SANDBOX_LEVEL_KEY, ...STRING_SETTING },
  ragSource: { storageKey: CHAT_RAG_SOURCE_KEY, ...JSON_SETTING },
  ragMode: { storageKey: CHAT_RAG_MODE_KEY, ...STRING_SETTING },
  ragTopK: { storageKey: CHAT_RAG_TOP_K_KEY, ...NUMBER_SETTING },
  ragAutoInject: {
    storageKey: CHAT_RAG_AUTOINJECT_KEY,
    ...RAG_AUTOINJECT_SETTING,
  },
  ragAutoInjectMinScore: {
    storageKey: CHAT_RAG_AUTOINJECT_MIN_SCORE_KEY,
    ...NUMBER_SETTING,
  },
  ragOcrScanned: { storageKey: CHAT_RAG_OCR_KEY, ...BOOLEAN_SETTING },
  ragCaptionFigures: { storageKey: CHAT_RAG_CAPTION_KEY, ...BOOLEAN_SETTING },
  speculativeType: { storageKey: CHAT_SPECULATIVE_TYPE_KEY, ...STRING_SETTING },
  gpuMemoryMode: { storageKey: CHAT_GPU_MEMORY_MODE_KEY, ...STRING_SETTING },
  expandQuantizations: {
    storageKey: CHAT_EXPAND_QUANTIZATIONS_KEY,
    ...BOOLEAN_SETTING,
  },
  showAllQuantizations: {
    storageKey: CHAT_SHOW_ALL_QUANTIZATIONS_KEY,
    ...BOOLEAN_SETTING,
  },
  fitOnDeviceOnly: {
    storageKey: MODELS_FIT_ON_DEVICE_ONLY_KEY,
    ...BOOLEAN_SETTING,
  },
} satisfies Partial<
  Record<ScalarSettingKey, { storageKey: string } & MirroredSettingCodec>
>;

type MirroredSettingKey = keyof typeof MIRRORED_SETTINGS;

const MIRRORED_SETTING_BY_STORAGE_KEY: ReadonlyMap<
  string,
  { field: MirroredSettingKey } & MirroredSettingCodec
> = new Map(
  Object.entries(MIRRORED_SETTINGS).map(([field, setting]) => [
    setting.storageKey,
    { field: field as MirroredSettingKey, ...setting },
  ]),
);

let mirroredSettingsHydrated = false;
let preHydrationPatch: SettingsPatch | null = null;

/** Edits before the initial GET are held and replayed after hydration to avoid a race. */
function mirrorSettingToBackend(key: string, raw: string): void {
  const setting = MIRRORED_SETTING_BY_STORAGE_KEY.get(key);
  if (!setting) return;
  const value = setting.decode(raw);
  if (value === undefined) return;
  scalarSettingMutationVersions[setting.field] += 1;
  const patch = { [setting.field]: value } as SettingsPatch;
  if (!mirroredSettingsHydrated) {
    preHydrationPatch ??= {};
    mergePatch(preHydrationPatch, patch);
    return;
  }
  saveSettingsPatch(patch);
}

function drainPreHydrationPatch(): void {
  if (!preHydrationPatch) return;
  mergePatch(pendingPatch, preHydrationPatch);
  preHydrationPatch = null;
}

function flushPreHydrationSettings(): void {
  if (!preHydrationPatch) return;
  const patch = preHydrationPatch;
  preHydrationPatch = null;
  saveSettingsPatch(patch);
}

function persistSetting(key: string, raw: string): void {
  const mirrored = MIRRORED_SETTING_BY_STORAGE_KEY.get(key);
  const writeGlobal = () => {
    if (!mirroredSettingsHydrated || readStorageValue(key) !== raw) {
      mirrorSettingToBackend(key, raw);
    }
    writeStorageValue(key, raw);
  };
  if (mirrored && captureThreadScopedEdit(mirrored.field, writeGlobal)) return;
  writeGlobal();
}

const THREAD_SETTINGS_DEBOUNCE_MS = 400;

let threadScopedSettingsThreadId: string | null = null;
let activeThreadScopedSettings: ThreadScopedSettings | null = null;
// Captured on entering a thread: edits with a chat open must not move the defaults.
let globalThreadScopedDefaults: ThreadScopedSettings | null = null;
let threadSettingsWriteTimer: ReturnType<typeof setTimeout> | null = null;
let threadSettingsWriteThreadId: string | null = null;
let threadSettingsWriteSnapshot: ThreadScopedSettings | null = null;

function readThreadScopedValue(
  state: ChatRuntimeStore,
  key: ThreadScopedSettingKey,
): unknown {
  return isThreadScopedParamKey(key)
    ? state.params[key]
    : (state as Record<string, unknown>)[key];
}

function readThreadScopedSettings(
  state: ChatRuntimeStore,
): ThreadScopedSettings {
  const source: Record<string, unknown> = {};
  for (const key of THREAD_SCOPED_SETTING_KEYS) {
    source[key] = readThreadScopedValue(state, key);
  }
  // Drops "full": a stored bypass would come back without the warning dialog.
  return sanitizeThreadScopedSettings(source);
}

export function threadScopedOverride<K extends ThreadScopedSettingKey>(
  key: K,
): ThreadScopedSettings[K] | undefined {
  if (key === "reasoningEffort" && activeThreadScopedSettings?.reasoningEffort !== undefined) {
    return activeThreadScopedSettings.reasoningEffort as ThreadScopedSettings[K];
  }
  if (
    threadSettingsWriteThreadId !== null &&
    threadSettingsWriteThreadId === threadScopedSettingsThreadId
  ) {
    if (threadSettingsWriteSnapshot !== null) {
      if (threadSettingsWriteSnapshot[key] !== undefined) {
        return threadSettingsWriteSnapshot[key];
      }
    } else {
      const live = readThreadScopedSettings(useChatRuntimeStore.getState());
      if (live[key] !== undefined) return live[key];
    }
  }
  return activeThreadScopedSettings?.[key];
}

const explicitlyEditedThreadFields = new Set<string>();

/** Fields a provider constraint moved without persisting; no snapshot may save them. */
const constraintSuppressedThreadFields = new Set<string>();

const CONSTRAINT_SUPPRESSIBLE_KEYS = ["reasoningEnabled", "toolsEnabled"] as const;

function noteConstraintSuppressedThreadField(
  field: (typeof CONSTRAINT_SUPPRESSIBLE_KEYS)[number],
): void {
  if (useChatRuntimeStore.getState().activeThreadId === null) return;
  constraintSuppressedThreadFields.add(field);
}

function keepsStoredValueUnderConstraint(
  key: (typeof CONSTRAINT_SUPPRESSIBLE_KEYS)[number],
  threadId: string,
  settings: ThreadScopedSettings,
): boolean {
  return (
    threadId === threadScopedSettingsThreadId &&
    constraintSuppressedThreadFields.has(key) &&
    !explicitlyEditedThreadFields.has(key) &&
    typeof activeThreadScopedSettings?.[key] === "boolean" &&
    settings[key] !== activeThreadScopedSettings[key]
  );
}

/** A per-model effort pin is shown in the store but belongs to the model, not the chat. */
function pinOwnsLiveReasoningEffort(state: ChatRuntimeStore): boolean {
  return (
    externalReasoningTakesEffort(state) &&
    pinnedReasoningEffort(
      state.params.checkpoint,
      state.reasoningEffortLevels,
    ) !== null
  );
}

function reasoningEffortOnRecord(): ReasoningEffort | undefined {
  return (
    activeThreadScopedSettings?.reasoningEffort ??
    globalThreadScopedDefaults?.reasoningEffort ??
    effortDisplacedByPin ??
    undefined
  );
}

const CLAMPED_PILL_KEYS = [
  "toolsEnabled",
  "codeToolsEnabled",
  "imageToolsEnabled",
  "webFetchToolsEnabled",
] as const;
type ClampedPillKey = (typeof CLAMPED_PILL_KEYS)[number];

/** Sampling keys share `params` with the model's values, so capture edits at edit time. */
function heldThreadScopedParamValue(key: string): unknown {
  for (let i = heldThreadScopedEdits.length - 1; i >= 0; i -= 1) {
    if (heldThreadScopedEdits[i].field === key) {
      return heldThreadScopedEdits[i].value;
    }
  }
  return undefined;
}

/** `??` is wrong here: a cleared `seed` is null and that is the chat's own choice. */
function firstSetThreadScopedValue(...values: unknown[]): unknown {
  return values.find((value) => value !== undefined);
}

function restoreThreadScopedParams(params: InferenceParams): InferenceParams {
  const kept: Record<string, unknown> = {};
  for (const key of THREAD_SCOPED_PARAM_KEYS) {
    const held = firstSetThreadScopedValue(
      heldThreadScopedParamValue(key),
      threadScopedOverride(key),
    );
    if (held === undefined || isSameThreadScopedValue(held, params[key])) {
      continue;
    }
    kept[key] = held;
  }
  return hasKeys(kept) ? { ...params, ...kept } : params;
}

function withoutActiveThreadParams(
  state: ChatRuntimeStore,
  params: InferenceParams,
): InferenceParams {
  if (threadScopedSettingsThreadId === null && pendingPairingThreadId === null) {
    return params;
  }
  const remembered = params.checkpoint
    ? state.paramsByModel[params.checkpoint]
    : undefined;
  const restored: Record<string, unknown> = {};
  for (const key of THREAD_SCOPED_PARAM_KEYS) {
    const held = heldThreadScopedParamValue(key);
    if (held === undefined && threadScopedOverride(key) === undefined) continue;
    const own = firstSetThreadScopedValue(
      remembered?.[key],
      globalThreadScopedDefaults?.[key],
      held !== undefined ? pairingWindowDefaults?.[key] : undefined,
    );
    if (own === undefined || isSameThreadScopedValue(own, params[key])) {
      continue;
    }
    restored[key] = own;
  }
  return hasKeys(restored) ? { ...params, ...restored } : params;
}

function withoutCapturedThreadEdits(
  changedParams: PersistedInferenceParams,
  fromModelDefaults: boolean,
): PersistedInferenceParams {
  const shared: PersistedInferenceParams = {};
  for (const [key, value] of Object.entries(changedParams)) {
    if (
      isThreadScopedParamKey(key) &&
      !fromModelDefaults &&
      // By value: inside the updater a read-back would find the pre-edit value.
      captureThreadScopedEdit(key, null, value)
    ) {
      continue;
    }
    (shared as Record<string, unknown>)[key] = value;
  }
  return shared;
}

function noteThreadScopedDefaults(shared: PersistedInferenceParams): void {
  let next: Record<string, unknown> | null = null;
  for (const [key, value] of Object.entries(shared)) {
    if (!isThreadScopedParamKey(key)) continue;
    if (isHeldThreadScopedField(key)) {
      hydratedDefaultsByHeldField.set(key, value);
    }
    if (globalThreadScopedDefaults === null) continue;
    next ??= { ...globalThreadScopedDefaults };
    next[key] = value;
  }
  if (next !== null) globalThreadScopedDefaults = next as ThreadScopedSettings;
}

function isSameThreadScopedValue(next: unknown, current: unknown): boolean {
  if (Object.is(next, current)) return true;
  if (isPlainObject(next) && isPlainObject(current)) {
    return next.type === current.type && next.kbId === current.kbId;
  }
  return false;
}

function buildThreadScopedSnapshot(
  threadId: string,
  snapshot: ThreadScopedSettings | null,
): ThreadScopedSettings {
  const settings =
    snapshot ?? readThreadScopedSettings(useChatRuntimeStore.getState());
  // Keep the stored level: the sanitizer drops a live "full", which would erase it.
  if (
    settings.permissionMode === undefined &&
    threadId === threadScopedSettingsThreadId &&
    activeThreadScopedSettings?.permissionMode !== undefined
  ) {
    settings.permissionMode = activeThreadScopedSettings.permissionMode;
  }
  if (
    threadId === threadScopedSettingsThreadId &&
    !explicitlyEditedThreadFields.has("deepResearchEnabled") &&
    activeThreadScopedSettings?.deepResearchEnabled === true &&
    settings.deepResearchEnabled !== true &&
    (externalCheckpointRefusesDeepResearch(
      useChatRuntimeStore.getState().params.checkpoint,
    ) ||
      useChatRuntimeStore.getState().incognito)
  ) {
    settings.deepResearchEnabled = true;
  }
  // A model that cannot stop thinking forces it on; do not persist that true.
  if (
    threadId === threadScopedSettingsThreadId &&
    !explicitlyEditedThreadFields.has("reasoningEnabled") &&
    activeThreadScopedSettings?.reasoningEnabled === false &&
    settings.reasoningEnabled !== false &&
    useChatRuntimeStore.getState().reasoningAlwaysOn
  ) {
    settings.reasoningEnabled = false;
  }
  if (threadId === threadScopedSettingsThreadId && pinHoldsLiveEffort()) {
    const onRecord = reasoningEffortOnRecord();
    if (onRecord !== undefined) settings.reasoningEffort = onRecord;
  }
  if (threadId === threadScopedSettingsThreadId) {
    const live = useChatRuntimeStore.getState();
    const modelLoaded = !!live.params.checkpoint && !live.modelLoading;
    const capable: Record<ClampedPillKey, boolean> = {
      toolsEnabled: live.supportsTools || live.supportsBuiltinWebSearch,
      codeToolsEnabled: live.supportsTools || live.supportsBuiltinCodeExecution,
      imageToolsEnabled: live.supportsBuiltinImageGeneration,
      webFetchToolsEnabled: live.supportsBuiltinWebFetch,
    };
    for (const key of CLAMPED_PILL_KEYS) {
      if (
        modelLoaded &&
        !capable[key] &&
        !explicitlyEditedThreadFields.has(key) &&
        activeThreadScopedSettings?.[key] === true &&
        settings[key] !== true
      ) {
        settings[key] = true;
      }
    }
  }
  if (keepsStoredValueUnderConstraint("reasoningEnabled", threadId, settings)) {
    settings.reasoningEnabled = activeThreadScopedSettings?.reasoningEnabled;
  }
  if (keepsStoredValueUnderConstraint("toolsEnabled", threadId, settings)) {
    settings.toolsEnabled = activeThreadScopedSettings?.toolsEnabled;
  }
  if (threadId === threadScopedSettingsThreadId) {
    activeThreadScopedSettings = settings;
  }
  explicitlyEditedThreadFields.clear();
  return settings;
}

const THREAD_SETTINGS_REPLAY_KEY = "unsloth_chat_thread_settings_replay";
const THREAD_SETTINGS_REPLAY_TIMEOUT_MS = 10_000;

/** The beacon cannot await and a row being created answers 404, so keep for replay. */
function rememberThreadSettingsForReplay(
  threadId: string,
  body: Record<string, unknown>,
): void {
  if (!canUseStorage()) return;
  try {
    const raw = localStorage.getItem(THREAD_SETTINGS_REPLAY_KEY);
    const pending = raw ? (JSON.parse(raw) as Record<string, unknown>) : {};
    pending[threadId] = body;
    localStorage.setItem(THREAD_SETTINGS_REPLAY_KEY, JSON.stringify(pending));
  } catch {
    // A full or unavailable store just means no replay; the beacon may still land.
  }
}

/** Safe to always run: the body carries its seq, so a landed write is refused. */
export function replayUnconfirmedThreadSettings(): void {
  // Once per session: sending each body twice would race two writes with the same seq.
  if (threadSettingsReplayStarted) return;
  threadSettingsReplayStarted = true;
  if (!canUseStorage()) return;
  let pending: Record<string, unknown> = {};
  try {
    const raw = localStorage.getItem(THREAD_SETTINGS_REPLAY_KEY);
    if (!raw) return;
    pending = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    localStorage.removeItem(THREAD_SETTINGS_REPLAY_KEY);
    return;
  }
  const sent: Promise<unknown>[] = [];
  for (const [threadId, body] of Object.entries(pending)) {
    // Bounded: every settings write waits on these.
    const timeout = new AbortController();
    const timer = setTimeout(() => timeout.abort(), THREAD_SETTINGS_REPLAY_TIMEOUT_MS);
    const request = authFetch(`/api/chat/threads/${encodeURIComponent(threadId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: timeout.signal,
    })
      .finally(() => clearTimeout(timer))
      // authFetch resolves for 404 and 5xx too, so check res.ok.
      .then((res) => {
        // Only forget this exact body: a newer one may have been stored meanwhile.
        if (res.ok) forgetReplayedThreadSettings(threadId, body);
      })
      .catch(() => undefined);
    sent.push(request);
  }
  // Every write waits on this: the replay uses the previous writer id and could revert edits.
  threadSettingsReplaySettled = Promise.all(sent).then(() => undefined);
}

let threadSettingsReplaySettled: Promise<void> = Promise.resolve();
let threadSettingsReplayStarted = false;

function forgetReplayedThreadSettings(
  threadId: string,
  expected?: unknown,
): void {
  if (!canUseStorage()) return;
  try {
    const raw = localStorage.getItem(THREAD_SETTINGS_REPLAY_KEY);
    if (!raw) return;
    const pending = JSON.parse(raw) as Record<string, unknown>;
    if (
      expected !== undefined &&
      JSON.stringify(pending[threadId]) !== JSON.stringify(expected)
    ) {
      return;
    }
    delete pending[threadId];
    if (Object.keys(pending).length === 0) {
      localStorage.removeItem(THREAD_SETTINGS_REPLAY_KEY);
    } else {
      localStorage.setItem(THREAD_SETTINGS_REPLAY_KEY, JSON.stringify(pending));
    }
  } catch {
    // Leaving it behind only costs one more replay next time.
  }
}

function sendThreadScopedSettingsBeacon(
  threadId: string,
  snapshot: ThreadScopedSettings | null,
  merge = false,
): void {
  // A merge for a never-read chat; a full replacement would erase the rest of its row.
  const body = merge
    ? {
        settingsPatch: snapshot,
        settingsSeq: nextThreadSettingsSeq(),
        settingsWriter: threadSettingsWriter,
      }
    : {
        settings: buildThreadScopedSnapshot(threadId, snapshot),
        settingsSeq: nextThreadSettingsSeq(),
        settingsWriter: threadSettingsWriter,
      };
  // Stand down queued and in-flight writes so an older one cannot land after the beacon.
  takeThreadSettingsWriteTicket(threadId);
  threadSettingsWriteAborts.get(threadId)?.abort();
  rememberThreadSettingsForReplay(threadId, body);
  void authFetch(`/api/chat/threads/${encodeURIComponent(threadId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    keepalive: true,
  }).catch(() => undefined);
}

// One chain per thread: writes replace settings_json, so they must be ordered.
const threadSettingsWriteChains = new Map<string, Promise<unknown>>();
const threadSettingsWriteTickets = new Map<string, number>();

const threadSettingsWriteAborts = new Map<string, AbortController>();

/** Writer id plus a counter (never a clock) lets the server refuse this tab's older writes. */
const threadSettingsWriter = crypto.randomUUID();
let lastThreadSettingsSeq = 0;

function nextThreadSettingsSeq(): number {
  lastThreadSettingsSeq += 1;
  return lastThreadSettingsSeq;
}

function takeThreadSettingsWriteTicket(threadId: string): number {
  const ticket = (threadSettingsWriteTickets.get(threadId) ?? 0) + 1;
  threadSettingsWriteTickets.set(threadId, ticket);
  return ticket;
}

function writeThreadScopedSettings(
  threadId: string,
  snapshot: ThreadScopedSettings | null,
): Promise<boolean> {
  const settings = buildThreadScopedSnapshot(threadId, snapshot);
  // The seq marks when the edit happened, not when the request was sent.
  const settingsSeq = nextThreadSettingsSeq();
  const ticket = takeThreadSettingsWriteTicket(threadId);
  const previous = threadSettingsWriteChains.get(threadId) ?? Promise.resolve();
  const next = previous
    .catch(() => undefined)
    // Previous session's replays go first: their writer id differs so nothing orders them.
    .then(() => threadSettingsReplaySettled)
    .catch(() => undefined)
    .then(async () => {
      // Superseded while queued: sending would undo the newer snapshot.
      if ((threadSettingsWriteTickets.get(threadId) ?? ticket) !== ticket) {
        return true;
      }
      const controller = new AbortController();
      threadSettingsWriteAborts.set(threadId, controller);
      try {
        const { updateStoredChatThread } = await import(
          "../utils/chat-history-storage"
        );
        await updateStoredChatThread(
          threadId,
          { settings, settingsSeq, settingsWriter: threadSettingsWriter },
          { signal: controller.signal },
        );
        forgetReplayedThreadSettings(threadId);
        return true;
      } catch {
        if (!controller.signal.aborted) warnSettingsPersistenceFailure();
        return controller.signal.aborted;
      } finally {
        if (threadSettingsWriteAborts.get(threadId) === controller) {
          threadSettingsWriteAborts.delete(threadId);
        }
      }
    })
    .finally(() => {
      if (threadSettingsWriteChains.get(threadId) === next) {
        threadSettingsWriteChains.delete(threadId);
        threadSettingsWriteTickets.delete(threadId);
      }
    });
  threadSettingsWriteChains.set(threadId, next);
  return next;
}

/** Wait for in-flight and debounced writes so a GET does not read a pre-edit snapshot. */
export async function settleThreadScopedSettingsForCopy(
  threadId: string,
): Promise<void> {
  if (pendingPairingThreadId === threadId) {
    await commitHeldThreadScopedEditsToTheirThread();
  }
  if (!(await awaitThreadScopedSettingsWrite(threadId))) {
    throw new Error("This chat's settings could not be saved before copying it");
  }
}

export async function awaitThreadScopedSettingsWrite(
  threadId: string,
): Promise<boolean> {
  if (threadSettingsWriteThreadId === threadId) {
    flushThreadScopedSettingsWrite();
  }
  const chain = threadSettingsWriteChains.get(threadId);
  if (chain === undefined) return true;
  const landed = await chain.catch(() => false);
  return landed !== false;
}

export async function awaitStartedThreadScopedSettingsWrites(): Promise<void> {
  // Repeats because a settled chain can leave a newer one; bounded so it cannot hang.
  for (let pass = 0; pass < 20 && threadSettingsWriteChains.size > 0; pass += 1) {
    await Promise.allSettled([...threadSettingsWriteChains.values()]);
  }
}

function flushThreadScopedSettingsWrite(keepalive = false): void {
  if (threadSettingsWriteTimer !== null) {
    clearTimeout(threadSettingsWriteTimer);
    threadSettingsWriteTimer = null;
  }
  const threadId = threadSettingsWriteThreadId;
  const snapshot = threadSettingsWriteSnapshot;
  threadSettingsWriteThreadId = null;
  threadSettingsWriteSnapshot = null;
  if (threadId === null) return;
  if (keepalive) {
    sendThreadScopedSettingsBeacon(threadId, snapshot);
    return;
  }
  trackUnsettledThreadSettingsWrite(threadId, snapshot);
}

/** Some browsers fire visibilitychange(hidden) then pagehide; keep it for the terminal resend. */
function trackUnsettledThreadSettingsWrite(
  threadId: string,
  snapshot: ThreadScopedSettings | null,
): void {
  // Identity, not value: two edits can both carry a null snapshot.
  const entry: UnsettledThreadSettingsWrite = { snapshot };
  unsettledThreadSettingsWrites.set(threadId, entry);
  void writeThreadScopedSettings(threadId, snapshot).then((landed) => {
    if (landed && unsettledThreadSettingsWrites.get(threadId) === entry) {
      unsettledThreadSettingsWrites.delete(threadId);
    }
  });
}

type UnsettledThreadSettingsWrite = { snapshot: ThreadScopedSettings | null };

const unsettledThreadSettingsWrites = new Map<
  string,
  UnsettledThreadSettingsWrite
>();

function beaconUnsettledThreadSettingsWrites(
  alreadySent: ReadonlySet<string>,
): void {
  const unsettled = [...unsettledThreadSettingsWrites];
  unsettledThreadSettingsWrites.clear();
  for (const [threadId, entry] of unsettled) {
    if (alreadySent.has(threadId)) continue;
    sendThreadScopedSettingsBeacon(threadId, entry.snapshot);
  }
}

function scheduleThreadScopedSettingsWrite(
  threadId: string,
  snapshot: ThreadScopedSettings | null = null,
): void {
  if (
    threadSettingsWriteThreadId !== null &&
    threadSettingsWriteThreadId !== threadId
  ) {
    flushThreadScopedSettingsWrite();
  }
  threadSettingsWriteThreadId = threadId;
  threadSettingsWriteSnapshot = snapshot;
  if (threadSettingsWriteTimer !== null) clearTimeout(threadSettingsWriteTimer);
  threadSettingsWriteTimer = setTimeout(() => {
    threadSettingsWriteTimer = null;
    const pendingThreadId = threadSettingsWriteThreadId;
    const pendingSnapshot = threadSettingsWriteSnapshot;
    threadSettingsWriteThreadId = null;
    threadSettingsWriteSnapshot = null;
    if (pendingThreadId !== null) {
      trackUnsettledThreadSettingsWrite(pendingThreadId, pendingSnapshot);
    }
  }, THREAD_SETTINGS_DEBOUNCE_MS);
}

// Edits made before this chat's snapshot lands; only the read says whose they are.
let pendingPairingThreadId: string | null = null;
let pairingWindowDefaults: ThreadScopedSettings | null = null;
let pairingWindowDefaultsThreadId: string | null = null;
let heldThreadScopedEdits: {
  field: string;
  writeGlobal: (() => void) | null;
  value?: unknown;
}[] = [];

export function beginThreadScopedPairing(threadId: string): void {
  if (pendingPairingThreadId === threadId) return;
  releaseHeldThreadScopedEdits();
  pendingPairingThreadId = threadId;
  // Sample defaults once per chat, not per attempt: a retry runs with the edit in the store.
  if (pairingWindowDefaultsThreadId !== threadId) {
    pairingWindowDefaultsThreadId = threadId;
    pairingWindowDefaults =
      threadScopedSettingsThreadId === null
        ? readThreadScopedSettings(useChatRuntimeStore.getState())
        : globalThreadScopedDefaults;
  }
  openThreadScopedPairingGate(threadId);
}

function openThreadScopedPairingGate(threadId: string): void {
  if (!useChatRuntimeStore.getState().threadScopedSettingsPending) {
    useChatRuntimeStore.setState({ threadScopedSettingsPending: true });
  }
  if (!pairingSettledByThreadId.has(threadId)) {
    let resolve!: () => void;
    const promise = new Promise<void>((r) => {
      resolve = r;
    });
    pairingSettledByThreadId.set(threadId, { promise, resolve });
  }
}

/** Release one chat's gate only, so B's pairing cannot free a run started for A. */
function closeThreadScopedPairingGate(threadId: string | null): void {
  if (threadId !== null) {
    pairingSettledByThreadId.get(threadId)?.resolve();
    pairingSettledByThreadId.delete(threadId);
  }
  const stillWaiting =
    pendingPairingThreadId !== null &&
    pairingSettledByThreadId.has(pendingPairingThreadId);
  if (useChatRuntimeStore.getState().threadScopedSettingsPending !== stillWaiting) {
    useChatRuntimeStore.setState({ threadScopedSettingsPending: stillWaiting });
  }
}

const pairingSettledByThreadId = new Map<
  string,
  { promise: Promise<void>; resolve: () => void }
>();

/** The adapter awaits this so a run never starts on installation defaults. */
export function awaitThreadScopedPairing(
  threadId: string | null | undefined,
): Promise<boolean> {
  if (!threadId) return Promise.resolve(true);
  const gate = pairingSettledByThreadId.get(threadId);
  if (!gate) return Promise.resolve(true);
  return Promise.race([
    gate.promise.then(() => true),
    new Promise<boolean>((resolve) =>
      setTimeout(() => resolve(false), THREAD_PAIRING_WAIT_MS),
    ),
  ]);
}

// Longer than the worst-case read with retries, so only an abandoned pairing refuses a run.
const THREAD_PAIRING_WAIT_MS = 30_000;

export function releaseHeldThreadScopedEdits(): void {
  const held = heldThreadScopedEdits;
  const threadId = pendingPairingThreadId;
  heldThreadScopedEdits = [];
  pendingPairingThreadId = null;
  pairingWindowDefaultsThreadId = null;
  closeThreadScopedPairingGate(threadId);
  for (const edit of held) {
    hydratedDefaultsByHeldField.delete(edit.field);
    edit.writeGlobal?.();
  }
}

/** Commit to the chat being left: writing to the defaults would move every other chat. */
export function commitHeldThreadScopedEditsToTheirThread(
  keepalive = false,
): Promise<void> {
  const threadId = pendingPairingThreadId;
  const held = heldThreadScopedEdits;
  heldThreadScopedEdits = [];
  pendingPairingThreadId = null;
  // pairingWindowDefaultsThreadId stays set: a retry would resample the edit as a default.
  closeThreadScopedPairingGate(null);
  if (threadId === null || held.length === 0) return Promise.resolve();
  const changes = heldThreadScopedChanges(held);
  restoreDefaultsOverCommittedEdits(threadId, held);
  if (keepalive) {
    sendThreadScopedSettingsBeacon(threadId, changes, true);
    return Promise.resolve();
  }
  // Returned, not fire-and-forget: forking copies settings_json server side.
  return mergeThreadScopedSettingsIntoRow(threadId, changes);
}

/** Restore installation values over a left chat's edits, or the next chat inherits them. */
function restoreDefaultsOverCommittedEdits(
  threadId: string,
  held: { field: string }[],
): void {
  if (useChatRuntimeStore.getState().activeThreadId === threadId) return;
  const before = (pairingWindowDefaults ?? globalThreadScopedDefaults) as Record<
    string,
    unknown
  > | null;
  const fields: Record<string, unknown> = {};
  const params: Record<string, unknown> = {};
  for (const edit of held) {
    const value = hydratedDefaultsByHeldField.has(edit.field)
      ? hydratedDefaultsByHeldField.get(edit.field)
      : before?.[edit.field];
    hydratedDefaultsByHeldField.delete(edit.field);
    if (value === undefined) continue;
    if (isThreadScopedParamKey(edit.field)) {
      params[edit.field] = value;
    } else {
      fields[edit.field] = value;
    }
  }
  if (!hasKeys(fields) && !hasKeys(params)) return;
  // setState, not setParams: the setter would persist these back to defaults and model memory.
  useChatRuntimeStore.setState((state) =>
    hasKeys(params)
      ? ({ ...fields, params: { ...state.params, ...params } } as Partial<ChatRuntimeStore>)
      : (fields as Partial<ChatRuntimeStore>),
  );
}

function heldThreadScopedChanges(
  held: { field: string; value?: unknown }[],
): ThreadScopedSettings {
  const edited: Record<string, unknown> = {};
  const live = useChatRuntimeStore.getState();
  // Sampling keys sit under `params`, so use the same reader as the snapshot path.
  for (const edit of held) {
    edited[edit.field] = edit.field === "reasoningEffort" && edit.value !== undefined
      ? edit.value
      : readThreadScopedValue(
          live,
          edit.field as ThreadScopedSettingKey,
        );
  }
  return sanitizeThreadScopedSettings(edited);
}

async function mergeThreadScopedSettingsIntoRow(
  threadId: string,
  changes: ThreadScopedSettings,
): Promise<void> {
  const settingsSeq = nextThreadSettingsSeq();
  const ticket = takeThreadSettingsWriteTicket(threadId);
  const previous = threadSettingsWriteChains.get(threadId) ?? Promise.resolve();
  const next = previous
    .catch(() => undefined)
    .then(() => threadSettingsReplaySettled)
    .catch(() => undefined)
    .then(async () => {
      if ((threadSettingsWriteTickets.get(threadId) ?? ticket) !== ticket) {
        return;
      }
      try {
        const { updateStoredChatThread } = await import(
          "../utils/chat-history-storage"
        );
        await updateStoredChatThread(threadId, {
          settingsPatch: changes,
          settingsSeq,
          settingsWriter: threadSettingsWriter,
        });
        forgetReplayedThreadSettings(threadId);
      } catch (error) {
        warnSettingsPersistenceFailure();
        // Rethrown: a fork awaits this and must not fork the pre-edit snapshot.
        throw error;
      }
    })
    .finally(() => {
      if (threadSettingsWriteChains.get(threadId) === next) {
        threadSettingsWriteChains.delete(threadId);
        threadSettingsWriteTickets.delete(threadId);
      }
    });
  threadSettingsWriteChains.set(threadId, next);
  return next;
}

const hydratedDefaultsByHeldField = new Map<string, unknown>();

function isHeldThreadScopedField(field: string): boolean {
  return heldThreadScopedEdits.some((edit) => edit.field === field);
}

function captureThreadScopedEdit(
  field: string,
  writeGlobal: (() => void) | null = null,
  value?: unknown,
): boolean {
  if (!isThreadOwnedSettingKey(field)) return false;
  const threadId = useChatRuntimeStore.getState().activeThreadId;
  if (threadId === null) return false;
  // Both ids: until the snapshot arrives the store still holds the old values.
  if (threadId === threadScopedSettingsThreadId) {
    if (field === "reasoningEffort" && value !== undefined) {
      activeThreadScopedSettings = {
        ...activeThreadScopedSettings,
        reasoningEffort: value as ReasoningEffort,
      };
    }
    explicitlyEditedThreadFields.add(field);
    constraintSuppressedThreadFields.delete(field);
    scheduleThreadScopedSettingsWrite(threadId);
    return true;
  }
  if (threadId === pendingPairingThreadId) {
    heldThreadScopedEdits.push({ field, writeGlobal, value });
    return true;
  }
  return false;
}

/** Before hydration a load-derived value only reflects this browser's cache. */
function persistLoadDerivedSetting(
  key: string,
  raw: string,
  stillCurrent: boolean,
): void {
  if (!mirroredSettingsHydrated) {
    writeStorageValue(key, raw);
    return;
  }
  if (!stillCurrent) return;
  persistSetting(key, raw);
}

function loadBool(key: string, fallback: boolean): boolean {
  const raw = loadOptionalBool(key);
  return raw ?? fallback;
}

export function loadOptionalBool(key: string): boolean | null {
  const raw = readStorageValue(key);
  if (raw === null) return null;
  return raw === "true";
}

export function resolveToolsEnabledOnLoad(supportsTools: boolean): {
  toolsEnabled: boolean;
  codeToolsEnabled: boolean;
} {
  if (!supportsTools) return { toolsEnabled: false, codeToolsEnabled: false };
  return {
    toolsEnabled:
      threadScopedOverride("toolsEnabled") ??
      loadOptionalBool(CHAT_TOOLS_ENABLED_KEY) ??
      false,
    codeToolsEnabled:
      threadScopedOverride("codeToolsEnabled") ??
      loadOptionalBool(CHAT_CODE_TOOLS_ENABLED_KEY) ??
      false,
  };
}

function saveBool(key: string, value: boolean): void {
  persistSetting(key, value ? "true" : "false");
}

let storedPreserveThinking: boolean | null = null;

function notePreserveThinkingPreference(value: boolean): void {
  storedPreserveThinking = value;
}

/** The backend family default only seeds an unanswered switch, never replaces an answer. */
export function resolvePreserveThinkingOnLoad(resp: {
  supports_preserve_thinking?: boolean | null;
  preserve_thinking_default?: boolean | null;
}): boolean {
  return storedPreserveThinking ?? preserveThinkingDefaultFromLoad(resp);
}


/** "full" is never restored: it disables the sandbox and every confirmation gate, so it needs
 *  the warning dialog each session. First run derives from the legacy confirm toggle. */
/** Anything but an explicit "low" (missing, garbled, a newer value) reads as the default, "high". */
export function normalizeSandboxLevel(raw: unknown): SandboxLevel {
  return raw === "low" ? "low" : "high";
}

export function loadSandboxLevel(): SandboxLevel {
  return normalizeSandboxLevel(readStorageValue(CHAT_SANDBOX_LEVEL_KEY));
}

function loadPermissionMode(): PermissionMode {
  return normalizeStoredPermissionMode(
    readStorageValue(CHAT_PERMISSION_MODE_KEY),
    loadOptionalBool(CHAT_CONFIRM_TOOL_CALLS_KEY),
  );
}

function savePermissionMode(mode: PermissionMode): void {
  if (mode === "full") return;
  persistSetting(CHAT_PERMISSION_MODE_KEY, mode);
}

const INITIAL_PERMISSION_MODE: PermissionMode = loadPermissionMode();

function codeDeclinedOnEnteringFullAccess(state: ChatRuntimeStore): boolean {
  return state.permissionMode === "full"
    ? state.codeToolsDeclinedUnderFullAccess
    : false;
}

/** Read the level live: Full access outlives a chat switch but codeToolsEnabled does not. */
export function codeToolsOn(
  state: Pick<
    ChatRuntimeStore,
    | "codeToolsEnabled"
    | "codeToolsDeclinedUnderFullAccess"
    | "permissionMode"
    | "supportsTools"
  > & { params: { checkpoint: string } },
): boolean {
  return (
    state.codeToolsEnabled ||
    (!state.codeToolsDeclinedUnderFullAccess &&
      state.permissionMode === "full" &&
      state.supportsTools &&
      !isExternalModelId(state.params.checkpoint))
  );
}

function loadString(key: string, fallback: string): string {
  return readStorageValue(key) ?? fallback;
}

function saveString(key: string, value: string): void {
  persistSetting(key, value);
}

export function normalizeSpeculativeType(
  v: string | null | undefined,
): string | null {
  if (v == null) return null;
  const s = String(v).trim().toLowerCase();
  if (!s) return null;
  if (s === "auto" || s === "default") return "auto";
  // Same spellings as _LEGACY_SPEC_MODE_MAP "off"; Auto here would re-send /load on every pick.
  if (s === "off" || s === "none" || s === "disable" || s === "disabled") {
    return "off";
  }
  if (s === "mtp" || s === "draft-mtp") return "mtp";
  if (s === "dspark" || s === "draft-dspark") return "dspark";
  if (s === "dflash" || s === "draft-dflash") return "dflash";
  if (s === "ngram" || s === "ngram-mod" || s === "ngram-simple") {
    return "ngram";
  }
  if (s === "mtp+ngram") return "mtp+ngram";
  const parts = s
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean);
  const hasMtp = parts.some((p) => p === "mtp" || p === "draft-mtp");
  const hasNgram = parts.some(
    (p) => p === "ngram" || p === "ngram-mod" || p === "ngram-simple",
  );
  if (hasMtp && hasNgram) return "mtp+ngram";
  if (hasMtp) return "mtp";
  if (hasNgram) return "ngram";
  return "auto";
}

export function resolveLoadedSpeculativeSettings(response: {
  speculative_type?: string | null;
  spec_draft_n_max?: number | null;
}): {
  speculativeType: string | null;
  loadedSpeculativeType: string | null;
  specDraftNMax: number | null;
  loadedSpecDraftNMax: number | null;
} {
  const loadedSpeculativeType = normalizeSpeculativeType(
    response.speculative_type,
  );
  const loadedSpecDraftNMax = response.spec_draft_n_max ?? null;
  return {
    speculativeType: loadedSpeculativeType,
    loadedSpeculativeType,
    specDraftNMax: loadedSpecDraftNMax,
    loadedSpecDraftNMax,
  };
}

export function readPersistedSpeculativeType(): string {
  const raw = loadString(CHAT_SPECULATIVE_TYPE_KEY, "auto");
  return PERSISTED_SPEC_MODES.has(raw) ? raw : "auto";
}

export function saveSpeculativeType(value: string | null): void {
  if (value && PERSISTED_SPEC_MODES.has(value)) {
    persistLoadDerivedSetting(
      CHAT_SPECULATIVE_TYPE_KEY,
      value,
      useChatRuntimeStore.getState().speculativeType === value,
    );
  }
}

export function readPersistedGpuMemoryMode(): "auto" | "manual" {
  return loadString(CHAT_GPU_MEMORY_MODE_KEY, "auto") === "manual" ? "manual" : "auto";
}

export function saveGpuMemoryMode(value: "auto" | "manual"): void {
  persistLoadDerivedSetting(
    CHAT_GPU_MEMORY_MODE_KEY,
    value,
    useChatRuntimeStore.getState().gpuMemoryMode === value,
  );
}

/** Only a non-diffusion GGUF: others report "auto" and must not clobber the preference. */
export function persistGpuMemoryModeOnLoad(
  resp: { is_gguf?: boolean; is_diffusion?: boolean },
  mode: "auto" | "manual",
): void {
  if (resp.is_gguf && !resp.is_diffusion) saveGpuMemoryMode(mode);
}

export { GPU_LAYERS_AUTO } from "../lib/gpu-placement";

function largestRemainder(shares: number[], total: number): number[] {
  const out = shares.map((x) => Math.floor(x));
  let rem = total - out.reduce((a, b) => a + b, 0);
  const byFrac = shares
    .map((x, i) => ({ i, frac: x - Math.floor(x) }))
    .sort((a, b) => b.frac - a.frac);
  for (let k = 0; rem > 0 && k < byFrac.length; k++, rem--) out[byFrac[k].i] += 1;
  return out;
}

// Mirrors llama.cpp's free-VRAM default split.
export function distributeByWeight(total: number, weights: number[]): number[] {
  if (weights.length === 0) return [];
  const t = Math.max(0, Math.floor(total));
  const sum = weights.reduce((a, b) => a + b, 0);
  const w = sum > 0 ? weights : weights.map(() => 1);
  const wSum = w.reduce((a, b) => a + b, 0);
  return largestRemainder(
    w.map((x) => (t * x) / wSum),
    t,
  );
}

// llama.cpp honors counts exactly only when gpu_layers == sum(counts).
export function rebalanceSplit(
  total: number,
  counts: number[],
  index: number,
  value: number,
): number[] {
  const v = Math.max(0, Math.min(value, total));
  const out = counts.slice();
  const otherIdx = counts.map((_, i) => i).filter((i) => i !== index);
  if (otherIdx.length === 0) {
    out[index] = total;
    return out;
  }
  out[index] = v;
  const dist = distributeByWeight(
    total - v,
    otherIdx.map((i) => counts[i]),
  );
  otherIdx.forEach((i, k) => (out[i] = dist[k]));
  return out;
}

// Null a stale persisted gpu_ids pick so it is not sent and rejected; cold cache leaves it.
export function reconcilePersistedGpuIds(
  ids: number[] | null,
  savedIndexKind?: GpuIndexKind | null,
  forDiffusion = false,
): number[] | null {
  return reconcilePersistedGpuSelection(
    ids,
    savedIndexKind,
    forDiffusion,
  ).ids;
}

export function reconcilePersistedGpuSelection(
  ids: number[] | null,
  savedIndexKind?: GpuIndexKind | null,
  forDiffusion = false,
): ReconciledGpuSelection {
  return reconcileCachedGpuSelection(ids, savedIndexKind, forDiffusion);
}

export function requestedGpuIdsFromResponse(resp: {
  gpu_ids?: number[] | null;
  requested_gpu_ids?: number[] | null;
}): number[] | null {
  return Object.prototype.hasOwnProperty.call(resp, "requested_gpu_ids")
    ? (resp.requested_gpu_ids ?? null)
    : (resp.gpu_ids ?? null);
}

export function loadedGpuMemoryFields(resp: {
  engine?: "auto" | "vllm" | "sglang";
  is_gguf?: boolean;
  is_diffusion?: boolean;
  gpu_memory_mode?: "auto" | "manual";
  gpu_layers?: number;
  cpu_fallback_reason?: "vulkan_startup_crash" | null;
  n_cpu_moe?: number;
  tensor_split?: number[] | null;
  n_layers?: number | null;
  n_moe_layers?: number;
  gpu_ids?: number[] | null;
  requested_gpu_ids?: number[] | null;
  diffusion_requested_ngl?: number | null;
}) {
  // Non-GGUF responses still say gpu_memory_mode "auto"; gate on is_gguf to keep the preference.
  if (!resp.is_gguf) {
    const managed = resp.engine === "vllm" || resp.engine === "sglang";
    const gpuIds = managed
      ? (requestedGpuIdsFromResponse(resp) ?? resp.gpu_ids ?? [0])
      : null;
    const indexKind = managed ? ("physical" as const) : null;
    return {
      selectedGpuIds: gpuIds,
      selectedGpuIndexKind: indexKind,
      loadedGpuIds: gpuIds,
      loadedGpuIndexKind: indexKind,
      loadedGpuMemoryMode: null,
      loadedCpuFallback: false,
      gpuLayers: GPU_LAYERS_AUTO,
      loadedGpuLayers: null,
      nCpuMoe: 0,
      loadedNCpuMoe: null,
      splitRatio: null,
      loadedSplitRatio: null,
      ggufLayerCount: null,
      moeLayerCount: null,
    };
  }
  const mode = resp.gpu_memory_mode ?? "auto";
  const hydratePlacementControls = shouldHydrateGpuPlacementControls(
    resp.cpu_fallback_reason,
  );
  const reportedGpuIds = requestedGpuIdsFromResponse(resp);
  const gpuIndexKind =
    reportedGpuIds == null
      ? null
      : cachedPinnableGpuIndexKind(resp.is_diffusion === true);
  // While discovery is cold, keep the pin or llama.cpp falls back to every device.
  const gpuIds =
    reportedGpuIds != null && gpuIndexKind !== null ? reportedGpuIds : null;
  // A shim without --ngl reports Auto while the backend holds the split, so recover it.
  const droppedSplit = recoverDroppedDiffusionSplit(
    resp.is_diffusion,
    mode,
    resp.diffusion_requested_ngl,
  );
  // Manual mode only; the server reports gpu_layers = -1 for Auto.
  const manualKnobs =
    mode === "manual"
      ? {
          loadedGpuLayers: resp.gpu_layers ?? null,
          loadedNCpuMoe: resp.n_cpu_moe ?? null,
          loadedSplitRatio: resp.tensor_split ?? null,
          ...(hydratePlacementControls
            ? {
                gpuLayers: resp.gpu_layers ?? GPU_LAYERS_AUTO,
                nCpuMoe: resp.n_cpu_moe ?? 0,
                splitRatio: resp.tensor_split ?? null,
              }
            : {}),
        }
      : {
          loadedGpuLayers: null,
          loadedNCpuMoe: null,
          loadedSplitRatio: null,
          ...(resp.is_diffusion
            ? droppedSplit != null
              ? { gpuLayers: droppedSplit }
              : {}
            : { gpuLayers: GPU_LAYERS_AUTO }),
          nCpuMoe: 0,
          splitRatio: null,
        };
  return {
    // A diffusion GGUF reporting "auto" ran on defaults; keep the manual preference.
    ...(hydratePlacementControls
      ? resp.is_diffusion && mode !== "manual"
        ? droppedSplit != null
          ? { gpuMemoryMode: "manual" as const }
          : {}
        : { gpuMemoryMode: mode }
      : {}),
    loadedGpuMemoryMode: mode,
    loadedCpuFallback: resp.cpu_fallback_reason === "vulkan_startup_crash",
    ggufLayerCount: resp.n_layers ?? null,
    moeLayerCount: resp.n_moe_layers ?? null,
    selectedGpuIds: gpuIds,
    selectedGpuIndexKind: gpuIds == null ? null : (gpuIndexKind ?? null),
    loadedGpuIds: gpuIds,
    loadedGpuIndexKind: gpuIds == null ? null : (gpuIndexKind ?? null),
    ...manualKnobs,
  };
}

export {
  hasGgufSource,
  isDownloadableHubRepo,
  isLocalModelPath,
  wantsDownloadManagerStaging,
} from "../utils/model-download-staging";

type ContextUsageSnapshot = {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  cachedTokens: number;
  cacheWriteTokens?: number;
  estimated?: boolean;
};

type ThreadRunOwner = {
  owner: () => void;
  local: boolean;
};

type ToolStatusEntry = {
  status: string;
  startedAt: number;
  owner?: () => void;
};

export type LoadedModelSummary = { id: string; quant?: string | null; checkpoint?: string };

type ChatRuntimeStore = {
  settingsHydrated: boolean;
  threadScopedSettingsPending: boolean;
  params: InferenceParams;
  paramsByModel: Record<string, PersistedInferenceParams>;
  rememberParamsPerModel: boolean;
  customPresets: Preset[];
  activePreset: string;
  activePresetSource: ChatPresetSource;
  models: ChatModelRow[];
  loras: ChatLoraSummary[];
  loraInventorySettled: boolean;
  runningByThreadId: Record<string, boolean>;
  localRunByThreadId: Record<string, boolean>;
  runOwnerByThreadId: Record<string, ThreadRunOwner[]>;
  cancelByThreadId: Record<string, () => void>;
  serverCancelByThreadId: Record<string, (() => void)[]>;
  autoTitle: boolean;
  hfToken: string;
  modelsError: string | null;
  // Set only when a LOAD fails, so attach gates can tell it from "no model picked".
  lastModelLoadError: string | null;
  activeGgufVariant: string | null;
  /** Resident per status, not the picker; undefined until first read to avoid a flash. */
  residentCheckpoint: string | null | undefined;
  loadedModels: LoadedModelSummary[];
  activeModelIsLocal: boolean;
  loadedContextLength: number | null;
  maxContextLength: number | null;
  nativeContextLength: number | null;
  loadedEngine: "auto" | "vllm" | "sglang";
  loadedEnginePrecision: NonNullable<InferenceParams["enginePrecision"]>;
  loadedEngineParallelism: NonNullable<InferenceParams["engineParallelism"]>;
  loadedIsGguf: boolean | null;
  /** Backend-reported: native-audio checkpoints are served off the MLX path. */
  loadedIsMlx: boolean | null;
  /** Null when the backend does not answer, which is not a confirmed false. */
  loadedContextEnforced: boolean | null;
  loadedContextUnboundedWhenBatched: boolean;
  loadedParallelSlots: number | null;
  loadedContextBudget: number | null;
  modelRequiresTrustRemoteCode: boolean;
  supportsReasoning: boolean;
  reasoningAlwaysOn: boolean;
  reasoningEnabled: boolean;
  lastOpenRouterChosenModel: string | null;
  reasoningStyle: ReasoningStyle;
  reasoningEffort: ReasoningEffort;
  supportsReasoningOff: boolean;
  reasoningEffortLevels: readonly ReasoningEffort[];
  supportsPreserveThinking: boolean;
  preserveThinking: boolean;
  supportsTools: boolean;
  supportsBuiltinWebSearch: boolean;
  supportsBuiltinCodeExecution: boolean;
  supportsBuiltinImageGeneration: boolean;
  supportsBuiltinWebFetch: boolean;
  keepModelsLoaded: boolean;
  toolsEnabled: boolean;
  /** Use codeToolsOn() for the effective value. */
  codeToolsEnabled: boolean;
  codeToolsDeclinedUnderFullAccess: boolean;
  imageToolsEnabled: boolean;
  deepResearchEnabled: boolean;
  researchWebsitePolicy: ResearchWebsitePolicy;
  researchModelTimeoutSeconds: number;
  // Whether the Canvas toggle is offered in the composer + menu (hidden by default).
  collapseHtmlArtifacts: boolean;
  allowArtifactNetworkAccess: boolean;
  searchImages: boolean;
  mcpEnabledForChat: boolean;
  ragEnabled: boolean;
  ragSource: RagSource;
  projectAttachmentTarget: ProjectAttachmentTarget;
  projectAttachmentTargetByThread: Record<string, ProjectAttachmentTarget>;
  ragMode: RagMode;
  ragTopK: number;
  ragAutoInject: RagAutoInject;
  ragAutoInjectMinScore: number;
  ragOcrScanned: boolean;
  ragCaptionFigures: boolean;
  confirmToolCalls: boolean;
  /** No confirmation gate and no sandbox; kept in sync with permissionMode "full". */
  bypassPermissions: boolean;
  /** Source of truth; bypassPermissions and confirmToolCalls mirror it. */
  permissionMode: PermissionMode;
  sandboxLevel: SandboxLevel;
  /** Whether the bypass warning dialog is open. Lifted out of the composer menu so confirming
   *  it does not leave the menu frozen. */
  bypassConfirmOpen: boolean;
  alwaysAllowToolsBySession: Map<string, Set<string>>;
  toolConfirmations: Record<
    string,
    {
      approvalId: string;
      sessionId: string;
      autoAllowKey: string;
      imageDisclosure?: ImageDisclosure;
    }
  >;
  webFetchToolsEnabled: boolean;
  /** A list: unresolved threads share "__default", so one clear must not remove a sibling. */
  toolStatusByThreadId: Record<string, ToolStatusEntry[]>;
  toolLiveOutput: Record<string, string>;
  toolFullOutput: Record<string, string>;
  generatingStatus: string | null;
  autoHealToolCalls: boolean;
  nudgeToolCalls: boolean;
  autoCompactEnabled: boolean;
  maxToolCallsPerMessage: number;
  toolCallTimeout: number;
  kvCacheDtype: string | null;
  mlxKvQuant: MlxKvQuant | null;
  loadedMlxKvQuantRequested: MlxKvQuant | null;
  mlxKvQuantReason: string | null;
  chatTemplateOverrideReason: string | null;
  mlxKvQuantNote: string | null;
  mlxInt8Prefill: boolean;
  loadedMlxInt8PrefillRequested: boolean;
  loadedKvCacheDtype: string | null;
  speculativeType: string | null;
  loadedSpeculativeType: string | null;
  specFallbackReason: string | null;
  mmprojFallbackReason: MmprojFallbackReason | null;
  specDrafterKind: string | null;
  specDraftNMax: number | null;
  loadedSpecDraftNMax: number | null;
  /** Never re-seeded from an echo: the resolved count would pin a blank control. */
  nParallel: number | null;
  loadedNParallel: number | null;
  reasoningBudget: number;
  loadedReasoningBudget: number | null;
  loadedReasoningBudgetRequested: number | null;
  reasoningBudgetMessage: string;
  loadedReasoningBudgetMessage: string | null;
  loadedReasoningBudgetMessageRequested: string | null;
  nBatch: number | null;
  loadedNBatch: number | null;
  loadedLlamaExtraArgs: string[] | null;
  nUbatch: number | null;
  loadedNUbatch: number | null;
  specDraftCacheDtype: string | null;
  loadedSpecDraftCacheDtype: string | null;
  loadMode: string | null;
  loadedLoadMode: string | null;
  ctxCheckpoints: number | null;
  loadedCtxCheckpoints: number | null;
  cacheRam: number | null;
  loadedCacheRam: number | null;
  tensorParallel: boolean;
  loadedTensorParallel: boolean | null;
  loadedDisableVision: boolean | null;
  disableVision: boolean;
  loadedVisionDisabledByUser: boolean | null;
  gpuMemoryMode: "auto" | "manual";
  loadedGpuMemoryMode: "auto" | "manual" | null;
  loadedCpuFallback: boolean;
  /** Manual mode: -1 = Auto (--fit); >= model layer count = all. */
  gpuLayers: number;
  loadedGpuLayers: number | null;
  nCpuMoe: number;
  loadedNCpuMoe: number | null;
  splitRatio: number[] | null;
  loadedSplitRatio: number[] | null;
  /** GGUF block_count; the manual ceiling is this + 1 for the output layer. */
  ggufLayerCount: number | null;
  moeLayerCount: number | null;
  selectedGpuIds: number[] | null;
  selectedGpuIndexKind: GpuIndexKind | null;
  loadedGpuIds: number[] | null;
  loadedGpuIndexKind: GpuIndexKind | null;
  expandQuantizations: boolean;
  showAllQuantizations: boolean;
  showMemoryBar: boolean;
  fitOnDeviceOnly: boolean;
  loadedIsMultimodal: boolean;
  loadedIsDiffusion: boolean;
  activeDiffusionCanvasByThreadId: Record<string, DiffusionCanvasFrame>;
  customContextLength: number | null;
  loadedCustomContextLength: number | null;
  defaultChatTemplate: string | null;
  chatTemplateOverride: string | null;
  loadedChatTemplateOverride: string | null;
  activeThreadId: string | null;
  activeThreadEpoch: number;
  queuedSettingsEpoch: number;
  activeProjectId: string | null;
  /** Lives only in memory, never studio.db, so a refresh always exits incognito. */
  incognito: boolean;
  settingsPanelOpen: boolean;
  editingMessageId: string | null;
  pendingAudioBase64: string | null;
  pendingAudioName: string | null;
  pendingImageEditReference: PendingImageEditReference | null;
  contextUsage: ContextUsageSnapshot | null;
  contextUsageByThreadId: Record<string, ContextUsageSnapshot>;
  modelLoading: boolean;
  loadingModelPick: (LoadingModelPick & { selectionSuperseded: boolean }) | null;
  activeLoadId: string | null;
  activeNativePathToken: string | null;
  // The desktop host prunes file leases on a TTL, so an expired token forces re-selection.
  activeNativePathExpiresAtMs: number | null;
  hydratePersistedSettings: () => Promise<void>;
  beginModelLoading: (phase?: ModelLifecyclePhase) => ModelLifecycleLease | null;
  endModelLoading: (lease: ModelLifecycleLease) => void;
  setLoadingModelPick: (pick: LoadingModelPick | null) => void;
  clearLoadingModelPick: (expected: LoadingModelPick) => void;
  setModelRequiresTrustRemoteCode: (required: boolean) => void;
  setParams: (
    params: InferenceParams,
    options?: {
      persist?: boolean;
      trackQueuedSettings?: boolean;
      minPChoiceEdited?: boolean;
      fromModelDefaults?: boolean;
      maxTokensCap?: number;
      migrateOwnedGlobalQwenDefaults?: boolean;
    },
  ) => void;
  setCustomPresets: (presets: Preset[]) => void;
  setActivePreset: (name: string) => void;
  setActivePresetSource: (source: ChatPresetSource) => void;
  setModels: (models: ChatModelRow[]) => void;
  setLoras: (loras: ChatLoraSummary[]) => void;
  setThreadRunning: (
    threadId: string,
    running: boolean,
    options?: { local?: boolean; owner?: () => void },
  ) => void;
  /** Runs file under "__default" until the thread is persisted; re-key them then. */
  adoptDefaultThreadRun: (threadId: string) => void;
  runKeyForOwner: (fallbackKey: string, owner: () => void) => string;
  registerThreadCancel: (threadId: string, cancel: () => void) => void;
  clearThreadCancel: (threadId: string, cancel?: () => void) => void;
  registerThreadServerCancel: (threadId: string, cancel: () => void) => void;
  clearThreadServerCancel: (threadId: string, cancel?: () => void) => void;
  setAutoTitle: (enabled: boolean) => void;
  setHfToken: (token: string) => void;
  setModelsError: (error: string | null) => void;
  setLastModelLoadError: (error: string | null) => void;
  setCheckpoint: (
    modelId: string,
    ggufVariant?: string | null,
    options?: {
      trackQueuedSettings?: boolean;
      persist?: boolean;
      maxTokensCap?: number;
    },
  ) => void;
  setActiveThreadId: (threadId: string | null) => void;
  applyThreadScopedSettings: (
    threadId: string | null,
    settings: ThreadScopedSettings | null,
  ) => void;
  setActiveProjectId: (projectId: string | null) => void;
  setIncognito: (incognito: boolean) => void;
  setSettingsPanelOpen: (open: boolean) => void;
  setEditingMessageId: (id: string | null) => void;
  clearCheckpoint: () => void;
  setReasoningEnabled: (
    enabled: boolean,
    options?: { persist?: boolean },
  ) => void;
  setRememberedParamsForModel: (
    modelId: string,
    patch: PersistedInferenceParams,
  ) => void;
  setLastOpenRouterChosenModel: (chosen: string | null) => void;
  setReasoningStyle: (style: ReasoningStyle) => void;
  setReasoningEffort: (effort: ReasoningEffort) => void;
  setPreserveThinking: (value: boolean) => void;
  setToolsEnabled: (enabled: boolean, options?: { persist?: boolean }) => void;
  setKeepModelsLoaded: (keep: boolean) => void;
  setCodeToolsEnabled: (enabled: boolean) => void;
  setImageToolsEnabled: (enabled: boolean) => void;
  setDeepResearchEnabled: (enabled: boolean) => void;
  setResearchWebsitePolicy: (policy: ResearchWebsitePolicy) => void;
  setResearchModelTimeoutSeconds: (seconds: number) => void;
  setCollapseHtmlArtifacts: (enabled: boolean) => void;
  setAllowArtifactNetworkAccess: (enabled: boolean) => void;
  setSearchImages: (enabled: boolean) => void;
  setMcpEnabledForChat: (enabled: boolean) => void;
  setConfirmToolCalls: (enabled: boolean) => void;
  setBypassPermissions: (enabled: boolean) => void;
  setPermissionMode: (mode: PermissionMode) => void;
  setSandboxLevel: (level: SandboxLevel) => void;
  setBypassConfirmOpen: (open: boolean) => void;
  allowToolAlways: (sessionId: string, toolName: string) => void;
  setToolConfirmation: (
    toolCallId: string,
    approvalId: string,
    sessionId: string,
    autoAllowKey: string,
    imageDisclosure?: ImageDisclosure,
  ) => void;
  clearToolConfirmation: (toolCallId: string) => void;
  setWebFetchToolsEnabled: (enabled: boolean) => void;
  setRagEnabled: (enabled: boolean) => void;
  setRememberParamsPerModel: (enabled: boolean) => void;
  setRagSource: (source: RagSource) => void;
  setProjectAttachmentTarget: (target: ProjectAttachmentTarget) => void;
  setThreadProjectAttachmentTarget: (
    threadId: string | null,
    target: ProjectAttachmentTarget,
  ) => void;
  adoptPendingProjectAttachmentTarget: (threadId: string, claim?: number) =>
    void;
  clearPendingProjectAttachmentTarget: () => void;
  setRagMode: (mode: RagMode) => void;
  setRagTopK: (topK: number) => void;
  setRagAutoInject: (value: RagAutoInject) => void;
  setRagAutoInjectMinScore: (score: number) => void;
  setRagOcrScanned: (enabled: boolean) => void;
  setRagCaptionFigures: (enabled: boolean) => void;
  setToolStatus: (
    threadId: string,
    status: string | null,
    owner?: () => void,
  ) => void;
  appendToolLiveOutput: (toolCallId: string, text: string) => void;
  clearToolLiveOutput: (toolCallId?: string) => void;
  setToolFullOutput: (toolCallId: string, text: string) => void;
  clearToolFullOutput: (toolCallId: string) => void;
  setGeneratingStatus: (status: string | null) => void;
  setActiveDiffusionCanvas: (
    threadId: string | null,
    canvas: DiffusionCanvasFrame,
  ) => void;
  clearActiveDiffusionCanvasForThread: (threadId: string | null) => void;
  setAutoHealToolCalls: (enabled: boolean) => void;
  setNudgeToolCalls: (enabled: boolean) => void;
  setAutoCompactEnabled: (enabled: boolean) => void;
  setMaxToolCallsPerMessage: (value: number) => void;
  setToolCallTimeout: (value: number) => void;
  setGpuMemoryMode: (mode: "auto" | "manual") => void;
  setGpuLayers: (value: number) => void;
  setNCpuMoe: (value: number) => void;
  setSplitRatio: (value: number[] | null) => void;
  setSelectedGpuIds: (
    ids: number[] | null,
    indexKind?: GpuIndexKind | null,
  ) => void;
  setExpandQuantizations: (value: boolean) => void;
  setShowAllQuantizations: (value: boolean) => void;
  setShowMemoryBar: (value: boolean) => void;
  setFitOnDeviceOnly: (value: boolean) => void;
  setPendingAudio: (base64: string, name: string) => void;
  clearPendingAudio: () => void;
  setPendingImageEditReference: (
    reference: PendingImageEditReference | null,
  ) => void;
  clearPendingImageEditReference: () => void;
  setContextUsage: (usage: ChatRuntimeStore["contextUsage"]) => void;
  setThreadContextUsage: (
    threadId: string,
    usage: ContextUsageSnapshot,
  ) => void;
};

type PersistedChatSettings = Awaited<
  ReturnType<typeof loadChatSettingsWithLegacyImport>
>["settings"];
type PersistedInferenceParams = NonNullable<
  PersistedChatSettings["inferenceParams"]
>;
type ScalarSettingKey =
  | "autoTitle"
  | "rememberParamsPerModel"
  | "reasoningEffort"
  | "preserveThinking"
  | "collapseHtmlArtifacts"
  | "allowArtifactNetworkAccess"
  | "searchImages"
  | "autoHealToolCalls"
  | "nudgeToolCalls"
  | "autoCompactEnabled"
  | "maxToolCallsPerMessage"
  | "toolCallTimeout"
  | "reasoningEnabled"
  | "toolsEnabled"
  | "codeToolsEnabled"
  | "imageToolsEnabled"
  | "webFetchToolsEnabled"
  | "deepResearchEnabled"
  | "researchWebsitePolicy"
  | "researchModelTimeoutSeconds"
  | "mcpEnabledForChat"
  | "confirmToolCalls"
  | "permissionMode"
  | "sandboxLevel"
  | "ragSource"
  | "ragMode"
  | "ragTopK"
  | "ragAutoInject"
  | "ragAutoInjectMinScore"
  | "ragOcrScanned"
  | "ragCaptionFigures"
  | "expandQuantizations"
  | "showAllQuantizations"
  | "fitOnDeviceOnly"
  | "speculativeType"
  | "gpuMemoryMode";

type PresetHydrationVersions = {
  customPresets: number;
  activePreset: number;
  activePresetSource: number;
};

type SettingsHydrationVersions = {
  inferenceParams: Record<PersistedInferenceParamKey, number>;
  scalarSettings: Record<ScalarSettingKey, number>;
  presets: PresetHydrationVersions;
};

const SCALAR_SETTING_KEYS = [
  "autoTitle",
  "rememberParamsPerModel",
  "reasoningEffort",
  "preserveThinking",
  "collapseHtmlArtifacts",
  "allowArtifactNetworkAccess",
  "searchImages",
  "autoHealToolCalls",
  "nudgeToolCalls",
  "autoCompactEnabled",
  "maxToolCallsPerMessage",
  "toolCallTimeout",
  "reasoningEnabled",
  "toolsEnabled",
  "codeToolsEnabled",
  "imageToolsEnabled",
  "webFetchToolsEnabled",
  "deepResearchEnabled",
  "researchWebsitePolicy",
  "researchModelTimeoutSeconds",
  "mcpEnabledForChat",
  "confirmToolCalls",
  "permissionMode",
  "sandboxLevel",
  "ragSource",
  "ragMode",
  "ragTopK",
  "ragAutoInject",
  "ragAutoInjectMinScore",
  "ragOcrScanned",
  "ragCaptionFigures",
  "expandQuantizations",
  "showAllQuantizations",
  "fitOnDeviceOnly",
  "speculativeType",
  "gpuMemoryMode",
] as const satisfies readonly ScalarSettingKey[];

const locallyRememberedModels = new Set<string>();
// Pre-hydration edits are held as patches: they name only touched keys and must merge.
const modelParamEditsBeforeHydration = new Map<string, PersistedInferenceParams>();
/** Lazy: an import cycle puts PERSISTED_INFERENCE_PARAM_KEYS in its TDZ at module load. */
let inferenceParamMutationVersionsCache: Record<
  PersistedInferenceParamKey,
  number
> | null = null;

function inferenceParamMutationVersions(): Record<
  PersistedInferenceParamKey,
  number
> {
  inferenceParamMutationVersionsCache ??= Object.fromEntries(
    PERSISTED_INFERENCE_PARAM_KEYS.map((key) => [key, 0]),
  ) as Record<PersistedInferenceParamKey, number>;
  return inferenceParamMutationVersionsCache;
}
const scalarSettingMutationVersions = Object.fromEntries(
  SCALAR_SETTING_KEYS.map((key) => [key, 0]),
) as Record<ScalarSettingKey, number>;

let loadedModelReasoningMode: {
  checkpoint: string;
  enabled: boolean;
  reasoningMutationVersion: number;
  fromLoad: boolean;
} | null = null;

export function noteLoadedModelReasoningMode(
  checkpoint: string,
  enabled: boolean,
  fromLoad = false,
): void {
  const state = useChatRuntimeStore.getState();
  const previous = loadedModelReasoningMode;
  loadedModelReasoningMode = {
    checkpoint,
    enabled:
      threadScopedOverride("reasoningEnabled") !== undefined
        ? installationReasoningEnabled(state)
        : enabled,
    reasoningMutationVersion:
      scalarSettingMutationVersions.reasoningEnabled,
    // Sticky: refresh() calls this again with false right after the load marks it.
    fromLoad:
      fromLoad ||
      (previous !== null &&
        sameCheckpointIdentity(previous.checkpoint, checkpoint) &&
        previous.reasoningMutationVersion ===
          scalarSettingMutationVersions.reasoningEnabled &&
        previous.fromLoad),
  };
}

function hasKeys(value: object): boolean {
  return Object.keys(value).length > 0;
}

function getSettingsHydrationVersions(): SettingsHydrationVersions {
  return {
    inferenceParams: { ...inferenceParamMutationVersions() },
    scalarSettings: { ...scalarSettingMutationVersions },
    presets: {
      customPresets: customPresetsMutationVersion,
      activePreset: activePresetMutationVersion,
      activePresetSource: activePresetSourceMutationVersion,
    },
  };
}

function cacheHydratedSettings(
  settings: PersistedChatSettings,
  versions: SettingsHydrationVersions,
): void {
  for (const [name, setting] of Object.entries(MIRRORED_SETTINGS)) {
    const field = name as MirroredSettingKey;
    const value = settings[field];
    if (value === undefined) continue;
    if (
      scalarSettingMutationVersions[field] !== versions.scalarSettings[field]
    ) {
      continue;
    }
    writeStorageValue(setting.storageKey, setting.encode(value));
  }
}

function readStoredSettingValue(
  setting: { storageKey: string } & MirroredSettingCodec,
): unknown {
  const raw = readStorageValue(setting.storageKey);
  return raw === null ? undefined : setting.decode(raw);
}

function backfillMirroredSettings(settings: PersistedChatSettings): void {
  const patch: SettingsPatch = {};
  for (const [name, setting] of Object.entries(MIRRORED_SETTINGS)) {
    const field = name as MirroredSettingKey;
    if (settings[field] !== undefined) continue;
    const value = setting.readForBackfill
      ? setting.readForBackfill()
      : readStoredSettingValue(setting);
    if (value === undefined) continue;
    (patch as Record<string, unknown>)[field] = value;
  }
  if (hasKeys(patch)) saveSettingsPatch(patch);
}

function getChangedInferenceParams(
  nextParams: InferenceParams,
  currentParams: InferenceParams,
  bumpVersions = true,
  minPChoiceEdited = false,
): PersistedInferenceParams {
  const changedParams: PersistedInferenceParams = {};
  const minPChoiceChanged =
    bumpVersions &&
    (minPChoiceEdited ||
      !Object.is(nextParams.minP, currentParams.minP) ||
      !Object.is(nextParams.minPMode, currentParams.minPMode));
  for (const key of PERSISTED_INFERENCE_PARAM_KEYS) {
    const nextValue = nextParams[key];
    if (
      Object.is(nextValue, currentParams[key]) &&
      !(minPChoiceChanged && (key === "minP" || key === "minPMode"))
    ) {
      continue;
    }
    if (bumpVersions) {
      inferenceParamMutationVersions()[key] += 1;
    }
    if (nextValue !== undefined) {
      setInferenceParam(changedParams as InferenceParams, key, nextValue);
    }
  }
  // A written number carries its mode, or the next read treats it as legacy.
  if (changedParams.minP !== undefined && nextParams.minPMode !== undefined) {
    changedParams.minPMode = nextParams.minPMode;
  }
  return changedParams;
}

function persistReplayedParams(
  state: ChatRuntimeStore,
  nextParams: InferenceParams,
  replayed: boolean,
): void {
  if (!replayed) {
    return;
  }
  const changed = getChangedInferenceParams(nextParams, state.params);
  if (state.settingsHydrated && hasKeys(changed)) {
    saveSettingsPatch({ inferenceParams: changed });
    noteThreadScopedDefaults(changed);
  }
}

function trackParamsByModel(
  state: ChatRuntimeStore,
  paramsByModel: Record<string, PersistedInferenceParams> | null,
  modelId: string | undefined,
): Record<string, PersistedInferenceParams> | null {
  if (!state.settingsHydrated) {
    return null;
  }
  if (paramsByModel && modelId) {
    locallyRememberedModels.add(modelId);
  }
  return paramsByModel;
}

function getReplayStatePatch(
  state: ChatRuntimeStore,
  nextParams: InferenceParams,
  outgoing: Record<string, PersistedInferenceParams> | null,
  baseParams: InferenceParams,
): Partial<ChatRuntimeStore> {
  persistReplayedParams(state, nextParams, baseParams !== state.params);
  return outgoing ? { paramsByModel: outgoing } : {};
}

function getParamsByModelAfterEdit(
  state: ChatRuntimeStore,
  outgoing: Record<string, PersistedInferenceParams> | null,
  nextParams: InferenceParams,
  changedParams: PersistedInferenceParams,
  persist: boolean,
): Record<string, PersistedInferenceParams> | null {
  if (!persist) {
    return outgoing;
  }
  const recorded = trackParamsByModel(
    state,
    getRememberedParamsPatch(
      state.rememberParamsPerModel,
      outgoing ?? state.paramsByModel,
      nextParams.checkpoint,
      changedParams,
      pickRememberedParams(withoutActiveThreadParams(state, nextParams)),
    ),
    nextParams.checkpoint,
  );
  return recorded ?? outgoing;
}

function rememberOutgoingModel(
  state: ChatRuntimeStore,
  outgoing: InferenceParams,
): Record<string, PersistedInferenceParams> | null {
  if (!state.settingsHydrated && outgoing.checkpoint) {
    modelLeftBeforeHydration = outgoing.checkpoint;
  }
  const snapshot = pickRememberedParams(
    withoutActiveThreadParams(state, outgoing),
  );
  // Seed only: a full snapshot would overwrite another tab's copy of untouched keys.
  const seeding = state.paramsByModel[outgoing.checkpoint] === undefined;
  const next = trackParamsByModel(
    state,
    getRememberedParamsPatch(
      state.rememberParamsPerModel,
      state.paramsByModel,
      outgoing.checkpoint,
      snapshot,
      snapshot,
    ),
    outgoing.checkpoint,
  );
  if (next && state.settingsHydrated && outgoing.checkpoint && seeding) {
    saveSettingsPatch({
      inferenceParamsByModel: { [outgoing.checkpoint]: snapshot },
    });
  }
  return next;
}

function persistParamEdit(
  changedParams: PersistedInferenceParams,
  paramsByModel: Record<string, PersistedInferenceParams> | null,
  modelId: string | undefined,
): void {
  if (!hasKeys(changedParams)) {
    return;
  }
  // Only what moved: the server merges per key.
  const rememberedChanges = pickRememberedChanges(changedParams);
  saveSettingsPatch({
    inferenceParams: changedParams,
    ...(paramsByModel && modelId && hasKeys(rememberedChanges)
      ? { inferenceParamsByModel: { [modelId]: rememberedChanges } }
      : {}),
  });
}

function getHydratedCustomPresets(
  settings: PersistedChatSettings,
  state: ChatRuntimeStore,
): Preset[] {
  settings = normalizeSavedChatSettings(settings);
  return (
    settings.customPresets?.map((preset) => {
      const loadConfig = normalizePresetLoadConfig(preset.loadConfig);
      return {
        name: preset.name,
        params: {
          ...DEFAULT_INFERENCE_PARAMS,
          ...preset.params,
        },
        ...(loadConfig ? { loadConfig } : {}),
      };
    }) ?? state.customPresets
  );
}

function getHydratedPresetState(
  settings: PersistedChatSettings,
  state: ChatRuntimeStore,
  versions: PresetHydrationVersions,
): Partial<
  Pick<
    ChatRuntimeStore,
    "customPresets" | "activePreset" | "activePresetSource"
  >
> {
  const nextState: Partial<
    Pick<
      ChatRuntimeStore,
      "customPresets" | "activePreset" | "activePresetSource"
    >
  > = {};
  if (customPresetsMutationVersion === versions.customPresets) {
    nextState.customPresets = getHydratedCustomPresets(settings, state);
  }
  if (activePresetMutationVersion === versions.activePreset) {
    nextState.activePreset = settings.activePreset ?? state.activePreset;
  }
  if (activePresetSourceMutationVersion === versions.activePresetSource) {
    const activePreset = nextState.activePreset ?? state.activePreset;
    nextState.activePresetSource =
      settings.activePresetSource ?? getPresetSource(activePreset);
  }
  return nextState;
}

function pickLocallyEditedParams(
  params: InferenceParams,
  versions: SettingsHydrationVersions,
): PersistedInferenceParams {
  const edited: PersistedInferenceParams = {};
  for (const key of REMEMBERED_INFERENCE_PARAM_KEYS) {
    if (inferenceParamMutationVersions()[key] !== versions.inferenceParams[key]) {
      setInferenceParam(edited as InferenceParams, key, params[key]);
    }
  }
  return edited;
}

let loadedContext: { checkpoint: string; cap: number } | null = null;

function noteLoadedContext(checkpoint: string, cap: number | undefined): void {
  if (cap !== undefined) {
    loadedContext = { checkpoint, cap };
  }
}

function loadedContextFor(checkpoint: string): number | null {
  return loadedContext?.checkpoint === checkpoint ? loadedContext.cap : null;
}

function capParamsToLoadedContext(
  state: ChatRuntimeStore,
  params: InferenceParams,
): InferenceParams {
  const residentContextCap = isExternalModelId(params.checkpoint)
    ? null
    : state.loadedContextLength;
  const cap = loadedContextFor(params.checkpoint) ?? residentContextCap;
  return cap !== null && params.maxTokens > cap
    ? { ...params, maxTokens: cap }
    : params;
}

let modelLoadedBeforeHydration: string | null = null;

let modelLeftBeforeHydration: string | null = null;

function noteModelDefaultsBeforeHydration(
  checkpoint: string,
  ownsPersistedGlobal: boolean,
): void {
  if (!ownsPersistedGlobal) {
    modelLoadedBeforeHydration = checkpoint;
    return;
  }
  if (!sameCheckpointIdentity(modelLoadedBeforeHydration, checkpoint)) {
    modelLoadedBeforeHydration = null;
  }
  if (sameCheckpointIdentity(unownedCheckpointBeforeHydration, checkpoint)) {
    unownedCheckpointBeforeHydration = null;
  }
}

/** An unresolved provider is left alone: refusing would drop a Codex preference. */
function externalCheckpointRefusesDeepResearch(
  checkpoint: string | null | undefined,
): boolean {
  const parsed = parseExternalModelId(checkpoint);
  if (!parsed) return false;
  const provider = useExternalProvidersStore
    .getState()
    .providers.find((candidate) => candidate.id === parsed.providerId);
  return provider != null && provider.providerType !== "openai_codex";
}

/** Kimi rejects search with thinking on, so a restore must keep the pills exclusive. */
function isKimiCheckpoint(checkpoint: string | null | undefined): boolean {
  const parsed = parseExternalModelId(checkpoint);
  if (!parsed) return false;
  return (
    useExternalProvidersStore
      .getState()
      .providers.find((candidate) => candidate.id === parsed.providerId)
      ?.providerType === "kimi"
  );
}

function getHydratedSettingsState(
  settings: PersistedChatSettings,
  state: ChatRuntimeStore,
  versions: SettingsHydrationVersions,
): Partial<ChatRuntimeStore> {
  // A migration's confirming read stays raw for CAS; normalize only on hydration.
  settings = normalizeSavedChatSettings(settings);
  const nextState: Partial<ChatRuntimeStore> = {};
  const checkpoint = state.params.checkpoint;
  const loadedBeforeHydration = sameCheckpointIdentity(
    modelLoadedBeforeHydration,
    checkpoint,
  );
  modelLoadedBeforeHydration = null;
  const remembersPerModel =
    settings.rememberParamsPerModel !== undefined &&
    scalarSettingMutationVersions.rememberParamsPerModel ===
      versions.scalarSettings.rememberParamsPerModel
      ? settings.rememberParamsPerModel
      : state.rememberParamsPerModel;
  // A model loaded mid-flight has no entry, so the global set would overwrite its defaults.
  const keepModelDefaults =
    remembersPerModel &&
    loadedBeforeHydration &&
    settings.inferenceParamsByModel?.[checkpoint] === undefined;
  const params = { ...state.params };
  for (const key of PERSISTED_INFERENCE_PARAM_KEYS) {
    const value = settings.inferenceParams?.[key];
    if (value !== undefined && isHeldThreadScopedField(key)) {
      hydratedDefaultsByHeldField.set(key, value);
      continue;
    }
    if (
      value !== undefined &&
      !keepModelDefaults &&
      // The context belongs to the load, not the previous model's global set.
      !(loadedBeforeHydration && key === "maxSeqLength") &&
      inferenceParamMutationVersions()[key] === versions.inferenceParams[key]
    ) {
      setInferenceParam(params, key, value);
    }
  }
  nextState.params = params;
  if (settings.inferenceParamsByModel !== undefined) {
    const hydrated: Record<string, PersistedInferenceParams> = {};
    for (const [modelId, entry] of Object.entries(
      settings.inferenceParamsByModel,
    )) {
      hydrated[modelId] = entry;
    }
    for (const modelId of locallyRememberedModels) {
      const local = state.paramsByModel[modelId];
      if (local) {
        hydrated[modelId] = local;
      }
    }
    for (const [modelId, patch] of modelParamEditsBeforeHydration) {
      hydrated[modelId] = { ...hydrated[modelId], ...patch };
      locallyRememberedModels.add(modelId);
    }
    modelParamEditsBeforeHydration.clear();
    if (checkpoint) {
      const edited = pickLocallyEditedParams(params, versions);
      if (hasKeys(edited)) {
        hydrated[checkpoint] = { ...hydrated[checkpoint], ...edited };
      }
    }
    nextState.paramsByModel = hydrated;
  } else if (checkpoint) {
    const edited = pickLocallyEditedParams(params, versions);
    if (hasKeys(edited)) {
      nextState.paramsByModel = {
        ...state.paramsByModel,
        [checkpoint]: { ...state.paramsByModel[checkpoint], ...edited },
      };
    }
  }
  // File the global set under a model stepped off before hydration, now that it is known.
  const left = modelLeftBeforeHydration;
  modelLeftBeforeHydration = null;
  const byModel = nextState.paramsByModel ?? state.paramsByModel;
  if (
    remembersPerModel &&
    left &&
    !sameCheckpointIdentity(left, checkpoint) &&
    !byModel[left]
  ) {
    const inherited: PersistedInferenceParams = {};
    for (const key of REMEMBERED_INFERENCE_PARAM_KEYS) {
      const value = settings.inferenceParams?.[key];
      if (value !== undefined) {
        setInferenceParam(inherited as InferenceParams, key, value);
      }
    }
    if (hasKeys(inherited)) {
      nextState.paramsByModel = { ...byModel, [left]: inherited };
    }
  }
  // A click made while this response was out is the newer answer.
  if (
    settings.preserveThinking !== undefined &&
    scalarSettingMutationVersions.preserveThinking ===
      versions.scalarSettings.preserveThinking
  ) {
    notePreserveThinkingPreference(settings.preserveThinking);
  }
  for (const key of SCALAR_SETTING_KEYS) {
    const value = settings[key];
    // Full access is session-only; a stored level must not drop the accepted bypass.
    if (
      state.permissionMode === "full" &&
      (key === "permissionMode" || key === "confirmToolCalls")
    ) {
      continue;
    }
    if (loadShadowOwnsMirroredSetting(key, state)) {
      continue;
    }
    // Only a load sets this; a stored false would ask a model that cannot stop thinking to stop.
    if (
      key === "reasoningEnabled" &&
      value === false &&
      state.reasoningAlwaysOn
    ) {
      continue;
    }
    if (
      key === "reasoningEnabled" &&
      loadEstablishedReasoningMode(state, true)
    ) {
      continue;
    }
    // Only local models or openai_codex can run deep research.
    if (
      key === "deepResearchEnabled" &&
      value === true &&
      externalCheckpointRefusesDeepResearch(state.params.checkpoint)
    ) {
      continue;
    }
    // Held edits advance no mutation version, so the server value would replace them.
    if (isHeldThreadScopedField(key)) {
      if (
        value !== undefined &&
        scalarSettingMutationVersions[key] === versions.scalarSettings[key]
      ) {
        hydratedDefaultsByHeldField.set(key, value);
      }
      continue;
    }
    if (
      value !== undefined &&
      scalarSettingMutationVersions[key] === versions.scalarSettings[key]
    ) {
      (nextState as Record<ScalarSettingKey, unknown>)[key] = value;
    }
  }
  // The already-selected model never crossed a transition, so replay its memory here.
  const remembered = (nextState.paramsByModel ?? state.paramsByModel)[
    params.checkpoint
  ];
  if (
    (nextState.rememberParamsPerModel ?? state.rememberParamsPerModel) &&
    remembered
  ) {
    // REMEMBERED, not PERSISTED: a stored maxSeqLength must not replace the loaded context.
    const replayed = { ...params };
    for (const key of REMEMBERED_INFERENCE_PARAM_KEYS) {
      const value = remembered[key];
      if (
        value !== undefined &&
        inferenceParamMutationVersions()[key] === versions.inferenceParams[key]
      ) {
        setInferenceParam(replayed, key, value);
      }
    }
    nextState.params = replayed;
  }
  const capped = nextState.params ?? params;
  // An external pick leaves the local model resident, so its context is not this cap.
  const residentContextCap = isExternalModelId(checkpoint)
    ? null
    : state.loadedContextLength;
  const cap = loadedContextFor(checkpoint) ?? residentContextCap;
  if (cap !== null && capped.maxTokens > cap) {
    nextState.params = { ...capped, maxTokens: cap };
  }
  return nextState;
}

function setScalarSettingVersion<K extends ScalarSettingKey>(
  key: K,
  value: ChatRuntimeStore[K],
  currentValue: ChatRuntimeStore[K],
): void {
  if (Object.is(value, currentValue)) {
    return;
  }
  const writeGlobal = () => {
    if (key === "reasoningEffort" && globalThreadScopedDefaults !== null) {
      globalThreadScopedDefaults = {
        ...globalThreadScopedDefaults,
        reasoningEffort: value as ReasoningEffort,
      };
    }
    scalarSettingMutationVersions[key] += 1;
    saveSettingsPatch({ [key]: value });
  };
  if (captureThreadScopedEdit(key, writeGlobal, value)) return;
  writeGlobal();
}

function localQwenMigrationSettings(
  state: ChatRuntimeStore,
): PersistedChatSettings {
  return {
    activePreset: state.activePreset,
    activePresetSource: state.activePresetSource,
    reasoningEnabled: installationReasoningEnabled(state),
    inferenceParams: pickRememberedParams(
      withoutActiveThreadParams(state, state.params),
    ),
    ...(Object.keys(state.paramsByModel).length > 0
      ? { inferenceParamsByModel: state.paramsByModel }
      : {}),
  };
}

// The chat's own effort displaced by a per-model pin, restored when the pin clears.
let effortDisplacedByPin: ReasoningEffort | null = null;

/** The first displacement wins, so pin-pin-clear returns to the chat's own level. */
export function noteEffortDisplacedByPin(current: ReasoningEffort): void {
  // Before hydration the live level is only the store default, not the chat's.
  if (!useChatRuntimeStore.getState().settingsHydrated) return;
  effortDisplacedByPin ??= current;
}

export function pinHoldsLiveEffort(): boolean {
  return (
    pinOwnsLiveReasoningEffort(useChatRuntimeStore.getState()) ||
    effortDisplacedByPin !== null
  );
}

/** Prefers the chat's snapshot level, which survives reloads unlike the in-memory record. */
export function takeEffortDisplacedByPin(): ReasoningEffort | null {
  const displaced = effortDisplacedByPin;
  effortDisplacedByPin = null;
  return (
    activeThreadScopedSettings?.reasoningEffort ??
    globalThreadScopedDefaults?.reasoningEffort ??
    displaced
  );
}

export function reconcilePinnedReasoningEffort(opts: {
  checkpoint: string;
  caps: ExternalReasoningCapabilities;
  providerType: string | null | undefined;
  apiType?: "chat_completions" | "responses";
}): void {
  const state = useChatRuntimeStore.getState();
  if (state.params.checkpoint !== opts.checkpoint) return;
  const pinned = externalReasoningTakesEffort(opts.caps)
    ? pinnedReasoningEffort(opts.checkpoint, opts.caps.reasoningEffortLevels)
    : null;
  if (!pinned && !pinHoldsLiveEffort()) return;
  const next = resolveExternalReasoningEffort({
    caps: opts.caps,
    providerType: opts.providerType,
    apiType: opts.apiType,
    current: pinned
      ? state.reasoningEffort
      : (takeEffortDisplacedByPin() ?? state.reasoningEffort),
    pinned,
    restore: pinned === null,
  });
  if (pinned) noteEffortDisplacedByPin(state.reasoningEffort);
  if (next === state.reasoningEffort) return;
  useChatRuntimeStore.setState((live) => ({
    reasoningEffort: next,
    queuedSettingsEpoch: live.queuedSettingsEpoch + 1,
  }));
}

function installationReasoningEnabled(state: ChatRuntimeStore): boolean {
  return threadScopedOverride("reasoningEnabled") !== undefined
    ? (globalThreadScopedDefaults?.reasoningEnabled ?? state.reasoningEnabled)
    : state.reasoningEnabled;
}

/** A load writes reasoningEnabled without a mutation version; hydration must not undo it. */
function loadEstablishedReasoningMode(
  state: ChatRuntimeStore,
  requireLoad = false,
): { enabled: boolean } | null {
  const loaded = loadedModelReasoningMode;
  if (
    loaded !== null &&
    sameCheckpointIdentity(loaded.checkpoint, state.params.checkpoint) &&
    loaded.reasoningMutationVersion ===
      scalarSettingMutationVersions.reasoningEnabled &&
    (!requireLoad || loaded.fromLoad)
  ) {
    return { enabled: loaded.enabled };
  }
  return null;
}

function qwenMigrationThinkingOn(
  settings: PersistedChatSettings,
  state: ChatRuntimeStore,
  reasoningMutationVersion = scalarSettingMutationVersions.reasoningEnabled,
): boolean {
  if (state.reasoningAlwaysOn) {
    return true;
  }
  // supportsReasoning starts false, so wait for status on this checkpoint before trusting it.
  const statusSeen = sameCheckpointIdentity(
    loadedModelReasoningMode?.checkpoint,
    state.params.checkpoint,
  );
  if (statusSeen && !state.supportsReasoning) {
    return false;
  }
  const established = loadEstablishedReasoningMode(state, true);
  if (established) {
    return established.enabled;
  }
  return settings.reasoningEnabled !== undefined &&
    scalarSettingMutationVersions.reasoningEnabled === reasoningMutationVersion
    ? settings.reasoningEnabled
    : installationReasoningEnabled(state);
}

function qwenMigrationRemembersPerModel(
  settings: PersistedChatSettings,
  state: ChatRuntimeStore,
  rememberParamsPerModelMutationVersion =
    scalarSettingMutationVersions.rememberParamsPerModel,
): boolean {
  return settings.rememberParamsPerModel !== undefined &&
    scalarSettingMutationVersions.rememberParamsPerModel ===
      rememberParamsPerModelMutationVersion
    ? settings.rememberParamsPerModel
    : state.rememberParamsPerModel;
}

const QWEN_MIGRATION_DECISION_FIELDS = [
  "activePreset",
  "activePresetSource",
  "reasoningEnabled",
  "rememberParamsPerModel",
  "inferenceParamsByModel",
] as const satisfies ReadonlyArray<keyof PersistedChatSettings>;

// Raw, not sanitized: the server tests `key in current` against what is stored.
/** A stored field that sanitizes away cannot be fenced by CAS, so decline to migrate. */
function qwenMigrationHasUnfenceableField(
  rawSettings: unknown,
  sanitized: PersistedChatSettings,
): boolean {
  const raw =
    typeof rawSettings === "object" && rawSettings !== null
      ? (rawSettings as Record<string, unknown>)
      : {};
  return QWEN_MIGRATION_DECISION_FIELDS.some(
    (field) => Object.hasOwn(raw, field) && sanitized[field] === undefined,
  );
}

function qwenMigrationExpectedAbsent(
  rawSettings: unknown,
): Array<keyof PersistedChatSettings> {
  const raw =
    typeof rawSettings === "object" && rawSettings !== null
      ? (rawSettings as Record<string, unknown>)
      : {};
  return QWEN_MIGRATION_DECISION_FIELDS.filter(
    (field) => !Object.hasOwn(raw, field),
  );
}

function qwenMigrationExpectedAbsentPaths(
  rawSettings: unknown,
  patch: PersistedChatSettings,
): Array<[keyof PersistedChatSettings, string]> {
  const nested = (field: keyof PersistedChatSettings): Record<string, unknown> =>
    typeof rawSettings === "object" && rawSettings !== null
      ? ((rawSettings as Record<string, unknown>)[field] as
          | Record<string, unknown>
          | undefined) ?? {}
      : {};
  const paths: Array<[keyof PersistedChatSettings, string]> = [];
  if (patch.inferenceParams !== undefined) {
    const global = nested("inferenceParams");
    for (const field of ["topK", "repetitionPenalty"] as const) {
      if (!Object.hasOwn(global, field)) {
        paths.push(["inferenceParams", field]);
      }
    }
  }
  // Fence the exact-key row too, or another tab's newer row would be overwritten.
  const stored = nested("inferenceParamsByModel");
  for (const modelId of Object.keys(patch.inferenceParamsByModel ?? {})) {
    if (!Object.hasOwn(stored, modelId)) {
      paths.push(["inferenceParamsByModel", modelId]);
    }
  }
  return paths;
}

function applyLegacyQwenDefaultsAfterPresetChange(
  ownedGlobalCheckpoint: string | null,
  migrateOwnedGlobalAlongsideModelMemory: boolean,
): void {
  useChatRuntimeStore.setState((state) => {
    if (
      !state.settingsHydrated ||
      state.activePresetSource !== "builtin-default"
    ) {
      return state;
    }
    const checkpoint = state.params.checkpoint;
    const includeOwnedGlobal = sameCheckpointIdentity(
      ownedGlobalCheckpoint,
      checkpoint,
    );
    const localSettings = localQwenMigrationSettings(state);
    const migration = migrateLegacyQwenDefaults(
      localSettings,
      checkpoint,
      qwenMigrationThinkingOn(localSettings, state),
      includeOwnedGlobal,
      migrateOwnedGlobalAlongsideModelMemory,
    );
    if (!migration.patch) return state;

    const activeModelId = migration.migratedModelIds.find(
      (modelId) => sameCheckpointIdentity(modelId, checkpoint),
    );
    const activePatch = activeModelId
      ? migration.patch.inferenceParamsByModel?.[activeModelId]
      : migration.patch.inferenceParams;
    if (activePatch) {
      noteThreadScopedDefaults(activePatch);
    }
    return {
      ...(migration.settings.inferenceParamsByModel
        ? { paramsByModel: migration.settings.inferenceParamsByModel }
        : {}),
      ...(activePatch
        ? {
            params: capParamsToLoadedContext(
              state,
              restoreThreadScopedParams({
                ...state.params,
                ...activePatch,
              }),
            ),
          }
        : {}),
    };
  });
}

/** After a migration raced a local edit, adopt only the fields the user did not touch. */
function adoptMigratedFieldsAfterLocalEdit(
  patch: PersistedChatSettings,
  checkpoint: string,
  versionsBefore: Record<PersistedInferenceParamKey, number>,
): void {
  const migratedRow =
    patch.inferenceParamsByModel?.[checkpoint] ?? patch.inferenceParams;
  if (migratedRow === undefined) {
    return;
  }
  useChatRuntimeStore.setState((state) => {
    const nextParams = { ...state.params };
    let changed = false;
    for (const [key, value] of Object.entries(migratedRow)) {
      const field = key as PersistedInferenceParamKey;
      if (
        versionsBefore[field] === undefined ||
        inferenceParamMutationVersions()[field] !== versionsBefore[field] ||
        nextParams[field] === value
      ) {
        continue;
      }
      (nextParams as Record<string, unknown>)[field] = value;
      changed = true;
    }
    if (!changed) {
      return state;
    }
    const storedRow = state.paramsByModel[checkpoint];
    return {
      params: restoreThreadScopedParams(nextParams),
      ...(storedRow
        ? {
            paramsByModel: {
              ...state.paramsByModel,
              [checkpoint]: { ...storedRow, ...migratedRow },
            },
          }
        : {}),
    };
  });
}

async function retryLegacyQwenDefaultsAfterPresetChange(
  ownedGlobalCheckpoint: string | null,
  migrateOwnedGlobalAlongsideModelMemory: boolean,
): Promise<void> {
  try {
    await flushPendingChatSettings();
    // A write outlasting the flush timeout could land after this CAS and restore the legacy row.
    if (!settingsWritesAreDrained()) {
      // Bounded: a refused patch is reflushed each pass, so unbounded rearm loops offline.
      if (qwenMigrationRearmsWhileBlocked < QWEN_MIGRATION_MAX_REARMS) {
        qwenMigrationRearmsWhileBlocked += 1;
        void inflightFlush.catch(() => undefined).then(() => {
          scheduleLegacyQwenDefaultsRetry(
            ownedGlobalCheckpoint,
            migrateOwnedGlobalAlongsideModelMemory,
          );
        });
      }
      return;
    }
    qwenMigrationRearmsWhileBlocked = 0;
    const state = useChatRuntimeStore.getState();
    if (
      !state.settingsHydrated ||
      state.activePresetSource !== "builtin-default"
    ) {
      return;
    }
    const checkpoint = state.params.checkpoint;
    const confirmedRaw = await getChatSettings();
    const confirmed = sanitizeChatSettings(confirmedRaw);
    const confirmedState = useChatRuntimeStore.getState();
    if (
      !confirmedState.settingsHydrated ||
      confirmedState.activePresetSource !== "builtin-default" ||
      confirmedState.params.checkpoint !== checkpoint
    ) {
      return;
    }
    const includeOwnedGlobal = sameCheckpointIdentity(
      ownedGlobalCheckpoint,
      checkpoint,
    );
    const migration = migrateLegacyQwenDefaults(
      confirmed,
      checkpoint,
      qwenMigrationThinkingOn(confirmed, confirmedState),
      includeOwnedGlobal,
      migrateOwnedGlobalAlongsideModelMemory &&
        !qwenMigrationRemembersPerModel(confirmed, confirmedState),
    );
    if (!migration.patch) return;
    if (qwenMigrationHasUnfenceableField(confirmedRaw, confirmed)) return;
    // Captured before the write: a mid-flight edit makes local refuse what the server took.
    const presetSourceBeforeWrite = activePresetSourceMutationVersion;
    const paramVersionsBeforeWrite = { ...inferenceParamMutationVersions() };
    const persisted = await savePersistedChatSettingsPatchIfCurrent(
      confirmed,
      migration.patch,
      qwenMigrationExpectedAbsent(confirmedRaw),
      qwenMigrationExpectedAbsentPaths(confirmedRaw, migration.patch),
    );
    // Apply locally only after persisting, and revalidate: the checkpoint can move meanwhile.
    if (
      persisted.applied &&
      sameCheckpointIdentity(
        useChatRuntimeStore.getState().params.checkpoint,
        checkpoint,
      )
    ) {
      if (activePresetSourceMutationVersion === presetSourceBeforeWrite) {
        applyLegacyQwenDefaultsAfterPresetChange(
          ownedGlobalCheckpoint,
          migrateOwnedGlobalAlongsideModelMemory,
        );
      } else if (
        migration.patch &&
        useChatRuntimeStore.getState().activePresetSource !== "custom"
      ) {
        adoptMigratedFieldsAfterLocalEdit(
          migration.patch,
          checkpoint,
          paramVersionsBeforeWrite,
        );
      }
    }
  } catch {
    warnSettingsPersistenceFailure();
  }
}

let qwenMigrationInFlight: Promise<void> | null = null;

const QWEN_MIGRATION_BARRIER_TIMEOUT_MS = 2000;

const QWEN_MIGRATION_MAX_REARMS = 3;
let qwenMigrationRearmsWhileBlocked = 0;

/** Lets a send wait for a scheduled Qwen defaults migration to persist. */
export async function awaitPendingQwenDefaultsMigration(): Promise<void> {
  if (qwenMigrationInFlight === null) return;
  // Bounded: neither request takes an abort signal.
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    await Promise.race([
      followPendingQwenDefaultsMigrations(),
      new Promise<void>((resolve) => {
        timer = setTimeout(resolve, QWEN_MIGRATION_BARRIER_TIMEOUT_MS);
      }),
    ]);
  } finally {
    if (timer !== undefined) clearTimeout(timer);
  }
}

// Re-read: a replacement scheduled while waiting owns the checkpoint.
async function followPendingQwenDefaultsMigrations(): Promise<void> {
  let pending = qwenMigrationInFlight;
  while (pending) {
    await pending;
    pending = qwenMigrationInFlight;
  }
}

let qwenDefaultsRetryScheduled = false;
let qwenDefaultsRetryOwnedGlobalCheckpoint: string | null = null;
let qwenDefaultsRetryOwnedGlobalCheckpointConflicted = false;
let qwenDefaultsRetryMigratesOwnedGlobalAlongsideModelMemory = false;

function scheduleLegacyQwenDefaultsRetry(
  ownedGlobalCheckpoint: string | null,
  migrateOwnedGlobalAlongsideModelMemory = ownedGlobalCheckpoint !== null,
): void {
  if (ownedGlobalCheckpoint !== null) {
    if (
      qwenDefaultsRetryOwnedGlobalCheckpoint !== null &&
      !sameCheckpointIdentity(
        qwenDefaultsRetryOwnedGlobalCheckpoint,
        ownedGlobalCheckpoint,
      )
    ) {
      qwenDefaultsRetryOwnedGlobalCheckpointConflicted = true;
    } else if (!qwenDefaultsRetryOwnedGlobalCheckpointConflicted) {
      qwenDefaultsRetryOwnedGlobalCheckpoint = ownedGlobalCheckpoint;
    }
  }
  qwenDefaultsRetryMigratesOwnedGlobalAlongsideModelMemory ||=
    migrateOwnedGlobalAlongsideModelMemory;
  if (qwenDefaultsRetryScheduled) {
    return;
  }
  qwenDefaultsRetryScheduled = true;
  // Published at scheduling time, or a send in between finds nothing to wait for.
  let settleMigration: () => void = () => undefined;
  const migration: Promise<void> = new Promise<void>((resolve) => {
    settleMigration = () => {
      if (qwenMigrationInFlight === migration) {
        qwenMigrationInFlight = null;
      }
      resolve();
    };
  });
  qwenMigrationInFlight = migration;
  queueMicrotask(() => {
    const scheduledOwnedGlobalCheckpoint =
      qwenDefaultsRetryOwnedGlobalCheckpointConflicted
        ? null
        : qwenDefaultsRetryOwnedGlobalCheckpoint;
    const migrateScheduledOwnedGlobalAlongsideModelMemory =
      qwenDefaultsRetryMigratesOwnedGlobalAlongsideModelMemory;
    qwenDefaultsRetryScheduled = false;
    qwenDefaultsRetryOwnedGlobalCheckpoint = null;
    qwenDefaultsRetryOwnedGlobalCheckpointConflicted = false;
    qwenDefaultsRetryMigratesOwnedGlobalAlongsideModelMemory = false;
    const state = useChatRuntimeStore.getState();
    if (
      !state.settingsHydrated ||
      state.activePresetSource !== "builtin-default"
    ) {
      settleMigration();
      return;
    }
    const localSettings = localQwenMigrationSettings(state);
    const includeOwnedGlobal = sameCheckpointIdentity(
      scheduledOwnedGlobalCheckpoint,
      state.params.checkpoint,
    );
    const hasLocalCandidate =
      migrateLegacyQwenDefaults(
        localSettings,
        state.params.checkpoint,
        qwenMigrationThinkingOn(localSettings, state),
        includeOwnedGlobal,
        migrateScheduledOwnedGlobalAlongsideModelMemory,
      ).patch !== null;
    const hasOwnedGlobalCandidate =
      includeOwnedGlobal && isPresenceBumpQwen(state.params.checkpoint);
    if (!hasLocalCandidate && !hasOwnedGlobalCandidate) {
      settleMigration();
      return;
    }
    void retryLegacyQwenDefaultsAfterPresetChange(
      scheduledOwnedGlobalCheckpoint,
      migrateScheduledOwnedGlobalAlongsideModelMemory,
    ).finally(settleMigration);
  });
}

export const useChatRuntimeStore = create<ChatRuntimeStore>((set, get) => ({
  settingsHydrated: false,
  threadScopedSettingsPending: false,
  // Only external checkpoints persist; local ids are re-derived from the backend.
  params: (() => {
    const persistedExternal = loadLastExternalCheckpoint();
    unownedCheckpointBeforeHydration = persistedExternal;
    return persistedExternal
      ? { ...DEFAULT_INFERENCE_PARAMS, checkpoint: persistedExternal }
      : DEFAULT_INFERENCE_PARAMS;
  })(),
  paramsByModel: {},
  rememberParamsPerModel: true,
  customPresets: [],
  activePreset: "Default",
  activePresetSource: getPresetSource("Default"),
  models: [],
  loras: [],
  loraInventorySettled: false,
  runningByThreadId: {},
  localRunByThreadId: {},
  runOwnerByThreadId: {},
  cancelByThreadId: {},
  serverCancelByThreadId: {},
  autoTitle: false,
  hfToken: useHfTokenStore.getState().token,
  modelsError: null,
  lastModelLoadError: null,
  activeGgufVariant: null,
  residentCheckpoint: undefined,
  loadedModels: [],
  activeModelIsLocal: false,
  loadedContextLength: null,
  maxContextLength: null,
  nativeContextLength: null,
  loadedEngine: "auto",
  loadedEnginePrecision: "auto",
  loadedEngineParallelism: "tensor",
  loadedIsGguf: null,
  loadedIsMlx: null,
  loadedContextEnforced: null,
  loadedContextUnboundedWhenBatched: false,
  loadedParallelSlots: null,
  loadedContextBudget: null,
  modelRequiresTrustRemoteCode: false,
  supportsReasoning: false,
  reasoningAlwaysOn: false,
  reasoningEnabled: loadBool(CHAT_REASONING_ENABLED_KEY, true),
  reasoningStyle: "enable_thinking",
  reasoningEffort: "medium",
  supportsReasoningOff: false,
  reasoningEffortLevels: ["low", "medium", "high"],
  lastOpenRouterChosenModel: null,
  supportsPreserveThinking: false,
  preserveThinking: false,
  supportsTools: false,
  supportsBuiltinWebSearch: false,
  supportsBuiltinCodeExecution: false,
  supportsBuiltinImageGeneration: false,
  supportsBuiltinWebFetch: false,
  toolsEnabled: loadBool(CHAT_TOOLS_ENABLED_KEY, false),
  keepModelsLoaded: false,
  codeToolsEnabled: loadBool(CHAT_CODE_TOOLS_ENABLED_KEY, false),
  codeToolsDeclinedUnderFullAccess: false,
  imageToolsEnabled: loadBool(CHAT_IMAGE_TOOLS_ENABLED_KEY, false),
  deepResearchEnabled: loadBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false),
  researchWebsitePolicy: loadResearchWebsitePolicy(),
  researchModelTimeoutSeconds: loadResearchModelTimeoutSeconds(),
  collapseHtmlArtifacts: loadBool(CHAT_COLLAPSE_HTML_ARTIFACTS_KEY, false),
  allowArtifactNetworkAccess: loadBool(
    CHAT_ALLOW_ARTIFACT_NETWORK_ACCESS_KEY,
    false,
  ),
  searchImages: loadBool(CHAT_SEARCH_IMAGES_KEY, false),
  mcpEnabledForChat: loadBool(CHAT_MCP_ENABLED_KEY, false),
  confirmToolCalls:
    INITIAL_PERMISSION_MODE === "ask" || INITIAL_PERMISSION_MODE === "auto",
  // Never restore Bypass Permissions: it needs the warning dialog each session.
  bypassPermissions: false,
  permissionMode: INITIAL_PERMISSION_MODE,
  sandboxLevel: loadSandboxLevel(),
  bypassConfirmOpen: false,
  alwaysAllowToolsBySession: new Map<string, Set<string>>(),
  toolConfirmations: {},
  webFetchToolsEnabled: loadBool(CHAT_WEB_FETCH_TOOLS_ENABLED_KEY, false),
  ragEnabled: false,
  ragSource: loadRagSource(),
  projectAttachmentTarget: loadProjectAttachmentTarget(),
  projectAttachmentTargetByThread: {},
  ragMode: loadRagMode(),
  ragTopK: loadRagTopK(),
  ragAutoInject: loadRagAutoInject(),
  ragAutoInjectMinScore: loadRagNumber(
    CHAT_RAG_AUTOINJECT_MIN_SCORE_KEY,
    DEFAULT_RAG_AUTOINJECT_MIN_SCORE,
    { min: 0, max: 1 },
  ),
  ragOcrScanned: loadBool(CHAT_RAG_OCR_KEY, DEFAULT_RAG_OCR),
  ragCaptionFigures: loadBool(CHAT_RAG_CAPTION_KEY, DEFAULT_RAG_CAPTION),
  toolStatusByThreadId: {},
  toolLiveOutput: {},
  toolFullOutput: {},
  generatingStatus: null,
  activeDiffusionCanvasByThreadId: {},
  autoHealToolCalls: true,
  nudgeToolCalls: true,
  autoCompactEnabled: DEFAULT_AUTO_COMPACT_ENABLED,
  maxToolCallsPerMessage: 25,
  toolCallTimeout: 5,
  kvCacheDtype: null,
  mlxKvQuant: null,
  loadedMlxKvQuantRequested: null,
  mlxKvQuantReason: null,
  chatTemplateOverrideReason: null,
  mlxKvQuantNote: null,
  mlxInt8Prefill: false,
  loadedMlxInt8PrefillRequested: false,
  loadedKvCacheDtype: null,
  speculativeType: readPersistedSpeculativeType(),
  loadedSpeculativeType: null,
  specFallbackReason: null,
  mmprojFallbackReason: null,
  specDrafterKind: null,
  specDraftNMax: null,
  loadedSpecDraftNMax: null,
  nParallel: null,
  loadedNParallel: null,
  reasoningBudget: -1,
  loadedReasoningBudget: null,
  loadedReasoningBudgetRequested: null,
  reasoningBudgetMessage: "",
  loadedReasoningBudgetMessage: null,
  loadedReasoningBudgetMessageRequested: null,
  nBatch: null,
  loadedNBatch: null,
  loadedLlamaExtraArgs: null,
  nUbatch: null,
  loadedNUbatch: null,
  specDraftCacheDtype: null,
  loadedSpecDraftCacheDtype: null,
  loadMode: null,
  loadedLoadMode: null,
  ctxCheckpoints: null,
  loadedCtxCheckpoints: null,
  cacheRam: null,
  loadedCacheRam: null,
  tensorParallel: false,
  loadedTensorParallel: null,
  loadedDisableVision: null,
  disableVision: false,
  loadedVisionDisabledByUser: null,
  gpuMemoryMode: readPersistedGpuMemoryMode(),
  loadedGpuMemoryMode: null,
  loadedCpuFallback: false,
  gpuLayers: GPU_LAYERS_AUTO,
  loadedGpuLayers: null,
  nCpuMoe: 0,
  loadedNCpuMoe: null,
  splitRatio: null,
  loadedSplitRatio: null,
  ggufLayerCount: null,
  moeLayerCount: null,
  selectedGpuIds: null,
  selectedGpuIndexKind: null,
  loadedGpuIds: null,
  loadedGpuIndexKind: null,
  expandQuantizations: loadBool(CHAT_EXPAND_QUANTIZATIONS_KEY, false),
  showAllQuantizations: loadBool(CHAT_SHOW_ALL_QUANTIZATIONS_KEY, false),
  showMemoryBar: loadBool(CHAT_SHOW_MEMORY_BAR_KEY, false),
  fitOnDeviceOnly: loadBool(MODELS_FIT_ON_DEVICE_ONLY_KEY, false),
  loadedIsMultimodal: false,
  loadedIsDiffusion: false,
  customContextLength: null,
  loadedCustomContextLength: null,
  defaultChatTemplate: null,
  chatTemplateOverride: null,
  loadedChatTemplateOverride: null,
  activeThreadId: null,
  activeThreadEpoch: 0,
  queuedSettingsEpoch: 0,
  activeProjectId: null,
  incognito: false,
  settingsPanelOpen: false,
  editingMessageId: null,
  pendingAudioBase64: null,
  pendingAudioName: null,
  pendingImageEditReference: null,
  contextUsage: null,
  contextUsageByThreadId: {},
  modelLoading: false,
  loadingModelPick: null,
  activeLoadId: null,
  activeNativePathToken: null,
  activeNativePathExpiresAtMs: null,
  hydratePersistedSettings: async () => {
    if (get().settingsHydrated) {
      return;
    }
    if (settingsHydrationPromise) {
      return settingsHydrationPromise;
    }
    settingsHydrationPromise = (async () => {
      const hydrationVersions = getSettingsHydrationVersions();
      try {
        const {
          settings,
          fromServer,
          persisted: settingsArePersisted,
        } = await loadChatSettingsWithLegacyImport();
        let confirmed: PersistedChatSettings | undefined;
        // A failed-to-save legacy merge exists only here; re-reading the server would discard it.
        const unmigrated = (
          snapshot: PersistedChatSettings,
        ): QwenDefaultsMigration => ({
          settings: settingsArePersisted ? snapshot : settings,
          patch: null,
          migratedModelIds: [],
        });
        const checkpoint = get().params.checkpoint;
        const thinkingOn = qwenMigrationThinkingOn(
          settings,
          get(),
          hydrationVersions.scalarSettings.reasoningEnabled,
        );
        // Only the model resident at startup can own a global-only legacy snapshot.
        const globalBelongsToActiveCheckpoint =
          !sameCheckpointIdentity(modelLoadedBeforeHydration, checkpoint) &&
          modelLeftBeforeHydration === null &&
          !sameCheckpointIdentity(unownedCheckpointBeforeHydration, checkpoint);
        const remembersPerModel = qwenMigrationRemembersPerModel(
          settings,
          get(),
          hydrationVersions.scalarSettings.rememberParamsPerModel,
        );
        let migration = migrateLegacyQwenDefaults(
          settings,
          checkpoint,
          thinkingOn,
          globalBelongsToActiveCheckpoint,
          globalBelongsToActiveCheckpoint && !remembersPerModel,
        );
        // Switch paths cannot schedule a retry here: settingsHydrated is still false.
        let checkpointMovedDuringConfirm = false;
        if (fromServer && migration.patch) {
          try {
            const confirmedRaw = await getChatSettings();
            confirmed = sanitizeChatSettings(confirmedRaw);
            const confirmedState = get();
            // A model switch or a debounced preset edit during the GET invalidates this migration.
            const presetSourceUnchanged =
              activePresetSourceMutationVersion ===
                hydrationVersions.presets.activePresetSource &&
              confirmedState.activePresetSource === "builtin-default";
            checkpointMovedDuringConfirm =
              confirmedState.params.checkpoint !== checkpoint;
            migration =
              confirmedState.params.checkpoint === checkpoint &&
              presetSourceUnchanged
                ? migrateLegacyQwenDefaults(
                    confirmed,
                    checkpoint,
                    qwenMigrationThinkingOn(
                      confirmed,
                      confirmedState,
                      hydrationVersions.scalarSettings.reasoningEnabled,
                    ),
                    globalBelongsToActiveCheckpoint,
                    globalBelongsToActiveCheckpoint &&
                      !qwenMigrationRemembersPerModel(
                        confirmed,
                        confirmedState,
                        hydrationVersions.scalarSettings
                          .rememberParamsPerModel,
                      ),
                  )
                : unmigrated(confirmed);
            // migrateLegacyQwenDefaults returns its input when nothing migrates; keep the unsaved merge.
            if (
              migration.patch === null ||
              qwenMigrationHasUnfenceableField(confirmedRaw, confirmed)
            ) {
              migration = unmigrated(confirmed);
            }
            if (migration.patch) {
              const persisted = await savePersistedChatSettingsPatchIfCurrent(
                confirmed,
                migration.patch,
                qwenMigrationExpectedAbsent(confirmedRaw),
                qwenMigrationExpectedAbsentPaths(
                  confirmedRaw,
                  migration.patch,
                ),
              );
              migration = {
                ...migration,
                settings: persisted.settings,
                patch: persisted.applied ? migration.patch : null,
                migratedModelIds: persisted.applied
                  ? migration.migratedModelIds
                  : [],
              };
            }
          } catch {
            // Never hydrate values the server did not accept; fall back to what was actually read.
            migration = unmigrated(confirmed ?? settings);
          }
        }
        const hydratedSettings = migration.settings;
        let applied = false;
        set((state) => {
          if (state.settingsHydrated) {
            return state;
          }
          applied = true;
          const nextState: Partial<ChatRuntimeStore> = {
            settingsHydrated: true,
            ...getHydratedPresetState(
              hydratedSettings,
              state,
              hydrationVersions.presets,
            ),
            ...getHydratedSettingsState(
              hydratedSettings,
              state,
              hydrationVersions,
            ),
          };
          return nextState;
        });
        if (applied) {
          cacheHydratedSettings(hydratedSettings, hydrationVersions);
          mirroredSettingsHydrated = true;
          // Only an authoritative read can say a mirrored field is unset; else backfill clobbers.
          if (fromServer) backfillMirroredSettings(hydratedSettings);
          // After the backfill, so a startup edit wins over the stored value.
          flushPreHydrationSettings();
          if (checkpointMovedDuringConfirm) {
            scheduleLegacyQwenDefaultsRetry(null);
          }
          replayUnconfirmedThreadSettings();
        }
      } catch {
        // Treat as hydrated so later setParams calls reach saveSettingsPatch, which toasts.
        warnSettingsPersistenceFailure();
        mirroredSettingsHydrated = true;
        flushPreHydrationSettings();
        // Independent of this endpoint: replay the last session's row writes anyway.
        replayUnconfirmedThreadSettings();
        set({ settingsHydrated: true });
      } finally {
        settingsHydrationPromise = null;
      }
    })();
    return settingsHydrationPromise;
  },
  beginModelLoading: (phase) => {
    const lease = chatModelLifecycleGate.tryAcquire(phase);
    if (lease !== null) {
      set({ modelLoading: true });
    }
    return lease;
  },
  endModelLoading: (lease) => {
    if (chatModelLifecycleGate.release(lease)) {
      set({ modelLoading: false });
    }
  },
  setLoadingModelPick: (pick) =>
    set({
      loadingModelPick: pick
        ? { ...pick, selectionSuperseded: false }
        : null,
    }),
  clearLoadingModelPick: (expected) =>
    set((state) => {
      const current = state.loadingModelPick;
      if (
        !current ||
        current.id !== expected.id ||
        current.ggufVariant !== expected.ggufVariant ||
        current.nativePathToken !== expected.nativePathToken
      ) {
        return state;
      }
      return { loadingModelPick: null };
    }),
  setModelRequiresTrustRemoteCode: (modelRequiresTrustRemoteCode) =>
    set({ modelRequiresTrustRemoteCode }),
  setParams: (params, options) => {
    set((state) => {
      // The local load path can move params.checkpoint via setParams() before setCheckpoint.
      const checkpointChanged = state.params.checkpoint !== params.checkpoint;
      const fromModelDefaults = options?.fromModelDefaults === true;
      const outgoing = checkpointChanged
        ? rememberOutgoingModel(state, state.params)
        : null;
      // An interactive load reaches setCheckpoint later, so replay the model's settings here.
      noteLoadedContext(params.checkpoint, options?.maxTokensCap);
      const replayed = checkpointChanged || fromModelDefaults;
      const incomingParams = fromModelDefaults
        ? { ...params, minPMode: withoutActiveThreadParams(state, params).minPMode }
        : params;
      const nextParams = getReplayedParams(
        state.rememberParamsPerModel,
        outgoing ?? state.paramsByModel,
        incomingParams,
        params.checkpoint,
        replayed,
        options?.maxTokensCap,
      );
      // The chat's pinned sampling goes on top of the replay; live store only.
      const effective = replayed
        ? restoreThreadScopedParams(nextParams)
        : nextParams;
      const changedParams = getChangedInferenceParams(
        nextParams,
        state.params,
        !fromModelDefaults,
        options?.minPChoiceEdited === true,
      );
      const queuedSettingsChanged =
        options?.minPChoiceEdited === true ||
        shouldAdvanceQueuedSettingsEpoch(
          state.params,
          effective,
          options?.trackQueuedSettings !== false,
        );
      const persistingGlobally =
        options?.persist !== false && state.settingsHydrated;
      // Chat-owned sampling edits reach neither defaults nor model memory, even pre-hydration.
      const sharedParams =
        options?.persist !== false
          ? withoutCapturedThreadEdits(changedParams, fromModelDefaults)
          : changedParams;
      const paramsByModel = getParamsByModelAfterEdit(
        state,
        outgoing,
        nextParams,
        sharedParams,
        options?.persist !== false && !fromModelDefaults,
      );
      if (persistingGlobally) {
        persistParamEdit(
          sharedParams,
          checkpointChanged ? null : paramsByModel,
          nextParams.checkpoint,
        );
        noteThreadScopedDefaults(sharedParams);
      } else if (fromModelDefaults && !state.settingsHydrated) {
        noteModelDefaultsBeforeHydration(
          nextParams.checkpoint,
          options?.migrateOwnedGlobalQwenDefaults === true,
        );
      }
      return {
        params: effective,
        ...(paramsByModel ? { paramsByModel } : {}),
        ...(queuedSettingsChanged
          ? { queuedSettingsEpoch: state.queuedSettingsEpoch + 1 }
          : {}),
        ...(checkpointChanged
          ? { contextUsage: null, contextUsageByThreadId: {} }
          : {}),
      };
    });
    if (options?.fromModelDefaults === true) {
      const retryState = get();
      const ownsPersistedGlobal =
        options.migrateOwnedGlobalQwenDefaults === true;
      scheduleLegacyQwenDefaultsRetry(
        ownsPersistedGlobal ? params.checkpoint : null,
        ownsPersistedGlobal && !retryState.rememberParamsPerModel,
      );
    }
  },
  setCustomPresets: (customPresets) =>
    set(() => {
      customPresetsMutationVersion += 1;
      saveSettingsPatch({ customPresets });
      return { customPresets };
    }),
  setActivePreset: (activePreset) =>
    set(() => {
      activePresetMutationVersion += 1;
      saveSettingsPatch({ activePreset });
      return { activePreset };
    }),
  setActivePresetSource: (activePresetSource) => {
    let returnedToBuiltInDefault = false;
    set((state) => {
      returnedToBuiltInDefault =
        activePresetSource === "builtin-default" &&
        state.activePresetSource !== "builtin-default";
      activePresetSourceMutationVersion += 1;
      saveSettingsPatch({ activePresetSource });
      return { activePresetSource };
    });
    if (returnedToBuiltInDefault) {
      // Defer a microtask so the final slider edit lands before the migration inspects it.
      scheduleLegacyQwenDefaultsRetry(
        useChatRuntimeStore.getState().params.checkpoint,
      );
    }
  },
  setModels: (models) => set({ models }),
  setLoras: (loras) => set({ loras, loraInventorySettled: true }),
  setThreadRunning: (threadId, running, options) =>
    set((state) => {
      const next = { ...state.runningByThreadId };
      const nextLocal = { ...state.localRunByThreadId };
      const nextOwner = { ...state.runOwnerByThreadId };
      const owners = state.runOwnerByThreadId[threadId] ?? [];
      const local = options?.local !== false;
      if (running) {
        next[threadId] = true;
        if (options?.owner) {
          nextOwner[threadId] = [...owners, { owner: options.owner, local }];
        }
        // Any local owner keeps the key counted, so an external run must not clear it.
        if (local) {
          nextLocal[threadId] = true;
        } else if (!owners.some((o) => o.local)) {
          delete nextLocal[threadId];
        }
      } else {
        const remaining = options?.owner
          ? owners.filter((o) => o.owner !== options.owner)
          : [];
        if (options?.owner && remaining.length === owners.length) return state;
        // An ownerless clear must not speak for runs that own the key.
        if (!options?.owner && owners.length > 0) return state;
        if (remaining.length > 0) {
          nextOwner[threadId] = remaining;
          if (remaining.some((o) => o.local)) {
            nextLocal[threadId] = true;
          } else {
            delete nextLocal[threadId];
          }
        } else {
          delete next[threadId];
          delete nextLocal[threadId];
          delete nextOwner[threadId];
        }
      }
      return {
        runningByThreadId: next,
        localRunByThreadId: nextLocal,
        runOwnerByThreadId: nextOwner,
      };
    }),
  adoptDefaultThreadRun: (threadId) =>
    set((state) => {
      const key = "__default";
      if (!threadId || threadId === key) return state;
      // Two first turns can share "__default"; moving wholesale could hand over a sibling's handle.
      if ((state.runOwnerByThreadId[key]?.length ?? 0) > 1) return state;
      const moved: Partial<ChatRuntimeStore> = {};
      const move = <T,>(
        map: Record<string, T>,
        name: keyof ChatRuntimeStore,
      ) => {
        const entry = map[key];
        if (entry === undefined || map[threadId] !== undefined) return;
        const next = { ...map };
        delete next[key];
        next[threadId] = entry;
        (moved as Record<string, unknown>)[name as string] = next;
      };
      move(state.runningByThreadId, "runningByThreadId");
      move(state.localRunByThreadId, "localRunByThreadId");
      move(state.runOwnerByThreadId, "runOwnerByThreadId");
      move(state.cancelByThreadId, "cancelByThreadId");
      move(state.serverCancelByThreadId, "serverCancelByThreadId");
      move(state.toolStatusByThreadId, "toolStatusByThreadId");
      move(
        state.activeDiffusionCanvasByThreadId,
        "activeDiffusionCanvasByThreadId",
      );
      return Object.keys(moved).length > 0 ? moved : state;
    }),
  runKeyForOwner: (fallbackKey, owner) => {
    for (const [key, entries] of Object.entries(get().runOwnerByThreadId)) {
      if (entries.some((e) => e.owner === owner)) return key;
    }
    return fallbackKey;
  },
  registerThreadCancel: (threadId, cancel) =>
    set((state) => {
      const next = { ...state.cancelByThreadId };
      next[threadId] = cancel;
      return { cancelByThreadId: next };
    }),
  clearThreadCancel: (threadId, cancel) =>
    set((state) => {
      if (!(threadId in state.cancelByThreadId)) return state;
      if (cancel && state.cancelByThreadId[threadId] !== cancel) return state;
      const next = { ...state.cancelByThreadId };
      delete next[threadId];
      return { cancelByThreadId: next };
    }),
  registerThreadServerCancel: (threadId, cancel) =>
    set((state) => {
      const next = { ...state.serverCancelByThreadId };
      next[threadId] = [...(state.serverCancelByThreadId[threadId] ?? []), cancel];
      return { serverCancelByThreadId: next };
    }),
  // Unresolved ids share "__default", so remove only this run's cancel.
  clearThreadServerCancel: (threadId, cancel) =>
    set((state) => {
      const current = state.serverCancelByThreadId[threadId];
      if (current === undefined) return state;
      const remaining =
        cancel === undefined ? [] : current.filter((c) => c !== cancel);
      if (remaining.length === current.length) return state;
      const next = { ...state.serverCancelByThreadId };
      if (remaining.length > 0) {
        next[threadId] = remaining;
      } else {
        delete next[threadId];
      }
      return { serverCancelByThreadId: next };
    }),
  setAutoTitle: (autoTitle) =>
    set((state) => {
      setScalarSettingVersion("autoTitle", autoTitle, state.autoTitle);
      return { autoTitle };
    }),
  setHfToken: (hfToken) => useHfTokenStore.getState().setToken(hfToken),
  setModelsError: (modelsError) => set({ modelsError }),
  setLastModelLoadError: (lastModelLoadError) => set({ lastModelLoadError }),
  setCheckpoint: (modelId, ggufVariant, options) => {
    let scheduleQwenMigration = false;
    set((state) => {
      // Only external selections persist; a stale local id would race the freshly loaded model.
      saveLastExternalCheckpoint(isExternalModelId(modelId) ? modelId : null);
      // Only disarm research when the connection cannot drive it.
      const clampsDeepResearch =
        isExternalModelId(modelId) && !externalModelSupportsStudioTools(modelId);
      if (clampsDeepResearch) {
        saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
      }
      const checkpointChanged = state.params.checkpoint !== modelId;
      if (checkpointChanged && !state.settingsHydrated) {
        unownedCheckpointBeforeHydration = modelId;
      }
      scheduleQwenMigration = checkpointChanged && state.settingsHydrated;
      const outgoing =
        checkpointChanged && options?.persist !== false
          ? rememberOutgoingModel(state, state.params)
          : null;
      const baseParams = getReplayedParams(
        state.rememberParamsPerModel,
        outgoing ?? state.paramsByModel,
        state.params,
        modelId,
        checkpointChanged,
        options?.maxTokensCap,
      );
      let nextMaxTokens = baseParams.maxTokens;
      if (checkpointChanged && isExternalModelId(modelId)) {
        const parsed = parseExternalModelId(modelId);
        const provider = parsed
          ? useExternalProvidersStore
              .getState()
              .providers.find((p) => p.id === parsed.providerId)
          : null;
        // Only when the provider is known: the 32,768 fallback would lower a value permanently.
        if (provider) {
          const cap = getExternalMaxOutputTokens(
            provider.providerType,
            parsed?.modelId,
            provider.maxOutputTokens,
          );
          if (nextMaxTokens > cap) {
            nextMaxTokens = cap;
          }
        }
      }
      const nextGgufVariant = ggufVariant ?? null;
      const nextDeepResearchEnabled = clampsDeepResearch
        ? false
        : state.deepResearchEnabled;
      const queuedSettingsChanged = shouldAdvanceQueuedSettingsEpoch(
        {
          checkpoint: state.params.checkpoint,
          maxTokens: state.params.maxTokens,
          ggufVariant: state.activeGgufVariant,
          deepResearchEnabled: state.deepResearchEnabled,
        },
        {
          checkpoint: modelId,
          maxTokens: nextMaxTokens,
          ggufVariant: nextGgufVariant,
          deepResearchEnabled: nextDeepResearchEnabled,
        },
        options?.trackQueuedSettings !== false,
      );
      const nextParams = {
        ...baseParams,
        checkpoint: modelId,
        maxTokens: nextMaxTokens,
      };
      const restoredParams = checkpointChanged
        ? restoreThreadScopedParams(nextParams)
        : nextParams;
      return {
        params: restoredParams,
        loadingModelPick:
          checkpointChanged && state.loadingModelPick
            ? { ...state.loadingModelPick, selectionSuperseded: true }
            : state.loadingModelPick,
        ...getReplayStatePatch(state, nextParams, outgoing, baseParams),
        activeGgufVariant: nextGgufVariant,
        ...(queuedSettingsChanged
          ? { queuedSettingsEpoch: state.queuedSettingsEpoch + 1 }
          : {}),
        // Provenance and the spec-fallback reason describe the replaced model; clear together.
        ...(checkpointChanged
          ? {
              contextUsage: null,
              contextUsageByThreadId: {},
              activeModelIsLocal: false,
              specFallbackReason: null,
              mmprojFallbackReason: null,
              specDrafterKind: null,
            }
          : {}),
        ...(clampsDeepResearch ? { deepResearchEnabled: false } : {}),
      };
    });
    // No load or status follows an external pick, so schedule the migration here.
    if (scheduleQwenMigration) {
      scheduleLegacyQwenDefaultsRetry(null);
    }
  },
  // Re-apply the incoming thread's usage: background runs never wrote the visible value.
  setActiveThreadId: (activeThreadId) =>
    set((state) => ({
      activeThreadId,
      activeThreadEpoch: state.activeThreadEpoch + 1,
      contextUsage: activeThreadId
        ? (state.contextUsageByThreadId[activeThreadId] ?? null)
        : null,
    })),
  applyThreadScopedSettings: (threadId, settings) =>
    set((state) => {
      settings = normalizeSavedThreadScopedSettings(settings);
      flushThreadScopedSettingsWrite();
      // Edits made while this chat's snapshot was in flight are kept and stored on the chat.
      const heldFields = new Set<string>();
      let heldEffort: ReasoningEffort | undefined;
      if (threadId !== null && threadId === pendingPairingThreadId) {
        for (const edit of heldThreadScopedEdits) {
          heldFields.add(edit.field);
          if (edit.field === "reasoningEffort" && edit.value !== undefined) {
            heldEffort = edit.value as ReasoningEffort;
          }
        }
        heldThreadScopedEdits = [];
        pendingPairingThreadId = null;
        pairingWindowDefaultsThreadId = null;
        closeThreadScopedPairingGate(threadId);
      } else if (
        threadId !== null ||
        pendingPairingThreadId === null ||
        pendingPairingThreadId !== state.activeThreadId
      ) {
        releaseHeldThreadScopedEdits();
      }
      // The updater's return merges last, so set the flag here rather than trusting earlier calls.
      const pending = pendingPairingThreadId !== null;
      if (threadScopedSettingsThreadId === null && threadId === null) {
        return state.threadScopedSettingsPending === pending
          ? state
          : { ...state, threadScopedSettingsPending: pending };
      }
      if (threadScopedSettingsThreadId === null) {
        // Capture the pre-window value: a held edit belongs to its chat, not the defaults.
        const captured = readThreadScopedSettings(state) as Record<
          string,
          unknown
        >;
        const beforeWindow = (pairingWindowDefaults ??
          globalThreadScopedDefaults) as Record<string, unknown> | null;
        for (const field of heldFields) {
          if (hydratedDefaultsByHeldField.has(field)) {
            captured[field] = hydratedDefaultsByHeldField.get(field);
          } else if (beforeWindow && field in beforeWindow) {
            captured[field] = beforeWindow[field];
          } else {
            delete captured[field];
          }
          hydratedDefaultsByHeldField.delete(field);
        }
        // A pin in the live effort must not become the default for snapshot-less chats.
        if (pinOwnsLiveReasoningEffort(state) && !heldFields.has("reasoningEffort")) {
          const onRecord = reasoningEffortOnRecord();
          if (onRecord === undefined) delete captured.reasoningEffort;
          else captured.reasoningEffort = onRecord;
        }
        globalThreadScopedDefaults = captured as ThreadScopedSettings;
      }
      threadScopedSettingsThreadId = threadId;
      explicitlyEditedThreadFields.clear();
      constraintSuppressedThreadFields.clear();
      const stored = hasThreadScopedSettings(settings)
        ? (settings as ThreadScopedSettings)
        : null;
      activeThreadScopedSettings = stored;
      const nextState: Partial<ChatRuntimeStore> = {};
      const target = nextState as Record<string, unknown>;
      const applied: Record<string, unknown> = {};
      const paramsPatch: Record<string, unknown> = {};
      for (const key of THREAD_SCOPED_SETTING_KEYS) {
        if (heldFields.has(key)) {
          if (key === "reasoningEffort" && heldEffort !== undefined) {
            applied[key] = heldEffort;
            activeThreadScopedSettings = { ...activeThreadScopedSettings, reasoningEffort: heldEffort };
            if (pinOwnsLiveReasoningEffort(state)) effortDisplacedByPin = heldEffort;
            else nextState.reasoningEffort = heldEffort;
          } else {
            applied[key] = readThreadScopedValue(state, key);
          }
          continue;
        }
        // Full access was accepted via a dialog, so a switch must not drop it.
        if (key === "permissionMode" && state.permissionMode === "full") {
          const underneath =
            stored?.permissionMode ??
            globalThreadScopedDefaults?.permissionMode ??
            loadPermissionMode();
          if (underneath !== "full") applied[key] = underneath;
          continue;
        }
        // setCheckpoint clears deep research only in the store; openai_codex is the exception.
        if (
          key === "deepResearchEnabled" &&
          (externalCheckpointRefusesDeepResearch(state.params.checkpoint) ||
            state.incognito)
        ) {
          continue;
        }
        // A key the snapshot omits falls back to defaults, not the outgoing chat's value.
        const value = firstSetThreadScopedValue(
          stored?.[key],
          globalThreadScopedDefaults?.[key],
        );
        if (value === undefined) continue;
        applied[key] = value;
        // The pin outranks the chat: record the incoming level but do not apply it.
        if (key === "reasoningEffort" && pinOwnsLiveReasoningEffort(state)) {
          effortDisplacedByPin = value as ReasoningEffort;
          continue;
        }
        if (isSameThreadScopedValue(value, readThreadScopedValue(state, key))) {
          continue;
        }
        if (isThreadScopedParamKey(key)) {
          paramsPatch[key] = value;
        } else {
          target[key] = value;
        }
      }
      if (hasKeys(paramsPatch)) {
        nextState.params = { ...state.params, ...paramsPatch };
      }
      // Kimi: the enforcing effect does not rerun on a thread switch, so drop thinking here.
      if (
        isKimiCheckpoint(state.params.checkpoint) &&
        (applied.toolsEnabled ?? state.toolsEnabled) === true &&
        (applied.reasoningEnabled ?? state.reasoningEnabled) === true
      ) {
        applied.reasoningEnabled = false;
        if (state.reasoningEnabled !== false) target.reasoningEnabled = false;
      }
      if (threadId !== null && (stored === null || heldFields.size > 0)) {
        scheduleThreadScopedSettingsWrite(
          threadId,
          stored === null ? sanitizeThreadScopedSettings(applied) : null,
        );
      }
      if (!hasKeys(nextState)) {
        return state.threadScopedSettingsPending === pending
          ? state
          : { ...state, threadScopedSettingsPending: pending };
      }
      if (nextState.permissionMode !== undefined) {
        nextState.confirmToolCalls =
          nextState.permissionMode === "ask" ||
          nextState.permissionMode === "auto";
      }
      return {
        ...nextState,
        threadScopedSettingsPending: pending,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setActiveProjectId: (activeProjectId) => set({ activeProjectId }),
  setIncognito: (incognito) => {
    if (incognito) saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
    set(
      incognito
        ? { incognito, deepResearchEnabled: false }
        : { incognito },
    );
  },
  setSettingsPanelOpen: (settingsPanelOpen) => set({ settingsPanelOpen }),
  setEditingMessageId: (id) => set({ editingMessageId: id }),
  clearCheckpoint: () => {
    saveLastExternalCheckpoint(null);
    saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
    const restoredEffort = pinHoldsLiveEffort()
      ? takeEffortDisplacedByPin()
      : null;
    return set((state) => ({
      queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      ...(() => {
        const outgoing = rememberOutgoingModel(state, state.params);
        return outgoing ? { paramsByModel: outgoing } : {};
      })(),
      params: {
        ...state.params,
        checkpoint: "",
      },
      activeGgufVariant: null,
      // Unknown, not null: null reads as "was evicted".
      residentCheckpoint: undefined,
      activeModelIsLocal: false,
      activeLoadId: null,
      activeNativePathToken: null,
      activeNativePathExpiresAtMs: null,
      ...loadedContextFields(null),
      modelRequiresTrustRemoteCode: false,
      contextUsage: null,
      contextUsageByThreadId: {},
      supportsReasoning: false,
      reasoningAlwaysOn: false,
      reasoningEnabled: true,
      reasoningStyle: "enable_thinking",
      reasoningEffort: restoredEffort ?? state.reasoningEffort,
      supportsReasoningOff: false,
      reasoningEffortLevels: ["low", "medium", "high"],
      supportsPreserveThinking: false,
      supportsTools: false,
      supportsBuiltinWebSearch: false,
      supportsBuiltinCodeExecution: false,
      supportsBuiltinImageGeneration: false,
      supportsBuiltinWebFetch: false,
      toolsEnabled: false,
      codeToolsEnabled: false,
      imageToolsEnabled: false,
      deepResearchEnabled: false,
      mcpEnabledForChat: false,
      webFetchToolsEnabled: false,
      ragEnabled: false,
      toolStatusByThreadId: {},
      toolLiveOutput: {},
      toolFullOutput: {},
      activeDiffusionCanvasByThreadId: {},
      kvCacheDtype: null,
      mlxKvQuant: null,
      loadedMlxKvQuantRequested: null,
      mlxKvQuantReason: null,
      chatTemplateOverrideReason: null,
      mlxKvQuantNote: null,
      mlxInt8Prefill: false,
      loadedMlxInt8PrefillRequested: false,
      loadedKvCacheDtype: null,
      speculativeType: readPersistedSpeculativeType(),
      loadedSpeculativeType: null,
      specFallbackReason: null,
      mmprojFallbackReason: null,
      specDrafterKind: null,
      specDraftNMax: null,
      loadedSpecDraftNMax: null,
      nParallel: null,
      loadedNParallel: null,
      reasoningBudget: -1,
      loadedReasoningBudget: null,
      loadedReasoningBudgetRequested: null,
      reasoningBudgetMessage: "",
      loadedReasoningBudgetMessage: null,
      loadedReasoningBudgetMessageRequested: null,
      nBatch: null,
      loadedNBatch: null,
      loadedLlamaExtraArgs: null,
      nUbatch: null,
      loadedNUbatch: null,
      specDraftCacheDtype: null,
      loadedSpecDraftCacheDtype: null,
      loadMode: null,
      loadedLoadMode: null,
      ctxCheckpoints: null,
      loadedCtxCheckpoints: null,
      cacheRam: null,
      loadedCacheRam: null,
      tensorParallel: false,
      loadedTensorParallel: null,
  loadedDisableVision: null,
      disableVision: false,
      loadedVisionDisabledByUser: null,
      gpuMemoryMode: readPersistedGpuMemoryMode(),
      loadedGpuMemoryMode: null,
      loadedCpuFallback: false,
      gpuLayers: GPU_LAYERS_AUTO,
      loadedGpuLayers: null,
      nCpuMoe: 0,
      loadedNCpuMoe: null,
      splitRatio: null,
      loadedSplitRatio: null,
      ggufLayerCount: null,
      moeLayerCount: null,
      selectedGpuIds: null,
      selectedGpuIndexKind: null,
      loadedGpuIds: null,
      loadedGpuIndexKind: null,
      loadedIsMultimodal: false,
      loadedIsDiffusion: false,
      customContextLength: null,
      loadedCustomContextLength: null,
      defaultChatTemplate: null,
      chatTemplateOverride: null,
      loadedChatTemplateOverride: null,
      pendingImageEditReference: null,
    }));
  },
  setReasoningEnabled: (reasoningEnabled, options) =>
    set((state) => {
      if (options?.persist !== false) {
        saveBool(CHAT_REASONING_ENABLED_KEY, reasoningEnabled);
      } else {
        noteConstraintSuppressedThreadField("reasoningEnabled");
      }
      return {
        reasoningEnabled,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setLastOpenRouterChosenModel: (lastOpenRouterChosenModel) =>
    set({ lastOpenRouterChosenModel }),
  setReasoningStyle: (reasoningStyle) =>
    set((state) => ({
      reasoningStyle,
      queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
    })),
  setRememberedParamsForModel: (modelId, patch) =>
    set((state) => {
      if (!modelId || !hasKeys(patch)) return state;
      const merged = { ...state.paramsByModel[modelId], ...patch };
      const next = { ...state.paramsByModel, [modelId]: merged };
      // Only the moved keys: the server merges per key, so a full snapshot clobbers other tabs.
      // Not gated on hydration, unlike a defaults write: this is a typed edit for one model that
      // can only set the keys it names, and gating it dropped an edit made while the initial
      // /api/chat/settings request was out.
      saveSettingsPatch({ inferenceParamsByModel: { [modelId]: patch } });
      // Hold pre-hydration edits, or the in-flight response rebuilds paramsByModel without them.
      if (!state.settingsHydrated) {
        modelParamEditsBeforeHydration.set(modelId, {
          ...modelParamEditsBeforeHydration.get(modelId),
          ...patch,
        });
      }
      // Apply via the switch restore: the open chat's systemPrompt outranks the model's.
      const live = state.params.checkpoint === modelId;
      const liveParams = live
        ? restoreThreadScopedParams({ ...state.params, ...patch })
        : null;
      const liveChanged =
        liveParams !== null &&
        shouldAdvanceQueuedSettingsEpoch(state.params, liveParams);
      if (liveChanged) getChangedInferenceParams(liveParams, state.params);
      return {
        paramsByModel: trackParamsByModel(state, next, modelId) ?? next,
        // Bump the epoch so queued prompts and pending pastes see the old prompt is stale.
        ...(liveChanged
          ? {
              params: liveParams,
              queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
            }
          : {}),
      };
    }),
  setReasoningEffort: (reasoningEffort) => {
    const checkpoint = useChatRuntimeStore.getState().params.checkpoint;
    const { effortByModel, setModelReasoningEffort } =
      useModelReasoningEffortStore.getState();
    const pinned = checkpoint ? effortByModel[checkpoint] : undefined;
    if (checkpoint && pinned !== undefined && pinned !== reasoningEffort) {
      setModelReasoningEffort(checkpoint, null);
    }
    set((state) => {
      if (pinned !== reasoningEffort) effortDisplacedByPin = null;
      setScalarSettingVersion(
        "reasoningEffort",
        reasoningEffort,
        state.reasoningEffort,
      );
      return {
        reasoningEffort,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    });
  },
  setPreserveThinking: (preserveThinking) =>
    set((state) => {
      setScalarSettingVersion(
        "preserveThinking",
        preserveThinking,
        state.preserveThinking,
      );
      notePreserveThinkingPreference(preserveThinking);
      return {
        preserveThinking,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setToolsEnabled: (toolsEnabled, options) =>
    set((state) => {
      if (options?.persist !== false) {
        saveBool(CHAT_TOOLS_ENABLED_KEY, toolsEnabled);
      } else {
        noteConstraintSuppressedThreadField("toolsEnabled");
      }
      if (toolsEnabled) saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
      return {
        ...(toolsEnabled
          ? { toolsEnabled, deepResearchEnabled: false }
          : { toolsEnabled }),
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setKeepModelsLoaded: (keepModelsLoaded) => set({ keepModelsLoaded }),
  setCodeToolsEnabled: (codeToolsEnabled) =>
    set((state) => {
      saveBool(CHAT_CODE_TOOLS_ENABLED_KEY, codeToolsEnabled);
      if (codeToolsEnabled) saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
      return {
        ...(codeToolsEnabled
          ? { codeToolsEnabled, deepResearchEnabled: false }
          : { codeToolsEnabled }),
        codeToolsDeclinedUnderFullAccess:
          !codeToolsEnabled && state.permissionMode === "full",
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setImageToolsEnabled: (imageToolsEnabled) =>
    set((state) => {
      saveBool(CHAT_IMAGE_TOOLS_ENABLED_KEY, imageToolsEnabled);
      if (imageToolsEnabled) saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
      return {
        ...(imageToolsEnabled
          ? { imageToolsEnabled, deepResearchEnabled: false }
          : { imageToolsEnabled }),
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setDeepResearchEnabled: (deepResearchEnabled) =>
    set((state) => {
      saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, deepResearchEnabled);
      const permissionMode =
        threadScopedOverride("permissionMode") ?? loadPermissionMode();
      if (deepResearchEnabled) {
        saveBool(CHAT_TOOLS_ENABLED_KEY, false);
        saveBool(CHAT_IMAGE_TOOLS_ENABLED_KEY, false);
        saveBool(CHAT_CODE_TOOLS_ENABLED_KEY, false);
        saveBool(CHAT_MCP_ENABLED_KEY, false);
        saveBool(CHAT_WEB_FETCH_TOOLS_ENABLED_KEY, false);
      }
      return deepResearchEnabled
        ? {
            deepResearchEnabled,
            toolsEnabled: false,
            codeToolsEnabled: false,
            codeToolsDeclinedUnderFullAccess: false,
            imageToolsEnabled: false,
            mcpEnabledForChat: false,
            webFetchToolsEnabled: false,
            bypassPermissions: false,
            permissionMode,
            confirmToolCalls:
              permissionMode === "ask" || permissionMode === "auto",
            queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
          }
        : {
            deepResearchEnabled,
            queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
          };
    }),
  setResearchWebsitePolicy: (researchWebsitePolicy) =>
    set((state) => {
      saveResearchWebsitePolicy(researchWebsitePolicy);
      return {
        researchWebsitePolicy,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setResearchModelTimeoutSeconds: (researchModelTimeoutSeconds) =>
    set((state) => {
      const seconds = isSupportedResearchModelTimeout(
        researchModelTimeoutSeconds,
      )
        ? researchModelTimeoutSeconds
        : DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS;
      persistSetting(CHAT_DEEP_RESEARCH_MODEL_TIMEOUT_KEY, String(seconds));
      return {
        researchModelTimeoutSeconds: seconds,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setCollapseHtmlArtifacts: (collapseHtmlArtifacts) =>
    set(() => {
      saveBool(CHAT_COLLAPSE_HTML_ARTIFACTS_KEY, collapseHtmlArtifacts);
      return { collapseHtmlArtifacts };
    }),
  setAllowArtifactNetworkAccess: (allowArtifactNetworkAccess) =>
    set(() => {
      saveBool(
        CHAT_ALLOW_ARTIFACT_NETWORK_ACCESS_KEY,
        allowArtifactNetworkAccess,
      );
      return { allowArtifactNetworkAccess };
    }),
  setSearchImages: (searchImages) =>
    set(() => {
      saveBool(CHAT_SEARCH_IMAGES_KEY, searchImages);
      return { searchImages };
    }),
  setMcpEnabledForChat: (mcpEnabledForChat) =>
    set((state) => {
      saveBool(CHAT_MCP_ENABLED_KEY, mcpEnabledForChat);
      if (mcpEnabledForChat) saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
      return {
        ...(mcpEnabledForChat
          ? { mcpEnabledForChat, deepResearchEnabled: false }
          : { mcpEnabledForChat }),
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setConfirmToolCalls: (confirmToolCalls) =>
    set((state) => {
      saveBool(CHAT_CONFIRM_TOOL_CALLS_KEY, confirmToolCalls);
      if (state.permissionMode === "full") {
        return {
          confirmToolCalls,
          queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
        };
      }
      const permissionMode: PermissionMode = confirmToolCalls ? "ask" : "off";
      savePermissionMode(permissionMode);
      return {
        confirmToolCalls,
        permissionMode,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setSandboxLevel: (sandboxLevel) =>
    set((state) => {
      saveString(CHAT_SANDBOX_LEVEL_KEY, sandboxLevel);
      return { sandboxLevel, queuedSettingsEpoch: state.queuedSettingsEpoch + 1 };
    }),
  setPermissionMode: (permissionMode) =>
    set((state) => {
      savePermissionMode(permissionMode);
      if (permissionMode === "full") {
        // Full access sends confirm_tool_calls=false; keep the store flag in sync.
        saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
        return {
          permissionMode,
          bypassPermissions: true,
          confirmToolCalls: false,
          deepResearchEnabled: false,
          codeToolsDeclinedUnderFullAccess: codeDeclinedOnEnteringFullAccess(state),
          queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
        };
      }
      const confirmToolCalls =
        permissionMode === "ask" || permissionMode === "auto";
      saveBool(CHAT_CONFIRM_TOOL_CALLS_KEY, confirmToolCalls);
      return {
        permissionMode,
        bypassPermissions: false,
        confirmToolCalls,
        codeToolsDeclinedUnderFullAccess: false,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setBypassPermissions: (bypassPermissions) =>
    // Not persisted: a reload must not keep the bypass unaccepted.
    set((state) => {
      if (bypassPermissions) {
        saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
        return {
          bypassPermissions,
          permissionMode: "full" as PermissionMode,
          confirmToolCalls: false,
          deepResearchEnabled: false,
          codeToolsDeclinedUnderFullAccess: codeDeclinedOnEnteringFullAccess(state),
          queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
        };
      }
      const permissionMode =
        threadScopedOverride("permissionMode") ?? loadPermissionMode();
      return {
        bypassPermissions,
        permissionMode,
        confirmToolCalls: permissionMode === "ask" || permissionMode === "auto",
        codeToolsDeclinedUnderFullAccess: false,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setBypassConfirmOpen: (bypassConfirmOpen) =>
    set(() => ({ bypassConfirmOpen })),
  allowToolAlways: (sessionId, toolName) =>
    set((state) => {
      const current = state.alwaysAllowToolsBySession.get(sessionId);
      if (current?.has(toolName)) return state;
      const next = new Map(state.alwaysAllowToolsBySession);
      next.set(sessionId, new Set(current ?? []).add(toolName));
      return { alwaysAllowToolsBySession: next };
    }),
  setToolConfirmation: (
    toolCallId,
    approvalId,
    sessionId,
    autoAllowKey,
    imageDisclosure,
  ) =>
    set((state) => ({
      toolConfirmations: {
        ...state.toolConfirmations,
        [toolCallId]: { approvalId, sessionId, autoAllowKey, imageDisclosure },
      },
    })),
  clearToolConfirmation: (toolCallId) =>
    set((state) => {
      if (
        !Object.prototype.hasOwnProperty.call(
          state.toolConfirmations,
          toolCallId,
        )
      ) {
        return state;
      }
      const next = { ...state.toolConfirmations };
      delete next[toolCallId];
      return { toolConfirmations: next };
    }),
  setWebFetchToolsEnabled: (webFetchToolsEnabled) =>
    set((state) => {
      saveBool(CHAT_WEB_FETCH_TOOLS_ENABLED_KEY, webFetchToolsEnabled);
      return {
        webFetchToolsEnabled,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setRagEnabled: (ragEnabled) =>
    set((state) => {
      // The only thread-scoped setting with no global slot, so no persist helper reaches it.
      if (ragEnabled !== state.ragEnabled) captureThreadScopedEdit("ragEnabled");
      return {
        ragEnabled,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setProjectAttachmentTarget: (projectAttachmentTarget) =>
    set(() => {
      saveString(CHAT_PROJECT_ATTACHMENT_TARGET_KEY, projectAttachmentTarget);
      return { projectAttachmentTarget };
    }),
  setThreadProjectAttachmentTarget: (threadId, target) =>
    set((state) => {
      if (threadId === null) {
        pendingAttachmentTargetClaim += 1;
      }
      return {
        projectAttachmentTargetByThread: {
          ...state.projectAttachmentTargetByThread,
          [threadId ?? PENDING_CHAT_ATTACHMENT_KEY]: target,
        },
      };
    }),
  adoptPendingProjectAttachmentTarget: (threadId, claim) =>
    set((state) => {
      if (claim !== undefined && claim !== pendingAttachmentTargetClaim) {
        return state;
      }
      const pending =
        state.projectAttachmentTargetByThread[PENDING_CHAT_ATTACHMENT_KEY];
      if (
        pending === undefined ||
        threadId in state.projectAttachmentTargetByThread
      ) {
        return state;
      }
      const next = { ...state.projectAttachmentTargetByThread };
      delete next[PENDING_CHAT_ATTACHMENT_KEY];
      next[threadId] = pending;
      return { projectAttachmentTargetByThread: next };
    }),
  clearPendingProjectAttachmentTarget: () =>
    set((state) => {
      const byThread = state.projectAttachmentTargetByThread;
      if (!(PENDING_CHAT_ATTACHMENT_KEY in byThread)) {
        return state;
      }
      pendingAttachmentTargetClaim += 1;
      const next = { ...byThread };
      delete next[PENDING_CHAT_ATTACHMENT_KEY];
      return { projectAttachmentTargetByThread: next };
    }),
  setRememberParamsPerModel: (rememberParamsPerModel) =>
    set((state) => {
      setScalarSettingVersion(
        "rememberParamsPerModel",
        rememberParamsPerModel,
        state.rememberParamsPerModel,
      );
      const snapshot = pickRememberedParams(
        withoutActiveThreadParams(state, state.params),
      );
      const paramsByModel = trackParamsByModel(
        state,
        getRememberedParamsPatch(
          rememberParamsPerModel,
          state.paramsByModel,
          state.params.checkpoint,
          snapshot,
          snapshot,
        ),
        state.params.checkpoint,
      );
      if (paramsByModel && state.settingsHydrated && state.params.checkpoint) {
        saveSettingsPatch({
          inferenceParamsByModel: {
            [state.params.checkpoint]: paramsByModel[state.params.checkpoint],
          },
        });
      }
      // The global set may still be the last model's, so write it on turning this off.
      if (!rememberParamsPerModel && state.settingsHydrated) {
        saveSettingsPatch({ inferenceParams: snapshot });
        noteThreadScopedDefaults(snapshot);
      }
      return {
        rememberParamsPerModel,
        ...(paramsByModel ? { paramsByModel } : {}),
      };
    }),
  setRagSource: (ragSource) =>
    set((state) => {
      saveRagSource(ragSource);
      return {
        ragSource,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setRagMode: (ragMode) =>
    set((state) => {
      saveString(CHAT_RAG_MODE_KEY, ragMode);
      return {
        ragMode,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setRagTopK: (ragTopK) =>
    set((state) => {
      saveString(CHAT_RAG_TOP_K_KEY, String(ragTopK));
      return {
        ragTopK,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setRagAutoInject: (ragAutoInject) =>
    set((state) => {
      saveString(CHAT_RAG_AUTOINJECT_KEY, ragAutoInject);
      return {
        ragAutoInject,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setRagAutoInjectMinScore: (ragAutoInjectMinScore) =>
    set((state) => {
      saveString(
        CHAT_RAG_AUTOINJECT_MIN_SCORE_KEY,
        String(ragAutoInjectMinScore),
      );
      return {
        ragAutoInjectMinScore,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setRagOcrScanned: (ragOcrScanned) =>
    set(() => {
      saveBool(CHAT_RAG_OCR_KEY, ragOcrScanned);
      return { ragOcrScanned };
    }),
  setRagCaptionFigures: (ragCaptionFigures) =>
    set(() => {
      saveBool(CHAT_RAG_CAPTION_KEY, ragCaptionFigures);
      return { ragCaptionFigures };
    }),
  setToolStatus: (threadId, status, owner) =>
    set((state) => {
      const next = { ...state.toolStatusByThreadId };
      const entries = state.toolStatusByThreadId[threadId] ?? [];
      const mine = entries.find((e) => e.owner === owner);
      if (!status) {
        // Drop only this run's entry: a sibling under the same key may still run a tool.
        if (mine === undefined) return state;
        const rest = entries.filter((e) => e !== mine);
        if (rest.length > 0) {
          next[threadId] = rest;
        } else {
          delete next[threadId];
        }
      } else {
        if (mine?.status === status) return state;
        const entry = { status, startedAt: Date.now(), owner };
        next[threadId] = mine
          ? entries.map((e) => (e === mine ? entry : e))
          : [...entries, entry];
      }
      return { toolStatusByThreadId: next };
    }),
  appendToolLiveOutput: (toolCallId, text) =>
    set((state) => ({
      toolLiveOutput: {
        ...state.toolLiveOutput,
        [toolCallId]: (state.toolLiveOutput[toolCallId] ?? "") + text,
      },
    })),
  setToolFullOutput: (toolCallId, text) =>
    set((state) => ({
      toolFullOutput: {
        ...state.toolFullOutput,
        [toolCallId]: text,
      },
    })),
  clearToolFullOutput: (toolCallId) =>
    set((state) => {
      if (!(toolCallId in state.toolFullOutput)) {
        return {};
      }
      const next = { ...state.toolFullOutput };
      delete next[toolCallId];
      return { toolFullOutput: next };
    }),
  clearToolLiveOutput: (toolCallId) =>
    set((state) => {
      if (toolCallId === undefined) {
        return Object.keys(state.toolLiveOutput).length
          ? { toolLiveOutput: {} }
          : {};
      }
      if (!(toolCallId in state.toolLiveOutput)) {
        return {};
      }
      const next = { ...state.toolLiveOutput };
      delete next[toolCallId];
      return { toolLiveOutput: next };
    }),
  setActiveDiffusionCanvas: (threadId, canvas) =>
    set((state) => ({
      activeDiffusionCanvasByThreadId: {
        ...state.activeDiffusionCanvasByThreadId,
        [threadId || "__default"]: canvas,
      },
    })),
  clearActiveDiffusionCanvasForThread: (threadId) =>
    set((state) => {
      const key = threadId || "__default";
      if (state.activeDiffusionCanvasByThreadId[key] === undefined) return state;
      const next = { ...state.activeDiffusionCanvasByThreadId };
      delete next[key];
      return { activeDiffusionCanvasByThreadId: next };
    }),
  setGeneratingStatus: (generatingStatus) => set({ generatingStatus }),
  setAutoHealToolCalls: (autoHealToolCalls) =>
    set((state) => {
      setScalarSettingVersion(
        "autoHealToolCalls",
        autoHealToolCalls,
        state.autoHealToolCalls,
      );
      return {
        autoHealToolCalls,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setNudgeToolCalls: (nudgeToolCalls) =>
    set((state) => {
      setScalarSettingVersion(
        "nudgeToolCalls",
        nudgeToolCalls,
        state.nudgeToolCalls,
      );
      return {
        nudgeToolCalls,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setAutoCompactEnabled: (autoCompactEnabled) =>
    set((state) => {
      setScalarSettingVersion(
        "autoCompactEnabled",
        autoCompactEnabled,
        state.autoCompactEnabled,
      );
      return {
        autoCompactEnabled,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setMaxToolCallsPerMessage: (maxToolCallsPerMessage) =>
    set((state) => {
      setScalarSettingVersion(
        "maxToolCallsPerMessage",
        maxToolCallsPerMessage,
        state.maxToolCallsPerMessage,
      );
      return {
        maxToolCallsPerMessage,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setToolCallTimeout: (toolCallTimeout) =>
    set((state) => {
      setScalarSettingVersion(
        "toolCallTimeout",
        toolCallTimeout,
        state.toolCallTimeout,
      );
      return {
        toolCallTimeout,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  // Persisted only on a successful load, so an unapplied pick does not stick.
  setGpuMemoryMode: (gpuMemoryMode) => set({ gpuMemoryMode }),
  setGpuLayers: (gpuLayers) => set({ gpuLayers }),
  setNCpuMoe: (nCpuMoe) => set({ nCpuMoe }),
  setSplitRatio: (splitRatio) => set({ splitRatio }),
  setSelectedGpuIds: (selectedGpuIds, selectedGpuIndexKind = null) =>
    set({
      selectedGpuIds,
      selectedGpuIndexKind:
        selectedGpuIds == null ? null : selectedGpuIndexKind,
    }),
  setExpandQuantizations: (expandQuantizations) => {
    saveBool(CHAT_EXPAND_QUANTIZATIONS_KEY, expandQuantizations);
    set({ expandQuantizations });
  },
  setShowAllQuantizations: (showAllQuantizations) => {
    saveBool(CHAT_SHOW_ALL_QUANTIZATIONS_KEY, showAllQuantizations);
    set({ showAllQuantizations });
  },
  setShowMemoryBar: (showMemoryBar) => {
    saveBool(CHAT_SHOW_MEMORY_BAR_KEY, showMemoryBar);
    set({ showMemoryBar });
  },
  setFitOnDeviceOnly: (fitOnDeviceOnly) => {
    saveBool(MODELS_FIT_ON_DEVICE_ONLY_KEY, fitOnDeviceOnly);
    set({ fitOnDeviceOnly });
  },
  setPendingAudio: (base64, name) =>
    set({ pendingAudioBase64: base64, pendingAudioName: name }),
  clearPendingAudio: () =>
    set({ pendingAudioBase64: null, pendingAudioName: null }),
  setPendingImageEditReference: (pendingImageEditReference) =>
    set({ pendingImageEditReference }),
  clearPendingImageEditReference: () =>
    set({ pendingImageEditReference: null }),
  // Write through to the thread's entry: the history loader runs once per mount.
  setContextUsage: (contextUsage) =>
    set((state) => {
      if (!state.activeThreadId) return { contextUsage };
      const next = { ...state.contextUsageByThreadId };
      if (contextUsage) {
        next[state.activeThreadId] = contextUsage;
      } else {
        delete next[state.activeThreadId];
      }
      return { contextUsage, contextUsageByThreadId: next };
    }),
  setThreadContextUsage: (threadId, usage) =>
    set((state) => ({
      contextUsageByThreadId: {
        ...state.contextUsageByThreadId,
        [threadId]: usage,
      },
    })),
}));

const unsubscribeHfTokenMirror = mirrorHfTokenInto(useChatRuntimeStore);
if (import.meta.hot) {
  import.meta.hot.dispose(unsubscribeHfTokenMirror);
}

export function resolveSpeculativeSettingsForLoad({
  usePersistedPreference = false,
}: {
  usePersistedPreference?: boolean;
} = {}): {
  speculativeType: string | null;
  specDraftNMax: number | null;
} {
  const state = useChatRuntimeStore.getState();
  const speculativeType = usePersistedPreference
    ? readPersistedSpeculativeType()
    : (state.speculativeType ?? readPersistedSpeculativeType());
  return {
    speculativeType,
    specDraftNMax:
      !usePersistedPreference &&
      speculativeType != null &&
      DRAFT_N_MAX_SPEC_TYPES.has(speculativeType)
        ? state.specDraftNMax
        : null,
  };
}
