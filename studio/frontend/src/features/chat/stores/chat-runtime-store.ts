// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { authFetch } from "@/features/auth";
import {
  mirrorHfTokenInto,
  useHfTokenStore,
} from "@/features/hub/stores/hf-token-store";
import { loadedContextFields } from "@/features/model-picker/model-config/per-model-config";
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
import { getExternalMaxOutputTokens } from "../provider-capabilities";
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
import {
  DEFAULT_AUTO_COMPACT_ENABLED,
  DEFAULT_COMPACTION_HEADROOM_RATIO,
  DEFAULT_CONTEXT_POLICY,
  type LocalContextPolicy,
} from "../utils/auto-compaction";
import { preserveThinkingDefaultFromLoad } from "../lib/resolve-preserve-thinking-default";
import {
  THREAD_SCOPED_PARAM_KEYS,
  THREAD_SCOPED_SETTING_KEYS,
  type ThreadScopedSettingKey,
  type ThreadScopedSettings,
  hasThreadScopedSettings,
  isThreadOwnedSettingKey,
  isThreadScopedParamKey,
  sanitizeThreadScopedSettings,
} from "../utils/thread-scoped-settings";
import {
  chatModelLifecycleGate,
  type ModelLifecycleLease,
} from "../utils/model-lifecycle-gate";
import { shouldAdvanceQueuedSettingsEpoch } from "../utils/queued-settings-epoch";
import type { MmprojFallbackReason } from "../types/api";
import type { ResearchWebsitePolicy } from "../types/research";
import {
  CHAT_GPU_MEMORY_MODE_KEY,
  CHAT_SPECULATIVE_TYPE_KEY,
} from "./chat-runtime-keys";
import { useExternalProvidersStore } from "./external-providers-store";
import { PLUS_MENU_PINS_STORAGE_KEY } from "./plus-menu-prefs-store";

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
export const CHAT_ARTIFACTS_ENABLED_KEY = "unsloth_chat_artifacts_enabled";
export const CHAT_SHOW_CANVAS_MENU_ITEM_KEY =
  "unsloth_chat_show_canvas_menu_item";
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

/** Local tool-call gate: "ask" every call, "auto" only high-risk ones, "off" never but keeps
 *  the sandbox, "full" drops both and is session-only. */
export type PermissionMode = "ask" | "auto" | "off" | "full";
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
// Only the model-agnostic intents (auto/ngram/off) persist: a saved drafter mode no-ops on a
// model with no MTP head or DSpark sidecar.
// The model-specific drafter modes and spec_draft_n_max stay session-only.
const PERSISTED_SPEC_MODES = new Set(["auto", "ngram", "off"]);

export type RagSource = { type: "thread" } | { type: "kb"; kbId: string };

/** Where the composer files an attachment in a project chat; `project` indexes it.
 *  Key a choice made in a chat with no id yet lives under until it gets one. */
export const PENDING_CHAT_ATTACHMENT_KEY = "__pending__";

/** Bumped when the pending entry changes hands, so a composer can tell it is still its own. */
let pendingAttachmentTargetClaim = 0;

export function readPendingAttachmentTargetClaim(): number {
  return pendingAttachmentTargetClaim;
}

export type RagMode = "hybrid" | "lexical" | "dense";

export const DEFAULT_RAG_SOURCE: RagSource = { type: "thread" };
export const DEFAULT_RAG_MODE: RagMode = "hybrid";
export const DEFAULT_RAG_TOP_K = 5;
// `auto` forces retrieval for smaller models (<=9B); `on`/`off` force it.
export type RagAutoInject = "auto" | "on" | "off";
export const DEFAULT_RAG_AUTOINJECT: RagAutoInject = "auto";
export const DEFAULT_RAG_AUTOINJECT_MIN_SCORE = 0.7;
// OCR scanned/image-only PDF pages at ingest; off skips the extra vision pass (only matters with a vision chat model).
export const DEFAULT_RAG_OCR = true;
// Describe figures/charts in PDFs at ingest so they become searchable; no-op without a vision model.
export const DEFAULT_RAG_CAPTION = true;
export const DEFAULT_RESEARCH_WEBSITE_POLICY: ResearchWebsitePolicy = {
  allowedDomains: [],
  blockedDomains: [],
};
export const DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS = 900;

/** 0 (unlimited) or a finite budget; the patch and run routes drop anything else and 400. */
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

// External picks ride in `params.checkpoint` as `external::<providerId>::<modelId>`.
// PersistedChatSettings omits `checkpoint`: only the local side is mirrored by /status.
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

/**
 * Two checkpoints naming the same model.
 *
 * normalizeModelIdentity, not toLowerCase: it folds case for repository ids and
 * the case-insensitive path forms while preserving it for POSIX paths, where
 * /home/u/Models/qwen3.8 and /home/u/models/qwen3.8 are two different files.
 */
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
  // Opaque ids compare exactly: a provider qualifies one and an encoded POSIX
  // path sits inside the other, and normalizeModelIdentity would fold both.
  if (isOpaqueModelRef(left) || isOpaqueModelRef(right)) {
    return left === right;
  }
  return normalizeModelIdentity(left) === normalizeModelIdentity(right);
}

// A checkpoint that arrived without any adoption signal: restored from
// localStorage at startup, or picked by the user while the settings GET was
// still in flight. Chat settings are installation-wide, so neither may claim a
// global snapshot another browser wrote for a different model.
let unownedCheckpointBeforeHydration: string | null = null;

function saveLastExternalCheckpoint(value: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (value && isExternalModelId(value)) {
      window.localStorage.setItem(LAST_EXTERNAL_CHECKPOINT_KEY, value);
    } else {
      // Cleared on a switch to a local/empty checkpoint, or the next refresh overrides it.
      window.localStorage.removeItem(LAST_EXTERNAL_CHECKPOINT_KEY);
    }
  } catch {
    // Storage quota / private-mode failures are non-fatal; the selection just will not survive a refresh.
  }
}

// "enable_thinking_effort" is a gate plus a level (GLM-5.2 high|max): it reuses the
// reasoning_effort dropdown but, unlike gpt-oss, can be turned off entirely.
export type ReasoningStyle =
  | "enable_thinking"
  | "reasoning_effort"
  | "enable_thinking_effort";
/** One live DiffusionGemma denoising snapshot: canvas text at a step of a block (0-based; total = steps in block). */
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

// Setting writes coalesce into one pendingPatch, flushed on a trailing debounce and on
// beforeunload so a pending patch survives tab close.
type SettingsPatch = Parameters<typeof savePersistedChatSettingsPatch>[0];

const SETTINGS_DEBOUNCE_MS = 400;
let pendingPatch: SettingsPatch = {};
let pendingTimer: ReturnType<typeof setTimeout> | null = null;
let inflightFlush: Promise<void> = Promise.resolve();

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Discriminated unions, not partial patches: merging a `thread` pick into a stored `kb` one
// keeps `kbId`, which the backend's thread variant forbids.
const ATOMIC_SETTING_KEYS = new Set<string>(["ragSource"]);

// Maps of per-model objects, merged one level further in: two edits to different fields in
// one debounce window must not replace each other.
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
    // extra="forbid" refuses the whole body on one bad field, so requeueing as-is would break
    // every later save. Keep the fields the server did not name; reschedule only if it shrank.
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

// Flushes handed to the network and not yet answered: pendingPatch and pendingTimer are both empty across that window.
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

// A wedged PATCH must not hold a send open; past this the run goes ahead on the value the server already has.
const SETTINGS_FLUSH_TIMEOUT_MS = 2000;

/** Flush the debounced patch and wait. The backend reads some settings out of SQLite at call
 *  time, so a message sent inside the window would run on the pre-toggle value. */
export async function flushPendingChatSettings(): Promise<void> {
  const queued = pendingTimer !== null || Object.keys(pendingPatch).length > 0;
  // Not just what is queued: the debounce may already have handed its patch to an unanswered
  // request, leaving both empty while the backend still reads the old value.
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

// keepalive lets the PUT outlive the unload; without it the last slider drag is dropped.
function flushSettingsOnPageHidden(terminal: boolean): void {
  if (pendingTimer !== null) clearTimeout(pendingTimer);
  // Terminal events only: the beacon PATCHes the row directly, and a row that does not exist
  // yet answers 404. Chats just sent are excluded, since each beacon takes a higher seq.
  const sentNewest = new Set<string>();
  const flushedThreadId = threadSettingsWriteThreadId;
  flushThreadScopedSettingsWrite(terminal);
  if (terminal && flushedThreadId !== null) sentNewest.add(flushedThreadId);
  // An edit made while its chat's read was out lives only in heldThreadScopedEdits, and effect
  // cleanup is not guaranteed during unload, so send it from here.
  if (terminal) {
    const heldThreadId = pendingPairingThreadId;
    void commitHeldThreadScopedEditsToTheirThread(true);
    if (heldThreadId !== null) sentNewest.add(heldThreadId);
    // And anything an earlier visibilitychange flushed the normal way, which this event would
    // otherwise leave to a cancelled fetch.
    beaconUnsettledThreadSettingsWrites(sentNewest);
  }
  // An edit still waiting on hydration is a user edit, so send it rather than let the next
  // session hydrate over it.
  drainPreHydrationPatch();
  if (Object.keys(pendingPatch).length === 0) return;
  // Counted like any other flush: a keepalive PUT still on the wire would land
  // after the compare-and-set and restore the legacy row.
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
  // beforeunload never fires on discarded mobile tabs, so pagehide and visibilitychange-hidden
  // are added too. Safe: the pending patch is swapped out before the request.
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
  /** Value to seed the backend with, for a setting stored under older keys. */
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

/** Settings describing the installation rather than the browser: each pairs a localStorage
 *  slot with the /api/chat/settings field that carries it to another browser. */
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
  artifactsEnabled: {
    storageKey: CHAT_ARTIFACTS_ENABLED_KEY,
    ...BOOLEAN_SETTING,
  },
  showCanvasMenuItem: {
    storageKey: CHAT_SHOW_CANVAS_MENU_ITEM_KEY,
    ...BOOLEAN_SETTING,
    // A profile predating the visibility flag keeps Canvas shown through its plus-menu pin.
    readForBackfill: () =>
      readStorageValue(CHAT_SHOW_CANVAS_MENU_ITEM_KEY) !== null ||
      readStorageValue(PLUS_MENU_PINS_STORAGE_KEY) !== null
        ? loadShowCanvasMenuItem()
        : undefined,
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
    // A profile predating permission levels holds only the confirm toggle; the level derives.
    readForBackfill: () =>
      readStorageValue(CHAT_PERMISSION_MODE_KEY) !== null ||
      loadOptionalBool(CHAT_CONFIRM_TOOL_CALLS_KEY) !== null
        ? loadPermissionMode()
        : undefined,
  },
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

/** An edit made before the initial GET lands is held and replayed after hydration, since the
 *  request would race it. The mutation version bumps either way, fencing the field. */
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

/** Move any held startup edits onto the outgoing patch. */
function drainPreHydrationPatch(): void {
  if (!preHydrationPatch) return;
  mergePatch(pendingPatch, preHydrationPatch);
  preHydrationPatch = null;
}

/** Replay the edits made while the initial GET was still in flight. */
function flushPreHydrationSettings(): void {
  if (!preHydrationPatch) return;
  const patch = preHydrationPatch;
  preHydrationPatch = null;
  saveSettingsPatch(patch);
}

/** Write a setting to localStorage and, when mirrored, to the backend. Storage is the
 *  synchronous boot cache; the backend copy is what another browser reads. */
function persistSetting(key: string, raw: string): void {
  const mirrored = MIRRORED_SETTING_BY_STORAGE_KEY.get(key);
  const writeGlobal = () => {
    // Before hydration the cache says nothing about the server's value, so an explicit write is
    // recorded even when it changes nothing locally.
    if (!mirroredSettingsHydrated || readStorageValue(key) !== raw) {
      mirrorSettingToBackend(key, raw);
    }
    writeStorageValue(key, raw);
  };
  if (mirrored && captureThreadScopedEdit(mirrored.field, writeGlobal)) return;
  writeGlobal();
}

const THREAD_SETTINGS_DEBOUNCE_MS = 400;

/** the thread whose snapshot the store shows, or null on a new chat. */
let threadScopedSettingsThreadId: string | null = null;
/** that thread's stored values, for the load paths that read a setting outside the store. */
let activeThreadScopedSettings: ThreadScopedSettings | null = null;
// Captured on the way into a thread: edits made with a chat open must not move the defaults.
let globalThreadScopedDefaults: ThreadScopedSettings | null = null;
let threadSettingsWriteTimer: ReturnType<typeof setTimeout> | null = null;
let threadSettingsWriteThreadId: string | null = null;
// Only when pinning: the values applied on open, not whatever a model-capability effect
// leaves in the store by the time the debounce fires.
let threadSettingsWriteSnapshot: ThreadScopedSettings | null = null;

/** Where a thread-scoped key's live value is: sampling under `params`, the rest fields. */
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
  // Drops "full" with it: a stored bypass would come back without the warning dialog.
  return sanitizeThreadScopedSettings(source);
}

// Keeps a model load from re-applying the global default over the pills the chat is running with.
export function threadScopedOverride<K extends ThreadScopedSettingKey>(
  key: K,
): ThreadScopedSettings[K] | undefined {
  // activeThreadScopedSettings only refreshes on the debounce, so for 400ms it holds pre-edit
  // values a load would revert and persist; prefer the store. A pending pin answers first.
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

/** Fields the user just set on the open chat, as opposed to ones a model's capabilities
 *  moved. The preservations below must not undo a choice only now made. */
const explicitlyEditedThreadFields = new Set<string>();

/** Fields a provider constraint moved WITHOUT persisting (Kimi's search turns thinking off
 *  with `{persist:false}`). That value is the provider's, so no snapshot write may save it. */
const constraintSuppressedThreadFields = new Set<string>();

/** Both pills of Kimi's search/thinking exclusion; the only non-persisting setters. */
const CONSTRAINT_SUPPRESSIBLE_KEYS = ["reasoningEnabled", "toolsEnabled"] as const;

function noteConstraintSuppressedThreadField(
  field: (typeof CONSTRAINT_SUPPRESSIBLE_KEYS)[number],
): void {
  if (useChatRuntimeStore.getState().activeThreadId === null) return;
  constraintSuppressedThreadFields.add(field);
}

/** Is the stored value the one to keep, rather than the constraint-derived live one? */
function keepsStoredValueUnderConstraint(
  key: (typeof CONSTRAINT_SUPPRESSIBLE_KEYS)[number],
  threadId: string,
  settings: ThreadScopedSettings,
): boolean {
  return (
    threadId === threadScopedSettingsThreadId &&
    constraintSuppressedThreadFields.has(key) &&
    // A choice the user has since made themselves is theirs, constraint or not.
    !explicitlyEditedThreadFields.has(key) &&
    typeof activeThreadScopedSettings?.[key] === "boolean" &&
    settings[key] !== activeThreadScopedSettings[key]
  );
}

// The pills chat-page clamps to the selected model's capabilities.
const CLAMPED_PILL_KEYS = [
  "toolsEnabled",
  "codeToolsEnabled",
  "imageToolsEnabled",
  "webFetchToolsEnabled",
] as const;
type ClampedPillKey = (typeof CLAMPED_PILL_KEYS)[number];

/** Sampling keys share `params` with the model's recommendation, so a load in this window
 *  would pin the model's value onto the chat. Captured at the edit; last entry wins. */
function heldThreadScopedParamValue(key: string): unknown {
  for (let i = heldThreadScopedEdits.length - 1; i >= 0; i -= 1) {
    if (heldThreadScopedEdits[i].field === key) {
      return heldThreadScopedEdits[i].value;
    }
  }
  return undefined;
}

/** First value actually set. `??` cannot do it: a cleared `seed` is null, and null there is
 *  the chat's own choice rather than a missing key. */
function firstSetThreadScopedValue(...values: unknown[]): unknown {
  return values.find((value) => value !== undefined);
}

/** Put back the sampling keys the open chat owns, so a load or status poll leaves it running
 *  what it stored. Only an unpinned chat falls through to the model's values. */
function restoreThreadScopedParams(params: InferenceParams): InferenceParams {
  const kept: Record<string, unknown> = {};
  for (const key of THREAD_SCOPED_PARAM_KEYS) {
    // Not ||, and not ?? either: 0, "", -1 and a cleared seed's null are all values a user sets on purpose.
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

/** Take the open chat's own values out of a snapshot about to be remembered against a model,
 *  or its sampling replays into the next chat opened there. A chat whose read is still out
 *  owns its keys too, so gating on the applied id alone leaked them. */
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
    // Only a key this chat actually owns; the rest are already the model's.
    const held = heldThreadScopedParamValue(key);
    if (held === undefined && threadScopedOverride(key) === undefined) continue;
    // For a held key the installation copy can still be null and the store no longer holds the
    // pre-edit value; the sample taken when the window opened is that value.
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

/** Drop the sampling keys the open chat just took: they must reach neither the installation
 *  defaults nor this model's memory, both shared with every other chat. */
function withoutCapturedThreadEdits(
  changedParams: PersistedInferenceParams,
  fromModelDefaults: boolean,
): PersistedInferenceParams {
  const shared: PersistedInferenceParams = {};
  for (const [key, value] of Object.entries(changedParams)) {
    if (
      isThreadScopedParamKey(key) &&
      !fromModelDefaults &&
      // By value: this runs inside the updater, so a read-back would find the pre-edit value.
      captureThreadScopedEdit(key, null, value)
    ) {
      continue;
    }
    (shared as Record<string, unknown>)[key] = value;
  }
  return shared;
}

/** Level the in-memory installation defaults with what was just written: a chat with no
 *  snapshot falls back to them, so a stale copy runs the previous model's sampling. */
function noteThreadScopedDefaults(shared: PersistedInferenceParams): void {
  let next: Record<string, unknown> | null = null;
  for (const [key, value] of Object.entries(shared)) {
    if (!isThreadScopedParamKey(key)) continue;
    // Held field: the pairing capture restores it from the pre-window sample, so without this
    // the in-memory defaults stay behind the server's all session.
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
  // ragSource is the only object among these, and its variants carry at most a kb id.
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
  // The write replaces the row and the sanitizer drops a live "full", so any pill toggled
  // under Full access would otherwise erase the chat's stored level.
  if (
    settings.permissionMode === undefined &&
    threadId === threadScopedSettingsThreadId &&
    activeThreadScopedSettings?.permissionMode !== undefined
  ) {
    settings.permissionMode = activeThreadScopedSettings.permissionMode;
  }
  // Same for deep research, which apply() also holds back, or toggling any other pill erases
  // the stored true. Unless the user just cleared it by enabling Search, Code or Images.
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
  // And for thinking, which a model that cannot stop thinking forces on: that true is the
  // model's, so persisting it would erase a chat's stored false.
  if (
    threadId === threadScopedSettingsThreadId &&
    !explicitlyEditedThreadFields.has("reasoningEnabled") &&
    activeThreadScopedSettings?.reasoningEnabled === false &&
    settings.reasoningEnabled !== false &&
    useChatRuntimeStore.getState().reasoningAlwaysOn
  ) {
    settings.reasoningEnabled = false;
  }
  // Same for every pill the model-selection pass clamps off without touching the snapshot:
  // each pill's own capability rule is exactly when the user could not have done it.
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
  // And for pills a provider constraint moved without persisting: the store holds the
  // provider's value, so the chat keeps the one it was stored with.
  if (keepsStoredValueUnderConstraint("reasoningEnabled", threadId, settings)) {
    settings.reasoningEnabled = activeThreadScopedSettings?.reasoningEnabled;
  }
  if (keepsStoredValueUnderConstraint("toolsEnabled", threadId, settings)) {
    settings.toolsEnabled = activeThreadScopedSettings?.toolsEnabled;
  }
  if (threadId === threadScopedSettingsThreadId) {
    activeThreadScopedSettings = settings;
  }
  // Spent: this snapshot has taken them into account.
  explicitlyEditedThreadFields.clear();
  return settings;
}

const THREAD_SETTINGS_REPLAY_KEY = "unsloth_chat_thread_settings_replay";
const THREAD_SETTINGS_REPLAY_TIMEOUT_MS = 10_000;

/** Snapshots a terminal event sent but could not confirm, kept where the next session finds
 *  them: the beacon cannot await, and a row still being created answers 404. */
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

/** Safe to run always: the body carries that session's own seq, so a replay of a write that
 *  did land is refused rather than reverting anything newer. */
export function replayUnconfirmedThreadSettings(): void {
  // Once per session: both hydration outcomes call it, and sending each body twice would race
  // two writes carrying the same seq.
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
    // Bounded: every settings write waits on these, so a socket that never settles would block
    // persistence for the whole session.
    const timeout = new AbortController();
    const timer = setTimeout(() => timeout.abort(), THREAD_SETTINGS_REPLAY_TIMEOUT_MS);
    const request = authFetch(`/api/chat/threads/${encodeURIComponent(threadId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: timeout.signal,
    })
      .finally(() => clearTimeout(timer))
      // Only an ok response means it landed: authFetch resolves for 404 and 5xx too, and the
      // missing-row case this exists for is exactly the one that 404s.
      .then((res) => {
        // Only the body this request carried: a terminal event can store a newer one meanwhile, and
        // dropping that because an older replay succeeded would lose the newest edit.
        if (res.ok) forgetReplayedThreadSettings(threadId, body);
      })
      .catch(() => undefined);
    sent.push(request);
  }
  // Every snapshot write waits for this: the replay carries the PREVIOUS session's writer id,
  // so the server applies it whenever it arrives and could revert a fresh edit.
  threadSettingsReplaySettled = Promise.all(sent).then(() => undefined);
}

// Resolved once the previous session's unconfirmed writes have been answered.
let threadSettingsReplaySettled: Promise<void> = Promise.resolve();
let threadSettingsReplayStarted = false;

/** `expected` narrows the drop to the exact body the caller sent; a caller that has itself
 *  written the row passes nothing, since its values supersede the entry. */
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

/** The ensure-then-update chain cannot finish during unload, so the row write goes out with
 *  keepalive and whatever it cannot confirm is left for the next session. */
function sendThreadScopedSettingsBeacon(
  threadId: string,
  snapshot: ThreadScopedSettings | null,
  merge = false,
): void {
  // A merge carries only what the user touched, for a chat whose snapshot was never read; a
  // replacement built from the defaults on screen would erase the rest of its row.
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
  // The beacon carries the newest values but skips the chain, so an older write would land
  // after it. The ticket stands down queued writes; the abort ends the one out.
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

// One chain per thread: the write REPLACES settings_json, so two unordered writes pick a
// winner and a slow first landing after a fast second reverts the user.
const threadSettingsWriteChains = new Map<string, Promise<unknown>>();
// Newest snapshot per thread, so a queued write can tell it was superseded and skip its PATCH.
const threadSettingsWriteTickets = new Map<string, number>();

// The request each thread currently has out, so a newer snapshot can stand it down.
const threadSettingsWriteAborts = new Map<string, AbortController>();

/** Stamps each snapshot write so the server refuses this tab's older ones: a keepalive sent
 *  on unload can be undone by a PATCH the server already had. A counter, never a clock:
 *  comparing machines' clocks silently loses the slower browser's edits. */
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

/** Resolves true when the snapshot reached the row, so callers can tell it landed. */
function writeThreadScopedSettings(
  threadId: string,
  snapshot: ThreadScopedSettings | null,
): Promise<boolean> {
  // Built now, not inside the chain: the snapshot must describe the store at the moment of the edit.
  const settings = buildThreadScopedSnapshot(threadId, snapshot);
  // Stamped with the snapshot, not the request: the seq says when the edit happened, not when its turn came up.
  const settingsSeq = nextThreadSettingsSeq();
  const ticket = takeThreadSettingsWriteTicket(threadId);
  const previous = threadSettingsWriteChains.get(threadId) ?? Promise.resolve();
  const next = previous
    .catch(() => undefined)
    // Last session's unconfirmed writes go first: one carries a different writer id, so nothing
    // on the server orders it against this edit.
    .then(() => threadSettingsReplaySettled)
    .catch(() => undefined)
    .then(async () => {
      // Superseded while queued: sending this would undo the newer snapshot, so it counts as landed for the caller.
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
        // This session's values are now the row's, so a leftover replay entry from the last one would revert them.
        forgetReplayedThreadSettings(threadId);
        return true;
      } catch {
        // The chat still behaves as edited; only the snapshot for the next visit is lost. An abort
        // lands here too, deliberately: a newer snapshot won.
        if (!controller.signal.aborted) warnSettingsPersistenceFailure();
        // An abort means a newer write won and is tracked in its place; a real failure means nothing reached the row.
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

/** A chat edited, left and re-entered has its PATCH in flight, and a GET overtaking it reads
 *  the pre-edit snapshot. Pending debounce included, or the read races the timer. */
export async function settleThreadScopedSettingsForCopy(
  threadId: string,
): Promise<void> {
  // An edit made while this chat's read was out lives in neither the debounce nor the chain.
  // Only for a caller about to read the row server side.
  if (pendingPairingThreadId === threadId) {
    await commitHeldThreadScopedEditsToTheirThread();
  }
  // A flushed replacement write that failed resolves false rather than throwing, so the chain
  // alone cannot tell a saved edit from a lost one.
  if (!(await awaitThreadScopedSettingsWrite(threadId))) {
    throw new Error("This chat's settings could not be saved before copying it");
  }
}

/** Resolves false only when a write for this chat is known to have failed. */
export async function awaitThreadScopedSettingsWrite(
  threadId: string,
): Promise<boolean> {
  if (threadSettingsWriteThreadId === threadId) {
    flushThreadScopedSettingsWrite();
  }
  const chain = threadSettingsWriteChains.get(threadId);
  if (chain === undefined) return true;
  // A rejection is the held-edit merge path, which throws on failure.
  const landed = await chain.catch(() => false);
  return landed !== false;
}

/** Every row write already started has landed. Not a flush: an unfired debounce is left
 *  alone. Loops rather than spinning a fixed count, since the chain ends in a dynamic
 *  import whose cost is a property of the machine. */
export async function awaitStartedThreadScopedSettingsWrites(): Promise<void> {
  // A settled chain can leave a newer one behind for the same chat, so this repeats until the
  // map is empty. Bounded, so a self-rescheduling write fails an assertion, not hangs.
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

/** Some browsers fire visibilitychange(hidden) then pagehide: the first flush clears the
 *  pending snapshot, so holding it lets the terminal path resend with keepalive. */
function trackUnsettledThreadSettingsWrite(
  threadId: string,
  snapshot: ThreadScopedSettings | null,
): void {
  // Identity, not value: two ordinary edits both carry a null snapshot, so comparing values
  // let one request's settle delete another's tracking.
  const entry: UnsettledThreadSettingsWrite = { snapshot };
  unsettledThreadSettingsWrites.set(threadId, entry);
  void writeThreadScopedSettings(threadId, snapshot).then((landed) => {
    // Only a write that reached the row stops being unsettled; dropping it on failure left
    // nothing for a terminal event to beacon.
    if (landed && unsettledThreadSettingsWrites.get(threadId) === entry) {
      unsettledThreadSettingsWrites.delete(threadId);
    }
  });
}

type UnsettledThreadSettingsWrite = { snapshot: ThreadScopedSettings | null };

/** Flushed but not yet acknowledged, so a terminal event can still resend it. */
const unsettledThreadSettingsWrites = new Map<
  string,
  UnsettledThreadSettingsWrite
>();

/** `alreadySent` names the threads just beaconed: they carry the newest values and every
 *  beacon takes a higher seq, so resending an older snapshot would let the stale one win. */
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
      // Tracked like a manual flush: once the timer fires there is no pending debounce for a
      // terminal event to find, so the write in flight is the only copy and teardown kills it.
      trackUnsettledThreadSettingsWrite(pendingThreadId, pendingSnapshot);
    }
  }, THREAD_SETTINGS_DEBOUNCE_MS);
}

// The chat whose snapshot is in flight, plus edits made before it landed: only the read can
// say whether they are the chat's or the defaults'. Writing them globally moved every
// other chat's default and was then overwritten by the arriving snapshot.
let pendingPairingThreadId: string | null = null;
/** The store's thread-scoped values as they stood when the current pairing began. */
let pairingWindowDefaults: ThreadScopedSettings | null = null;
/** And the chat it was sampled for, so a retry does not resample over its own edit. */
let pairingWindowDefaultsThreadId: string | null = null;
let heldThreadScopedEdits: {
  field: string;
  writeGlobal: (() => void) | null;
  /** Sampling keys only, captured by value; see heldThreadScopedParamValue. */
  value?: unknown;
}[] = [];

/** Start of the window: the read for `threadId` is out but its snapshot has not landed. */
export function beginThreadScopedPairing(threadId: string): void {
  if (pendingPairingThreadId === threadId) return;
  releaseHeldThreadScopedEdits();
  pendingPairingThreadId = threadId;
  // What the installation defaults were before this chat could edit anything; nothing later
  // can recover it. Once per chat, not per attempt: a retry runs with the edit in the store.
  if (pairingWindowDefaultsThreadId !== threadId) {
    pairingWindowDefaultsThreadId = threadId;
    // Switching straight between saved chats the store still holds the OUTGOING chat's values,
    // so the capture made when it was paired is the source of defaults.
    pairingWindowDefaults =
      threadScopedSettingsThreadId === null
        ? readThreadScopedSettings(useChatRuntimeStore.getState())
        : globalThreadScopedDefaults;
  }
  openThreadScopedPairingGate(threadId);
}

/** Sends are held while this is true; see the field's own note. */
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

/** Release ONE chat's gate: releasing all let a run started for A be freed by B's pairing
 *  ending. The composer flag is separate and tracks only the chat on screen. */
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

// Per chat, not one for all: a run started for A must not be released by B's pairing ending.
const pairingSettledByThreadId = new Map<
  string,
  { promise: Promise<void>; resolve: () => void }
>();

/** Resolves once `threadId`'s own settings are known; the adapter awaits it so a run cannot
 *  start on the installation defaults (a chat stored "ask" would run "off"). False means
 *  the wait ran out and the store now describes some other chat, so callers must stop. */
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

// Longer than the read can take (retries, timeouts, and a give-up that opens the gate), so
// a run is only refused for a chat whose pairing was genuinely abandoned.
// THREAD_READ_RETRIES retries, each bounded by THREAD_READ_TIMEOUT_MS and spaced
// THREAD_READ_RETRY_MS apart.
const THREAD_PAIRING_WAIT_MS = 30_000;

/** The chat turned out to own no snapshot: send the held edits to the defaults, as before. */
export function releaseHeldThreadScopedEdits(): void {
  const held = heldThreadScopedEdits;
  const threadId = pendingPairingThreadId;
  heldThreadScopedEdits = [];
  pendingPairingThreadId = null;
  pairingWindowDefaultsThreadId = null;
  // The answer is in: this chat owns no snapshot, so the defaults ARE its settings and a run waiting on it can go.
  closeThreadScopedPairingGate(threadId);
  for (const edit of held) {
    // Written to the defaults now, so the value hydration held back is history.
    hydratedDefaultsByHeldField.delete(edit.field);
    edit.writeGlobal?.();
  }
}

/** The user left before the read finished. The edit belongs to its chat, so write it there:
 *  replaying it into the installation defaults would move every snapshot-less chat. */
export function commitHeldThreadScopedEditsToTheirThread(
  keepalive = false,
): Promise<void> {
  const threadId = pendingPairingThreadId;
  const held = heldThreadScopedEdits;
  heldThreadScopedEdits = [];
  pendingPairingThreadId = null;
  // pairingWindowDefaultsThreadId is deliberately NOT cleared: a failed read re-pairs the same
  // chat with the edit in the store, and resampling would take it for a default. The gate
  // stays shut too, since the store now shows the chat the user moved to.
  closeThreadScopedPairingGate(null);
  if (threadId === null || held.length === 0) return Promise.resolve();
  const changes = heldThreadScopedChanges(held);
  // Read off the store above, so only now that the values are safely captured.
  restoreDefaultsOverCommittedEdits(threadId, held);
  if (keepalive) {
    sendThreadScopedSettingsBeacon(threadId, changes, true);
    return Promise.resolve();
  }
  // Returned rather than fired and forgotten: forking copies settings_json server side, so it
  // must wait for a held edit to reach the row.
  return mergeThreadScopedSettingsIntoRow(threadId, changes);
}

/** Put the installation values back over the edits just written to the chat being left, or
 *  the next chat takes its temperature and prompt. Only on a real leave: a retried read
 *  or a fork commits the same way while the chat stays open. */
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
    // The server answered for this field while the window was open, so that value is the
    // installation's; the pre-window sample is only this browser's cache.
    const value = hydratedDefaultsByHeldField.has(edit.field)
      ? hydratedDefaultsByHeldField.get(edit.field)
      : before?.[edit.field];
    hydratedDefaultsByHeldField.delete(edit.field);
    // Nothing known to go back to: leaving the edit up is no worse than blanking it.
    if (value === undefined) continue;
    if (isThreadScopedParamKey(edit.field)) {
      params[edit.field] = value;
    } else {
      fields[edit.field] = value;
    }
  }
  if (!hasKeys(fields) && !hasKeys(params)) return;
  // setState, not setParams: these values are already the installation's, and the setter would
  // persist them back to it and to the loaded model's memory.
  useChatRuntimeStore.setState((state) =>
    hasKeys(params)
      ? ({ ...fields, params: { ...state.params, ...params } } as Partial<ChatRuntimeStore>)
      : (fields as Partial<ChatRuntimeStore>),
  );
}

/** What the user actually touched, read off the store, which still holds their edits. */
function heldThreadScopedChanges(
  held: { field: string }[],
): ThreadScopedSettings {
  const edited: Record<string, unknown> = {};
  const live = useChatRuntimeStore.getState();
  // Same reader the snapshot path uses: sampling keys sit under `params`, so a direct field
  // read returns undefined and the sanitizer drops them.
  for (const edit of held) {
    edited[edit.field] = readThreadScopedValue(
      live,
      edit.field as ThreadScopedSettingKey,
    );
  }
  return sanitizeThreadScopedSettings(edited);
}

/** PATCH only `changes`. For when the store cannot be trusted to describe the chat (its read
 *  has not landed), since a replacement built from the defaults would erase the rest. */
async function mergeThreadScopedSettingsIntoRow(
  threadId: string,
  changes: ThreadScopedSettings,
): Promise<void> {
  const settingsSeq = nextThreadSettingsSeq();
  const ticket = takeThreadSettingsWriteTicket(threadId);
  const previous = threadSettingsWriteChains.get(threadId) ?? Promise.resolve();
  const next = previous
    .catch(() => undefined)
    // Last session's unconfirmed writes go first: one carries a different writer id, so nothing
    // on the server orders it against this edit.
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
        // Rethrown as well as toasted: a fork waits on this to know the row holds the edit, and a
        // resolved promise would let it fork the pre-edit snapshot.
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

/** What the server called an installation default, for fields hydration skipped because the
 *  user had just set them inside a chat whose read was out. When the pairing window closes
 *  the default returns to this rather than the browser's pre-hydration copy. */
const hydratedDefaultsByHeldField = new Map<string, unknown>();

/** Is this field an edit waiting on its chat's read, and so not the installation's to set? */
function isHeldThreadScopedField(field: string): boolean {
  return heldThreadScopedEdits.some((edit) => edit.field === field);
}

// Reports whether the edit was taken; with no chat open the caller persists globally as before.
function captureThreadScopedEdit(
  field: string,
  writeGlobal: (() => void) | null = null,
  value?: unknown,
): boolean {
  if (!isThreadOwnedSettingKey(field)) return false;
  const threadId = useChatRuntimeStore.getState().activeThreadId;
  if (threadId === null) return false;
  // Both ids: between a switch and its snapshot arriving the store still holds the old values.
  if (threadId === threadScopedSettingsThreadId) {
    explicitlyEditedThreadFields.add(field);
    // Set by the user now, so the chat stores this rather than what it had before a constraint moved the same field.
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

/** Persist a value a model load applied rather than one the user picked: before hydration
 *  such a write only reflects this browser's cache. `stillCurrent` says it matches the live
 *  store, so a load spanning hydration cannot write its stale capture over the answer. */
function persistLoadDerivedSetting(
  key: string,
  raw: string,
  stillCurrent: boolean,
): void {
  if (!mirroredSettingsHydrated) {
    writeStorageValue(key, raw);
    return;
  }
  // Dropped outright: the cache now holds the hydrated preference too.
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

/** Pill state to apply on a model load: the persisted preference wins in both directions, the
 *  open chat beats the installation default, and with no preference the pills stay off. */
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

// The installation's own answer to the preserve-thinking switch, or null while it has never
// given one. Only hydration and the composer toggle write it.
let storedPreserveThinking: boolean | null = null;

/** Record the preference a stored value or a toggle just expressed. */
function notePreserveThinkingPreference(value: boolean): void {
  storedPreserveThinking = value;
}

/** The backend's family default is a DEFAULT: it seeds the switch where the installation
 *  never answered and never replaces an answer, as resolveToolsEnabledOnLoad does. That is
 *  what makes a cold boot deterministic when the settings GET and the status race. */
export function resolvePreserveThinkingOnLoad(resp: {
  supports_preserve_thinking?: boolean | null;
  preserve_thinking_default?: boolean | null;
}): boolean {
  return storedPreserveThinking ?? preserveThinkingDefaultFromLoad(resp);
}

// The visibility flag shipped after the menu pins, so when absent an explicit Canvas pin wins.
function loadShowCanvasMenuItem(): boolean {
  const stored = loadOptionalBool(CHAT_SHOW_CANVAS_MENU_ITEM_KEY);
  if (stored !== null) return stored;
  if (!canUseStorage()) return false;
  try {
    const raw = localStorage.getItem(PLUS_MENU_PINS_STORAGE_KEY);
    if (raw === null) return false;
    const parsed = JSON.parse(raw) as {
      state?: { pins?: { canvas?: boolean } };
    };
    return parsed.state?.pins?.canvas === true;
  } catch {
    return false;
  }
}

/** "full" is never restored: it disables the sandbox and every confirmation gate, so it needs
 *  the warning dialog each session. First run derives from the legacy confirm toggle. */
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

function loadString(key: string, fallback: string): string {
  return readStorageValue(key) ?? fallback;
}

function saveString(key: string, value: string): void {
  persistSetting(key, value);
}

// Canonicalises any backend value onto the Speculative Decoding modes; legacy backend-only
// aliases map to their closest UI mode.
export function normalizeSpeculativeType(
  v: string | null | undefined,
): string | null {
  if (v == null) return null;
  const s = String(v).trim().toLowerCase();
  if (!s) return null;
  if (s === "auto" || s === "default") return "auto";
  // The same four spellings _LEGACY_SPEC_MODE_MAP reads as "off". A stored override
  // reaches here raw (model-overrides.ts binds speculative_type unnormalized), so
  // falling through to Auto would compare Auto against a status the backend already
  // canonicalised to "off" and re-send /load on every pick.
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
  // Comma-chained legacy values (e.g. from older backend echoes).
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
  // Unknown -> safe fallback to Auto so the dropdown stays controlled.
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

// The user's standing preference, sanitized to the universal set.
export function readPersistedSpeculativeType(): string {
  const raw = loadString(CHAT_SPECULATIVE_TYPE_KEY, "auto");
  return PERSISTED_SPEC_MODES.has(raw) ? raw : "auto";
}

// MTP / null / unknown values stay unwritten and so session-only. Called from the load path,
// so only an applied preference persists.
export function saveSpeculativeType(value: string | null): void {
  if (value && PERSISTED_SPEC_MODES.has(value)) {
    persistLoadDerivedSetting(
      CHAT_SPECULATIVE_TYPE_KEY,
      value,
      useChatRuntimeStore.getState().speculativeType === value,
    );
  }
}

// A standing preference, not per-model: a "manual" choice survives model switches and reloads.
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

/** Persist the GPU Memory mode after a load, but only for a non-diffusion GGUF: non-GGUF has
 *  no such mode and diffusion reports "auto", so neither may clobber the preference. */
export function persistGpuMemoryModeOnLoad(
  resp: { is_gguf?: boolean; is_diffusion?: boolean },
  mode: "auto" | "manual",
): void {
  if (resp.is_gguf && !resp.is_diffusion) saveGpuMemoryMode(mode);
}

// Re-exported from its dependency-free home so existing imports keep working.
export { GPU_LAYERS_AUTO } from "../lib/gpu-placement";

// Round real-valued shares to integers summing to `total`, leftovers to the largest
// fractional parts (largest-remainder method).
function largestRemainder(shares: number[], total: number): number[] {
  const out = shares.map((x) => Math.floor(x));
  let rem = total - out.reduce((a, b) => a + b, 0);
  const byFrac = shares
    .map((x, i) => ({ i, frac: x - Math.floor(x) }))
    .sort((a, b) => b.frac - a.frac);
  for (let k = 0; rem > 0 && k < byFrac.length; k++, rem--) out[byFrac[k].i] += 1;
  return out;
}

// Spread `total` layers over GPUs in proportion to `weights`, summing to `total`; even split
// for all-zero weights. Mirrors llama.cpp's free-VRAM default.
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

// Set GPU `index` to `value` and rebalance so the counts still sum to `total`. Counts are
// sent verbatim, and llama.cpp honors each exactly when gpu_layers == sum(counts).
export function rebalanceSplit(
  total: number,
  counts: number[],
  index: number,
  value: number,
): number[] {
  const v = Math.max(0, Math.min(value, total));
  const out = counts.slice();
  const otherIdx = counts.map((_, i) => i).filter((i) => i !== index);
  // No other GPU to absorb the remainder: this one holds everything.
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

// Validate a persisted gpu_ids pick against the GPUs present now, returning null (automatic)
// when stale, so a saved [1] on a 1-GPU host is not sent and rejected unclearably. A cold
// device cache leaves it alone; an absent namespace is a legacy physical-ID pick.
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

// Store fields derived from a load/status response's GPU-memory settings, shared by every
// load path so the manual-knob round-trip cannot drift.
export function loadedGpuMemoryFields(resp: {
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
  // Meaningful only for a GGUF chat load, and a non-GGUF response still serializes
  // gpu_memory_mode "auto", so gate on is_gguf or a transformers load resets the preference.
  if (!resp.is_gguf) {
    // Clear the GPU pick / offload baseline a prior GGUF load left, else a stale loadedGpuIds
    // reads as dirty. gpuMemoryMode stays but its baseline clears, so Reset keeps the mode.
    // gpuIdsDirty is ungated, so Reset would restore it while the picker is hidden. gpuMemoryMode is
    // kept as the standing preference, but its loaded baseline clears to null.
    return {
      selectedGpuIds: null,
      selectedGpuIndexKind: null,
      loadedGpuIds: null,
      loadedGpuIndexKind: null,
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
  // Keep the user's placement pool editable across hydration; gpu_ids stays the fitted subset.
  const reportedGpuIds = requestedGpuIdsFromResponse(resp);
  const gpuIndexKind =
    reportedGpuIds == null
      ? null
      : cachedPinnableGpuIndexKind(resp.is_diffusion === true);
  // A numeric id is unsafe to adopt once discovery says it is not a physical CUDA/ROCm id or
  // Vulkan ordinal. While discovery is cold the namespace is merely deferred, so keep the
  // pin, or a reload in that window lets llama.cpp fall back to every device.
  const gpuIds =
    reportedGpuIds != null && gpuIndexKind !== null ? reportedGpuIds : null;
  // A shim without --ngl reports Auto while the backend still holds the ask, so recover it:
  // in-memory state survives a reload but not a refresh.
  const droppedSplit = recoverDroppedDiffusionSplit(
    resp.is_diffusion,
    mode,
    resp.diffusion_requested_ngl,
  );
  // The layer/MoE/split knobs apply only in manual mode, so auto must not seed their baselines.
  // In manual the server reports gpu_layers = -1 for Auto, round-tripping the slider.
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
          // Auto ignores these, so reset the editable knobs or a switch back to Manual sends a previous
          // model's values. Diffusion excepted: an "auto" response may be an older shim that dropped
          // a manual split, so restore the ask when the response carries it. Resetting the slider
          // would turn it into manual/-1, unapplyable even after the unsloth_zoo upgrade adds --ngl.
          ...(resp.is_diffusion
            ? droppedSplit != null
              ? { gpuLayers: droppedSplit }
              : {}
            : { gpuLayers: GPU_LAYERS_AUTO }),
          nCpuMoe: 0,
          splitRatio: null,
        };
  return {
    // A diffusion GGUF reporting "auto" ran on the runner's defaults, so an inert manual
    // preference must survive it; "manual" means a split was applied (#7574), so adopt it.
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
    // MoE expert-layer count: the n_cpu_moe slider max, and 0 hides the slider.
    moeLayerCount: resp.n_moe_layers ?? null,
    // The picker reflects the requested placement pool, not a fitted subset.
    selectedGpuIds: gpuIds,
    // gpuIndexKind is `undefined` only in the deferred cache-cold case, since a `null` kind
    // already forced gpuIds to null; normalize to the explicit-null-namespace convention.
    selectedGpuIndexKind: gpuIds == null ? null : (gpuIndexKind ?? null),
    loadedGpuIds: gpuIds,
    loadedGpuIndexKind: gpuIds == null ? null : (gpuIndexKind ?? null),
    ...manualKnobs,
  };
}

// Re-exported so every importer keeps its path; defined in a leaf module a test can load.
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
  // Anthropic-only; optional so pre-cache-stats persisted entries load.
  cacheWriteTokens?: number;
};

/** One live run behind `runningByThreadId[id]`, with the `local` flag it started with so the
 *  model-swap gate can tell llama-server runs from external ones on a shared key. */
type ThreadRunOwner = {
  owner: () => void;
  local: boolean;
};

type ToolStatusEntry = {
  status: string;
  startedAt: number;
  owner?: () => void;
};

type ChatRuntimeStore = {
  settingsHydrated: boolean;
  /** The open chat's settings were asked for but have not arrived, so the store shows the
   *  installation defaults and a run started now would be captured with them. Sending waits. */
  threadScopedSettingsPending: boolean;
  params: InferenceParams;
  /** Last-used sampling params per checkpoint id, replayed on model switch. */
  paramsByModel: Record<string, PersistedInferenceParams>;
  rememberParamsPerModel: boolean;
  customPresets: Preset[];
  activePreset: string;
  activePresetSource: ChatPresetSource;
  models: ChatModelRow[];
  loras: ChatLoraSummary[];
  loraInventorySettled: boolean;
  runningByThreadId: Record<string, boolean>;
  /** Runs decoding on the local llama-server. Swapping the local model neither interrupts an
   *  external chat nor needs its consent, which is why the backend excludes them too. */
  localRunByThreadId: Record<string, boolean>;
  /** Which runs set `runningByThreadId[id]`; see `setThreadRunning`'s `owner`. A list, since
   *  runs with no resolved thread id share the "__default" key. */
  runOwnerByThreadId: Record<string, ThreadRunOwner[]>;
  cancelByThreadId: Record<string, () => void>;
  /** Backend cancels for background threads: `cancelByThreadId` holds only the visible thread's
   *  `cancelRun()`. A list, since unresolved ids share "__default". */
  serverCancelByThreadId: Record<string, (() => void)[]>;
  autoTitle: boolean;
  hfToken: string;
  modelsError: string | null;
  // Set only when a LOAD fails (not refresh/list/unload, which use modelsError), so the attach
  // gates can tell a failed load from "no model picked".
  lastModelLoadError: string | null;
  activeGgufVariant: string | null;
  /** What /api/inference/status says is resident, as opposed to what the picker selected.
   *  undefined until the first read, so the header does not flash "not loaded". */
  residentCheckpoint: string | null | undefined;
  /** Whether the backend loaded the active model from a filesystem path. */
  activeModelIsLocal: boolean;
  loadedContextLength: number | null;
  maxContextLength: number | null;
  nativeContextLength: number | null;
  /** The backend's own is_gguf for the loaded model; null until one loads. Set wherever
   *  loadedContextLength is, so a context never arrives unattributed. */
  loadedIsGguf: boolean | null;
  /** The backend's own is_mlx for the loaded model; null until one loads. The platform
   *  cannot answer it: the worker serves native-audio checkpoints off the MLX path. */
  loadedIsMlx: boolean | null;
  /** Whether loadedContextLength actually bounds the cache. Null where the backend does
   *  not answer, which is not the same as a confirmed false. */
  loadedContextEnforced: boolean | null;
  modelRequiresTrustRemoteCode: boolean;
  supportsReasoning: boolean;
  reasoningAlwaysOn: boolean;
  reasoningEnabled: boolean;
  /** The model the OpenRouter router picked for the latest stream under the openrouter/free
   *  meta-model. Cleared on a non-OpenRouter model; display only. */
  lastOpenRouterChosenModel: string | null;
  reasoningStyle: ReasoningStyle;
  reasoningEffort: ReasoningEffort;
  supportsReasoningOff: boolean;
  reasoningEffortLevels: readonly ReasoningEffort[];
  supportsPreserveThinking: boolean;
  preserveThinking: boolean;
  supportsTools: boolean;
  /** Whether the active external provider exposes a server-side web_search (OpenAI /v1/responses).
   *  Distinct from `supportsTools`: this only lights the Search pill for external models. */
  supportsBuiltinWebSearch: boolean;
  /** Whether the provider exposes server-side code execution (Anthropic `code_execution_20250825`).
   *  Distinct from `supportsTools`, since Anthropic dispatches it itself. */
  supportsBuiltinCodeExecution: boolean;
  /** Whether the provider exposes server-side image generation (OpenAI Responses API).
   *  Local models never receive it. */
  supportsBuiltinImageGeneration: boolean;
  /** Whether the provider exposes server-side web_fetch (Anthropic `web_fetch_*`). Gates the
   *  composer's Fetch pill, independent of Search. */
  supportsBuiltinWebFetch: boolean;
  toolsEnabled: boolean;
  codeToolsEnabled: boolean;
  imageToolsEnabled: boolean;
  deepResearchEnabled: boolean;
  researchWebsitePolicy: ResearchWebsitePolicy;
  researchModelTimeoutSeconds: number;
  artifactsEnabled: boolean;
  // Whether the Canvas toggle is offered in the composer + menu (hidden by default).
  showCanvasMenuItem: boolean;
  collapseHtmlArtifacts: boolean;
  allowArtifactNetworkAccess: boolean;
  // web_search also returns images the model can place inline; read by the backend per call.
  searchImages: boolean;
  mcpEnabledForChat: boolean;
  ragEnabled: boolean;
  ragSource: RagSource;
  /** Default composer attachment scope for chats inside a project. */
  projectAttachmentTarget: ProjectAttachmentTarget;
  /** Per-chat override of that default, so a pick in one chat does not redirect the rest. Session-only. */
  projectAttachmentTargetByThread: Record<string, ProjectAttachmentTarget>;
  ragMode: RagMode;
  ragTopK: number;
  // autoInject = forced first-pass retrieval before answering.
  ragAutoInject: RagAutoInject;
  ragAutoInjectMinScore: number;
  // OCR scanned/image-only PDF pages at ingest time (vision model required).
  ragOcrScanned: boolean;
  // Describe figures/charts at ingest time (vision model required).
  ragCaptionFigures: boolean;
  /** When on, local Unsloth tool calls pause for an explicit allow/deny before they run. */
  confirmToolCalls: boolean;
  /** Tool calls run with no confirmation gate AND no python/terminal sandbox (secrets still
   *  stripped). Outranks confirmToolCalls; kept in sync with permissionMode "full". */
  bypassPermissions: boolean;
  /** Permission level. Single source of truth for the bypass dropdowns; bypassPermissions and
   *  confirmToolCalls mirror it. "full" is session-only. */
  permissionMode: PermissionMode;
  /** Whether the bypass warning dialog is open. Lifted out of the composer menu so confirming
   *  it does not leave the menu frozen. */
  bypassConfirmOpen: boolean;
  /** Per-chat tool names auto-approved via "Always allow", keyed by UI confirmation scope
   *  rather than the backend sandbox session id. Not persisted. */
  alwaysAllowToolsBySession: Map<string, Set<string>>;
  /** Calls paused for allow/deny, keyed by scoped frontend tool-call id. Each carries the
   *  backend `approvalId` and the run's `sessionId` so the exact call resolves;
   *  `autoAllowKey` scopes "Always allow" per chat. Backend-gated local calls only. */
  toolConfirmations: Record<
    string,
    { approvalId: string; sessionId: string; autoAllowKey: string }
  >;
  /** Fetch pill state, independent of `toolsEnabled` (Search). Read only when the provider
   *  supports builtin web_fetch. */
  webFetchToolsEnabled: boolean;
  /** Live tool status per conversation with its start time. Keyed by thread, or one chat's
   *  tool call shows above every composer; the timestamp keeps the counter running. */
  /** Per-run entries, newest last. Unresolved threads share "__default", so one scalar per key
   *  let a finishing run's clear remove a sibling's status. */
  toolStatusByThreadId: Record<string, ToolStatusEntry[]>;
  /** Live stdout/stderr from running tools by toolCallId; cleared on tool_end or run end. */
  toolLiveOutput: Record<string, string>;
  /** Full live output of finished tools whose result was truncated for the model. Finished
   *  cards prefer it over the truncated result. */
  toolFullOutput: Record<string, string>;
  generatingStatus: string | null;
  autoHealToolCalls: boolean;
  nudgeToolCalls: boolean;
  autoCompactEnabled: boolean;
  contextPolicy: LocalContextPolicy;
  compactionHeadroomRatio: number;
  maxToolCallsPerMessage: number;
  toolCallTimeout: number;
  kvCacheDtype: string | null;
  mlxKvBits: number | null;
  /** Width the backend was last asked for; the verdict belongs beside it. */
  loadedMlxKvBitsRequested: number | null;
  mlxKvQuantReason: string | null;
  chatTemplateOverrideReason: string | null;
  mlxKvQuantNote: string | null;
  loadedKvCacheDtype: string | null;
  speculativeType: string | null;
  loadedSpeculativeType: string | null;
  /** Why speculative decoding was disabled despite being requested, or null. Mirrors
   *  InferenceStatusResponse.spec_fallback_reason. */
  specFallbackReason: string | null;
  /** Projector recovery outcome for the active GGUF, or null. */
  mmprojFallbackReason: MmprojFallbackReason | null;
  /** Which drafter the speculative resolution was about ("mtp", "dspark", "dflash").
   *  specFallbackReason alone cannot name the file to fix, since Auto resolves it server-side. */
  specDrafterKind: string | null;
  /** User --spec-draft-n-max override (null = platform default). */
  specDraftNMax: number | null;
  loadedSpecDraftNMax: number | null;
  /** --parallel slots override for GGUF loads (null = server default). Never re-seeded from an
   *  echo: the resolved count would pin a blank control. */
  nParallel: number | null;
  /** Slots the last successful load sent (null = default); a rollback re-sends them so a failed
   *  switch cannot lose the override. */
  loadedNParallel: number | null;
  /** user --batch-size override for gguf loads (null = llama.cpp default 2048) */
  nBatch: number | null;
  /** batch size the last successful load sent (null = default) */
  loadedNBatch: number | null;
  /** Pass-through args the resident model is running, as far as this client knows. A rollback
   *  resends them: by then the target load has replaced the backend's inheritance source. */
  loadedLlamaExtraArgs: string[] | null;
  /** user --ubatch-size override for gguf loads (null = llama.cpp default 512) */
  nUbatch: number | null;
  /** micro-batch size the last successful load sent (null = default) */
  loadedNUbatch: number | null;
  /** --spec-draft-type-k/-v override, the DRAFT context's KV dtype (null = f16). Separate from
   *  kvCacheDtype, which is the target model's. */
  specDraftCacheDtype: string | null;
  /** draft cache dtype the last successful load sent (null = default) */
  loadedSpecDraftCacheDtype: string | null;
  /** user --load-mode override (null = llama.cpp's own `auto`) */
  loadMode: string | null;
  /** load mode the last successful load sent (null = default) */
  loadedLoadMode: string | null;
  /** user --ctx-checkpoints override (null = llama.cpp default 32) */
  ctxCheckpoints: number | null;
  /** checkpoint count the last successful load sent (null = default) */
  loadedCtxCheckpoints: number | null;
  /** user --cache-ram override in MiB (null = llama.cpp default 8192) */
  cacheRam: number | null;
  /** host prompt cache size the last successful load sent (null = default) */
  loadedCacheRam: number | null;
  /** Tensor-parallel split (--split-mode tensor) toggle, GGUF multi-GPU only. */
  tensorParallel: boolean;
  /** Backend-reported tensor-parallel state; null until first hydrated. */
  loadedTensorParallel: boolean | null;
  /** What the RUNNING server was loaded with, as opposed to what the control shows: a pending
   *  per-model config reaches disableVision before a switch captures its baseline. */
  loadedDisableVision: boolean | null;
  /** Load a vision GGUF without its mmproj, freeing the projector's VRAM. */
  disableVision: boolean;
  /** Backend-reported: image input is off by request, not by absence of a projector. Null until first hydrated. */
  loadedVisionDisabledByUser: boolean | null;
  /** GPU memory strategy for GGUF loads. "auto" fits GPUs and context for you; "manual" owns
   *  the offload (gpuLayers < 0 = Auto/--fit, >= 0 pins layers + nCpuMoe). */
  gpuMemoryMode: "auto" | "manual";
  /** Backend-reported gpu memory mode; null until first hydrated. */
  loadedGpuMemoryMode: "auto" | "manual" | null;
  /** The active model must use the staged CPU-only runtime when it is reloaded. */
  loadedCpuFallback: boolean;
  /** Manual mode: layers to offload to GPU. -1 = Auto (--fit); >= model layer count = all. */
  gpuLayers: number;
  loadedGpuLayers: number | null;
  /** Manual mode: MoE expert layers to keep on CPU (--n-cpu-moe); 0 = none. */
  nCpuMoe: number;
  loadedNCpuMoe: number | null;
  /** Manual mode: per-GPU layer counts (--tensor-split), in GPU-in-use order; null = unset. */
  splitRatio: number[] | null;
  /** Backend-reported per-GPU split ratio (--tensor-split); null = unset. */
  loadedSplitRatio: number[] | null;
  /** Model layer count (GGUF block_count); the manual ceiling is this + 1, the output layer too. */
  ggufLayerCount: number | null;
  /** MoE expert-layer count: the nCpuMoe slider max; 0/null hides the slider. */
  moeLayerCount: number | null;
  /** Picked IDs in the backend-declared GPU namespace (null = automatic). */
  selectedGpuIds: number[] | null;
  /** Namespace used by selectedGpuIds; kept with deferred persisted picks. */
  selectedGpuIndexKind: GpuIndexKind | null;
  loadedGpuIds: number[] | null;
  /** Backend-reported namespace paired with loadedGpuIds. */
  loadedGpuIndexKind: GpuIndexKind | null;
  /** Persisted: expand every On Device GGUF repo's quantizations by default. */
  expandQuantizations: boolean;
  /** Persisted: show non-downloaded quantizations too, not just downloaded. */
  showAllQuantizations: boolean;
  /** Persisted, off by default: chart each model's VRAM footprint. Opt-in, being estimates. */
  showMemoryBar: boolean;
  /** Persisted, shared with the Hub page: list only models that fit this device's budget. */
  fitOnDeviceOnly: boolean;
  loadedIsMultimodal: boolean;
  /** Active model is a block-diffusion model (DiffusionGemma): drives the denoising-canvas artifact auto-render. */
  loadedIsDiffusion: boolean;
  /** Live denoising frame per conversation ("__default" until the id exists). Transient and
   *  keyed, since two denoising chats overwrote each other's frame. */
  activeDiffusionCanvasByThreadId: Record<string, DiffusionCanvasFrame>;
  customContextLength: number | null;
  /** The pinned context the loaded model used (null = Auto), so dirty-tracking and a later fit
   *  Apply can tell an explicit pin from Auto. */
  loadedCustomContextLength: number | null;
  defaultChatTemplate: string | null;
  chatTemplateOverride: string | null;
  loadedChatTemplateOverride: string | null;
  activeThreadId: string | null;
  activeThreadEpoch: number;
  queuedSettingsEpoch: number;
  activeProjectId: string | null;
  /** Incognito toggle: the conversation lives only in assistant-ui's in-memory repository and
   *  never reaches studio.db, so a refresh always exits incognito. */
  incognito: boolean;
  settingsPanelOpen: boolean;
  editingMessageId: string | null;
  pendingAudioBase64: string | null;
  pendingAudioName: string | null;
  pendingImageEditReference: PendingImageEditReference | null;
  contextUsage: ContextUsageSnapshot | null;
  /** Per-thread copy of the above, so the bar survives a switch: `contextUsage` is the VISIBLE
   *  conversation's, and a background run may not write it. */
  contextUsageByThreadId: Record<string, ContextUsageSnapshot>;
  modelLoading: boolean;
  loadingModelPick: LoadingModelPick | null;
  // What the resident model loaded from, when that is not its id: a reload rebuilds its target
  // from the checkpoint, so without this it goes back down the ref the pin avoided.
  activeLoadId: string | null;
  activeNativePathToken: string | null;
  // Expiry (ms) of the active native path token: the desktop host prunes file leases on a TTL,
  // so a reload prompts re-selection instead of reusing a dead token.
  activeNativePathExpiresAtMs: number | null;
  hydratePersistedSettings: () => Promise<void>;
  beginModelLoading: () => ModelLifecycleLease | null;
  endModelLoading: (lease: ModelLifecycleLease) => void;
  setLoadingModelPick: (pick: LoadingModelPick | null) => void;
  clearLoadingModelPick: (expected: LoadingModelPick) => void;
  setModelRequiresTrustRemoteCode: (required: boolean) => void;
  setParams: (
    params: InferenceParams,
    options?: {
      persist?: boolean;
      trackQueuedSettings?: boolean;
      /** These params are the model's defaults, so its remembered settings are laid back over them
       *  even with no checkpoint change; being the model's, they move the installation defaults. */
      fromModelDefaults?: boolean;
      /** The context the model just loaded with. */
      maxTokensCap?: number;
      /** Defaults came from the server-resident model at startup, so a legacy
       * global snapshot can be attributed to it. */
      migrateOwnedGlobalQwenDefaults?: boolean;
    },
  ) => void;
  setCustomPresets: (presets: Preset[]) => void;
  setActivePreset: (name: string) => void;
  setActivePresetSource: (source: ChatPresetSource) => void;
  setModels: (models: ChatModelRow[]) => void;
  setLoras: (loras: ChatLoraSummary[]) => void;
  /** `local` defaults true so an unqualified caller still counts for the model-swap gate.
   *  `owner` narrows the clear, since unresolved ids share "__default" and owners accumulate. */
  setThreadRunning: (
    threadId: string,
    running: boolean,
    options?: { local?: boolean; owner?: () => void },
  ) => void;
  /** Re-key a first turn's run handles once its thread is persisted. Everything files under
   *  "__default" before the id exists, so the sidebar found no run and Stop had no handle. */
  adoptDefaultThreadRun: (threadId: string) => void;
  /** Which key this run's handles live under now: `adoptDefaultThreadRun` re-keys them mid-run,
   *  so a run started under "__default" must look its owner up. */
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
      /** False when the switch only puts back what was on screen before a hidden load, so the model
       *  it steps off was never the user's. */
      persist?: boolean;
      /** The context the model just loaded with, when the caller knows it. */
      maxTokensCap?: number;
    },
  ) => void;
  setActiveThreadId: (threadId: string | null) => void;
  /** show `threadId`'s own settings; a null threadId or snapshot restores the global ones. */
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
  setLastOpenRouterChosenModel: (chosen: string | null) => void;
  setReasoningStyle: (style: ReasoningStyle) => void;
  setReasoningEffort: (effort: ReasoningEffort) => void;
  setPreserveThinking: (value: boolean) => void;
  setToolsEnabled: (enabled: boolean, options?: { persist?: boolean }) => void;
  setCodeToolsEnabled: (enabled: boolean) => void;
  setImageToolsEnabled: (enabled: boolean) => void;
  setDeepResearchEnabled: (enabled: boolean) => void;
  setResearchWebsitePolicy: (policy: ResearchWebsitePolicy) => void;
  setResearchModelTimeoutSeconds: (seconds: number) => void;
  setArtifactsEnabled: (
    enabled: boolean,
    options?: { persist?: boolean },
  ) => void;
  setShowCanvasMenuItem: (enabled: boolean) => void;
  setCollapseHtmlArtifacts: (enabled: boolean) => void;
  setAllowArtifactNetworkAccess: (enabled: boolean) => void;
  setSearchImages: (enabled: boolean) => void;
  setMcpEnabledForChat: (enabled: boolean) => void;
  setConfirmToolCalls: (enabled: boolean) => void;
  setBypassPermissions: (enabled: boolean) => void;
  setPermissionMode: (mode: PermissionMode) => void;
  setBypassConfirmOpen: (open: boolean) => void;
  allowToolAlways: (sessionId: string, toolName: string) => void;
  setToolConfirmation: (
    toolCallId: string,
    approvalId: string,
    sessionId: string,
    autoAllowKey: string,
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
  /** Carry a choice made before the chat existed onto its new id. `claim` is what
   *  readPendingAttachmentTargetClaim gave; a newer one means another composer owns it. */
  adoptPendingProjectAttachmentTarget: (threadId: string, claim?: number) =>
    void;
  /** Drop a choice made in a composer that never became a chat. */
  clearPendingProjectAttachmentTarget: () => void;
  setRagMode: (mode: RagMode) => void;
  setRagTopK: (topK: number) => void;
  setRagAutoInject: (value: RagAutoInject) => void;
  setRagAutoInjectMinScore: (score: number) => void;
  setRagOcrScanned: (enabled: boolean) => void;
  setRagCaptionFigures: (enabled: boolean) => void;
  /** `owner` is the run's identity token, as for `setThreadRunning`: unresolved threads share
   *  "__default", so without it one run's cleanup clears a concurrent run's status. */
  setToolStatus: (
    threadId: string,
    status: string | null,
    owner?: () => void,
  ) => void;
  appendToolLiveOutput: (toolCallId: string, text: string) => void;
  /** Clear one tool's live output, or all when no id is given. */
  clearToolLiveOutput: (toolCallId?: string) => void;
  /** Preserve a finished tool's full live-streamed output for display. */
  setToolFullOutput: (toolCallId: string, text: string) => void;
  /** Drop a stale preserved full output (a new run is reusing the id). */
  clearToolFullOutput: (toolCallId: string) => void;
  setGeneratingStatus: (status: string | null) => void;
  setActiveDiffusionCanvas: (
    threadId: string | null,
    canvas: DiffusionCanvasFrame,
  ) => void;
  /** Drop only `threadId`'s canvas: a run ending in a background chat must not wipe another's. */
  clearActiveDiffusionCanvasForThread: (threadId: string | null) => void;
  setAutoHealToolCalls: (enabled: boolean) => void;
  setNudgeToolCalls: (enabled: boolean) => void;
  setAutoCompactEnabled: (enabled: boolean) => void;
  setContextPolicy: (policy: LocalContextPolicy) => void;
  setCompactionHeadroomRatio: (ratio: number) => void;
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
  /** A finished run's usage, kept per thread so switching back re-applies it. */
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
  | "contextPolicy"
  | "compactionHeadroomRatio"
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
  | "artifactsEnabled"
  | "showCanvasMenuItem"
  | "mcpEnabledForChat"
  | "confirmToolCalls"
  | "permissionMode"
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
  "contextPolicy",
  "compactionHeadroomRatio",
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
  "artifactsEnabled",
  "showCanvasMenuItem",
  "mcpEnabledForChat",
  "confirmToolCalls",
  "permissionMode",
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

// Ids this browser holds a local answer for. Hydration keeps these and merges the rest, so a
// pre-hydration edit cannot drop other models.
const locallyRememberedModels = new Set<string>();
const inferenceParamMutationVersions = Object.fromEntries(
  PERSISTED_INFERENCE_PARAM_KEYS.map((key) => [key, 0]),
) as Record<PersistedInferenceParamKey, number>;
const scalarSettingMutationVersions = Object.fromEntries(
  SCALAR_SETTING_KEYS.map((key) => [key, 0]),
) as Record<ScalarSettingKey, number>;

let loadedModelReasoningMode: {
  checkpoint: string;
  enabled: boolean;
  reasoningMutationVersion: number;
  fromLoad: boolean;
} | null = null;

/**
 * Record the mode a load/status response put the active model in. Persisted
 * settings can describe the previous model, so they cannot pick the migration
 * table after a family default changed on load.
 *
 * `fromLoad` marks a model this browser actually loaded. A status refresh only
 * echoes the reasoningEnabled already in the store, which before hydration is
 * this browser's local default, so it must not outrank the persisted toggle.
 */
export function noteLoadedModelReasoningMode(
  checkpoint: string,
  enabled: boolean,
  fromLoad = false,
): void {
  const state = useChatRuntimeStore.getState();
  const previous = loadedModelReasoningMode;
  loadedModelReasoningMode = {
    checkpoint,
    // A thread pin is not a shared model default. When one is active, retain
    // the installation value captured before that thread was applied.
    enabled:
      threadScopedOverride("reasoningEnabled") !== undefined
        ? installationReasoningEnabled(state)
        : enabled,
    reasoningMutationVersion:
      scalarSettingMutationVersions.reasoningEnabled,
    // Sticky per checkpoint: performLoad marks the load, then awaits refresh(),
    // whose status merge calls this again with the default false. Downgrading
    // there would drop the load's claim before hydration could read it.
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
    inferenceParams: { ...inferenceParamMutationVersions },
    scalarSettings: { ...scalarSettingMutationVersions },
    presets: {
      customPresets: customPresetsMutationVersion,
      activePreset: activePresetMutationVersion,
      activePresetSource: activePresetSourceMutationVersion,
    },
  };
}

/** Refresh the localStorage cache from the values hydration just applied. */
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

/** Seed the backend from this browser for mirrored settings it never stored; without it an
 *  existing install keeps its preferences local until each is next changed. */
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

/** `bumpVersions` fences the moved keys against a hydration response in flight. A model's own
 *  defaults do not get that: they must not outrank the settings saved for it. */
function getChangedInferenceParams(
  nextParams: InferenceParams,
  currentParams: InferenceParams,
  bumpVersions = true,
): PersistedInferenceParams {
  const changedParams: PersistedInferenceParams = {};
  for (const key of PERSISTED_INFERENCE_PARAM_KEYS) {
    const nextValue = nextParams[key];
    if (Object.is(nextValue, currentParams[key])) {
      continue;
    }
    if (bumpVersions) {
      inferenceParamMutationVersions[key] += 1;
    }
    if (nextValue !== undefined) {
      setInferenceParam(changedParams as InferenceParams, key, nextValue);
    }
  }
  return changedParams;
}

/** Bump the hydration versions for the replayed keys and mirror them globally, so a reload
 *  lands on this model's settings. */
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
    // Same reason as the setParams write: the in-memory copy is what a chat with no snapshot falls back to.
    noteThreadScopedDefaults(changed);
  }
}

/** Same fence as the inference-param versions, per model so only the edited one is protected.
 *  Before hydration the params are placeholders, so nothing is recorded. */
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

/** State a model switch owes to the memory: the outgoing model's snapshot. */
function getReplayStatePatch(
  state: ChatRuntimeStore,
  nextParams: InferenceParams,
  outgoing: Record<string, PersistedInferenceParams> | null,
  baseParams: InferenceParams,
): Partial<ChatRuntimeStore> {
  persistReplayedParams(state, nextParams, baseParams !== state.params);
  return outgoing ? { paramsByModel: outgoing } : {};
}

/** Falls back to the outgoing snapshot: a non-persisting update (a background load) must not
 *  rewrite durable memory. Only an edit is recorded; defaults ride the leaving snapshot. */
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
      // Same filter as the outgoing snapshot: an edit to a key the memory keeps but the chat does
      // not still records the WHOLE snapshot, riding the chat's sampling into the entry.
      pickRememberedParams(withoutActiveThreadParams(state, nextParams)),
    ),
    nextParams.checkpoint,
  );
  return recorded ?? outgoing;
}

/** Snapshot the model being switched away from, so a model that was never edited still keeps what it ran with. */
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
  // Only to seed a model with no entry: later changes are written key by key, and a full
  // snapshot would put this browser's copy of untouched keys over another tab's.
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
  // A full snapshot rewrites every field, so send one only when this browser has something to
  // say: an edit made here, or an entry that does not exist yet.
  if (next && state.settingsHydrated && outgoing.checkpoint && seeding) {
    saveSettingsPatch({
      inferenceParamsByModel: { [outgoing.checkpoint]: snapshot },
    });
  }
  return next;
}

/** Write an edit to the global set, and to the selected model's own set. */
function persistParamEdit(
  changedParams: PersistedInferenceParams,
  paramsByModel: Record<string, PersistedInferenceParams> | null,
  modelId: string | undefined,
): void {
  if (!hasKeys(changedParams)) {
    return;
  }
  // Only what moved: the server merges per key, so sending the rest would overwrite another
  // tab's copy of every other key.
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

/** Keys the user moved while the request was in flight: the same fence, read the other way round. */
function pickLocallyEditedParams(
  params: InferenceParams,
  versions: SettingsHydrationVersions,
): PersistedInferenceParams {
  const edited: PersistedInferenceParams = {};
  for (const key of REMEMBERED_INFERENCE_PARAM_KEYS) {
    if (inferenceParamMutationVersions[key] !== versions.inferenceParams[key]) {
      setInferenceParam(edited as InferenceParams, key, params[key]);
    }
  }
  return edited;
}

/** The context the last load or status published for the model on screen.
 * loadedContextLength covers only a backend that sizes a window, so without this
 * a safetensors model has nothing to clamp the hydration replay against. Keyed
 * by checkpoint. */
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

/** A model selected while the request was in flight. Its defaults lose to its
 * own entry but outrank the global set, which belongs to the last model used. */
let modelLoadedBeforeHydration: string | null = null;

/** A model stepped off before hydration: nothing can be filed for it yet, but the global set
 *  the response delivers is what it ran with, so this says who to file it under. */
let modelLeftBeforeHydration: string | null = null;

function noteModelDefaultsBeforeHydration(
  checkpoint: string,
  ownsPersistedGlobal: boolean,
): void {
  // A model the user selected while settings were in flight does not own the
  // previous session's global snapshot, even as this tab's first checkpoint.
  // Server-resident startup adoption is the one exception.
  if (!ownsPersistedGlobal) {
    modelLoadedBeforeHydration = checkpoint;
    return;
  }
  // The model already resident at startup is the one the saved global set
  // describes, so its defaults must not stand in front of it.
  if (!sameCheckpointIdentity(modelLoadedBeforeHydration, checkpoint)) {
    modelLoadedBeforeHydration = null;
  }
  // setCheckpoint marks any pre-hydration switch as an interactive pick, and
  // this is the adoption signal that says otherwise.
  if (sameCheckpointIdentity(unownedCheckpointBeforeHydration, checkpoint)) {
    unownedCheckpointBeforeHydration = null;
  }
}

/** Whether the checkpoint is an external model that cannot run deep research, as the composer
 *  rules it. An unresolved provider is left alone: refusing would drop a Codex preference. */
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

/** Kimi's $web_search requires thinking disabled and the composer keeps the pills exclusive,
 *  so a restore must too, or a chat stored elsewhere sends a combination Kimi rejects. */
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
  const nextState: Partial<ChatRuntimeStore> = {};
  const checkpoint = state.params.checkpoint;
  const loadedBeforeHydration = sameCheckpointIdentity(
    modelLoadedBeforeHydration,
    checkpoint,
  );
  modelLoadedBeforeHydration = null;
  // The toggle as it will read once this response lands, under the same fence the scalar loop below applies.
  const remembersPerModel =
    settings.rememberParamsPerModel !== undefined &&
    scalarSettingMutationVersions.rememberParamsPerModel ===
      versions.scalarSettings.rememberParamsPerModel
      ? settings.rememberParamsPerModel
      : state.rememberParamsPerModel;
  // A model loaded mid-flight has no entry to restore defaults from, so the global set would
  // overwrite them with the last model's. Only while the memory is on.
  const keepModelDefaults =
    remembersPerModel &&
    loadedBeforeHydration &&
    settings.inferenceParamsByModel?.[checkpoint] === undefined;
  const params = { ...state.params };
  for (const key of PERSISTED_INFERENCE_PARAM_KEYS) {
    const value = settings.inferenceParams?.[key];
    // A slider moved before this response landed is held for the open chat. The edit wins in the
    // store but belongs to the chat, so keep the server's value for the restore.
    if (value !== undefined && isHeldThreadScopedField(key)) {
      hydratedDefaultsByHeldField.set(key, value);
      continue;
    }
    if (
      value !== undefined &&
      !keepModelDefaults &&
      // The context belongs to the load, not the previous model's global set, and no entry carries
      // one for the replay below.
      !(loadedBeforeHydration && key === "maxSeqLength") &&
      inferenceParamMutationVersions[key] === versions.inferenceParams[key]
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
      // Stored as written: a gap is a key this model never pinned, and there is no honest value to invent.
      hydrated[modelId] = entry;
    }
    for (const modelId of locallyRememberedModels) {
      const local = state.paramsByModel[modelId];
      if (local) {
        hydrated[modelId] = local;
      }
    }
    // The entry arriving for this model predates the fenced edit, so lay the edit over it or the
    // next defaults update replays the stale one.
    if (checkpoint) {
      const edited = pickLocallyEditedParams(params, versions);
      if (hasKeys(edited)) {
        hydrated[checkpoint] = { ...hydrated[checkpoint], ...edited };
      }
    }
    nextState.paramsByModel = hydrated;
  } else if (checkpoint) {
    // No map in the response: an install upgraded from before this feature. With no entry the
    // next defaults update puts the recommendation back over the fenced edit.
    const edited = pickLocallyEditedParams(params, versions);
    if (hasKeys(edited)) {
      nextState.paramsByModel = {
        ...state.paramsByModel,
        [checkpoint]: { ...state.paramsByModel[checkpoint], ...edited },
      };
    }
  }
  // A model stepped off before this response landed could not be filed then, and only now is
  // the global set it ran with known; without this, switching back inherits a stranger.
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
  // Same fence as the scalar loop: a click made while this response was out is the newer
  // answer, and recording the stored value over it leaves the switch visibly wrong.
  if (
    settings.preserveThinking !== undefined &&
    scalarSettingMutationVersions.preserveThinking ===
      versions.scalarSettings.preserveThinking
  ) {
    notePreserveThinkingPreference(settings.preserveThinking);
  }
  for (const key of SCALAR_SETTING_KEYS) {
    const value = settings[key];
    // Full access is session-only, so a stored level must not silently drop the sandbox bypass
    // the user accepted a warning for.
    if (
      state.permissionMode === "full" &&
      (key === "permissionMode" || key === "confirmToolCalls")
    ) {
      continue;
    }
    // Both describe the running model through a loaded* shadow this loop cannot set, so skip
    // them while a shadow owns them. With none resident the store field is what a load reads.
    if (loadShadowOwnsMirroredSetting(key, state)) {
      continue;
    }
    // A load sets this from the model's own capability, and only a load sets reasoningAlwaysOn,
    // so a stored false would ask a model that cannot stop thinking to stop.
    if (
      key === "reasoningEnabled" &&
      value === false &&
      state.reasoningAlwaysOn
    ) {
      continue;
    }
    // Same reason, the other direction: a load that landed while this response
    // was in flight already chose the mode for the running model and advances
    // no mutation version, so the stored toggle describes the previous one. Let
    // the load win, as its sampling table does. Only an actual load, though: a
    // status refresh echoes local state and must not outrank the installation.
    if (
      key === "reasoningEnabled" &&
      loadEstablishedReasoningMode(state, true)
    ) {
      continue;
    }
    // Only a local model or an openai_codex provider can run deep research, so a
    // stored true must not arm the pill for any other external checkpoint.
    if (
      key === "deepResearchEnabled" &&
      value === true &&
      externalCheckpointRefusesDeepResearch(state.params.checkpoint)
    ) {
      continue;
    }
    // A click made before this response landed is held for the open chat rather than written
    // globally, so it advances no mutation version and the server's value would replace it.
    if (isHeldThreadScopedField(key)) {
      // The click wins in the STORE but belongs to the chat: when the window closes the default
      // has to go back to something, so keep the server's authoritative value here.
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
  // The model already selected when this lands never crossed a checkpoint transition, so
  // nothing replayed its memory and it would run on the last model's global set.
  const remembered = (nextState.paramsByModel ?? state.paramsByModel)[
    params.checkpoint
  ];
  if (
    (nextState.rememberParamsPerModel ?? state.rememberParamsPerModel) &&
    remembered
  ) {
    // Same fence as the global set. REMEMBERED, not PERSISTED, as in getReplayedParams: a
    // maxSeqLength the row carries must not replace the loaded context.
    const replayed = { ...params };
    for (const key of REMEMBERED_INFERENCE_PARAM_KEYS) {
      const value = remembered[key];
      if (
        value !== undefined &&
        inferenceParamMutationVersions[key] === versions.inferenceParams[key]
      ) {
        setInferenceParam(replayed, key, value);
      }
    }
    // The same cap the load and status replays apply.
    nextState.params = replayed;
  }
  // Outside the replay: an install with only a global set has no entry, and the budget
  // restored from it does not fit the load either.
  const capped = nextState.params ?? params;
  // loadedContextLength describes whatever is resident, which an external pick
  // leaves loaded, so it is not this checkpoint's context to clamp against.
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
    scalarSettingMutationVersions[key] += 1;
    saveSettingsPatch({ [key]: value });
  };
  if (captureThreadScopedEdit(key, writeGlobal)) return;
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

function installationReasoningEnabled(state: ChatRuntimeStore): boolean {
  return threadScopedOverride("reasoningEnabled") !== undefined
    ? (globalThreadScopedDefaults?.reasoningEnabled ?? state.reasoningEnabled)
    : state.reasoningEnabled;
}

/**
 * Whether a load or status response established the reasoning mode for the
 * active model, with no user toggle since.
 *
 * A load writes reasoningEnabled without advancing its mutation version, so
 * hydration would otherwise replay a toggle describing the previous model. The
 * migration table and the hydrated pill both read this, so they cannot disagree
 * and show a thinking pill above non-thinking sampling.
 */
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
  // Ahead of the established mode: a model that cannot reason never had the
  // thinking table applied at load, so its recorded mode is just this browser's
  // toggle. Only once status reported this checkpoint, though, since
  // supportsReasoning starts false and "not asked yet" is not "cannot".
  const statusSeen = sameCheckpointIdentity(
    loadedModelReasoningMode?.checkpoint,
    state.params.checkpoint,
  );
  if (statusSeen && !state.supportsReasoning) {
    return false;
  }
  // requireLoad, as the hydrated pill uses: a status refresh echoes the toggle
  // already in the store, which before hydration is this browser's own default.
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

// Raw, not sanitized: the server tests `key in current` against what is stored,
// so asserting absence from the sanitized copy would fence a key the row has.
/**
 * True when a decision field is stored but sanitizes away, such as an explicit
 * null or an empty inferenceParamsByModel map: the compare-and-set can assert a
 * value or an absence and neither fits, and `expected` matches by recursive
 * subset, so an empty map matches a populated one. Decline rather than migrate
 * unfenced. An absent key is different, and stays fenced by expectedAbsent.
 */
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
  // Raw for the same reason as qwenMigrationExpectedAbsent: the server tests the
  // stored row, so a key sanitizing away must not be fenced as absent.
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
  // Normalizing a differently-cased key writes a new exact-key row. Unfenced,
  // the subset compare only checks the old spelling, so an exact-key row
  // another tab added after the confirming read would be overwritten whole.
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
      // The open chat may pin one of these, but a later snapshot-less chat
      // falls back to the installation copy captured when pairing began. Move
      // it with the migrated defaults before restoring the thread's params.
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

/**
 * Bring local params back in line after a migration the server accepted while a
 * local edit was in flight.
 *
 * The edit flips provenance to "modified", so the ordinary apply refuses and
 * the tab keeps generating from values the server no longer holds. Only the
 * fields the user did not touch are adopted, which is what the per-field
 * mutation versions captured before the write identify.
 */
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
        inferenceParamMutationVersions[field] !== versionsBefore[field] ||
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
      // As the normal application path does: applying a thread pin advances no
      // mutation version, so nothing above would have noticed it.
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
    // Land the preset selection first so the confirming GET can recognize the
    // legacy snapshot; a newer edit in another tab fails the fingerprint
    // safely.
    await flushPendingChatSettings();
    // A write outstanding past the flush timeout would reach the backend merge
    // after this CAS and restore the legacy row. Rearm on its settlement rather
    // than wait: an external selection gets no later status callback.
    if (!settingsWritesAreDrained()) {
      // Bounded: a refused patch is retained and reflushed by each pass, so an
      // unbounded rearm loops on an unreachable backend. Cleared once writes
      // drain, so a healthy install never spends it.
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
    // A model switch during the GET invalidates both the row and the reasoning
    // mode this retry would persist. The new model schedules its own.
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
      // From the confirming read, as hydration derives it: memory turned on
      // elsewhere means the global is no longer this checkpoint's to rewrite.
      migrateOwnedGlobalAlongsideModelMemory &&
        !qwenMigrationRemembersPerModel(confirmed, confirmedState),
    );
    if (!migration.patch) return;
    if (qwenMigrationHasUnfenceableField(confirmedRaw, confirmed)) return;
    // Captured before the write: an edit landing mid-flight is the one case
    // where the server takes the migration and local refuses it.
    const presetSourceBeforeWrite = activePresetSourceMutationVersion;
    const paramVersionsBeforeWrite = { ...inferenceParamMutationVersions };
    const persisted = await savePersistedChatSettingsPatchIfCurrent(
      confirmed,
      migration.patch,
      qwenMigrationExpectedAbsent(confirmedRaw),
      qwenMigrationExpectedAbsentPaths(confirmedRaw, migration.patch),
    );
    // Only now touch local state: applying before persisting would leave this
    // tab generating with values the server rejected. Revalidated after the
    // await too, since the checkpoint can move while the response is in flight,
    // and an external pick gets no callback to schedule another retry.
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
        // "modified" is an ordinary edit and belongs here; a custom preset can
        // hold a legacy value on purpose and bumps no counter for it.
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

// Same bound as SETTINGS_FLUSH_TIMEOUT_MS, for the same reason.
const QWEN_MIGRATION_BARRIER_TIMEOUT_MS = 2000;

const QWEN_MIGRATION_MAX_REARMS = 3;
let qwenMigrationRearmsWhileBlocked = 0;

/**
 * Settles once a scheduled migration has decided and persisted, so a prompt sent
 * right after picking a Qwen with a dormant legacy row does not generate from
 * the values getReplayedParams just put on screen.
 */
export async function awaitPendingQwenDefaultsMigration(): Promise<void> {
  if (qwenMigrationInFlight === null) return;
  // Bounded like the settings flush: neither request takes an abort signal, so
  // a wedged backend would hold every send open. Past this the run proceeds.
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

// Re-read rather than await once: a replacement scheduled while this waits owns
// the checkpoint the send is about to generate from.
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
  // Published from scheduling time, not from when the microtask runs, or a send
  // landing in between would find nothing to wait for.
  let settleMigration: () => void = () => undefined;
  // Cleared only while the pointer still refers to this retry: the scheduler
  // frees its slot before the async work ends, so a later retry can replace it.
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
    // A status refresh needs a confirming GET only when the local store still
    // holds a legacy row. Adoption is the exception: it owns a server snapshot
    // model defaults already replaced locally, so only the server still has it.
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
  // Hydrate the last external checkpoint so the picker survives a refresh; local ids are
  // re-derived from the backend and deliberately not persisted.
  params: (() => {
    const persistedExternal = loadLastExternalCheckpoint();
    unownedCheckpointBeforeHydration = persistedExternal;
    return persistedExternal
      ? { ...DEFAULT_INFERENCE_PARAMS, checkpoint: persistedExternal }
      : DEFAULT_INFERENCE_PARAMS;
  })(),
  paramsByModel: {},
  // On by default; a model with nothing remembered keeps the current settings.
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
  activeModelIsLocal: false,
  loadedContextLength: null,
  maxContextLength: null,
  nativeContextLength: null,
  loadedIsGguf: null,
  loadedIsMlx: null,
  loadedContextEnforced: null,
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
  codeToolsEnabled: loadBool(CHAT_CODE_TOOLS_ENABLED_KEY, false),
  imageToolsEnabled: loadBool(CHAT_IMAGE_TOOLS_ENABLED_KEY, false),
  deepResearchEnabled: loadBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false),
  researchWebsitePolicy: loadResearchWebsitePolicy(),
  researchModelTimeoutSeconds: loadResearchModelTimeoutSeconds(),
  artifactsEnabled: loadBool(CHAT_ARTIFACTS_ENABLED_KEY, false),
  showCanvasMenuItem: loadShowCanvasMenuItem(),
  collapseHtmlArtifacts: loadBool(CHAT_COLLAPSE_HTML_ARTIFACTS_KEY, false),
  allowArtifactNetworkAccess: loadBool(
    CHAT_ALLOW_ARTIFACT_NETWORK_ACCESS_KEY,
    false,
  ),
  searchImages: loadBool(CHAT_SEARCH_IMAGES_KEY, false),
  mcpEnabledForChat: loadBool(CHAT_MCP_ENABLED_KEY, false),
  // Mirrors permissionMode (gate requested for ask/auto) so both controls agree on load.
  confirmToolCalls:
    INITIAL_PERMISSION_MODE === "ask" || INITIAL_PERMISSION_MODE === "auto",
  // Never restore Bypass Permissions from storage: it disables the sandbox and the
  // confirmation gate, so it needs the warning dialog each session.
  bypassPermissions: false,
  permissionMode: INITIAL_PERMISSION_MODE,
  bypassConfirmOpen: false,
  alwaysAllowToolsBySession: new Map<string, Set<string>>(),
  toolConfirmations: {},
  webFetchToolsEnabled: loadBool(CHAT_WEB_FETCH_TOOLS_ENABLED_KEY, false),
  // RAG is opt-in per session: always starts off, never restored from storage.
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
  contextPolicy: DEFAULT_CONTEXT_POLICY,
  compactionHeadroomRatio: DEFAULT_COMPACTION_HEADROOM_RATIO,
  maxToolCallsPerMessage: 25,
  toolCallTimeout: 5,
  kvCacheDtype: null,
  mlxKvBits: null,
  loadedMlxKvBitsRequested: null,
  mlxKvQuantReason: null,
  chatTemplateOverrideReason: null,
  mlxKvQuantNote: null,
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
  // Off by default: On Device lists what is on disk, not the whole repo.
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
        // Assigned by the confirming read below, so the failure path can prefer
        // it over the older hydration response.
        let confirmed: PersistedChatSettings | undefined;
        // What to hydrate when nothing was migrated. The confirming read is the
        // newer server truth, but a legacy merge that failed to save exists
        // only here, and re-reading the server would discard it.
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
        // A model loaded during the settings GET replaced the one whose global
        // fallback was saved. Only the model resident at startup can own a
        // global-only legacy snapshot.
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
        // Neither switch path can schedule the new model's retry from here:
        // both gate on settingsHydrated, still false.
        let checkpointMovedDuringConfirm = false;
        if (fromServer && migration.patch) {
          try {
            // Re-read immediately before the write, so the patch derives from
            // this confirmation rather than the earlier hydration response and
            // a newer edit from another tab is left untouched.
            const confirmedRaw = await getChatSettings();
            confirmed = sanitizeChatSettings(confirmedRaw);
            const confirmedState = get();
            // A model switch during the confirming GET invalidates the
            // checkpoint and mode this migration would persist; the new model
            // schedules its own retry. The preset can move too: a slider
            // touched now sets the local source to "modified" but queues its
            // write behind the debounce, so the server still reads
            // "builtin-default". Migrating on that stale read would rewrite a
            // preset the user already modified, so trust the local version.
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
            // migrateLegacyQwenDefaults returns its own input when nothing
            // migrates, so a read finding no candidate would otherwise replace
            // an unsaved legacy merge with server-only settings.
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
            // Best effort, but that cannot mean hydrating values the server
            // never accepted: the sheet and every request would show migrated
            // sampling while storage still held the old row, on each start.
            // Fall back to what was actually read, preferring the confirming
            // snapshot when it landed, since the first read may already be
            // stale and backfill would then push this browser's copy over it.
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
          // Only an authoritative read says a mirrored field is unset on the
          // server. A GET that fell back to legacy storage knows nothing about
          // it, and backfilling then pushes this browser's stale values over
          // whatever another browser wrote.
          if (fromServer) backfillMirroredSettings(hydratedSettings);
          // After the backfill, so a startup edit wins over the stored value.
          flushPreHydrationSettings();
          // Hydrated now, so the model switched to can be repaired.
          if (checkpointMovedDuringConfirm) {
            scheduleLegacyQwenDefaultsRetry(null);
          }
          // The previous session's tab-close writes, for the rows that did not exist
          // yet when it sent them. A replay of one that did land is refused by its own
          // seq, so this cannot revert anything newer.
          replayUnconfirmedThreadSettings();
        }
      } catch {
        // Hydrate failed: treat as hydrated-with-defaults so later setParams calls reach
        // saveSettingsPatch, which toasts on real network failure.
        warnSettingsPersistenceFailure();
        mirroredSettingsHydrated = true;
        flushPreHydrationSettings();
        // Independent of this endpoint: the tab-close snapshots are rows' own settings, and leaving
        // them unsent because /api/chat/settings blipped strands the last session's edit.
        replayUnconfirmedThreadSettings();
        set({ settingsHydrated: true });
      } finally {
        settingsHydrationPromise = null;
      }
    })();
    return settingsHydrationPromise;
  },
  beginModelLoading: () => {
    const lease = chatModelLifecycleGate.tryAcquire();
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
  setLoadingModelPick: (pick) => set({ loadingModelPick: pick }),
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
      // Mirror setCheckpoint: the local load path can move params.checkpoint via setParams()
      // first, leaving stale per-turn counters under the new checkpoint.
      const checkpointChanged = state.params.checkpoint !== params.checkpoint;
      const fromModelDefaults = options?.fromModelDefaults === true;
      // Remember what the outgoing model was running with before replacing it.
      const outgoing = checkpointChanged
        ? rememberOutgoingModel(state, state.params)
        : null;
      // An interactive local load arrives with the destination checkpoint and recommended params,
      // reaching setCheckpoint only later, so replay here or the switch never restores the
      // model's own settings. fromModelDefaults marks the updates memory goes back over.
      noteLoadedContext(params.checkpoint, options?.maxTokensCap);
      const replayed = checkpointChanged || fromModelDefaults;
      const nextParams = getReplayedParams(
        state.rememberParamsPerModel,
        outgoing ?? state.paramsByModel,
        params,
        params.checkpoint,
        replayed,
        options?.maxTokensCap,
      );
      // A chat outranks both the model's memory and its defaults, so its pinned sampling goes back
      // on top of the replay. Live store only; persistence is decided from nextParams.
      const effective = replayed
        ? restoreThreadScopedParams(nextParams)
        : nextParams;
      // A user edit fences the keys it moved against a hydration response in flight; only the HTTP
      // write is gated on settingsHydrated.
      const changedParams = getChangedInferenceParams(
        nextParams,
        state.params,
        !fromModelDefaults,
      );
      const queuedSettingsChanged = shouldAdvanceQueuedSettingsEpoch(
        state.params,
        effective,
        options?.trackQueuedSettings !== false,
      );
      const persistingGlobally =
        options?.persist !== false && state.settingsHydrated;
      // A sampling key moved with a chat open belongs to that chat, so it reaches neither the
      // defaults nor the model's memory. NOT gated on hydration, unlike the global write: the
      // composer is live while the initial request is out, and an uncaptured edit leaked.
      const sharedParams =
        options?.persist !== false
          ? withoutCapturedThreadEdits(changedParams, fromModelDefaults)
          : changedParams;
      // An edit belongs to the model the params now describe, so a call moving checkpoint and
      // sliders at once files them under the destination.
      const paramsByModel = getParamsByModelAfterEdit(
        state,
        outgoing,
        nextParams,
        sharedParams,
        options?.persist !== false && !fromModelDefaults,
      );
      if (persistingGlobally) {
        // A switch replays the destination's entry over the params, so writing it back says nothing
        // new and, merged per key, would overwrite another tab's copy.
        persistParamEdit(
          sharedParams,
          checkpointChanged ? null : paramsByModel,
          nextParams.checkpoint,
        );
        // Level with what was just written, or a chat opened later this session falls back to the
        // sampling from before this model loaded.
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
    // Startup can hydrate before the server's active model is adopted. Once
    // status applies its defaults the checkpoint and reasoning mode are known,
    // so the deferred active-row migration can run.
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
      // The sheet updates provenance before applying the parameter edit. Defer
      // a microtask so a final slider move restoring built-in Default has
      // landed before the migration inspects and flushes it.
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
        // Any local owner keeps the key counted by the model-swap gate, so an external run on a
        // shared key must not clear a sibling's flag.
        if (local) {
          nextLocal[threadId] = true;
        } else if (!owners.some((o) => o.local)) {
          delete nextLocal[threadId];
        }
      } else {
        const remaining = options?.owner
          ? owners.filter((o) => o.owner !== options.owner)
          : [];
        // An owner missing from the list was already cleared, or the key belongs to siblings only.
        if (options?.owner && remaining.length === owners.length) return state;
        // An ownerless clear predates per-run tracking, so it must not speak for runs that own the key.
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
      // Two first turns can share "__default" with nothing linking a run to the thread being
      // persisted, so moving the arrays wholesale handed over a sibling's stop handle. Adopt
      // only when the key holds a single run.
      if ((state.runOwnerByThreadId[key]?.length ?? 0) > 1) return state;
      // Only the transient run maps move; anything filed under the real id is better identified.
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
  // `cancel` narrows removal to the run that registered it: unresolved ids share "__default",
  // so a blind delete would drop a live sibling.
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
      // Persist external selections so they survive a refresh. Local ids are re-derived on mount,
      // and a stale persisted one would race the freshly-loaded model. See
      // LAST_EXTERNAL_CHECKPOINT_KEY notes.
      saveLastExternalCheckpoint(isExternalModelId(modelId) ? modelId : null);
      // Only disarm research for a connection that cannot drive it: the id prefix alone switched
      // it off for capable providers, and saveBool would write that off for every browser.
      const clampsDeepResearch =
        isExternalModelId(modelId) && !externalModelSupportsStudioTools(modelId);
      if (clampsDeepResearch) {
        saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
      }
      // Clear stale per-turn usage on model change, or the relaxed external render gate shows it.
      const checkpointChanged = state.params.checkpoint !== modelId;
      // An interactive pick during hydration has no adoption signal either, so
      // the previous session's global does not become this model's.
      if (checkpointChanged && !state.settingsHydrated) {
        unownedCheckpointBeforeHydration = modelId;
      }
      scheduleQwenMigration = checkpointChanged && state.settingsHydrated;
      // Remember what the outgoing model was running with before replacing it.
      // Not for a restore: the model it steps off is the one a background load
      // put there, and its load defaults are not settings the user chose.
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
      // Clamp maxTokens to the new model's cap when switching into an external model, so a value
      // carried over from a local session cannot exceed the slider's max.
      let nextMaxTokens = baseParams.maxTokens;
      if (checkpointChanged && isExternalModelId(modelId)) {
        const parsed = parseExternalModelId(modelId);
        const provider = parsed
          ? useExternalProvidersStore
              .getState()
              .providers.find((p) => p.id === parsed.providerId)
          : null;
        // Only when the connection is known: a checkpoint restored before the provider store hydrates
        // reads the 32,768 fallback and lowers a value nothing puts back.
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
      // The chat outranks the model it switches to, so its pinned sampling and prompt go back over
      // the replay; an external switch has no load to do it. Live store only.
      // Live store only: getReplayStatePatch still persists from the unrestored object, so the model's
      // own values reach the installation defaults.
      const restoredParams = checkpointChanged
        ? restoreThreadScopedParams(nextParams)
        : nextParams;
      return {
        params: restoredParams,
        ...getReplayStatePatch(state, nextParams, outgoing, baseParams),
        activeGgufVariant: nextGgufVariant,
        ...(queuedSettingsChanged
          ? { queuedSettingsEpoch: state.queuedSettingsEpoch + 1 }
          : {}),
        // Provenance and the spec-fallback reason both describe the model being replaced, so they go
        // together; dropping one pairs a stale reason with the wrong recovery text.
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
        // Switching to a provider that cannot run Unsloth's tool loop disables Deep Research; a
        // capable one keeps the user's choice.
        ...(clampsDeepResearch ? { deepResearchEnabled: false } : {}),
      };
    });
    // An external pick is the whole story for that model: no load and no status
    // follows it, so nothing else would ever schedule the migration and the
    // replayed legacy row would keep being sent until a reload. Never with
    // ownership, since selecting a model says nothing about whose global the
    // installation's snapshot is. The scheduler's own dry run makes this cheap
    // when there is no legacy row to repair.
    if (scheduleQwenMigration) {
      scheduleLegacyQwenDefaultsRetry(null);
    }
  },
  // Re-apply the incoming thread's own usage rather than blanking the bar: a run that finished
  // in the background never wrote the visible value, and a still-mounted runtime skips the
  // history loader on the way back.
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
      // The pending write belongs to the outgoing thread, so it goes out before the swap.
      flushThreadScopedSettingsWrite();
      // Edits made while this chat's snapshot was in flight: keep them and store them on the chat,
      // or the read would silently undo a click the user already saw.
      const heldFields = new Set<string>();
      if (threadId !== null && threadId === pendingPairingThreadId) {
        for (const edit of heldThreadScopedEdits) heldFields.add(edit.field);
        heldThreadScopedEdits = [];
        pendingPairingThreadId = null;
        // The window is over, so the next visit samples afresh rather than reusing this round's.
        pairingWindowDefaultsThreadId = null;
        // Its snapshot is now the one in the store, so anything waiting on it can go.
        closeThreadScopedPairingGate(threadId);
      } else if (
        // A drop to the defaults for a chat still open and still waiting on its read keeps holding:
        // those edits are that chat's, and releasing them here is the leak this path prevents.
        threadId !== null ||
        pendingPairingThreadId === null ||
        pendingPairingThreadId !== state.activeThreadId
      ) {
        releaseHeldThreadScopedEdits();
      }
      // Set from here rather than trusting the calls above: this updater's return value merges
      // last, so a `return state` would put the old flag back.
      const pending = pendingPairingThreadId !== null;
      // Nothing was overridden while unpaired, so there is nothing to restore.
      if (threadScopedSettingsThreadId === null && threadId === null) {
        return state.threadScopedSettingsPending === pending
          ? state
          : { ...state, threadScopedSettingsPending: pending };
      }
      if (threadScopedSettingsThreadId === null) {
        // A held edit is in the store but belongs to its chat, so capturing it here would promote it
        // to the default every snapshot-less chat follows; take the pre-window value. Deleting the
        // key leaves no fallback and leaks the edited value into the next chat.
        const captured = readThreadScopedSettings(state) as Record<
          string,
          unknown
        >;
        const beforeWindow = (pairingWindowDefaults ??
          globalThreadScopedDefaults) as Record<string, unknown> | null;
        for (const field of heldFields) {
          // The server answered for this field while the window was open and hydration had to skip it:
          // that value is the installation's, the pre-window copy only this browser's cache.
          if (hydratedDefaultsByHeldField.has(field)) {
            captured[field] = hydratedDefaultsByHeldField.get(field);
          } else if (beforeWindow && field in beforeWindow) {
            captured[field] = beforeWindow[field];
          } else {
            delete captured[field];
          }
          hydratedDefaultsByHeldField.delete(field);
        }
        globalThreadScopedDefaults = captured as ThreadScopedSettings;
      }
      threadScopedSettingsThreadId = threadId;
      explicitlyEditedThreadFields.clear();
      // The constraint belongs to the chat it was applied in, and this chat's own provider effects
      // will say so again if it still holds.
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
        // The user set this one while the read was in flight, so it wins over what came back.
        if (heldFields.has(key)) {
          applied[key] = readThreadScopedValue(state, key);
          continue;
        }
        // Full access was accepted through a warning dialog, so a switch must not drop it. The chat
        // is still pinned with the level underneath, or it would store no level at all.
        if (key === "permissionMode" && state.permissionMode === "full") {
          const underneath =
            stored?.permissionMode ??
            globalThreadScopedDefaults?.permissionMode ??
            loadPermissionMode();
          if (underneath !== "full") applied[key] = underneath;
          continue;
        }
        // setCheckpoint clears deep research for external models in the store only, so a stored true
        // comes back and fails every send. openai_codex is the composer's own exception.
        if (
          key === "deepResearchEnabled" &&
          (externalCheckpointRefusesDeepResearch(state.params.checkpoint) ||
            state.incognito)
        ) {
          continue;
        }
        // A key the snapshot omits falls back to the defaults, not to the outgoing chat's value.
        const value = firstSetThreadScopedValue(
          stored?.[key],
          globalThreadScopedDefaults?.[key],
        );
        if (value === undefined) continue;
        applied[key] = value;
        if (isSameThreadScopedValue(value, readThreadScopedValue(state, key))) {
          continue;
        }
        // The sampling ones are one object, gathered and applied together below.
        if (isThreadScopedParamKey(key)) {
          paramsPatch[key] = value;
        } else {
          target[key] = value;
        }
      }
      if (hasKeys(paramsPatch)) {
        nextState.params = { ...state.params, ...paramsPatch };
      }
      // Search and Thinking are exclusive on Kimi and the enforcing effect does not rerun on a
      // thread switch, so the restore drops thinking as clicking the Search pill does.
      if (
        isKimiCheckpoint(state.params.checkpoint) &&
        (applied.toolsEnabled ?? state.toolsEnabled) === true &&
        (applied.reasoningEnabled ?? state.reasoningEnabled) === true
      ) {
        applied.reasoningEnabled = false;
        if (state.reasoningEnabled !== false) target.reasoningEnabled = false;
      }
      // Pin what the chat shows now, or changing the defaults later would rewrite its modes. A
      // chat that already had a snapshot only needs a write if it carries a held edit.
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
    // Mirror setCheckpoint's persistence: dropping the checkpoint must also clear any stored external selection.
    saveLastExternalCheckpoint(null);
    saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
    return set((state) => ({
      queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      // An unload leaves the model the same way a switch does, so record what it was running with.
      ...(() => {
        const outgoing = rememberOutgoingModel(state, state.params);
        return outgoing ? { paramsByModel: outgoing } : {};
      })(),
      params: {
        ...state.params,
        checkpoint: "",
      },
      activeGgufVariant: null,
      // Nothing is picked, so residency has nothing to describe. Unknown, not null: null reads as
      // "was evicted".
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
      artifactsEnabled: false,
      mcpEnabledForChat: false,
      webFetchToolsEnabled: false,
      // Only the per-session enable pill resets; source/mode/top_k persist.
      ragEnabled: false,
      toolStatusByThreadId: {},
      toolLiveOutput: {},
      toolFullOutput: {},
      activeDiffusionCanvasByThreadId: {},
      kvCacheDtype: null,
      mlxKvBits: null,
      loadedMlxKvBitsRequested: null,
      mlxKvQuantReason: null,
      chatTemplateOverrideReason: null,
      mlxKvQuantNote: null,
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
      // Standing preference: survives unload, unlike the per-model knobs above.
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
  setReasoningEffort: (reasoningEffort) =>
    set((state) => {
      setScalarSettingVersion(
        "reasoningEffort",
        reasoningEffort,
        state.reasoningEffort,
      );
      return {
        reasoningEffort,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
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
  setCodeToolsEnabled: (codeToolsEnabled) =>
    set((state) => {
      saveBool(CHAT_CODE_TOOLS_ENABLED_KEY, codeToolsEnabled);
      if (codeToolsEnabled) saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
      return {
        ...(codeToolsEnabled
          ? { codeToolsEnabled, deepResearchEnabled: false }
          : { codeToolsEnabled }),
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
      // The level this chat carries, or the global one when it carries none.
      const permissionMode =
        threadScopedOverride("permissionMode") ?? loadPermissionMode();
      if (deepResearchEnabled) {
        saveBool(CHAT_TOOLS_ENABLED_KEY, false);
        saveBool(CHAT_IMAGE_TOOLS_ENABLED_KEY, false);
        saveBool(CHAT_CODE_TOOLS_ENABLED_KEY, false);
        saveBool(CHAT_ARTIFACTS_ENABLED_KEY, false);
        saveBool(CHAT_MCP_ENABLED_KEY, false);
        saveBool(CHAT_WEB_FETCH_TOOLS_ENABLED_KEY, false);
      }
      return deepResearchEnabled
        ? {
            deepResearchEnabled,
            toolsEnabled: false,
            codeToolsEnabled: false,
            imageToolsEnabled: false,
            artifactsEnabled: false,
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
  setArtifactsEnabled: (artifactsEnabled, options) =>
    set((state) => {
      if (options?.persist !== false) {
        saveBool(CHAT_ARTIFACTS_ENABLED_KEY, artifactsEnabled);
      }
      if (artifactsEnabled) saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
      return {
        ...(artifactsEnabled
          ? { artifactsEnabled, deepResearchEnabled: false }
          : { artifactsEnabled }),
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setShowCanvasMenuItem: (showCanvasMenuItem) =>
    set(() => {
      saveBool(CHAT_SHOW_CANVAS_MENU_ITEM_KEY, showCanvasMenuItem);
      return { showCanvasMenuItem };
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
      // The legacy toggle is a view over the level: on -> "ask", off -> "off". "full" is untouched.
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
  setPermissionMode: (permissionMode) =>
    set((state) => {
      // "full" is session-only (see init); ask/auto/off persist and keep the legacy confirm toggle in sync.
      savePermissionMode(permissionMode);
      if (permissionMode === "full") {
        // Full access sends confirm_tool_calls=false; keep the store flag in sync so metadata does
        // not report confirmations as enabled.
        saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
        return {
          permissionMode,
          bypassPermissions: true,
          confirmToolCalls: false,
          deepResearchEnabled: false,
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
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setBypassPermissions: (bypassPermissions) =>
    // Deliberately not persisted (see init): a reload must not keep the sandbox/confirmation
    // bypass unaccepted. Turning it off returns to the last persisted level.
    set((state) => {
      if (bypassPermissions) {
        // Full access never prompts; mirror confirm_tool_calls=false so metadata agrees.
        saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
        return {
          bypassPermissions,
          permissionMode: "full" as PermissionMode,
          confirmToolCalls: false,
          deepResearchEnabled: false,
          queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
        };
      }
      // Back to the level this chat carries, or the global one when it carries none.
      const permissionMode =
        threadScopedOverride("permissionMode") ?? loadPermissionMode();
      return {
        bypassPermissions,
        permissionMode,
        confirmToolCalls: permissionMode === "ask" || permissionMode === "auto",
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
  setToolConfirmation: (toolCallId, approvalId, sessionId, autoAllowKey) =>
    set((state) => ({
      toolConfirmations: {
        ...state.toolConfirmations,
        [toolCallId]: { approvalId, sessionId, autoAllowKey },
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
      // The entry under the shared key need not be the caller's: an abandoned composer has had its
      // own dropped and the next one's is sitting there.
      if (claim !== undefined && claim !== pendingAttachmentTargetClaim) {
        return state;
      }
      const pending =
        state.projectAttachmentTargetByThread[PENDING_CHAT_ATTACHMENT_KEY];
      // A chat that already made its own choice keeps it: the pending entry belongs to a chat that does not exist yet.
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
      // Turning it on adopts the settings on screen for the active model. Inside a chat those are
      // the chat's, so the outgoing snapshot's filter takes those keys out first.
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
      // Turning it on is an explicit statement about the model on screen, so the whole snapshot is
      // what it means, not a key-by-key patch.
      if (paramsByModel && state.settingsHydrated && state.params.checkpoint) {
        saveSettingsPatch({
          inferenceParamsByModel: {
            [state.params.checkpoint]: paramsByModel[state.params.checkpoint],
          },
        });
      }
      // Turning it off makes the settings on screen the one shared set; the global set can still
      // be the last model's, so write it or the next launch restores that.
      if (!rememberParamsPerModel && state.settingsHydrated) {
        saveSettingsPatch({ inferenceParams: snapshot });
        // Third write of the installation-wide sampling; level the copy as the others do.
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
        // Drop only this run's entry: a sibling behind the same key may still be running a tool.
        if (mine === undefined) return state;
        const rest = entries.filter((e) => e !== mine);
        if (rest.length > 0) {
          next[threadId] = rest;
        } else {
          delete next[threadId];
        }
      } else {
        // Same text from the same run means the same call, so keep startedAt.
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
  setContextPolicy: (contextPolicy) =>
    set((state) => {
      setScalarSettingVersion(
        "contextPolicy",
        contextPolicy,
        state.contextPolicy,
      );
      return {
        contextPolicy,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setCompactionHeadroomRatio: (compactionHeadroomRatio) =>
    set((state) => {
      setScalarSettingVersion(
        "compactionHeadroomRatio",
        compactionHeadroomRatio,
        state.compactionHeadroomRatio,
      );
      return {
        compactionHeadroomRatio,
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
  // A standing preference, persisted only on a successful load (see use-chat-model-runtime),
  // so an unapplied pick does not stick to the next session.
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
  // Write through to the visible thread's own entry so a value restored by the history loader
  // survives a switch: that loader runs once per mount and setActiveThreadId reads the map.
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

// Mirror token edits made through the shared store (e.g. Unsloth's field).
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
