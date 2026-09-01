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
import {
  type ChatPresetSource,
  type Preset,
  getPresetSource,
} from "../presets/preset-policy";
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
  savePersistedChatSettingsPatch,
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

/**
 * Permission level for local tool calls:
 * - "ask": always ask before every tool call runs.
 * - "auto" ("Approve for me", the default): only ask for calls the backend
 *   detects as high risk; ordinary dev commands run immediately. Sandbox stays on.
 * - "off": never ask; tool calls run automatically inside the sandbox
 *   (the original default before permission levels existed).
 * - "full" ("Full access"): no confirmations and the python/terminal sandbox
 *   is disabled. Session-only; never restored from storage.
 */
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
// Persist only the model-agnostic intents (auto/ngram/off). The model-specific
// drafter modes (mtp/mtp+ngram/dspark/dflash) and spec_draft_n_max stay session-only:
// a persisted choice would silently no-op on a model with no MTP head or no
// DSpark sidecar. Unknown -> auto.
const PERSISTED_SPEC_MODES = new Set(["auto", "ngram", "off"]);

export type RagSource = { type: "thread" } | { type: "kb"; kbId: string };

/** Where the composer files an attachment in a project chat. `project` indexes

/** Key a choice made in a chat that has no id yet lives under until it gets one. */
export const PENDING_CHAT_ATTACHMENT_KEY = "__pending__";

/** Bumped whenever the pending entry changes hands, so a composer that read it
 * can tell whether the one sitting there afterwards is still its own. */
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
// OCR scanned/image-only PDF pages at ingest time. On by default; off skips the
// extra vision pass (only matters when the loaded chat model has vision).
export const DEFAULT_RAG_OCR = true;
// Describe figures/charts in PDFs at ingest time so they become searchable. On by
// default (no-op without a vision model); off skips the per-figure vision calls.
export const DEFAULT_RAG_CAPTION = true;
export const DEFAULT_RESEARCH_WEBSITE_POLICY: ResearchWebsitePolicy = {
  allowedDomains: [],
  blockedDomains: [],
};
export const DEFAULT_RESEARCH_MODEL_TIMEOUT_SECONDS = 900;

/** 0 (unlimited) or a finite budget the settings patch and the run route both accept.
 * Anything else would be dropped from the patch and 400 the run, so the default stands in. */
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

// External provider selection is encoded into `params.checkpoint` as
// `external::<providerId>::<modelId>`. PersistedChatSettings omits `checkpoint`
// because the local-model side is mirrored by the backend's
// /api/inference/status.active_model. External selections have no such mirror,
// so without explicit localStorage persistence here the user's external pick
// is reset to the default on every refresh.
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

function saveLastExternalCheckpoint(value: string | null): void {
  if (typeof window === "undefined") return;
  try {
    if (value && isExternalModelId(value)) {
      window.localStorage.setItem(LAST_EXTERNAL_CHECKPOINT_KEY, value);
    } else {
      // Clear on switch to a local/empty checkpoint so the next refresh
      // won't override the now-active local selection.
      window.localStorage.removeItem(LAST_EXTERNAL_CHECKPOINT_KEY);
    }
  } catch {
    // Storage quota / private-mode failures are non-fatal; selection just
    // won't survive the refresh.
  }
}

// "enable_thinking_effort" is a hybrid: an on/off gate (enable_thinking) plus an
// effort level among a discrete set (e.g. GLM-5.2's high|max). It reuses the
// reasoning_effort dropdown UI but, unlike gpt-oss, can be fully disabled.
export type ReasoningStyle =
  | "enable_thinking"
  | "reasoning_effort"
  | "enable_thinking_effort";
/** One live DiffusionGemma denoising snapshot: the current canvas text at a
 *  given step of a given block (block/step are 0-based; total = steps in block). */
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

// Coalesce setting writes into one pendingPatch (deep merge for nested keys),
// flush on a trailing-edge debounce and on beforeunload so a pending patch
// survives tab close. Slider drags produce one HTTP write per quiet window.
type SettingsPatch = Parameters<typeof savePersistedChatSettingsPatch>[0];

const SETTINGS_DEBOUNCE_MS = 400;
let pendingPatch: SettingsPatch = {};
let pendingTimer: ReturnType<typeof setTimeout> | null = null;
let inflightFlush: Promise<void> = Promise.resolve();

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

// Discriminated unions, not partial patches: merging a `thread` pick into a
// stored `kb` one keeps `kbId`, which the backend's thread variant forbids.
const ATOMIC_SETTING_KEYS = new Set<string>(["ragSource"]);

// Maps of per-model objects, merged a level further in: two edits to different
// fields inside one debounce window must not replace each other.
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
    // A rejected patch is NOT requeued as it stands. The endpoint is
    // extra="forbid" and refuses the whole body on one bad field, so a patch the
    // server will never accept, requeued forever, makes every later save fail
    // too and the tab can no longer persist any chat setting. Keep the fields
    // the server did not name, drop the ones it did, and only reschedule when
    // the patch actually got smaller so the retry chain is bounded.
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

// Flushes handed to the network and not yet answered. pendingPatch and
// pendingTimer are both empty across that window, so they cannot answer "is a
// settings write still outstanding" on their own.
let unsettledFlushes = 0;

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

// A wedged PATCH must not hold a send open. Past this the run goes ahead on the
// value the server already has, which is exactly where it stood before.
const SETTINGS_FLUSH_TIMEOUT_MS = 2000;

/**
 * Send the debounced settings patch now and wait for it.
 *
 * Some settings are read by the backend out of SQLite at call time rather than
 * being carried in the request -- Search images picks the web_search schema that
 * way -- and the mirror above is a trailing-edge debounce, so a message sent
 * inside that window would run on the value before the toggle. Returns
 * immediately when nothing is queued, which is every send but one right after a
 * settings change.
 */
export async function flushPendingChatSettings(): Promise<void> {
  const queued = pendingTimer !== null || Object.keys(pendingPatch).length > 0;
  // Not just what is queued: the debounce may have fired already and handed its
  // patch to a request the server has not answered, which leaves both of those
  // empty while the value the backend reads is still the old one.
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

// Best-effort flush of any pending patch when the page is going away. keepalive
// lets the PUT outlive the unload; without it the browser cancels the fetch and
// the user's last slider drag is dropped.
function flushSettingsOnPageHidden(terminal: boolean): void {
  if (pendingTimer !== null) clearTimeout(pendingTimer);
  // A captured edit lives only in the debounce, so send it before the tab goes.
  // Only on a terminal event, though: the beacon PATCHes the row directly and a
  // thread whose row has not been created yet answers 404, where the normal path
  // would have created it first. visibilitychange is not terminal (it fires on
  // every tab switch), and the page is still there to await the ensure.
  // Chats whose newest values have just gone out, so an older unsettled snapshot for
  // the same chat is not sent after them: each beacon takes a higher seq than the last,
  // which would make the stale one the winner.
  const sentNewest = new Set<string>();
  const flushedThreadId = threadSettingsWriteThreadId;
  flushThreadScopedSettingsWrite(terminal);
  if (terminal && flushedThreadId !== null) sentNewest.add(flushedThreadId);
  // An edit made while its chat's read was still out lives only in heldThreadScopedEdits,
  // so the flush above does not see it. Effect cleanup is not guaranteed during unload and
  // its ordinary fetch would not outlive the page anyway, so send it from here, keepalive.
  if (terminal) {
    const heldThreadId = pendingPairingThreadId;
    void commitHeldThreadScopedEditsToTheirThread(true);
    if (heldThreadId !== null) sentNewest.add(heldThreadId);
    // And anything an earlier visibilitychange already flushed the normal way, which
    // this event would otherwise leave to a fetch the page is about to cancel.
    beaconUnsettledThreadSettingsWrites(sentNewest);
  }
  // An edit still waiting on hydration is a user edit like any other, and the
  // tab is going away, so send it rather than let the next session hydrate over it.
  drainPreHydrationPatch();
  if (Object.keys(pendingPatch).length === 0) return;
  inflightFlush = inflightFlush
    .catch(() => undefined)
    .then(() => flushSettingsPatch(true));
}

if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", () => flushSettingsOnPageHidden(true));
  // beforeunload is not the end of a page's life on every platform: mobile
  // Safari and backgrounded Android tabs are routinely discarded without it, and
  // a page restored from the back/forward cache never unloaded at all. pagehide
  // and the hidden transition of visibilitychange are the two the platform does
  // guarantee, so the debounced patch is normally already gone by the time an
  // unload would have run. Safe to add rather than replace: the pending patch is
  // swapped out before the request, so a second call with nothing queued returns
  // immediately and the two never send the same edit twice.
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

/**
 * Settings that describe the installation rather than the browser holding them.
 * Each entry pairs a localStorage slot with the /api/chat/settings field that
 * carries it to another browser or a remote session.
 */
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
    // A profile predating the visibility flag keeps Canvas shown through an
    // explicit plus-menu pin, which loadShowCanvasMenuItem reads.
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
    // A profile predating permission levels holds only the confirm toggle, and
    // loadPermissionMode derives the level from it.
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

/**
 * Send a changed mirrored setting to the backend. An edit made before the
 * initial GET lands is held rather than dropped: the request would race the
 * hydrating value, so it is replayed once hydration has applied the server's.
 * The mutation version is bumped straight away either way, so hydration leaves
 * a field the user has just set alone.
 */
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

/**
 * Write a setting to its localStorage slot and, for the settings mirrored to
 * /api/chat/settings, to the backend as well. Storage is the synchronous boot
 * cache; the backend copy is what a second browser or a remote session reads.
 */
function persistSetting(key: string, raw: string): void {
  const mirrored = MIRRORED_SETTING_BY_STORAGE_KEY.get(key);
  const writeGlobal = () => {
    // Before hydration the cache says nothing about the server's value, so an
    // explicit write is recorded even where it changes nothing locally. A
    // constraint write (deep research turning the tool pills off) has to reach
    // the backend, or hydration restores the toggle it was there to clear.
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
// captured on the way into a thread: edits made with a chat open must not move the defaults.
let globalThreadScopedDefaults: ThreadScopedSettings | null = null;
let threadSettingsWriteTimer: ReturnType<typeof setTimeout> | null = null;
let threadSettingsWriteThreadId: string | null = null;
// set only when pinning: the values applied on open, not whatever a model-capability effect
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
  // drops "full" with it: a stored bypass would come back without the warning dialog.
  return sanitizeThreadScopedSettings(source);
}

// keeps a model load from re-applying the global default over the pills the chat is running with.
export function threadScopedOverride<K extends ThreadScopedSettingKey>(
  key: K,
): ThreadScopedSettings[K] | undefined {
  // activeThreadScopedSettings is only refreshed when the debounce fires, so for the 400ms
  // after an edit it still holds the pre-edit values, and the capture path deliberately
  // wrote nothing to localStorage either. A model load or a status poll landing in that
  // window would read the old value, revert the pill, and then be persisted by the write
  // the edit itself scheduled. The store already holds the edit, so prefer it.
  // A pending PIN carries its own snapshot, which is what the chat is about to be
  // stored as, so that answers instead of the live store. Falling through to
  // activeThreadScopedSettings here reads null, because a chat being pinned had no
  // snapshot to begin with, and the load then puts the global values back over the
  // edit the queued write is about to persist.
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

/**
 * Fields the user has just set on the open chat, as opposed to ones a model's
 * capabilities moved in the store. The preservations below exist to stop a clamp erasing
 * a stored choice; they must not also undo a choice the user has only now made.
 */
const explicitlyEditedThreadFields = new Set<string>();

/**
 * Fields a provider constraint has moved in the store WITHOUT persisting them. Kimi's
 * builtin web search may not run with thinking, so the composer turns the other pill off
 * with `{ persist: false }`; that write bypasses persistSetting, and so the capture path,
 * on purpose. The value in the store is then the provider's, not the user's, and the
 * next full-snapshot write must not save it over what the chat has stored, or a chat
 * that asked for thinking comes back without it on a model that allows it.
 */
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

/**
 * The value of a sampling edit still waiting on its chat's read, if there is one.
 *
 * Pills can be read back off the store when the window closes, but the sampling keys
 * share `params` with the model's recommendation, so a load landing in the same window
 * would overwrite the edit first, pinning the model's value onto the chat as if the user
 * had chosen it. Captured at the edit instead; last entry wins, as the read-back did.
 */
function heldThreadScopedParamValue(key: string): unknown {
  for (let i = heldThreadScopedEdits.length - 1; i >= 0; i -= 1) {
    if (heldThreadScopedEdits[i].field === key) {
      return heldThreadScopedEdits[i].value;
    }
  }
  return undefined;
}

/**
 * The first value that is actually set. `??` cannot do this any more: `seed` is null when
 * the pin is cleared, and null there is the chat's own choice rather than a missing key.
 */
function firstSetThreadScopedValue(...values: unknown[]): unknown {
  return values.find((value) => value !== undefined);
}

/**
 * Put back the sampling keys the open chat owns, so a model load or status poll applying
 * that model's recommendation leaves the chat running on what it stored. Only an unpinned
 * chat falls through to the model's values.
 */
function restoreThreadScopedParams(params: InferenceParams): InferenceParams {
  const kept: Record<string, unknown> = {};
  for (const key of THREAD_SCOPED_PARAM_KEYS) {
    // Not ||, and not ?? either: 0, "", -1 and a cleared seed's null are all values a
    // user sets on purpose here.
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

/**
 * Take the open chat's own values back out of a snapshot about to be remembered against
 * a model: they belong to the chat, so leaving them in replays one chat's prompt and
 * sampling into the next chat opened on that model. What the model already remembered
 * wins over the installation values, so stepping off a model inside a chat does not
 * flatten a preference it was given outside one.
 *
 * A chat whose read is still out owns its keys just as much as an applied one (the edit
 * sits in heldThreadScopedEdits, not a snapshot); gating on the applied id alone let a
 * model switch in that window snapshot the chat's sampling into the outgoing model's
 * memory, which every other chat on that model then replays.
 */
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
    // For a held key the installation copy can still be null and the store no longer
    // holds the pre-edit value; the sample taken when the window opened is that value.
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

/**
 * Drop the sampling keys the open chat just took, so they reach neither the installation
 * defaults nor this model's memory: both are shared with every other chat. A model's own
 * values are never taken and pass straight through.
 */
function withoutCapturedThreadEdits(
  changedParams: PersistedInferenceParams,
  fromModelDefaults: boolean,
): PersistedInferenceParams {
  const shared: PersistedInferenceParams = {};
  for (const [key, value] of Object.entries(changedParams)) {
    if (
      isThreadScopedParamKey(key) &&
      !fromModelDefaults &&
      // By value: this runs inside the updater, so the store still holds the pre-edit
      // value, and a load in the same pairing window is what a read-back would find.
      captureThreadScopedEdit(key, null, value)
    ) {
      continue;
    }
    (shared as Record<string, unknown>)[key] = value;
  }
  return shared;
}

/**
 * Move the in-memory copy of the installation defaults to what was just written.
 * `applyThreadScopedSettings` falls back to it for a chat with no snapshot, so leaving it
 * stale means such a chat runs the sampling of whichever model was loaded before it.
 */
function noteThreadScopedDefaults(shared: PersistedInferenceParams): void {
  let next: Record<string, unknown> | null = null;
  for (const [key, value] of Object.entries(shared)) {
    if (!isThreadScopedParamKey(key)) continue;
    // Held field: the pairing capture restores it from the sample taken when the window
    // opened, so without this the pre-window value goes back and the in-memory defaults
    // stay behind the server's for the session, pinning onto the next snapshot-less chat.
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
  // the write replaces the row, and the sanitizer drops a live "full", so without this the
  // level the chat had stored would be erased by any pill toggled under Full access.
  if (
    settings.permissionMode === undefined &&
    threadId === threadScopedSettingsThreadId &&
    activeThreadScopedSettings?.permissionMode !== undefined
  ) {
    settings.permissionMode = activeThreadScopedSettings.permissionMode;
  }
  // Same for deep research, which apply() also holds back (external models and incognito
  // cannot run it). Without this, toggling any other pill in such a chat erases the true
  // it had stored, and it comes back off once the chat is on a local model again.
  //
  // Unless the user just changed it themselves: enabling Search, Code or Images clears
  // deep research deliberately, and restoring it here would ignore that and bring it
  // back, alongside Search, the moment the chat is on a local model again.
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
  // And for thinking, which a model that cannot stop thinking forces on in the store.
  // That true is the model's, so persisting it would erase a chat's stored false and
  // leave thinking on once the chat is back on a model where it is optional.
  if (
    threadId === threadScopedSettingsThreadId &&
    !explicitlyEditedThreadFields.has("reasoningEnabled") &&
    activeThreadScopedSettings?.reasoningEnabled === false &&
    settings.reasoningEnabled !== false &&
    useChatRuntimeStore.getState().reasoningAlwaysOn
  ) {
    settings.reasoningEnabled = false;
  }
  // Same again for every pill the model-selection pass in chat-page clamps off in the
  // store without touching the snapshot. The clamp is the model's, not the user's, so it
  // must not erase what the chat had stored; the condition is each pill's own capability
  // rule, which is exactly when the user could not have turned it off themselves.
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
  // And for the pills a provider constraint moved without persisting them: the store
  // holds the provider's value there, so the chat keeps the one it was stored with.
  if (keepsStoredValueUnderConstraint("reasoningEnabled", threadId, settings)) {
    settings.reasoningEnabled = activeThreadScopedSettings?.reasoningEnabled;
  }
  if (keepsStoredValueUnderConstraint("toolsEnabled", threadId, settings)) {
    settings.toolsEnabled = activeThreadScopedSettings?.toolsEnabled;
  }
  if (threadId === threadScopedSettingsThreadId) {
    activeThreadScopedSettings = settings;
  }
  // Spent: this snapshot has taken them into account, and the next one is about
  // whatever the user does next.
  explicitlyEditedThreadFields.clear();
  return settings;
}

const THREAD_SETTINGS_REPLAY_KEY = "unsloth_chat_thread_settings_replay";
const THREAD_SETTINGS_REPLAY_TIMEOUT_MS = 10_000;

/**
 * Snapshots a terminal event sent but could not confirm, kept where the next session
 * will find them. The beacon cannot await anything: a chat whose row is still being
 * created answers 404 and the edit is gone, and the creation that follows knows nothing
 * about it. Writing the attempt down costs nothing and makes the loss recoverable.
 */
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

/**
 * Re-send anything the last session could not confirm. Safe to run always: the seq the
 * body carries is that session's own, so a replay of a write that did land is refused
 * by the server rather than reverting anything newer.
 */
export function replayUnconfirmedThreadSettings(): void {
  // Once per session: it is called from both hydration outcomes, and sending each body
  // twice would race two writes carrying the same seq for the same row.
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
    // Bounded: every settings write in the session waits for these, so a socket that
    // never settles would leave the whole session unable to persist anything.
    const timeout = new AbortController();
    const timer = setTimeout(() => timeout.abort(), THREAD_SETTINGS_REPLAY_TIMEOUT_MS);
    const request = authFetch(`/api/chat/threads/${encodeURIComponent(threadId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: timeout.signal,
    })
      .finally(() => clearTimeout(timer))
      // Only an ok response means it landed. authFetch resolves for 404 and 5xx too,
      // and the missing-row case this exists for is exactly the one that 404s, so
      // dropping the entry on any settled promise would throw the edit away for good.
      .then((res) => {
        // Only the body this request carried. A terminal event in THIS session can
        // store a newer body for the same thread while the replay is still out, and
        // that one is unconfirmed by definition: dropping it because an older replay
        // succeeded would leave the newest edit with nothing to recover it from.
        if (res.ok) forgetReplayedThreadSettings(threadId, body);
      })
      .catch(() => undefined);
    sent.push(request);
  }
  // Every snapshot write waits for this. The replay carries the PREVIOUS session's
  // writer id, so the server treats it as unrelated to anything this session sends and
  // will apply it whenever it arrives: on a slow connection a full snapshot from last
  // time could land after an edit made just now and revert it.
  threadSettingsReplaySettled = Promise.all(sent).then(() => undefined);
}

// Resolved once the previous session's unconfirmed writes have been answered.
let threadSettingsReplaySettled: Promise<void> = Promise.resolve();
let threadSettingsReplayStarted = false;

/**
 * Drop one entry, leaving the rest for the next attempt. `expected` narrows that to the
 * exact body the caller sent, for callers whose success says nothing about a newer entry
 * stored since; a caller that has itself just written the row passes nothing, because its
 * values supersede whatever the entry holds.
 */
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

/**
 * Send the snapshot on tab close. The ensure-then-update chain cannot finish during unload, so
 * the row write goes straight out with keepalive, and what it could not confirm is left for
 * the next session to replay: a thread whose row is still being created answers 404, and the
 * creation that follows would otherwise land without the user's last edit.
 */
function sendThreadScopedSettingsBeacon(
  threadId: string,
  snapshot: ThreadScopedSettings | null,
  merge = false,
): void {
  // A merge carries only what the user touched, for the chat whose own snapshot was
  // never read; sending a replacement built from the defaults on screen would erase
  // the rest of its row. Everything else replaces, as the debounced write does.
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
  // The beacon carries the newest values but skips the chain, so an older write would
  // otherwise land after it and put the stale snapshot back. The ticket stands down the
  // ones still queued; the abort ends the one already out, which no ticket can reach.
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

// One chain per thread. The write REPLACES settings_json rather than merging it, so two
// unordered writes do not merge into the newer one, they pick a winner: a slow first
// request landing after a fast second restores the settings the user just moved away from.
const threadSettingsWriteChains = new Map<string, Promise<unknown>>();
// Newest snapshot issued per thread, so a write that is still waiting its turn can tell it
// has been superseded and skip its PATCH rather than reinstate what it captured.
const threadSettingsWriteTickets = new Map<string, number>();

// The request each thread currently has out, so a newer snapshot can stand it down.
const threadSettingsWriteAborts = new Map<string, AbortController>();

/**
 * Stamps each snapshot write so the server can refuse this tab's own older ones, which
 * is the case that needs it: a keepalive sent on unload can be undone by a PATCH the
 * server already had in hand, and no client-side abort reaches that.
 *
 * The id makes the ordering per-tab. A plain counter, and never a clock: comparing one
 * machine's numbers with another's means the browser that happens to be behind has
 * every edit refused while still being told it saved. Across tabs the last write wins,
 * as it did before any of this.
 */
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
  // Built now, not inside the chain: the snapshot must describe the store as it is at the
  // moment of the edit, not as it will be once the previous write has finished.
  const settings = buildThreadScopedSnapshot(threadId, snapshot);
  // Stamped with the snapshot, not with the request: the seq has to say when the edit
  // happened, not when its turn in the chain came up.
  const settingsSeq = nextThreadSettingsSeq();
  const ticket = takeThreadSettingsWriteTicket(threadId);
  const previous = threadSettingsWriteChains.get(threadId) ?? Promise.resolve();
  const next = previous
    .catch(() => undefined)
    // Last session's unconfirmed writes go first, or one of them can land on top of
    // this edit: it carries a different writer id, so nothing on the server orders it.
    .then(() => threadSettingsReplaySettled)
    .catch(() => undefined)
    .then(async () => {
      // superseded while queued: sending this would undo the newer snapshot, and the
      // newer one is what the replay entry should be measured against, so this counts
      // as landed for the purposes of the caller.
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
        // This session's values are now the row's, so a leftover replay entry from the
        // last one would revert them if it were ever retried.
        forgetReplayedThreadSettings(threadId);
        return true;
      } catch {
        // the chat still behaves as edited; only the snapshot for the next visit is lost.
        // An abort lands here too, and that one is deliberate: a newer snapshot won.
        if (!controller.signal.aborted) warnSettingsPersistenceFailure();
        // An abort means a newer write won and is tracked in its place; a real failure
        // means nothing reached the row, and the caller needs to know.
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

/**
 * Settle this chat's snapshot before anything reads it back. A chat edited, left and
 * re-entered has its PATCH in flight, and a GET overtaking it returns the pre-edit
 * snapshot, which then gets applied over the values the user set and written back on the
 * next edit. Pending debounce included, or the read races the timer instead.
 */
export async function settleThreadScopedSettingsForCopy(
  threadId: string,
): Promise<void> {
  // An edit made while this chat's read was still out lives in neither the debounce nor
  // the chain, so settling those alone leaves it behind and the copy takes the old row.
  // Only for a caller that is about to read the row server side: the pairing itself
  // waits on this function's sibling, which must not close the window it just opened.
  if (pendingPairingThreadId === threadId) {
    await commitHeldThreadScopedEditsToTheirThread();
  }
  // A flushed replacement write that failed resolves false rather than throwing, so
  // awaiting the chain alone cannot tell a saved edit from a lost one. The copy is made
  // server side from the row: going ahead on a failed write hands the new chat the
  // pre-edit snapshot and tells the user nothing.
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

/**
 * Every row write that has already been started, landed.
 *
 * Not a flush: a debounce that has not fired yet is left where it is, so a caller cannot
 * use this to make a write happen earlier than the store would have. It is for a caller
 * that has just let the debounce fire and now needs the rows before it reads them back,
 * and does not know which chats the store decided to write.
 *
 * The chain a write runs on ends in `await import("../utils/chat-history-storage")`. How
 * many event-loop turns that costs is a property of the machine -- the specifier has no
 * extension, so it resolves through a hook, which means filesystem work under a test
 * runner and a chunk fetch in a browser. A caller that instead spins a fixed number of
 * turns, or races the loader with an import of its own, is asserting something about the
 * machine rather than about the store. Awaiting the chains asserts the thing itself.
 */
export async function awaitStartedThreadScopedSettingsWrites(): Promise<void> {
  // A chain that settles can leave a newer one behind it for the same chat, so this
  // repeats until the map is empty rather than awaiting one snapshot of it. Bounded, so a
  // write that keeps rescheduling itself surfaces as a failed assertion, not as a hang.
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

/**
 * Start a normal write and keep its snapshot until it settles.
 *
 * Some browsers fire visibilitychange(hidden) and then pagehide. The first flushes
 * normally and clears the pending snapshot, so the terminal event would find nothing to
 * beacon and a discarded page takes the ordinary fetch with it. Holding the snapshot
 * lets the terminal path send it again, keepalive.
 */
function trackUnsettledThreadSettingsWrite(
  threadId: string,
  snapshot: ThreadScopedSettings | null,
): void {
  // Identity, not value: two ordinary edits both carry a null snapshot, so comparing
  // the value let the first request's settle delete the second one's tracking, and a
  // terminal event then found nothing to resend for a write still in flight.
  const entry: UnsettledThreadSettingsWrite = { snapshot };
  unsettledThreadSettingsWrites.set(threadId, entry);
  void writeThreadScopedSettings(threadId, snapshot).then((landed) => {
    // Only a write that reached the row stops being unsettled. Dropping it on failure
    // left nothing for a terminal event to beacon, so the edit was lost on close and
    // came back reverted, which is the opposite of what this tracking is for.
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

/**
 * Re-send anything a normal flush has not landed yet, with keepalive.
 *
 * `alreadySent` names the threads the terminal flush has just beaconed. Those carry the
 * newest values, and every beacon takes a higher seq than the last, so re-sending an
 * older unsettled snapshot for the same chat afterwards would make the stale one win.
 */
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
      // Tracked the same way a manual flush is: once the timer has fired there is no
      // pending debounce left for a terminal event to find, so without this the write
      // in flight is the only copy and the page teardown cancels it.
      trackUnsettledThreadSettingsWrite(pendingThreadId, pendingSnapshot);
    }
  }, THREAD_SETTINGS_DEBOUNCE_MS);
}

// the chat whose snapshot is in flight, and the edits made before it landed. The store already
// shows them; only the read can say whether they belong to this chat or to the defaults, so they
// wait here. Writing them globally in the meantime moved every other chat's default and was then
// overwritten by the arriving snapshot, so the click both leaked and appeared to do nothing.
let pendingPairingThreadId: string | null = null;
/** The store's thread-scoped values as they stood when the current pairing began. */
let pairingWindowDefaults: ThreadScopedSettings | null = null;
/** and the chat it was sampled for, so a retry does not resample over its own edit. */
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
  // What the installation defaults were before this chat could edit anything. Recorded
  // here because there is nowhere later to recover it from: an edit made during the
  // window overwrites the store, and on the first pairing of a session there is no
  // earlier capture to fall back on.
  //
  // Once per chat, not once per attempt: a retry after a failed read runs with the held
  // edit already in the store, and re-sampling would take that edit for a default.
  if (pairingWindowDefaultsThreadId !== threadId) {
    pairingWindowDefaultsThreadId = threadId;
    // Switching straight from one saved chat to another, the store still holds the
    // OUTGOING chat's values at this point, so it is not a source of defaults. The
    // capture made when that chat was paired is, and it exists whenever a chat is
    // applied. Only with no chat applied does the store itself hold the defaults.
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

/**
 * Release ONE chat's gate. Releasing all of them let a run started for A be freed by B's
 * pairing ending, which is the whole reason these are held per chat; the composer flag
 * is separate, and only tracks whether the chat now on screen is still waiting.
 */
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

// Per chat, not one for all of them: a run started for A must not be released by B's
// pairing ending, or it resumes and reads B's settings for A's run.
const pairingSettledByThreadId = new Map<
  string,
  { promise: Promise<void>; resolve: () => void }
>();

/**
 * Resolves once `threadId`'s own settings are known, or immediately if they already
 * are. Every run goes through the adapter, so awaiting this there is what stops a
 * Reload, a Continue or an edit-and-send from starting on the installation defaults
 * that stand in while the read is out: a chat stored as "ask" would run as "off".
 * Bounded, so it cannot wait for the life of the tab.
 *
 * Resolves false when the wait ran out. The caller must NOT go ahead on that: the only
 * way to reach it is a chat left mid-read, whose gate is held shut deliberately and
 * whose settings never arrived, so the store now describes some other chat. Running
 * anyway is exactly the mix-up the gate exists to prevent, only 10 seconds later.
 */
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

// Longer than the read can take (THREAD_READ_RETRIES retries, each bounded by
// THREAD_READ_TIMEOUT_MS and spaced THREAD_READ_RETRY_MS apart, then a give-up that
// opens the gate itself), so a slow read never reaches this and a run is only refused
// for a chat whose pairing has genuinely been abandoned.
const THREAD_PAIRING_WAIT_MS = 30_000;

/** The chat turned out to own no snapshot: send the held edits to the defaults, as before. */
export function releaseHeldThreadScopedEdits(): void {
  const held = heldThreadScopedEdits;
  const threadId = pendingPairingThreadId;
  heldThreadScopedEdits = [];
  pendingPairingThreadId = null;
  pairingWindowDefaultsThreadId = null;
  // The answer is in: this chat owns no snapshot, so the defaults ARE its settings and
  // a run waiting on it can go.
  closeThreadScopedPairingGate(threadId);
  for (const edit of held) {
    // Written to the defaults now, so the value hydration held back is history.
    hydratedDefaultsByHeldField.delete(edit.field);
    edit.writeGlobal?.();
  }
}

/**
 * The user left before the read finished. The edit belongs to the chat it was made in, so
 * write it there; replaying it into the installation defaults would move every
 * snapshot-less chat, which is the leak this path exists to prevent.
 *
 * The store still holds the edited values here: the incoming chat's own read has not
 * resolved yet, so nothing has overwritten them. A chat with no row of its own writes
 * nothing (`ensureStoredChatThread` only adopts a record that already exists), which
 * correctly leaves an unsaved chat following the defaults.
 */
export function commitHeldThreadScopedEditsToTheirThread(
  keepalive = false,
): Promise<void> {
  const threadId = pendingPairingThreadId;
  const held = heldThreadScopedEdits;
  heldThreadScopedEdits = [];
  pendingPairingThreadId = null;
  // pairingWindowDefaultsThreadId is deliberately NOT cleared: a failed read commits
  // and then re-pairs the same chat, and the store holds the edit by then, so letting
  // the retry resample would take that edit for the installation default. Leaving for a
  // different chat resamples anyway, because the id no longer matches.
  //
  // The gate is NOT opened here. This runs when the user leaves mid-read, and this
  // chat's snapshot was never loaded, so a run still waiting on it must not be told the
  // settings are known: the store now shows the chat the user moved to.
  closeThreadScopedPairingGate(null);
  if (threadId === null || held.length === 0) return Promise.resolve();
  const changes = heldThreadScopedChanges(held);
  // Read off the store above, so only now that the values are safely captured.
  restoreDefaultsOverCommittedEdits(threadId, held);
  if (keepalive) {
    sendThreadScopedSettingsBeacon(threadId, changes, true);
    return Promise.resolve();
  }
  // Returned rather than fired and forgotten: forking copies settings_json server side,
  // so it has to be able to wait for a held edit to reach the row first.
  return mergeThreadScopedSettingsIntoRow(threadId, changes);
}

/**
 * Put the installation values back over the edits just written to the chat the user left.
 * Without this the store keeps showing that chat's temperature and system prompt, and the
 * next chat opened takes them: a snapshot-less one captures the store as the installation
 * defaults and is pinned with them, a brand new one simply runs on them.
 *
 * Only when the chat is actually being left. A failed read about to be retried commits the
 * same way while the chat stays open, as does a fork; putting the defaults back there
 * would undo an edit the user is looking at.
 */
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
    // The server answered for this field while the window was open, so that value is
    // the installation's; the pre-window sample is only what this browser had cached.
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
  // setState and not setParams: these values are already the installation's, and going
  // through the setter would persist them back to it and to the loaded model's memory.
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
  // Same reader the snapshot path uses: sampling keys sit under `params`, so a direct
  // field read returns undefined and the sanitizer drops them.
  for (const edit of held) {
    edited[edit.field] = readThreadScopedValue(
      live,
      edit.field as ThreadScopedSettingKey,
    );
  }
  return sanitizeThreadScopedSettings(edited);
}

/**
 * PATCH just the fields in `changes`, leaving the rest of the row alone. Used when the
 * store cannot be trusted to describe the chat, which is any time its read has not
 * landed: the store is showing the installation defaults, and a replacement built from
 * those would erase everything the chat had stored that the user did not touch.
 */
async function mergeThreadScopedSettingsIntoRow(
  threadId: string,
  changes: ThreadScopedSettings,
): Promise<void> {
  const settingsSeq = nextThreadSettingsSeq();
  const ticket = takeThreadSettingsWriteTicket(threadId);
  const previous = threadSettingsWriteChains.get(threadId) ?? Promise.resolve();
  const next = previous
    .catch(() => undefined)
    // Last session's unconfirmed writes go first, or one of them can land on top of
    // this edit: it carries a different writer id, so nothing on the server orders it.
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
        // Rethrown as well as toasted: a fork waits on this to know the row holds the
        // edit before the backend copies it, and a resolved promise would let it make
        // a fork carrying the pre-edit snapshot.
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

/**
 * What the server said an installation default is, for fields hydration had to skip
 * because the user had just set them inside a chat whose read was still out. Those edits
 * belong to the chat, so when its pairing window closes the default goes back to the
 * value from before the window, which is this browser's pre-hydration copy. The server's
 * is the authoritative one and lands nowhere else.
 *
 * A default published by a model that loaded inside the window is recorded here for the
 * same reason: setParams already sent it to the installation, so restoring the pre-window
 * sample over it would leave this session's copy behind the server's.
 */
const hydratedDefaultsByHeldField = new Map<string, unknown>();

/** Is this field an edit waiting on its chat's read, and so not the installation's to set? */
function isHeldThreadScopedField(field: string): boolean {
  return heldThreadScopedEdits.some((edit) => edit.field === field);
}

// reports whether the edit was taken; with no chat open the caller persists globally as before.
function captureThreadScopedEdit(
  field: string,
  writeGlobal: (() => void) | null = null,
  value?: unknown,
): boolean {
  if (!isThreadOwnedSettingKey(field)) return false;
  const threadId = useChatRuntimeStore.getState().activeThreadId;
  if (threadId === null) return false;
  // both ids: between a switch and its snapshot arriving the store still holds the old values.
  if (threadId === threadScopedSettingsThreadId) {
    explicitlyEditedThreadFields.add(field);
    // Set by the user now, so the chat stores this rather than what it had before a
    // constraint moved the same field.
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

/**
 * Persist a value a model load applied rather than one the user picked. Before
 * hydration such a write only reflects this browser's cache, so treating it as
 * an edit would both block the stored preference from hydrating and replay the
 * load's default over it. It reaches the cache and waits for the server's.
 *
 * `stillCurrent` says the value matches the live store. A load captures its
 * settings up front and persists them on success, so one that started before
 * hydration and finished after it would otherwise write the stale capture over
 * the preference that just arrived.
 */
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

/**
 * Resolve the web-search / code-execution pill state to apply when a model
 * loads. Honors the user's persisted preference so a tool-capable model never
 * re-enables a pill the user turned off, and never re-disables one they turned
 * on, and the open chat's own state wins over the installation default. When no
 * preference has been expressed the pills stay off: tool execution is opt-in, so
 * the person enables it with a click rather than a model turning it on for them.
 */
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

// The installation's own answer to the preserve-thinking switch, or null while it has
// never given one. Hydration records the stored preference and the composer toggle
// records the click; nothing else writes it.
let storedPreserveThinking: boolean | null = null;

/** Record the preference a stored value or a toggle just expressed. */
function notePreserveThinkingPreference(value: boolean): void {
  storedPreserveThinking = value;
}

/**
 * The preserve-thinking value a model load or a status adoption should publish. The
 * family default the backend resolves (on for Qwen3.8, off everywhere else) is a
 * DEFAULT: it seeds the switch where the installation has never answered, and never
 * replaces an answer it gave, the same rule resolveToolsEnabledOnLoad applies to the
 * tool pills. That is also what makes a cold boot deterministic -- the settings GET and
 * the inference status race each other, and a load write that cannot overwrite a stored
 * preference leaves the same result whichever lands first.
 */
export function resolvePreserveThinkingOnLoad(resp: {
  supports_preserve_thinking?: boolean | null;
  preserve_thinking_default?: boolean | null;
}): boolean {
  return storedPreserveThinking ?? preserveThinkingDefaultFromLoad(resp);
}

// The visibility flag shipped after the menu pins, so when it is absent,
// profiles that had explicitly pinned Canvas keep it visible.
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

/**
 * "full" is intentionally not restorable: it disables the sandbox and every
 * confirmation gate, so it must be re-enabled (through the warning dialog)
 * each session. First run falls back to the legacy "Confirm tool calls"
 * toggle so existing users keep their behavior (on -> ask, explicitly
 * off -> "off", i.e. no prompts); fresh installs default to "auto".
 */
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

// Canonicalises any backend value onto the Speculative Decoding dropdown's
// modes ("auto"/"mtp"/"dspark"/"dflash"/"ngram"/"mtp+ngram"/"off"/null). Backend-only
// legacy aliases map to their closest UI mode.
export function normalizeSpeculativeType(
  v: string | null | undefined,
): string | null {
  if (v == null) return null;
  const s = String(v).trim().toLowerCase();
  if (!s) return null;
  if (s === "auto" || s === "default") return "auto";
  if (s === "off") return "off";
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

// MTP / null / unknown values are left unwritten so they stay session-only.
// Called from the load path so only an applied preference is persisted, not an
// unapplied dropdown edit the user might Reset or abandon before Apply.
export function saveSpeculativeType(value: string | null): void {
  if (value && PERSISTED_SPEC_MODES.has(value)) {
    persistLoadDerivedSetting(
      CHAT_SPECULATIVE_TYPE_KEY,
      value,
      useChatRuntimeStore.getState().speculativeType === value,
    );
  }
}

// GPU Memory strategy is a standing preference (like speculative type), not a
// per-model setting: a "manual" choice persists across model switches and reloads.
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

/** Persist the GPU Memory mode after a load, but only for a non-diffusion GGUF:
 *  non-GGUF has no such mode, and diffusion runs mode-agnostic (reports "auto"),
 *  so neither must clobber the standing manual preference. */
export function persistGpuMemoryModeOnLoad(
  resp: { is_gguf?: boolean; is_diffusion?: boolean },
  mode: "auto" | "manual",
): void {
  if (resp.is_gguf && !resp.is_diffusion) saveGpuMemoryMode(mode);
}

// Re-exported from its dependency-free home so existing imports keep working.
export { GPU_LAYERS_AUTO } from "../lib/gpu-placement";

// Round real-valued shares to integers summing exactly to `total`, giving the
// leftover units to the largest fractional parts (largest-remainder method).
function largestRemainder(shares: number[], total: number): number[] {
  const out = shares.map((x) => Math.floor(x));
  let rem = total - out.reduce((a, b) => a + b, 0);
  const byFrac = shares
    .map((x, i) => ({ i, frac: x - Math.floor(x) }))
    .sort((a, b) => b.frac - a.frac);
  for (let k = 0; rem > 0 && k < byFrac.length; k++, rem--) out[byFrac[k].i] += 1;
  return out;
}

// Spread `total` layers across GPUs in proportion to `weights` (e.g. per-GPU
// VRAM), as integers summing exactly to `total`; even split for all-zero/empty
// weights. Default per-GPU layer split before the user edits it (mirrors
// llama.cpp's free-VRAM default).
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

// Set GPU `index` to `value` and rebalance the rest so per-GPU counts still sum
// to `total`; others absorb the remainder in proportion to their counts (evenly
// if all zero). The --tensor-split editor: counts are sent verbatim, and
// llama.cpp gives each GPU exactly its count when gpu_layers == sum(counts).
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

// Validate a persisted gpu_ids pick against the GPUs present right now, before
// restoring it from remembered settings. Returns null (= automatic) when the
// pick is stale (none of the saved ids exist, or the host can't pin a multi-GPU
// set), so a saved [1] on a now-1-GPU host doesn't get sent and rejected with no
// way to clear it. A null pick (= automatic) passes through unchanged, and an
// unpopulated device cache leaves the pick alone (the backend still guards).
// An explicit null namespace means discovery had not completed when the live
// state was captured, while an absent namespace is a legacy physical-ID pick.
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

// Store fields derived from a load/status response's GPU-memory settings.
// Shared by every load path so the manual-knob round-trip can't drift.
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
  // GPU-memory state is meaningful only for a GGUF chat load. A non-GGUF response
  // still carries gpu_memory_mode (its default "auto" is serialized), so gate on
  // the authoritative is_gguf flag, not the field's presence -- otherwise loading
  // a transformers model would reset the standing manual preference.
  if (!resp.is_gguf) {
    // Clear the GPU pick / offload baseline a prior GGUF load may have left, so it
    // reflects the non-GGUF model (no pin) -- else a stale loadedGpuIds reads as
    // dirty (gpuIdsDirty is ungated) and Reset restores it while the picker is
    // hidden. gpuMemoryMode (the standing preference) is kept, but its loaded
    // baseline clears to null so Reset preserves the preference, not a stale mode.
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
  // Keep the user's placement pool editable across status/load hydration.
  // gpu_ids remains the effective fitted subset for diagnostics.
  const reportedGpuIds = requestedGpuIdsFromResponse(resp);
  const gpuIndexKind =
    reportedGpuIds == null
      ? null
      : cachedPinnableGpuIndexKind(resp.is_diffusion === true);
  // A numeric ID is unsafe to adopt or persist once discovery says it is NOT a
  // physical CUDA/ROCm ID or a Vulkan ordinal (gpuIndexKind === null, cache
  // warm). But while discovery is still cold (gpuIndexKind === undefined) the
  // namespace is merely deferred, not rejected -- keep the just-applied pin so
  // a reload/rollback in that window doesn't omit gpu_ids and let llama.cpp
  // fall back to every device. A later status refresh resolves the namespace
  // once the shared system cache warms.
  const gpuIds =
    reportedGpuIds != null && gpuIndexKind !== null ? reportedGpuIds : null;
  // A shim without --ngl reports Auto while the backend still holds the ask, so recover
  // it: in-memory state survives a reload but not a refresh.
  const droppedSplit = recoverDroppedDiffusionSplit(
    resp.is_diffusion,
    mode,
    resp.diffusion_requested_ngl,
  );
  // Layer/MoE/split knobs apply (and are reported) only in manual mode; in auto
  // the server ignores them, so don't seed the loaded baseline or the editable
  // knobs with values it never applied. In manual, the server reports gpu_layers
  // = -1 under Auto, which round-trips the slider back to its Auto position.
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
          // Auto ignores these, so reset the editable knobs too (not just the
          // loaded baseline) -- else a later switch back to Manual would snapshot
          // and send a previous model's stale gpuLayers/nCpuMoe/split that this
          // load never applied. Mirrors the non-GGUF branch above.
          // Diffusion excepted: an "auto" diffusion response may be an older shim
          // DROPPING a manual split. Restore the ask when the response carries it (a
          // refresh has none left in memory), else keep what is standing. Resetting the
          // slider would turn the ask into manual/-1, unapplyable even after the
          // unsloth_zoo upgrade that adds --ngl.
          ...(resp.is_diffusion
            ? droppedSplit != null
              ? { gpuLayers: droppedSplit }
              : {}
            : { gpuLayers: GPU_LAYERS_AUTO }),
          nCpuMoe: 0,
          splitRatio: null,
        };
  return {
    // A diffusion GGUF reporting "auto" ran on the runner's defaults, so an inert standing
    // manual preference must survive it. But "manual" means a split was actually applied
    // (#7574): adopt it, or a refresh hydrates back to "auto" while the runner serves one.
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
    // gpuIndexKind is `undefined` only in the deferred (cache-cold) case here,
    // since a `null` kind already forced gpuIds to null above. Normalize to the
    // explicit-null-namespace convention (discovery not complete yet).
    selectedGpuIndexKind: gpuIds == null ? null : (gpuIndexKind ?? null),
    loadedGpuIds: gpuIds,
    loadedGpuIndexKind: gpuIds == null ? null : (gpuIndexKind ?? null),
    ...manualKnobs,
  };
}

// re-exported here so every existing importer keeps its path; defined in a leaf module
// that a test can load without pulling the whole store in.
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

/**
 * One live run behind `runningByThreadId[id]`, with the `local` flag it started with so the
 * model-swap gate can tell llama-server runs from external ones when runs share a key.
 */
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
  /**
   * The open chat's own settings have been asked for but have not arrived. The store is
   * showing the installation defaults meanwhile, so a run started now would be captured
   * with them: a chat stored as "ask" could run tools without asking. Sending waits.
   */
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
  /**
     * The subset of `runningByThreadId` decoding on the local llama-server. Swapping the local
     * model neither interrupts an external-provider chat nor needs its consent, which is why
     * the backend keeps those out of `active_generations` too.
     */
  localRunByThreadId: Record<string, boolean>;
  /**
     * Which runs set `runningByThreadId[id]`; see `setThreadRunning`'s `owner`. A list, not one
     * entry: runs without a resolved thread id share the "__default" key, so one entry would let
     * a newer run's clear delete an older run's flag while it still generates.
     */
  runOwnerByThreadId: Record<string, ThreadRunOwner[]>;
  cancelByThreadId: Record<string, () => void>;
  /**
     * Backend cancels for the threads generating in the background. `cancelByThreadId` only holds
     * the visible thread's `cancelRun()`, so the adapter parks a closure here that POSTs that
     * run's own cancel_id. A list for the same reason as `runOwnerByThreadId`: "__default" is shared.
     */
  serverCancelByThreadId: Record<string, (() => void)[]>;
  autoTitle: boolean;
  hfToken: string;
  modelsError: string | null;
  // Set only when a LOAD fails (not refresh/list/unload, which use modelsError);
  // lets the attach gates flag a failed load vs "no model picked".
  lastModelLoadError: string | null;
  activeGgufVariant: string | null;
  /**
   * What /api/inference/status says is resident, as opposed to what the picker
   * has selected. undefined until the first status read, so the header does not
   * flash "not loaded" before anything is known. Loading an image or video
   * model evicts the chat model (one GPU owner at a time), which is otherwise
   * invisible here: the selection survives it.
   */
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
  /**
   * The model id the OpenRouter router picked for the most recent stream when
   * the active checkpoint is the openrouter/free meta-model. Updated when a
   * chunk's `model` field differs from the requested id; cleared on a
   * non-OpenRouter model. UI display only (appended after `openrouter/free:`).
   */
  lastOpenRouterChosenModel: string | null;
  reasoningStyle: ReasoningStyle;
  reasoningEffort: ReasoningEffort;
  supportsReasoningOff: boolean;
  reasoningEffortLevels: readonly ReasoningEffort[];
  supportsPreserveThinking: boolean;
  preserveThinking: boolean;
  supportsTools: boolean;
  /**
   * Whether the active external provider exposes a server-side web_search tool
   * (OpenAI's /v1/responses today). Distinct from `supportsTools` (the local
   * tool runtime): this only enables the composer's Search pill for external
   * models. Local models keep `supportsTools` only.
   */
  supportsBuiltinWebSearch: boolean;
  /**
   * Whether the active external provider exposes a server-side code-execution
   * tool (Anthropic's `code_execution_20250825` on Claude 4.x). Distinct from
   * `supportsTools` like supportsBuiltinWebSearch: Anthropic dispatches it
   * server-side. Read by both composers' Code pill gate.
   */
  supportsBuiltinCodeExecution: boolean;
  /**
   * Whether the active external provider exposes a server-side image-generation
   * tool (OpenAI's Responses-API `image_generation`). Gates the composer's
   * Images pill. Local models never receive it (their runtime can't dispatch it).
   */
  supportsBuiltinImageGeneration: boolean;
  /**
   * Whether the active external provider exposes a server-side web_fetch tool
   * (Anthropic's `web_fetch_20250910` / `web_fetch_20260209`). Gates the
   * composer's Fetch pill, independent of Search.
   */
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
  /** Per-chat override of that default, so a pick in one chat does not redirect
   * the rest. Session-only: a reload falls back to the saved default. */
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
  /**
   * When on, local Unsloth tool calls pause for an explicit allow/deny in the
   * chat before they run.
   */
  confirmToolCalls: boolean;
  /**
   * Bypass Permissions: when on, tool calls run with no confirmation gate
   * AND the python/terminal execution sandbox is disabled on the backend
   * (secrets are still stripped). Takes precedence over confirmToolCalls.
   * Kept in sync with permissionMode ("full" <=> true).
   */
  bypassPermissions: boolean;
  /**
   * Permission level. Single source of truth for the bypass dropdowns;
   * bypassPermissions and confirmToolCalls mirror it so legacy call sites
   * keep working. "full" is session-only (never persisted).
   */
  permissionMode: PermissionMode;
  /** Whether the "Enable Bypass Permissions?" warning dialog is open. Lifted out
   *  of the composer menu so confirming/cancelling it doesn't leave the menu frozen. */
  bypassConfirmOpen: boolean;
  /**
   * Per-chat set of tool names the user chose to auto-approve via "Always
   * allow". Keyed by UI confirmation scope, not necessarily the backend
   * sandbox session id. Not persisted across reloads.
   */
  alwaysAllowToolsBySession: Map<string, Set<string>>;
  /**
   * Tool calls currently paused awaiting the user's allow/deny decision,
   * keyed by the scoped frontend tool-call id. Each entry carries the backend
   * ``approvalId`` to echo back and the ``sessionId`` the generation runs
   * under, so the confirmation always resolves the exact pending call. The
   * ``autoAllowKey`` scopes the UI-only "Always allow" bucket per chat.
   * Only backend-gated local tool calls are added here.
   */
  toolConfirmations: Record<
    string,
    { approvalId: string; sessionId: string; autoAllowKey: string }
  >;
  /**
   * Fetch pill state, independent of `toolsEnabled` (Search). Only
   * consulted when `providerSupportsBuiltinWebFetch` is true.
   */
  webFetchToolsEnabled: boolean;
  /**
     * Live tool status per conversation ("Running Python: ...") with its start time. Keyed by
     * thread, or one chat's tool call shows above every other composer; the timestamp keeps the
     * counter running across a thread switch.
     */
  /**
     * Per-run entries, newest last. Unresolved threads share "__default", so one scalar per key
     * meant a finishing run's clear removed a sibling's status while its tool was still running.
     */
  toolStatusByThreadId: Record<string, ToolStatusEntry[]>;
  /** Live stdout/stderr from running tools, keyed by toolCallId. Transient:
   *  appended by tool_output, cleared on tool_end or run end. */
  toolLiveOutput: Record<string, string>;
  /** Full live output of finished tools whose result was truncated for the
   *  model, keyed by toolCallId. Set from tool_end; finished cards prefer it
   *  over the truncated result. Session-transient. */
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
  /**
   * Why speculative decoding was disabled despite being requested, or null.
   * Mirrors InferenceStatusResponse.spec_fallback_reason.
   */
  specFallbackReason: string | null;
  /** Projector recovery outcome for the active GGUF, or null. */
  mmprojFallbackReason: MmprojFallbackReason | null;
  /**
   * Which drafter the loaded model's speculative resolution was about: "mtp",
   * "dspark" or "dflash". Paired with specFallbackReason: the reason alone cannot name the
   * file to fix, since Auto resolves the kind server-side and the requested mode
   * still reads "auto".
   */
  specDrafterKind: string | null;
  /** User --spec-draft-n-max override (null = platform default). */
  specDraftNMax: number | null;
  loadedSpecDraftNMax: number | null;
  /** User --parallel slots override for GGUF loads (null = server default).
   *  Never re-seeded from an echo: the resolved count would pin a blank control. */
  nParallel: number | null;
  /** Slots the last successful load sent (null = default); the rollback
   *  re-sends it so a failed switch can't lose the override. */
  loadedNParallel: number | null;
  /** user --batch-size override for gguf loads (null = llama.cpp default 2048) */
  nBatch: number | null;
  /** batch size the last successful load sent (null = default) */
  loadedNBatch: number | null;
  /**
   * Pass-through llama-server arguments the resident model is running, as far as
   * this client knows (null = none, or never told). Kept so a rollback after a
   * failed switch can put the previous model back with them: by then the target
   * load has already replaced the backend's inheritance source, so omitting the
   * field on that rollback reads as a cross-model pickup and restores the model
   * without the arguments it was running.
   */
  loadedLlamaExtraArgs: string[] | null;
  /** user --ubatch-size override for gguf loads (null = llama.cpp default 512) */
  nUbatch: number | null;
  /** micro-batch size the last successful load sent (null = default) */
  loadedNUbatch: number | null;
  /** user --spec-draft-type-k/-v override, the DRAFT context's KV cache dtype
   *  (null = llama.cpp default f16). Separate from kvCacheDtype, which is the
   *  target model's. */
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
  /** What the RUNNING server was loaded with, as opposed to what the control now
   * shows: a pending per-model config is applied to disableVision before a switch
   * captures its rollback baseline, so only this survives to restore. */
  loadedDisableVision: boolean | null;
  /** Load a vision GGUF without its mmproj, freeing the projector's VRAM. */
  disableVision: boolean;
  /** Backend-reported: image input is off by request, not by absence of a
   *  projector. Null until first hydrated. */
  loadedVisionDisabledByUser: boolean | null;
  /** GPU memory strategy for GGUF loads. "auto" = Unsloth picks GPUs and context
   *  to fit; "manual" = you own the offload (gpuLayers < 0 = Auto/--fit, >= 0
   *  pins layers + nCpuMoe). */
  gpuMemoryMode: "auto" | "manual";
  /** Backend-reported gpu memory mode; null until first hydrated. */
  loadedGpuMemoryMode: "auto" | "manual" | null;
  /** The active model must use the staged CPU-only runtime when it is reloaded. */
  loadedCpuFallback: boolean;
  /** Manual mode: layers to offload to GPU. -1 = Auto (--fit); >= model layer
   *  count = all. */
  gpuLayers: number;
  loadedGpuLayers: number | null;
  /** Manual mode: MoE expert layers to keep on CPU (--n-cpu-moe); 0 = none. */
  nCpuMoe: number;
  loadedNCpuMoe: number | null;
  /** Manual mode: per-GPU layer counts (--tensor-split), in GPU-in-use order;
   *  null = unset (llama.cpp splits by free VRAM). */
  splitRatio: number[] | null;
  /** Backend-reported per-GPU split ratio (--tensor-split); null = unset. */
  loadedSplitRatio: number[] | null;
  /** Model layer count (GGUF block_count); the manual gpu-layers ceiling is
   * this + 1 (the output layer is offloadable too). */
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
  /** Persisted: expand every On Device GGUF repo's quantizations by default
   *  instead of waiting for a click. */
  expandQuantizations: boolean;
  /** Persisted: show non-downloaded quantizations too, not just downloaded. */
  showAllQuantizations: boolean;
  /** Persisted, off by default: chart each downloaded model's VRAM footprint
   *  under its row. Opt-in because the figures are estimates, and a row that
   *  cannot be sized is better left plain than annotated with a guess. */
  showMemoryBar: boolean;
  /** Persisted, shared by the chat model selector and the Hub page: list only
   *  models whose size fits this device's memory budget. */
  fitOnDeviceOnly: boolean;
  loadedIsMultimodal: boolean;
  /** Active model is a block-diffusion model (DiffusionGemma): drives the
   *  denoising-canvas artifact auto-render. */
  loadedIsDiffusion: boolean;
  /**
     * Live denoising frame per conversation ("__default" until the id exists). Transient: set per
     * step, cleared when the run ends, never persisted. Keyed, not global: two denoising chats
     * overwrote each other's frame, so the visible preview flickered or vanished.
     */
  activeDiffusionCanvasByThreadId: Record<string, DiffusionCanvasFrame>;
  customContextLength: number | null;
  /** The pinned context the loaded model used (null = Auto), so dirty-tracking
   *  and a later fit Apply can tell an explicit pin apart from Auto. */
  loadedCustomContextLength: number | null;
  defaultChatTemplate: string | null;
  chatTemplateOverride: string | null;
  loadedChatTemplateOverride: string | null;
  activeThreadId: string | null;
  activeThreadEpoch: number;
  queuedSettingsEpoch: number;
  activeProjectId: string | null;
  /**
   * Temporary / incognito chat toggle. When on, the active conversation
   * lives only in assistant-ui's in-memory repository and is never
   * persisted to studio.db -- so it stays out of history and vanishes on
   * reload. Deliberately ephemeral: NOT mirrored to localStorage or the
   * backend settings, so a refresh always exits incognito.
   */
  incognito: boolean;
  settingsPanelOpen: boolean;
  editingMessageId: string | null;
  pendingAudioBase64: string | null;
  pendingAudioName: string | null;
  pendingImageEditReference: PendingImageEditReference | null;
  contextUsage: ContextUsageSnapshot | null;
  /**
     * Per-thread copy of the above, so the bar survives a switch away and back. `contextUsage` is
     * the VISIBLE conversation's usage and a background run may not write it, so without this a
     * run finishing off-screen leaves nothing to restore.
     */
  contextUsageByThreadId: Record<string, ContextUsageSnapshot>;
  modelLoading: boolean;
  loadingModelPick: LoadingModelPick | null;
  // What the resident model loaded from, when that is not its id. A reload rebuilds its target
  // from the checkpoint, so without this it goes back down the ref the pin avoided.
  activeLoadId: string | null;
  activeNativePathToken: string | null;
  // Wall-clock expiry (ms) of the active native path token. The desktop host
  // prunes file leases after a TTL, so a reload checks this to prompt
  // re-selection instead of reusing a dead token.
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
      /** These params are the model's defaults, so its remembered settings are laid back
       * over them even though the checkpoint did not change. Being the model's and not
       * the user's, they also move the installation defaults, never the chat's. */
      fromModelDefaults?: boolean;
      /** The context the model just loaded with. */
      maxTokensCap?: number;
    },
  ) => void;
  setCustomPresets: (presets: Preset[]) => void;
  setActivePreset: (name: string) => void;
  setActivePresetSource: (source: ChatPresetSource) => void;
  setModels: (models: ChatModelRow[]) => void;
  setLoras: (loras: ChatLoraSummary[]) => void;
  /**
     * `local` defaults to true, so an unqualified caller still counts for the model-swap gate.
     * `owner` narrows the clear to the run that set the flag: unresolved thread ids share the
     * "__default" key, so a blind delete would drop a sibling's live entry. Owners accumulate,
     * so the flag survives until the last one clears.
     */
  setThreadRunning: (
    threadId: string,
    running: boolean,
    options?: { local?: boolean; owner?: () => void },
  ) => void;
  /**
     * Re-key a first turn's run handles once its thread is persisted.
     *
     * A run that starts before its id exists files everything under "__default". Nothing moved it
     * afterwards, so once the user navigated away the sidebar found no run and showed no spinner;
     * stopChatThread had no handle either and the generation carried on holding a slot.
     */
  adoptDefaultThreadRun: (threadId: string) => void;
  /**
     * Which key this run's handles live under now. `adoptDefaultThreadRun` re-keys them mid-run,
     * so a run that started under "__default" must look its owner up instead of reusing the key
     * it captured, or its writes and its final clear miss the entries.
     */
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
      /** False when the switch only puts back what was on screen before a
       * hidden load, so the model it steps off was never the user's. */
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
  /** Carry a choice made before the chat existed onto its new id. `claim` is
   * the value readPendingAttachmentTargetClaim gave when the choice was made;
   * a newer one means the entry now belongs to a different composer. */
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
  /**
     * `owner` is the run's identity token, as for `setThreadRunning`: unresolved threads share
     * "__default", so without it one run's cleanup clears a concurrent run's status.
     */
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
  /** Drop only `threadId`'s canvas: a run ending in a background chat must not wipe the
     * frame another chat is still painting. */
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

// Ids this browser holds a local answer for. Hydration keeps these and merges
// the rest, so a pre-hydration edit cannot drop other models.
const locallyRememberedModels = new Set<string>();
const inferenceParamMutationVersions = Object.fromEntries(
  PERSISTED_INFERENCE_PARAM_KEYS.map((key) => [key, 0]),
) as Record<PersistedInferenceParamKey, number>;
const scalarSettingMutationVersions = Object.fromEntries(
  SCALAR_SETTING_KEYS.map((key) => [key, 0]),
) as Record<ScalarSettingKey, number>;

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

/**
 * Seed the backend from this browser for mirrored settings it has never stored.
 * Without it an existing install would keep its preferences local until each
 * one is next changed.
 */
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

/** `bumpVersions` fences the moved keys against a hydration response in flight.
 * A model's own defaults do not get that: they must not outrank the settings the
 * user saved for it. */
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

/** Bump the hydration versions for the replayed keys and mirror them into the
 * global set, so a reload lands on this model's settings. */
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
    // Same reason as the setParams write: the in-memory copy is what a chat with
    // no snapshot falls back to, so it has to move with what was just stored.
    noteThreadScopedDefaults(changed);
  }
}

/** Same fence as the inference-param versions, recorded per model so only the
 * edited one is protected. Before hydration the params are placeholders, so
 * nothing is recorded. */
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

/** The map after a setParams call, falling back to the outgoing snapshot. A
 * non-persisting update (a background load) must not rewrite durable memory.
 *
 * Only an edit is recorded: defaults are not settings the model was used with,
 * and staged load params describe the model about to load. Both are picked up
 * by the snapshot taken when the model is left. */
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
      // Same filter as the outgoing snapshot: an edit to a key the memory keeps but the
      // chat does not (maxTokens, fastMode) still records the WHOLE snapshot, so the open
      // chat's sampling would ride into the model's entry and replay into the next chat.
      pickRememberedParams(withoutActiveThreadParams(state, nextParams)),
    ),
    nextParams.checkpoint,
  );
  return recorded ?? outgoing;
}

/** Snapshot the model being switched away from, so a model that was never edited
 * still keeps what it ran with. */
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
  // Only to seed a model with no entry: later changes are written key by key by
  // the edit that made them, and a full snapshot would put this browser's copy
  // of untouched keys over another tab's.
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
  // A full snapshot rewrites every field, so send one only when this browser has
  // something to say: an edit made here, or an entry that does not exist yet.
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
  // Only what moved: the server merges per key, so sending the rest would put
  // this browser's copy of every other key over another tab's.
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

/** Keys the user moved while the request was in flight: the same fence, read the
 * other way round. */
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

/** A model that took over from another while the request was in flight. Its
 * defaults lose to its own entry but outrank the global set, which belongs to
 * whichever model was used last. Not narrowed to the keys that moved. */
let modelLoadedBeforeHydration: string | null = null;

/** A model stepped off before hydration. Nothing can be filed for it yet, but the
 * global set the response will deliver is what it ran with, so this says which
 * model to file that under. */
let modelLeftBeforeHydration: string | null = null;

function noteModelDefaultsBeforeHydration(
  checkpoint: string,
  replacedAnotherModel: boolean,
): void {
  if (replacedAnotherModel) {
    modelLoadedBeforeHydration = checkpoint;
    return;
  }
  // The model already resident at startup is the one the saved global set
  // describes, so its defaults must not stand in front of it.
  if (modelLoadedBeforeHydration !== checkpoint) {
    modelLoadedBeforeHydration = null;
  }
}

/**
 * Whether the active checkpoint is an external model that cannot run deep
 * research, matching the composer's own rule. An unresolved provider is left
 * alone: the connection list may not have loaded, and refusing on that would
 * drop a Codex user's stored preference.
 */
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

/**
 * Kimi's $web_search builtin requires thinking disabled, so the composer keeps the two
 * pills mutually exclusive. A restore has to do the same, or a chat stored under another
 * provider sends a combination Kimi rejects.
 */
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
  const loadedBeforeHydration = modelLoadedBeforeHydration === checkpoint;
  modelLoadedBeforeHydration = null;
  // The toggle as it will read once this response lands, under the same fence
  // the scalar loop below applies.
  const remembersPerModel =
    settings.rememberParamsPerModel !== undefined &&
    scalarSettingMutationVersions.rememberParamsPerModel ===
      versions.scalarSettings.rememberParamsPerModel
      ? settings.rememberParamsPerModel
      : state.rememberParamsPerModel;
  // A model loaded mid-flight has no entry to restore its defaults from, so the
  // global set would overwrite them with the last model's. Only while the memory
  // is on: with it off that global set IS this model's settings.
  const keepModelDefaults =
    remembersPerModel &&
    loadedBeforeHydration &&
    settings.inferenceParamsByModel?.[checkpoint] === undefined;
  const params = { ...state.params };
  for (const key of PERSISTED_INFERENCE_PARAM_KEYS) {
    const value = settings.inferenceParams?.[key];
    // A slider moved before this response landed is held for the open chat, like a pill
    // clicked in the same window. The edit wins in the store, but it is the chat's, so
    // keep the server's value for the restore that runs when the pairing window closes:
    // without it the default falls back to this browser's pre-hydration copy and is
    // pinned onto the next snapshot-less chat. The edit already moved the version below.
    if (value !== undefined && isHeldThreadScopedField(key)) {
      hydratedDefaultsByHeldField.set(key, value);
      continue;
    }
    if (
      value !== undefined &&
      !keepModelDefaults &&
      // The context belongs to the load, not to the previous model's global set,
      // and no entry carries one for the replay below to put back.
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
      // Stored as written: a gap is a key this model never pinned, and there is
      // no honest value to invent. The replay lays the entry over what a load
      // just published, which is where a gap belongs.
      hydrated[modelId] = entry;
    }
    for (const modelId of locallyRememberedModels) {
      const local = state.paramsByModel[modelId];
      if (local) {
        hydrated[modelId] = local;
      }
    }
    // The entry arriving for this model predates the fenced edit, so lay the
    // edit over it or the next defaults update replays the stale one.
    if (checkpoint) {
      const edited = pickLocallyEditedParams(params, versions);
      if (hasKeys(edited)) {
        hydrated[checkpoint] = { ...hydrated[checkpoint], ...edited };
      }
    }
    nextState.paramsByModel = hydrated;
  } else if (checkpoint) {
    // No map in the response: an install upgraded from before this feature. With
    // no entry the next defaults update has nothing to replay and puts the
    // recommendation back over the fenced edit.
    const edited = pickLocallyEditedParams(params, versions);
    if (hasKeys(edited)) {
      nextState.paramsByModel = {
        ...state.paramsByModel,
        [checkpoint]: { ...state.paramsByModel[checkpoint], ...edited },
      };
    }
  }
  // A model stepped off before this response landed could not be filed then, and
  // the global set it ran with is only now known. Without this it has no entry,
  // so switching back inherits whatever replaced it.
  const left = modelLeftBeforeHydration;
  modelLeftBeforeHydration = null;
  const byModel = nextState.paramsByModel ?? state.paramsByModel;
  if (remembersPerModel && left && left !== checkpoint && !byModel[left]) {
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
  // Under the same fence as the scalar loop below, and for the same reason: a click
  // made while this response was out is the newer answer. Recording the stored value
  // over it would leave the switch visibly on -- the fence keeps the store field --
  // while the next model load quietly resolved to the value the user just replaced.
  if (
    settings.preserveThinking !== undefined &&
    scalarSettingMutationVersions.preserveThinking ===
      versions.scalarSettings.preserveThinking
  ) {
    notePreserveThinkingPreference(settings.preserveThinking);
  }
  for (const key of SCALAR_SETTING_KEYS) {
    const value = settings[key];
    // Full access is session-only, so a stored level must not silently drop the
    // sandbox bypass the user accepted a warning for, and Full access never
    // confirms a tool call whatever the saved toggle says.
    if (
      state.permissionMode === "full" &&
      (key === "permissionMode" || key === "confirmToolCalls")
    ) {
      continue;
    }
    // Both describe the running model through a loaded* shadow this loop cannot
    // set, so skip them while a shadow owns them. With none resident the store
    // field is what the next load reads and then persists back, so the server's
    // preference has to land here or the load overwrites it with a default.
    if (loadShadowOwnsMirroredSetting(key, state)) {
      continue;
    }
    // A load sets this from the model's own capability, and only a load sets
    // reasoningAlwaysOn, so a stored false would ask a model that cannot stop
    // thinking to stop thinking.
    if (
      key === "reasoningEnabled" &&
      value === false &&
      state.reasoningAlwaysOn
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
    // A click made before this response landed is held for the open chat rather than
    // written globally, so it advances no mutation version and the server's value would
    // silently replace it. The user is looking at their click; it wins.
    if (isHeldThreadScopedField(key)) {
      // The click wins in the STORE, but it is the chat's, not the installation's. When
      // the window closes the default has to go back to a value, and the one captured
      // before the window is this browser's pre-hydration copy. Keep the server's here
      // so the restore has the authoritative value to go back to.
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
  // The model already selected when this lands never crossed a checkpoint
  // transition, so nothing replayed its memory and it would run on the global
  // set, which belongs to whichever model was used last.
  const remembered = (nextState.paramsByModel ?? state.paramsByModel)[
    params.checkpoint
  ];
  if (
    (nextState.rememberParamsPerModel ?? state.rememberParamsPerModel) &&
    remembered
  ) {
    // Same fence as the global set above: a key the user moved mid-flight is
    // their edit. REMEMBERED, not PERSISTED, as in getReplayedParams: a
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
  // Outside the replay: an install with only a global set has no entry, and the
  // budget restored from it does not fit the load either.
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

export const useChatRuntimeStore = create<ChatRuntimeStore>((set, get) => ({
  settingsHydrated: false,
  threadScopedSettingsPending: false,
  // Hydrate the last external checkpoint so the external picker survives a
  // refresh. Local checkpoints are re-derived from the backend in
  // useChatModelRuntime and intentionally NOT persisted here.
  params: (() => {
    const persistedExternal = loadLastExternalCheckpoint();
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
  // Mirrors permissionMode (gate requested for ask/auto) so both controls
  // agree on load.
  confirmToolCalls:
    INITIAL_PERMISSION_MODE === "ask" || INITIAL_PERMISSION_MODE === "auto",
  // Never restore Bypass Permissions from storage: it disables the sandbox and
  // the confirmation gate, so it must be re-enabled (through the warning
  // dialog) each session rather than silently reactivating on reload.
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
        const { settings, fromServer } = await loadChatSettingsWithLegacyImport();
        let applied = false;
        set((state) => {
          if (state.settingsHydrated) {
            return state;
          }
          applied = true;
          const nextState: Partial<ChatRuntimeStore> = {
            settingsHydrated: true,
            ...getHydratedPresetState(
              settings,
              state,
              hydrationVersions.presets,
            ),
            ...getHydratedSettingsState(settings, state, hydrationVersions),
          };
          return nextState;
        });
        if (applied) {
          cacheHydratedSettings(settings, hydrationVersions);
          mirroredSettingsHydrated = true;
          // Only an authoritative read says a mirrored field is unset on the
          // server. A GET that fell back to legacy storage knows nothing about
          // it, and backfilling then pushes this browser's stale values over
          // whatever another browser wrote.
          if (fromServer) backfillMirroredSettings(settings);
          // After the backfill, so a startup edit wins over the stored value.
          flushPreHydrationSettings();
          // The previous session's tab-close writes, for the rows that did not exist
          // yet when it sent them. A replay of one that did land is refused by its own
          // seq, so this cannot revert anything newer.
          replayUnconfirmedThreadSettings();
        }
      } catch {
        // Hydrate failed: treat as hydrated-with-defaults so future setParams
        // calls reach saveSettingsPatch (which toasts on real network failure).
        warnSettingsPersistenceFailure();
        mirroredSettingsHydrated = true;
        flushPreHydrationSettings();
        // Independent of this endpoint: the tab-close snapshots waiting in storage are
        // rows' own settings, and leaving them unsent because /api/chat/settings was
        // briefly unavailable strands the last session's edit for the whole of this one.
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
  setParams: (params, options) =>
    set((state) => {
      // Mirror setCheckpoint: the local load path can mutate params.checkpoint
      // via setParams() before setCheckpoint runs, leaving stale per-turn
      // counters under the new checkpoint.
      const checkpointChanged = state.params.checkpoint !== params.checkpoint;
      const fromModelDefaults = options?.fromModelDefaults === true;
      // Remember what the outgoing model was running with before replacing it.
      const outgoing = checkpointChanged
        ? rememberOutgoingModel(state, state.params)
        : null;
      // An interactive local load lands here with the destination checkpoint and
      // the backend's recommended params, and only reaches setCheckpoint later,
      // once params.checkpoint already matches. Replay here or that switch, the
      // common one, never restores the model's own settings. fromModelDefaults
      // marks the updates that re-apply model defaults after a load or a status
      // poll: they overwrite remembered values, so memory goes back over them.
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
      // A chat outranks both the model's memory and its defaults, so the sampling it
      // pinned goes back on top of the replay. Live store only: persistence is decided
      // from nextParams below, so a pinning chat does not withhold the model's value.
      const effective = replayed
        ? restoreThreadScopedParams(nextParams)
        : nextParams;
      // A user edit fences the keys it moved against a hydration response still
      // in flight; only the HTTP write is gated on settingsHydrated.
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
      // A sampling key moved with a chat open belongs to that chat, so it reaches neither
      // the installation defaults nor this model's memory, both shared with every other
      // chat. What is left over is still the installation's.
      //
      // Capture is NOT gated on hydration, unlike the global write: the composer and the
      // settings sheet are live while the initial /api/chat/settings request is still out,
      // and this chat's pairing has already begun. Leaving such an edit uncaptured let the
      // arriving snapshot apply over it and let the pairing capture take it for an
      // installation default, pinning it onto the next snapshot-less chat.
      const sharedParams =
        options?.persist !== false
          ? withoutCapturedThreadEdits(changedParams, fromModelDefaults)
          : changedParams;
      // An edit belongs to the model the params now describe, so a call that moves
      // checkpoint and sliders at once files them under the destination.
      const paramsByModel = getParamsByModelAfterEdit(
        state,
        outgoing,
        nextParams,
        sharedParams,
        options?.persist !== false && !fromModelDefaults,
      );
      if (persistingGlobally) {
        // A switch replays the destination's entry over the params, so writing
        // it back says nothing new and, merged per key on the server, would put
        // this browser's copy over one another tab has since changed.
        persistParamEdit(
          sharedParams,
          checkpointChanged ? null : paramsByModel,
          nextParams.checkpoint,
        );
        // Level with what was just written, or a chat opened later in the same
        // session falls back to the sampling from before this model loaded.
        noteThreadScopedDefaults(sharedParams);
      } else if (fromModelDefaults && !state.settingsHydrated) {
        noteModelDefaultsBeforeHydration(
          nextParams.checkpoint,
          checkpointChanged && state.params.checkpoint !== "",
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
    }),
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
  setActivePresetSource: (activePresetSource) =>
    set(() => {
      activePresetSourceMutationVersion += 1;
      saveSettingsPatch({ activePresetSource });
      return { activePresetSource };
    }),
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
        // Any local owner keeps the key counted by the model-swap gate, so an external run
        // joining a shared key must not clear a sibling's flag.
        if (local) {
          nextLocal[threadId] = true;
        } else if (!owners.some((o) => o.local)) {
          delete nextLocal[threadId];
        }
      } else {
        const remaining = options?.owner
          ? owners.filter((o) => o.owner !== options.owner)
          : [];
        // An owner missing from the list was already cleared, or the key belongs to siblings
        // only: either way this run must change nothing.
        if (options?.owner && remaining.length === owners.length) return state;
        // An ownerless clear predates per-run tracking, so it must not speak for runs that
        // own the key: leave them to clear themselves.
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
      // Two first turns can share "__default", and nothing links a run there to the thread being
      // persisted. Moving the arrays wholesale handed this thread the sibling's owner and stop
      // handle too, so stopping one aborted both. Adopt only when the key holds a single run.
      if ((state.runOwnerByThreadId[key]?.length ?? 0) > 1) return state;
      // Only the transient run maps move. Anything already filed under the real id wins,
      // since that is a later, better-identified run.
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
  // `cancel` narrows removal to the run that registered it: unresolved thread ids share the
  // "__default" key, so a blind delete would drop a live sibling.
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
  setCheckpoint: (modelId, ggufVariant, options) =>
    set((state) => {
      // Persist external selections so they survive a refresh. Local ids are
      // NOT persisted -- they're re-derived from the backend on mount, and a
      // stale persisted local id would race the freshly-loaded model. See
      // LAST_EXTERNAL_CHECKPOINT_KEY notes.
      saveLastExternalCheckpoint(isExternalModelId(modelId) ? modelId : null);
      // Only disarm research for a connection that cannot drive it. Gating on the id
      // prefix alone silently switched it off for capable providers too, and saveBool
      // now reaches the backend, so that would write the preference off for every
      // browser on the install. Hoisted because all three writes below share it.
      const clampsDeepResearch =
        isExternalModelId(modelId) && !externalModelSupportsStudioTools(modelId);
      if (clampsDeepResearch) {
        saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
      }
      // Clear stale per-turn usage on model change; the relaxed external-provider
      // render gate would otherwise show old counters until the next completion.
      const checkpointChanged = state.params.checkpoint !== modelId;
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
      // Clamp maxTokens to the new model's cap when switching into an external
      // model so a value carried over from a local session doesn't exceed the
      // slider's max.
      let nextMaxTokens = baseParams.maxTokens;
      if (checkpointChanged && isExternalModelId(modelId)) {
        const parsed = parseExternalModelId(modelId);
        const provider = parsed
          ? useExternalProvidersStore
              .getState()
              .providers.find((p) => p.id === parsed.providerId)
          : null;
        // Only when the connection is known. A checkpoint restored before the
        // provider store hydrates would otherwise read the 32,768 fallback and lower
        // a value nothing puts back. No provider means unknown, not 32,768.
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
      // The chat outranks the model it switches to, so its pinned sampling and prompt go
      // back over the replay; an external switch has no load after it to do that. Live
      // store only: getReplayStatePatch still persists from the unrestored object, so the
      // model's own values reach the installation defaults.
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
        // Provenance and the spec-fallback reason both describe the model
        // being replaced, so they go together on a real change. Dropping only
        // one leaves the settings sheet pairing a stale reason with the wrong
        // recovery text. The load or status response reseeds both.
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
        // Switching to a connection whose provider cannot run Unsloth's tool
        // loop disables Deep Research; a capable one keeps the user's choice.
        ...(clampsDeepResearch ? { deepResearchEnabled: false } : {}),
      };
    }),
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
      // the pending write belongs to the outgoing thread, so it goes out before the swap.
      flushThreadScopedSettingsWrite();
      // edits made while this chat's snapshot was in flight: keep them and store them on the
      // chat. Anything else and the read would silently undo a click the user already saw.
      const heldFields = new Set<string>();
      if (threadId !== null && threadId === pendingPairingThreadId) {
        for (const edit of heldThreadScopedEdits) heldFields.add(edit.field);
        heldThreadScopedEdits = [];
        pendingPairingThreadId = null;
        // The window is over, so the next visit to this chat samples afresh rather than
        // reusing what the installation defaults were the last time round.
        pairingWindowDefaultsThreadId = null;
        // Its snapshot is now the one in the store, so anything waiting on it can go.
        closeThreadScopedPairingGate(threadId);
      } else if (
        // A drop to the defaults for the chat that is still open and still waiting on its
        // read keeps holding: those edits are that chat's, and releasing them here is the
        // leak this whole path exists to prevent.
        threadId !== null ||
        pendingPairingThreadId === null ||
        pendingPairingThreadId !== state.activeThreadId
      ) {
        releaseHeldThreadScopedEdits();
      }
      // Set from here rather than trusting the calls above: this updater's own return
      // value is merged last, so a `return state` would put the old flag back.
      const pending = pendingPairingThreadId !== null;
      // nothing was overridden while unpaired, so there is nothing to restore.
      if (threadScopedSettingsThreadId === null && threadId === null) {
        return state.threadScopedSettingsPending === pending
          ? state
          : { ...state, threadScopedSettingsPending: pending };
      }
      if (threadScopedSettingsThreadId === null) {
        // A held edit is already in the store but belongs to its chat, not to the
        // installation. Capturing it here would promote it to the default that every
        // snapshot-less chat follows, so take the value from before the window opened.
        // Deleting the key instead leaves it with no fallback at all, and the edited
        // value then stays live into the next chat, which is the same leak.
        const captured = readThreadScopedSettings(state) as Record<
          string,
          unknown
        >;
        const beforeWindow = (pairingWindowDefaults ??
          globalThreadScopedDefaults) as Record<string, unknown> | null;
        for (const field of heldFields) {
          // The server answered for this field while the window was open, and hydration
          // had to skip it. That value is the installation's; the pre-window copy is
          // only what this browser had cached before the answer arrived.
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
      // The constraint belongs to the chat it was applied in, and this chat's own
      // provider effects will say so again if it still holds here.
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
        // the user set this one while the read was in flight, so it wins over what came back.
        if (heldFields.has(key)) {
          applied[key] = readThreadScopedValue(state, key);
          continue;
        }
        // full access was accepted through a warning dialog: a switch must not drop it.
        // The chat is still pinned with the level underneath it, or a chat first opened
        // under full access would store no level at all and follow the installation one
        // forever after, which is the opposite of what pinning on open is for.
        if (key === "permissionMode" && state.permissionMode === "full") {
          const underneath =
            stored?.permissionMode ??
            globalThreadScopedDefaults?.permissionMode ??
            loadPermissionMode();
          if (underneath !== "full") applied[key] = underneath;
          continue;
        }
        // setCheckpoint clears deep research for external models in the store only, so a
        // stored true would come back and fail every send in that chat. openai_codex is the
        // exception the composer already makes, so use the same predicate rather than
        // refusing every external checkpoint.
        if (
          key === "deepResearchEnabled" &&
          (externalCheckpointRefusesDeepResearch(state.params.checkpoint) ||
            state.incognito)
        ) {
          continue;
        }
        // a key the snapshot omits falls back to the defaults, not to the outgoing chat's value.
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
      // Search and Thinking are mutually exclusive on Kimi, and the model-selection effect
      // that enforces it does not rerun on a thread switch. Restoring both, which a chat
      // stored under another provider can hold, would send an unsupported combination, so
      // the restore drops thinking exactly as clicking the Search pill does.
      if (
        isKimiCheckpoint(state.params.checkpoint) &&
        (applied.toolsEnabled ?? state.toolsEnabled) === true &&
        (applied.reasoningEnabled ?? state.reasoningEnabled) === true
      ) {
        applied.reasoningEnabled = false;
        if (state.reasoningEnabled !== false) target.reasoningEnabled = false;
      }
      // pin what the chat shows now, or changing the defaults later would rewrite its modes.
      // A chat that already had a snapshot only needs a write if it is carrying a held edit.
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
    // Mirror setCheckpoint's persistence: dropping the checkpoint must also
    // clear any stored external selection so the next refresh doesn't snap
    // back to a model the user intentionally cleared.
    saveLastExternalCheckpoint(null);
    saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
    return set((state) => ({
      queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      // An unload leaves the model the same way a switch does, so record what it
      // was running with.
      ...(() => {
        const outgoing = rememberOutgoingModel(state, state.params);
        return outgoing ? { paramsByModel: outgoing } : {};
      })(),
      params: {
        ...state.params,
        checkpoint: "",
      },
      activeGgufVariant: null,
      // Nothing is picked, so there is nothing for residency to describe. Back
      // to unknown rather than null: null would be read as "was evicted".
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
      // the level this chat carries, or the global one when it carries none.
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
      // The legacy toggle is a view over the permission level: on -> "ask",
      // off -> "off" (no prompts). While "full" is active the level is left
      // alone (the toggle is disabled in the UI anyway).
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
      // "full" is session-only (never persisted, see init); ask/auto/off
      // persist and keep the legacy confirm toggle in sync (the gate is
      // requested for both ask and auto).
      savePermissionMode(permissionMode);
      if (permissionMode === "full") {
        // Full access sends confirm_tool_calls=false; keep the store flag in
        // sync so response metadata does not report confirmations as enabled.
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
    // Deliberately not persisted (see init): a reload must not silently keep
    // the sandbox/confirmation bypass active without re-accepting the warning.
    // Turning bypass off returns to the last persisted ask/auto level.
    set((state) => {
      if (bypassPermissions) {
        // Full access never prompts; mirror confirm_tool_calls=false in the
        // store so metadata does not report confirmations as enabled.
        saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
        return {
          bypassPermissions,
          permissionMode: "full" as PermissionMode,
          confirmToolCalls: false,
          deepResearchEnabled: false,
          queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
        };
      }
      // back to the level this chat carries, or the global one when it carries none.
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
      // the only thread-scoped setting with no global slot, so no persist helper reaches it.
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
      // The entry under the shared key need not be the caller's: an abandoned
      // composer has had its own dropped and the next one's is sitting there,
      // which handing it over would consume.
      if (claim !== undefined && claim !== pendingAttachmentTargetClaim) {
        return state;
      }
      const pending =
        state.projectAttachmentTargetByThread[PENDING_CHAT_ATTACHMENT_KEY];
      // A chat that already made its own choice keeps it: the pending entry
      // belongs to a chat that does not exist yet.
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
      // Turning it on adopts the settings on screen for the active model.
      // Inside a chat the sampling on screen is that chat's and both writes below are
      // shared, so the outgoing snapshot's filter takes those keys back out first.
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
      // Turning it on is an explicit statement about the model on screen, so the
      // whole snapshot is what it means, not a key-by-key patch.
      if (paramsByModel && state.settingsHydrated && state.params.checkpoint) {
        saveSettingsPatch({
          inferenceParamsByModel: {
            [state.params.checkpoint]: paramsByModel[state.params.checkpoint],
          },
        });
      }
      // Turning it off makes the settings on screen the one shared set. The
      // global set can still be the last model's, so write it or the next launch
      // restores that instead.
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
        // Drop only this run's entry: a sibling behind the same key may still be running a tool,
        // and its status has to survive this clear.
        if (mine === undefined) return state;
        const rest = entries.filter((e) => e !== mine);
        if (rest.length > 0) {
          next[threadId] = rest;
        } else {
          delete next[threadId];
        }
      } else {
        // Same text from the same run means the same call, so keep startedAt: only a new tool restarts it.
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
  // Standing preference, but persisted only on a successful load (see
  // use-chat-model-runtime), not on selection -- so an unapplied pick the user
  // resets/abandons doesn't stick to the next session.
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
  // Write through to the visible thread's own entry, so a value restored by the history loader
  // survives a switch away and back: that loader runs once per mount and setActiveThreadId
  // reads the map, so without this the bar goes blank on return.
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
