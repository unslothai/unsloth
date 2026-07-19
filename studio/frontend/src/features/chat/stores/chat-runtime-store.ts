// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { RememberedLoadSettings } from "@/components/assistant-ui/model-selector/remembered-load-settings";
import {
  cancelStagedModelDownload,
  mirrorHfTokenInto,
  useHfTokenStore,
} from "@/features/hub";
import { toast } from "@/lib/toast";
import { create } from "zustand";
import { isExternalModelId, parseExternalModelId } from "../external-providers";
import {
  type ChatPresetSource,
  type Preset,
  getPresetSource,
} from "../presets/preset-policy";
import { getExternalMaxOutputTokens } from "../provider-capabilities";
import {
  type ChatLoraSummary,
  type ChatModelSummary,
  DEFAULT_INFERENCE_PARAMS,
  type InferenceParams,
} from "../types/runtime";
import {
  loadChatSettingsWithLegacyImport,
  savePersistedChatSettingsPatch,
} from "../utils/chat-settings-storage";
import { useExternalProvidersStore } from "./external-providers-store";
import { PLUS_MENU_PINS_STORAGE_KEY } from "./plus-menu-prefs-store";

export const CHAT_REASONING_ENABLED_KEY = "unsloth_chat_reasoning_enabled";
export const CHAT_TOOLS_ENABLED_KEY = "unsloth_chat_tools_enabled";
export const CHAT_CODE_TOOLS_ENABLED_KEY = "unsloth_chat_code_tools_enabled";
export const CHAT_IMAGE_TOOLS_ENABLED_KEY = "unsloth_chat_image_tools_enabled";
export const CHAT_ARTIFACTS_ENABLED_KEY = "unsloth_chat_artifacts_enabled";
export const CHAT_SHOW_CANVAS_MENU_ITEM_KEY =
  "unsloth_chat_show_canvas_menu_item";
export const CHAT_COLLAPSE_HTML_ARTIFACTS_KEY =
  "unsloth_chat_collapse_html_artifacts";
export const CHAT_ALLOW_ARTIFACT_NETWORK_ACCESS_KEY =
  "unsloth_chat_allow_artifact_network_access";
export const CHAT_MCP_ENABLED_KEY = "unsloth_chat_mcp_enabled";
export const CHAT_CONFIRM_TOOL_CALLS_KEY = "unsloth_chat_confirm_tool_calls";
export const CHAT_LOAD_ON_SELECTION_KEY = "unsloth_chat_load_on_selection";
export const CHAT_EXPAND_QUANTIZATIONS_KEY =
  "unsloth_chat_expand_quantizations";
export const CHAT_SHOW_ALL_QUANTIZATIONS_KEY =
  "unsloth_chat_show_all_quantizations";
export const MODELS_FIT_ON_DEVICE_ONLY_KEY =
  "unsloth_models_fit_on_device_only";
export const CHAT_BYPASS_PERMISSIONS_KEY = "unsloth_chat_bypass_permissions";
export const CHAT_PERMISSION_MODE_KEY = "unsloth_chat_permission_mode";

/**
 * Permission level for local tool calls:
 * - "ask": always ask before every tool call runs.
 * - "auto" ("Approve for me"): only ask for calls the backend detects as
 *   potentially unsafe; read-only calls run immediately. Sandbox stays on.
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
export const CHAT_SPECULATIVE_TYPE_KEY = "unsloth_chat_speculative_type";

// Persist only the model-agnostic intents (auto/ngram/off). MTP modes
// (mtp/mtp+ngram) and spec_draft_n_max stay session-only: a persisted MTP
// choice would silently no-op on models without an MTP head. Unknown -> auto.
const PERSISTED_SPEC_MODES = new Set(["auto", "ngram", "off"]);

export type RagSource = { type: "thread" } | { type: "kb"; kbId: string };

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
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(CHAT_RAG_SOURCE_KEY, JSON.stringify(value));
  } catch {
    // Ignore storage failures; the default RAG source still works for this session.
  }
}

function loadRagMode(): RagMode {
  const raw = loadString(CHAT_RAG_MODE_KEY, DEFAULT_RAG_MODE);
  return raw === "lexical" || raw === "dense" ? raw : "hybrid";
}

function loadRagAutoInject(): RagAutoInject {
  const raw = loadString(CHAT_RAG_AUTOINJECT_KEY, DEFAULT_RAG_AUTOINJECT);
  if (raw === "auto" || raw === "on" || raw === "off") return raw;
  // Legacy boolean migration: false -> Off, else Auto.
  return raw === "false" ? "off" : "auto";
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

function mergePatch(into: SettingsPatch, more: SettingsPatch): void {
  for (const [key, value] of Object.entries(more)) {
    const intoAny = into as Record<string, unknown>;
    const prev = intoAny[key];
    if (isPlainObject(prev) && isPlainObject(value)) {
      intoAny[key] = { ...prev, ...value };
    } else {
      intoAny[key] = value;
    }
  }
}

async function flushSettingsPatch(keepalive = false): Promise<void> {
  if (Object.keys(pendingPatch).length === 0) return;
  const patch = pendingPatch;
  pendingPatch = {};
  try {
    await savePersistedChatSettingsPatch(patch, { keepalive });
  } catch {
    const retryPatch: SettingsPatch = {};
    mergePatch(retryPatch, patch);
    mergePatch(retryPatch, pendingPatch);
    pendingPatch = retryPatch;
    warnSettingsPersistenceFailure();
  }
}

function saveSettingsPatch(patch: SettingsPatch): void {
  mergePatch(pendingPatch, patch);
  if (pendingTimer !== null) clearTimeout(pendingTimer);
  pendingTimer = setTimeout(() => {
    pendingTimer = null;
    inflightFlush = inflightFlush
      .catch(() => undefined)
      .then(() => flushSettingsPatch());
  }, SETTINGS_DEBOUNCE_MS);
}

// Best-effort flush of any pending patch on tab close. keepalive lets the PUT
// outlive the unload; without it the browser cancels the fetch and the user's
// last slider drag is dropped.
if (typeof window !== "undefined") {
  window.addEventListener("beforeunload", () => {
    if (pendingTimer !== null) clearTimeout(pendingTimer);
    if (Object.keys(pendingPatch).length === 0) return;
    inflightFlush = inflightFlush
      .catch(() => undefined)
      .then(() => flushSettingsPatch(true));
  });
}

function canUseStorage(): boolean {
  return typeof window !== "undefined";
}

function loadBool(key: string, fallback: boolean): boolean {
  const raw = loadOptionalBool(key);
  return raw ?? fallback;
}

export function loadOptionalBool(key: string): boolean | null {
  if (!canUseStorage()) return null;
  try {
    const raw = localStorage.getItem(key);
    if (raw === null) return null;
    return raw === "true";
  } catch {
    return null;
  }
}

/**
 * Resolve the web-search / code-execution pill state to apply when a model
 * loads. Honors the user's persisted preference so a tool-capable model never
 * re-enables a pill the user turned off, and never re-disables one they turned
 * on. When no preference has been expressed the pills stay off: tool execution
 * is opt-in, so the person enables it with a click rather than a tool-capable
 * model turning it on for them.
 */
export function resolveToolsEnabledOnLoad(supportsTools: boolean): {
  toolsEnabled: boolean;
  codeToolsEnabled: boolean;
} {
  if (!supportsTools) return { toolsEnabled: false, codeToolsEnabled: false };
  return {
    toolsEnabled: loadOptionalBool(CHAT_TOOLS_ENABLED_KEY) ?? false,
    codeToolsEnabled: loadOptionalBool(CHAT_CODE_TOOLS_ENABLED_KEY) ?? false,
  };
}

function saveBool(key: string, value: boolean): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(key, value ? "true" : "false");
  } catch {
    // ignore
  }
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
  if (!canUseStorage()) return "auto";
  try {
    const raw = localStorage.getItem(CHAT_PERMISSION_MODE_KEY);
    if (raw === "ask" || raw === "auto" || raw === "off") return raw;
  } catch {
    // ignore
  }
  const legacyConfirm = loadOptionalBool(CHAT_CONFIRM_TOOL_CALLS_KEY);
  if (legacyConfirm === null) return "auto";
  return legacyConfirm ? "ask" : "off";
}

function savePermissionMode(mode: PermissionMode): void {
  if (!canUseStorage() || mode === "full") return;
  try {
    localStorage.setItem(CHAT_PERMISSION_MODE_KEY, mode);
  } catch {
    // ignore
  }
}

const INITIAL_PERMISSION_MODE: PermissionMode = loadPermissionMode();

function loadString(key: string, fallback: string): string {
  if (!canUseStorage()) return fallback;
  try {
    return localStorage.getItem(key) ?? fallback;
  } catch {
    return fallback;
  }
}

function saveString(key: string, value: string): void {
  if (!canUseStorage()) return;
  try {
    localStorage.setItem(key, value);
  } catch {
    // ignore
  }
}

// Canonicalises any backend value onto the Speculative Decoding dropdown's
// modes ("auto"/"mtp"/"ngram"/"mtp+ngram"/"off"/null). Backend-only
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
    saveString(CHAT_SPECULATIVE_TYPE_KEY, value);
  }
}

/** A local model staged for a deferred load (see `pendingSelection`). Shape is
 *  a subset of the load hook's `SelectedModelInput`, structurally assignable. */
export type PendingModelSelection = {
  id: string;
  isLora?: boolean;
  ggufVariant?: string;
  isDownloaded?: boolean;
  expectedBytes?: number;
  /** Native (drag-drop / picked-from-disk) GGUF: the path token used to read
   *  the header and to load. Absent for HF-repo models. */
  nativePathToken?: string;
  /** Expiry for the native path token. Native tokens are short-lived grants,
   *  not durable loaded-model identities. */
  nativePathTokenExpiresAtMs?: number | null;
  /** Direct local .gguf file (custom folder / LM Studio): a GGUF source even
   *  though it carries neither an HF variant nor a native path token. */
  isGguf?: boolean;
  /** Native context length read from the GGUF header once the file is local.
   *  Scoped here (not the shared `ggufContextLength`) so a staged model's
   *  metadata never pollutes the currently-loaded model's context display. */
  contextLength?: number | null;
  /** "Load on selection" on + un-cached GGUF: download via the manager (global
   *  indicator) without opening the sheet, then load once the download finishes. */
  autoLoad?: boolean;
  /** Uncached non-GGUF HF repo: download the full snapshot via the manager
   *  (variant null) the same way GGUF picks download a variant. */
  isHubRepo?: boolean;
};

/** A pick is a GGUF (HF variant, native file, or a direct local .gguf) and so
 *  has pre-load options worth staging. Works on a selection or a staged pick. */
export function hasGgufSource(x: {
  ggufVariant?: string;
  nativePathToken?: string;
  isGguf?: boolean;
}): boolean {
  return (
    x.ggufVariant != null || x.nativePathToken != null || x.isGguf === true
  );
}

export function isNativePathTokenExpired(
  expiresAtMs?: number | null,
  now = Date.now(),
): boolean {
  return (
    typeof expiresAtMs === "number" &&
    Number.isFinite(expiresAtMs) &&
    expiresAtMs <= now
  );
}

export function hasUsableNativePathToken(x: {
  nativePathToken?: string | null;
  nativePathTokenExpiresAtMs?: number | null;
}): boolean {
  return (
    typeof x.nativePathToken === "string" &&
    x.nativePathToken.length > 0 &&
    !isNativePathTokenExpired(x.nativePathTokenExpiresAtMs)
  );
}

export function hasLoadedGgufSource(x: {
  activeGgufVariant?: string | null;
  activeNativePathToken?: string | null;
  activeNativePathTokenExpiresAtMs?: number | null;
  ggufContextLength?: number | null;
  params: { checkpoint: string };
}): boolean {
  return (
    x.activeGgufVariant != null ||
    hasUsableNativePathToken({
      nativePathToken: x.activeNativePathToken,
      nativePathTokenExpiresAtMs: x.activeNativePathTokenExpiresAtMs,
    }) ||
    x.activeNativePathTokenExpiresAtMs != null ||
    (isLocalModelPath(x.params.checkpoint) &&
      x.params.checkpoint.toLowerCase().endsWith(".gguf")) ||
    x.ggufContextLength != null
  );
}

/** A local-disk model id: Unix absolute (/), relative (./ ../), tilde (~/),
 *  Windows drive (C:\) or UNC (\\server). Shared so the loader and the
 *  hub-repo predicate classify ids identically. */
export function isLocalModelPath(id: string): boolean {
  return /^(\/|\.{1,2}[\\/]|~[\\/]|[A-Za-z]:[\\/]|\\\\)/.test(id);
}

function clearedLoadedGgufMetadata() {
  return {
    activeNativePathToken: null,
    activeNativePathTokenExpiresAtMs: null,
    ggufContextLength: null,
    ggufMaxContextLength: null,
    ggufNativeContextLength: null,
  };
}

/** An uncached HF hub repo we can download as a full snapshot (non-GGUF
 *  safetensors / MLX). Excludes GGUF sources, local paths, native files, LoRA,
 *  and external provider models so none are mis-routed into a snapshot. */
export function isDownloadableHubRepo(x: {
  id: string;
  source?: string;
  isLora?: boolean;
  ggufVariant?: string;
  nativePathToken?: string;
  isGguf?: boolean;
}): boolean {
  return (
    x.source === "hub" &&
    !hasGgufSource(x) &&
    x.isLora !== true &&
    x.nativePathToken == null &&
    !isLocalModelPath(x.id)
  );
}

export function isPendingGguf(pending: PendingModelSelection | null): boolean {
  return pending != null && hasGgufSource(pending);
}

/** Whether `pending` refers to the same model as `pick` (id + GGUF variant +
 *  native path token, optionals null-normalized). Native ids are display labels
 *  that can collide, so the token must match too — id alone can land on the
 *  wrong file. */
export function pendingSelectionMatches(
  pending: PendingModelSelection | null,
  pick: {
    id: string;
    ggufVariant?: string | null;
    nativePathToken?: string | null;
  },
): boolean {
  return (
    pending != null &&
    pending.id === pick.id &&
    (pending.ggufVariant ?? null) === (pick.ggufVariant ?? null) &&
    (pending.nativePathToken ?? null) === (pick.nativePathToken ?? null)
  );
}

type ChatRuntimeStore = {
  settingsHydrated: boolean;
  params: InferenceParams;
  customPresets: Preset[];
  activePreset: string;
  activePresetSource: ChatPresetSource;
  models: ChatModelSummary[];
  loras: ChatLoraSummary[];
  runningByThreadId: Record<string, boolean>;
  cancelByThreadId: Record<string, () => void>;
  autoTitle: boolean;
  hfToken: string;
  modelsError: string | null;
  // Set only when a LOAD fails (not refresh/list/unload, which use modelsError);
  // lets the attach gates flag a failed load vs "no model picked".
  lastModelLoadError: string | null;
  activeGgufVariant: string | null;
  ggufContextLength: number | null;
  ggufMaxContextLength: number | null;
  ggufNativeContextLength: number | null;
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
  artifactsEnabled: boolean;
  // Whether the Canvas toggle is offered in the composer + menu (hidden by default).
  showCanvasMenuItem: boolean;
  collapseHtmlArtifacts: boolean;
  allowArtifactNetworkAccess: boolean;
  mcpEnabledForChat: boolean;
  ragEnabled: boolean;
  ragSource: RagSource;
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
  toolStatus: string | null;
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
  maxToolCallsPerMessage: number;
  toolCallTimeout: number;
  kvCacheDtype: string | null;
  loadedKvCacheDtype: string | null;
  speculativeType: string | null;
  loadedSpeculativeType: string | null;
  /**
   * Why MTP was disabled on the loaded model despite being requested, or null.
   * Mirrors InferenceStatusResponse.spec_fallback_reason.
   */
  specFallbackReason: string | null;
  /** User --spec-draft-n-max override (null = platform default). */
  specDraftNMax: number | null;
  loadedSpecDraftNMax: number | null;
  /** Tensor-parallel split (--split-mode tensor) toggle, GGUF multi-GPU only. */
  tensorParallel: boolean;
  /** Backend-reported tensor-parallel state; null until first hydrated. */
  loadedTensorParallel: boolean | null;
  /** Persisted: when false, picking a local model stages it as
   *  `pendingSelection` (and opens settings) instead of loading immediately,
   *  so load settings can be set before the single load. */
  loadOnSelection: boolean;
  /** Persisted: expand every On Device GGUF repo's quantizations by default
   *  instead of waiting for a click. */
  expandQuantizations: boolean;
  /** Persisted: show non-downloaded quantizations too, not just downloaded. */
  showAllQuantizations: boolean;
  /** Persisted, shared by the chat model selector and the Hub page: list only
   *  models whose size fits this device's memory budget. */
  fitOnDeviceOnly: boolean;
  /** A local model picked while `loadOnSelection` is off: staged, not loaded.
   *  The settings sheet shows its load knobs and a Load button. */
  pendingSelection: PendingModelSelection | null;
  loadedIsMultimodal: boolean;
  /** Active model is a block-diffusion model (DiffusionGemma): drives the
   *  denoising-canvas artifact auto-render. */
  loadedIsDiffusion: boolean;
  /** Live denoising frame for the in-progress diffusion message. Transient: set
   *  per step, cleared when the run ends, never persisted into the transcript. */
  activeDiffusionCanvas: DiffusionCanvasFrame | null;
  customContextLength: number | null;
  defaultChatTemplate: string | null;
  chatTemplateOverride: string | null;
  loadedChatTemplateOverride: string | null;
  activeThreadId: string | null;
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
  contextUsage: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    cachedTokens: number;
    // Anthropic-only; optional so pre-cache-stats persisted entries load.
    cacheWriteTokens?: number;
  } | null;
  modelLoading: boolean;
  activeNativePathToken: string | null;
  activeNativePathTokenExpiresAtMs: number | null;
  hydratePersistedSettings: () => Promise<void>;
  setModelLoading: (loading: boolean) => void;
  setModelRequiresTrustRemoteCode: (required: boolean) => void;
  setParams: (params: InferenceParams) => void;
  setCustomPresets: (presets: Preset[]) => void;
  setActivePreset: (name: string) => void;
  setActivePresetSource: (source: ChatPresetSource) => void;
  setModels: (models: ChatModelSummary[]) => void;
  setLoras: (loras: ChatLoraSummary[]) => void;
  setThreadRunning: (threadId: string, running: boolean) => void;
  registerThreadCancel: (threadId: string, cancel: () => void) => void;
  clearThreadCancel: (threadId: string) => void;
  setAutoTitle: (enabled: boolean) => void;
  setHfToken: (token: string) => void;
  setModelsError: (error: string | null) => void;
  setLastModelLoadError: (error: string | null) => void;
  setCheckpoint: (modelId: string, ggufVariant?: string | null) => void;
  setActiveThreadId: (threadId: string | null) => void;
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
  setArtifactsEnabled: (
    enabled: boolean,
    options?: { persist?: boolean },
  ) => void;
  setShowCanvasMenuItem: (enabled: boolean) => void;
  setCollapseHtmlArtifacts: (enabled: boolean) => void;
  setAllowArtifactNetworkAccess: (enabled: boolean) => void;
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
  setRagSource: (source: RagSource) => void;
  setRagMode: (mode: RagMode) => void;
  setRagTopK: (topK: number) => void;
  setRagAutoInject: (value: RagAutoInject) => void;
  setRagAutoInjectMinScore: (score: number) => void;
  setRagOcrScanned: (enabled: boolean) => void;
  setRagCaptionFigures: (enabled: boolean) => void;
  setToolStatus: (status: string | null) => void;
  appendToolLiveOutput: (toolCallId: string, text: string) => void;
  /** Clear one tool's live output, or all when no id is given. */
  clearToolLiveOutput: (toolCallId?: string) => void;
  /** Preserve a finished tool's full live-streamed output for display. */
  setToolFullOutput: (toolCallId: string, text: string) => void;
  /** Drop a stale preserved full output (a new run is reusing the id). */
  clearToolFullOutput: (toolCallId: string) => void;
  setGeneratingStatus: (status: string | null) => void;
  setActiveDiffusionCanvas: (canvas: DiffusionCanvasFrame | null) => void;
  setAutoHealToolCalls: (enabled: boolean) => void;
  setNudgeToolCalls: (enabled: boolean) => void;
  setMaxToolCallsPerMessage: (value: number) => void;
  setToolCallTimeout: (value: number) => void;
  setKvCacheDtype: (dtype: string | null) => void;
  setSpeculativeType: (type: string | null) => void;
  setSpecDraftNMax: (value: number | null) => void;
  /** Revert the editable load knobs to the loaded model's baseline (or defaults
   *  when nothing is loaded). Used by the settings-sheet Reset button and to
   *  start each deferred-staging session clean so one staged pick's settings
   *  don't leak onto the next. */
  resetModelSettingsToLoaded: () => void;
  /** Seed the editable load knobs from a model's remembered settings. Shared by
   *  the settings sheet's restore effect and the "Load on selection" paths,
   *  which skip the sheet but must still honor a saved config. */
  applyRememberedLoadSettings: (settings: RememberedLoadSettings) => void;
  setTensorParallel: (value: boolean) => void;
  setLoadOnSelection: (value: boolean) => void;
  setExpandQuantizations: (value: boolean) => void;
  setShowAllQuantizations: (value: boolean) => void;
  setFitOnDeviceOnly: (value: boolean) => void;
  setPendingSelection: (selection: PendingModelSelection | null) => void;
  /** Stage a pick for a deferred load: revert knobs to the loaded baseline,
   *  record the selection, and open the settings sheet. */
  stageModel: (selection: PendingModelSelection) => void;
  /** Abandon a staged pick without loading: revert knobs to the loaded baseline
   *  and clear the pending selection. Cancels its in-flight download too, unless
   *  `keepDownload` is set (navigation keeps the transfer running, like Hub). */
  abandonStagedModel: (opts?: { keepDownload?: boolean }) => void;
  setCustomContextLength: (v: number | null) => void;
  setChatTemplateOverride: (template: string | null) => void;
  setPendingAudio: (base64: string, name: string) => void;
  clearPendingAudio: () => void;
  setPendingImageEditReference: (
    reference: PendingImageEditReference | null,
  ) => void;
  clearPendingImageEditReference: () => void;
  setContextUsage: (usage: ChatRuntimeStore["contextUsage"]) => void;
};

type PersistedChatSettings = Awaited<
  ReturnType<typeof loadChatSettingsWithLegacyImport>
>;
type PersistedInferenceParams = NonNullable<
  PersistedChatSettings["inferenceParams"]
>;
type PersistedInferenceParamKey = keyof PersistedInferenceParams;
type ScalarSettingKey =
  | "autoTitle"
  | "reasoningEffort"
  | "preserveThinking"
  | "collapseHtmlArtifacts"
  | "allowArtifactNetworkAccess"
  | "autoHealToolCalls"
  | "nudgeToolCalls"
  | "maxToolCallsPerMessage"
  | "toolCallTimeout";

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

const PERSISTED_INFERENCE_PARAM_KEYS = [
  "temperature",
  "topP",
  "topK",
  "minP",
  "repetitionPenalty",
  "presencePenalty",
  "maxSeqLength",
  "maxTokens",
  "systemPrompt",
  "systemVariables",
  "trustRemoteCode",
  "fastMode",
] as const satisfies readonly PersistedInferenceParamKey[];

const SCALAR_SETTING_KEYS = [
  "autoTitle",
  "reasoningEffort",
  "preserveThinking",
  "collapseHtmlArtifacts",
  "allowArtifactNetworkAccess",
  "autoHealToolCalls",
  "nudgeToolCalls",
  "maxToolCallsPerMessage",
  "toolCallTimeout",
] as const satisfies readonly ScalarSettingKey[];

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

function setInferenceParam(
  params: InferenceParams,
  key: PersistedInferenceParamKey,
  value: PersistedInferenceParams[PersistedInferenceParamKey],
): void {
  (params as Record<PersistedInferenceParamKey, unknown>)[key] = value;
}

function getChangedInferenceParams(
  nextParams: InferenceParams,
  currentParams: InferenceParams,
): PersistedInferenceParams {
  const changedParams: PersistedInferenceParams = {};
  for (const key of PERSISTED_INFERENCE_PARAM_KEYS) {
    const nextValue = nextParams[key];
    if (Object.is(nextValue, currentParams[key])) {
      continue;
    }
    inferenceParamMutationVersions[key] += 1;
    if (nextValue !== undefined) {
      setInferenceParam(changedParams as InferenceParams, key, nextValue);
    }
  }
  return changedParams;
}

function getHydratedCustomPresets(
  settings: PersistedChatSettings,
  state: ChatRuntimeStore,
): Preset[] {
  return (
    settings.customPresets?.map((preset) => ({
      name: preset.name,
      params: {
        ...DEFAULT_INFERENCE_PARAMS,
        ...preset.params,
      },
    })) ?? state.customPresets
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

function getHydratedSettingsState(
  settings: PersistedChatSettings,
  state: ChatRuntimeStore,
  versions: SettingsHydrationVersions,
): Partial<ChatRuntimeStore> {
  const nextState: Partial<ChatRuntimeStore> = {};
  const params = { ...state.params };
  for (const key of PERSISTED_INFERENCE_PARAM_KEYS) {
    const value = settings.inferenceParams?.[key];
    if (
      value !== undefined &&
      inferenceParamMutationVersions[key] === versions.inferenceParams[key]
    ) {
      setInferenceParam(params, key, value);
    }
  }
  nextState.params = params;
  for (const key of SCALAR_SETTING_KEYS) {
    const value = settings[key];
    if (
      value !== undefined &&
      scalarSettingMutationVersions[key] === versions.scalarSettings[key]
    ) {
      (nextState as Record<ScalarSettingKey, unknown>)[key] = value;
    }
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
  scalarSettingMutationVersions[key] += 1;
  saveSettingsPatch({ [key]: value });
}

/** The "revert to the loaded model" baseline for the editable load knobs.
 *  Shared by resetModelSettingsToLoaded (full revert) and stageModel (which
 *  overrides speculative to start a fresh pick from the standing default). */
function loadedBaselineSettings(s: ChatRuntimeStore) {
  const hasLoadedModel = Boolean(s.params.checkpoint);
  return {
    customContextLength: null,
    kvCacheDtype: s.loadedKvCacheDtype,
    tensorParallel: s.loadedTensorParallel ?? false,
    speculativeType: hasLoadedModel
      ? s.loadedSpeculativeType
      : readPersistedSpeculativeType(),
    specDraftNMax: hasLoadedModel ? s.loadedSpecDraftNMax : null,
    chatTemplateOverride: s.loadedChatTemplateOverride,
  };
}

export const useChatRuntimeStore = create<ChatRuntimeStore>((set, get) => ({
  settingsHydrated: false,
  // Hydrate the last external checkpoint so the external picker survives a
  // refresh. Local checkpoints are re-derived from the backend in
  // useChatModelRuntime and intentionally NOT persisted here.
  params: (() => {
    const persistedExternal = loadLastExternalCheckpoint();
    return persistedExternal
      ? { ...DEFAULT_INFERENCE_PARAMS, checkpoint: persistedExternal }
      : DEFAULT_INFERENCE_PARAMS;
  })(),
  customPresets: [],
  activePreset: "Default",
  activePresetSource: getPresetSource("Default"),
  models: [],
  loras: [],
  runningByThreadId: {},
  cancelByThreadId: {},
  autoTitle: false,
  hfToken: useHfTokenStore.getState().token,
  modelsError: null,
  lastModelLoadError: null,
  activeGgufVariant: null,
  ggufContextLength: null,
  ggufMaxContextLength: null,
  ggufNativeContextLength: null,
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
  artifactsEnabled: loadBool(CHAT_ARTIFACTS_ENABLED_KEY, false),
  showCanvasMenuItem: loadShowCanvasMenuItem(),
  collapseHtmlArtifacts: loadBool(CHAT_COLLAPSE_HTML_ARTIFACTS_KEY, false),
  allowArtifactNetworkAccess: loadBool(
    CHAT_ALLOW_ARTIFACT_NETWORK_ACCESS_KEY,
    false,
  ),
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
  toolStatus: null,
  toolLiveOutput: {},
  toolFullOutput: {},
  generatingStatus: null,
  activeDiffusionCanvas: null,
  autoHealToolCalls: true,
  nudgeToolCalls: true,
  maxToolCallsPerMessage: 25,
  toolCallTimeout: 5,
  kvCacheDtype: null,
  loadedKvCacheDtype: null,
  speculativeType: readPersistedSpeculativeType(),
  loadedSpeculativeType: null,
  specFallbackReason: null,
  specDraftNMax: null,
  loadedSpecDraftNMax: null,
  tensorParallel: false,
  loadedTensorParallel: null,
  loadOnSelection: loadBool(CHAT_LOAD_ON_SELECTION_KEY, true),
  expandQuantizations: loadBool(CHAT_EXPAND_QUANTIZATIONS_KEY, false),
  showAllQuantizations: loadBool(CHAT_SHOW_ALL_QUANTIZATIONS_KEY, true),
  fitOnDeviceOnly: loadBool(MODELS_FIT_ON_DEVICE_ONLY_KEY, false),
  pendingSelection: null,
  loadedIsMultimodal: false,
  loadedIsDiffusion: false,
  customContextLength: null,
  defaultChatTemplate: null,
  chatTemplateOverride: null,
  loadedChatTemplateOverride: null,
  activeThreadId: null,
  activeProjectId: null,
  incognito: false,
  settingsPanelOpen: false,
  editingMessageId: null,
  pendingAudioBase64: null,
  pendingAudioName: null,
  pendingImageEditReference: null,
  contextUsage: null,
  modelLoading: false,
  activeNativePathToken: null,
  activeNativePathTokenExpiresAtMs: null,
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
        const settings = await loadChatSettingsWithLegacyImport();
        set((state) => {
          if (state.settingsHydrated) {
            return state;
          }
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
      } catch {
        // Hydrate failed: treat as hydrated-with-defaults so future setParams
        // calls reach saveSettingsPatch (which toasts on real network failure).
        warnSettingsPersistenceFailure();
        set({ settingsHydrated: true });
      } finally {
        settingsHydrationPromise = null;
      }
    })();
    return settingsHydrationPromise;
  },
  setModelLoading: (loading) => set({ modelLoading: loading }),
  setModelRequiresTrustRemoteCode: (modelRequiresTrustRemoteCode) =>
    set({ modelRequiresTrustRemoteCode }),
  setParams: (params) =>
    set((state) => {
      // Bump version unconditionally so a late hydration response won't clobber
      // a pre-hydrate user edit; only the HTTP write is gated on settingsHydrated.
      const changedParams = getChangedInferenceParams(params, state.params);
      if (state.settingsHydrated && hasKeys(changedParams)) {
        saveSettingsPatch({ inferenceParams: changedParams });
      }
      // Mirror setCheckpoint: the local load path can mutate params.checkpoint
      // via setParams() before setCheckpoint runs, leaving stale per-turn
      // counters or GGUF identity metadata under the new checkpoint.
      const checkpointChanged = state.params.checkpoint !== params.checkpoint;
      return {
        params,
        ...(checkpointChanged
          ? {
              contextUsage: null,
              activeGgufVariant: null,
              ...clearedLoadedGgufMetadata(),
            }
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
  setLoras: (loras) => set({ loras }),
  setThreadRunning: (threadId, running) =>
    set((state) => {
      const next = { ...state.runningByThreadId };
      if (running) {
        next[threadId] = true;
      } else {
        delete next[threadId];
      }
      return { runningByThreadId: next };
    }),
  registerThreadCancel: (threadId, cancel) =>
    set((state) => {
      const next = { ...state.cancelByThreadId };
      next[threadId] = cancel;
      return { cancelByThreadId: next };
    }),
  clearThreadCancel: (threadId) =>
    set((state) => {
      if (!(threadId in state.cancelByThreadId)) return state;
      const next = { ...state.cancelByThreadId };
      delete next[threadId];
      return { cancelByThreadId: next };
    }),
  setAutoTitle: (autoTitle) =>
    set((state) => {
      setScalarSettingVersion("autoTitle", autoTitle, state.autoTitle);
      return { autoTitle };
    }),
  setHfToken: (hfToken) => useHfTokenStore.getState().setToken(hfToken),
  setModelsError: (modelsError) => set({ modelsError }),
  setLastModelLoadError: (lastModelLoadError) => set({ lastModelLoadError }),
  setCheckpoint: (modelId, ggufVariant) =>
    set((state) => {
      // Persist external selections so they survive a refresh. Local ids are
      // NOT persisted -- they're re-derived from the backend on mount, and a
      // stale persisted local id would race the freshly-loaded model. See
      // LAST_EXTERNAL_CHECKPOINT_KEY notes.
      saveLastExternalCheckpoint(isExternalModelId(modelId) ? modelId : null);
      // Clear stale per-turn usage on model change; the relaxed external-provider
      // render gate would otherwise show old counters until the next completion.
      const checkpointChanged = state.params.checkpoint !== modelId;
      const nextGgufVariant = ggufVariant ?? null;
      const loadedGgufSourceChanged =
        checkpointChanged || state.activeGgufVariant !== nextGgufVariant;
      const pendingToClear =
        checkpointChanged && state.params.checkpoint
          ? state.pendingSelection
          : null;
      if (pendingToClear) {
        cancelStagedModelDownload(pendingToClear);
      }
      // Clamp maxTokens to the new model's cap when switching into an external
      // model so a value carried over from a local session doesn't exceed the
      // slider's max.
      let nextMaxTokens = state.params.maxTokens;
      if (checkpointChanged && isExternalModelId(modelId)) {
        const parsed = parseExternalModelId(modelId);
        const provider = parsed
          ? useExternalProvidersStore
              .getState()
              .providers.find((p) => p.id === parsed.providerId)
          : null;
        const cap = getExternalMaxOutputTokens(
          provider?.providerType,
          parsed?.modelId,
        );
        if (nextMaxTokens > cap) {
          nextMaxTokens = cap;
        }
      }
      return {
        params: {
          ...state.params,
          checkpoint: modelId,
          maxTokens: nextMaxTokens,
        },
        activeGgufVariant: nextGgufVariant,
        ...(checkpointChanged ? { contextUsage: null } : {}),
        ...(loadedGgufSourceChanged ? clearedLoadedGgufMetadata() : {}),
        // Switching away from a loaded model (e.g. picking an external provider)
        // abandons any staged pick, so its Load button and edited knobs don't
        // linger over the newly active model. Same revert as abandonStagedModel.
        // Guarded on a non-empty current checkpoint: an establishing set from a
        // background status sync (empty -> active) must not wipe a fresh stage.
        ...(pendingToClear
          ? { ...loadedBaselineSettings(state), pendingSelection: null }
          : {}),
      };
    }),
  setActiveThreadId: (activeThreadId) =>
    set({ activeThreadId, contextUsage: null }),
  setActiveProjectId: (activeProjectId) => set({ activeProjectId }),
  setIncognito: (incognito) => set({ incognito }),
  setSettingsPanelOpen: (settingsPanelOpen) => set({ settingsPanelOpen }),
  setEditingMessageId: (id) => set({ editingMessageId: id }),
  clearCheckpoint: () => {
    // Mirror setCheckpoint's persistence: dropping the checkpoint must also
    // clear any stored external selection so the next refresh doesn't snap
    // back to a model the user intentionally cleared.
    saveLastExternalCheckpoint(null);
    cancelStagedModelDownload(get().pendingSelection);
    return set((state) => ({
      params: {
        ...state.params,
        checkpoint: "",
      },
      activeGgufVariant: null,
      activeNativePathToken: null,
      activeNativePathTokenExpiresAtMs: null,
      pendingSelection: null,
      ggufContextLength: null,
      ggufMaxContextLength: null,
      ggufNativeContextLength: null,
      modelRequiresTrustRemoteCode: false,
      contextUsage: null,
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
      artifactsEnabled: false,
      mcpEnabledForChat: false,
      webFetchToolsEnabled: false,
      // Only the per-session enable pill resets; source/mode/top_k persist.
      ragEnabled: false,
      toolStatus: null,
      toolLiveOutput: {},
      toolFullOutput: {},
      activeDiffusionCanvas: null,
      kvCacheDtype: null,
      loadedKvCacheDtype: null,
      speculativeType: readPersistedSpeculativeType(),
      loadedSpeculativeType: null,
      specFallbackReason: null,
      specDraftNMax: null,
      loadedSpecDraftNMax: null,
      tensorParallel: false,
      loadedTensorParallel: null,
      loadedIsMultimodal: false,
      loadedIsDiffusion: false,
      customContextLength: null,
      defaultChatTemplate: null,
      chatTemplateOverride: null,
      loadedChatTemplateOverride: null,
      pendingImageEditReference: null,
    }));
  },
  setReasoningEnabled: (reasoningEnabled, options) =>
    set(() => {
      if (options?.persist !== false) {
        saveBool(CHAT_REASONING_ENABLED_KEY, reasoningEnabled);
      }
      return { reasoningEnabled };
    }),
  setLastOpenRouterChosenModel: (lastOpenRouterChosenModel) =>
    set({ lastOpenRouterChosenModel }),
  setReasoningStyle: (reasoningStyle) => set({ reasoningStyle }),
  setReasoningEffort: (reasoningEffort) =>
    set((state) => {
      setScalarSettingVersion(
        "reasoningEffort",
        reasoningEffort,
        state.reasoningEffort,
      );
      return { reasoningEffort };
    }),
  setPreserveThinking: (preserveThinking) =>
    set((state) => {
      setScalarSettingVersion(
        "preserveThinking",
        preserveThinking,
        state.preserveThinking,
      );
      return { preserveThinking };
    }),
  setToolsEnabled: (toolsEnabled, options) =>
    set(() => {
      if (options?.persist !== false) {
        saveBool(CHAT_TOOLS_ENABLED_KEY, toolsEnabled);
      }
      return { toolsEnabled };
    }),
  setCodeToolsEnabled: (codeToolsEnabled) =>
    set(() => {
      saveBool(CHAT_CODE_TOOLS_ENABLED_KEY, codeToolsEnabled);
      return { codeToolsEnabled };
    }),
  setImageToolsEnabled: (imageToolsEnabled) =>
    set(() => {
      saveBool(CHAT_IMAGE_TOOLS_ENABLED_KEY, imageToolsEnabled);
      return { imageToolsEnabled };
    }),
  setArtifactsEnabled: (artifactsEnabled, options) =>
    set(() => {
      if (options?.persist !== false) {
        saveBool(CHAT_ARTIFACTS_ENABLED_KEY, artifactsEnabled);
      }
      return { artifactsEnabled };
    }),
  setShowCanvasMenuItem: (showCanvasMenuItem) =>
    set(() => {
      saveBool(CHAT_SHOW_CANVAS_MENU_ITEM_KEY, showCanvasMenuItem);
      return { showCanvasMenuItem };
    }),
  setCollapseHtmlArtifacts: (collapseHtmlArtifacts) =>
    set((state) => {
      saveBool(CHAT_COLLAPSE_HTML_ARTIFACTS_KEY, collapseHtmlArtifacts);
      setScalarSettingVersion(
        "collapseHtmlArtifacts",
        collapseHtmlArtifacts,
        state.collapseHtmlArtifacts,
      );
      return { collapseHtmlArtifacts };
    }),
  setAllowArtifactNetworkAccess: (allowArtifactNetworkAccess) =>
    set((state) => {
      saveBool(
        CHAT_ALLOW_ARTIFACT_NETWORK_ACCESS_KEY,
        allowArtifactNetworkAccess,
      );
      setScalarSettingVersion(
        "allowArtifactNetworkAccess",
        allowArtifactNetworkAccess,
        state.allowArtifactNetworkAccess,
      );
      return { allowArtifactNetworkAccess };
    }),
  setMcpEnabledForChat: (mcpEnabledForChat) =>
    set(() => {
      saveBool(CHAT_MCP_ENABLED_KEY, mcpEnabledForChat);
      return { mcpEnabledForChat };
    }),
  setConfirmToolCalls: (confirmToolCalls) =>
    set((state) => {
      saveBool(CHAT_CONFIRM_TOOL_CALLS_KEY, confirmToolCalls);
      // The legacy toggle is a view over the permission level: on -> "ask",
      // off -> "off" (no prompts). While "full" is active the level is left
      // alone (the toggle is disabled in the UI anyway).
      if (state.permissionMode === "full") return { confirmToolCalls };
      const permissionMode: PermissionMode = confirmToolCalls ? "ask" : "off";
      savePermissionMode(permissionMode);
      return { confirmToolCalls, permissionMode };
    }),
  setPermissionMode: (permissionMode) =>
    set(() => {
      // "full" is session-only (never persisted, see init); ask/auto/off
      // persist and keep the legacy confirm toggle in sync (the gate is
      // requested for both ask and auto).
      savePermissionMode(permissionMode);
      if (permissionMode === "full") {
        // Full access sends confirm_tool_calls=false; keep the store flag in
        // sync so response metadata does not report confirmations as enabled.
        return { permissionMode, bypassPermissions: true, confirmToolCalls: false };
      }
      const confirmToolCalls =
        permissionMode === "ask" || permissionMode === "auto";
      saveBool(CHAT_CONFIRM_TOOL_CALLS_KEY, confirmToolCalls);
      return { permissionMode, bypassPermissions: false, confirmToolCalls };
    }),
  setBypassPermissions: (bypassPermissions) =>
    // Deliberately not persisted (see init): a reload must not silently keep
    // the sandbox/confirmation bypass active without re-accepting the warning.
    // Turning bypass off returns to the last persisted ask/auto level.
    set(() => {
      if (bypassPermissions) {
        // Full access never prompts; mirror confirm_tool_calls=false in the
        // store so metadata does not report confirmations as enabled.
        return {
          bypassPermissions,
          permissionMode: "full" as PermissionMode,
          confirmToolCalls: false,
        };
      }
      const permissionMode = loadPermissionMode();
      return {
        bypassPermissions,
        permissionMode,
        confirmToolCalls: permissionMode === "ask" || permissionMode === "auto",
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
    set(() => {
      saveBool(CHAT_WEB_FETCH_TOOLS_ENABLED_KEY, webFetchToolsEnabled);
      return { webFetchToolsEnabled };
    }),
  setRagEnabled: (ragEnabled) => set(() => ({ ragEnabled })),
  setRagSource: (ragSource) =>
    set(() => {
      saveRagSource(ragSource);
      return { ragSource };
    }),
  setRagMode: (ragMode) =>
    set(() => {
      saveString(CHAT_RAG_MODE_KEY, ragMode);
      return { ragMode };
    }),
  setRagTopK: (ragTopK) =>
    set(() => {
      saveString(CHAT_RAG_TOP_K_KEY, String(ragTopK));
      return { ragTopK };
    }),
  setRagAutoInject: (ragAutoInject) =>
    set(() => {
      saveString(CHAT_RAG_AUTOINJECT_KEY, ragAutoInject);
      return { ragAutoInject };
    }),
  setRagAutoInjectMinScore: (ragAutoInjectMinScore) =>
    set(() => {
      saveString(
        CHAT_RAG_AUTOINJECT_MIN_SCORE_KEY,
        String(ragAutoInjectMinScore),
      );
      return { ragAutoInjectMinScore };
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
  setToolStatus: (toolStatus) => set({ toolStatus }),
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
  setActiveDiffusionCanvas: (activeDiffusionCanvas) =>
    set({ activeDiffusionCanvas }),
  setGeneratingStatus: (generatingStatus) => set({ generatingStatus }),
  setAutoHealToolCalls: (autoHealToolCalls) =>
    set((state) => {
      setScalarSettingVersion(
        "autoHealToolCalls",
        autoHealToolCalls,
        state.autoHealToolCalls,
      );
      return { autoHealToolCalls };
    }),
  setNudgeToolCalls: (nudgeToolCalls) =>
    set((state) => {
      setScalarSettingVersion(
        "nudgeToolCalls",
        nudgeToolCalls,
        state.nudgeToolCalls,
      );
      return { nudgeToolCalls };
    }),
  setMaxToolCallsPerMessage: (maxToolCallsPerMessage) =>
    set((state) => {
      setScalarSettingVersion(
        "maxToolCallsPerMessage",
        maxToolCallsPerMessage,
        state.maxToolCallsPerMessage,
      );
      return { maxToolCallsPerMessage };
    }),
  setToolCallTimeout: (toolCallTimeout) =>
    set((state) => {
      setScalarSettingVersion(
        "toolCallTimeout",
        toolCallTimeout,
        state.toolCallTimeout,
      );
      return { toolCallTimeout };
    }),
  setKvCacheDtype: (kvCacheDtype) => set({ kvCacheDtype }),
  setSpeculativeType: (speculativeType) => set({ speculativeType }),
  setSpecDraftNMax: (specDraftNMax) => set({ specDraftNMax }),
  setTensorParallel: (tensorParallel) => set({ tensorParallel }),
  resetModelSettingsToLoaded: () => set((s) => loadedBaselineSettings(s)),
  applyRememberedLoadSettings: (settings) =>
    // Coalesce every field: a blob persisted by an older/newer build can omit
    // keys, and a raw spread would push `undefined` into fields typed non-null.
    set({
      customContextLength: settings.contextLength ?? null,
      kvCacheDtype: settings.kvCacheDtype ?? null,
      speculativeType: settings.speculativeType ?? "auto",
      specDraftNMax: settings.specDraftNMax ?? null,
      tensorParallel: settings.tensorParallel ?? false,
    }),
  setLoadOnSelection: (loadOnSelection) => {
    saveBool(CHAT_LOAD_ON_SELECTION_KEY, loadOnSelection);
    set({ loadOnSelection });
  },
  setExpandQuantizations: (expandQuantizations) => {
    saveBool(CHAT_EXPAND_QUANTIZATIONS_KEY, expandQuantizations);
    set({ expandQuantizations });
  },
  setShowAllQuantizations: (showAllQuantizations) => {
    saveBool(CHAT_SHOW_ALL_QUANTIZATIONS_KEY, showAllQuantizations);
    set({ showAllQuantizations });
  },
  setFitOnDeviceOnly: (fitOnDeviceOnly) => {
    saveBool(MODELS_FIT_ON_DEVICE_ONLY_KEY, fitOnDeviceOnly);
    set({ fitOnDeviceOnly });
  },
  setPendingSelection: (pendingSelection) => set({ pendingSelection }),
  stageModel: (selection) => {
    // Refuse staging mid-load: post-load cleanup would silently drop the queued
    // pick. stageOrLoad toasts first for callers that can.
    if (get().modelLoading) return;
    // Rebinding to a new pick keeps the prior pick's download running so the
    // user can queue multiple downloads at once (Hub-style).
    set((s) => {
      return {
        ...loadedBaselineSettings(s),
        pendingSelection: selection,
        // autoLoad downloads silently and loads on completion, so keep the sheet shut.
        settingsPanelOpen: !selection.autoLoad,
        // Speculative starts from the standing default, not the loaded model's
        // mode, so a fresh pick doesn't inherit (and then carry, via the staged
        // Load's keepSpeculative) a forced MTP mode onto a model that may lack it.
        speculativeType: readPersistedSpeculativeType(),
        specDraftNMax: null,
      };
    });
  },
  abandonStagedModel: (opts) => {
    const { pendingSelection } = get();
    if (!pendingSelection) return;
    // Cancel the staged pick's in-flight download (centralized for every abandon
    // path: sheet close, thread switch, route exit, new chat). `keepDownload`
    // opts out so navigation leaves the transfer running, like a Hub download.
    if (!opts?.keepDownload) cancelStagedModelDownload(pendingSelection);
    set((s) => ({ ...loadedBaselineSettings(s), pendingSelection: null }));
  },
  setCustomContextLength: (customContextLength) => set({ customContextLength }),
  setChatTemplateOverride: (chatTemplateOverride) =>
    set({ chatTemplateOverride }),
  setPendingAudio: (base64, name) =>
    set({ pendingAudioBase64: base64, pendingAudioName: name }),
  clearPendingAudio: () =>
    set({ pendingAudioBase64: null, pendingAudioName: null }),
  setPendingImageEditReference: (pendingImageEditReference) =>
    set({ pendingImageEditReference }),
  clearPendingImageEditReference: () =>
    set({ pendingImageEditReference: null }),
  setContextUsage: (contextUsage) => set({ contextUsage }),
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
      (speculativeType === "mtp" || speculativeType === "mtp+ngram")
        ? state.specDraftNMax
        : null,
  };
}
