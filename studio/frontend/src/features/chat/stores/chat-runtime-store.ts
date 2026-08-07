// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { mirrorHfTokenInto, useHfTokenStore } from "@/features/hub";
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
import { isExternalModelId, parseExternalModelId } from "../external-providers";
import {
  type ChatPresetSource,
  type Preset,
  getPresetSource,
} from "../presets/preset-policy";
import { normalizePresetLoadConfig } from "../presets/preset-load-config";
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
import {
  chatModelLifecycleGate,
  type ModelLifecycleLease,
} from "../utils/model-lifecycle-gate";
import { shouldAdvanceQueuedSettingsEpoch } from "../utils/queued-settings-epoch";
import type { ResearchWebsitePolicy } from "../types/research";
import { useExternalProvidersStore } from "./external-providers-store";
import { PLUS_MENU_PINS_STORAGE_KEY } from "./plus-menu-prefs-store";

export const CHAT_REASONING_ENABLED_KEY = "unsloth_chat_reasoning_enabled";
export const CHAT_TOOLS_ENABLED_KEY = "unsloth_chat_tools_enabled";
export const CHAT_CODE_TOOLS_ENABLED_KEY = "unsloth_chat_code_tools_enabled";
export const CHAT_IMAGE_TOOLS_ENABLED_KEY = "unsloth_chat_image_tools_enabled";
export const CHAT_DEEP_RESEARCH_ENABLED_KEY =
  "unsloth_chat_deep_research_enabled";
export const CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY =
  "unsloth_chat_deep_research_website_policy";
export const CHAT_ARTIFACTS_ENABLED_KEY = "unsloth_chat_artifacts_enabled";
export const CHAT_SHOW_CANVAS_MENU_ITEM_KEY =
  "unsloth_chat_show_canvas_menu_item";
export const CHAT_COLLAPSE_HTML_ARTIFACTS_KEY =
  "unsloth_chat_collapse_html_artifacts";
export const CHAT_ALLOW_ARTIFACT_NETWORK_ACCESS_KEY =
  "unsloth_chat_allow_artifact_network_access";
export const CHAT_MCP_ENABLED_KEY = "unsloth_chat_mcp_enabled";
export const CHAT_CONFIRM_TOOL_CALLS_KEY = "unsloth_chat_confirm_tool_calls";
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
export const CHAT_SPECULATIVE_TYPE_KEY = "unsloth_chat_speculative_type";
export const CHAT_GPU_MEMORY_MODE_KEY = "unsloth_chat_gpu_memory_mode";

// Persist only the model-agnostic intents (auto/ngram/off). The model-specific
// drafter modes (mtp/mtp+ngram/dspark) and spec_draft_n_max stay session-only:
// a persisted choice would silently no-op on a model with no MTP head or no
// DSpark sidecar. Unknown -> auto.
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
export const DEFAULT_RESEARCH_WEBSITE_POLICY: ResearchWebsitePolicy = {
  allowedDomains: [],
  blockedDomains: [],
};

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
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      CHAT_DEEP_RESEARCH_WEBSITE_POLICY_KEY,
      JSON.stringify(policy),
    );
  } catch {
    // Keep the in-memory setting when storage is unavailable.
  }
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
  if (s === "dspark" || s === "draft-dspark") return "dspark";
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

// GPU Memory strategy is a standing preference (like speculative type), not a
// per-model setting: a "manual" choice persists across model switches and reloads.
export function readPersistedGpuMemoryMode(): "auto" | "manual" {
  return loadString(CHAT_GPU_MEMORY_MODE_KEY, "auto") === "manual" ? "manual" : "auto";
}

export function saveGpuMemoryMode(value: "auto" | "manual"): void {
  saveString(CHAT_GPU_MEMORY_MODE_KEY, value);
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

/** A pick is a GGUF: HF variant, native file, or a direct local .gguf. */
export function hasGgufSource(x: {
  ggufVariant?: string;
  nativePathToken?: string;
  isGguf?: boolean;
}): boolean {
  return (
    x.ggufVariant != null || x.nativePathToken != null || x.isGguf === true
  );
}

/** A local-disk model id: Unix absolute (/), relative (./ ../), tilde (~/),
 *  Windows drive (C:\) or UNC (\\server). Shared so the loader and the
 *  hub-repo predicate classify ids identically. */
export function isLocalModelPath(id: string): boolean {
  return /^(\/|\.{1,2}[\\/]|~[\\/]|[A-Za-z]:[\\/]|\\\\)/.test(id);
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
  params: InferenceParams;
  customPresets: Preset[];
  activePreset: string;
  activePresetSource: ChatPresetSource;
  models: ChatModelSummary[];
  loras: ChatLoraSummary[];
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
  /** Whether the backend loaded the active model from a filesystem path. */
  activeModelIsLocal: boolean;
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
  deepResearchEnabled: boolean;
  researchWebsitePolicy: ResearchWebsitePolicy;
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
  /**
   * Which drafter the loaded model's speculative resolution was about, "mtp" or
   * "dspark". Paired with specFallbackReason: the reason alone cannot name the
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
  /** Tensor-parallel split (--split-mode tensor) toggle, GGUF multi-GPU only. */
  tensorParallel: boolean;
  /** Backend-reported tensor-parallel state; null until first hydrated. */
  loadedTensorParallel: boolean | null;
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
    options?: { persist?: boolean; trackQueuedSettings?: boolean },
  ) => void;
  setCustomPresets: (presets: Preset[]) => void;
  setActivePreset: (name: string) => void;
  setActivePresetSource: (source: ChatPresetSource) => void;
  setModels: (models: ChatModelSummary[]) => void;
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
    options?: { trackQueuedSettings?: boolean },
  ) => void;
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
  setDeepResearchEnabled: (enabled: boolean) => void;
  setResearchWebsitePolicy: (policy: ResearchWebsitePolicy) => void;
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
  localRunByThreadId: {},
  runOwnerByThreadId: {},
  cancelByThreadId: {},
  serverCancelByThreadId: {},
  autoTitle: false,
  hfToken: useHfTokenStore.getState().token,
  modelsError: null,
  lastModelLoadError: null,
  activeGgufVariant: null,
  activeModelIsLocal: false,
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
  deepResearchEnabled: loadBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false),
  researchWebsitePolicy: loadResearchWebsitePolicy(),
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
  toolStatusByThreadId: {},
  toolLiveOutput: {},
  toolFullOutput: {},
  generatingStatus: null,
  activeDiffusionCanvasByThreadId: {},
  autoHealToolCalls: true,
  nudgeToolCalls: true,
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
  specDrafterKind: null,
  specDraftNMax: null,
  loadedSpecDraftNMax: null,
  nParallel: null,
  loadedNParallel: null,
  tensorParallel: false,
  loadedTensorParallel: null,
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
      // Bump version unconditionally so a late hydration response won't clobber
      // a pre-hydrate user edit; only the HTTP write is gated on settingsHydrated.
      const changedParams = getChangedInferenceParams(params, state.params);
      const queuedSettingsChanged = shouldAdvanceQueuedSettingsEpoch(
        state.params,
        params,
        options?.trackQueuedSettings !== false,
      );
      if (
        options?.persist !== false &&
        state.settingsHydrated &&
        hasKeys(changedParams)
      ) {
        saveSettingsPatch({ inferenceParams: changedParams });
      }
      // Mirror setCheckpoint: the local load path can mutate params.checkpoint
      // via setParams() before setCheckpoint runs, leaving stale per-turn
      // counters under the new checkpoint.
      const checkpointChanged = state.params.checkpoint !== params.checkpoint;
      return {
        params,
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
  setLoras: (loras) => set({ loras }),
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
      if (isExternalModelId(modelId)) {
        saveBool(CHAT_DEEP_RESEARCH_ENABLED_KEY, false);
      }
      // Clear stale per-turn usage on model change; the relaxed external-provider
      // render gate would otherwise show old counters until the next completion.
      const checkpointChanged = state.params.checkpoint !== modelId;
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
      const nextGgufVariant = ggufVariant ?? null;
      const nextDeepResearchEnabled = isExternalModelId(modelId)
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
      return {
        params: {
          ...state.params,
          checkpoint: modelId,
          maxTokens: nextMaxTokens,
        },
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
              specDrafterKind: null,
            }
          : {}),
        // Switching to an external provider disables Deep Research, which only
        // applies to the local base model.
        ...(isExternalModelId(modelId) ? { deepResearchEnabled: false } : {}),
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
      params: {
        ...state.params,
        checkpoint: "",
      },
      activeGgufVariant: null,
      activeModelIsLocal: false,
      activeLoadId: null,
      activeNativePathToken: null,
      activeNativePathExpiresAtMs: null,
      ggufContextLength: null,
      ggufMaxContextLength: null,
      ggufNativeContextLength: null,
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
      specDrafterKind: null,
      specDraftNMax: null,
      loadedSpecDraftNMax: null,
      nParallel: null,
      loadedNParallel: null,
      tensorParallel: false,
      loadedTensorParallel: null,
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
      return {
        preserveThinking,
        queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
      };
    }),
  setToolsEnabled: (toolsEnabled, options) =>
    set((state) => {
      if (options?.persist !== false) {
        saveBool(CHAT_TOOLS_ENABLED_KEY, toolsEnabled);
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
      const permissionMode = loadPermissionMode();
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
      const permissionMode = loadPermissionMode();
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
    set((state) => ({
      ragEnabled,
      queuedSettingsEpoch: state.queuedSettingsEpoch + 1,
    })),
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
