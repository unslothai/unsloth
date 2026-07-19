// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthToken } from "@/features/auth";
import {
  loadRememberedLoadSettings,
  rememberedLoadSettingsKey,
} from "@/components/assistant-ui/model-selector/remembered-load-settings";
import { projectHasSources } from "@/features/rag/api/rag-api";
import { apiUrl } from "@/lib/api-base";
import { parseParamCountB } from "@/lib/model-size";
import { toast } from "@/lib/toast";
import type { MessageTiming, ToolCallMessagePart } from "@assistant-ui/core";
import type { ChatModelAdapter } from "@assistant-ui/react";
import { parsePartialJsonObject } from "assistant-stream/utils";
import {
  getExternalProviderApiKey,
  isCustomProviderType,
  isPromptCacheTtl,
  loadExternalProviders,
  parseExternalModelId,
  providerTypeSupportsVision,
  supportsProviderPromptCacheTtl,
  supportsProviderPromptCaching,
  toExternalBackendProviderType,
} from "../external-providers";
import { pickFriendlyContainerName } from "../lib/friendly-names";
import {
  reasoningCapsFromLoad,
  tryAdoptServerActiveModel,
} from "../lib/apply-inference-status-to-store";
import {
  clampReasoningEffortToLevels,
  getExternalMaxOutputTokens,
  getExternalMinOutputTokens,
  getExternalReasoningCapabilities,
  getProviderCapabilities,
  isGeminiCustomOpenAICompatBase,
  providerSupportsBuiltinCodeExecution,
  providerSupportsBuiltinImageGeneration,
  providerSupportsBuiltinWebFetch,
  providerSupportsBuiltinWebSearch,
  providerSupportsFastMode,
} from "../provider-capabilities";
import {
  type PendingImageEditReference,
  type RagAutoInject,
  GPU_LAYERS_AUTO,
  loadedGpuMemoryFieldsUnlessStaged,
  reconcilePersistedGpuIds,
  resolveLoadedSpeculativeSettings,
  resolveSpeculativeSettingsForLoad,
  persistGpuMemoryModeOnLoad,
  resolveToolsEnabledOnLoad,
  saveSpeculativeType,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";
import { resolveFitMaxSeqLength, resolveManualAutoCtxPin } from "../presets/preset-policy";
import { ensureGpuDeviceCache } from "@/hooks/use-gpu-info";
import { useExternalProvidersStore } from "../stores/external-providers-store";
import {
  shouldPreserveFullOutput,
  toolOutputKey,
  toolPaneScope,
} from "../tool-output-scope";
import type { ModelType } from "../types";
import { isMultimodalResponse } from "../types/api";
import type {
  GgufVariantDetail,
  OpenAIChatCompletionsRequest,
  OpenAIChatMessage,
  OpenAIMessageContent,
  OpenAIReasoningContentPart,
} from "../types/api";
import type { ChatModelSummary } from "../types/runtime";
import {
  getStoredChatThread,
  getStoredChatProject,
  listStoredChatThreads,
  updateStoredChatThread,
} from "../utils/chat-history-storage";
import {
  readLastLocalModelLoad,
  recordLastLocalModelLoad,
  type LastLocalModelKind,
} from "../utils/last-local-model-load";
import { getImageInputUnavailableReason } from "../utils/image-input-support";
import {
  hasClosedThinkTag,
  parseAssistantContent,
} from "../utils/parse-assistant-content";
import { resolveLoadMaxSeqLength } from "../presets/preset-policy";
import {
  generateAudio,
  listCachedGguf,
  listCachedModels,
  listGgufVariants,
  loadModel,
  streamChatCompletions,
  StreamInterruptedError,
  validateModel,
} from "./chat-api";
import {
  createOpenAIContainer,
  listOpenAIContainers,
} from "./openai-containers";
import {
  encryptProviderApiKey,
  isProviderKeyRotationError,
} from "./providers-api";

// Small models (<=9B) answer from memory instead of calling search, so "auto"
// forces retrieval for them and leaves it to larger ones.
const AUTOINJECT_AUTO_MAX_SIZE_B = 9;

function resolveAutoInject(mode: RagAutoInject, checkpoint: string): boolean {
  if (mode === "on") return true;
  if (mode === "off") return false;
  const size = parseParamCountB(checkpoint);
  // Unknown size -> enable.
  return size === null || size <= AUTOINJECT_AUTO_MAX_SIZE_B;
}

/** Server-side usage data from llama-server (via stream_options.include_usage). */
interface ServerUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  // External prompt-cache fields (see _build_usage_chunk in
  // external_provider.py); cache_creation is Anthropic-only.
  prompt_tokens_details?: {
    cached_tokens?: number;
  };
  cache_creation_input_tokens?: number;
  cache_read_input_tokens?: number;
}

/** Server-side timing data from llama-server's timings object. */
interface ServerTimings {
  prompt_n: number;
  cache_n: number;
  prompt_ms: number;
  prompt_per_token_ms: number;
  prompt_per_second: number;
  predicted_n: number;
  predicted_ms: number;
  predicted_per_token_ms: number;
  predicted_per_second: number;
  // DiffusionGemma-only extras (present when serving a diffusion model; ignored otherwise).
  diffusion?: boolean;
  diffusion_blocks?: number;
  diffusion_steps?: number;
  diffusion_canvas?: number;
  diffusion_prompt_n?: number;
  diffusion_prompt_prepare_ms?: number;
  diffusion_decode_ms?: number;
  diffusion_wall_ms?: number;
  // Honest throughput, matching the standalone diffusion CLI:
  //   effective = canvas*blocks/wall, parallel = canvas/per_step, output = answer tokens/wall.
  diffusion_effective_tok_s?: number;
  diffusion_parallel_tok_s?: number;
  diffusion_output_tok_s?: number;
  diffusion_steps_per_second?: number;
}

interface ResponseDetailsMetadata {
  modelId: string;
  modelLabel: string;
  responseModelId: string;
  providerId?: string;
  providerName: string;
  providerType: string;
  startedAt: number;
  finishedAt: number;
  durationMs: number;
  sessionId?: string;
  cancelId: string;
  toolCalls: string[];
  tools: {
    search: boolean;
    fetch: boolean;
    code: boolean;
    images: boolean;
    mcp: boolean;
    docs: boolean;
    artifacts: boolean;
    confirmToolCalls: boolean;
    bypassPermissions: boolean;
    permissionMode?: string;
  };
}

type RunMessages = Parameters<ChatModelAdapter["run"]>[0]["messages"];
type RunMessage = RunMessages[number];

type OpenAIStreamAdapterOptions = {
  modelType?: ModelType;
  pairId?: string;
};

/** Tracks which user messages were sent with an audio file (messageId → filename). */
export const sentAudioNames = new Map<string, string>();

// Synthetic provider-side tool names; backend stamps args._server_tool so
// user functions with the same name aren't dropped. Mirror of backend
// _SERVER_SIDE_BUILTIN_TOOL_NAMES.
const SERVER_SIDE_BUILTIN_TOOL_NAMES = new Set<string>([
  "web_search",
  "web_fetch",
  "code_execution",
  "image_generation",
]);

/**
 * Whether a persisted tool-call part is provider-side synthetic and should
 * be stripped from outbound history. Matches on the args._server_tool marker
 * or a Gemini native_part payload (no shape heuristic, since user functions
 * can legitimately share a name).
 */
function isServerSideBuiltinToolPart(
  toolNameLower: string,
  _argsObj: Record<string, unknown> | null,
  hasServerToolMarker: boolean,
  hasNativePart: boolean,
): boolean {
  if (!SERVER_SIDE_BUILTIN_TOOL_NAMES.has(toolNameLower)) return false;
  if (hasServerToolMarker) return true;
  return hasNativePart;
}

const FIRST_THREAD_SAVE_TIMEOUT_MS = 250;

type ThreadAutosaveHandle = {
  registerFirstSave(threadId: string, promise: Promise<void>): Promise<void>;
  awaitFirstSave(threadId: string | undefined): Promise<void>;
};

const pendingFirstThreadSaves = new Map<string, Promise<void>>();

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Best-effort partial parse of a live tool_args stream into a tool part's
 * `args`, so cards render the payload while the model is still writing it. The
 * structured path streams raw arguments JSON; the text path wraps it in call
 * markup, unwrapped here. Returns null until something parses; never throws.
 */
function parseLiveToolArgs(
  raw: string,
): { args: Record<string, unknown>; argsText: string } | null {
  let candidate = raw.trimStart();
  if (!candidate.startsWith("{")) {
    const brace = candidate.indexOf("{");
    if (brace < 0) return null;
    candidate = candidate.slice(brace);
  }
  const parsed = parsePartialJsonObject(candidate) as
    | Record<string, unknown>
    | undefined;
  if (!parsed || typeof parsed !== "object") return null;
  // Call envelope from the text path: unwrap to the arguments payload.
  const inner = parsed.arguments ?? parsed.parameters;
  if (typeof parsed.name === "string" && inner !== undefined) {
    if (typeof inner === "string") {
      // Stringified arguments: partial-parse the inner JSON string.
      const innerParsed = parsePartialJsonObject(inner) as
        | Record<string, unknown>
        | undefined;
      if (innerParsed && typeof innerParsed === "object") {
        return { args: innerParsed, argsText: inner };
      }
      return null;
    }
    if (inner && typeof inner === "object" && !Array.isArray(inner)) {
      return {
        args: inner as Record<string, unknown>,
        argsText: JSON.stringify(inner),
      };
    }
    return null;
  }
  return { args: parsed, argsText: candidate };
}

function parseSystemVariablesMap(raw: string): Record<string, unknown> {
  if (!raw.trim()) {
    return {};
  }
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
  } catch {
    // Invalid JSON: keep unresolved placeholders in output prompt.
  }
  return {};
}

function hasOwn(object: object, key: string): boolean {
  return Object.prototype.hasOwnProperty.call(object, key);
}

function getNestedValue(
  values: Record<string, unknown>,
  path: string,
): unknown | undefined {
  const parts = path.split(".").map((part) => part.trim()).filter(Boolean);
  if (parts.length === 0) {
    return undefined;
  }
  let current: unknown = values;
  for (const part of parts) {
    if (!current || typeof current !== "object" || Array.isArray(current)) {
      return undefined;
    }
    if (!hasOwn(current, part)) {
      return undefined;
    }
    current = (current as Record<string, unknown>)[part];
  }
  return current;
}

function padDatePart(value: number): string {
  return String(value).padStart(2, "0");
}

function formatLocalDate(now: Date): string {
  return [
    now.getFullYear(),
    padDatePart(now.getMonth() + 1),
    padDatePart(now.getDate()),
  ].join("-");
}

function formatLocalTime(now: Date): string {
  return [
    padDatePart(now.getHours()),
    padDatePart(now.getMinutes()),
    padDatePart(now.getSeconds()),
  ].join(":");
}

function formatTimezoneOffset(now: Date): string {
  const offsetMinutes = -now.getTimezoneOffset();
  const sign = offsetMinutes >= 0 ? "+" : "-";
  const abs = Math.abs(offsetMinutes);
  const hours = Math.floor(abs / 60);
  const minutes = abs % 60;
  return `${sign}${padDatePart(hours)}:${padDatePart(minutes)}`;
}

function stringifyTemplateValue(value: unknown): string {
  if (value == null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function resolveSystemPromptVariables(
  prompt: string,
  customVariablesRaw: string,
): string {
  if (!prompt) {
    return prompt;
  }
  const now = new Date();
  const localDate = formatLocalDate(now);
  const localTime = formatLocalTime(now);
  const systemVariables: Record<string, string> = {
    $date: localDate,
    $time: localTime,
    $now: `${localDate}T${localTime}${formatTimezoneOffset(now)}`,
  };
  const customVariables = parseSystemVariablesMap(customVariablesRaw);
  return prompt.replaceAll(
    /{{\s*([a-zA-Z_$][a-zA-Z0-9_$.-]*)\s*}}/g,
    (full, keyRaw) => {
      const key = String(keyRaw).trim();
      if (hasOwn(systemVariables, key)) {
        return systemVariables[key] ?? full;
      }
      const resolved = getNestedValue(customVariables, key);
      if (resolved === undefined) {
        return full;
      }
      return stringifyTemplateValue(resolved);
    },
  );
}

export const ThreadAutosaveHandle: ThreadAutosaveHandle = {
  registerFirstSave(threadId, promise) {
    const trackedPromise = promise.catch(() => {});
    const cleanupPromise = trackedPromise.finally(() => {
      if (pendingFirstThreadSaves.get(threadId) === cleanupPromise) {
        pendingFirstThreadSaves.delete(threadId);
      }
    });
    pendingFirstThreadSaves.set(threadId, cleanupPromise);
    return cleanupPromise;
  },

  async awaitFirstSave(threadId) {
    if (!threadId) {
      return;
    }
    const pending = pendingFirstThreadSaves.get(threadId);
    if (!pending) {
      return;
    }
    await Promise.race([pending, wait(FIRST_THREAD_SAVE_TIMEOUT_MS)]);
  },
};

export function useThreadAutosaveHandle(): ThreadAutosaveHandle {
  return ThreadAutosaveHandle;
}

/**
 * Match error messages indicating the request filled (or would fill) the KV
 * cache, so the UI can toast a pointer at the ``Context Length`` setting.
 *
 * Two wordings reach the client and both must hit:
 *   1. Raw llama-server text when ``--no-context-shift`` trips:
 *      "the request exceeds the available context size (N tokens)".
 *   2. The friendly rewrite from
 *      ``backend/routes/inference.py::_friendly_error``: "Message too long:
 *      ... context window. Try increasing the Context Length ..." (the one
 *      most users see on the streaming GGUF path).
 *
 * Match substrings, not full regexes: both layers have drifted across
 * versions (llama.cpp phrasing and _friendly_error copy edits).
 */
export function isContextLimitError(message: string): boolean {
  if (!message) return false;
  const m = message.toLowerCase();
  return (
    // Raw llama-server wording.
    m.includes("context size") ||
    m.includes("context shift") ||
    m.includes("exceeds the available context") ||
    // Backend _friendly_error rewrite.
    m.includes("message too long") ||
    m.includes("context window") ||
    // n_ctx mentions that carry an "exceed"/"full" signal.
    (m.includes("n_ctx") && (m.includes("exceed") || m.includes("full")))
  );
}

async function updateStoredChatThreadEventually(
  threadId: string,
  patch: Parameters<typeof updateStoredChatThread>[1],
): Promise<void> {
  for (let attempt = 0; attempt < 10; attempt++) {
    const updated = await updateStoredChatThread(threadId, patch).catch(
      () => undefined,
    );
    if (updated) return;
    await wait(50);
  }
}

/**
 * Return ``raw`` when it is a safe-to-navigate http(s) URL, else "".
 * Rejects non-string input, CR/LF (header injection), and non-http(s)
 * schemes (``javascript:`` / ``data:`` / ``vbscript:``) so provider/tool-
 * controlled strings cannot land in an <a href>.
 */
function isSafeNavigableSourceUrl(raw: unknown): string {
  if (typeof raw !== "string") return "";
  const value = raw.trim();
  if (!value || /[\r\n]/.test(value)) return "";
  try {
    const parsed = new URL(value);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") {
      return value;
    }
  } catch {
    // Fall through.
  }
  return "";
}

/** Convert an Anthropic document citation dict into a Sources-panel source. */
function documentCitationToSource(
  cit: Record<string, unknown>,
  fallbackIdx: number,
): {
  type: "source";
  sourceType: "url";
  id: string;
  url: string;
  title: string;
  metadata?: { description: string };
} | null {
  const source = typeof cit.source === "string" && cit.source ? cit.source : "";
  const docTitle =
    (typeof cit.document_title === "string" && cit.document_title) ||
    (typeof cit.title === "string" && cit.title) ||
    "";
  const docIndex =
    typeof cit.document_index === "number" ? cit.document_index : undefined;
  // Only treat ``source`` as navigable when it is real http(s);
  // search_result_location can carry a free-form id (e.g. ``kb-doc-42``)
  // or a hostile scheme. Fall back to a stable doc anchor otherwise.
  const url =
    isSafeNavigableSourceUrl(source) ||
    `#anthropic-doc-${docIndex ?? fallbackIdx}`;
  const title = docTitle || source || `Document ${fallbackIdx + 1}`;
  const cited = typeof cit.cited_text === "string" ? cit.cited_text.trim() : "";
  // Trim the cited snippet so the Sources panel stays scannable.
  const description = cited.length > 240 ? `${cited.slice(0, 240)}...` : cited;
  // Anthropic numbers inline [N] per citation, not per source URL.
  // Fold citation type + position-bearing fields into the id so distinct
  // citations on the same source keep separate Sources entries.
  const citationType =
    typeof cit.type === "string" ? String(cit.type) : "";
  const positionParts = [
    cit.search_result_index,
    cit.start_char_index,
    cit.end_char_index,
    cit.start_page_number,
    cit.end_page_number,
    cit.start_block_index,
    cit.end_block_index,
  ]
    .filter((v) => typeof v === "number")
    .map((v) => String(v))
    .join(":");
  const idAnchor = positionParts
    ? `${citationType}:${positionParts}`
    : `${citationType}:${fallbackIdx}`;
  const id = `${url}#${idAnchor}`;
  return {
    type: "source" as const,
    sourceType: "url" as const,
    id,
    url,
    title,
    ...(description ? { metadata: { description } } : {}),
  };
}

/** Parse "Title: ...\nURL: ...\nSnippet: ..." blocks into source content parts. */
function parseSourcesFromResult(raw: string): {
  type: "source";
  sourceType: "url";
  id: string;
  url: string;
  title: string;
  metadata?: { description: string };
}[] {
  if (!raw) return [];
  const blocks = raw.split(/\n---\n/).filter(Boolean);
  const sources: {
    type: "source";
    sourceType: "url";
    id: string;
    url: string;
    title: string;
    metadata?: { description: string };
  }[] = [];
  for (const block of blocks) {
    const titleMatch = block.match(/Title:\s*(.+)/);
    const urlMatch = block.match(/URL:\s*(.+)/);
    const snippetMatch = block.match(/Snippet:\s*(.+)/);
    if (titleMatch && urlMatch) {
      // Drop blocks whose ``URL:`` is not safe http(s); provider/tool
      // output is attacker-controllable, so a hostile scheme must not
      // reach the Sources panel <a href>.
      const url = isSafeNavigableSourceUrl(urlMatch[1]);
      if (!url) continue;
      const snippet = snippetMatch?.[1]?.trim();
      sources.push({
        type: "source" as const,
        sourceType: "url" as const,
        id: url,
        url,
        title: titleMatch[1].trim(),
        ...(snippet ? { metadata: { description: snippet } } : {}),
      });
    }
  }
  return sources;
}

function estimateTokenCount(text: string): number | undefined {
  const trimmed = text.trim();
  if (!trimmed) {
    return undefined;
  }
  return Math.max(1, Math.round(trimmed.length / 4));
}

/**
 * Normalize a streamed `delta.content` to a plain text string.
 *
 * OpenAI Chat Completions originally typed `delta.content` as a string, but
 * some providers now emit an array of structured content parts; concatenating
 * those directly would stringify each as `[object Object]`. This guards that.
 *
 * Handled part shapes:
 *   { type: "text" | "output_text", text | content: "..." }   → text body
 *   { type: "thinking" | "reasoning", thinking | text: "..." } → wrapped as
 *       inline `<think>...</think>` so `parseAssistantContent` lifts it into
 *       a reasoning part (else Mistral magistral and similar reasoning-part
 *       providers lose their thinking panel).
 *
 * Unknown part types are skipped — better to drop a stray field than
 * stringify an object into the rendered chat.
 */
function extractDeltaText(delta: unknown): string {
  const extractReasoningText = (payload: unknown): string => {
    if (typeof payload === "string") return payload;
    if (Array.isArray(payload)) {
      return payload.map((item) => extractReasoningText(item)).join("");
    }
    if (!payload || typeof payload !== "object") return "";

    const obj = payload as Record<string, unknown>;
    for (const key of ["thinking", "text", "content", "reasoning", "summary"]) {
      if (key in obj) {
        const text = extractReasoningText(obj[key]);
        if (text) return text;
      }
    }
    return "";
  };

  if (typeof delta === "string") return delta;
  if (!Array.isArray(delta)) return "";
  let out = "";
  for (const part of delta) {
    if (typeof part === "string") {
      out += part;
      continue;
    }
    if (!part || typeof part !== "object") continue;
    const obj = part as {
      type?: string;
      text?: string;
      content?: string;
      thinking?: string;
    };
    if (obj.type === "text" || obj.type === "output_text") {
      if (typeof obj.text === "string") out += obj.text;
      else if (typeof obj.content === "string") out += obj.content;
    } else if (obj.type === "thinking" || obj.type === "reasoning") {
      const thinking = extractReasoningText(obj);
      if (thinking) out += `<think>${thinking}</think>`;
    }
  }
  return out;
}

function buildTiming(
  streamStartTime: number,
  totalChunks: number,
  firstTokenTime?: number,
  totalStreamTime?: number,
  tokenCount?: number,
  toolCallCount = 0,
  tokensPerSecondOverride?: number,
): MessageTiming {
  return {
    streamStartTime,
    firstTokenTime,
    totalStreamTime,
    tokenCount,
    tokensPerSecond:
      tokensPerSecondOverride ??
      (typeof totalStreamTime === "number" &&
      totalStreamTime > 0 &&
      typeof tokenCount === "number"
        ? tokenCount / (totalStreamTime / 1000)
        : undefined),
    totalChunks,
    toolCallCount,
  };
}

function collectTextParts(message: RunMessage): string[] {
  const textParts = message.content
    .filter((part) => part.type === "text")
    .map((part) => part.text);

  if ("attachments" in message && (message.attachments?.length ?? 0) > 0) {
    for (const attachment of message.attachments ?? []) {
      for (const part of attachment.content ?? []) {
        if (part.type === "text") {
          textParts.push(part.text);
        }
      }
    }
  }

  return textParts;
}

function collectImageParts(
  message: RunMessage,
): Array<{ type: "image_url"; image_url: { url: string } }> {
  const parts: Array<{ type: "image_url"; image_url: { url: string } }> = [];
  const pushImagePart = (part: { type: string }) => {
    if (part.type !== "image" || !("image" in part)) {
      return;
    }
    const src = (part as { image: string }).image;
    if (!src) {
      return;
    }
    parts.push({
      type: "image_url",
      image_url: {
        url: src.startsWith("data:") ? src : `data:image/png;base64,${src}`,
      },
    });
  };

  for (const part of message.content ?? []) {
    pushImagePart(part);
  }

  if ("attachments" in message && (message.attachments?.length ?? 0) > 0) {
    for (const attachment of message.attachments ?? []) {
      for (const part of attachment.content ?? []) {
        pushImagePart(part);
      }
    }
  }

  return parts;
}

function normalizeOpenAIReasoningItem(
  value: unknown,
): OpenAIReasoningContentPart | null {
  if (!value || typeof value !== "object") {
    return null;
  }
  const item = value as Record<string, unknown>;
  if (item.type !== "reasoning" || typeof item.id !== "string" || !item.id) {
    return null;
  }
  const summary = Array.isArray(item.summary)
    ? item.summary.flatMap((part) => {
        if (!part || typeof part !== "object") {
          return [];
        }
        const summaryPart = part as Record<string, unknown>;
        return summaryPart.type === "summary_text" &&
          typeof summaryPart.text === "string"
          ? [{ type: "summary_text" as const, text: summaryPart.text }]
          : [];
      })
    : [];
  const normalized: OpenAIReasoningContentPart = {
    type: "reasoning",
    id: item.id,
    summary,
  };
  if (
    item.status === "in_progress" ||
    item.status === "completed" ||
    item.status === "incomplete"
  ) {
    normalized.status = item.status;
  }
  return normalized;
}

function toOpenAIImageEditReferenceMessage(
  reference: PendingImageEditReference,
): OpenAIChatMessage | null {
  if (!reference.openaiImageGenerationCallId) {
    return null;
  }
  const content: Exclude<OpenAIMessageContent, string> = [];
  const reasoningItem = normalizeOpenAIReasoningItem(
    reference.openaiReasoningItem,
  );
  if (reasoningItem) {
    content.push(reasoningItem);
  }
  content.push({
    type: "image_generation_call",
    id: reference.openaiImageGenerationCallId,
    ...(reference.openaiResponseId
      ? { response_id: reference.openaiResponseId }
      : {}),
  });
  return { role: "assistant", content };
}

// Refusal flag stamped on assistant metadata when the backend emits the
// `anthropic_refusal` _toolEvent. We drop the refused pair from the next
// request body (Anthropic: leaving refusals in context keeps refusing).
// Using metadata, not text, prevents content from spoofing a reset.
function isAnthropicRefusalMessage(message: RunMessage): boolean {
  if (message.role !== "assistant") return false;
  const metadata = (message as { metadata?: unknown }).metadata as
    | { custom?: Record<string, unknown> }
    | undefined;
  return metadata?.custom?.anthropicRefusal === true;
}

type SerializedMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: OpenAIMessageContent | null;
  tool_calls?: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
    extra_content?: unknown;
  }>;
  tool_call_id?: string;
  name?: string;
  /**
   * Gemini text-part thoughtSignature stashed during streaming on the
   * last text MessagePart. Backend reads
   * `extra_content.google.thought_signature` and attaches it to the
   * matching Gemini text part on the outbound turn.
   */
  extra_content?: unknown;
};

type SerializedToolCall = NonNullable<SerializedMessage["tool_calls"]>[number];
type SerializedToolResult = {
  role: "tool";
  content: string;
  tool_call_id: string;
  name?: string;
};

type ToolPartReplayMetadata = {
  argsObj: Record<string, unknown> | null;
  argsGoogle: Record<string, unknown> | null;
  hasNativePart: boolean;
  isServerSideBuiltin: boolean;
};

function getToolPartReplayMetadata(
  tc: ToolCallMessagePart,
): ToolPartReplayMetadata {
  const toolNameLower = (tc.toolName ?? "").toLowerCase();
  const argsObj =
    tc.args && typeof tc.args === "object"
      ? (tc.args as Record<string, unknown>)
      : null;
  const argsGoogle =
    argsObj && typeof argsObj.google === "object" && argsObj.google !== null
      ? (argsObj.google as Record<string, unknown>)
      : null;
  const hasNativePart = Boolean(
    argsGoogle &&
      typeof argsGoogle.native_part === "object" &&
      argsGoogle.native_part !== null,
  );
  const hasServerToolMarker = Boolean(
    argsObj && (argsObj as Record<string, unknown>)._server_tool === true,
  );
  return {
    argsObj,
    argsGoogle,
    hasNativePart,
    isServerSideBuiltin: isServerSideBuiltinToolPart(
      toolNameLower,
      argsObj,
      hasServerToolMarker,
      hasNativePart,
    ),
  };
}

type ToolReplayProvenance = {
  source?: string;
  [key: string]: unknown;
};

function getToolReplayProvenance(
  part: ToolCallMessagePart,
): ToolReplayProvenance | null {
  const provenance = (part as { provenance?: unknown }).provenance;
  if (
    !provenance ||
    typeof provenance !== "object" ||
    Array.isArray(provenance)
  ) {
    return null;
  }
  return provenance as ToolReplayProvenance;
}

function hasToolReplayResult(part: ToolCallMessagePart): boolean {
  const result = (part as { result?: unknown }).result;
  return result !== undefined && result !== null;
}

function shouldFlushCompletedLocalToolPair(part: ToolCallMessagePart): boolean {
  const provenance = getToolReplayProvenance(part);
  if (provenance?.source !== "local") {
    return false;
  }
  if (getToolPartReplayMetadata(part).isServerSideBuiltin) {
    return false;
  }
  return hasToolReplayResult(part);
}

function serializeAssistantToolCallPart(
  part: ToolCallMessagePart,
): SerializedToolCall | null {
  const tc = part as ToolCallMessagePart & {
    argsText?: string;
    extra_content?: unknown;
  };
  const { argsGoogle, hasNativePart, isServerSideBuiltin } =
    getToolPartReplayMetadata(tc);

  if (isServerSideBuiltin && !hasNativePart) {
    return null;
  }

  const argumentsStr =
    typeof tc.argsText === "string" && tc.argsText.length > 0
      ? tc.argsText
      : JSON.stringify(tc.args ?? {});
  const entry: SerializedToolCall = {
    id: tc.toolCallId,
    type: "function" as const,
    function: {
      name: tc.toolName ?? "",
      arguments: argumentsStr,
    },
  };
  // Promote args.google to extra_content.google so the backend
  // native_part replay branch can find it. The backend only inspects
  // extra_content, not function.arguments.
  if (tc.extra_content !== undefined) {
    entry.extra_content = tc.extra_content;
  } else if (argsGoogle) {
    entry.extra_content = { google: argsGoogle };
  }
  return entry;
}

export interface McpImageToolResult {
  text: string;
  images: { data: string; mimeType: string }[];
}

export function isMcpImageToolResult(
  val: unknown,
): val is McpImageToolResult {
  if (typeof val !== "object" || val === null) {
    return false;
  }
  const v = val as { text?: unknown; images?: unknown; sessionId?: unknown };
  return (
    typeof v.text === "string" &&
    v.sessionId === undefined &&
    Array.isArray(v.images) &&
    v.images.length > 0 &&
    v.images.every(
      (img: unknown) =>
        typeof img === "object" &&
        img !== null &&
        typeof (img as { data?: unknown }).data === "string" &&
        typeof (img as { mimeType?: unknown }).mimeType === "string",
    )
  );
}

function serializeToolResultPart(
  part: ToolCallMessagePart,
): SerializedToolResult | null {
  const tc = part as ToolCallMessagePart;
  const result = (tc as { result?: unknown }).result;
  const { isServerSideBuiltin } = getToolPartReplayMetadata(tc);

  // Skip provider-side builtins; see isServerSideBuiltinToolPart.
  if (isServerSideBuiltin) {
    return null;
  }
  if (result === undefined || result === null) return null;

  let content: string;
  if (typeof result === "string") {
    // Backend ChatMessage validator rejects role="tool" with empty
    // content; serialise a sentinel JSON so legitimately empty tool
    // outputs still round-trip the follow-up turn to the provider.
    content = result.length > 0 ? result : JSON.stringify({ result: "" });
  } else if (isMcpImageToolResult(result)) {
    content = result.text.length > 0 ? result.text : JSON.stringify({ result: "" });
  } else {
    try {
      content = JSON.stringify(result);
    } catch {
      content = String(result);
    }
  }

  return {
    role: "tool" as const,
    content,
    tool_call_id: tc.toolCallId,
    ...(tc.toolName ? { name: tc.toolName } : {}),
  };
}

function canReplayToolCallWithoutRoleTool(part: ToolCallMessagePart): boolean {
  // Gemini/OpenAI provider-native builtin cards replay through
  // extra_content/native parts and intentionally do not produce role="tool"
  // messages. Local/user tool calls must have a concrete tool result before
  // they are replayed ahead of later assistant text.
  return getToolPartReplayMetadata(part).isServerSideBuiltin;
}

function sanitizeAssistantReplayText(text: string): string {
  return text.replace(
    /data:audio\/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+/g,
    "[audio]",
  );
}

function buildReplayContent(
  textContent: string,
  imageParts: Array<{ type: "image_url"; image_url: { url: string } }>,
): OpenAIMessageContent {
  return imageParts.length > 0
    ? [{ type: "text", text: textContent }, ...imageParts]
    : textContent;
}

function collectAssistantTextThoughtSignature(
  message: RunMessage,
): string | undefined {
  if (!Array.isArray(message.content)) return undefined;
  for (let i = message.content.length - 1; i >= 0; i -= 1) {
    const part = message.content[i] as { type?: string } & Record<
      string,
      unknown
    >;
    if (part?.type !== "text") continue;
    const sig = part._google_thought_signature;
    if (typeof sig === "string" && sig) return sig;
  }
  return undefined;
}

function attachAssistantThoughtSignature(
  messages: SerializedMessage[],
  thoughtSignature: string | undefined,
): void {
  if (!thoughtSignature) return;
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (message.role !== "assistant") continue;
    const extra =
      message.extra_content &&
      typeof message.extra_content === "object" &&
      !Array.isArray(message.extra_content)
        ? (message.extra_content as Record<string, unknown>)
        : {};
    const google =
      extra.google &&
      typeof extra.google === "object" &&
      !Array.isArray(extra.google)
        ? (extra.google as Record<string, unknown>)
        : {};
    message.extra_content = {
      ...extra,
      google: { ...google, thought_signature: thoughtSignature },
    };
    return;
  }
}

function serializeAssistantReplayMessages(
  message: RunMessage,
): SerializedMessage[] {
  if (isAnthropicRefusalMessage(message)) {
    // Prune refused assistant turn from outbound history; the
    // rendered transcript still shows the user-visible notice.
    return [];
  }

  const imageParts = collectImageParts(message);
  const messages: SerializedMessage[] = [];
  const pendingTextParts: string[] = [];
  let pendingToolCalls: SerializedToolCall[] = [];
  let pendingToolResults: SerializedToolResult[] = [];
  let imagePartsPending = imageParts.length > 0;

  const flushAssistantAndToolResults = (force = false): void => {
    const textContent = sanitizeAssistantReplayText(
      pendingTextParts.join("\n"),
    );
    const includeImageParts = imagePartsPending ? imageParts : [];
    const hasContent = textContent.length > 0 || includeImageParts.length > 0;
    const hasToolCalls = pendingToolCalls.length > 0;

    if (!force && !hasContent && !hasToolCalls) {
      return;
    }

    const assistantMessage: SerializedMessage = {
      role: "assistant",
      content: hasContent
        ? buildReplayContent(textContent, includeImageParts)
        : "",
    };
    if (hasToolCalls) {
      assistantMessage.tool_calls = pendingToolCalls;
      // OpenAI requires content === null on assistant turns whose
      // payload is entirely tool_calls (matches the wire shape Gemini
      // expects for the next functionCall replay).
      if (!hasContent) {
        assistantMessage.content = null;
      }
    }

    messages.push(assistantMessage);
    if (pendingToolResults.length > 0) {
      messages.push(...pendingToolResults);
    }

    pendingTextParts.length = 0;
    pendingToolCalls = [];
    pendingToolResults = [];
    imagePartsPending = false;
  };

  for (const part of message.content ?? []) {
    if (part.type === "text") {
      if (pendingToolCalls.length > 0) {
        flushAssistantAndToolResults();
      }
      pendingTextParts.push(part.text);
      continue;
    }

    if (part.type === "tool-call") {
      const toolPart = part as ToolCallMessagePart;
      const toolCall = serializeAssistantToolCallPart(toolPart);
      if (!toolCall) continue;

      const toolResult = serializeToolResultPart(toolPart);
      if (!toolResult && !canReplayToolCallWithoutRoleTool(toolPart)) {
        continue;
      }

      const flushLocalPair = shouldFlushCompletedLocalToolPair(toolPart);
      if (flushLocalPair && pendingToolCalls.length > 0) {
        flushAssistantAndToolResults();
      }

      pendingToolCalls.push(toolCall);
      if (toolResult) {
        pendingToolResults.push(toolResult);
      }

      if (flushLocalPair) {
        flushAssistantAndToolResults();
      }
    }
  }

  flushAssistantAndToolResults(messages.length === 0);
  attachAssistantThoughtSignature(
    messages,
    collectAssistantTextThoughtSignature(message),
  );
  return messages;
}

function toOpenAIMessages(message: RunMessage): SerializedMessage[] {
  if (
    message.role !== "system" &&
    message.role !== "user" &&
    message.role !== "assistant"
  ) {
    return [];
  }

  if (message.role === "assistant") {
    return serializeAssistantReplayMessages(message);
  }

  const textContent = collectTextParts(message).join("\n");
  const imageParts = collectImageParts(message);
  return [
    {
      role: message.role,
      content: buildReplayContent(textContent, imageParts),
    },
  ];
}

function extractImageBase64(input: string): string | undefined {
  if (!input) {
    return undefined;
  }
  if (input.startsWith("data:")) {
    const commaIndex = input.indexOf(",");
    return commaIndex >= 0 ? input.slice(commaIndex + 1) : undefined;
  }
  return input;
}

function findLatestUserImageBase64(messages: RunMessages): string | undefined {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (!message || message.role !== "user") {
      continue;
    }

    // Image in message.content (e.g. compare view).
    for (const part of message.content ?? []) {
      if (part.type === "image" && "image" in part) {
        const encoded = extractImageBase64(part.image);
        if (encoded) return encoded;
      }
    }

    // Image in message.attachments (e.g. chat composer).
    if ("attachments" in message && (message.attachments?.length ?? 0) > 0) {
      for (const attachment of message.attachments ?? []) {
        for (const part of attachment.content ?? []) {
          if (part.type !== "image") {
            continue;
          }
          const encoded = extractImageBase64(part.image);
          if (encoded) {
            return encoded;
          }
        }
      }
    }
  }

  return undefined;
}

function extractAudioPartBase64(
  part: { type: string } | null | undefined,
): string | undefined {
  if (!part || part.type !== "audio" || !("audio" in part)) return undefined;
  const audioPart = (
    part as unknown as {
      type: "audio";
      audio: string | { data: string; format: string };
    }
  ).audio;
  const raw = typeof audioPart === "string" ? audioPart : audioPart?.data;
  if (!raw) return undefined;
  return raw.startsWith("data:") ? raw.split(",")[1] : raw;
}

// Exported for tests.
export function findLatestUserAudioBase64(
  messages: RunMessages,
): string | undefined {
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (!message || message.role !== "user") continue;

    // Message content parts (from compare view's CompareMessagePart with type: "audio")
    for (const part of message.content ?? []) {
      const base64 = extractAudioPartBase64(part);
      if (base64) return base64;
    }

    // Attachment content parts (from AudioAttachmentAdapter)
    if ("attachments" in message) {
      for (const attachment of message.attachments ?? []) {
        for (const part of attachment.content ?? []) {
          const base64 = extractAudioPartBase64(part);
          if (base64) return base64;
        }
      }
    }

    // Only the newest user message counts. audio_base64 switches the
    // backend onto the audio generation path, so replaying audio from an
    // older turn would hijack text follow-ups (Whisper would retranscribe
    // the stale clip). Matches the consumed-on-send semantics of the
    // legacy pendingAudio path.
    break;
  }

  // Runtime store (main composer's audio upload).
  const pendingAudio = useChatRuntimeStore.getState().pendingAudioBase64;
  return pendingAudio ?? undefined;
}

async function resolveUseAdapter(
  threadId: string | undefined,
  options: OpenAIStreamAdapterOptions = {},
): Promise<boolean | undefined> {
  if (options.modelType === "model1" || options.modelType === "model2") {
    return undefined;
  }
  if (
    options.pairId &&
    (options.modelType === "base" || options.modelType === "lora")
  ) {
    return options.modelType === "lora";
  }
  if (!threadId) {
    return undefined;
  }
  try {
    const thread = await getStoredChatThread(threadId);
    if (!thread?.pairId) {
      return undefined;
    }
    // model1/model2 threads skip the adapter toggle — each side loads
    // its own model via /api/inference/load before generation.
    if (thread.modelType === "model1" || thread.modelType === "model2") {
      return undefined;
    }
    return thread.modelType === "lora";
  } catch {
    return undefined;
  }
}

async function resolveProjectInstructions(
  threadId: string | undefined,
): Promise<string> {
  const projectId = await resolveProjectId(threadId);
  if (!projectId) {
    return "";
  }

  const project = await getStoredChatProject(projectId).catch(() => null);
  if (!project || project.archived) {
    return "";
  }
  return project.instructions?.trim() ?? "";
}

async function resolveProjectId(
  threadId: string | undefined,
): Promise<string | null> {
  if (threadId) {
    const thread = await getStoredChatThread(threadId).catch(() => null);
    return thread?.projectId ?? null;
  }
  const projectId = useChatRuntimeStore.getState().activeProjectId;
  if (!projectId) {
    return null;
  }
  return projectId;
}

async function resolveSandboxSessionId(
  threadId: string | undefined,
): Promise<string | undefined> {
  const projectId = await resolveProjectId(threadId);
  return projectId ? `project-${projectId}` : threadId;
}

/** Wait for an in-progress model load to finish (polls store every 500ms). */
function waitForModelReady(abortSignal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    const check = () => {
      if (abortSignal?.aborted) {
        reject(new Error("Aborted"));
        return;
      }
      if (!useChatRuntimeStore.getState().modelLoading) {
        resolve();
        return;
      }
      setTimeout(check, 500);
    };
    check();
  });
}

/**
 * Auto-load the smallest downloaded model when the user chats without
 * selecting one. Prefers GGUF (smallest cached variant), then smallest
 * cached safetensors model.
 */
// Cap cascade so broken cached repos can't spam /api/inference/load.
const MAX_AUTO_LOAD_ATTEMPTS = 3;
const BIG_ENDIAN_GGUF_FILENAME_RE = /(^|[-_])be(?:[._-]|$)/gi;
const GGUF_KNOWN_QUANT_RE =
  /(UD-)?(MXFP[0-9]+(?:_[A-Z0-9]+)*|IQ[0-9]+_[A-Z]+(?:_[A-Z0-9]+)?|TQ[0-9]+_[0-9]+|Q[0-9]+_K_[A-Z]+|Q[0-9]+_[0-9]+|Q[0-9]+_K|BF16|F16|F32)/i;

type AutoLoadCandidate = {
  id: string;
  kind: LastLocalModelKind;
  ggufVariant: string | null;
  maxSeqLength: number;
  successLabel: string;
};

function autoLoadCandidateKey(
  kind: LastLocalModelKind,
  id: string,
  ggufVariant?: string | null,
): string {
  return `${kind}:${id.toLowerCase()}:${(ggufVariant ?? "").toLowerCase()}`;
}

function findCachedRepo<T extends { repo_id: string }>(
  repos: T[],
  id: string,
): T | undefined {
  const normalized = id.toLowerCase();
  return repos.find((repo) => repo.repo_id.toLowerCase() === normalized);
}

function hasBigEndianGgufMarker(filename: string, quant?: string | null): boolean {
  const normalized = filename.replace(/\\/g, "/").toLowerCase();
  const separatorIndex = normalized.lastIndexOf("/");
  const basename = separatorIndex >= 0 ? normalized.slice(separatorIndex + 1) : normalized;
  const parent = separatorIndex >= 0 ? normalized.slice(0, separatorIndex) : "";
  const stem = basename.replace(/\.[^.]*$/, "");
  const quantKey = quant?.trim().toLowerCase() || "";
  const quantIndex = quantKey ? stem.indexOf(quantKey) : -1;
  const quantInParentOnly =
    !!parent &&
    quantIndex < 0 &&
    ((!!quantKey && parent.includes(quantKey)) ||
      (!quantKey && GGUF_KNOWN_QUANT_RE.test(parent)));
  for (const match of stem.matchAll(BIG_ENDIAN_GGUF_FILENAME_RE)) {
    if (quantIndex >= 0 && quantIndex < (match.index ?? 0)) {
      return true;
    }
    const tail = stem.slice((match.index ?? 0) + match[0].length).replace(/^[._-]+/, "");
    if (!tail || !GGUF_KNOWN_QUANT_RE.test(tail)) {
      return !quantInParentOnly;
    }
  }
  return false;
}

function isAutoLoadableGgufVariant(variant: GgufVariantDetail | null): boolean {
  if (!variant?.filename) {
    return false;
  }
  const filename = variant.filename.trim().toLowerCase();
  if (!filename) {
    return false;
  }
  return !hasBigEndianGgufMarker(filename, variant.quant);
}

async function autoLoadSmallestModel(): Promise<{
  loaded: boolean;
  blockedByTrustRemoteCode: boolean;
}> {
  if (await tryAdoptServerActiveModel()) {
    return { loaded: true, blockedByTrustRemoteCode: false };
  }

  const store = useChatRuntimeStore.getState();
  const hfToken = store.hfToken || null;
  const trustRemoteCode = store.params.trustRemoteCode ?? false;
  const specSettings = resolveSpeculativeSettingsForLoad();
  const lastLoaded = readLastLocalModelLoad();
  const toastId = toast("Loading a model…", {
    description: lastLoaded
      ? "Loading last used model."
      : "Auto-selecting the smallest downloaded model.",
    duration: 5000,
    closeButton: true,
  });
  let blockedByTrustRemoteCode = false;
  let hadNonTrustFailure = false;
  let loadAttempts = 0;
  const skippedAutoLoadCandidates = new Set<string>();

  async function canAutoLoad(payload: {
    model_path: string;
    max_seq_length: number;
    is_lora: boolean;
    gguf_variant?: string | null;
    // GGUF-only: scopes the training guard to the same placement policy /load
    // will use. Manual mode must match because it makes placement user-owned.
    // The layer/MoE/split/KV/spec knobs are deliberately not sent: Auto mode's
    // guard sizes conservatively, while Manual mode bypasses that estimate.
    // The safetensors fallback omits both fields and uses HF auto-placement.
    gpu_ids?: number[];
    gpu_memory_mode?: "auto" | "manual";
  }): Promise<boolean> {
    const validation = await validateModel({
      ...payload,
      hf_token: hfToken,
      load_in_4bit: true,
      trust_remote_code: trustRemoteCode,
    });
    // Background auto-load never runs a repo's custom code or loads Hub-flagged unsafe
    // files on its own; both are deferred to the explicit consent dialog instead.
    if (
      validation.requires_trust_remote_code ||
      validation.requires_security_review
    ) {
      blockedByTrustRemoteCode = true;
      return false;
    }
    // Never install packages from a background load; explicit loads raise the upgrade dialog.
    if (validation.requires_transformers_upgrade) {
      hadNonTrustFailure = true;
      return false;
    }
    return true;
  }

  async function loadAutoLoadCandidate(
    candidate: AutoLoadCandidate,
  ): Promise<boolean> {
    if (loadAttempts >= MAX_AUTO_LOAD_ATTEMPTS) {
      return false;
    }
    const currentStore = useChatRuntimeStore.getState();
    // Blobs are saved for GGUF picks only (the sheet gates on it), so don't
    // let a legacy non-GGUF blob feed a stale context/spec choice into a
    // safetensors auto-load.
    const remembered =
      candidate.kind === "gguf"
        ? loadRememberedLoadSettings(
            rememberedLoadSettingsKey({
              id: candidate.id,
              ggufVariant: candidate.ggufVariant,
            }),
          )
        : null;
    const effectiveMaxSeqLength = resolveLoadMaxSeqLength({
      modelId: candidate.id,
      ggufVariant: candidate.ggufVariant,
      isGguf: candidate.kind === "gguf",
      customContextLength: remembered?.contextLength ?? null,
      ggufContextLength: null,
      currentCheckpoint: currentStore.params.checkpoint,
      activeGgufVariant: currentStore.activeGgufVariant,
      maxSeqLength: candidate.maxSeqLength,
      presetSource: currentStore.activePresetSource,
    });
    // The GPU knobs are per-model, so read them from the same remembered
    // settings that fed effectiveMaxSeqLength -- on a background auto-load the
    // live store holds session defaults, not the saved Manual mode / layer pin /
    // GPU pick. Absent fields fall back the way applyRememberedLoadSettings
    // does: the mode to the store (a persisted standing preference), the
    // per-model knobs to their defaults. The saved GPU pick is reconciled
    // against the GPUs present now, like the interactive restore.
    const effectiveGpuMemoryMode =
      remembered?.gpuMemoryMode ?? currentStore.gpuMemoryMode;
    const effectiveGpuLayers = remembered?.gpuLayers ?? GPU_LAYERS_AUTO;
    const effectiveNCpuMoe = remembered?.nCpuMoe ?? 0;
    if (remembered?.selectedGpuIds != null) {
      // Warm the device cache first: on a cold cache the reconcile passes the
      // saved pick through unvalidated, and a stale cross-host pick then fails
      // the load with the picker hidden.
      await ensureGpuDeviceCache();
    }
    const effectiveGpuIds =
      remembered?.selectedGpuIds !== undefined
        ? reconcilePersistedGpuIds(remembered.selectedGpuIds)
        : null;
    // Under Manual GPU memory + Auto layers, llama.cpp's --fit owns context
    // sizing, so send 0 (or the pinned length). GGUF-only; a no-op otherwise.
    // The context pin is per-model too, so it comes from remembered settings,
    // not the live store.
    const fitMaxSeqLength = resolveFitMaxSeqLength(
      candidate.kind === "gguf",
      effectiveGpuMemoryMode,
      effectiveGpuLayers,
      remembered?.contextLength ?? null,
      effectiveMaxSeqLength,
    );
    const effectiveSpeculativeType =
      remembered?.speculativeType ?? specSettings.speculativeType;
    const effectiveSpecDraftNMax =
      remembered?.specDraftNMax ?? specSettings.specDraftNMax;
    if (
      !(await canAutoLoad({
        model_path: candidate.id,
        max_seq_length: fitMaxSeqLength,
        is_lora: false,
        gguf_variant: candidate.ggufVariant,
        // The same remembered-derived GPU pick the load below sends.
        ...(candidate.kind === "gguf"
          ? {
              gpu_ids: effectiveGpuIds ?? undefined,
              gpu_memory_mode: effectiveGpuMemoryMode,
            }
          : {}),
      }))
    ) {
      skippedAutoLoadCandidates.add(
        autoLoadCandidateKey(candidate.kind, candidate.id, candidate.ggufVariant),
      );
      return false;
    }
    loadAttempts += 1;
    const loadResp = await loadModel({
      model_path: candidate.id,
      hf_token: hfToken,
      max_seq_length: fitMaxSeqLength,
      load_in_4bit: true,
      is_lora: false,
      gguf_variant: candidate.ggufVariant,
      trust_remote_code: trustRemoteCode,
      cache_type_kv: remembered?.kvCacheDtype ?? null,
      speculative_type: effectiveSpeculativeType,
      spec_draft_n_max: effectiveSpecDraftNMax,
      tensor_parallel: remembered?.tensorParallel ?? false,
      // GGUF-only: the safetensors fallback loads via HF auto-placement (no
      // explicit pins). The split ratio is deliberately never remembered
      // (positionally bound to an exact GPU set), so auto-load leaves llama.cpp's
      // free-VRAM default in charge rather than sending a stale store value.
      ...(candidate.kind === "gguf"
        ? {
            gpu_memory_mode: effectiveGpuMemoryMode,
            gpu_layers: effectiveGpuLayers,
            n_cpu_moe: effectiveNCpuMoe,
            gpu_ids: effectiveGpuIds ?? undefined,
          }
        : {}),
    });
    saveSpeculativeType(effectiveSpeculativeType);
    // Self-gates on is_gguf (skips diffusion), so persists only for a real GGUF load.
    persistGpuMemoryModeOnLoad(loadResp, effectiveGpuMemoryMode);
    useChatRuntimeStore
      .getState()
      .setCheckpoint(candidate.id, candidate.ggufVariant ?? undefined);
    const store = useChatRuntimeStore.getState();
    store.setModelRequiresTrustRemoteCode(
      loadResp.requires_trust_remote_code ?? false,
    );
    store.setParams({
      ...store.params,
      maxTokens:
        candidate.kind === "gguf"
          ? loadResp.context_length ?? 131072
          : effectiveMaxSeqLength,
    });
    const autoModel: ChatModelSummary = {
      id: candidate.id,
      name: loadResp.display_name ?? candidate.id,
      isVision: loadResp.is_vision ?? false,
      isLora: loadResp.is_lora ?? false,
      isGguf: loadResp.is_gguf ?? candidate.kind === "gguf",
      isAudio: loadResp.is_audio ?? false,
      audioType: loadResp.audio_type ?? null,
      hasAudioInput: loadResp.has_audio_input ?? false,
    };
    if (!store.models.some((m) => m.id === candidate.id)) {
      store.setModels([...store.models, autoModel]);
    }
    if (candidate.kind === "gguf") {
      // Keep an explicit Manual+Auto context pin the load just applied (so a
      // later Apply doesn't silently revert it to auto-fit sizing), mirroring
      // the interactive path's keepCustomCtx; other cases baseline on
      // ggufContextLength.
      const keepCustomCtx = resolveManualAutoCtxPin(
        effectiveGpuMemoryMode,
        effectiveGpuLayers,
        remembered?.contextLength ?? null,
      );
      useChatRuntimeStore.setState({
        ggufContextLength: loadResp.context_length ?? 131072,
        ggufMaxContextLength:
          loadResp.max_context_length ?? loadResp.context_length ?? 131072,
        ggufNativeContextLength: loadResp.native_context_length ?? null,
        supportsReasoning: loadResp.supports_reasoning ?? false,
        reasoningAlwaysOn: loadResp.reasoning_always_on ?? false,
        reasoningEnabled: loadResp.supports_reasoning ?? false,
        ...reasoningCapsFromLoad(loadResp),
        supportsPreserveThinking: loadResp.supports_preserve_thinking ?? false,
        supportsTools: loadResp.supports_tools ?? false,
        ...resolveToolsEnabledOnLoad(loadResp.supports_tools ?? false),
        kvCacheDtype: loadResp.cache_type_kv ?? null,
        loadedKvCacheDtype: loadResp.cache_type_kv ?? null,
        tensorParallel: loadResp.tensor_parallel ?? false,
        loadedTensorParallel: loadResp.tensor_parallel ?? false,
        ...loadedGpuMemoryFieldsUnlessStaged(loadResp, {
          customContextLength: keepCustomCtx,
        }),
        loadedCustomContextLength: keepCustomCtx,
        defaultChatTemplate: loadResp.chat_template ?? null,
        chatTemplateOverride: null,
        loadedChatTemplateOverride: null,
        loadedIsMultimodal: isMultimodalResponse(loadResp),
        loadedIsDiffusion: loadResp.is_diffusion ?? false,
        ...resolveLoadedSpeculativeSettings(loadResp),
      });
    } else {
      useChatRuntimeStore.setState({
        supportsReasoning: loadResp.supports_reasoning ?? false,
        reasoningAlwaysOn: loadResp.reasoning_always_on ?? false,
        reasoningEnabled: loadResp.supports_reasoning ?? false,
        ...reasoningCapsFromLoad(loadResp),
        supportsPreserveThinking: loadResp.supports_preserve_thinking ?? false,
        supportsTools: loadResp.supports_tools ?? false,
        ...resolveToolsEnabledOnLoad(loadResp.supports_tools ?? false),
        kvCacheDtype: loadResp.cache_type_kv ?? null,
        loadedKvCacheDtype: loadResp.cache_type_kv ?? null,
        tensorParallel: loadResp.tensor_parallel ?? false,
        loadedTensorParallel: loadResp.tensor_parallel ?? false,
        // Non-GGUF response: clears any stale GPU baseline a prior manual-GPU
        // GGUF load left, matching the interactive/status sibling load paths.
        ...loadedGpuMemoryFieldsUnlessStaged(loadResp),
        defaultChatTemplate: loadResp.chat_template ?? null,
        chatTemplateOverride: null,
        loadedChatTemplateOverride: null,
        ...resolveLoadedSpeculativeSettings(loadResp),
        loadedIsMultimodal: isMultimodalResponse(loadResp),
        loadedIsDiffusion: loadResp.is_diffusion ?? false,
      });
    }
    if (!(loadResp.is_lora ?? false)) {
      recordLastLocalModelLoad({
        id: candidate.id,
        kind: candidate.kind,
        ggufVariant: candidate.ggufVariant,
      });
    }
    toast.success(candidate.successLabel, { id: toastId });
    return true;
  }
  try {
    const [ggufRepos, modelRepos] = await Promise.all([
      listCachedGguf().catch(() => []),
      listCachedModels().catch(() => []),
    ]);

    if (lastLoaded) {
      if (lastLoaded.kind === "gguf") {
        const repo = findCachedRepo(ggufRepos, lastLoaded.id);
        if (repo && lastLoaded.ggufVariant) {
          try {
            const variants = await listGgufVariants(repo.repo_id);
            const variant = variants.variants.find(
              (entry) =>
                entry.downloaded &&
                entry.quant?.toLowerCase() ===
                  lastLoaded.ggufVariant?.toLowerCase() &&
                isAutoLoadableGgufVariant(entry),
            );
            if (variant) {
              toast("Loading last used model…", {
                id: toastId,
                description: `${repo.repo_id} (${variant.quant})`,
                duration: 5000,
              });
              if (
                await loadAutoLoadCandidate({
                  id: repo.repo_id,
                  kind: "gguf",
                  ggufVariant: variant.quant,
                  maxSeqLength: 0,
                  successLabel: `Loaded ${repo.repo_id} (${variant.quant})`,
                })
              ) {
                return { loaded: true, blockedByTrustRemoteCode: false };
              }
            }
          } catch {
            hadNonTrustFailure = true;
            skippedAutoLoadCandidates.add(
              autoLoadCandidateKey("gguf", repo.repo_id, lastLoaded.ggufVariant),
            );
          }
        }
      } else {
        const repo = findCachedRepo(modelRepos, lastLoaded.id);
        if (repo) {
          try {
            toast("Loading last used model…", {
              id: toastId,
              description: repo.repo_id,
              duration: 5000,
            });
            if (
              await loadAutoLoadCandidate({
                id: repo.repo_id,
                kind: "model",
                ggufVariant: null,
                maxSeqLength: store.params.maxSeqLength,
                successLabel: `Loaded ${repo.repo_id}`,
              })
            ) {
              return { loaded: true, blockedByTrustRemoteCode: false };
            }
          } catch {
            hadNonTrustFailure = true;
            skippedAutoLoadCandidates.add(
              autoLoadCandidateKey("model", repo.repo_id),
            );
          }
        }
      }
      toast("Loading a model…", {
        id: toastId,
        description: "Auto-selecting the smallest downloaded model.",
        duration: 5000,
      });
    }

    // GGUF first: smallest-total-size repo, then its smallest variant.
    if (ggufRepos.length > 0) {
      const sorted = [...ggufRepos].sort((a, b) => a.size_bytes - b.size_bytes);
      for (const repo of sorted) {
        if (loadAttempts >= MAX_AUTO_LOAD_ATTEMPTS) break;
        try {
          const variants = await listGgufVariants(repo.repo_id);
          const downloaded = variants.variants
            .filter((v) => v.downloaded && isAutoLoadableGgufVariant(v))
            .sort((a, b) => a.size_bytes - b.size_bytes);
          if (downloaded.length > 0) {
            const variant = downloaded[0];
            if (
              skippedAutoLoadCandidates.has(
                autoLoadCandidateKey("gguf", repo.repo_id, variant.quant),
              )
            ) {
              continue;
            }
            if (
              await loadAutoLoadCandidate({
                id: repo.repo_id,
                kind: "gguf",
                ggufVariant: variant.quant,
                maxSeqLength: 0,
                successLabel: `Loaded ${repo.repo_id} (${variant.quant})`,
              })
            ) {
              return { loaded: true, blockedByTrustRemoteCode: false };
            }
          }
        } catch {
          hadNonTrustFailure = true;
          continue;
        }
      }
    }

    // Fall back to safetensors models.
    if (modelRepos.length > 0) {
      const sorted = [...modelRepos].sort(
        (a, b) => a.size_bytes - b.size_bytes,
      );
      for (const repo of sorted) {
        if (loadAttempts >= MAX_AUTO_LOAD_ATTEMPTS) break;
        try {
          if (
            skippedAutoLoadCandidates.has(
              autoLoadCandidateKey("model", repo.repo_id),
            )
          ) {
            continue;
          }
          if (
            await loadAutoLoadCandidate({
              id: repo.repo_id,
              kind: "model",
              ggufVariant: null,
              maxSeqLength: 4096,
              successLabel: `Loaded ${repo.repo_id}`,
            })
          ) {
            return { loaded: true, blockedByTrustRemoteCode: false };
          }
        } catch {
          hadNonTrustFailure = true;
          continue;
        }
      }
    }

    // Cap also gates the default download, so total /api/inference/load
    // budget across cached + fallback is MAX_AUTO_LOAD_ATTEMPTS, not +1.
    if (loadAttempts >= MAX_AUTO_LOAD_ATTEMPTS) {
      toast.dismiss(toastId);
      return {
        loaded: false,
        blockedByTrustRemoteCode:
          blockedByTrustRemoteCode && !hadNonTrustFailure,
      };
    }

    // No cached models — try downloading a small default GGUF.
    toast("Downloading a small model…", {
      id: toastId,
      description:
        "No downloaded models found. Fetching Qwen3.5-4B-MTP (UD-Q4_K_XL).",
      duration: 30000,
    });
    try {
      const rt = useChatRuntimeStore.getState();
      if (
        !(await canAutoLoad({
          model_path: "unsloth/Qwen3.5-4B-MTP-GGUF",
          max_seq_length: 0,
          is_lora: false,
          gguf_variant: "UD-Q4_K_XL",
          // The same live-store GPU pick the load below sends (a fresh default
          // model has no remembered settings to prefer).
          gpu_ids: rt.selectedGpuIds ?? undefined,
          gpu_memory_mode: rt.gpuMemoryMode,
        }))
      ) {
        toast.dismiss(toastId);
        return { loaded: false, blockedByTrustRemoteCode };
      }
      loadAttempts += 1;
      const loadResp = await loadModel({
        model_path: "unsloth/Qwen3.5-4B-MTP-GGUF",
        hf_token: hfToken,
        // Model default under both modes: Auto layers + no pin means
        // resolveFitMaxSeqLength returns 0 for every mode (the canAutoLoad
        // preflight above sends the same).
        max_seq_length: 0,
        load_in_4bit: true,
        is_lora: false,
        gguf_variant: "UD-Q4_K_XL",
        trust_remote_code: trustRemoteCode,
        speculative_type: specSettings.speculativeType,
        spec_draft_n_max: specSettings.specDraftNMax,
        // GPU Memory mode is a standing preference, so honor it on auto-load.
        // The layer/MoE/split knobs and the context pin are per-model: the
        // live store may hold edits drafted for a staged pick, and a fresh
        // default model has no remembered settings, so those stay at their
        // defaults like the cached-candidate path. The GPU pick deliberately
        // differs from that path (remembered-or-nothing): it's the picker's
        // current on-screen selection, and the canAutoLoad preflight above
        // already committed to it.
        gpu_memory_mode: rt.gpuMemoryMode,
        gpu_layers: GPU_LAYERS_AUTO,
        n_cpu_moe: 0,
        gpu_ids: rt.selectedGpuIds ?? undefined,
      });
      saveSpeculativeType(specSettings.speculativeType);
      persistGpuMemoryModeOnLoad(loadResp, rt.gpuMemoryMode);
      useChatRuntimeStore
        .getState()
        .setCheckpoint("unsloth/Qwen3.5-4B-MTP-GGUF", "UD-Q4_K_XL");
      const store = useChatRuntimeStore.getState();
      store.setModelRequiresTrustRemoteCode(
        loadResp.requires_trust_remote_code ?? false,
      );
      store.setParams({
        ...store.params,
        maxTokens: loadResp.context_length ?? 131072,
      });
      const defaultModel: ChatModelSummary = {
        id: "unsloth/Qwen3.5-4B-MTP-GGUF",
        name: loadResp.display_name ?? "Qwen3.5-4B-MTP-GGUF",
        isVision: loadResp.is_vision ?? false,
        isLora: false,
        isGguf: true,
      };
      if (!store.models.some((m) => m.id === "unsloth/Qwen3.5-4B-MTP-GGUF")) {
        store.setModels([...store.models, defaultModel]);
      }
      useChatRuntimeStore.setState({
        ggufContextLength: loadResp.context_length ?? 131072,
        ggufMaxContextLength:
          loadResp.max_context_length ?? loadResp.context_length ?? 131072,
        supportsReasoning: loadResp.supports_reasoning ?? false,
        reasoningAlwaysOn: loadResp.reasoning_always_on ?? false,
        reasoningEnabled: loadResp.supports_reasoning ?? false,
        ...reasoningCapsFromLoad(loadResp),
        supportsPreserveThinking: loadResp.supports_preserve_thinking ?? false,
        supportsTools: loadResp.supports_tools ?? false,
        ...resolveToolsEnabledOnLoad(loadResp.supports_tools ?? false),
        kvCacheDtype: loadResp.cache_type_kv ?? null,
        loadedKvCacheDtype: loadResp.cache_type_kv ?? null,
        tensorParallel: loadResp.tensor_parallel ?? false,
        loadedTensorParallel: loadResp.tensor_parallel ?? false,
        ...loadedGpuMemoryFieldsUnlessStaged(loadResp),
        // Drives the GPU Memory controls' diffusion gate; set alongside the
        // GPU fields on every load path so the gate can't read stale.
        loadedIsDiffusion: loadResp.is_diffusion ?? false,
        defaultChatTemplate: loadResp.chat_template ?? null,
        chatTemplateOverride: null,
        loadedIsMultimodal: isMultimodalResponse(loadResp),
        ...resolveLoadedSpeculativeSettings(loadResp),
      });
      recordLastLocalModelLoad({
        id: "unsloth/Qwen3.5-4B-MTP-GGUF",
        kind: "gguf",
        ggufVariant: "UD-Q4_K_XL",
      });
      toast.success("Loaded Qwen3.5-4B-MTP (UD-Q4_K_XL)", { id: toastId });
      return { loaded: true, blockedByTrustRemoteCode: false };
    } catch {
      toast.dismiss(toastId);
      hadNonTrustFailure = true;
      return {
        loaded: false,
        blockedByTrustRemoteCode:
          blockedByTrustRemoteCode && !hadNonTrustFailure,
      };
    }
  } catch {
    toast.dismiss(toastId);
    hadNonTrustFailure = true;
    return {
      loaded: false,
      blockedByTrustRemoteCode: blockedByTrustRemoteCode && !hadNonTrustFailure,
    };
  }
}

export function createOpenAIStreamAdapter(
  options: OpenAIStreamAdapterOptions = {},
): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal, unstable_threadId }) {
      await useChatRuntimeStore.getState().hydratePersistedSettings();
      let runtime = useChatRuntimeStore.getState();
      // Capture the thread ID once so it stays stable even if the user
      // switches chats while waiting for model load / auto-load.
      const resolvedThreadId =
        (unstable_threadId ?? runtime.activeThreadId) || undefined;
      const sandboxSessionId = await resolveSandboxSessionId(resolvedThreadId);
      const toolConfirmationScopeId = resolvedThreadId
        ? `${sandboxSessionId || "_default"}:${resolvedThreadId}`
        : sandboxSessionId || "_default";
      const toolConfirmationIdsByBackendId = new Map<string, string>();
      // Store keys are pane-scoped since local tool ids ("call_0") repeat across
      // turns and concurrent panes (compare mode). Track this run's keys so
      // cleanup can't wipe another pane's.
      const toolOutputPaneScope = toolPaneScope(
        options.modelType,
        options.pairId,
      );
      const scopedToolOutputKey = (id: string) =>
        toolOutputKey(toolOutputPaneScope, id);
      const runToolLiveOutputKeys = new Set<string>();
      const resolvedThreadKey = resolvedThreadId ?? null;
      const pendingImageEditReferenceForRun = runtime.pendingImageEditReference;
      const selectedImageEditReference =
        (pendingImageEditReferenceForRun?.threadId ?? null) ===
        resolvedThreadKey
          ? pendingImageEditReferenceForRun
          : null;
      const clearSelectedImageEditReference = () => {
        if (!selectedImageEditReference) {
          return;
        }
        const store = useChatRuntimeStore.getState();
        const pending = store.pendingImageEditReference;
        if (
          pending?.openaiImageGenerationCallId ===
            selectedImageEditReference.openaiImageGenerationCallId &&
          pending.openaiResponseId ===
            selectedImageEditReference.openaiResponseId &&
          (pending.threadId ?? null) ===
            (selectedImageEditReference.threadId ?? null)
        ) {
          store.clearPendingImageEditReference();
        }
      };

      // Wait for in-progress model load before inferring.
      if (runtime.modelLoading) {
        toast.info("Waiting for model to finish loading…");
        try {
          await waitForModelReady(abortSignal);
        } catch (error) {
          clearSelectedImageEditReference();
          throw error;
        }
      }

      if (!useChatRuntimeStore.getState().params.checkpoint) {
        // Prefer a model already loaded by the CLI/API before auto-loading.
        let loaded: boolean;
        let blockedByTrustRemoteCode: boolean;
        try {
          ({ loaded, blockedByTrustRemoteCode } =
            await autoLoadSmallestModel());
        } catch (error) {
          clearSelectedImageEditReference();
          throw error;
        }
        if (!loaded) {
          toast.error(
            blockedByTrustRemoteCode
              ? "This model needs custom code approval"
              : "No model loaded",
            {
              description: blockedByTrustRemoteCode
                ? "Select it from the top bar to review and approve its custom code, or pick another model."
                : "Pick a model in the top bar, then retry.",
            },
          );
          clearSelectedImageEditReference();
          throw new Error("Load a model first.");
        }
      }

      // Re-read store after auto-load / model-ready wait.
      runtime = useChatRuntimeStore.getState();
      const { params } = runtime;
      const {
        supportsTools,
        toolsEnabled,
        codeToolsEnabled,
        imageToolsEnabled,
        artifactsEnabled,
        mcpEnabledForChat,
        confirmToolCalls,
        bypassPermissions,
        permissionMode,
        webFetchToolsEnabled,
        ragEnabled,
        ragSource,
        ragMode,
        ragTopK,
        ragAutoInject,
        ragAutoInjectMinScore,
      } = runtime;
      // Project sources auto-scope: a chat inside a project retrieves from the
      // project's indexed sources even when the Docs pill is off. The probe is
      // cached, so this is one round trip per project every ~30s at most.
      const ragProjectId = await resolveProjectId(resolvedThreadId);
      const projectRagEnabled = ragProjectId
        ? await projectHasSources(ragProjectId)
        : false;
      const externalSelection = parseExternalModelId(params.checkpoint);
      const isExternalRequest = externalSelection !== null;
      if (
        isExternalRequest &&
        !useExternalProvidersStore.getState().connectionsEnabled
      ) {
        toast.error("Connections are disabled.", {
          description:
            "Turn on Enable connections in Settings → Connections to use hosted models.",
        });
        clearSelectedImageEditReference();
        throw new Error("Connections disabled.");
      }
      const externalProvider = isExternalRequest
        ? loadExternalProviders().find(
            (provider) => provider.id === externalSelection.providerId,
          )
        : null;
      const selectedModelSummary = runtime.models.find(
        (model) => model.id === params.checkpoint,
      );
      const externalApiKey = externalProvider
        ? getExternalProviderApiKey(externalProvider.id).trim()
        : "";

      if (isExternalRequest && !externalProvider) {
        toast.error("Connection not found.", {
          description: "Open Settings → Connections and add it again.",
        });
        clearSelectedImageEditReference();
        throw new Error("Connection not found.");
      }
      // Local providers and custom Gemini bases allow an empty key.
      const externalProviderIsCustom = externalProvider
        ? isCustomProviderType(externalProvider.providerType)
        : false;
      const externalProviderIsGeminiCustomBase = Boolean(
        externalProvider &&
          externalProvider.providerType === "gemini" &&
          isGeminiCustomOpenAICompatBase(externalProvider.baseUrl),
      );
      if (
        isExternalRequest &&
        !externalApiKey &&
        !externalProviderIsCustom &&
        !externalProviderIsGeminiCustomBase
      ) {
        toast.error("Missing API key for selected connection.", {
          description: "Open Settings → Connections and set the API key again.",
        });
        clearSelectedImageEditReference();
        throw new Error("Missing connection API key.");
      }

      // Image-generation flag (OpenAI cloud + Responses-capable model);
      // computed first so Gemini image mode can suppress Search/Code.
      const imageGenerationEnabledForThisTurn = Boolean(
        externalProvider &&
          externalSelection &&
          imageToolsEnabled &&
          providerSupportsBuiltinImageGeneration(
            externalProvider.providerType,
            externalSelection.modelId,
            externalProvider.baseUrl,
          ),
      );
      // Per-model Search/Code allowances live in
      // providerSupportsBuiltin*; this flag just signals image-mode.
      const geminiImageModeForThisTurn =
        externalProvider?.providerType === "gemini" &&
        imageGenerationEnabledForThisTurn;
      const webSearchEnabledForThisTurn = Boolean(
        externalProvider &&
          externalSelection &&
          toolsEnabled &&
          providerSupportsBuiltinWebSearch(
            externalProvider.providerType,
            externalSelection.modelId,
            externalProvider.baseUrl,
          ),
      );
      const codeExecEnabledForThisTurn = Boolean(
        externalProvider &&
          externalSelection &&
          codeToolsEnabled &&
          !geminiImageModeForThisTurn &&
          providerSupportsBuiltinCodeExecution(
            externalProvider.providerType,
            externalSelection.modelId,
            externalProvider.baseUrl,
          ),
      );
      // Fetch pill is independent of Search (Anthropic bills web_fetch
      // separately). Sourced from `webFetchToolsEnabled`; on providers
      // without web_fetch the toggle is forced off in chat-page setState.
      const webFetchEnabledForThisTurn = Boolean(
        externalProvider &&
          webFetchToolsEnabled &&
          providerSupportsBuiltinWebFetch(externalProvider.providerType),
      );
      const providerShipsWebFetch = Boolean(
        externalProvider &&
          providerSupportsBuiltinWebFetch(externalProvider.providerType),
      );

      if (selectedImageEditReference && !imageGenerationEnabledForThisTurn) {
        clearSelectedImageEditReference();
        toast.error("Image editing is unavailable", {
          description:
            "Select an OpenAI image-generation model, then retry the edit.",
        });
        throw new Error("Image generation edit unavailable.");
      }

      // Drop refused assistant turns + their triggering user prompt;
      // otherwise context re-triggers the classifier.
      const survivingMessages: RunMessage[] = [];
      for (const message of messages) {
        if (isAnthropicRefusalMessage(message)) {
          const last = survivingMessages.at(-1);
          if (last && last.role === "user") {
            survivingMessages.pop();
          }
          continue;
        }
        survivingMessages.push(message);
      }

      // toOpenAIMessages emits assistant tool_calls + role="tool"
      // follow-ups; the backend Gemini translator rebuilds the
      // functionCall / functionResponse parts (with thoughtSignature).
      const outboundMessages = survivingMessages
        .flatMap(toOpenAIMessages)
        .filter((message): message is NonNullable<typeof message> =>
          Boolean(message),
        );
      if (selectedImageEditReference) {
        const referenceMessage = toOpenAIImageEditReferenceMessage(
          selectedImageEditReference,
        );
        if (!referenceMessage) {
          clearSelectedImageEditReference();
          toast.error("This generated image cannot be edited", {
            description:
              "The original image reference is missing. Generate the image again, then retry the edit.",
          });
          throw new Error("Generated image edit reference missing.");
        }
        let insertAt = outboundMessages.length;
        for (let i = outboundMessages.length - 1; i >= 0; i -= 1) {
          if (outboundMessages[i]?.role === "user") {
            insertAt = i;
            break;
          }
        }
        // OpenAIChatMessage is a structural superset of SerializedMessage
        // on the role/content axis; cast through unknown since
        // referenceMessage carries no tool_calls (plain assistant turn).
        outboundMessages.splice(
          insertAt,
          0,
          referenceMessage as unknown as SerializedMessage,
        );
      }

      const safeSystemPrompt =
        typeof params.systemPrompt === "string"
          ? resolveSystemPromptVariables(
              params.systemPrompt,
              typeof params.systemVariables === "string"
                ? params.systemVariables
                : "",
            )
          : "";
      const projectInstructions =
        await resolveProjectInstructions(resolvedThreadId);
      const combinedSystemPrompt = [
        projectInstructions
          ? `<project_instructions>\n${projectInstructions}\n</project_instructions>`
          : "",
        safeSystemPrompt.trim(),
      ]
        .filter(Boolean)
        .join("\n\n");
      if (combinedSystemPrompt) {
        outboundMessages.unshift({
          role: "system",
          content: combinedSystemPrompt,
        });
      }
      let disabledToolGuard: string | null = null;
      const disabledToolGuardProviderType = externalProvider?.providerType;
      if (
        disabledToolGuardProviderType === "anthropic" ||
        disabledToolGuardProviderType === "openai"
      ) {
        const webLabel = providerShipsWebFetch
          ? "web search or web fetch"
          : "web search";
        // Treat search and fetch as one "any web tool" axis so the guard
        // only warns when neither pill is on; checking webSearch alone
        // mis-fired when only Fetch was on and suppressed web_fetch.
        const anyWebEnabledForThisTurn =
          webSearchEnabledForThisTurn || webFetchEnabledForThisTurn;
        if (
          !anyWebEnabledForThisTurn &&
          !codeExecEnabledForThisTurn &&
          !imageGenerationEnabledForThisTurn
        ) {
          disabledToolGuard =
            `You do not have ${webLabel}, code execution, or image generation tools in this conversation. ` +
            "Answer from your own knowledge. " +
            "If a request genuinely requires tool use, live data fetch, running code, or image generation, " +
            "inform the user that you do not have access to these capabilities. " +
            "Do not return tool-call syntax inside your response.";
        } else if (!anyWebEnabledForThisTurn && !codeExecEnabledForThisTurn) {
          disabledToolGuard =
            `You do not have ${webLabel} or code execution tools in this conversation. ` +
            "You may still use image generation tools when they are available and useful. " +
            "If a request genuinely requires live data fetch or running code, " +
            "inform the user that you do not have access to these capabilities. " +
            "Do not return tool-call syntax inside your response.";
        } else if (!anyWebEnabledForThisTurn) {
          const availableTools = [
            codeExecEnabledForThisTurn ? "code execution" : null,
            imageGenerationEnabledForThisTurn ? "image generation" : null,
          ].filter(Boolean);
          disabledToolGuard =
            `You do not have ${webLabel} tools in this conversation. ` +
            (availableTools.length > 0
              ? `You may still use ${availableTools.join(" and ")} tools when they are available and useful. `
              : "") +
            "If a request genuinely requires live data fetch or web search tool use, " +
            "inform the user that you do not have access to these capabilities. " +
            "Do not return tool-call syntax inside your response.";
        } else if (!codeExecEnabledForThisTurn) {
          const availableTools = [
            webLabel,
            imageGenerationEnabledForThisTurn ? "image generation" : null,
          ].filter(Boolean);
          disabledToolGuard =
            "You do not have code execution tools in this conversation. " +
            `You may still use ${availableTools.join(" and ")} tools when they are available and useful. ` +
            "If a request genuinely requires running code or code execution tool use, " +
            "inform the user that you do not have access to these capabilities. " +
            "Do not return tool-call syntax inside your response.";
        }
      }
      type OutboundMessage = (typeof outboundMessages)[number];
      function addSystemInstruction(
        targetMessages: OutboundMessage[],
        text: string | null,
      ): void {
        if (!text) return;
        const firstMessage = targetMessages[0];
        if (firstMessage?.role === "system") {
          if (typeof firstMessage.content === "string") {
            targetMessages[0] = {
              ...firstMessage,
              content: `${firstMessage.content}\n\n${text}`,
            };
          } else {
            targetMessages[0] = {
              ...firstMessage,
              content: [
                ...(Array.isArray(firstMessage.content)
                  ? firstMessage.content
                  : []),
                { type: "text", text: `\n\n${text}` },
              ],
            };
          }
          return;
        }
        targetMessages.unshift({ role: "system", content: text });
      }

      // Scan post-prune history so a refused user turn's image/audio
      // doesn't gate or mis-attribute the next turn.
      const imageBase64 = findLatestUserImageBase64(survivingMessages);
      const audioBase64 = findLatestUserAudioBase64(survivingMessages);
      const hasOutboundImage = Boolean(imageBase64);

      // Keep render_html local-only and mirror the backend image-turn gate.
      // Canvas is independent of Search/Code: a local tool-capable model
      // with Canvas on exposes render_html even with no other pills active.
      const renderHtmlToolEnabledForThisTurn = Boolean(
        !isExternalRequest &&
          supportsTools &&
          artifactsEnabled &&
          !hasOutboundImage,
      );
      const artifactInstruction = artifactsEnabled
        ? renderHtmlToolEnabledForThisTurn
          ? "When the user asks for an HTML, CSS, or JavaScript canvas, call render_html once with one complete self-contained HTML document in the code argument. Embed CSS and JavaScript inside the document. After render_html succeeds, do not call it again in the same response unless the user asks for changes. Future user requests for new canvases may call render_html once."
          : "When the user asks for an HTML, CSS, or JavaScript canvas, return one complete self-contained fenced html code block. Embed CSS and JavaScript inside the document. Do not emit tool-call syntax."
        : null;
      const effectiveDisabledToolGuard =
        disabledToolGuard && artifactsEnabled
          ? `${disabledToolGuard} HTML, CSS, or JavaScript canvas requests can still be answered by following the canvas fallback instruction.`
          : disabledToolGuard;
      addSystemInstruction(outboundMessages, effectiveDisabledToolGuard);
      addSystemInstruction(outboundMessages, artifactInstruction);

      // Block when ANY image is in the outbound payload (current or prior
      // turns) and the loaded model can't process images. Once a chat
      // contains an image, a non-vision model can't respond — the user
      // starts a new chat to switch models.
      if (imageBase64) {
        const activeModel = runtime.models.find(
          (m) => m.id === params.checkpoint,
        );
        const imageGateReason = getImageInputUnavailableReason({
          activeModel,
          isExternalModel: isExternalRequest,
          externalSupportsVision: providerTypeSupportsVision(
            externalProvider?.providerType,
          ),
          externalModelLabel: externalSelection?.modelId ?? null,
          loadedIsMultimodal: runtime.loadedIsMultimodal,
          modelLoaded: !!params.checkpoint && !runtime.modelLoading,
          loadError: runtime.lastModelLoadError,
        });
        if (imageGateReason) {
          toast.error(imageGateReason);
          // Flip the per-thread running flag on→off so compare-mode
          // waitForRunEnd resolves instead of hanging: this gate fires
          // before the streaming path's setThreadRunning(true).
          const gatedThreadKey = resolvedThreadId || "__default";
          runtime.setThreadRunning(gatedThreadKey, true);
          runtime.setThreadRunning(gatedThreadKey, false);
          clearSelectedImageEditReference();
          throw new Error(imageGateReason);
        }
      }
      // Clear pending audio from store after extracting (consumed on send).
      if (audioBase64) {
        const audioName = runtime.pendingAudioName;
        if (audioName) {
          const lastUserMsg = [...survivingMessages]
            .reverse()
            .find((m) => m.role === "user");
          if (lastUserMsg) sentAudioNames.set(lastUserMsg.id, audioName);
        }
        runtime.clearPendingAudio();
      }
      const useAdapter = await resolveUseAdapter(resolvedThreadId, options);

      // ── Audio model path (non-streaming) ─────────────────────
      const activeModel = runtime.models.find(
        (m) => m.id === params.checkpoint,
      );
      if (activeModel?.isAudio && !activeModel?.hasAudioInput) {
        const threadKey = resolvedThreadId || "__default";
        runtime.setThreadRunning(threadKey, true);
        try {
          yield {
            content: [{ type: "text" as const, text: "Generating audio..." }],
          };

          const result = await generateAudio(
            {
              model: params.checkpoint,
              messages: outboundMessages,
              stream: false,
              temperature: params.temperature,
              top_p: params.topP,
              max_tokens: params.maxTokens,
              top_k: params.topK,
              min_p: params.minP,
              repetition_penalty: params.repetitionPenalty,
              presence_penalty: params.presencePenalty,
              ...(useAdapter === undefined ? {} : { use_adapter: useAdapter }),
            },
            abortSignal,
          );

          const audioUrl = `data:audio/wav;base64,${result.audio.data}`;
          yield {
            content: [
              {
                type: "text" as const,
                text: `<audio-player src="${audioUrl}" />`,
              },
            ],
          };
        } catch (err) {
          if (!abortSignal.aborted) {
            toast.error("Audio generation failed", {
              description: err instanceof Error ? err.message : "Unknown error",
            });
          }
          throw err;
        } finally {
          runtime.setThreadRunning(threadKey, false);
        }
        return;
      }

      const threadKey = resolvedThreadId || "__default";
      let waitingFirstChunk = true;
      let firstTokenSettled = false;
      const streamStartTime = Date.now();
      let responseModelId = externalSelection?.modelId ?? params.checkpoint;
      let firstTokenTime: number | undefined;
      let totalChunks = 0;
      let resolveFirstToken: (() => void) | null = null;
      let rejectFirstToken: ((err: unknown) => void) | null = null;
      const firstTokenPromise = new Promise<void>((resolve, reject) => {
        resolveFirstToken = resolve;
        rejectFirstToken = reject;
      });
      // Avoid unhandled rejections if toast.promise never attached.
      void firstTokenPromise.catch(() => {});

      function settleFirstTokenOk(): void {
        if (firstTokenSettled) return;
        firstTokenSettled = true;
        resolveFirstToken?.();
      }

      function settleFirstTokenErr(err: unknown): void {
        if (firstTokenSettled) return;
        firstTokenSettled = true;
        rejectFirstToken?.(err);
      }

      const warmupDelayMs = 450;
      const warmupTimer = setTimeout(() => {
        if (!waitingFirstChunk) return;
        if (abortSignal.aborted) return;
        runtime.setGeneratingStatus("waiting");
      }, warmupDelayMs);
      runtime.setThreadRunning(threadKey, true);
      let cumulativeText = "";
      let reasoningStartAt: number | null = null;
      let reasoningDuration = 0;
      // True while wrapping a `delta.reasoning_content` stream in
      // <think>...</think> for parseAssistantContent. Lives outside the
      // SSE loop because the close tag fires when content arrives.
      let reasoningContentOpen = false;
      type ToolCallProvenance = {
        source?: string;
        healed?: boolean;
        forced?: boolean;
        provisional?: boolean;
        duplicate?: boolean;
        reason?: string;
        [key: string]: unknown;
      };
      type PositionedToolCallPart = ToolCallMessagePart & {
        textCursor?: number;
        _delta_index?: number;
        extra_content?: unknown;
        provenance?: ToolCallProvenance;
      };
      // Tool call parts, cumulative; result lands on tool_end.
      const toolCallParts: PositionedToolCallPart[] = [];
      // Raw tool_args accumulator per card: the backend forwards arguments while
      // the model is still WRITING them, and the partial parse below feeds the
      // card's args so the code renders live.
      const liveArgsTextById = new Map<string, string>();
      // Backend tool ids ("call_0", ...) restart every response, so a bare id as
      // store key lets a later turn's stream overwrite the preserved output an
      // earlier still-mounted finished card reads (the tool_start stale-clear
      // only guards the forward direction). Mint one per-run-unique part id per
      // backend id (confirmation ids already synthesize their own) so each card
      // key is unique; every tool_start/output/args/end resolves the same id via
      // this map, dropped at tool_end.
      const toolPartIdByBackendId = new Map<string, string>();
      const resolveToolPartId = (backendToolCallId: string): string => {
        if (!backendToolCallId) {
          return toolCallParts[toolCallParts.length - 1]?.toolCallId ?? "";
        }
        const confirmationId =
          toolConfirmationIdsByBackendId.get(backendToolCallId);
        if (confirmationId) {
          return confirmationId;
        }
        let partId = toolPartIdByBackendId.get(backendToolCallId);
        if (!partId) {
          partId = `${backendToolCallId}:${crypto.randomUUID()}`;
          toolPartIdByBackendId.set(backendToolCallId, partId);
        }
        return partId;
      };
      // Latest Gemini text-part thoughtSignature; pinned onto the final
      // text MessagePart so next-turn replay carries it.
      let latestTextThoughtSignature: string | undefined;
      const pinTextThoughtSignature = <T extends { type: string }>(
        parts: T[],
      ): T[] => {
        if (!latestTextThoughtSignature || parts.length === 0) return parts;
        for (let i = parts.length - 1; i >= 0; i -= 1) {
          if (parts[i].type === "text") {
            parts[i] = {
              ...parts[i],
              _google_thought_signature: latestTextThoughtSignature,
            } as T;
            break;
          }
        }
        return parts;
      };
      const buildAssistantContent = (rawText: string) => {
        const positionedTools = toolCallParts
          .map((part, index) => {
            const cursor = (part as PositionedToolCallPart).textCursor;
            return {
              part,
              index,
              cursor:
                typeof cursor === "number" && Number.isFinite(cursor)
                  ? Math.min(Math.max(cursor, 0), rawText.length)
                  : 0,
            };
          })
          .sort((a, b) => a.cursor - b.cursor || a.index - b.index);

        const assembled: Array<
          ReturnType<typeof parseAssistantContent>[number] | ToolCallMessagePart
        > = [];
        let textCursor = 0;
        let toolIndex = 0;

        const appendTextThrough = (nextCursor: number) => {
          if (nextCursor <= textCursor) return;
          assembled.push(
            ...parseAssistantContent(rawText.slice(textCursor, nextCursor)),
          );
          textCursor = nextCursor;
        };

        while (toolIndex < positionedTools.length) {
          const cursor = positionedTools[toolIndex].cursor;
          appendTextThrough(cursor);
          while (
            toolIndex < positionedTools.length &&
            positionedTools[toolIndex].cursor === cursor
          ) {
            assembled.push(positionedTools[toolIndex].part);
            toolIndex += 1;
          }
        }
        appendTextThrough(rawText.length);

        return pinTextThoughtSignature(assembled);
      };
      const parseToolProvenance = (
        value: unknown,
      ): ToolCallProvenance | undefined => {
        if (!value || typeof value !== "object" || Array.isArray(value)) {
          return undefined;
        }
        return { ...(value as Record<string, unknown>) } as ToolCallProvenance;
      };
      const mergeToolProvenance = (
        existing: ToolCallProvenance | undefined,
        incoming: ToolCallProvenance | undefined,
      ): ToolCallProvenance | undefined => {
        if (!incoming) return existing;
        if (!existing) return incoming;
        const merged: ToolCallProvenance = { ...existing, ...incoming };
        for (const key of [
          "healed",
          "forced",
          "provisional",
          "duplicate",
        ] as const) {
          if (existing[key] === true || incoming[key] === true) {
            merged[key] = true;
          }
        }
        return merged;
      };
      const closeReasoningContent = () => {
        if (!reasoningContentOpen) return;
        cumulativeText += "</think>";
        reasoningContentOpen = false;
      };
      // Anthropic document_citations payload, converted to Sources-panel
      // parts at end-of-stream so inline [N] markers have matching entries.
      const documentCitationParts: Array<{
        type: "source";
        sourceType: "url";
        id: string;
        url: string;
        title: string;
        metadata?: { description: string };
      }> = [];
      // Latched on the `anthropic_refusal` tool event; stamped onto final
      // assistant metadata as `custom.anthropicRefusal` to drive the
      // history-prune above.
      let anthropicRefusalSeen = false;
      let serverMetadata: {
        usage?: ServerUsage;
        timings?: ServerTimings;
      } | null = null;

      // Per-run cancellation token so a delayed stop POST can't match
      // the next run on the same thread.
      const cancelId =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

      // Colab-style proxies can swallow fetch aborts, so also POST
      // /inference/cancel explicitly on abort.
      const onAbortCancel = () => {
        // assistant-ui aborts with AbortError(detach=true) when a thread's runtime
        // unmounts (navigation / background thread switch) and detach=false for an
        // explicit Stop. Only a real Stop cancels the backend run; a detach must
        // leave a backgrounded generation streaming.
        if ((abortSignal.reason as { detach?: boolean } | undefined)?.detach) {
          return;
        }
        const body: Record<string, string> = { cancel_id: cancelId };
        if (sandboxSessionId) body.session_id = sandboxSessionId;
        // Plain fetch, not authFetch: authFetch redirects to login on
        // 401, which would kick the user out mid-stop.
        const token = getAuthToken();
        // Use apiUrl so the cancel POST reaches the right origin in Tauri
        // production builds (webview origin != backend at 127.0.0.1:<port>).
        // Browser/dev builds get the empty base, so the path is unchanged.
        void fetch(apiUrl("/api/inference/cancel"), {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: JSON.stringify(body),
          keepalive: true,
        }).catch(() => {});
      };
      try {
        if (abortSignal.aborted) {
          onAbortCancel();
        } else {
          abortSignal.addEventListener("abort", onAbortCancel, { once: true });
        }

        const {
          supportsReasoning,
          reasoningEnabled,
          reasoningStyle,
          reasoningEffort,
          reasoningEffortLevels,
          supportsPreserveThinking,
          preserveThinking,
        } = runtime;
        const externalBackendProviderType = toExternalBackendProviderType(
          externalProvider?.providerType,
        );
        const buildResponseDetails = (
          finishedAt: number,
        ): ResponseDetailsMetadata => ({
          modelId: params.checkpoint,
          modelLabel:
            (isExternalRequest || responseModelId !== params.checkpoint
              ? responseModelId
              : selectedModelSummary?.name || responseModelId) ||
            params.checkpoint ||
            "Unknown model",
          responseModelId:
            responseModelId ||
            externalSelection?.modelId ||
            params.checkpoint,
          ...(externalProvider?.id ? { providerId: externalProvider.id } : {}),
          providerName:
            externalProvider?.name ??
            (isExternalRequest ? "External provider" : "Local model"),
          providerType: externalProvider?.providerType ?? "local",
          startedAt: streamStartTime,
          finishedAt,
          durationMs: finishedAt - streamStartTime,
          ...(sandboxSessionId ? { sessionId: sandboxSessionId } : {}),
          cancelId,
          toolCalls: Array.from(
            new Set(
              toolCallParts
                .map((part) => part.toolName)
                .filter(
                  (toolName): toolName is string =>
                    typeof toolName === "string" && toolName.length > 0,
                ),
            ),
          ),
          tools: {
            search:
              webSearchEnabledForThisTurn ||
              (!isExternalRequest && supportsTools && toolsEnabled),
            fetch: webFetchEnabledForThisTurn,
            code:
              codeExecEnabledForThisTurn ||
              (!isExternalRequest && supportsTools && codeToolsEnabled),
            images: imageGenerationEnabledForThisTurn,
            mcp: !isExternalRequest && supportsTools && mcpEnabledForChat,
            docs:
              !isExternalRequest &&
              supportsTools &&
              (ragEnabled || projectRagEnabled),
            artifacts: renderHtmlToolEnabledForThisTurn,
            confirmToolCalls,
            bypassPermissions,
            permissionMode,
          },
        });
        const externalCapabilities = getProviderCapabilities(
          externalProvider?.providerType,
        );
        const externalReasoningCaps: ReturnType<
          typeof getExternalReasoningCapabilities
        > =
          externalSelection && externalProvider
            ? getExternalReasoningCapabilities(
                externalProvider.providerType,
                externalSelection.modelId,
                {
                  isReasoningProvider:
                    externalProvider.isReasoningModel === true,
                  baseUrl: externalProvider.baseUrl ?? null,
                },
              )
            : {
                supportsReasoning,
                reasoningStyle,
                reasoningAlwaysOn: false,
                supportsReasoningOff: false,
                reasoningEffortLevels: ["low", "medium", "high"] as const,
              };
        type RequestReasoningEffort = Extract<
          NonNullable<OpenAIChatCompletionsRequest["reasoning_effort"]>,
          "none" | "minimal" | "low" | "medium" | "high" | "max" | "xhigh"
        >;
        const fallbackExternalEffort = (externalReasoningCaps
          .reasoningEffortLevels[0] ?? "low") as RequestReasoningEffort;
        const selectedExternalEffort: RequestReasoningEffort =
          clampReasoningEffortToLevels(
            reasoningEffort,
            externalReasoningCaps.reasoningEffortLevels,
          ) as RequestReasoningEffort;
        // Clamp to the loaded local model's advertised levels so a stale value
        // (e.g. "max" carried over from an external model, or a level this model
        // lacks) becomes one the backend will honor instead of being dropped:
        // gpt-oss-style reasoning_effort gets low|medium|high, GLM-style
        // enable_thinking_effort gets high|max.
        const localReasoningEffort = clampReasoningEffortToLevels(
          reasoningEffort,
          reasoningEffortLevels,
        );
        const externalReasoningEnabled =
          !externalReasoningCaps.supportsReasoningOff ? true : reasoningEnabled;
        const buildRequestPayload = async (
          forceRefreshPublicKey = false,
        ): Promise<OpenAIChatCompletionsRequest> => {
          if (externalSelection && externalProvider) {
            // Per-thread container reuse; empty/undefined falls back to
            // container_auto. Anthropic uses anthropicCodeExecContainerId.
            let openaiCodeExecContainerId: string | null = null;
            let anthropicCodeExecContainerId: string | null = null;
            if (codeExecEnabledForThisTurn && resolvedThreadId) {
              try {
                const thread = await getStoredChatThread(resolvedThreadId);
                openaiCodeExecContainerId =
                  thread?.openaiCodeExecContainerId ?? null;
                anthropicCodeExecContainerId =
                  thread?.anthropicCodeExecContainerId ?? null;
              } catch {
                openaiCodeExecContainerId = null;
                anthropicCodeExecContainerId = null;
              }
              // Pre-send container validation (OpenAI). Stale ids drop
              // silently and fall through to lazy-create; on list-call
              // failure, rely on the backend's retry path.
              let activeContainerIds: Set<string> | null = null;
              if (externalProvider.providerType === "openai") {
                try {
                  const list = await listOpenAIContainers({
                    apiKey: externalApiKey,
                    baseUrl: externalProvider.baseUrl || null,
                  });
                  activeContainerIds = new Set(list.map((c) => c.id));
                } catch {
                  activeContainerIds = null;
                }
                if (
                  activeContainerIds &&
                  openaiCodeExecContainerId &&
                  !activeContainerIds.has(openaiCodeExecContainerId)
                ) {
                  void updateStoredChatThreadEventually(resolvedThreadId, {
                    openaiCodeExecContainerId: null,
                  }).catch(() => {});
                  openaiCodeExecContainerId = null;
                }
              }
              // Cross-thread inheritance: reuse the most recent container
              // from any other thread; opt-out via the picker.
              if (
                !openaiCodeExecContainerId &&
                externalProvider.providerType === "openai"
              ) {
                try {
                  const others = await listStoredChatThreads({
                    includeArchived: true,
                  });
                  for (const t of others) {
                    if (t.id === resolvedThreadId) continue;
                    if (!t.openaiCodeExecContainerId) continue;
                    // Skip ids not in active set; null the source thread so
                    // the next pass doesn't re-pick a dead id.
                    if (
                      activeContainerIds &&
                      !activeContainerIds.has(t.openaiCodeExecContainerId)
                    ) {
                      void updateStoredChatThreadEventually(t.id, {
                        openaiCodeExecContainerId: null,
                      }).catch(() => {});
                      continue;
                    }
                    openaiCodeExecContainerId = t.openaiCodeExecContainerId;
                    void updateStoredChatThreadEventually(resolvedThreadId, {
                      openaiCodeExecContainerId,
                    }).catch(() => {});
                    break;
                  }
                } catch {
                  /* fall through to lazy-create below */
                }
              }
              // Pre-create our own container (vs container_auto) so it shows
              // in the picker with a friendly name and the configured TTL.
              // Falls back to container_auto on failure.
              if (
                !openaiCodeExecContainerId &&
                externalProvider.providerType === "openai"
              ) {
                const ttl = externalProvider.openaiContainerTtlMinutes;
                const ttlToUse = typeof ttl === "number" && ttl >= 1 ? ttl : 20;
                try {
                  const created = await createOpenAIContainer(
                    {
                      apiKey: externalApiKey,
                      baseUrl: externalProvider.baseUrl || null,
                    },
                    {
                      // Friendly English-word name so the container is
                      // human-readable in the picker (e.g. "kestrel-3f9c")
                      // instead of a thread-id slug or blank default.
                      name: pickFriendlyContainerName(),
                      ttlMinutes: ttlToUse,
                    },
                  );
                  openaiCodeExecContainerId = created.id;
                  void updateStoredChatThreadEventually(resolvedThreadId, {
                    openaiCodeExecContainerId: created.id,
                  }).catch(() => {});
                } catch {
                  // Fall back to the backend's container_auto path on
                  // failure — keeps the chat moving (the auto-created
                  // container is unnamed); the next turn can retry.
                  openaiCodeExecContainerId = null;
                }
              }
            }
            return {
              model: externalSelection.modelId,
              messages: outboundMessages,
              stream: true,
              // Reasoning-class models (OpenAI gpt-5.x / o3) reject
              // temperature and top_p; forward only when supported.
              ...(externalCapabilities?.temperature !== false
                ? { temperature: params.temperature }
                : {}),
              ...(externalCapabilities?.topP !== false
                ? { top_p: params.topP }
                : {}),
              // Floor at the provider's documented min (Kimi thinking
              // needs >=16k); clamp at the per-model max.
              max_tokens: Math.min(
                Math.max(
                  params.maxTokens,
                  getExternalMinOutputTokens(externalProvider?.providerType),
                ),
                getExternalMaxOutputTokens(
                  externalProvider?.providerType,
                  externalSelection?.modelId,
                ),
              ),
              // Forward only sampling knobs the provider accepts.
              ...(externalCapabilities?.topK ? { top_k: params.topK } : {}),
              ...(externalCapabilities?.presencePenalty
                ? { presence_penalty: params.presencePenalty }
                : {}),
              // enabled_tools from active pills; backend maps each name
              // to the provider's tool schema.
              ...(webSearchEnabledForThisTurn ||
              webFetchEnabledForThisTurn ||
              codeExecEnabledForThisTurn ||
              imageGenerationEnabledForThisTurn
                ? {
                    enable_tools: true,
                    enabled_tools: [
                      ...(webSearchEnabledForThisTurn ? ["web_search"] : []),
                      ...(webFetchEnabledForThisTurn ? ["web_fetch"] : []),
                      ...(codeExecEnabledForThisTurn ? ["code_execution"] : []),
                      ...(imageGenerationEnabledForThisTurn
                        ? ["image_generation"]
                        : []),
                    ],
                  }
                : {}),
              provider_id: externalProvider.id,
              provider_type: externalBackendProviderType,
              external_model: externalSelection.modelId,
              ...(externalApiKey
                ? {
                    encrypted_api_key: await encryptProviderApiKey(
                      externalApiKey,
                      forceRefreshPublicKey,
                    ),
                  }
                : {}),
              provider_base_url: externalProvider.baseUrl || null,
              ...(openaiCodeExecContainerId
                ? {
                    openai_code_exec_container_id: openaiCodeExecContainerId,
                  }
                : {}),
              ...(anthropicCodeExecContainerId
                ? {
                    anthropic_code_exec_container_id:
                      anthropicCodeExecContainerId,
                  }
                : {}),
              ...(supportsProviderPromptCaching(externalProvider.providerType)
                ? {
                    enable_prompt_caching:
                      externalProvider.enablePromptCaching ?? true,
                  }
                : {}),
              // Anthropic prompt-cache TTL; unknown values no-op on backend.
              ...(supportsProviderPromptCacheTtl(
                externalProvider.providerType,
              ) &&
              (externalProvider.enablePromptCaching ?? true) &&
              isPromptCacheTtl(externalProvider.promptCacheTtl)
                ? { prompt_cache_ttl: externalProvider.promptCacheTtl }
                : {}),
              // Anthropic fast mode (Opus 4.6 / 4.7 only); backend
              // silently drops on unsupported models as a backstop.
              ...(params.fastMode &&
              providerSupportsFastMode(
                externalProvider.providerType,
                externalSelection.modelId,
              )
                ? { fast_mode: true }
                : {}),
              ...(externalReasoningCaps.supportsReasoning
                ? externalReasoningCaps.reasoningStyle === "reasoning_effort"
                  ? externalReasoningEnabled
                    ? { reasoning_effort: selectedExternalEffort }
                    : externalReasoningCaps.supportsReasoningOff
                      ? { reasoning_effort: "none" }
                      : {
                          reasoning_effort: fallbackExternalEffort,
                        }
                  : { thinking: { type: reasoningEnabled ? "enabled" : "disabled" } }
                : {}),
            };
          }

          return {
            model: params.checkpoint,
            messages: outboundMessages,
            stream: true,
            // Opt into the trailing usage chunk so the context-usage bar
            // and tok/s readout populate (backend gates it on include_usage).
            stream_options: { include_usage: true },
            temperature: params.temperature,
            top_p: params.topP,
            max_tokens: params.maxTokens,
            top_k: params.topK,
            min_p: params.minP,
            repetition_penalty: params.repetitionPenalty,
            presence_penalty: params.presencePenalty,
            image_base64: imageBase64,
            audio_base64: audioBase64,
            cancel_id: cancelId,
            ...(sandboxSessionId ? { session_id: sandboxSessionId } : {}),
            ...(resolvedThreadId ? { thread_id: resolvedThreadId } : {}),
            ...(useAdapter === undefined ? {} : { use_adapter: useAdapter }),
            ...(supportsReasoning
              ? reasoningStyle === "enable_thinking_effort"
                ? // GLM-5.2-style: on/off gate plus an effort level. Disabling
                  // sends enable_thinking=false (a real disable); enabling sends
                  // the chosen level (e.g. high|max).
                  reasoningEnabled
                  ? {
                      enable_thinking: true,
                      reasoning_effort: localReasoningEffort,
                    }
                  : { enable_thinking: false }
                : reasoningStyle === "reasoning_effort"
                  ? reasoningEnabled
                    ? { reasoning_effort: localReasoningEffort }
                    : {}
                  : {
                      thinking: {
                        type: reasoningEnabled ? "enabled" : "disabled",
                      },
                    }
              : {}),
            ...(supportsPreserveThinking
              ? { preserve_thinking: preserveThinking }
              : {}),
            // Permission level for local tool calls is sent for every local
            // chat, not only when a tool pill is on: a process policy
            // (unsloth run --enable-tools) can open the tool loop with no pill,
            // and the backend must still see the selected gate. ask/auto request
            // the confirm gate ("auto" only pauses calls flagged unsafe); off
            // and full never prompt, full also drops the sandbox.
            permission_mode: permissionMode,
            confirm_tool_calls:
              permissionMode === "ask" || permissionMode === "auto",
            bypass_permissions: bypassPermissions,
            ...(supportsTools &&
            (toolsEnabled ||
              codeToolsEnabled ||
              renderHtmlToolEnabledForThisTurn ||
              mcpEnabledForChat ||
              ragEnabled ||
              projectRagEnabled)
              ? {
                  enable_tools: true,
                  enabled_tools: [
                    // First so retrieval is the primary tool when Docs is on.
                    ...(ragEnabled || projectRagEnabled
                      ? ["search_knowledge_base"]
                      : []),
                    ...(toolsEnabled ? ["web_search"] : []),
                    ...(codeToolsEnabled ? ["python", "terminal"] : []),
                    ...(renderHtmlToolEnabledForThisTurn
                      ? ["render_html"]
                      : []),
                  ],
                  mcp_enabled: mcpEnabledForChat,
                  // Scope: thread_id = this thread's docs, kb_id = a KB,
                  // project_id = the thread's project sources (auto-on whenever
                  // the project has indexed sources, no Docs pill needed).
                  ...(ragEnabled || projectRagEnabled
                    ? {
                        rag_scope: {
                          ...(ragEnabled && ragSource.type === "kb"
                            ? { kb_id: ragSource.kbId }
                            : {
                                ...(ragEnabled && resolvedThreadId
                                  ? { thread_id: resolvedThreadId }
                                  : {}),
                                ...(projectRagEnabled && ragProjectId
                                  ? { project_id: ragProjectId }
                                  : {}),
                              }),
                          default_top_k: ragTopK,
                          mode: ragMode,
                          autoinject: resolveAutoInject(
                            ragAutoInject,
                            params.checkpoint,
                          ),
                          autoinject_min_score: ragAutoInjectMinScore,

                          ...(ragAutoInject === "off"
                            ? { whole_doc: false }
                            : {}),
                          context_length:
                            runtime.ggufContextLength ?? params.maxSeqLength ?? undefined,
                        },
                      }
                    : {}),
                  auto_heal_tool_calls:
                    useChatRuntimeStore.getState().autoHealToolCalls,
                  nudge_tool_calls: useChatRuntimeStore.getState().nudgeToolCalls,
                  max_tool_calls_per_message:
                    useChatRuntimeStore.getState().maxToolCallsPerMessage,
                  tool_call_timeout: (() => {
                    const mins = useChatRuntimeStore.getState().toolCallTimeout;
                    return mins >= 9999 ? 9999 : mins * 60;
                  })(),
                }
              : {}),
          };
        };

        let retriedWithRefreshedKey = false;
        while (true) {
          try {
            let requestPayload: OpenAIChatCompletionsRequest;
            try {
              requestPayload = await buildRequestPayload(
                retriedWithRefreshedKey,
              );
            } catch (error) {
              clearSelectedImageEditReference();
              throw error;
            }
            clearSelectedImageEditReference();
            await ThreadAutosaveHandle.awaitFirstSave(resolvedThreadId);
            const stream = streamChatCompletions(requestPayload, abortSignal);

            for await (const chunk of stream) {
              const chunkModel = (chunk as { model?: unknown }).model;
              if (typeof chunkModel === "string" && chunkModel.length > 0) {
                responseModelId = chunkModel;
              }

              // Handle tool status events
              const toolStatusText = (
                chunk as unknown as { _toolStatus?: string }
              )._toolStatus;
              if (toolStatusText !== undefined) {
                runtime.setToolStatus(toolStatusText || null);
                continue;
              }

              // Local GGUF sends server-timed reasoning duration. Guard the type
              // so a malformed or proxied chunk (string/null/NaN duration) can
              // never turn the label into NaN.
              const reasoningMs = (
                chunk as { _reasoningDurationMs?: number } | null | undefined
              )?._reasoningDurationMs;
              if (typeof reasoningMs === "number" && Number.isFinite(reasoningMs)) {
                reasoningDuration = Math.max(0, Math.round(reasoningMs / 1000));
                continue;
              }

              // Diffusion frame: a transient canvas snapshot. Route it to the transient
              // store (the in-bubble renderer reads it) and skip it; it has no assistant
              // text, so it never enters the transcript or the counters below.
              const diffusionFrame = (
                chunk as unknown as {
                  _diffusionFrame?: {
                    block?: number;
                    step?: number;
                    total?: number;
                    text?: string;
                  };
                }
              )._diffusionFrame;
              if (diffusionFrame !== undefined) {
                runtime.setActiveDiffusionCanvas({
                  block: diffusionFrame.block ?? 0,
                  step: diffusionFrame.step ?? 0,
                  total: diffusionFrame.total ?? 0,
                  text: diffusionFrame.text ?? "",
                });
                continue;
              }

              // Emit tool-call content parts for assistant-ui.
              // tool_start: add a part (renders "running").
              // tool_end: set result on the part (transitions to "complete").
              const toolEvent = (
                chunk as unknown as { _toolEvent?: Record<string, unknown> }
              )._toolEvent;
              if (toolEvent !== undefined) {
                // Persist container_id onto the thread (OpenAI / Anthropic).
                if (toolEvent.type === "container_ready") {
                  const newContainerId = toolEvent.container_id as
                    | string
                    | undefined;
                  if (newContainerId && resolvedThreadId) {
                    const field =
                      externalProvider?.providerType === "anthropic"
                        ? "anthropicCodeExecContainerId"
                        : "openaiCodeExecContainerId";
                    void updateStoredChatThreadEventually(resolvedThreadId, {
                      [field]: newContainerId,
                    }).catch(() => {});
                  }
                  continue;
                }
                if (toolEvent.type === "document_citations") {
                  // Convert citations_delta footnotes into Sources-panel
                  // entries matching the inline [N] markers.
                  const cits = toolEvent.citations;
                  if (Array.isArray(cits)) {
                    cits.forEach((entry, idx) => {
                      if (!entry || typeof entry !== "object") return;
                      const part = documentCitationToSource(
                        entry as Record<string, unknown>,
                        idx,
                      );
                      if (
                        part &&
                        !documentCitationParts.some((p) => p.id === part.id)
                      ) {
                        documentCitationParts.push(part);
                      }
                    });
                  }
                  continue;
                }
                if (toolEvent.type === "container_invalidated") {
                  if (resolvedThreadId) {
                    const field =
                      externalProvider?.providerType === "anthropic"
                        ? "anthropicCodeExecContainerId"
                        : "openaiCodeExecContainerId";
                    void updateStoredChatThreadEventually(resolvedThreadId, {
                      [field]: null,
                    }).catch(() => {});
                  }
                  continue;
                }
                if (toolEvent.type === "anthropic_refusal") {
                  // Latch the backend refusal signal so final message
                  // metadata can drive the prune.
                  anthropicRefusalSeen = true;
                  continue;
                }
                if (toolEvent.type === "tool_output") {
                  // Incremental stdout from a running tool: append to the live
                  // store so the card renders it while the spinner runs. Final
                  // result arrives via tool_end.
                  const backendToolCallId =
                    (toolEvent.tool_call_id as string) || "";
                  const liveId = resolveToolPartId(backendToolCallId);
                  const liveText =
                    typeof toolEvent.text === "string" ? toolEvent.text : "";
                  if (liveId && liveText) {
                    const liveKey = scopedToolOutputKey(liveId);
                    runToolLiveOutputKeys.add(liveKey);
                    useChatRuntimeStore
                      .getState()
                      .appendToolLiveOutput(liveKey, liveText);
                  }
                  continue;
                }
                if (toolEvent.type === "tool_args") {
                  // The model is still WRITING this call's arguments: accumulate
                  // the raw stream and feed a partial parse into the part's args
                  // so the card shows the code live. tool_start later replaces
                  // args with the authoritative parse.
                  const backendToolCallId =
                    (toolEvent.tool_call_id as string) || "";
                  const liveId = resolveToolPartId(backendToolCallId);
                  const fragment =
                    typeof toolEvent.text === "string" ? toolEvent.text : "";
                  if (liveId && fragment) {
                    const accum =
                      (liveArgsTextById.get(liveId) ?? "") + fragment;
                    liveArgsTextById.set(liveId, accum);
                    const partial = parseLiveToolArgs(accum);
                    const idx = toolCallParts.findIndex(
                      (p) => p.toolCallId === liveId,
                    );
                    if (partial && idx !== -1) {
                      const existing = toolCallParts[
                        idx
                      ] as PositionedToolCallPart;
                      toolCallParts[idx] = {
                        ...existing,
                        args: partial.args as ToolCallMessagePart["args"],
                        argsText: partial.argsText,
                      };
                      yield {
                        content: buildAssistantContent(cumulativeText),
                        metadata: {
                          timing: buildTiming(
                            streamStartTime,
                            totalChunks,
                            firstTokenTime,
                          ),
                          custom: { reasoningDuration },
                        },
                      };
                    }
                  }
                  continue;
                }
                closeReasoningContent();
                const toolProvenance = parseToolProvenance(
                  toolEvent.provenance,
                );
                if (toolEvent.type === "tool_start") {
                  const backendToolCallId =
                    (toolEvent.tool_call_id as string) || "";
                  const approvalId = (toolEvent.approval_id as string) || "";
                  const awaitingConfirmation =
                    toolEvent.awaiting_confirmation === true;
                  const id =
                    awaitingConfirmation && approvalId
                      ? `${toolConfirmationScopeId}:${approvalId}`
                      : backendToolCallId
                        ? resolveToolPartId(backendToolCallId)
                        : approvalId ||
                          `${toolEvent.tool_name}_${Date.now()}`;
                  if (awaitingConfirmation && backendToolCallId) {
                    toolConfirmationIdsByBackendId.set(backendToolCallId, id);
                  }
                  // "call_0" restarts every response: drop stale live/preserved
                  // output under this key, else the card shows the previous call's.
                  const staleKey = scopedToolOutputKey(id);
                  useChatRuntimeStore.getState().clearToolLiveOutput(staleKey);
                  useChatRuntimeStore.getState().clearToolFullOutput(staleKey);
                  const toolArgs = (toolEvent.arguments ??
                    {}) as ToolCallMessagePart["args"];
                  const idx = toolCallParts.findIndex(
                    (p) => p.toolCallId === id,
                  );
                  if (idx !== -1) {
                    const existing = toolCallParts[
                      idx
                    ] as PositionedToolCallPart;
                    toolCallParts[idx] = {
                      ...existing,
                      toolName: toolEvent.tool_name as string,
                      argsText: JSON.stringify(toolArgs),
                      args: toolArgs,
                      provenance: mergeToolProvenance(
                        existing.provenance,
                        toolProvenance,
                      ),
                    };
                  } else {
                    toolCallParts.push({
                      type: "tool-call" as const,
                      toolCallId: id,
                      toolName: toolEvent.tool_name as string,
                      argsText: JSON.stringify(toolArgs),
                      args: toolArgs,
                      textCursor: cumulativeText.length,
                      ...(toolProvenance ? { provenance: toolProvenance } : {}),
                    } as PositionedToolCallPart);
                  }
                  if (awaitingConfirmation) {
                    useChatRuntimeStore
                      .getState()
                      .setToolConfirmation(
                        id,
                        approvalId,
                        sandboxSessionId ?? "",
                        toolConfirmationScopeId,
                      );
                  }
                } else if (toolEvent.type === "tool_end") {
                  const backendToolCallId =
                    (toolEvent.tool_call_id as string) || "";
                  const id = resolveToolPartId(backendToolCallId);
                  if (backendToolCallId) {
                    toolConfirmationIdsByBackendId.delete(backendToolCallId);
                    toolPartIdByBackendId.delete(backendToolCallId);
                  }
                  useChatRuntimeStore.getState().clearToolConfirmation(id);
                  // The result replaces the live output, but if the stream
                  // captured MORE than the truncated result, preserve it so the
                  // finished card keeps everything. Uses the shared predicate,
                  // not a length compare (footer / "Exit code N:" / __IMAGES__
                  // tail can make the result longer by byte).
                  const liveKey = scopedToolOutputKey(id);
                  const liveOutput =
                    useChatRuntimeStore.getState().toolLiveOutput[liveKey] ??
                    "";
                  if (
                    id &&
                    shouldPreserveFullOutput(
                      liveOutput,
                      (toolEvent.result as string) ?? "",
                    )
                  ) {
                    useChatRuntimeStore
                      .getState()
                      .setToolFullOutput(liveKey, liveOutput);
                  }
                  useChatRuntimeStore.getState().clearToolLiveOutput(liveKey);
                  runToolLiveOutputKeys.delete(liveKey);
                  liveArgsTextById.delete(id);
                  const idx = toolCallParts.findIndex(
                    (p) => p.toolCallId === id,
                  );
                  if (idx !== -1) {
                    const rawResult = (toolEvent.result as string) ?? "";
                    const imgMarker = "\n__IMAGES__:";
                    const imgIdx = rawResult.lastIndexOf(imgMarker);
                    const mcpImgMarker = "\n__MCP_IMAGES__:";
                    const mcpImgIdx = rawResult.lastIndexOf(mcpImgMarker);
                    let parsedResult:
                      | string
                      | { text: string; images: string[]; sessionId: string }
                      | McpImageToolResult
                      | {
                          image_b64: string;
                          image_mime: string;
                          size?: string;
                          quality?: string;
                          background?: string;
                          prompt?: string;
                        };
                    const imageB64 = toolEvent.image_b64 as string | undefined;
                    // A valid MCP image envelope wins; an invalid marker falls
                    // through so a sandbox __IMAGES__ suffix still renders and
                    // legit text round-trips unchanged.
                    let mcpImages: McpImageToolResult | null = null;
                    if (mcpImgIdx !== -1) {
                      try {
                        const images = JSON.parse(
                          rawResult.slice(mcpImgIdx + mcpImgMarker.length),
                        );
                        const candidate = {
                          text: rawResult.slice(0, mcpImgIdx),
                          images,
                        };
                        if (isMcpImageToolResult(candidate)) mcpImages = candidate;
                      } catch {
                        // Not a valid envelope; fall through below.
                      }
                    }
                    if (
                      toolCallParts[idx].toolName === "image_generation" &&
                      typeof imageB64 === "string" &&
                      imageB64
                    ) {
                      // Backend keeps base64 on separate image_b64 /
                      // image_mime fields so logs stay small; repackage here.
                      parsedResult = {
                        image_b64: imageB64,
                        image_mime:
                          (toolEvent.image_mime as string | undefined) ??
                          "image/png",
                        size: toolEvent.size as string | undefined,
                        quality: toolEvent.quality as string | undefined,
                        background: toolEvent.background as string | undefined,
                        prompt: toolEvent.prompt as string | undefined,
                      };
                    } else if (mcpImages !== null) {
                      parsedResult = mcpImages;
                    } else if (imgIdx !== -1) {
                      const text = rawResult.slice(0, imgIdx);
                      // Fall back to "_default" to match the backend sandbox
                      // dir used when no session_id (see tools.py _get_workdir).
                      const sessionId = sandboxSessionId || "_default";
                      try {
                        const images = JSON.parse(
                          rawResult.slice(imgIdx + imgMarker.length),
                        ) as string[];
                        parsedResult = { text, images, sessionId };
                      } catch {
                        parsedResult = rawResult;
                      }
                    } else {
                      parsedResult = rawResult;
                    }
                    // Merge tool_end args first, then Gemini native_part.
                    const nextArgs =
                      toolEvent.arguments &&
                      typeof toolEvent.arguments === "object"
                        ? (toolEvent.arguments as ToolCallMessagePart["args"])
                        : undefined;
                    const mergedArgs: ToolCallMessagePart["args"] = {
                      ...(toolCallParts[idx].args ?? {}),
                      ...(nextArgs ?? {}),
                    } as ToolCallMessagePart["args"];
                    // Merge tool_end native_part into args.google so the
                    // outbound translator replays both start (executableCode)
                    // and end (result / inlineData) on the same turn.
                    // Concatenate so each part keeps its own thoughtSignature.
                    const endGoogle = (
                      toolEvent as { google?: { native_part?: unknown } }
                    ).google;
                    if (
                      endGoogle &&
                      typeof endGoogle === "object" &&
                      endGoogle.native_part &&
                      typeof endGoogle.native_part === "object"
                    ) {
                      const argsObj = mergedArgs as Record<string, unknown>;
                      const existingGoogle = (argsObj.google ?? {}) as Record<
                        string,
                        unknown
                      >;
                      const existingNative =
                        (existingGoogle.native_part as Record<
                          string,
                          unknown
                        >) ?? {};
                      const endNative = endGoogle.native_part as Record<
                        string,
                        unknown
                      >;
                      // Extract part entries from parts:[...] or legacy
                      // single-object native_part. Legacy thoughtSignature
                      // always belongs on executableCode.
                      const collectParts = (
                        native: Record<string, unknown>,
                      ): Record<string, unknown>[] => {
                        if (Array.isArray(native.parts)) {
                          return (native.parts as unknown[]).filter(
                            (entry): entry is Record<string, unknown> =>
                              Boolean(entry) &&
                              typeof entry === "object" &&
                              !Array.isArray(entry),
                          );
                        }
                        const out: Record<string, unknown>[] = [];
                        const legacySig =
                          typeof native.thoughtSignature === "string"
                            ? native.thoughtSignature
                            : typeof native.thought_signature === "string"
                              ? (native.thought_signature as string)
                              : null;
                        for (const key of [
                          "executableCode",
                          "codeExecutionResult",
                          "inlineData",
                        ] as const) {
                          const sub = native[key];
                          if (sub && typeof sub === "object") {
                            const entry: Record<string, unknown> = {
                              [key]: sub,
                            };
                            if (key === "executableCode" && legacySig) {
                              entry.thoughtSignature = legacySig;
                            }
                            out.push(entry);
                          }
                        }
                        return out;
                      };
                      const mergedParts = [
                        ...collectParts(existingNative),
                        ...collectParts(endNative),
                      ];
                      argsObj.google = {
                        ...existingGoogle,
                        native_part: { parts: mergedParts },
                      };
                    }
                    const existing = toolCallParts[
                      idx
                    ] as PositionedToolCallPart;
                    toolCallParts[idx] = {
                      ...existing,
                      args: mergedArgs,
                      argsText: JSON.stringify(mergedArgs ?? {}),
                      result: parsedResult,
                      provenance: mergeToolProvenance(
                        existing.provenance,
                        toolProvenance,
                      ),
                    };
                  }
                }
                yield {
                  content: buildAssistantContent(cumulativeText),
                  metadata: {
                    timing: buildTiming(
                      streamStartTime,
                      totalChunks,
                      firstTokenTime,
                    ),
                    custom: { reasoningDuration },
                  },
                };
                continue;
              }

              // OpenAI-standard usage chunk: choices=[], usage populated.
              if (chunk.choices?.length === 0 && chunk.usage) {
                serverMetadata = {
                  usage: chunk.usage,
                  timings: (chunk as Record<string, unknown>).timings as
                    | ServerTimings
                    | undefined,
                };
                continue;
              }

              totalChunks += 1;
              // Latch the chunk's `model` field so the openrouter/free chip
              // shows the chosen underlying model.
              if (
                isExternalRequest &&
                externalProvider?.providerType === "openrouter" &&
                externalSelection?.modelId === "openrouter/free"
              ) {
                const chunkModel = (chunk as { model?: unknown }).model;
                if (
                  typeof chunkModel === "string" &&
                  chunkModel.length > 0 &&
                  chunkModel !== externalSelection.modelId
                ) {
                  const storeState = useChatRuntimeStore.getState();
                  if (storeState.lastOpenRouterChosenModel !== chunkModel) {
                    storeState.setLastOpenRouterChosenModel(chunkModel);
                  }
                }
              }
              const rawDelta = chunk.choices?.[0]?.delta?.content;
              // Normalize structured delta.content (mistral magistral).
              const delta = extractDeltaText(rawDelta);
              // Latest Gemini text-part thoughtSignature for next-turn replay.
              const deltaExtraContent = (
                chunk.choices?.[0]?.delta as
                  | { extra_content?: unknown }
                  | undefined
              )?.extra_content;
              if (deltaExtraContent && typeof deltaExtraContent === "object") {
                const eGoogle = (deltaExtraContent as Record<string, unknown>)
                  .google;
                if (eGoogle && typeof eGoogle === "object") {
                  const sig = (eGoogle as Record<string, unknown>)
                    .thought_signature;
                  if (typeof sig === "string" && sig) {
                    latestTextThoughtSignature = sig;
                  }
                }
              }
              // Kimi / DeepSeek stream thinking via delta.reasoning_content;
              // wrap inline as <think>...</think> for parseAssistantContent.
              const rawReasoning = (
                chunk.choices?.[0]?.delta as
                  | { reasoning_content?: unknown }
                  | undefined
              )?.reasoning_content;
              // OpenRouter ships reasoning as delta.reasoning_details[]
              // regardless of provider; merge into the same wrap path.
              const rawReasoningDetails = (
                chunk.choices?.[0]?.delta as
                  | { reasoning_details?: unknown }
                  | undefined
              )?.reasoning_details;
              const reasoningFromDetails = Array.isArray(rawReasoningDetails)
                ? rawReasoningDetails
                    .map((part) => {
                      if (!part || typeof part !== "object") return "";
                      const text = (part as { text?: unknown }).text;
                      return typeof text === "string" ? text : "";
                    })
                    .join("")
                : "";
              const reasoning =
                (typeof rawReasoning === "string" ? rawReasoning : "") +
                reasoningFromDetails;
              // OpenAI delta.tool_calls: streams fragments by index;
              // accumulate into one part. extra_content carries Gemini 3
              // thoughtSignature for replay.
              const rawDeltaToolCalls = (
                chunk.choices?.[0]?.delta as
                  | { tool_calls?: unknown }
                  | undefined
              )?.tool_calls;
              if (
                Array.isArray(rawDeltaToolCalls) &&
                rawDeltaToolCalls.length > 0
              ) {
                closeReasoningContent();
                for (const tc of rawDeltaToolCalls) {
                  if (!tc || typeof tc !== "object") continue;
                  const call = tc as {
                    id?: string;
                    index?: number;
                    function?: { name?: string; arguments?: string };
                    extra_content?: unknown;
                  };
                  const idx =
                    typeof call.index === "number" ? call.index : undefined;
                  const stableId = call.id;
                  // Match an existing fragment by id first (canonical), then
                  // by index slot; fall back to a minted tool_call_<n> id
                  // for streams that send neither.
                  let existing = stableId
                    ? toolCallParts.find((p) => p.toolCallId === stableId)
                    : undefined;
                  if (!existing && idx !== undefined) {
                    existing = toolCallParts.find(
                      (p) => (p as PositionedToolCallPart)._delta_index === idx,
                    );
                  }
                  const argsFragment = call.function?.arguments ?? "";
                  if (existing) {
                    const prevName = existing.toolName ?? "";
                    const nextName = call.function?.name ?? prevName;
                    const merged = (existing.argsText ?? "") + argsFragment;
                    let parsedArgs: ToolCallMessagePart["args"] =
                      existing.args ?? {};
                    if (merged) {
                      try {
                        parsedArgs = JSON.parse(
                          merged,
                        ) as ToolCallMessagePart["args"];
                      } catch {
                        parsedArgs = {
                          _raw: merged,
                        } as ToolCallMessagePart["args"];
                      }
                    }
                    const prevExtra = (existing as PositionedToolCallPart)
                      .extra_content;
                    const updated: PositionedToolCallPart = {
                      ...(existing as PositionedToolCallPart),
                      toolName: nextName,
                      argsText: merged,
                      args: parsedArgs,
                      ...(call.extra_content !== undefined
                        ? { extra_content: call.extra_content }
                        : prevExtra !== undefined
                          ? { extra_content: prevExtra }
                          : {}),
                      ...(idx !== undefined ? { _delta_index: idx } : {}),
                    };
                    const replaceIdx = toolCallParts.indexOf(existing);
                    if (replaceIdx >= 0) {
                      toolCallParts[replaceIdx] = updated;
                    }
                  } else {
                    const callId =
                      stableId || `tool_call_${idx ?? toolCallParts.length}`;
                    const argsText = argsFragment;
                    let parsedArgs: ToolCallMessagePart["args"] = {};
                    if (argsText) {
                      try {
                        parsedArgs = JSON.parse(
                          argsText,
                        ) as ToolCallMessagePart["args"];
                      } catch {
                        parsedArgs = {
                          _raw: argsText,
                        } as ToolCallMessagePart["args"];
                      }
                    }
                    const fresh: PositionedToolCallPart = {
                      type: "tool-call" as const,
                      toolCallId: callId,
                      toolName: call.function?.name ?? "",
                      argsText,
                      args: parsedArgs,
                      textCursor: cumulativeText.length,
                      ...(call.extra_content !== undefined
                        ? { extra_content: call.extra_content }
                        : {}),
                      ...(idx !== undefined ? { _delta_index: idx } : {}),
                    };
                    toolCallParts.push(fresh);
                  }
                }
                yield {
                  content: buildAssistantContent(cumulativeText),
                  metadata: {
                    timing: buildTiming(
                      streamStartTime,
                      totalChunks,
                      firstTokenTime,
                    ),
                    custom: { reasoningDuration },
                  },
                };
                continue;
              }
              if (!delta && !reasoning) {
                continue;
              }
              if (waitingFirstChunk) {
                waitingFirstChunk = false;
                firstTokenTime = Date.now() - streamStartTime;
                settleFirstTokenOk();
                runtime.setGeneratingStatus(null);
              }

              if (reasoning) {
                if (!reasoningContentOpen) {
                  cumulativeText += `<think>${reasoning}`;
                  reasoningContentOpen = true;
                } else {
                  cumulativeText += reasoning;
                }
              }
              if (delta) {
                closeReasoningContent();
                cumulativeText += delta;
              }
              // Strip a trailing ${...} template-literal fragment from
              // external streams (mistral magistral occasionally emits one).
              if (isExternalRequest) {
                cumulativeText = cumulativeText.replace(
                  /\s*\$\{[^}]*\}\s*$/,
                  "",
                );
              }
              const textParts = parseAssistantContent(cumulativeText);

              // Fallback when no server-side reasoning_summary arrives.
              if (
                textParts.some((part) => part.type === "reasoning") &&
                !reasoningStartAt
              ) {
                reasoningStartAt = Date.now();
              }
              if (
                hasClosedThinkTag(cumulativeText) &&
                reasoningStartAt &&
                !reasoningDuration
              ) {
                reasoningDuration = Math.round(
                  (Date.now() - reasoningStartAt) / 1000,
                );
              }

              if (textParts.length > 0 || toolCallParts.length > 0) {
                yield {
                  content: buildAssistantContent(cumulativeText),
                  metadata: {
                    timing: buildTiming(
                      streamStartTime,
                      totalChunks,
                      firstTokenTime,
                    ),
                    custom: { reasoningDuration },
                  },
                };
              }
            }
            break;
          } catch (streamError) {
            if (
              isExternalRequest &&
              !retriedWithRefreshedKey &&
              isProviderKeyRotationError(streamError)
            ) {
              retriedWithRefreshedKey = true;
              continue;
            }
            throw streamError;
          }
        }
        // If the stream ended while we were still inside a
        // delta.reasoning_content block (Kimi / DeepSeek path), close
        // the open <think> tag so the reasoning panel parses cleanly.
        closeReasoningContent();
        settleFirstTokenOk();

        // Extract source parts from completed web_search and web_fetch
        // calls. Both emit the same `Title:` / `URL:` / `Snippet:` block
        // shape, so the parser need not branch on tool name.
        const sourceParts = toolCallParts.flatMap((tc) => {
          if (
            (tc.toolName !== "web_search" && tc.toolName !== "web_fetch") ||
            !tc.result
          ) {
            return [];
          }
          return parseSourcesFromResult(
            typeof tc.result === "string" ? tc.result : "",
          );
        });

        const meta = serverMetadata;
        const finalTokenCount =
          meta?.usage?.completion_tokens ?? estimateTokenCount(cumulativeText);
        const finalTokPerSec = meta?.timings?.predicted_per_second;
        const serverPromptEvalTime = meta?.timings?.prompt_ms;

        // Prefer llama-server timings; fall back to provider usage envelope.
        const cachedTokens =
          meta?.timings?.cache_n ??
          meta?.usage?.prompt_tokens_details?.cached_tokens ??
          meta?.usage?.cache_read_input_tokens ??
          0;
        // Anthropic-only (billed at the write premium).
        const cacheWriteTokens = meta?.usage?.cache_creation_input_tokens ?? 0;

        // Gate on the captured checkpoint still being active so a late
        // completion from provider A doesn't populate the bar after a
        // mid-stream switch to provider B.
        if (
          meta?.usage &&
          typeof meta.usage.prompt_tokens === "number" &&
          typeof meta.usage.completion_tokens === "number" &&
          typeof meta.usage.total_tokens === "number" &&
          useChatRuntimeStore.getState().params.checkpoint === params.checkpoint
        ) {
          useChatRuntimeStore.getState().setContextUsage({
            promptTokens: meta.usage.prompt_tokens,
            completionTokens: meta.usage.completion_tokens,
            totalTokens: meta.usage.total_tokens,
            cachedTokens,
            cacheWriteTokens,
          });
        }

        const finishedAt = Date.now();
        const finalTiming = buildTiming(
          streamStartTime,
          totalChunks,
          serverPromptEvalTime ?? firstTokenTime,
          finishedAt - streamStartTime,
          finalTokenCount,
          toolCallParts.length,
          finalTokPerSec,
        );

        // Finalize reasoning-only streams.
        if (reasoningStartAt && !reasoningDuration) {
          reasoningDuration = Math.max(
            0,
            Math.round((Date.now() - reasoningStartAt) / 1000),
          );
        }
        yield {
          content: [
            ...buildAssistantContent(cumulativeText),
            ...sourceParts,
            ...documentCitationParts,
          ],
          metadata: {
            timing: finalTiming,
            custom: {
              reasoningDuration,
              // Persisted refusal flag driving the two-pass prune.
              anthropicRefusal: anthropicRefusalSeen || undefined,
              serverTimings: meta?.timings ?? undefined,
              contextUsage: meta?.usage
                ? {
                    promptTokens: meta.usage.prompt_tokens,
                    completionTokens: meta.usage.completion_tokens,
                    totalTokens: meta.usage.total_tokens,
                    cachedTokens,
                    cacheWriteTokens,
                    modelId: params.checkpoint,
                  }
                : undefined,
              responseDetails: buildResponseDetails(finishedAt),
              timing: finalTiming,
            },
          },
        };
      } catch (err) {
        settleFirstTokenErr(
          err instanceof Error ? err : new Error("Generation failed"),
        );
        if (!abortSignal.aborted) {
          const msg = err instanceof Error ? err.message : String(err);
          if (err instanceof StreamInterruptedError) {
            // Connection dropped mid-turn: surface it explicitly (the rethrow
            // below also marks the message with an inline error + Retry).
            toast.error("Response interrupted", {
              description:
                "The connection dropped before the model finished. " +
                "The partial answer is kept. Use Retry to regenerate.",
              duration: 8000,
            });
          } else if (isContextLimitError(msg)) {
            // llama-server runs with --no-context-shift, returning a hard
            // error instead of silently dropping old KV-cache turns. Point
            // the user at the control that raises the ceiling.
            toast.error("Context limit reached", {
              description:
                "The conversation has filled the model's context window. " +
                'Increase "Context Length" in the chat Settings panel (⚙ in the top-right), ' +
                "or start a new chat.",
              duration: 8000,
            });
          } else {
            toast.error("Generation failed", {
              description: msg || "Unknown error",
            });
          }
        }
        throw err;
      } finally {
        abortSignal.removeEventListener("abort", onAbortCancel);
        const confirmStore = useChatRuntimeStore.getState();
        for (const part of toolCallParts) {
          confirmStore.clearToolConfirmation(part.toolCallId);
        }
        runtime.setGeneratingStatus(null);
        runtime.setToolStatus(null);
        // Clear only this run's live keys (a concurrent pane owns its own). A
        // key still here streamed stdout but never reached tool_end (SSE drop or
        // cancel), so promote it to full output first, else the partial
        // diagnostics the user was watching vanish from the card.
        for (const liveKey of runToolLiveOutputKeys) {
          const store = useChatRuntimeStore.getState();
          const liveOutput = store.toolLiveOutput[liveKey] ?? "";
          if (liveOutput) {
            store.setToolFullOutput(liveKey, liveOutput);
          }
          store.clearToolLiveOutput(liveKey);
        }
        runToolLiveOutputKeys.clear();
        // Drop the transient denoising canvas so the finished bubble shows only
        // the committed markdown answer (cancellation/error included).
        runtime.setActiveDiffusionCanvas(null);
        clearTimeout(warmupTimer);
        if (waitingFirstChunk) {
          if (firstTokenSettled) {
            settleFirstTokenOk();
          } else if (abortSignal.aborted) {
            settleFirstTokenErr(new Error("Cancelled"));
          } else {
            settleFirstTokenErr(new Error("No tokens received"));
          }
        }
        runtime.setThreadRunning(threadKey, false);
      }
    },
  };
}
