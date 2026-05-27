// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { getAuthToken } from "@/features/auth";
import { apiUrl } from "@/lib/api-base";
import { toast } from "@/lib/toast";
import type { MessageTiming, ToolCallMessagePart } from "@assistant-ui/core";
import type { ChatModelAdapter } from "@assistant-ui/react";
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
  EXTERNAL_MAX_OUTPUT_TOKENS,
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
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";
import { useExternalProvidersStore } from "../stores/external-providers-store";
import { isMultimodalResponse } from "../types/api";
import type {
  OpenAIChatCompletionsRequest,
  OpenAIChatMessage,
  OpenAIMessageContent,
  OpenAIReasoningContentPart,
} from "../types/api";
import type { ChatModelSummary } from "../types/runtime";
import { getImageInputUnavailableReason } from "../utils/image-input-support";
import {
  getStoredChatThread,
  listStoredChatThreads,
  updateStoredChatThread,
} from "../utils/chat-history-storage";
import {
  hasClosedThinkTag,
  parseAssistantContent,
} from "../utils/parse-assistant-content";
import {
  generateAudio,
  listCachedGguf,
  listCachedModels,
  listGgufVariants,
  loadModel,
  streamChatCompletions,
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

/** Server-side usage data from llama-server (via stream_options.include_usage). */
interface ServerUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  // External prompt-cache fields (see _build_usage_chunk in
  // external_provider.py). cache_creation is Anthropic-only.
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
}

type RunMessages = Parameters<ChatModelAdapter["run"]>[0]["messages"];
type RunMessage = RunMessages[number];

/** Tracks which user messages were sent with an audio file (messageId → filename). */
export const sentAudioNames = new Map<string, string>();

// Synthetic provider-side tool names; backend stamps args._server_tool
// so user functions with the same name aren't dropped. Mirror of
// _SERVER_SIDE_BUILTIN_TOOL_NAMES on the backend.
const SERVER_SIDE_BUILTIN_TOOL_NAMES = new Set<string>([
  "web_search",
  "web_fetch",
  "code_execution",
  "image_generation",
]);

/**
 * Whether a persisted tool-call part is provider-side synthetic and
 * should be stripped from outbound history. Match on the
 * args._server_tool marker or a Gemini native_part payload — no shape
 * heuristic, because user functions can legitimately share a name.
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

/**
 * Match error messages that indicate the request filled or would fill
 * the KV cache, so the UI can show a dedicated toast pointing at the
 * ``Context Length`` setting.
 *
 * Two wordings reach the client and both must hit:
 *
 *   1. The raw llama-server text when ``--no-context-shift`` trips --
 *      "the request exceeds the available context size (N tokens)".
 *   2. The rewritten friendly text emitted by
 *      ``backend/routes/inference.py::_friendly_error`` -- "Message too
 *      long: X tokens exceeds the Y-token context window. Try
 *      increasing the Context Length ..." This is the one most users
 *      see on the streaming GGUF path.
 *
 * We match on substrings rather than full regexes because both layers
 * have drifted across versions (llama.cpp master has tweaked the
 * phrasing; ``_friendly_error`` has gone through several copy edits).
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

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
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
 * Return ``raw`` when it is a safe-to-navigate http(s) URL, or "" otherwise.
 * Rejects non-string input, CR/LF (header injection), and non-http(s)
 * schemes (``javascript:`` / ``data:`` / ``vbscript:``) so provider /
 * tool-controlled strings cannot land in an <a href>.
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
  const source =
    typeof cit.source === "string" && cit.source ? cit.source : "";
  const docTitle =
    (typeof cit.document_title === "string" && cit.document_title) ||
    (typeof cit.title === "string" && cit.title) ||
    "";
  const docIndex =
    typeof cit.document_index === "number" ? cit.document_index : undefined;
  // Only treat ``source`` as a navigable URL when it is real http(s);
  // search_result_location can carry a free-form id (e.g. ``kb-doc-42``)
  // or a hostile ``javascript:`` / ``data:`` / ``vbscript:`` string.
  // Fall back to a stable doc anchor otherwise.
  const url =
    isSafeNavigableSourceUrl(source) || `#anthropic-doc-${docIndex ?? fallbackIdx}`;
  const title = docTitle || source || `Document ${fallbackIdx + 1}`;
  const cited =
    typeof cit.cited_text === "string" ? cit.cited_text.trim() : "";
  // Trim the cited snippet so the Sources panel stays scannable.
  const description =
    cited.length > 240 ? `${cited.slice(0, 240)}...` : cited;
  // Anthropic numbers inline [N] per citation, not per source URL.
  // Fold citation type + position-bearing fields into the id so two
  // distinct citations on the same source (or two search_result_locations
  // with different search_result_index) keep separate Sources entries.
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
      // output is attacker-controllable so a hostile ``javascript:`` /
      // ``data:`` line must not reach the Sources panel <a href>.
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
 * a number of providers now emit it as an array of structured content parts.
 * Concatenating that with `cumulativeText += delta` would stringify each
 * part as `[object Object]` — this function is the guard against that.
 *
 * Handled part shapes:
 *   { type: "text" | "output_text", text | content: "..." }   → text body
 *   { type: "thinking" | "reasoning", thinking | text: "..." } → wrapped as
 *       inline `<think>...</think>` so the downstream parser
 *       (`parseAssistantContent`) lifts it into a reasoning part the same way
 *       it does for providers that emit thinking inline. Without this wrap,
 *       Mistral magistral and similar reasoning-part providers would lose
 *       their thinking panel.
 *
 * Unknown part types are skipped — better to drop a stray field than to
 * stringify an object and pollute the rendered chat with `[object Object]`.
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
// request body (Anthropic guidance: leaving refusals in context keeps
// refusing). Metadata (not text) prevents content from spoofing a reset.
function isAnthropicRefusalMessage(message: RunMessage): boolean {
  if (message.role !== "assistant") return false;
  const metadata = (message as { metadata?: unknown }).metadata as
    | { custom?: Record<string, unknown> }
    | undefined;
  return metadata?.custom?.anthropicRefusal === true;
}

function collectAssistantToolCalls(
  message: RunMessage,
): Array<{
  id: string;
  type: "function";
  function: { name: string; arguments: string };
  extra_content?: unknown;
}> {
  const out: Array<{
    id: string;
    type: "function";
    function: { name: string; arguments: string };
    extra_content?: unknown;
  }> = [];
  for (const part of message.content ?? []) {
    if (part.type !== "tool-call") continue;
    const tc = part as ToolCallMessagePart & {
      argsText?: string;
      extra_content?: unknown;
    };
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
    const isServerSideBuiltin = isServerSideBuiltinToolPart(
      toolNameLower,
      argsObj,
      hasServerToolMarker,
      hasNativePart,
    );
    if (isServerSideBuiltin) {
      // Gemini code_execution / image_generation still need to round-
      // trip the native_part payload for native replay; drop the rest.
      if (!hasNativePart) continue;
    }
    const argumentsStr =
      typeof tc.argsText === "string" && tc.argsText.length > 0
        ? tc.argsText
        : JSON.stringify(tc.args ?? {});
    const entry: {
      id: string;
      type: "function";
      function: { name: string; arguments: string };
      extra_content?: unknown;
    } = {
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
    out.push(entry);
  }
  return out;
}

function collectToolResultMessages(
  message: RunMessage,
): Array<{
  role: "tool";
  content: string;
  tool_call_id: string;
  name?: string;
}> {
  const out: Array<{
    role: "tool";
    content: string;
    tool_call_id: string;
    name?: string;
  }> = [];
  for (const part of message.content ?? []) {
    if (part.type !== "tool-call") continue;
    const tc = part as ToolCallMessagePart;
    const result = (tc as { result?: unknown }).result;
    // Skip provider-side builtins; see isServerSideBuiltinToolPart.
    const argsObj =
      tc.args && typeof tc.args === "object"
        ? (tc.args as Record<string, unknown>)
        : null;
    const argsGoogle =
      argsObj && typeof argsObj.google === "object" && argsObj.google !== null
        ? (argsObj.google as Record<string, unknown>)
        : null;
    const toolNameLower = (tc.toolName ?? "").toLowerCase();
    const hasServerToolMarker = Boolean(
      argsObj && argsObj._server_tool === true,
    );
    const hasNativePart = Boolean(
      argsGoogle &&
        typeof argsGoogle.native_part === "object" &&
        argsGoogle.native_part !== null,
    );
    if (
      isServerSideBuiltinToolPart(
        toolNameLower,
        argsObj,
        hasServerToolMarker,
        hasNativePart,
      )
    ) {
      continue;
    }
    if (result === undefined || result === null) continue;
    let content: string;
    if (typeof result === "string") {
      // Backend ChatMessage validator rejects role="tool" with empty
      // content; serialise a sentinel JSON so legitimately empty tool
      // outputs still round-trip the follow-up turn to the provider.
      content = result.length > 0 ? result : JSON.stringify({ result: "" });
    } else {
      try {
        content = JSON.stringify(result);
      } catch {
        content = String(result);
      }
    }
    out.push({
      role: "tool",
      content,
      tool_call_id: tc.toolCallId,
      ...(tc.toolName ? { name: tc.toolName } : {}),
    });
  }
  return out;
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

function toOpenAIMessages(message: RunMessage): SerializedMessage[] {
  if (
    message.role !== "system" &&
    message.role !== "user" &&
    message.role !== "assistant"
  ) {
    return [];
  }

  let textContent = collectTextParts(message).join("\n");
  if (message.role === "assistant") {
    textContent = textContent.replace(
      /data:audio\/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]+/g,
      "[audio]",
    );
    if (isAnthropicRefusalMessage(message)) {
      // Prune refused assistant turn from outbound history; the
      // rendered transcript still shows the user-visible notice.
      return [];
    }
  }

  const imageParts = collectImageParts(message);
  const toolCalls =
    message.role === "assistant" ? collectAssistantToolCalls(message) : [];
  const toolResults =
    message.role === "assistant" ? collectToolResultMessages(message) : [];

  const base: SerializedMessage = {
    role: message.role,
    content:
      imageParts.length > 0
        ? [{ type: "text", text: textContent }, ...imageParts]
        : textContent,
  };
  if (toolCalls.length > 0) {
    base.tool_calls = toolCalls;
    // OpenAI requires content === null on assistant turns whose
    // payload is entirely tool_calls (matches the wire shape Gemini
    // expects for the next functionCall replay).
    if (!textContent && imageParts.length === 0) {
      base.content = null;
    }
  }
  if (message.role === "assistant") {
    const sig = collectAssistantTextThoughtSignature(message);
    if (sig) {
      base.extra_content = { google: { thought_signature: sig } };
    }
  }

  return toolResults.length > 0 ? [base, ...toolResults] : [base];
}

// Thin singular wrapper: returns only the first serialized message
// (without tool_calls or tool follow-ups) so the OpenAI image-edit
// replay path can map a thread to flat OpenAI chat messages without
// pulling in tool history.
function toOpenAIMessage(message: RunMessage): {
  role: "system" | "user" | "assistant";
  content: OpenAIMessageContent;
} | null {
  const serialized = toOpenAIMessages(message);
  if (serialized.length === 0) return null;
  const first = serialized[0];
  if (
    first.role !== "system" &&
    first.role !== "user" &&
    first.role !== "assistant"
  ) {
    return null;
  }
  if (first.content === null || first.content === undefined) {
    return null;
  }
  if (typeof first.content === "string" && !first.content) {
    return null;
  }
  return {
    role: first.role,
    content: first.content as OpenAIMessageContent,
  };
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

    // Image in message.content (e.g. compare view appends content with image parts)
    for (const part of message.content ?? []) {
      if (part.type === "image" && "image" in part) {
        const encoded = extractImageBase64(part.image);
        if (encoded) return encoded;
      }
    }

    // Image in message.attachments (e.g. chat composer)
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

function findLatestUserAudioBase64(messages: RunMessages): string | undefined {
  // Check message content parts (from compare view's CompareMessagePart with type: "audio")
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const message = messages[i];
    if (!message || message.role !== "user") continue;

    for (const part of message.content ?? []) {
      if (part.type === "audio" && "audio" in part) {
        const audioPart = (
          part as unknown as {
            type: "audio";
            audio: string | { data: string; format: string };
          }
        ).audio;
        const raw = typeof audioPart === "string" ? audioPart : audioPart?.data;
        if (raw) return raw.startsWith("data:") ? raw.split(",")[1] : raw;
      }
    }
  }

  // Check the runtime store (from main composer's audio upload)
  const pendingAudio = useChatRuntimeStore.getState().pendingAudioBase64;
  return pendingAudio ?? undefined;
}

async function resolveUseAdapter(
  threadId: string | undefined,
): Promise<boolean | undefined> {
  if (!threadId) {
    return undefined;
  }
  try {
    const thread = await getStoredChatThread(threadId);
    if (!thread?.pairId) {
      return undefined;
    }
    // model1/model2 threads don't use the adapter toggle — each side
    // loads its own model via /api/inference/load before generation.
    if (thread.modelType === "model1" || thread.modelType === "model2") {
      return undefined;
    }
    return thread.modelType === "lora";
  } catch {
    return undefined;
  }
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
 * Auto-load the smallest downloaded model when the user tries to chat
 * without selecting one. Prefers GGUF (picks smallest cached variant),
 * falls back to smallest cached safetensors model.
 */
// Cap cascade so broken cached repos can't spam /api/inference/load.
const MAX_AUTO_LOAD_ATTEMPTS = 3;

async function autoLoadSmallestModel(): Promise<{
  loaded: boolean;
  blockedByTrustRemoteCode: boolean;
}> {
  const store = useChatRuntimeStore.getState();
  const hfToken = store.hfToken || null;
  const trustRemoteCode = store.params.trustRemoteCode ?? false;
  const toastId = toast("Loading a model…", {
    description: "Auto-selecting the smallest downloaded model.",
    duration: 5000,
    closeButton: true,
  });
  let blockedByTrustRemoteCode = false;
  let hadNonTrustFailure = false;
  let loadAttempts = 0;

  async function canAutoLoad(payload: {
    model_path: string;
    max_seq_length: number;
    is_lora: boolean;
    gguf_variant?: string | null;
  }): Promise<boolean> {
    const validation = await validateModel({
      ...payload,
      hf_token: hfToken,
      load_in_4bit: true,
      trust_remote_code: trustRemoteCode,
    });
    if (validation.requires_trust_remote_code && !trustRemoteCode) {
      blockedByTrustRemoteCode = true;
      return false;
    }
    return true;
  }
  try {
    const [ggufRepos, modelRepos] = await Promise.all([
      listCachedGguf().catch(() => []),
      listCachedModels().catch(() => []),
    ]);

    // Try GGUF first: pick the repo with the smallest total size,
    // then pick its smallest downloaded variant.
    if (ggufRepos.length > 0) {
      const sorted = [...ggufRepos].sort((a, b) => a.size_bytes - b.size_bytes);
      for (const repo of sorted) {
        if (loadAttempts >= MAX_AUTO_LOAD_ATTEMPTS) break;
        try {
          const variants = await listGgufVariants(repo.repo_id);
          const downloaded = variants.variants
            .filter((v) => v.downloaded)
            .sort((a, b) => a.size_bytes - b.size_bytes);
          if (downloaded.length > 0) {
            const variant = downloaded[0];
            if (
              !(await canAutoLoad({
                model_path: repo.repo_id,
                max_seq_length: 0,
                is_lora: false,
                gguf_variant: variant.quant,
              }))
            ) {
              continue;
            }
            loadAttempts += 1;
            const loadResp = await loadModel({
              model_path: repo.repo_id,
              hf_token: hfToken,
              max_seq_length: 0,
              load_in_4bit: true,
              is_lora: false,
              gguf_variant: variant.quant,
              trust_remote_code: trustRemoteCode,
            });
            useChatRuntimeStore
              .getState()
              .setCheckpoint(repo.repo_id, variant.quant);
            const store = useChatRuntimeStore.getState();
            store.setModelRequiresTrustRemoteCode(
              loadResp.requires_trust_remote_code ?? false,
            );
            store.setParams({
              ...store.params,
              maxTokens: loadResp.context_length ?? 131072,
            });
            // Add model to store so the selector shows the name
            const autoModel: ChatModelSummary = {
              id: repo.repo_id,
              name: loadResp.display_name ?? repo.repo_id,
              isVision: loadResp.is_vision ?? false,
              isLora: loadResp.is_lora ?? false,
              isGguf: loadResp.is_gguf ?? false,
              isAudio: loadResp.is_audio ?? false,
              audioType: loadResp.audio_type ?? null,
              hasAudioInput: loadResp.has_audio_input ?? false,
            };
            const existingModels = store.models;
            if (!existingModels.some((m) => m.id === repo.repo_id)) {
              store.setModels([...existingModels, autoModel]);
            }
            useChatRuntimeStore.setState({
              ggufContextLength: loadResp.context_length ?? 131072,
              ggufMaxContextLength:
                loadResp.max_context_length ??
                loadResp.context_length ??
                131072,
              supportsReasoning: loadResp.supports_reasoning ?? false,
              reasoningAlwaysOn: loadResp.reasoning_always_on ?? false,
              reasoningEnabled: loadResp.supports_reasoning ?? false,
              reasoningStyle: loadResp.reasoning_style ?? "enable_thinking",
              supportsPreserveThinking:
                loadResp.supports_preserve_thinking ?? false,
              supportsTools: loadResp.supports_tools ?? false,
              toolsEnabled: loadResp.supports_tools ?? false,
              codeToolsEnabled: loadResp.supports_tools ?? false,
              kvCacheDtype: loadResp.cache_type_kv ?? null,
              loadedKvCacheDtype: loadResp.cache_type_kv ?? null,
              defaultChatTemplate: loadResp.chat_template ?? null,
              chatTemplateOverride: null,
              loadedChatTemplateOverride: null,
              loadedIsMultimodal: isMultimodalResponse(loadResp),
            });
            toast.success(`Loaded ${repo.repo_id} (${variant.quant})`, {
              id: toastId,
            });
            return { loaded: true, blockedByTrustRemoteCode: false };
          }
        } catch {
          hadNonTrustFailure = true;
          continue;
        }
      }
    }

    // Fall back to safetensors models
    if (modelRepos.length > 0) {
      const sorted = [...modelRepos].sort(
        (a, b) => a.size_bytes - b.size_bytes,
      );
      for (const repo of sorted) {
        if (loadAttempts >= MAX_AUTO_LOAD_ATTEMPTS) break;
        try {
          if (
            !(await canAutoLoad({
              model_path: repo.repo_id,
              max_seq_length: 4096,
              is_lora: false,
              gguf_variant: null,
            }))
          ) {
            continue;
          }
          loadAttempts += 1;
          const sfLoadResp = await loadModel({
            model_path: repo.repo_id,
            hf_token: hfToken,
            max_seq_length: 4096,
            load_in_4bit: true,
            is_lora: false,
            gguf_variant: null,
            trust_remote_code: trustRemoteCode,
          });
          useChatRuntimeStore.getState().setCheckpoint(repo.repo_id);
          const store = useChatRuntimeStore.getState();
          store.setModelRequiresTrustRemoteCode(
            sfLoadResp.requires_trust_remote_code ?? false,
          );
          store.setParams({ ...store.params, maxTokens: 4096 });
          useChatRuntimeStore.setState({
            supportsReasoning: sfLoadResp.supports_reasoning ?? false,
            reasoningAlwaysOn: sfLoadResp.reasoning_always_on ?? false,
            reasoningEnabled: sfLoadResp.supports_reasoning ?? false,
            reasoningStyle: sfLoadResp.reasoning_style ?? "enable_thinking",
            supportsPreserveThinking:
              sfLoadResp.supports_preserve_thinking ?? false,
            supportsTools: sfLoadResp.supports_tools ?? false,
            // Parity with the GGUF branch above.
            toolsEnabled: sfLoadResp.supports_tools ?? false,
            codeToolsEnabled: sfLoadResp.supports_tools ?? false,
            defaultChatTemplate: sfLoadResp.chat_template ?? null,
            chatTemplateOverride: null,
            loadedChatTemplateOverride: null,
          });
          const sfModel: ChatModelSummary = {
            id: repo.repo_id,
            name: sfLoadResp.display_name ?? repo.repo_id,
            isVision: sfLoadResp.is_vision ?? false,
            isLora: sfLoadResp.is_lora ?? false,
            isGguf: sfLoadResp.is_gguf ?? false,
          };
          if (!store.models.some((m) => m.id === repo.repo_id)) {
            store.setModels([...store.models, sfModel]);
          }
          useChatRuntimeStore.setState({
            loadedIsMultimodal: isMultimodalResponse(sfLoadResp),
          });
          toast.success(`Loaded ${repo.repo_id}`, { id: toastId });
          return { loaded: true, blockedByTrustRemoteCode: false };
        } catch {
          hadNonTrustFailure = true;
          continue;
        }
      }
    }

    // Cap also gates the default download so the total /api/inference/load
    // budget across cached + fallback is MAX_AUTO_LOAD_ATTEMPTS, not +1.
    if (loadAttempts >= MAX_AUTO_LOAD_ATTEMPTS) {
      toast.dismiss(toastId);
      return {
        loaded: false,
        blockedByTrustRemoteCode:
          blockedByTrustRemoteCode && !hadNonTrustFailure,
      };
    }

    // No cached models found — try downloading a small default GGUF
    toast("Downloading a small model…", {
      id: toastId,
      description:
        "No downloaded models found. Fetching Gemma-4-E2B-it (UD-Q4_K_XL).",
      duration: 30000,
    });
    try {
      if (
        !(await canAutoLoad({
          model_path: "unsloth/gemma-4-E2B-it-GGUF",
          max_seq_length: 0,
          is_lora: false,
          gguf_variant: "UD-Q4_K_XL",
        }))
      ) {
        toast.dismiss(toastId);
        return { loaded: false, blockedByTrustRemoteCode };
      }
      loadAttempts += 1;
      const loadResp = await loadModel({
        model_path: "unsloth/gemma-4-E2B-it-GGUF",
        hf_token: hfToken,
        max_seq_length: 0,
        load_in_4bit: true,
        is_lora: false,
        gguf_variant: "UD-Q4_K_XL",
        trust_remote_code: trustRemoteCode,
      });
      useChatRuntimeStore
        .getState()
        .setCheckpoint("unsloth/gemma-4-E2B-it-GGUF", "UD-Q4_K_XL");
      const store = useChatRuntimeStore.getState();
      store.setModelRequiresTrustRemoteCode(
        loadResp.requires_trust_remote_code ?? false,
      );
      store.setParams({
        ...store.params,
        maxTokens: loadResp.context_length ?? 131072,
      });
      const defaultModel: ChatModelSummary = {
        id: "unsloth/gemma-4-E2B-it-GGUF",
        name: loadResp.display_name ?? "gemma-4-E2B-it-GGUF",
        isVision: loadResp.is_vision ?? false,
        isLora: false,
        isGguf: true,
      };
      if (!store.models.some((m) => m.id === "unsloth/gemma-4-E2B-it-GGUF")) {
        store.setModels([...store.models, defaultModel]);
      }
      useChatRuntimeStore.setState({
        ggufContextLength: loadResp.context_length ?? 131072,
        ggufMaxContextLength:
          loadResp.max_context_length ?? loadResp.context_length ?? 131072,
        supportsReasoning: loadResp.supports_reasoning ?? false,
        reasoningAlwaysOn: loadResp.reasoning_always_on ?? false,
        reasoningEnabled: loadResp.supports_reasoning ?? false,
        reasoningStyle: loadResp.reasoning_style ?? "enable_thinking",
        supportsPreserveThinking: loadResp.supports_preserve_thinking ?? false,
        supportsTools: loadResp.supports_tools ?? false,
        toolsEnabled: loadResp.supports_tools ?? false,
        codeToolsEnabled: loadResp.supports_tools ?? false,
        kvCacheDtype: loadResp.cache_type_kv ?? null,
        loadedKvCacheDtype: loadResp.cache_type_kv ?? null,
        defaultChatTemplate: loadResp.chat_template ?? null,
        chatTemplateOverride: null,
        loadedIsMultimodal: isMultimodalResponse(loadResp),
      });
      toast.success("Loaded Gemma-4-E2B-it (UD-Q4_K_XL)", { id: toastId });
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

export function createOpenAIStreamAdapter(): ChatModelAdapter {
  return {
    async *run({ messages, abortSignal, unstable_threadId }) {
      await useChatRuntimeStore.getState().hydratePersistedSettings();
      let runtime = useChatRuntimeStore.getState();
      // Capture the thread ID once at the start so it stays stable even if
      // the user switches chats while waiting for model load / auto-load.
      const resolvedThreadId =
        (unstable_threadId ?? runtime.activeThreadId) || undefined;
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

      // Wait for in-progress model load to finish before inferring
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
        // Auto-load the smallest downloaded model
        let loaded: boolean;
        let blockedByTrustRemoteCode: boolean;
        try {
          ({ loaded, blockedByTrustRemoteCode } = await autoLoadSmallestModel());
        } catch (error) {
          clearSelectedImageEditReference();
          throw error;
        }
        if (!loaded) {
          toast.error(
            blockedByTrustRemoteCode
              ? "Enable custom code to auto-load this model"
              : "No model loaded",
            {
              description: blockedByTrustRemoteCode
                ? 'Turn on "Enable custom code" in Chat Settings, or pick another model in the top bar.'
                : "Pick a model in the top bar, then retry.",
            },
          );
          clearSelectedImageEditReference();
          throw new Error("Load a model first.");
        }
      }

      // Re-read store after potential auto-load / model ready wait
      runtime = useChatRuntimeStore.getState();
      const { params } = runtime;
      const {
        supportsTools,
        toolsEnabled,
        codeToolsEnabled,
        imageToolsEnabled,
        mcpEnabledForChat,
        webFetchToolsEnabled,
      } = runtime;
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

      // Image-generation flag (OpenAI cloud + Responses-capable model).
      // Computed first so Gemini image mode can suppress Search/Code.
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
      // separately from web_search). Sourced from `webFetchToolsEnabled`;
      // on providers without web_fetch the toggle is forced off in
      // chat-page's runtime setState.
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
      // functionCall/functionResponse parts (with thoughtSignature).
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
        // for the role/content axis the outbound pipeline consumes; cast
        // through unknown since referenceMessage carries no tool_calls
        // (the image_edit reference is a plain assistant turn).
        outboundMessages.splice(
          insertAt,
          0,
          referenceMessage as unknown as SerializedMessage,
        );
      }

      const safeSystemPrompt =
        typeof params.systemPrompt === "string" ? params.systemPrompt : "";
      if (safeSystemPrompt.trim()) {
        outboundMessages.unshift({
          role: "system",
          content: safeSystemPrompt.trim(),
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
        // Treat search and fetch as a single "any web tool" axis so
        // the guard only warns when neither pill is on; checking
        // webSearchEnabledForThisTurn alone mis-fired when only Fetch
        // was on and suppressed live web_fetch calls.
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
      if (disabledToolGuard) {
        const firstMessage = outboundMessages[0];
        if (firstMessage?.role === "system") {
          if (typeof firstMessage.content === "string") {
            outboundMessages[0] = {
              ...firstMessage,
              content: `${firstMessage.content}\n\n${disabledToolGuard}`,
            };
          } else {
            outboundMessages[0] = {
              ...firstMessage,
              content: [
                ...(Array.isArray(firstMessage.content)
                  ? firstMessage.content
                  : []),
                { type: "text", text: `\n\n${disabledToolGuard}` },
              ],
            };
          }
        } else {
          outboundMessages.unshift({
            role: "system",
            content: disabledToolGuard,
          });
        }
      }
      // Scan post-prune history so a refused user turn's image/audio
      // doesn't gate or mis-attribute the next non-refused turn.
      const imageBase64 = findLatestUserImageBase64(survivingMessages);
      const audioBase64 = findLatestUserAudioBase64(survivingMessages);

      // Block when ANY image is in the outbound payload (current or
      // prior turns) and the loaded model can't process images. Keeps
      // the gate simple: once a chat contains an image, a non-vision
      // model can't respond — user starts a new chat to switch models.
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
        });
        if (imageGateReason) {
          toast.error(imageGateReason);
          // Flip the per-thread running flag on→off so the compare-mode
          // waitForRunEnd resolves instead of hanging. This gate fires
          // before the streaming path's setThreadRunning(true), so the
          // wait promise would otherwise never settle.
          const gatedThreadKey = resolvedThreadId || "__default";
          runtime.setThreadRunning(gatedThreadKey, true);
          runtime.setThreadRunning(gatedThreadKey, false);
          clearSelectedImageEditReference();
          throw new Error(imageGateReason);
        }
      }
      // Clear pending audio from store after extracting (consumed on send)
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
      const useAdapter = await resolveUseAdapter(resolvedThreadId);

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
      // <think>...</think> for parseAssistantContent. Lives outside
      // the SSE loop because the close tag fires when content arrives.
      let reasoningContentOpen = false;
      // Tool call parts, cumulative; result lands on tool_end.
      const toolCallParts: ToolCallMessagePart[] = [];
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
      const orderAssistantContent = (
        textParts: ReturnType<typeof parseAssistantContent>,
      ) => {
        const imageToolParts = toolCallParts.filter(
          (part) => part.toolName === "image_generation",
        );
        const otherToolParts = toolCallParts.filter(
          (part) => part.toolName !== "image_generation",
        );
        return [...otherToolParts, ...textParts, ...imageToolParts];
      };
      // Anthropic document_citations tool_event payload, converted to
      // Sources-panel source parts at end-of-stream so the inline [N]
      // markers have matching entries.
      const documentCitationParts: Array<{
        type: "source";
        sourceType: "url";
        id: string;
        url: string;
        title: string;
        metadata?: { description: string };
      }> = [];
      // Latched on the `anthropic_refusal` tool event; stamped onto the
      // final assistant metadata as `custom.anthropicRefusal` to drive
      // the history-prune above.
      let anthropicRefusalSeen = false;
      let serverMetadata: {
        usage?: ServerUsage;
        timings?: ServerTimings;
      } | null = null;

      // Per-run cancellation token so a delayed stop POST cannot match
      // the next run on the same thread.
      const cancelId =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(36).slice(2)}`;

      // Colab-style proxies can swallow fetch aborts, so also POST
      // /inference/cancel explicitly on abort.
      const onAbortCancel = () => {
        const body: Record<string, string> = { cancel_id: cancelId };
        if (resolvedThreadId) body.session_id = resolvedThreadId;
        // Plain fetch, not authFetch: authFetch redirects to login on
        // 401, which would kick the user out mid-stop.
        const token = getAuthToken();
        // Use apiUrl so the cancel POST reaches the right origin in
        // Tauri production builds (where the webview origin is not the
        // backend at 127.0.0.1:<port>). Browser/dev builds get the empty
        // base, so the path is unchanged there.
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
          supportsPreserveThinking,
          preserveThinking,
        } = runtime;
        const externalBackendProviderType = toExternalBackendProviderType(
          externalProvider?.providerType,
        );
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
        const localReasoningEffort =
          reasoningEffort === "low" ||
          reasoningEffort === "medium" ||
          reasoningEffort === "high"
            ? reasoningEffort
            : "low";
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
              // silently and fall through to lazy-create. On list-call
              // failure, skip and rely on the backend's retry path.
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
              // Cross-thread inheritance: reuse the most recently used
              // container from any other thread; opt-out via the picker.
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
                    // Skip ids not in active set; null on source thread so
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
              // Pre-create our own container (vs container_auto) so it
              // shows up in the picker with a friendly name and the
              // configured TTL. Falls back to container_auto on failure.
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
                      // Friendly English-word name so the container
                      // is human-readable in the picker list (e.g.
                      // "kestrel-3f9c") instead of a thread-id slug
                      // or OpenAI's default blank name.
                      name: pickFriendlyContainerName(),
                      ttlMinutes: ttlToUse,
                    },
                  );
                  openaiCodeExecContainerId = created.id;
                  void updateStoredChatThreadEventually(resolvedThreadId, {
                    openaiCodeExecContainerId: created.id,
                  }).catch(() => {});
                } catch {
                  // Fall back to backend's container_auto path on
                  // failure — keeps the chat moving; the next turn
                  // can retry. The auto-created container will be
                  // unnamed, but the chat doesn't break.
                  openaiCodeExecContainerId = null;
                }
              }
            }
            return {
              model: externalSelection.modelId,
              messages: outboundMessages,
              stream: true,
              // Reasoning-class models (OpenAI gpt-5.x / o3) reject temperature
              // and top_p; only forward when the active provider supports them.
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
              // Compose the enabled_tools list from the active pills;
              // backend maps each name to the provider's tool schema.
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
              // silently drops on unsupported models as a second
              // line of defence.
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
                  : { enable_thinking: reasoningEnabled }
                : {}),
            };
          }

          return {
            model: params.checkpoint,
            messages: outboundMessages,
            stream: true,
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
            ...(resolvedThreadId ? { session_id: resolvedThreadId } : {}),
            ...(useAdapter === undefined ? {} : { use_adapter: useAdapter }),
            ...(supportsReasoning
              ? reasoningStyle === "reasoning_effort"
                ? reasoningEnabled
                  ? { reasoning_effort: localReasoningEffort }
                  : {}
                : { enable_thinking: reasoningEnabled }
              : {}),
            ...(supportsPreserveThinking
              ? { preserve_thinking: preserveThinking }
              : {}),
            ...(supportsTools && (toolsEnabled || codeToolsEnabled || mcpEnabledForChat)
              ? {
                  enable_tools: true,
                  enabled_tools: [
                    ...(toolsEnabled ? ["web_search"] : []),
                    ...(codeToolsEnabled ? ["python", "terminal"] : []),
                  ],
                  mcp_enabled: mcpEnabledForChat,
                  auto_heal_tool_calls:
                    useChatRuntimeStore.getState().autoHealToolCalls,
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
              requestPayload = await buildRequestPayload(retriedWithRefreshedKey);
            } catch (error) {
              clearSelectedImageEditReference();
              throw error;
            }
            clearSelectedImageEditReference();
            const stream = streamChatCompletions(requestPayload, abortSignal);

            for await (const chunk of stream) {
              // Handle tool status events
              const toolStatusText = (
                chunk as unknown as { _toolStatus?: string }
              )._toolStatus;
              if (toolStatusText !== undefined) {
                runtime.setToolStatus(toolStatusText || null);
                continue;
              }

              // Emit tool-call content parts for assistant-ui.
              // On tool_start: add a new tool-call part (renders in "running" state).
              // On tool_end: set result on the existing part (transitions to "complete").
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
                  // Convert Anthropic citations_delta footnotes into
                  // Sources-panel entries matching the inline [N] markers.
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
                  // Latch the backend refusal signal so the final
                  // message metadata can drive the prune.
                  anthropicRefusalSeen = true;
                  continue;
                }
                if (toolEvent.type === "tool_start") {
                  const id =
                    (toolEvent.tool_call_id as string) ||
                    `${toolEvent.tool_name}_${Date.now()}`;
                  const toolArgs = (toolEvent.arguments ??
                    {}) as ToolCallMessagePart["args"];
                  toolCallParts.push({
                    type: "tool-call" as const,
                    toolCallId: id,
                    toolName: toolEvent.tool_name as string,
                    argsText: JSON.stringify(toolArgs),
                    args: toolArgs,
                  });
                } else if (toolEvent.type === "tool_end") {
                  const id =
                    (toolEvent.tool_call_id as string) ||
                    toolCallParts[toolCallParts.length - 1]?.toolCallId ||
                    "";
                  const idx = toolCallParts.findIndex(
                    (p) => p.toolCallId === id,
                  );
                  if (idx !== -1) {
                    const rawResult = (toolEvent.result as string) ?? "";
                    const imgMarker = "\n__IMAGES__:";
                    const imgIdx = rawResult.lastIndexOf(imgMarker);
                    let parsedResult:
                      | string
                      | { text: string; images: string[]; sessionId: string }
                      | {
                          image_b64: string;
                          image_mime: string;
                          size?: string;
                          quality?: string;
                          background?: string;
                          prompt?: string;
                        };
                    const imageB64 = toolEvent.image_b64 as string | undefined;
                    if (
                      toolCallParts[idx].toolName === "image_generation" &&
                      typeof imageB64 === "string" &&
                      imageB64
                    ) {
                      // Backend keeps base64 on separate image_b64 /
                      // image_mime fields so logs stay small; repackage.
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
                    } else if (imgIdx !== -1) {
                      const text = rawResult.slice(0, imgIdx);
                      // Fall back to "_default" to match the backend sandbox directory
                      // used when no session_id is provided (see tools.py _get_workdir).
                      const sessionId = resolvedThreadId || "_default";
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
                    // Concatenate parts so each keeps its own thoughtSignature.
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
                      // Extract part entries from either parts:[...] or
                      // legacy single-object native_part. Legacy
                      // thoughtSignature always belongs on executableCode.
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
                    toolCallParts[idx] = {
                      ...toolCallParts[idx],
                      args: mergedArgs,
                      argsText: JSON.stringify(mergedArgs ?? {}),
                      result: parsedResult,
                    };
                  }
                }
                // Cumulative yield. orderAssistantContent puts search/
                // code before text and generated images after.
                const textParts = pinTextThoughtSignature(
                  parseAssistantContent(cumulativeText),
                );
                yield {
                  content: orderAssistantContent(textParts),
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

              // OpenAI-standard usage chunk: choices=[], usage populated
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
              // Latch the chunk's `model` field so the openrouter/free
              // chip can show the chosen underlying model.
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
              // Normalize structured delta.content (mistral magistral) to text.
              const delta = extractDeltaText(rawDelta);
              // Latest Gemini text-part thoughtSignature for next-turn replay.
              const deltaExtraContent = (
                chunk.choices?.[0]?.delta as
                  | { extra_content?: unknown }
                  | undefined
              )?.extra_content;
              if (
                deltaExtraContent &&
                typeof deltaExtraContent === "object"
              ) {
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
              // Kimi / DeepSeek stream thinking via delta.reasoning_content.
              // Wrap inline as <think>...</think> for parseAssistantContent.
              const rawReasoning = (
                chunk.choices?.[0]?.delta as
                  | { reasoning_content?: unknown }
                  | undefined
              )?.reasoning_content;
              // OpenRouter ships reasoning as delta.reasoning_details[]
              // regardless of underlying provider; merge into the same wrap path.
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
              // thoughtSignature for next-turn replay.
              const rawDeltaToolCalls = (
                chunk.choices?.[0]?.delta as
                  | { tool_calls?: unknown }
                  | undefined
              )?.tool_calls;
              if (
                Array.isArray(rawDeltaToolCalls) &&
                rawDeltaToolCalls.length > 0
              ) {
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
                  // Match an existing fragment by id first (canonical),
                  // then by index slot. Fall back to a freshly-minted
                  // tool_call_<n> id for streams that send neither.
                  let existing = stableId
                    ? toolCallParts.find((p) => p.toolCallId === stableId)
                    : undefined;
                  if (!existing && idx !== undefined) {
                    existing = toolCallParts.find(
                      (p) =>
                        (
                          p as ToolCallMessagePart & { _delta_index?: number }
                        )._delta_index === idx,
                    );
                  }
                  const argsFragment = call.function?.arguments ?? "";
                  if (existing) {
                    const prevName = existing.toolName ?? "";
                    const nextName = call.function?.name ?? prevName;
                    const merged =
                      (existing.argsText ?? "") + argsFragment;
                    let parsedArgs:
                      ToolCallMessagePart["args"] = existing.args ?? {};
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
                    const prevExtra = (
                      existing as ToolCallMessagePart & {
                        extra_content?: unknown;
                      }
                    ).extra_content;
                    const updated: ToolCallMessagePart & {
                      _delta_index?: number;
                      extra_content?: unknown;
                    } = {
                      ...(existing as ToolCallMessagePart),
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
                      stableId ||
                      `tool_call_${idx ?? toolCallParts.length}`;
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
                    const fresh: ToolCallMessagePart & {
                      _delta_index?: number;
                      extra_content?: unknown;
                    } = {
                      type: "tool-call" as const,
                      toolCallId: callId,
                      toolName: call.function?.name ?? "",
                      argsText,
                      args: parsedArgs,
                      ...(call.extra_content !== undefined
                        ? { extra_content: call.extra_content }
                        : {}),
                      ...(idx !== undefined ? { _delta_index: idx } : {}),
                    };
                    toolCallParts.push(fresh);
                  }
                }
                yield {
                  content: [
                    ...toolCallParts,
                    ...pinTextThoughtSignature(
                      parseAssistantContent(cumulativeText),
                    ),
                  ],
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
                if (reasoningContentOpen) {
                  cumulativeText += "</think>";
                  reasoningContentOpen = false;
                }
                cumulativeText += delta;
              }
              // Strip a trailing ${...} template-literal artifact from
              // external streams (mistral magistral occasionally emits one).
              if (isExternalRequest) {
                cumulativeText = cumulativeText.replace(
                  /\s*\$\{[^}]*\}\s*$/,
                  "",
                );
              }
              const parts = pinTextThoughtSignature(
                parseAssistantContent(cumulativeText),
              );

              if (
                parts.some((part) => part.type === "reasoning") &&
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

              if (parts.length > 0 || toolCallParts.length > 0) {
                yield {
                  content: orderAssistantContent(parts),
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
        if (reasoningContentOpen) {
          cumulativeText += "</think>";
          reasoningContentOpen = false;
        }
        settleFirstTokenOk();

        // Extract source parts from completed web_search and web_fetch
        // tool calls. Both emit the same `Title:` / `URL:` / `Snippet:`
        // block shape from the Anthropic backend, so the parser does
        // not need to branch on tool name.
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
        // completion from provider A doesn't populate the bar after the
        // user switched to provider B mid-stream.
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

        const finalTiming = buildTiming(
          streamStartTime,
          totalChunks,
          serverPromptEvalTime ?? firstTokenTime,
          Date.now() - streamStartTime,
          finalTokenCount,
          toolCallParts.length,
          finalTokPerSec,
        );

        yield {
          content: [
            ...orderAssistantContent(
              pinTextThoughtSignature(parseAssistantContent(cumulativeText)),
            ),
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
          if (isContextLimitError(msg)) {
            // llama-server was launched with --no-context-shift, so it
            // returns a hard error instead of silently dropping old
            // turns from the KV cache. Point the user at the exact
            // control that raises the ceiling.
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
        runtime.setGeneratingStatus(null);
        runtime.setToolStatus(null);
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
