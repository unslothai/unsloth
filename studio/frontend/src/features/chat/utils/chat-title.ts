// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  buildExternalModelId,
  parseExternalModelId,
} from "../external-providers";
import { encryptProviderApiKey } from "../api/providers-api";
import {
  type CustomReasoningConfig,
  normalizeCustomReasoningConfig,
} from "../custom-reasoning";
import {
  type ExternalProviderConfig,
  getExternalProviderApiKey,
  isCustomProviderType,
  loadExternalProviders,
  toExternalBackendProviderType,
} from "../external-providers";
import {
  clampReasoningEffortToLevels,
  getExternalMinOutputTokens,
  getExternalReasoningCapabilities,
  getProviderCapabilities,
  isGeminiCustomOpenAICompatBase,
} from "../provider-capabilities";
import { useExternalProvidersStore } from "../stores/external-providers-store";
import type { MessageRecord, ThreadRecord } from "../types";
import type {
  OpenAIChatChunk,
  OpenAIChatCompletionsRequest,
} from "../types/api";
import { extractDeltaText } from "./parse-assistant-content";

/** Store the whole first line and let the sidebar clip it with CSS, so a wider one shows more.
 *  Matches the rename input's maxLength: UTF-16 units, ellipsis included. */
export const FALLBACK_TITLE_MAX = 120;

/** Older titles were stored pre-cut at 48 chars with a literal "...". Kept to find and rewrite those rows. */
export const LEGACY_FALLBACK_TITLE_MAX = 48;
const LEGACY_FALLBACK_SUFFIX = "...";

/** Drop unpaired surrogates: they render as nothing, and one reaching the backend fails its
 *  SQLite bind and 500s the title write. Iteration yields a valid pair whole, so a length-1
 *  unit in the range is a lone surrogate. */
function dropLoneSurrogates(text: string): string {
  let out = "";
  for (const character of text) {
    const code = character.codePointAt(0) ?? 0;
    if (character.length === 1 && code >= 0xd800 && code <= 0xdfff) continue;
    out += character;
  }
  return out;
}

function firstLineOf(text: string): string {
  const firstLine = (text || "").split(/\r?\n/, 1)[0] ?? "";
  // Drop surrogates first, or removing one can leave a double or trailing space.
  return dropLoneSurrogates(firstLine).replace(/\s+/g, " ").trim();
}

/** Cut to at most `maxUnits` UTF-16 units without splitting an astral character: a lone surrogate
 *  parses fine and then fails the backend's SQLite bind. */
function cutToUnits(text: string, maxUnits: number): string {
  let out = "";
  for (const character of text) {
    if (out.length + character.length > maxUnits) break;
    out += character;
  }
  return out;
}

export function fallbackTitleFromUserText(userText: string): string {
  const cleaned = firstLineOf(userText);
  if (!cleaned) return "New Chat";
  if (cleaned.length <= FALLBACK_TITLE_MAX) return cleaned;
  // The ellipsis takes one of the budget, so the title still fits the input.
  return cutToUnits(cleaned, FALLBACK_TITLE_MAX - 1).trimEnd() + "…";
}

/** Pre-filter on the title alone: only these are worth fetching messages for. */
export function couldBeLegacyClippedTitle(title: string | undefined): boolean {
  return (
    typeof title === "string" &&
    title.endsWith(LEGACY_FALLBACK_SUFFIX) &&
    title.length === LEGACY_FALLBACK_TITLE_MAX + LEGACY_FALLBACK_SUFFIX.length
  );
}

/** True when `title` is exactly the old 48-character cut of `userText`. */
export function isLegacyClippedTitle(
  title: string | undefined,
  userText: string,
): boolean {
  if (!couldBeLegacyClippedTitle(title)) return false;
  const kept = (title as string).slice(0, LEGACY_FALLBACK_TITLE_MAX);
  const cleaned = firstLineOf(userText);
  return (
    cleaned.length > LEGACY_FALLBACK_TITLE_MAX &&
    cleaned.slice(0, LEGACY_FALLBACK_TITLE_MAX) === kept
  );
}

function textOf(message: MessageRecord | undefined): string {
  if (!message) return "";
  const content = message.content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .filter(
      (part): part is Extract<typeof part, { type: "text" }> =>
        part.type === "text",
    )
    .map((part) => part.text)
    .join("")
    .trim();
}

export interface LegacyTitleRepair {
  threadId: string;
  /** The clipped title the rewrite is based on, guarding the write. */
  previousTitle: string;
  /** The message the title came from, guarded too: deleting it must not leave its text expanded into the title. */
  openingMessageId: string;
  title: string;
}

export interface LegacyRepairPage {
  candidates: ThreadRecord[];
  /** What this page skipped. The next page reads it, so a row this page failed on is not redrawn by the same drain. */
  rest: ThreadRecord[];
  hasMore: boolean;
}

/** One page of rows to look at, skipping the ones already tried. */
export function selectLegacyRepairPage(
  threads: ThreadRecord[],
  attempted: ReadonlySet<string>,
  limit: number,
): LegacyRepairPage {
  const pending = threads.filter(
    (thread) =>
      couldBeLegacyClippedTitle(thread.title) && !attempted.has(thread.id),
  );
  const candidates = pending.slice(0, limit);
  const taken = new Set(candidates.map((thread) => thread.id));
  return {
    candidates,
    rest: threads.filter((thread) => !taken.has(thread.id)),
    hasMore: pending.length > limit,
  };
}

/** Threads the backend holds no messages for. */
export function threadsMissingMessages(
  ids: readonly string[],
  messagesByThreadId: ReadonlyMap<string, MessageRecord[]>,
): string[] {
  return ids.filter((id) => (messagesByThreadId.get(id) ?? []).length === 0);
}

/** Of those, the ones still worth retrying: no record of their import finishing, so their
 *  messages may yet land. One the ledger knows is simply empty, and retrying it would re-read it
 *  on every refresh for the session, since its title stays clipped. */
export function threadsAwaitingImport(
  ids: readonly string[],
  messagesByThreadId: ReadonlyMap<string, MessageRecord[]>,
  importedThreadIds: ReadonlySet<string>,
): string[] {
  return threadsMissingMessages(ids, messagesByThreadId).filter(
    (id) => !importedThreadIds.has(id),
  );
}

/** Rows to rewrite. A title must be the exact old cut of its own first message, so a rename
 *  ending in "..." is left alone. */
export function planLegacyTitleRepairs(
  threads: ThreadRecord[],
  messagesByThreadId: Map<string, MessageRecord[]>,
): LegacyTitleRepair[] {
  const repairs: LegacyTitleRepair[] = [];
  for (const thread of threads) {
    const messages = messagesByThreadId.get(thread.id) ?? [];
    // Earliest, not first in the array; ties break on id, as the backend does.
    const opening = messages
      .filter((m) => m.role === "user")
      .reduce<MessageRecord | undefined>((earliest, m) => {
        if (earliest === undefined) return m;
        if (m.createdAt !== earliest.createdAt) {
          return m.createdAt < earliest.createdAt ? m : earliest;
        }
        return m.id < earliest.id ? m : earliest;
      }, undefined);
    const userText = textOf(opening);
    if (opening === undefined) continue;
    if (!isLegacyClippedTitle(thread.title, userText)) continue;
    const title = fallbackTitleFromUserText(userText);
    if (title === thread.title) continue;
    repairs.push({
      threadId: thread.id,
      previousTitle: thread.title,
      openingMessageId: opening.id,
      title,
    });
  }
  return repairs;
}

// Routing lives here, not in the adapter, so node --test can load it (the provider is JSX).

/** The backend dispatches on provider_id / provider_type, never parsing the external:: model id. */
export interface ExternalRoutingFields {
  provider_id: string;
  provider_type: string;
  external_model: string;
  provider_base_url: string | null;
  provider_api_type: "chat_completions" | "responses";
  provider_reasoning_config?: CustomReasoningConfig;
  encrypted_api_key?: string;
}

export type ExternalRoutingUnavailableReason =
  | "connections-disabled"
  | "connection-missing"
  | "missing-api-key";

export interface ResolvedExternalConnection {
  provider: ExternalProviderConfig;
  modelId: string;
  /** Browser-held key, or "" when the backend holds one or none is needed. */
  apiKey: string;
}

export type ExternalRoutingTarget =
  | { kind: "local" }
  | { kind: "unavailable"; reason: ExternalRoutingUnavailableReason }
  | ({ kind: "external" } & ResolvedExternalConnection);

/** A request answering only some of these is served by the local model instead (#9045). */
export function resolveExternalRouting(
  checkpoint: string | null | undefined,
): ExternalRoutingTarget {
  const selection = parseExternalModelId(checkpoint);
  if (selection === null) return { kind: "local" };

  if (!useExternalProvidersStore.getState().connectionsEnabled) {
    return { kind: "unavailable", reason: "connections-disabled" };
  }

  const provider = loadExternalProviders().find(
    (c) => c.id === selection.providerId,
  );
  if (!provider) return { kind: "unavailable", reason: "connection-missing" };

  // Installation-saved key wins: the browser copy may be stale.
  const apiKey = provider.hasApiKey
    ? ""
    : getExternalProviderApiKey(provider.id).trim();
  const keyOptional =
    Boolean(provider.hasApiKey) ||
    provider.authKind === "chatgpt_oauth" ||
    isCustomProviderType(provider.providerType) ||
    (provider.providerType === "gemini" &&
      isGeminiCustomOpenAICompatBase(provider.baseUrl));
  if (!apiKey && !keyOptional)
    return { kind: "unavailable", reason: "missing-api-key" };

  return { kind: "external", provider, modelId: selection.modelId, apiKey };
}

/** Encrypted per attempt so a rotated-key retry can rebuild with forceRefreshPublicKey. */
export async function buildExternalRoutingFields(
  connection: ResolvedExternalConnection,
  options: { forceRefreshPublicKey?: boolean } = {},
): Promise<ExternalRoutingFields> {
  const { provider, modelId, apiKey } = connection;
  const reasoningConfig =
    provider.providerType === "custom" &&
    (provider.backendProviderType === undefined || provider.backendProviderType === "custom") &&
    provider.apiType !== "responses" && !provider.decisionsOnly
      ? normalizeCustomReasoningConfig(provider.reasoningConfig)
      : undefined;
  return {
    provider_id: provider.id,
    provider_type: toExternalBackendProviderType(provider.providerType),
    external_model: modelId,
    provider_base_url: provider.baseUrl || null,
    provider_api_type: provider.apiType ?? "chat_completions",
    ...(reasoningConfig?.enabled ? { provider_reasoning_config: reasoningConfig } : {}),
    ...(apiKey
      ? {
          encrypted_api_key: await encryptProviderApiKey(
            apiKey,
            options.forceRefreshPublicKey ?? false,
          ),
        }
      : {}),
  };
}

/** A deep research run's config is evidence only once it completed. */
export function answeringCheckpoint(custom: unknown): string {
  const meta = (custom ?? {}) as {
    responseDetails?: { modelId?: unknown };
    researchRun?: {
      status?: unknown;
      config?: { inferenceRequest?: Record<string, unknown> };
    };
  };
  const stamped = meta.responseDetails?.modelId;
  if (typeof stamped === "string" && stamped) return stamped;

  const run = meta.researchRun;
  if (run?.status !== "completed") return "";

  const inference = run.config?.inferenceRequest ?? {};
  const { providerId, providerType, externalModel } = inference;
  const routed = [providerId, providerType, externalModel].every(
    (field) => typeof field === "string" && field !== "",
  );
  if (routed)
    return buildExternalModelId(providerId as string, externalModel as string);
  const model = typeof inference.model === "string" ? inference.model : "";
  return parseExternalModelId(model) === null ? model : "";
}

/** Follows the connection that answered, not the live selection, so the excerpt never reaches an unused connection; local models follow the selection so an evicted one is not reloaded. */
export function titleCheckpoint(
  answeredWith: string,
  activeCheckpoint: string,
): string {
  if (parseExternalModelId(answeredWith) !== null) return answeredWith;
  return parseExternalModelId(activeCheckpoint) === null
    ? activeCheckpoint
    : "";
}

const VERBATIM_EFFORT_PROVIDER_TYPES = new Set(["openai", "openai_codex"]);

type TitleReasoningFields = Pick<
  OpenAIChatCompletionsRequest,
  "enable_thinking" | "reasoning_effort"
>;

function titleReasoningCaps(connection: ResolvedExternalConnection) {
  const { provider, modelId } = connection;
  return getExternalReasoningCapabilities(provider.providerType, modelId, {
    isReasoningProvider: provider.isReasoningModel === true,
    baseUrl: provider.baseUrl ?? null,
    apiType: provider.apiType,
    reasoningConfig: provider.reasoningConfig,
  });
}

/** Responses route: either field becomes reasoning.effort (400 on non-reasoning models), sent verbatim, so omit or clamp. */
function titleReasoningFields(
  connection: ResolvedExternalConnection,
): TitleReasoningFields {
  const { provider } = connection;
  const caps = titleReasoningCaps(connection);
  const responsesRoute =
    VERBATIM_EFFORT_PROVIDER_TYPES.has(provider.providerType) ||
    provider.apiType === "responses";
  if (responsesRoute && !caps.supportsReasoning) return {};
  const clamp = responsesRoute && caps.reasoningStyle === "reasoning_effort";
  return {
    enable_thinking: false,
    reasoning_effort: clamp
      ? clampReasoningEffortToLevels("none", caps.reasoningEffortLevels)
      : "none",
  };
}

/** Reasoning that cannot be turned off counts toward the cap (Gemini 2.5 Pro forces 128), so it gets headroom. */
function titleMaxTokens(connection: ResolvedExternalConnection): number {
  const floor = getExternalMinOutputTokens(connection.provider.providerType);
  const caps = titleReasoningCaps(connection);
  const forcedReasoning = caps.supportsReasoning && !caps.supportsReasoningOff;
  return Math.max(24, floor, forcedReasoning ? 1024 : 0);
}

const TITLE_SYSTEM_PROMPT =
  "Write 1 concise chat title summarizing the conversation topic, not the user's exact wording. Use the assistant reply as context when provided. Rules: 2-6 words, no quotes, no punctuation, ASCII only, do not echo input. Output title only.";

export async function buildTitleRequest(
  checkpoint: string,
  prompt: string,
): Promise<OpenAIChatCompletionsRequest | null> {
  const routing = resolveExternalRouting(checkpoint);
  if (routing.kind === "unavailable") return null;

  // The backend forwards sampling fields verbatim, so gate them as the chat request does.
  const caps =
    routing.kind === "external"
      ? getProviderCapabilities(
          routing.provider.providerType,
          routing.provider.apiType,
          routing.modelId,
          routing.provider.baseUrl,
        )
      : null;
  const local = routing.kind === "local";

  return {
    model: checkpoint,
    // Required: the proxy answers SSE, so stream:false has no readable body.
    stream: true,
    ...(local || caps?.temperature !== false ? { temperature: 0.2 } : {}),
    ...(local || caps?.topP !== false ? { top_p: 0.9 } : {}),
    max_tokens: routing.kind === "external" ? titleMaxTokens(routing) : 24,
    ...(local || caps?.topK ? { top_k: 20 } : {}),
    ...(local || caps?.repetitionPenalty ? { repetition_penalty: 1.0 } : {}),
    ...(routing.kind === "external"
      ? titleReasoningFields(routing)
      : { enable_thinking: false, reasoning_effort: "none" as const }),
    // Else the server's tools-on default adds tool schemas.
    enable_tools: false,
    messages: [
      { role: "system", content: TITLE_SYSTEM_PROMPT },
      { role: "user", content: prompt },
    ],
    ...(routing.kind === "external"
      ? await buildExternalRoutingFields(routing)
      : {}),
  };
}

/** Truncated answers are discarded: hitting the token cap means it wrote something else. */
export async function titleFromStream(
  chunks: AsyncIterable<OpenAIChatChunk>,
): Promise<string | null> {
  let content = "";
  let finishReason: string | null = null;
  for await (const chunk of chunks) {
    const choice = chunk.choices?.[0];
    content += extractDeltaText(choice?.delta?.content).text;
    // A later usage chunk has no finish reason; it must not erase this one.
    finishReason = choice?.finish_reason ?? finishReason;
  }

  if (finishReason === "length") return null;
  // A model that cannot turn reasoning off streams its summary as a closed think block first.
  const visible = content.replace(/<think>[\s\S]*?<\/think>/gi, "");
  if (!visible || /<\/?think>/i.test(visible)) return null;
  return normalizeTitle(visible);
}

export function normalizeTitle(raw: string): string | null {
  let title = raw.split(/\r?\n/, 1)[0] ?? "";
  title = title.replace(/^\s*title\s*:\s*/i, "");
  title = title.replace(/[^\x20-\x7E]+/g, " ");
  title = title.replace(/["'`]+/g, "");

  // Echo fail-safe: reject leading role labels before punctuation strips the ":".
  if (/^\s*(user|assistant|base|lora)\s*:/i.test(title)) {
    return null;
  }

  title = title.replace(/[.!?:;,]+/g, " ");
  title = title.replace(/\s+/g, " ").trim();

  const words = title.split(" ").filter(Boolean).slice(0, 6);
  const joined = words.join(" ").trim();
  if (!joined) return null;
  return joined.length > 60 ? joined.slice(0, 60).trimEnd() : joined;
}
