// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type GeminiAnswerReplayPart,
  type GeminiContinuationReplayTurn,
  type GeminiThoughtReplayPart,
  parseGeminiAnswerReplayParts,
  parseGeminiContinuationReplayTurns,
  parseGeminiThoughtReplayParts,
} from "../gemini-thought-replay.ts";
import type { ProviderCompactionContentPart } from "../types/api";
import { providerCompactionPart } from "./provider-compaction.ts";

/** Resuming a response that stopped early (`length`, `cancelled`, `interrupted`): the conversation is re-sent
 *  with the partial as the final assistant turn plus `continue_final_message`, so the prompt ends mid-sentence
 *  and the new text is appended to the partial. */

/** Why a turn ended before the model was done. `context_window` is a `length` cut the same
 *  request can never fit into, hence its own reason. `empty` is a clean finish that produced
 *  nothing, which is a failure to report rather than an answer. `quote_cut` flags a
 *  possible mid-quote stop. */
export type IncompleteReason =
  | "length"
  | "cancelled"
  | "interrupted"
  | "context_window"
  | "empty"
  | "quote_cut";

export type IncompleteInfo = {
  reason: IncompleteReason;
};

/** Whether a finished turn left anything on screen, which is what separates `empty` from a
 *  real answer. Structured parts (tool calls, images, sources) always count; text and
 *  reasoning have to contain more than whitespace. */
export function hasRenderableContent(
  content: readonly { type: string; text?: string }[],
): boolean {
  return content.some((part) => {
    if (part.type === "text" || part.type === "reasoning") {
      return (part.text ?? "").trim().length > 0;
    }
    return true;
  });
}

const INCOMPLETE_REASONS: readonly IncompleteReason[] = [
  "length",
  "cancelled",
  "interrupted",
  "context_window",
  "empty",
  "quote_cut",
];

/** Below this a shared boundary is likely coincidence, and trimming would eat output. */
const MIN_OVERLAP = 12;

const MAX_OVERLAP = 400;

const RESTART_PROBE = 48;

/** A provider-reported filled window outranks every inferred reason. */
export function resolveIncompleteReason<T extends IncompleteReason | null>(
  reason: T,
  contextWindowExceeded: boolean,
): T | "context_window" {
  return contextWindowExceeded ? "context_window" : reason;
}

export function incompleteReasonAfterError(
  latched: IncompleteReason | null,
  fromError: IncompleteReason,
): IncompleteReason {
  if (latched === "length" && fromError === "context_window") {
    return fromError;
  }
  return latched ?? fromError;
}

export function isProviderReportedReason(
  reason: IncompleteReason | null | undefined,
): boolean {
  return reason === "context_window";
}

export function readIncompleteInfo(metadata: unknown): IncompleteInfo | null {
  const custom = (metadata as { custom?: Record<string, unknown> } | undefined)
    ?.custom;
  const incomplete = custom?.incomplete as { reason?: unknown } | undefined;
  const reason = incomplete?.reason;
  if (
    typeof reason === "string" &&
    (INCOMPLETE_REASONS as readonly string[]).includes(reason)
  ) {
    return { reason: reason as IncompleteReason };
  }
  return null;
}

/** `length` maps to a non-error status: the Continue bar already covers it. */
const STATUS_REASON: Record<
  IncompleteReason,
  "cancelled" | "length" | "error"
> = {
  cancelled: "cancelled",
  length: "length",
  interrupted: "error",
  context_window: "length",
  // Not `cancelled` (drops the stamped reason on reload) and not `error` (red box over the bar).
  empty: "length",
  quote_cut: "length",
};

export function restoredAssistantStatus(
  metadata: unknown,
): import("@assistant-ui/react").MessageStatus {
  const incomplete = readIncompleteInfo(metadata);
  if (!incomplete) {
    return { type: "complete", reason: "unknown" };
  }
  return { type: "incomplete", reason: STATUS_REASON[incomplete.reason] };
}

const INCOMPLETE_LABELS: Record<IncompleteReason, string> = {
  length: "Response hit the Max Tokens limit",
  cancelled: "Response stopped",
  interrupted: "Response interrupted",
  context_window: "Response filled the model's context window",
  empty: "The model returned an empty response",
  quote_cut: "This response may have ended early",
};

export function incompleteLabel(reason: IncompleteReason): string {
  return INCOMPLETE_LABELS[reason];
}

const INCOMPLETE_REMEDIES: Partial<Record<IncompleteReason, string>> = {
  context_window: "Start a new chat, or shorten this one, to keep going",
  empty: "Try again, or pick a different model",
  quote_cut:
    "The model may have emitted a special token while quoting it. Write special tokens with a space inside, like < |im_end|> or < end_of_turn>, ask the model to do the same, and try again",
};

export function incompleteRemedy(reason: IncompleteReason): string | null {
  return INCOMPLETE_REMEDIES[reason] ?? null;
}

/** Providers ignoring assistant prefill can restate the last words; trim that leading overlap. */
export function stripContinuationOverlap(
  partial: string,
  continuation: string,
): string {
  if (partial.length === 0 || continuation.length === 0) {
    return continuation;
  }
  const limit = Math.min(partial.length, continuation.length, MAX_OVERLAP);
  for (let size = limit; size >= MIN_OVERLAP; size -= 1) {
    if (continuation.startsWith(partial.slice(partial.length - size))) {
      return continuation.slice(size);
    }
  }
  return continuation;
}

export function isRestart(partial: string, continuation: string): boolean {
  const head = partial.trimStart().slice(0, RESTART_PROBE);
  if (head.length < RESTART_PROBE) {
    return false;
  }
  return continuation.trimStart().startsWith(head);
}

export function joinContinuation(
  partial: string,
  continuation: string,
  { streaming = false }: { streaming?: boolean } = {},
): string {
  if (!partial) {
    return continuation;
  }
  if (!streaming && isRestart(partial, continuation)) {
    return continuation;
  }
  return `${partial}${stripContinuationOverlap(partial, continuation)}`;
}

/** Never run isRestart per arrival: it would publish the continuation alone and Stop persists it. */
export function createContinuationMerger(
  partial: string,
  repair: boolean,
): (cumulative: string, options?: { final?: boolean }) => string {
  return (cumulative, { final = false } = {}) => {
    if (!partial || !repair) {
      return cumulative;
    }
    return joinContinuation(partial, cumulative.slice(partial.length), {
      streaming: !final,
    });
  };
}

/** MLX reports finish_reason "stop" at the token cap, so infer truncation from the budget. */
export function budgetImpliesTruncation({
  isMlx,
  maxTokens,
  completionTokens,
}: {
  isMlx: boolean;
  maxTokens: number | undefined;
  completionTokens: number | undefined;
}): boolean {
  if (!isMlx) {
    return false;
  }
  return (
    typeof maxTokens === "number" &&
    typeof completionTokens === "number" &&
    completionTokens >= maxTokens
  );
}

/** Mirrors the backend guard: tool calls block; reasoning-only needs `thought`. */
export function isContinuableContent(
  content: readonly unknown[] | undefined,
  {
    thought = false,
    replay = false,
  }: { thought?: boolean; replay?: boolean } = {},
): boolean {
  if (!content) {
    return replay;
  }
  let hasText = false;
  let hasReasoning = false;
  for (const part of content) {
    const type = (part as { type?: string })?.type;
    if (type === "text") {
      hasText = hasText || ((part as { text?: string }).text ?? "").length > 0;
      continue;
    }
    if (type === "reasoning") {
      hasReasoning =
        hasReasoning || ((part as { text?: string }).text ?? "").trim().length > 0;
      continue;
    }
    if (type === "source") {
      continue;
    }
    return false;
  }
  return hasText || (thought && hasReasoning) || replay;
}

export function readContinuationSource(
  content: readonly unknown[] | undefined,
): { partial: string; reasoning: string } {
  let partial = "";
  const thoughts: string[] = [];
  let ordered = true;
  for (const part of content ?? []) {
    const { type, text } = (part ?? {}) as { type?: string; text?: unknown };
    if (typeof text !== "string") {
      continue;
    }
    if (type === "text") {
      partial += text;
    } else if (type === "reasoning") {
      ordered = ordered && partial.length === 0;
      thoughts.push(text);
    }
  }
  return { partial, reasoning: ordered ? thoughts.join("\n") : "" };
}

export function continuationSeed(partial: string, thought: string): string {
  if (!thought) {
    return partial;
  }
  return partial ? `<think>${thought}</think>${partial}` : `<think>${thought}`;
}

export function readTextThoughtSignature(
  content: readonly unknown[] | undefined,
): string | undefined {
  if (!content) {
    return undefined;
  }
  for (let i = content.length - 1; i >= 0; i -= 1) {
    const part = content[i] as
      | { type?: string; _google_thought_signature?: unknown }
      | undefined;
    if (part?.type !== "text") {
      continue;
    }
    const signature = part._google_thought_signature;
    if (typeof signature === "string" && signature) {
      return signature;
    }
  }
  return undefined;
}

/** Anthropic, Gemini and Mistral reject a trailing assistant turn; send an instruction instead. */
const PREFILL_REJECTING_PROVIDERS = new Set(["anthropic", "gemini", "mistral"]);

export function rejectsAssistantPrefill(
  providerType: string | undefined,
): boolean {
  return providerType != null && PREFILL_REJECTING_PROVIDERS.has(providerType);
}

/** Mirrors `_CONTINUATION_FLAG_PROVIDERS` in external_provider.py; these resume token-exactly. */
const EXACT_RESUME_PROVIDERS = new Set(["vllm", "llama_cpp"]);

export function resumesExactly(providerType: string | undefined): boolean {
  return providerType != null && EXACT_RESUME_PROVIDERS.has(providerType);
}

/** Audio input or output would regenerate rather than continue, so Continue is hidden. */
export function modeAllowsContinuation({
  fromAudioInput,
  audioOutputModel,
}: {
  fromAudioInput: boolean;
  audioOutputModel: boolean;
}): boolean {
  return !(fromAudioInput || audioOutputModel);
}

export const CONTINUE_INSTRUCTION =
  "Continue your previous response from exactly where it stopped. " +
  "Do not repeat any text you already wrote and do not restate the answer.";

export const CONTINUATION_RUN_CONFIG_KEY = "unslothContinuation";

export type ContinuationRequest = {
  partial: string;
  reasoning?: string;
  reasoningDuration?: number;
  /** The sibling run drops the original message, so replaying this keeps history signed. */
  thoughtSignature?: string;
  /** Signed Gemini thought-summary parts from the turn being resumed. */
  thoughtParts?: GeminiThoughtReplayPart[];
  /** Exact Gemini answer-part boundaries from the turn being resumed. */
  answerParts?: GeminiAnswerReplayPart[];
  /** Gemini responses hidden behind the merged continuation bubble, in provider order. */
  geminiReplayTurns?: GeminiContinuationReplayTurn[];
  providerCompaction?: ProviderCompactionContentPart;
  providerCompactionAfterToolCalls?: number;
  providerCompactionProviderType?: string;
  providerCompactionModelId?: string;
  providerCompactionConnectionKey?: string;
};

type ProviderCompactionContinuationFields = Pick<
  Required<ContinuationRequest>,
  | "providerCompaction"
  | "providerCompactionAfterToolCalls"
  | "providerCompactionProviderType"
  | "providerCompactionModelId"
  | "providerCompactionConnectionKey"
>;

function providerCompactionFields(
  value: unknown,
): ProviderCompactionContinuationFields | Record<string, never> {
  const fields = value as
    | {
        providerCompaction?: unknown;
        providerCompactionAfterToolCalls?: unknown;
        providerCompactionProviderType?: unknown;
        providerCompactionModelId?: unknown;
        providerCompactionConnectionKey?: unknown;
      }
    | undefined;
  const compaction = providerCompactionPart(fields?.providerCompaction);
  const boundary = fields?.providerCompactionAfterToolCalls;
  const providerType = fields?.providerCompactionProviderType;
  const modelId = fields?.providerCompactionModelId;
  const connectionKey = fields?.providerCompactionConnectionKey;
  if (
    !compaction ||
    !Number.isInteger(boundary) ||
    (boundary as number) < 0 ||
    typeof providerType !== "string" ||
    !providerType ||
    typeof modelId !== "string" ||
    !modelId ||
    typeof connectionKey !== "string" ||
    !connectionKey
  ) {
    return {};
  }
  return {
    providerCompaction: compaction,
    providerCompactionAfterToolCalls: boundary as number,
    providerCompactionProviderType: providerType,
    providerCompactionModelId: modelId,
    providerCompactionConnectionKey: connectionKey,
  };
}

export function providerCompactionContinuationFields(
  metadata: unknown,
): ProviderCompactionContinuationFields | Record<string, never> {
  const custom = (metadata as { custom?: unknown } | undefined)?.custom;
  return providerCompactionFields(custom);
}

/** Read a continuation request out of a run's `runConfig`, if it is one. */
export function readContinuationRequest(
  runConfig: unknown,
): ContinuationRequest | null {
  const custom = (runConfig as { custom?: Record<string, unknown> } | undefined)
    ?.custom;
  const request = custom?.[CONTINUATION_RUN_CONFIG_KEY] as
    | {
        partial?: unknown;
        reasoning?: unknown;
        reasoningDuration?: unknown;
        thoughtSignature?: unknown;
        thoughtParts?: unknown;
        answerParts?: unknown;
        geminiReplayTurns?: unknown;
        providerCompaction?: unknown;
        providerCompactionAfterToolCalls?: unknown;
        providerCompactionProviderType?: unknown;
        providerCompactionModelId?: unknown;
        providerCompactionConnectionKey?: unknown;
      }
    | undefined;
  const partial = typeof request?.partial === "string" ? request.partial : "";
  const reasoning =
    typeof request?.reasoning === "string" && request.reasoning.trim()
      ? request.reasoning
      : "";
  const signature = request?.thoughtSignature;
  const thoughtParts = parseGeminiThoughtReplayParts(request?.thoughtParts);
  const answerParts = parseGeminiAnswerReplayParts(request?.answerParts);
  const geminiReplayTurns = parseGeminiContinuationReplayTurns(
    request?.geminiReplayTurns,
  );
  const hasGeminiReplay = Boolean(
    (typeof signature === "string" && signature) ||
      thoughtParts.length > 0 ||
      answerParts.length > 0 ||
      geminiReplayTurns.length > 0,
  );
  if (!partial && !reasoning && !hasGeminiReplay) {
    return null;
  }
  const duration = request?.reasoningDuration;
  return {
    partial,
    ...(reasoning ? { reasoning } : {}),
    ...(reasoning &&
    typeof duration === "number" &&
    Number.isFinite(duration) &&
    duration >= 0
      ? { reasoningDuration: duration }
      : {}),
    ...(typeof signature === "string" && signature
      ? { thoughtSignature: signature }
      : {}),
    ...(thoughtParts.length > 0 ? { thoughtParts } : {}),
    ...(answerParts.length > 0 ? { answerParts } : {}),
    ...(geminiReplayTurns.length > 0 ? { geminiReplayTurns } : {}),
    ...providerCompactionFields(request),
  };
}

/** Auto-resume only Max Tokens cuts, bounded so a model that never stops cannot loop forever. */
export const AUTO_CONTINUE_LIMIT = 3;

/** Keyed by parent: a continuation runs as a sibling, so a per-message counter would reset. */
const spent = new Map<string, number>();

export function shouldAutoContinue(
  reason: IncompleteReason | null | undefined,
  key: string | null | undefined,
  {
    limit = AUTO_CONTINUE_LIMIT,
    fits,
    partialTokens,
    promptTarget,
  }: {
    limit?: number;
    fits?: boolean;
    partialTokens?: number;
    promptTarget?: number;
  } = {},
): boolean {
  if (reason !== "length" || !key) {
    return false;
  }
  if (fits === false) {
    // A `fits: false` turn cannot be resumed: nothing is left to evict and continuing needs more.
    return false;
  }
  if (
    typeof partialTokens === "number" &&
    typeof promptTarget === "number" &&
    promptTarget > 0 &&
    partialTokens >= promptTarget
  ) {
    return false;
  }
  return (spent.get(key) ?? 0) < limit;
}

/** Called before the run starts, so a failed run still consumes its budget. */
export function recordAutoContinue(key: string): void {
  spent.set(key, (spent.get(key) ?? 0) + 1);
}

export function autoContinueCount(key: string | null | undefined): number {
  return key ? (spent.get(key) ?? 0) : 0;
}

/** A lease, not a flag, so a crashed tab cannot block forever; hidden tabs tick about once a minute. */
export const AUTO_CONTINUE_LEASE_TTL_MS = 180_000;

export const AUTO_CONTINUE_LEASE_RENEW_MS = 30_000;

/** Keeps a finished record so a stale second tab cannot continue the same message again. */
export const AUTO_CONTINUE_CONTINUED_TTL_MS = 86_400_000;

export const AUTO_CONTINUE_LEASE_KEY = "unsloth_chat_auto_continue_leases";

export const AUTO_CONTINUE_LOCK_NAME = "unsloth_chat_auto_continue_claim";

/** `held-elsewhere` restores the manual Continue button; `skipped` is this tab's own duplicate. */
export type AutoContinueClaim = "started" | "skipped" | "held-elsewhere";

export type AutoContinueLeaseStorage = {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
};

export type AutoContinueLockManager = {
  request<T>(name: string, callback: () => T | Promise<T>): Promise<T>;
};

type Lease = {
  token: string;
  expires: number;
  done?: boolean;
};

/** Resolved per call: absent in tests and SSR, and Safari private mode throws on access. */
function browserLeaseStorage(): AutoContinueLeaseStorage | null {
  try {
    if (typeof window === "undefined") {
      return null;
    }
    return (
      (window.localStorage as AutoContinueLeaseStorage | undefined) ?? null
    );
  } catch {
    return null;
  }
}

/** Web Locks make the read-modify-write exclusive; localStorage alone is not across statements. */
function browserLockManager(): AutoContinueLockManager | null {
  try {
    const locks = (globalThis.navigator as { locks?: unknown } | undefined)
      ?.locks as AutoContinueLockManager | undefined;
    return typeof locks?.request === "function" ? locks : null;
  } catch {
    return null;
  }
}

function readLeases(storage: AutoContinueLeaseStorage): Record<string, Lease> {
  const raw = storage.getItem(AUTO_CONTINUE_LEASE_KEY);
  if (!raw) {
    return {};
  }
  const parsed = JSON.parse(raw) as unknown;
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return {};
  }
  const out: Record<string, Lease> = {};
  for (const [id, value] of Object.entries(parsed as Record<string, unknown>)) {
    const token = (value as Lease | undefined)?.token;
    const expires = (value as Lease | undefined)?.expires;
    if (typeof token === "string" && typeof expires === "number") {
      out[id] = (value as Lease).done === true
        ? { token, expires, done: true }
        : { token, expires };
    }
  }
  return out;
}

function newLeaseToken(): string {
  const uuid = (globalThis.crypto as Crypto | undefined)?.randomUUID;
  if (typeof uuid === "function") {
    return uuid.call(globalThis.crypto);
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

export function createAutoContinueTab({
  storage,
  locks,
}: {
  storage?: AutoContinueLeaseStorage | null;
  locks?: AutoContinueLockManager | null;
} = {}): {
  claim: (
    messageId: string | null | undefined,
    options?: { now?: number; holder?: string },
  ) => Promise<AutoContinueClaim>;
  claimed: (
    messageId: string | null | undefined,
    options?: { now?: number },
  ) => boolean;
  renew: (
    messageId: string | null | undefined,
    holder: string,
    options?: { now?: number },
  ) => Promise<void>;
  release: (
    messageId: string | null | undefined,
    holder: string,
    options?: { now?: number },
  ) => Promise<void>;
  forget: (messageId: string | null | undefined) => void;
  reset: () => void;
} {
  const continued = new Set<string>();

  const claiming = new Set<string>();

  /** Tagged by holder: compare mode mounts two panes in one tab. */
  const ownTokens = new Map<string, { token: string; holder: string }>();

  const leaseSeam = (): AutoContinueLeaseStorage | null =>
    storage === undefined ? browserLeaseStorage() : storage;

  const lockSeam = (): AutoContinueLockManager | null =>
    locks === undefined ? browserLockManager() : locks;

  async function exclusively<T>(mutate: () => T, fallback: T): Promise<T> {
    const manager = lockSeam();
    if (!manager) {
      return mutate();
    }
    let ran = false;
    try {
      return await manager.request(AUTO_CONTINUE_LOCK_NAME, () => {
        ran = true;
        return mutate();
      });
    } catch {
      return ran ? fallback : mutate();
    }
  }

  function prune(
    leases: Record<string, Lease>,
    now: number,
  ): Record<string, Lease> {
    const out: Record<string, Lease> = {};
    for (const [id, lease] of Object.entries(leases)) {
      if (lease.expires > now) {
        out[id] = lease;
      }
    }
    return out;
  }

  function liveLease(messageId: string, now: number): Lease | null {
    const store = leaseSeam();
    if (!store) {
      return null;
    }
    try {
      const lease = readLeases(store)[messageId];
      return lease && lease.expires > now ? lease : null;
    } catch {
      return null;
    }
  }

  function takeLease(
    messageId: string,
    now: number,
    holder: string,
  ): AutoContinueClaim {
    const store = leaseSeam();
    if (!store) {
      return "started";
    }
    let leases: Record<string, Lease>;
    try {
      leases = readLeases(store);
    } catch {
      return "started";
    }
    const held = leases[messageId];
    if (
      held &&
      held.expires > now &&
      held.token !== ownTokens.get(messageId)?.token
    ) {
      return "held-elsewhere";
    }
    const token = newLeaseToken();
    const next = prune(leases, now);
    next[messageId] = { token, expires: now + AUTO_CONTINUE_LEASE_TTL_MS };
    try {
      store.setItem(AUTO_CONTINUE_LEASE_KEY, JSON.stringify(next));
      // Without a lock manager, read back: the last writer's token wins.
      if (readLeases(store)[messageId]?.token !== token) {
        return "held-elsewhere";
      }
    } catch {
      return "started";
    }
    ownTokens.set(messageId, { token, holder });
    return "started";
  }

  function heldFor(messageId: string, holder: string): boolean {
    return ownTokens.get(messageId)?.holder === holder;
  }

  /** Restamp ONE message: the next round is claimed before the finished one is released. */
  function restamp(
    messageId: string,
    holder: string,
    expires: number,
    now: number,
    done = false,
  ): boolean {
    const held = ownTokens.get(messageId);
    if (!held || held.holder !== holder) {
      return false;
    }
    const store = leaseSeam();
    if (!store) {
      return false;
    }
    try {
      const leases = prune(readLeases(store), now);
      if (leases[messageId]?.token !== held.token) {
        ownTokens.delete(messageId);
        return false;
      }
      leases[messageId] = done
        ? { token: held.token, expires, done: true }
        : { token: held.token, expires };
      store.setItem(AUTO_CONTINUE_LEASE_KEY, JSON.stringify(leases));
      return true;
    } catch {
      return false;
    }
  }

  return {
    async claim(messageId, { now = Date.now(), holder = "" } = {}) {
      if (!messageId || continued.has(messageId) || claiming.has(messageId)) {
        return "skipped";
      }
      if (liveLease(messageId, now)) {
        return "held-elsewhere";
      }
      claiming.add(messageId);
      try {
        const outcome = await exclusively(
          () => takeLease(messageId, now, holder),
          "held-elsewhere" as AutoContinueClaim,
        );
        if (outcome === "started") {
          continued.add(messageId);
        }
        return outcome;
      } finally {
        claiming.delete(messageId);
      }
    },
    claimed(messageId, { now = Date.now() } = {}) {
      if (!messageId) {
        return false;
      }
      return continued.has(messageId) || Boolean(liveLease(messageId, now));
    },
    async renew(messageId, holder, { now = Date.now() } = {}) {
      if (!messageId) {
        return;
      }
      if (!heldFor(messageId, holder)) {
        return;
      }
      await exclusively(
        () => restamp(messageId, holder, now + AUTO_CONTINUE_LEASE_TTL_MS, now),
        false,
      );
    },
    async release(messageId, holder, { now = Date.now() } = {}) {
      if (!messageId) {
        return;
      }
      if (!heldFor(messageId, holder)) {
        return;
      }
      await exclusively(
        () =>
          restamp(
            messageId,
            holder,
            now + AUTO_CONTINUE_CONTINUED_TTL_MS,
            now,
            true,
          ),
        false,
      );
      // Marked done rather than released: only a lapsing lease should hand the message on.
      ownTokens.delete(messageId);
    },
    /** Drops only this tab's record; the storage lease runs out its TTL. */
    forget(messageId) {
      if (!messageId) {
        return;
      }
      continued.delete(messageId);
    },
    reset() {
      continued.clear();
      claiming.clear();
      const store = leaseSeam();
      if (!store || ownTokens.size === 0) {
        ownTokens.clear();
        return;
      }
      try {
        const leases = readLeases(store);
        for (const [id, held] of ownTokens) {
          if (leases[id]?.token === held.token) {
            delete leases[id];
          }
        }
        store.setItem(AUTO_CONTINUE_LEASE_KEY, JSON.stringify(leases));
      } catch {
        // Nothing to undo: an unwritable seam held no lease of ours either.
      }
      ownTokens.clear();
    },
  };
}

const tab = createAutoContinueTab();

export function claimAutoContinue(
  messageId: string | null | undefined,
  holder: string,
): Promise<AutoContinueClaim> {
  return tab.claim(messageId, { holder });
}

export function wasAutoContinued(messageId: string | null | undefined): boolean {
  return tab.claimed(messageId);
}

export function forgetAutoContinue(messageId: string | null | undefined): void {
  tab.forget(messageId);
}

export function renewAutoContinueLease(
  messageId: string,
  holder: string,
): Promise<void> {
  return tab.renew(messageId, holder);
}

/** Cut to the settle window rather than deleted, so a stale tab cannot start a duplicate. */
export function releaseAutoContinueLease(
  messageId: string,
  holder: string,
): Promise<void> {
  return tab.release(messageId, holder);
}

/** Not the on-screen thread: a run keeps streaming after the user opens another chat. */
export type AutoContinueRunSignal = {
  isRunning(threadId: string): boolean;
  subscribe(onChange: () => void): () => void;
};

/** Must be the run's own promise: the next round is claimed while the previous winds down. */
export type AutoContinueIssuedRun = {
  whenSettled(onSettled: () => void): void;
};

/** Holds renew until their own run appears; preflight has no time bound (e.g. GGUF loads). */
export function createAutoContinueLeaseKeeper({
  signal,
  renew = (messageId, holder, now) => tab.renew(messageId, holder, { now }),
  release = (messageId, holder, now) => tab.release(messageId, holder, { now }),
  now = Date.now,
}: {
  signal: AutoContinueRunSignal;
  renew?: (messageId: string, holder: string, now: number) => void;
  release?: (messageId: string, holder: string, now: number) => void;
  now?: () => number;
}): {
  hold: (messageId: string, threadId: string) => void;
  settleOn: (
    messageId: string,
    threadId: string,
    issued: AutoContinueIssuedRun | undefined,
  ) => void;
  observe: () => void;
  failed: (threadId: string) => void;
  tick: () => void;
  held: () => number;
  stop: () => void;
} {
  type Hold = {
    messageId: string;
    threadId: string;
    idle: boolean;
    /** The key was free when taken, so a running reading is this hold's own run. */
    ownsTheKey: boolean;
    armed: boolean;
    settled: boolean;
  };
  const holds = new Map<string, Hold>();
  let unsubscribe: (() => void) | null = null;

  const key = (messageId: string, threadId: string) =>
    `${threadId}\u0000${messageId}`;

  function observe(): void {
    const at = now();
    for (const [id, hold] of [...holds]) {
      if (hold.settled && !hold.armed && hold.ownsTheKey) {
        // Stop during preflight: discard (not release) an unarmed hold so no `done` marker is written.
        holds.delete(id);
        continue;
      }
      if (signal.isRunning(hold.threadId)) {
        hold.armed ||= hold.idle;
        continue;
      }
      hold.idle = true;
      if (hold.armed) {
        holds.delete(id);
        release(hold.messageId, hold.threadId, at);
        continue;
      }
      // Unarmed and unsettled means preflight, which has no bound: keep renewing, never time out.
    }
    if (holds.size === 0 && unsubscribe) {
      unsubscribe();
      unsubscribe = null;
    }
  }

  return {
    hold(messageId, threadId) {
      if (!messageId) {
        return;
      }
      if (!threadId) {
        return;
      }
      holds.set(key(messageId, threadId), {
        messageId,
        threadId,
        idle: !signal.isRunning(threadId),
        ownsTheKey: !signal.isRunning(threadId),
        armed: false,
        settled: false,
      });
      unsubscribe ??= signal.subscribe(observe);
    },
    /** Separate from `hold`: the hold is taken before the run starts, so no promise exists yet. */
    settleOn(messageId, threadId, issued) {
      if (!issued || !messageId || !threadId) {
        return;
      }
      const hold = holds.get(key(messageId, threadId));
      if (!hold) {
        return;
      }
      issued.whenSettled(() => {
        // Use the captured hold: the key is reclaimed as soon as the next round hits Max Tokens.
        hold.settled = true;
        observe();
      });
    },
    observe,
    /** The adapter threw before the run signal; discard only unarmed holds, letting the TTL lapse. */
    failed(threadId) {
      if (!threadId) {
        return;
      }
      for (const [id, hold] of [...holds]) {
        if (hold.threadId === threadId && !hold.armed) {
          holds.delete(id);
        }
      }
      if (holds.size === 0 && unsubscribe) {
        unsubscribe();
        unsubscribe = null;
      }
    },
    tick() {
      observe();
      const at = now();
      for (const hold of holds.values()) {
        renew(hold.messageId, hold.threadId, at);
      }
    },
    held() {
      return holds.size;
    },
    stop() {
      holds.clear();
      unsubscribe?.();
      unsubscribe = null;
    },
  };
}

/** `spent` resets on reload, so only runs started this page may auto-continue. */
const startedThisSession = new Set<string>();

export function noteRunStartedThisSession(
  messageId: string | null | undefined,
): void {
  if (messageId) {
    startedThisSession.add(messageId);
  }
}

export function runStartedThisSession(
  messageId: string | null | undefined,
): boolean {
  return Boolean(messageId) && startedThisSession.has(messageId as string);
}

/** The budget is per turn but the claim is per message; check both or a spinner never resolves. */
export function shouldAutoContinueMessage(
  messageId: string | null | undefined,
  reason: IncompleteReason | null | undefined,
  key: string | null | undefined,
  options: Parameters<typeof shouldAutoContinue>[2] = {},
): boolean {
  if (!runStartedThisSession(messageId) || wasAutoContinued(messageId)) {
    return false;
  }
  return shouldAutoContinue(reason, key, options);
}

/** Test seam; releases only leases this tab wrote. */
export function resetAutoContinue(key?: string): void {
  if (key === undefined) {
    spent.clear();
    startedThisSession.clear();
    tab.reset();
  } else {
    spent.delete(key);
  }
}
