// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Resuming a response that stopped early: Max Tokens ran out (`length`), Stop was
 * pressed (`cancelled`), or the stream was cut (`interrupted`).
 *
 * Continuing re-sends the conversation with the partial as the final assistant turn
 * plus `continue_final_message`, so the prompt ends mid-sentence and the model emits
 * the next token. The new text is appended to the partial.
 */

/** Why a turn ended before the model was done. */
export type IncompleteReason = "length" | "cancelled" | "interrupted";

/** Metadata stamped on an assistant message that stopped early. */
export type IncompleteInfo = {
  reason: IncompleteReason;
};

const INCOMPLETE_REASONS: readonly IncompleteReason[] = [
  "length",
  "cancelled",
  "interrupted",
];

/** Below this a shared boundary is likely coincidence, and trimming would eat output. */
const MIN_OVERLAP = 12;

/** Only the tail can be re-emitted, so the scan stays bounded. */
const MAX_OVERLAP = 400;

/** How much of the partial's opening a restart has to reproduce to be called a restart. */
const RESTART_PROBE = 48;

/** Read the incomplete marker off an assistant message's metadata. */
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

/**
 * assistant-ui's reason for each of ours. `length` is a truthful stop, not a failure:
 * mapping it to `error` paints MessagePrimitive.Error's red box and a Retry button over
 * a turn that already offers the Continue bar below it, on every reloaded max-tokens
 * answer including ones written before this feature existed. `interrupted` keeps `error`
 * on purpose -- a cut stream is the thing the user has to be told about.
 */
const STATUS_REASON: Record<
  IncompleteReason,
  "cancelled" | "length" | "error"
> = {
  cancelled: "cancelled",
  length: "length",
  interrupted: "error",
};

/** Restore assistant-ui's status without losing the product-specific stop reason. */
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
};

/** The user-facing explanation of why a turn stopped. */
export function incompleteLabel(reason: IncompleteReason): string {
  return INCOMPLETE_LABELS[reason];
}

/**
 * Drop text the continuation repeated from the end of the partial. Local models
 * continue token-exactly, but a provider that ignores assistant prefill can restate
 * the last few words. Only a suffix the continuation opens with is removed.
 */
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

/**
 * True when the "continuation" is really a fresh answer. Judged on the partial's
 * opening, which a genuine continuation never reproduces; appending a restart would
 * read as a stutter, so the caller keeps it alone.
 */
export function isRestart(partial: string, continuation: string): boolean {
  const head = partial.trimStart().slice(0, RESTART_PROBE);
  if (head.length < RESTART_PROBE) {
    // Too short to tell a restart from a coincidence.
    return false;
  }
  return continuation.trimStart().startsWith(head);
}

/**
 * Merge a partial answer with its continuation.
 *
 * `streaming` skips the restart check, which needs more text than early chunks carry;
 * it runs once at the end.
 */
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

/**
 * The merge a resumed turn applies to its cumulative text, streaming and at the end.
 *
 * The bug this fixes is `isRestart` running per arrival. It cannot fire until the continuation
 * reaches RESTART_PROBE characters, and the moment it does it publishes the continuation ALONE.
 * Over a 1602-character partial that took the value from 1649 characters to 48 in one arrival,
 * measured character by character. That is not only a visible collapse: Stop persists the last
 * STREAMED yield, because assistant-ui drops what a run yields after an abort and the terminal
 * merge sits behind `!abortSignal.aborted`, so stopping in that window SAVED the 48 characters
 * and discarded the whole partial the user was reading.
 *
 * `streaming` exists for exactly this and production never passed it. It suppresses the restart
 * check, which needs more text than early chunks carry, and leaves the overlap trim, which only
 * ever removes text the continuation itself repeated. So a streamed merge keeps the partial and
 * the new text, always, and a genuine restart is collapsed once at the end where the evidence
 * for it is complete.
 *
 * Two things were tried first and are recorded because both were worse:
 *
 *   * repairing ONLY at the end. That saves `partial + repeated tail` on Stop, so a Continue
 *     built on it replays the duplicate. It swapped a display bug for a stored one.
 *   * holding the partial until the trim could no longer change (at `min(partial.length,
 *     MAX_OVERLAP)`). Monotonic, but Stop inside that window persisted the STALE PARTIAL and
 *     discarded every newly generated character, which is worse than either.
 *
 * The overlap trim can still shift the join by up to MAX_OVERLAP characters as a longer overlap
 * starts matching, so a publish is not strictly monotonic. That is bounded, rare, confined to
 * external-provider continuations, and never loses text. It is deliberately not worth holding
 * generated output back to avoid.
 */
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

/**
 * Whether a finished MLX turn used its whole budget, i.e. stopped for length.
 *
 * MLX alone needs the inference: it reports finish_reason "stop" at the cap. Everyone
 * else reports "length" themselves, transformers included (from the token count
 * `generate` returned, which distinguishes an exhausted budget from a stop token that
 * happened to land at the cap), so inferring for them would relabel a finished answer.
 */
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

/**
 * Whether an assistant turn can be resumed at all.
 *
 * A turn that called a tool cannot: the continuation runs as a sibling, so the call and
 * its result are absent from the outbound history and the model would be asked to carry
 * on from an answer whose evidence is missing. Matches the backend guard.
 */
export function isContinuableContent(
  content: readonly unknown[] | undefined,
): boolean {
  if (!content) {
    return false;
  }
  let hasText = false;
  for (const part of content) {
    const type = (part as { type?: string })?.type;
    if (type === "text") {
      hasText = hasText || ((part as { text?: string }).text ?? "").length > 0;
      continue;
    }
    // Reasoning and citations are never replayed, so they neither block nor enable.
    if (type === "reasoning" || type === "source") {
      continue;
    }
    return false;
  }
  return hasText;
}

/**
 * The newest Gemini text-part thoughtSignature on an assistant turn. The streaming
 * adapter pins it onto the final text part; the continuation carries it so the resumed
 * turn is replayed signed rather than as bare text.
 */
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

/**
 * Providers that reject a trailing assistant turn.
 *
 * Anthropic removed assistant prefill in Claude 4.6 and never allowed it with extended
 * thinking. Gemini requires a multiturn request to end in a user turn or a function
 * response. Mistral takes a prefill only on an assistant message carrying `prefix: true`,
 * which neither the outbound message type nor the backend schema has. All get the partial
 * plus an instruction turn instead, keeping the last message a user turn.
 */
const PREFILL_REJECTING_PROVIDERS = new Set(["anthropic", "gemini", "mistral"]);

export function rejectsAssistantPrefill(
  providerType: string | undefined,
): boolean {
  return providerType != null && PREFILL_REJECTING_PROVIDERS.has(providerType);
}

/**
 * Self-hosted servers the backend sends the continuation flags to.
 *
 * Mirrors `_CONTINUATION_FLAG_PROVIDERS` in `core/inference/external_provider.py`: vLLM
 * and llama-server document `continue_final_message` + `add_generation_prompt`, so they
 * resume at the exact token boundary and need no overlap repair. Every other connection
 * may ignore the trailing assistant turn and restart.
 */
const EXACT_RESUME_PROVIDERS = new Set(["vllm", "llama_cpp"]);

export function resumesExactly(providerType: string | undefined): boolean {
  return providerType != null && EXACT_RESUME_PROVIDERS.has(providerType);
}

/**
 * Whether the run this thread would start can resume a partial at all.
 *
 * These two answer from scratch before the continuation request is read: audio input
 * re-listens and an audio-output model regenerates the clip. Continue would restart, so
 * it is hidden. Armed Deep Research is not here: the model decides whether an armed turn
 * becomes research, and a turn that did is hidden by its run, not by the toggle.
 */
export function modeAllowsContinuation({
  fromAudioInput,
  audioOutputModel,
}: {
  fromAudioInput: boolean;
  audioOutputModel: boolean;
}): boolean {
  return !(fromAudioInput || audioOutputModel);
}

/** Asks for a continuation when the partial cannot be sent as a prefill. */
export const CONTINUE_INSTRUCTION =
  "Continue your previous response from exactly where it stopped. " +
  "Do not repeat any text you already wrote and do not restate the answer.";

/** The `runConfig.custom` key carrying a continuation request to the chat adapter. */
export const CONTINUATION_RUN_CONFIG_KEY = "unslothContinuation";

export type ContinuationRequest = {
  /** The partial answer to resume, exactly as it was rendered. */
  partial: string;
  /**
   * Gemini text-part thoughtSignature from the turn being resumed. The sibling run drops
   * the original assistant message, so replaying it here keeps the history signed.
   */
  thoughtSignature?: string;
};

/** Read a continuation request out of a run's `runConfig`, if it is one. */
export function readContinuationRequest(
  runConfig: unknown,
): ContinuationRequest | null {
  const custom = (runConfig as { custom?: Record<string, unknown> } | undefined)
    ?.custom;
  const request = custom?.[CONTINUATION_RUN_CONFIG_KEY] as
    | { partial?: unknown; thoughtSignature?: unknown }
    | undefined;
  const partial = request?.partial;
  if (typeof partial === "string" && partial.length > 0) {
    const signature = request?.thoughtSignature;
    return typeof signature === "string" && signature
      ? { partial, thoughtSignature: signature }
      : { partial };
  }
  return null;
}

/**
 * Resuming a Max Tokens cut WITHOUT asking.
 *
 * Hitting the cap is not a decision the user made; it is the reply running out of room
 * mid-sentence, and the only sensible answer to "the answer is not finished" is to finish
 * it. Every other reason is left alone: `cancelled` is the user pressing Stop, so
 * resuming would restart the thing they just stopped, and `interrupted` means the
 * connection dropped, where a silent retry can hide a broken link.
 *
 * Bounded, because a model that will not stop would otherwise loop forever, and each
 * round grows the transcript and drives compaction harder. After the budget is spent the
 * bar comes back and the user decides.
 */
export const AUTO_CONTINUE_LIMIT = 3;

/**
 * Rounds already spent per logical turn, keyed by the parent the continuation hangs off.
 *
 * Keyed on the PARENT, not the message: a continuation runs as a sibling of the turn it
 * resumes, so each round produces a new message id and a per-message counter would reset
 * every time and never reach its limit. The parent is the one id every round of the same
 * turn shares. In memory only; a reload is a fresh decision by a present user.
 */
const spent = new Map<string, number>();

/**
 * Whether this cut should resume on its own.
 *
 * `fits` and `promptTarget` come from the resumed turn's own truncation metadata, and
 * both exist because resuming FIGHTS the context window. A continuation replays the
 * partial as the final assistant turn, and the fit protects the final group, so the
 * partial is the one thing compaction may not evict. Once it is large enough that the
 * system turn plus the carried-forward block will not fit beside it, the request is
 * irreducible and every further round produces the same refusal.
 *
 * Measured at a 4,864-token context: a 3,217-token partial plus system and X came to
 * 4,218 against a 3,648-token target, so the answer could not be resumed at all -- and
 * three automatic rounds each asked anyway and each failed identically.
 */
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
    /** `contextTruncation.fits` of the turn being resumed. */
    fits?: boolean;
    /** Estimated size of the partial that would be replayed. */
    partialTokens?: number;
    /** `contextTruncation.prompt_target`: what the window leaves for the prompt. */
    promptTarget?: number;
  } = {},
): boolean {
  if (reason !== "length" || !key) {
    return false;
  }
  if (fits === false) {
    // Neither shape of `fits: false` can be resumed. A refusal never fitted, and a rescue
    // is reached only once eviction ran out of eligible turns, so its prompt is already
    // the floor. Nothing is left to evict either way, while a continuation replays the
    // partial as the final assistant turn, which the fit protects, asking for MORE prompt
    // against the same window: on a 460-of-500-token rescue a 10-character partial
    // already refuses.
    return false;
  }
  if (
    typeof partialTokens === "number" &&
    typeof promptTarget === "number" &&
    promptTarget > 0 &&
    partialTokens >= promptTarget
  ) {
    // The partial alone meets the budget, so nothing can be sent beside it -- not the
    // system prompt, not the user's own question. Raising Context Length is the only
    // remedy and the bar says so.
    return false;
  }
  return (spent.get(key) ?? 0) < limit;
}

/** Record a round against `key`. Called before the run starts, so a run that fails to
 * produce anything still consumes its budget rather than retrying forever. */
export function recordAutoContinue(key: string): void {
  spent.set(key, (spent.get(key) ?? 0) + 1);
}

/** Rounds spent so far, for the indicator and for tests. */
export function autoContinueCount(key: string | null | undefined): number {
  return key ? (spent.get(key) ?? 0) : 0;
}

/**
 * How long a lease survives without being renewed.
 *
 * A lease, not a permanent flag: the tab that wins can be closed or crash mid-run, and a
 * flag it never gets to clear would leave the message unresumable for the life of the
 * profile. The winner renews for as long as its run is live (see
 * `renewAutoContinueLease`), so the TTL is not a guess at how long a continuation takes:
 * it is how long a tab that stopped renewing is still believed to be alive.
 *
 * Three minutes because renewals are timers, and a hidden tab is throttled to roughly one
 * timer callback a minute in Chrome. Three minutes leaves a backgrounded winner about
 * three chances to check in before anyone else may touch its message, while a tab that
 * really died gives the message back inside the pause a user notices anyway. The manual
 * Continue button is never gated on any of this, so the way out is always open.
 */
export const AUTO_CONTINUE_LEASE_TTL_MS = 180_000;

/**
 * How often the holder renews while its run is live. Well inside the TTL even at the
 * once-a-minute rate a hidden tab's timers are throttled to.
 */
export const AUTO_CONTINUE_LEASE_RENEW_MS = 30_000;

/**
 * How long a record stays behind once the run it covered has finished.
 *
 * A lease answers "is somebody running this message"; this answers "has this message
 * already been continued", which is what the per-tab claim set answers inside the tab that
 * won and what no other tab has any way of learning. Nothing tells a second tab that the
 * sibling was written: `notifyChatHistoryUpdated` dispatches a same-window event, no chat
 * key has a `storage` listener, and persisted messages are re-read only by `history.load`
 * when a thread is opened. So a tab still holding the pre-continuation branch in memory
 * takes the message straight back the moment the record goes -- on the next remount of the
 * bar, which leaving the chat and returning is enough to produce -- and the user pays for a
 * second continuation from a history that is missing the first.
 *
 * Long enough to outlive the sitting the continuation happened in, not permanent: the ids
 * are pruned like any other record, so the key cannot grow without bound.
 */
export const AUTO_CONTINUE_CONTINUED_TTL_MS = 86_400_000;

/** The `localStorage` key holding the leases, one record per claimed message id. */
export const AUTO_CONTINUE_LEASE_KEY = "unsloth_chat_auto_continue_leases";

/** The Web Locks name every read-modify-write of that key is taken under. */
export const AUTO_CONTINUE_LOCK_NAME = "unsloth_chat_auto_continue_claim";

/**
 * What a claim attempt did.
 *
 * Three answers, not a boolean, because the two ways of not starting a run want opposite
 * things on screen. `held-elsewhere` is another tab running this message: this one has to
 * drop the "Continuing automatically" spinner and put its manual Continue button back, or
 * it shows a spinner for a run it will never own. `skipped` is this tab declining its own
 * duplicate call -- a `<StrictMode>` replay, or a claim already in flight -- where the run
 * IS coming and changing what is on screen would be wrong.
 */
export type AutoContinueClaim = "started" | "skipped" | "held-elsewhere";

/** The slice of `Storage` the lease needs. Narrow so a test can hand over a fake. */
export type AutoContinueLeaseStorage = {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
};

/** The slice of `navigator.locks` the claim needs: an exclusive request by name. */
export type AutoContinueLockManager = {
  request<T>(name: string, callback: () => T | Promise<T>): Promise<T>;
};

type Lease = {
  token: string;
  expires: number;
  /**
   * The run this record covered reached a terminal state, so the message HAS been
   * continued. A plain lease that lapses means only that its holder stopped answering and
   * the message is free again; this one means the work is done and nobody should repeat it.
   */
  done?: boolean;
};

/**
 * The browser's own storage, or nothing when there is not one to read.
 *
 * Resolved per call rather than once at import: `localStorage` is absent under the test
 * runner and in SSR, and Safari's private mode throws on the property itself, not just on
 * a write. Every caller treats null as "no cross-tab seam" and falls back to the module
 * claim below, which is exactly what shipped before the lease existed.
 */
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

/**
 * The browser's lock manager, or nothing where there is not one.
 *
 * `navigator.locks` is what makes the claim exclusive: read, decide and write happen
 * inside one held lock, so no other tab can be between them. localStorage alone cannot do
 * this -- its individual operations are atomic, but a read-modify-write across three
 * statements is not, and the storage mutex that used to serialize such sequences was
 * dropped from the spec, so nothing holds a lock across them.
 */
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

/**
 * One tab's view of which messages have been continued.
 *
 * Constructible because a second instance is precisely what a second browser tab is: its
 * own module scope, its own empty claim set, the same saved thread, and the same storage
 * and lock manager underneath. That is the case the lease exists for, and the only way to
 * write it down as a test.
 */
export function createAutoContinueTab({
  storage,
  locks,
}: {
  /** Omit for the browser's `localStorage`; pass null for a tab with no seam. */
  storage?: AutoContinueLeaseStorage | null;
  /** Omit for `navigator.locks`; pass null for a browser that has none. */
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
  /**
   * Messages this tab has already continued automatically, by message id.
   *
   * Separate from `spent`, which counts rounds per logical turn and is what bounds a
   * runaway loop. This answers a different question: has THIS message already been
   * resumed once. A component-local ref cannot, because it dies with the component.
   * Leave the chat with a truncated branch selected and come back, and the ref is fresh
   * while the parent still has budget, so the effect fires again and creates another
   * sibling and another paid request. Module scope outlives the remount; the ids are
   * short and bounded by how many replies a session truncates.
   */
  const continued = new Set<string>();

  /** Claims this tab has in flight. The lock is async, so a second call can arrive. */
  const claiming = new Set<string>();

  /**
   * Lease tokens this tab holds, by message id, each tagged with the holder that took it.
   *
   * Tagged, because "this tab" is not one runtime. Compare mode mounts two of them side by
   * side, each with its own thread and its own run, and either can be resuming a truncated
   * reply while the other is idle. A holder renews and releases what IT took; the pane that
   * finishes first must not hand the other pane's still-generating message to anyone.
   */
  const ownTokens = new Map<string, { token: string; holder: string }>();

  const leaseSeam = (): AutoContinueLeaseStorage | null =>
    storage === undefined ? browserLeaseStorage() : storage;

  const lockSeam = (): AutoContinueLockManager | null =>
    locks === undefined ? browserLockManager() : locks;

  /**
   * Run `mutate` with nobody else inside it.
   *
   * With a lock manager this is genuinely exclusive across tabs. Without one -- an older
   * browser, an embedded webview, the test runner -- it degrades to running the body
   * straight through, which is the write-then-read-back below and no worse than what
   * shipped before. A lock manager that rejects the request is treated the same way,
   * and a body that has already run is never run twice.
   */
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

  /**
   * Drop what has lapsed, so the key cannot grow with every truncated reply.
   *
   * The one mutation here that is deliberately not per message. It only ever removes
   * records that are already dead by their own expiry -- `liveLease` reads them as free
   * whether they are in the key or not -- so it cannot end anybody's hold, and it is the
   * only thing bounding the size of a key every tab writes to.
   */
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

  /** A lease held by ANY tab, this one included, or null. Expired reads as free. */
  function liveLease(messageId: string, now: number): Lease | null {
    const store = leaseSeam();
    if (!store) {
      return null;
    }
    try {
      const lease = readLeases(store)[messageId];
      return lease && lease.expires > now ? lease : null;
    } catch {
      // Unreadable storage is no worse than no storage: fall back to module scope.
      return null;
    }
  }

  /**
   * Take the lease, or report who has it. Runs inside the lock.
   *
   * A storage that cannot be read or written is not an error: the tab keeps the
   * module-only claim, which is what single-tab use has always relied on, rather than
   * refusing a continuation the user is waiting for.
   */
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
      // Read back, which is the only defence left when there is no lock manager: whoever
      // wrote last is the token in storage, and the other tab does not find its own.
      if (readLeases(store)[messageId]?.token !== token) {
        return "held-elsewhere";
      }
    } catch {
      return "started";
    }
    ownTokens.set(messageId, { token, holder });
    return "started";
  }

  /** Whether `holder` is the one this tab took `messageId`'s lease for. */
  function heldFor(messageId: string, holder: string): boolean {
    return ownTokens.get(messageId)?.holder === holder;
  }

  /**
   * Rewrite ONE lease with a new expiry, if this holder is the one that took it. Runs
   * inside the lock, and reports whether the lease is still this tab's to stamp.
   *
   * One message, never a holder's whole set. A holder is a runtime, and a runtime outlives
   * the run inside it: rounds of the same turn are claimed one after another under the same
   * holder, and the next round is claimed BEFORE the finished one is given back, because
   * React runs the bar's effect before the thread's. Anything that operated on "every lease
   * this holder owns" would settle the round that just started along with the round that
   * ended.
   */
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
        // Lapsed, or taken over while this tab was not looking. Not ours to stamp.
        ownTokens.delete(messageId);
        return false;
      }
      leases[messageId] = done
        ? { token: held.token, expires, done: true }
        : { token: held.token, expires };
      store.setItem(AUTO_CONTINUE_LEASE_KEY, JSON.stringify(leases));
      return true;
    } catch {
      // An unwritable seam holds no lease of ours to renew or release either.
      return false;
    }
  }

  return {
    async claim(messageId, { now = Date.now(), holder = "" } = {}) {
      if (!messageId || continued.has(messageId) || claiming.has(messageId)) {
        return "skipped";
      }
      if (liveLease(messageId, now)) {
        // Cheap answer before touching the lock. Not recorded locally: if the tab holding
        // it dies, this one takes the message back when the lease lapses.
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
      // A claim of this tab's own that is still in flight deliberately does NOT count: the
      // spinner it is rendering is the right thing to show, and treating it as taken would
      // flash the manual bar in the middle of winning. Its duplicate calls are refused by
      // `claim` itself.
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
      // Held no longer, and marked done rather than handed back: the message HAS been
      // continued now, and the only thing that could hand it to another tab is a lease that
      // lapses, which is reserved for a holder that stopped answering. Only this message, so
      // the next round of the same turn -- claimed under this same holder before this
      // release lands -- keeps its own lease and its own renewals.
      ownTokens.delete(messageId);
    },
    /**
     * Take back a claim whose run was never issued.
     *
     * Only this tab's own record of having continued the message. The storage lease is left
     * exactly where it is, to run out its own TTL: that is what a tab closed mid-claim
     * already leaves behind, and dropping it here would hand the message to a second tab
     * while this one may still be deciding what to do with it.
     *
     * Without this the message stays in `continued` for the life of the tab, and every
     * later claim of it -- the branch picker back to the truncated sibling, or returning to
     * the chat -- answers "skipped", so the automatic continuation never runs again for a
     * request that was never made.
     */
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
      // Synchronous, and so outside the lock: a reset is a test seam and a new thread
      // starting from zero, neither of which races another tab for this key.
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

/** This tab. */
const tab = createAutoContinueTab();

/**
 * Claim `messageId` for an automatic continuation, and say what happened.
 *
 * Asynchronous because exclusivity is: the read-decide-write is taken under a Web Lock so
 * that two tabs cannot both come out of it believing they won. The caller starts its run
 * on `started` and only on `started`.
 */
export function claimAutoContinue(
  messageId: string | null | undefined,
  holder: string,
): Promise<AutoContinueClaim> {
  return tab.claim(messageId, { holder });
}

/**
 * Whether `messageId` is already being continued automatically -- by this tab, or by
 * another one whose lease is still live.
 *
 * Asked at render time, so the answer has to match what `claimAutoContinue` will do a tick
 * later as closely as a synchronous read can. It cannot be exact, because the lock that
 * decides a genuine race resolves later; the caller therefore also re-renders off the
 * claim's own answer rather than trusting this alone.
 */
export function wasAutoContinued(messageId: string | null | undefined): boolean {
  return tab.claimed(messageId);
}

/**
 * Give a claim back when the run it was taken for turned out never to be issued.
 *
 * The claim is what stops this tab continuing the same message twice, so it is only ever
 * dropped where nothing was started: the caller checked, in the same tick and off the same
 * store it starts the run from, that the message it claimed is no longer there to resume.
 * The cross-tab lease is deliberately untouched and lapses on its own.
 */
export function forgetAutoContinue(messageId: string | null | undefined): void {
  tab.forget(messageId);
}

/**
 * Keep ONE message's lease alive while the run resuming it is still going.
 *
 * Called on a timer for as long as that runtime's thread is running. Without it the TTL
 * would have to guess the longest continuation anyone might generate, and a slow local
 * model on a big Max Tokens would outlive that guess and have its message taken by another
 * tab mid-stream.
 *
 * Named by message and by holder, not by holder alone. The holder keeps two panes of
 * compare mode apart; the message id is what keeps one round of a turn apart from the next,
 * which the same holder claims moments later.
 */
export function renewAutoContinueLease(
  messageId: string,
  holder: string,
): Promise<void> {
  return tab.renew(messageId, holder);
}

/**
 * Give ONE message's lease back when the run resuming it reaches a terminal state,
 * cancelled included.
 *
 * It is cut to the settle window rather than deleted, so a tab still showing the
 * pre-continuation branch cannot immediately start the duplicate this file exists to
 * prevent, and the full TTL is left to mean one thing only: the holder stopped answering.
 *
 * By message, because a holder can own two at once and they are not in the same state. A
 * turn that hits Max Tokens again is claimed by the bar's effect before this thread's
 * effect observes the first run ending, so a holder-wide release settles a round that has
 * only just begun, and one settle window later another tab takes it.
 */
export function releaseAutoContinueLease(
  messageId: string,
  holder: string,
): Promise<void> {
  return tab.release(messageId, holder);
}

/**
 * Whether the thread a lease was taken for is generating right now.
 *
 * Deliberately NOT "is the thread on screen running". A run keeps streaming when the user
 * opens another chat -- the chat view is keyed by project so the provider is not remounted,
 * and the runtime just points its selection somewhere else -- so a lease whose lifetime is
 * read off the selected thread is released the moment the user looks away, and one settle
 * window later another tab can claim a message that is still being written. The signal has
 * to be per thread and indifferent to selection.
 */
export type AutoContinueRunSignal = {
  isRunning(threadId: string): boolean;
  subscribe(onChange: () => void): () => void;
};

/**
 * Holds the lease of each continuation this tab is running, for as long as its own run runs.
 *
 * A hold is (message, thread) and its lifetime is the run's: it arms when THAT thread starts
 * generating and is given back when THAT thread stops, whatever is on screen meanwhile and
 * whichever component has mounted or unmounted. Nothing here reads React state.
 *
 * Arming waits for a run that starts after the hold is taken. A round that hits Max Tokens
 * again is claimed while the finished round is still winding down, so a hold that armed on
 * "the thread is running" would arm on its predecessor's run and be released when THAT one
 * ended, mid-stream. So a hold taken while the thread is busy waits to see it idle first.
 *
 * A hold whose run has not appeared YET is kept and renewed, never timed out. No deadline
 * separates "this run is never coming" from "this run is still in preflight", because the
 * preflight has no bound: `chat-adapter.ts` awaits the thread's own settings pairing (up to
 * 30 seconds by itself), then `waitForModelReady`, which polls every 500ms for as long as a
 * model is loading -- minutes for a large local GGUF, and exactly the state a tab just
 * opened on a truncated reply is in. A fixed arming timeout dropped the hold in the middle
 * of that, its renewals stopped, and the lease lapsed under a run that had since started
 * streaming, so another tab could claim the message and pay for the same continuation twice.
 * Renewing instead costs only this: a claim whose run genuinely never starts holds its lease
 * until the tab is closed, and the manual Continue button is never gated on any of it.
 */
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
  observe: () => void;
  failed: (threadId: string) => void;
  tick: () => void;
  held: () => number;
  stop: () => void;
} {
  type Hold = {
    messageId: string;
    threadId: string;
    /** Seen idle since the hold was taken, so the next run to start is this hold's own. */
    idle: boolean;
    /** That run has started. Only an armed hold is ever released. */
    armed: boolean;
  };
  const holds = new Map<string, Hold>();
  let unsubscribe: (() => void) | null = null;

  const key = (messageId: string, threadId: string) =>
    `${threadId}\u0000${messageId}`;

  /** Arm, release, or forget. No writes to storage beyond the release itself. */
  function observe(): void {
    const at = now();
    for (const [id, hold] of [...holds]) {
      if (signal.isRunning(hold.threadId)) {
        // Only a run that started after this hold was taken can be its own.
        hold.armed ||= hold.idle;
        continue;
      }
      hold.idle = true;
      if (hold.armed) {
        // This hold's own run has ended: finished, cancelled or failed.
        holds.delete(id);
        release(hold.messageId, hold.threadId, at);
        continue;
      }
      // Not armed yet, so its run is still in preflight, which has no upper bound (see the
      // note above this function). Kept and renewed rather than timed out: dropping it here
      // stopped the renewals while the run was still on its way, and the lease then lapsed
      // under a live continuation.
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
        // Claimed while the thread is between runs, which is the ordinary case: the bar
        // only fires on a reply that has finished.
        idle: !signal.isRunning(threadId),
        armed: false,
      });
      unsubscribe ??= signal.subscribe(observe);
    },
    observe,
    /**
     * `threadId`'s run failed on its way out, before it ever reached the run signal.
     *
     * The one thing that can end a hold which never armed, and it is a fact rather than a
     * deadline: the adapter threw, so this run is not slow, it is over. Not a timeout by
     * another name -- a preflight that is merely long emits nothing here and keeps its hold
     * and its renewals, which is the whole reason arming has no deadline.
     *
     * Armed holds are left alone. Their run did reach the signal, so the thread going idle
     * is what settles them, with the `done` marker that says the message HAS been continued
     * -- which a run that never produced a token has no right to write. This one only
     * discards: the lease it stops renewing lapses on its own TTL and the message comes
     * back to whoever is still looking at it, rather than being renewed for the life of the
     * tab and refused to every other one.
     */
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
        // Armed or still waiting for its run: either way this tab is alive and expects to
        // be running that message, which is the whole question the lease asks.
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

/**
 * Whether THIS message is the one to continue automatically.
 *
 * `shouldAutoContinue` answers about the turn: is the reason right, does the partial
 * still fit, is the round budget unspent. It keeps saying yes after a message has been
 * claimed, because the budget is per turn and one spent round out of three leaves the
 * turn resumable, while the claim is per message and is what actually decides whether a
 * run starts. Anything rendering off the turn's answer alone therefore reports an
 * already-claimed message as continuing while `claimAutoContinue` refuses to start
 * anything: a spinner that never resolves, over the manual Continue button it replaces.
 *
 * Reachable without a reload. The continuation runs as a sibling and the branch picker
 * leads straight back to the truncated partial, and leaving the chat and returning lands
 * on it too whenever it is still the selected branch.
 */
export function shouldAutoContinueMessage(
  messageId: string | null | undefined,
  reason: IncompleteReason | null | undefined,
  key: string | null | undefined,
  options: Parameters<typeof shouldAutoContinue>[2] = {},
): boolean {
  if (wasAutoContinued(messageId)) {
    return false;
  }
  return shouldAutoContinue(reason, key, options);
}

/**
 * Test seam; also lets a new thread start from zero.
 *
 * A full reset gives back the leases this tab wrote as well, so "start from zero" means
 * the same thing it did before the lease existed. Leases other tabs hold are left alone:
 * they are not this tab's to release, and a run they are driving is still going.
 *
 * Deliberately wholesale, unlike renewal and release, which are per holder. A run ending
 * is one runtime saying it is finished, and says nothing about the others; a reset is the
 * whole tab being told it has no claims at all, which is what a test between cases and a
 * cleared history both mean. Nothing in the run lifecycle calls it.
 */
export function resetAutoContinue(key?: string): void {
  if (key === undefined) {
    spent.clear();
    tab.reset();
  } else {
    spent.delete(key);
  }
}
