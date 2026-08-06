// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function isBrowserOffline(): boolean {
  return isNavigatorOffline();
}

const NETWORK_STATUS_EVENT = "unsloth-network-status";
const REMOTE_OFFLINE_TTL_MS = 30_000;
const HUGGING_FACE_ORIGIN = "https://huggingface.co";
const noopUnsubscribe = () => undefined;

type RemoteNetworkScope = string | readonly string[];

/**
 * Why a Hub request failed. Browsers collapse CORS, DNS, TLS interception and
 * real outages into one opaque TypeError, so "network-opaque" says what we can
 * prove, not what happened. Only CSP can be named, via its violation event.
 */
export type HubFailureKind =
  | "aborted"
  | "timeout"
  | "browser-offline"
  | "network-opaque"
  | "unknown";

export interface HubFailure {
  kind: HubFailureKind;
  /** Already sanitised: safe to render. Never contains a full URL or a token. */
  message: string;
  /** Origin only, never the full request URL (which carries the search query). */
  origin: string | null;
  status?: number;
  retryable: boolean;
}

/** Error thrown by fetchWithTimeout carrying a classified, renderable failure. */
export class HubFetchError extends Error {
  readonly failure: HubFailure;

  constructor(failure: HubFailure, options?: { cause?: unknown }) {
    super(failure.message, options);
    this.name = "HubFetchError";
    this.failure = failure;
  }
}

export function isHubFetchError(error: unknown): error is HubFetchError {
  return error instanceof HubFetchError;
}

const remoteOfflineUntilByOrigin = new Map<string, number>();
// The TTL controls when we retry; this controls what we tell the user. Cleared
// only by a success, so the cause outlives the backoff window and the panel can
// still say why after the window has lapsed.
const lastFailureByOrigin = new Map<string, HubFailure>();

function isNavigatorOffline(): boolean {
  return typeof navigator !== "undefined" && navigator.onLine === false;
}

function emitNetworkStatusChange(): void {
  if (typeof window === "undefined") {
    return;
  }
  window.dispatchEvent(new Event(NETWORK_STATUS_EVENT));
}

export function getBrowserOfflineRetryDelayMs(): number {
  // Keyed off the empirical remote-offline TTL, not navigator.onLine, so
  // recovery doesn't stall on platforms where navigator.onLine is stuck false.
  // The earliest live window, not the latest: the phase reads one feed's, so
  // waking on the longest would leave it reporting a state it had already left.
  return Math.max(
    0,
    getEarliestRemoteOfflineUntil() - Date.now(),
  );
}

function normalizeScope(scope: RemoteNetworkScope): readonly string[] {
  return typeof scope === "string" ? [scope] : scope;
}

function offlineUntil(origin: string): number {
  const value = remoteOfflineUntilByOrigin.get(origin) ?? 0;
  if (value <= Date.now()) {
    remoteOfflineUntilByOrigin.delete(origin);
    return 0;
  }
  return value;
}

function getRemoteOfflineUntil(scope: RemoteNetworkScope): number {
  let until = 0;
  for (const origin of normalizeScope(scope)) {
    until = Math.max(until, offlineUntil(origin));
  }
  return until;
}

/** The soonest live window anywhere, or 0 when nothing is backing off. */
function getEarliestRemoteOfflineUntil(): number {
  const now = Date.now();
  let until = 0;
  for (const [origin, value] of remoteOfflineUntilByOrigin) {
    if (value <= now) {
      remoteOfflineUntilByOrigin.delete(origin);
      continue;
    }
    if (until === 0 || value < until) {
      until = value;
    }
  }
  return until;
}

export function isRemoteNetworkOffline(
  scope: RemoteNetworkScope = HUGGING_FACE_ORIGIN,
): boolean {
  return getRemoteOfflineUntil(scope) > Date.now();
}

export function isHuggingFaceOffline(): boolean {
  // navigator.onLine is advisory only (false-reports offline on WSL2 / some
  // WebKitGTK/Tauri webviews). The authoritative signal is the empirical
  // remote-offline TTL, set when a real fetch fails and cleared on next success;
  // navigator's online/offline events still drive re-evaluation.
  return isRemoteNetworkOffline(HUGGING_FACE_ORIGIN);
}

/**
 * Availability of a Hub origin. A lapsed backoff means "probing", not
 * "available": only a success promotes an origin, which stops the flapping.
 */
export type HubPhase = "available" | "probing" | "unavailable";

export function getHubPhase(origin: string = HUGGING_FACE_ORIGIN): HubPhase {
  if (!lastFailureByOrigin.has(origin)) {
    return "available";
  }
  return isRemoteNetworkOffline(origin) ? "unavailable" : "probing";
}

export function getLastHubFailure(
  origin: string = HUGGING_FACE_ORIGIN,
): HubFailure | null {
  return lastFailureByOrigin.get(origin) ?? null;
}

export function markRemoteNetworkOnline(origin?: string): void {
  if (origin === undefined) {
    if (
      remoteOfflineUntilByOrigin.size === 0 &&
      lastFailureByOrigin.size === 0
    ) {
      return;
    }
    remoteOfflineUntilByOrigin.clear();
    lastFailureByOrigin.clear();
    emitNetworkStatusChange();
    return;
  }
  // The cause goes with the window: a success is what proves the block lifted.
  const hadWindow = remoteOfflineUntilByOrigin.delete(origin);
  const hadFailure = lastFailureByOrigin.delete(origin);
  if (!hadWindow && !hadFailure) {
    return;
  }
  emitNetworkStatusChange();
}

export function markRemoteNetworkOffline(
  originOrTtl: string | number = HUGGING_FACE_ORIGIN,
  ttlMs = REMOTE_OFFLINE_TTL_MS,
  failure?: HubFailure,
): void {
  const origin =
    typeof originOrTtl === "string" ? originOrTtl : HUGGING_FACE_ORIGIN;
  const ttl = typeof originOrTtl === "number" ? originOrTtl : ttlMs;
  const nextUntil = Date.now() + ttl;
  const previousUntil = remoteOfflineUntilByOrigin.get(origin) ?? 0;
  // The cause has to describe the window in force: recording a newer cause while
  // keeping a longer window left the panel naming a spent failure while a
  // different, still-live one held it unavailable. A first cause is always
  // taken, so nothing the user sees goes unexplained.
  const takesWindow = nextUntil > previousUntil;
  const records =
    failure !== undefined && (takesWindow || !lastFailureByOrigin.has(origin));
  const failureChanged =
    records && lastFailureByOrigin.get(origin)?.kind !== failure?.kind;
  if (records && failure !== undefined) {
    lastFailureByOrigin.set(origin, failure);
  }
  if (!takesWindow) {
    if (failureChanged) {
      emitNetworkStatusChange();
    }
    return;
  }
  remoteOfflineUntilByOrigin.set(origin, nextUntil);
  emitNetworkStatusChange();
}

/** Let Retry re-probe now. The failure stays until a request succeeds. */
export function clearRemoteBackoff(
  origin: string = HUGGING_FACE_ORIGIN,
): void {
  if (!remoteOfflineUntilByOrigin.delete(origin)) {
    return;
  }
  emitNetworkStatusChange();
}

export function subscribeNetworkStatus(listener: () => void): () => void {
  if (typeof window === "undefined") {
    return noopUnsubscribe;
  }
  window.addEventListener("online", listener);
  window.addEventListener("offline", listener);
  window.addEventListener(NETWORK_STATUS_EVENT, listener);
  return () => {
    window.removeEventListener("online", listener);
    window.removeEventListener("offline", listener);
    window.removeEventListener(NETWORK_STATUS_EVENT, listener);
  };
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

function isNetworkFetchError(error: unknown): boolean {
  if (isAbortError(error)) {
    return false;
  }
  return error instanceof TypeError;
}

function originFromFetchInput(
  input: Parameters<typeof fetch>[0],
): string | null {
  try {
    const raw =
      typeof input === "string"
        ? input
        : input instanceof URL
          ? input.toString()
          : input.url;
    const base =
      typeof window !== "undefined" ? window.location.href : "http://localhost";
    return new URL(raw, base).origin;
  } catch {
    return null;
  }
}

function hostLabel(origin: string | null): string {
  if (!origin) {
    return "Hugging Face";
  }
  try {
    return new URL(origin).host;
  } catch {
    return origin;
  }
}

/**
 * Build a renderable failure. Drops the request URL: it carries the search query
 * and can carry an internal hostname. Only the origin's host survives.
 */
export function classifyFetchFailure(
  error: unknown,
  origin: string | null,
  options: { timedOut?: boolean } = {},
): HubFailure {
  const host = hostLabel(origin);
  if (options.timedOut) {
    return {
      kind: "timeout",
      message: `The request to ${host} timed out.`,
      origin,
      retryable: true,
    };
  }
  if (isAbortError(error)) {
    return {
      kind: "aborted",
      message: "The request was cancelled.",
      origin,
      retryable: true,
    };
  }
  if (isNetworkFetchError(error)) {
    if (isNavigatorOffline()) {
      return {
        kind: "browser-offline",
        message: "This browser reports no network connection.",
        origin,
        retryable: true,
      };
    }
    return {
      kind: "network-opaque",
      message: `The browser could not reach ${host}. A DNS or content filter, TLS-inspecting antivirus, a browser extension, or a CORS policy can all cause this, and the browser does not say which.`,
      origin,
      retryable: true,
    };
  }
  return {
    kind: "unknown",
    message: `The request to ${host} failed.`,
    origin,
    retryable: true,
  };
}

/**
 * Drop the trailer @huggingface/hub's createApiError appends to every message
 * ("... URL: <full request url>. Request ID: ..."). The URL carries the user's
 * search query, and on a private deployment the mirror's hostname.
 */
export function sanitizeHubErrorMessage(message: string): string {
  if (!message) return message;
  const cleaned = message.replace(/\.?\s*URL:\s*\S+(\.\s*Request ID:\s*\S+)?\.?\s*$/, "");
  return cleaned.trim() || message;
}

export async function fetchWithTimeout(
  input: Parameters<typeof fetch>[0],
  init: Parameters<typeof fetch>[1] = {},
  timeoutMs = 15_000,
): Promise<Response> {
  const parentSignal = init.signal;
  const controller = new AbortController();
  let timedOut = false;
  const timeout = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);
  const abortFromParent = () => controller.abort();

  if (parentSignal?.aborted) {
    abortFromParent();
  } else {
    parentSignal?.addEventListener("abort", abortFromParent, { once: true });
  }

  const origin = originFromFetchInput(input);

  try {
    const response = await fetch(input, { ...init, signal: controller.signal });
    if (origin) {
      markRemoteNetworkOnline(origin);
    }
    return response;
  } catch (error) {
    // A superseded query must not blacklist the Hub or overwrite a diagnosis.
    if (parentSignal?.aborted && !timedOut) {
      throw error;
    }
    const failure = classifyFetchFailure(error, origin, { timedOut });
    if (origin && (timedOut || isNetworkFetchError(error))) {
      markRemoteNetworkOffline(origin, REMOTE_OFFLINE_TTL_MS, failure);
    }
    throw new HubFetchError(failure, { cause: error });
  } finally {
    clearTimeout(timeout);
    parentSignal?.removeEventListener("abort", abortFromParent);
  }
}
