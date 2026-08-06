// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export function isBrowserOffline(): boolean {
  return isNavigatorOffline();
}

const NETWORK_STATUS_EVENT = "unsloth-network-status";
const REMOTE_OFFLINE_TTL_MS = 30_000;
// Discovery and repository pages are served by the main Hugging Face origin.
// Keep optional services such as datasets-server separate so an outage there
// cannot make the whole Hub appear offline.
const HUGGING_FACE_ORIGIN = "https://huggingface.co";
export const DATASETS_SERVER_ORIGIN = "https://datasets-server.huggingface.co";
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
  | "csp-blocked"
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
  effectiveDirective?: string;
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

// Keyed like the failures, not by origin alone. A block can be per-path, so an
// avatar succeeding does not prove the listing works: an origin-wide window let
// that success retire the feed's backoff and resume probing a still-dead feed.
const remoteOfflineUntilByKey = new Map<string, number>();
// The TTL controls when we retry; this controls what we tell the user. Cleared
// only by a success, so the cause outlives the backoff window.
const lastFailureByKey = new Map<string, HubFailure>();

/**
 * Which feed a request belongs to. A block can be per-path, so the catalog panel
 * reads discovery's own history: an avatar result, good or bad, cannot move it.
 *
 * "info" is the repo-path lookup the listing runs alongside itself. Its failure
 * is swallowed by its caller, so nothing gates on it; on either of the other
 * keys it would instead retire the catalog's diagnosis or, worse, suppress
 * every asset client for 30s over a lookup that nobody was waiting on.
 */
export type HubService = "discovery" | "other" | "info";

const HUB_SERVICES: readonly HubService[] = ["discovery", "other", "info"];
// What the origin-wide reachability question reads. "info" is swept by Retry,
// which clears everything on the origin, but must not answer "can this browser
// reach the Hub" for the clients that do wait on an answer.
const GATING_SERVICES: readonly HubService[] = ["discovery", "other"];

function failureKey(origin: string, service: HubService): string {
  return `${service}|${origin}`;
}

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

function offlineUntil(origin: string, service: HubService): number {
  const key = failureKey(origin, service);
  const value = remoteOfflineUntilByKey.get(key) ?? 0;
  if (value <= Date.now()) {
    remoteOfflineUntilByKey.delete(key);
    return 0;
  }
  return value;
}

function getRemoteOfflineUntil(
  scope: RemoteNetworkScope,
  service?: HubService,
): number {
  const services = service ? [service] : GATING_SERVICES;
  let until = 0;
  for (const origin of normalizeScope(scope)) {
    for (const s of services) {
      until = Math.max(until, offlineUntil(origin, s));
    }
  }
  return until;
}

/**
 * The soonest live window anywhere, or 0 when nothing is backing off. Every
 * origin, not just the Hub's: a client gated on another one (dataset sizes read
 * datasets-server) would otherwise back off with nothing scheduled to wake it,
 * and its own gate is shut, so no request could report the recovery either.
 */
function getEarliestRemoteOfflineUntil(): number {
  const now = Date.now();
  let until = 0;
  for (const [key, value] of remoteOfflineUntilByKey) {
    if (value <= now) {
      remoteOfflineUntilByKey.delete(key);
      continue;
    }
    if (until === 0 || value < until) {
      until = value;
    }
  }
  return until;
}

/** Omit `service` to ask whether anything is backing off this origin. */
export function isRemoteNetworkOffline(
  scope: RemoteNetworkScope = HUGGING_FACE_ORIGIN,
  service?: HubService,
): boolean {
  return getRemoteOfflineUntil(scope, service) > Date.now();
}

/**
 * For the clients that fetch repo assets directly (repo cards, owner avatars,
 * quant listings). They ask whether this browser can reach the Hub for *their*
 * requests, so a blocked catalog must not suppress them: under a per-path block
 * the listing can be dead while everything else answers, and their own success
 * is what clears this.
 */
export function isDirectHubOffline(
  origin: string = HUGGING_FACE_ORIGIN,
): boolean {
  return isRemoteNetworkOffline(origin, "other");
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

export function getHubPhase(
  origin: string = HUGGING_FACE_ORIGIN,
  service: HubService = "discovery",
): HubPhase {
  if (!lastFailureByKey.has(failureKey(origin, service))) {
    return "available";
  }
  return isRemoteNetworkOffline(origin, service) ? "unavailable" : "probing";
}

export function getLastHubFailure(
  origin: string = HUGGING_FACE_ORIGIN,
  service: HubService = "discovery",
): HubFailure | null {
  return lastFailureByKey.get(failureKey(origin, service)) ?? null;
}

export function markRemoteNetworkOnline(
  origin?: string,
  service: HubService = "discovery",
): void {
  if (origin === undefined) {
    if (remoteOfflineUntilByKey.size === 0 && lastFailureByKey.size === 0) {
      return;
    }
    remoteOfflineUntilByKey.clear();
    lastFailureByKey.clear();
    emitNetworkStatusChange();
    return;
  }
  // Both the window and the cause are per feed, so a success retires only its
  // own: under a per-path block one feed recovering says nothing about another.
  const key = failureKey(origin, service);
  const hadWindow = remoteOfflineUntilByKey.delete(key);
  const hadFailure = lastFailureByKey.delete(key);
  if (!hadWindow && !hadFailure) {
    return;
  }
  emitNetworkStatusChange();
}

export function markRemoteNetworkOffline(
  originOrTtl: string | number = HUGGING_FACE_ORIGIN,
  ttlMs = REMOTE_OFFLINE_TTL_MS,
  failure?: HubFailure,
  service: HubService = "discovery",
): void {
  const origin =
    typeof originOrTtl === "string"
      ? originOrTtl
      : HUGGING_FACE_ORIGIN;
  const ttl = typeof originOrTtl === "number" ? originOrTtl : ttlMs;
  const nextUntil = Date.now() + ttl;
  const key = failureKey(origin, service);
  const previousUntil = remoteOfflineUntilByKey.get(key) ?? 0;
  // The cause has to describe the window that is in force. Recording a newer
  // cause while keeping a longer window left the panel naming an older
  // response while a different, still-live failure was what held it
  // unavailable. A first cause is always taken, so nothing goes unexplained.
  const takesWindow = nextUntil > previousUntil;
  const records =
    failure !== undefined && (takesWindow || !lastFailureByKey.has(key));
  const failureChanged =
    records && lastFailureByKey.get(key)?.kind !== failure?.kind;
  if (records && failure !== undefined) {
    lastFailureByKey.set(key, failure);
  }
  if (!takesWindow) {
    if (failureChanged) {
      emitNetworkStatusChange();
    }
    return;
  }
  remoteOfflineUntilByKey.set(key, nextUntil);
  emitNetworkStatusChange();
}

/** Let Retry re-probe now. The failure stays until a request succeeds. */
export function clearRemoteBackoff(
  origin: string = HUGGING_FACE_ORIGIN,
): void {
  // An explicit Retry tests the network now, so every feed's window goes.
  let cleared = false;
  for (const service of HUB_SERVICES) {
    cleared = remoteOfflineUntilByKey.delete(failureKey(origin, service)) || cleared;
  }
  if (!cleared) {
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

// ---------------------------------------------------------------------------
// CSP correlation
// ---------------------------------------------------------------------------

// Violations fire on the document, not the rejected promise, so attribution
// means remembering them briefly and matching by origin.
const CSP_VIOLATION_TTL_MS = 3_000;
const cspViolationsByOrigin = new Map<
  string,
  { at: number; effectiveDirective: string }
>();
let cspListenerInstalled = false;

function recordCspViolation(event: SecurityPolicyViolationEvent): void {
  const directive = event.effectiveDirective || event.violatedDirective || "";
  // Only connect-src can explain a failed fetch.
  if (!directive.startsWith("connect-src")) {
    return;
  }
  const origin = originFromFetchInput(event.blockedURI);
  if (!origin) {
    return;
  }
  cspViolationsByOrigin.set(origin, {
    at: Date.now(),
    effectiveDirective: directive,
  });
}

export function installCspViolationListener(): () => void {
  if (typeof document === "undefined" || cspListenerInstalled) {
    return noopUnsubscribe;
  }
  cspListenerInstalled = true;
  document.addEventListener("securitypolicyviolation", recordCspViolation);
  return () => {
    document.removeEventListener("securitypolicyviolation", recordCspViolation);
    cspListenerInstalled = false;
  };
}

function takeCspViolation(
  origin: string | null,
  since: number,
): { effectiveDirective: string } | null {
  if (!origin) {
    return null;
  }
  const hit = cspViolationsByOrigin.get(origin);
  if (!hit) {
    return null;
  }
  const now = Date.now();
  if (now - hit.at > CSP_VIOLATION_TTL_MS || hit.at < since) {
    cspViolationsByOrigin.delete(origin);
    return null;
  }
  // Kept for the rest of its TTL rather than consumed. Concurrent requests to
  // one origin fail under one policy, so consuming it let the second be
  // classified network-opaque and overwrite the more actionable diagnosis.
  return { effectiveDirective: hit.effectiveDirective };
}

/**
 * Build a renderable failure. Drops the request URL: it carries the search query
 * and can carry an internal hostname. Only the origin's host survives.
 */
export function classifyFetchFailure(
  error: unknown,
  origin: string | null,
  options: { timedOut?: boolean; startedAt?: number } = {},
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
  const csp = takeCspViolation(origin, options.startedAt ?? 0);
  if (csp) {
    return {
      kind: "csp-blocked",
      message: `The browser blocked the connection to ${host} under Content Security Policy (${csp.effectiveDirective}).`,
      origin,
      effectiveDirective: csp.effectiveDirective,
      retryable: false,
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
  options: { service?: HubService } = {},
): Promise<Response> {
  // Defaults to the feed with a panel: an unmarked caller's failure is one the
  // user should see. Auxiliary clients pass "other".
  const service = options.service ?? "discovery";
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
  const startedAt = Date.now();

  try {
    const response = await fetch(input, { ...init, signal: controller.signal });
    if (origin) {
      markRemoteNetworkOnline(origin, service);
    }
    return response;
  } catch (error) {
    // A superseded query must not blacklist the Hub or overwrite a diagnosis.
    if (parentSignal?.aborted && !timedOut) {
      throw error;
    }
    const failure = classifyFetchFailure(error, origin, { timedOut, startedAt });
    if (origin && (timedOut || isNetworkFetchError(error))) {
      markRemoteNetworkOffline(origin, REMOTE_OFFLINE_TTL_MS, failure, service);
    }
    throw new HubFetchError(failure, { cause: error });
  } finally {
    clearTimeout(timeout);
    parentSignal?.removeEventListener("abort", abortFromParent);
  }
}
