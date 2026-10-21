// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Which saved Hugging Face token the Hub refused (a 401 that then succeeded anonymously).
 * Keyed by a fingerprint, never the token; dependency free so hub-fetch runs under bare node. */

/** "rejected" when a token is newly refused, "cleared" when a refusal is forgotten. */
export type HfTokenRejectionEvent = "rejected" | "cleared";

type Listener = (event: HfTokenRejectionEvent) => void;

/** After this long the token is tried again, so a briefly failing verifier cannot hide private
 * repos for the whole session. */
export const HF_TOKEN_REJECTION_RECHECK_MS = 10 * 60 * 1000;

// One refusal per Hub (scope): a mirror can refuse a token another endpoint accepts.
const refusals = new Map<string | null, { fingerprint: string; at: number }>();
let version = 0;
const listeners = new Set<Listener>();
// Accepted-token evidence: a refusal older than a newer success is ignored.
let successes = 0;
const lastSuccess = new Map<string | null, number>();
let lastSuccessEverywhere = 0;

/** FNV-1a over the token: enough to tell two tokens apart, useless for recovering one. */
function fingerprint(token: string): string {
  let hash = 0x811c9dc5;
  for (let i = 0; i < token.length; i += 1) {
    hash ^= token.charCodeAt(i);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return `${token.length}:${hash.toString(16)}`;
}

function normalized(token: string | null | undefined): string {
  return token?.trim() ?? "";
}

function notify(event: HfTokenRejectionEvent): void {
  version += 1;
  for (const listener of listeners) {
    try {
      listener(event);
    } catch {
      // One broken subscriber must not stop the others hearing about it.
    }
  }
}

/** A mark to pass to ``noteHfTokenRejected`` from a request that is about to start. */
export function hfTokenRejectionMark(): number {
  return successes;
}

/** Record that the Hub at *scope* refused *token*. True only when this is news for that Hub.
 * With *since* (a mark taken when the request started), a refusal that Hub has answered with
 * an accepted token after that mark is stale and ignored. */
export function noteHfTokenRejected(
  token: string | null | undefined,
  scope: string | null = null,
  since?: number,
): boolean {
  const value = normalized(token);
  if (!value) return false;
  if (since !== undefined && Math.max(lastSuccessEverywhere, lastSuccess.get(scope) ?? 0) > since) {
    return false;
  }
  const next = fingerprint(value);
  const known = refusals.get(scope);
  refusals.set(scope, { fingerprint: next, at: Date.now() });
  if (known?.fingerprint === next) return false;
  notify("rejected");
  return true;
}

/** Whether the Hub at *scope* refused *token* recently enough to skip it. Without a scope:
 * whether any Hub has refused it this session. A different token starts clean. */
export function isHfTokenRejected(
  token: string | null | undefined,
  scope?: string | null,
): boolean {
  const value = normalized(token);
  if (!value) return false;
  const wanted = fingerprint(value);
  if (scope === undefined) {
    return [...refusals.values()].some((entry) => entry.fingerprint === wanted);
  }
  const entry = refusals.get(scope);
  return (
    entry?.fingerprint === wanted && Date.now() - entry.at < HF_TOKEN_REJECTION_RECHECK_MS
  );
}

/** Whether any token is currently recorded as refused. */
export function hasRejectedHfToken(): boolean {
  return refusals.size > 0;
}

/** Forget the refusal by *scope* (every refusal without one) and record the success. */
export function clearHfTokenRejected(scope?: string | null): void {
  successes += 1;
  if (scope === undefined) lastSuccessEverywhere = successes;
  else lastSuccess.set(scope, successes);
  const changed = scope === undefined ? refusals.size > 0 : refusals.has(scope);
  if (!changed) return;
  if (scope === undefined) refusals.clear();
  else refusals.delete(scope);
  notify("cleared");
}

/** For useSyncExternalStore: changes whenever the rejected token does. */
export function hfTokenRejectionVersion(): number {
  return version;
}

export function subscribeHfTokenRejected(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
