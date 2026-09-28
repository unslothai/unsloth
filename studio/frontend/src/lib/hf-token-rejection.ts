// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Which saved Hugging Face token the Hub refused, so the UI can say so instead of "offline".
 *
 * A token the Hub accepts gets 404 for a repo it cannot see, never 401, so a 401 on a read that
 * carried a token which then succeeded anonymously means the token itself was refused (an
 * expired OAuth token, a revoked key). Keyed by a fingerprint, never the token, and dependency
 * free so hub-fetch stays importable under bare node. */

/** "rejected" when a token is newly refused, "cleared" when a refusal is forgotten. */
export type HfTokenRejectionEvent = "rejected" | "cleared";

type Listener = (event: HfTokenRejectionEvent) => void;

/** After this long the token is tried again: a verifier that failed for a while must not hide
 * private and gated repos for the rest of the session. Still refused, it is recorded again
 * without a second notification. */
export const HF_TOKEN_REJECTION_RECHECK_MS = 10 * 60 * 1000;

// One refusal per Hub (scope): a mirror or the datasets server can refuse a token another
// endpoint accepts, and recording one must not forget another.
const refusals = new Map<string | null, { fingerprint: string; at: number }>();
let version = 0;
const listeners = new Set<Listener>();

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

/** Record that the Hub at *scope* refused *token*. True only when this is news for that Hub. */
export function noteHfTokenRejected(
  token: string | null | undefined,
  scope: string | null = null,
): boolean {
  const value = normalized(token);
  if (!value) return false;
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

/** Forget the refusal by *scope*, or every refusal without one. */
export function clearHfTokenRejected(scope?: string | null): void {
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
