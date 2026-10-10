// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Keyed by fingerprint, never the token; dependency free so hub-fetch runs under bare node. */

export type HfTokenRejectionEvent = "rejected" | "cleared";

type Listener = (event: HfTokenRejectionEvent) => void;

/** So a briefly failing verifier cannot hide private repos all session. */
export const HF_TOKEN_REJECTION_RECHECK_MS = 10 * 60 * 1000;

// Per Hub: a mirror can refuse a token another endpoint accepts.
const refusals = new Map<string | null, { fingerprint: string; at: number }>();
let version = 0;
const listeners = new Set<Listener>();
// A refusal older than a newer success is ignored.
let successes = 0;
const lastSuccess = new Map<string | null, number>();
let lastSuccessEverywhere = 0;

/** FNV-1a: tells tokens apart without recovering one. */
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
      // One broken subscriber must not stop the others.
    }
  }
}

export function hfTokenRejectionMark(): number {
  return successes;
}

/** True only when this is news. With *since*, a refusal older than a later success is ignored. */
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

/** Without a scope: whether any Hub refused it this session. */
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

export function hasRejectedHfToken(): boolean {
  return refusals.size > 0;
}

/** No scope clears every refusal. */
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

export function hfTokenRejectionVersion(): number {
  return version;
}

export function subscribeHfTokenRejected(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}
