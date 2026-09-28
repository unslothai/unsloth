// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Which saved Hugging Face token the Hub refused, so the UI can say so instead of "offline".
 *
 * A token the Hub accepts gets 404 for a repo it cannot see, never 401, so a 401 on a read that
 * carried a token which then succeeded anonymously means the token itself was refused (an
 * expired OAuth token, a revoked key). Keyed by a fingerprint, never the token, and dependency
 * free so hub-fetch stays importable under bare node. */

type Listener = () => void;

let rejectedFingerprint: string | null = null;
// The Hub the refusal came from: a mirror can refuse a token another endpoint accepts.
let rejectedScope: string | null = null;
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

/** Record that the Hub at *scope* refused *token*. True only the first time for that pair. */
export function noteHfTokenRejected(
  token: string | null | undefined,
  scope: string | null = null,
): boolean {
  const value = normalized(token);
  if (!value) return false;
  const next = fingerprint(value);
  if (rejectedFingerprint === next && rejectedScope === scope) return false;
  rejectedFingerprint = next;
  rejectedScope = scope;
  version += 1;
  for (const listener of listeners) {
    try {
      listener();
    } catch {
      // One broken subscriber must not stop the others hearing about it.
    }
  }
  return true;
}

/** Whether the Hub has refused *token* in this session. A different token starts clean, and
 * with a *scope* so does a different Hub endpoint. */
export function isHfTokenRejected(
  token: string | null | undefined,
  scope?: string | null,
): boolean {
  const value = normalized(token);
  if (!value || rejectedFingerprint !== fingerprint(value)) return false;
  return scope === undefined || rejectedScope === scope;
}

/** Whether any token is currently recorded as refused. */
export function hasRejectedHfToken(): boolean {
  return rejectedFingerprint !== null;
}

export function clearHfTokenRejected(): void {
  if (rejectedFingerprint === null) return;
  rejectedFingerprint = null;
  rejectedScope = null;
  version += 1;
  for (const listener of listeners) {
    try {
      listener();
    } catch {
      // As above.
    }
  }
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
