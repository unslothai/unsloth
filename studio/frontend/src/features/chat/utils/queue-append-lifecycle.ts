// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Public ThreadRuntimeImpl.append() returns void and LocalThreadRuntimeCore
// drops history.append()'s Promise. A plain string token carried in
// metadata.custom lets our own history adapter settle the queued append.
// Strings survive every conversion path; symbols do not.

export const QUEUE_APPEND_TOKEN_KEY = "unslothQueueAppendToken";

export class QueueAppendAbandonedError extends Error {
  constructor(message = "The queued append never reached history.") {
    super(message);
    this.name = "QueueAppendAbandonedError";
  }
}

type QueueAppendWaiter = {
  readonly promise: Promise<void>;
  readonly resolve: () => void;
  readonly reject: (error: unknown) => void;
};

const waiters = new Map<string, QueueAppendWaiter>();

export function createQueueAppendToken(): string {
  const cryptoApi = globalThis.crypto;
  if (cryptoApi && typeof cryptoApi.randomUUID === "function") {
    return cryptoApi.randomUUID();
  }
  return `queue-append-${Date.now().toString(36)}-${Math.random()
    .toString(36)
    .slice(2)}`;
}

type TokenCarrier = {
  readonly metadata?: {
    readonly custom?: Record<string, unknown>;
  };
};

export function queueAppendTokenForMessage(
  message: TokenCarrier,
): string | null {
  const token = message.metadata?.custom?.[QUEUE_APPEND_TOKEN_KEY];
  return typeof token === "string" && token.length > 0 ? token : null;
}

/**
 * Reads the token and clears it from the live thread message. The converted
 * message shares metadata.custom with the caller's message-like, so this also
 * keeps the transport marker out of edit/resend and later re-exports.
 */
export function takeQueueAppendToken(message: TokenCarrier): string | null {
  const custom = message.metadata?.custom;
  if (custom === undefined) {
    return null;
  }
  const token = custom[QUEUE_APPEND_TOKEN_KEY];
  if (typeof token !== "string" || token.length === 0) {
    return null;
  }
  // A frozen caller object must not make history append throw; persistence
  // still uses the stripped copy below.
  Reflect.deleteProperty(custom, QUEUE_APPEND_TOKEN_KEY);
  return token;
}

/** Removes the transport token before the row is written or re-exported. */
export function stripQueueAppendToken(
  custom: Record<string, unknown> | undefined,
): Record<string, unknown> | undefined {
  if (custom === undefined) {
    return undefined;
  }
  const rest: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(custom)) {
    if (key !== QUEUE_APPEND_TOKEN_KEY) {
      rest[key] = value;
    }
  }
  return Object.keys(rest).length > 0 ? rest : undefined;
}

export function waitForQueueAppend(token: string): Promise<void> {
  const existing = waiters.get(token);
  if (existing !== undefined) {
    return existing.promise;
  }
  let resolve!: () => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<void>((settle, fail) => {
    resolve = settle;
    reject = fail;
  });
  // The queue observer attaches its own rejection handler. This guards paths
  // that intentionally abandon a waiter after cancellation.
  promise.catch(() => undefined);
  waiters.set(token, { promise, resolve, reject });
  return promise;
}

function takeWaiter(token: string): QueueAppendWaiter | null {
  const waiter = waiters.get(token);
  if (waiter === undefined) {
    return null;
  }
  waiters.delete(token);
  return waiter;
}

export function settleQueueAppend(token: string): void {
  takeWaiter(token)?.resolve();
}

export function failQueueAppend(token: string, error: unknown): void {
  takeWaiter(token)?.reject(error);
}

/** Cancellation and no-op append paths must not leave a dispatched item. */
export function abandonQueueAppend(token: string): void {
  takeWaiter(token)?.reject(new QueueAppendAbandonedError());
}
