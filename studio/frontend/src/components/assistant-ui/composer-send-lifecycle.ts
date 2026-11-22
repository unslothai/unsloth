// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The installed public ComposerRuntimeImpl.send() returns void, and its core
// Promise is dropped. Observe the public success event and our attachment
// adapter's failure path instead of inspecting an impossible return value.
//
// The `send` event payload is `{}` in @assistant-ui/core@0.1.17, so a quoted
// send is additionally correlated through the quote context id echoed by our
// history adapter. Only one composer attempt may be in flight at once; the
// pre-stream reservation is the app-level interlock and observeComposerSend
// enforces the same invariant at this boundary.

export class ComposerAttachmentPreparationError extends Error {
  readonly token: symbol | null;

  constructor(token: symbol | null, cause: unknown) {
    super(
      cause instanceof Error
        ? cause.message
        : cause === undefined
          ? "Attachment preparation failed"
          : String(cause),
    );
    this.name = "ComposerAttachmentPreparationError";
    this.token = token;
    this.cause = cause;
  }
}

export type ComposerSendObservable = {
  unstable_on(event: "send", callback: (payload: unknown) => void): () => void;
  getState(): {
    readonly isEmpty?: boolean;
    readonly dictation?: unknown;
  };
};

type SendCallbacks = {
  readonly onSent: () => void;
  readonly onFailed: (error: unknown) => void;
};

/** Ownership handle returned to the send site that installed the observer. */
export type ComposerSendAttempt = {
  readonly token: symbol | null;
  readonly stop: () => void;
  readonly isActive: () => boolean;
};

export type ComposerSendRegistration =
  | { readonly kind: "observing"; readonly attempt: ComposerSendAttempt }
  | { readonly kind: "busy"; readonly activeToken: symbol | null }
  | {
      readonly kind: "ignored";
      readonly reason: "empty" | "dictating";
    };

type SendAttempt = {
  readonly composer: ComposerSendObservable;
  readonly token: symbol | null;
  readonly threadIds: ReadonlySet<string>;
  readonly expectedContextId: string | null;
  readonly callbacks: SendCallbacks;
  active: boolean;
  sendEventSeen: boolean;
  historyContextSeen: boolean;
  sendUnsubscribe: (() => void) | null;
  removeRejectionGuard: (() => void) | null;
  failure: ComposerAttachmentPreparationError | null;
  fail: (error: ComposerAttachmentPreparationError) => void;
  maybeSent: () => void;
};

const attemptsByToken = new Map<symbol, SendAttempt>();
const attemptsByThread = new Map<string, Set<SendAttempt>>();
const attemptsByContext = new Map<string, Set<SendAttempt>>();
const activeByComposer = new WeakMap<object, SendAttempt>();

function rememberThreadAttempt(attempt: SendAttempt): void {
  for (const threadId of attempt.threadIds) {
    let set = attemptsByThread.get(threadId);
    if (set === undefined) {
      set = new Set();
      attemptsByThread.set(threadId, set);
    }
    set.add(attempt);
  }
}

function forgetThreadAttempt(attempt: SendAttempt): void {
  for (const threadId of attempt.threadIds) {
    const set = attemptsByThread.get(threadId);
    if (set === undefined) {
      continue;
    }
    set.delete(attempt);
    if (set.size === 0) {
      attemptsByThread.delete(threadId);
    }
  }
}

function rememberContextAttempt(attempt: SendAttempt): void {
  if (attempt.expectedContextId === null) {
    return;
  }
  let set = attemptsByContext.get(attempt.expectedContextId);
  if (set === undefined) {
    set = new Set();
    attemptsByContext.set(attempt.expectedContextId, set);
  }
  set.add(attempt);
}

function forgetContextAttempt(attempt: SendAttempt): void {
  if (attempt.expectedContextId === null) {
    return;
  }
  const set = attemptsByContext.get(attempt.expectedContextId);
  if (set === undefined) {
    return;
  }
  set.delete(attempt);
  if (set.size === 0) {
    attemptsByContext.delete(attempt.expectedContextId);
  }
}

function attemptFor(
  token: symbol | null,
  threadIds: readonly (string | null | undefined)[],
): SendAttempt | null {
  if (token !== null) {
    const exact = attemptsByToken.get(token);
    if (exact !== undefined) {
      return exact;
    }
  }
  for (const threadId of threadIds) {
    if (!threadId) {
      continue;
    }
    const set = attemptsByThread.get(threadId);
    if (set !== undefined && set.size > 0) {
      return set.values().next().value ?? null;
    }
  }
  return null;
}

/**
 * The dropped core Promise cannot be caught at the call site. Suppress only
 * the tagged rejection from the in-flight send, and only until its unhandled
 * rejection has been reported. Untagged rejections are never prevented.
 */
function installRejectionGuard(attempt: SendAttempt): () => void {
  if (typeof window === "undefined") {
    return () => undefined;
  }
  const handler = (event: Event): void => {
    const reason = (event as { reason?: unknown }).reason;
    if (
      reason instanceof ComposerAttachmentPreparationError &&
      reason.token === attempt.token
    ) {
      event.preventDefault();
    }
  };
  window.addEventListener("unhandledrejection", handler);
  return () => {
    window.removeEventListener("unhandledrejection", handler);
  };
}

export type ObserveComposerSendOptions = {
  readonly expectedContextId: string | null;
};

export function observeComposerSend(
  composer: ComposerSendObservable,
  threadIds: readonly string[],
  token: symbol | null,
  callbacks: SendCallbacks,
  options: ObserveComposerSendOptions = { expectedContextId: null },
): ComposerSendRegistration {
  // BaseComposerRuntimeCore.send() returns before emitting `send` when the
  // composer is empty or dictation is active. Do not occupy the composer with
  // an observer that can never receive a terminal signal.
  const state = composer.getState();
  if (state.isEmpty === true) {
    return { kind: "ignored", reason: "empty" };
  }
  if (state.dictation !== undefined) {
    return { kind: "ignored", reason: "dictating" };
  }

  const active = activeByComposer.get(composer);
  if (active !== undefined) {
    // A second send must not be attributed to the first one's event. The
    // caller's reservation normally prevents this before it reaches here.
    return { kind: "busy", activeToken: active.token };
  }

  const attempt: SendAttempt = {
    composer,
    token,
    threadIds: new Set(threadIds),
    expectedContextId: options.expectedContextId,
    callbacks,
    active: true,
    sendEventSeen: false,
    historyContextSeen: false,
    sendUnsubscribe: null,
    removeRejectionGuard: null,
    failure: null,
    fail: () => undefined,
    maybeSent: () => undefined,
  };
  attempt.removeRejectionGuard = installRejectionGuard(attempt);
  activeByComposer.set(composer, attempt);
  if (token !== null) {
    attemptsByToken.set(token, attempt);
  }
  rememberThreadAttempt(attempt);
  rememberContextAttempt(attempt);

  const cleanup = (): void => {
    if (token !== null && attemptsByToken.get(token) === attempt) {
      attemptsByToken.delete(token);
    }
    forgetThreadAttempt(attempt);
    forgetContextAttempt(attempt);
    if (activeByComposer.get(composer) === attempt) {
      activeByComposer.delete(composer);
    }
    attempt.sendUnsubscribe?.();
    attempt.sendUnsubscribe = null;
  };

  const releaseGuard = (): void => {
    const remove = attempt.removeRejectionGuard;
    attempt.removeRejectionGuard = null;
    remove?.();
  };

  const settleSent = (): void => {
    if (!attempt.active) {
      return;
    }
    attempt.active = false;
    cleanup();
    releaseGuard();
    attempt.callbacks.onSent();
  };

  const maybeSent = (): void => {
    if (!attempt.sendEventSeen) {
      return;
    }
    // A quoted send settles only when our history adapter has seen the exact
    // context id. This makes an interleaved send event unable to misattribute.
    if (attempt.expectedContextId === null || attempt.historyContextSeen) {
      settleSent();
    }
  };

  const failed = (error: ComposerAttachmentPreparationError): void => {
    if (!attempt.active) {
      return;
    }
    attempt.active = false;
    attempt.failure = error;
    cleanup();
    try {
      attempt.callbacks.onFailed(error);
    } finally {
      // The core Promise rejects after this synchronous failure callback.
      // Keep the guard for that one task, then remove it.
      setTimeout(releaseGuard, 0);
    }
  };

  attempt.sendUnsubscribe = composer.unstable_on("send", () => {
    attempt.sendEventSeen = true;
    maybeSent();
  });
  attempt.fail = failed;
  attempt.maybeSent = maybeSent;

  const stop = (): void => {
    if (!attempt.active) {
      return;
    }
    attempt.active = false;
    cleanup();
    // An unmount before the failure cannot know a rejection is coming.
    releaseGuard();
  };

  return {
    kind: "observing",
    attempt: {
      token,
      stop,
      isActive: () => attempt.active,
    },
  };
}

/**
 * Our history adapter calls this synchronously when it receives the appended
 * message. It ties the payload-less public `send` event to the quote dispatch.
 */
export function noteComposerSendMessage(quoteContextId: string | null): void {
  if (quoteContextId === null) {
    return;
  }
  const set = attemptsByContext.get(quoteContextId);
  if (set === undefined) {
    return;
  }
  for (const attempt of [...set]) {
    attempt.historyContextSeen = true;
    attempt.maybeSent();
  }
}

/** Reports before rethrowing so the caller never inspects the void send(). */
export function reportComposerAttachmentFailure(
  token: symbol | null,
  threadIds: readonly (string | null | undefined)[],
  cause: unknown,
): ComposerAttachmentPreparationError {
  const attempt = attemptFor(token, threadIds);
  const effectiveToken = token ?? attempt?.token ?? null;
  const error =
    cause instanceof ComposerAttachmentPreparationError &&
    cause.token === effectiveToken
      ? cause
      : new ComposerAttachmentPreparationError(effectiveToken, cause);
  attempt?.fail(error);
  return error;
}
