// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// One keystroke in the composer must not cost a pass over the thread (#12552).
// assistant-ui has one notification manager per client tree: `composer.setText` on every keystroke
// notifies every `useAuiState` subscriber, and each one re-runs its selector. At 2,000 messages that
// is tens of thousands of selector runs per character, and the keystroke stalls for hundreds of ms
// on a fast desktop and seconds on a laptop. None of those rows can change: inside a message,
// `composer` is that message's EDIT composer, and nothing under a message reads the thread's.
// So the message rows get a client whose `subscribe` drops the notifications in which no scope's
// STATE changed except the thread composer's text. Every other state change, including the
// composer's attachments, dictation or edit state, passes through untouched. A change visible only
// through scope methods would also be dropped; in assistant-ui 0.12 methods change only with the
// runtime, which changes `thread.messages` too, so re-check this on an upgrade. Reads are not
// gated: a row that renders for any other reason still reads current state.

import type { AssistantClient } from "@assistant-ui/react";

type Listener = () => void;
type Fingerprint = unknown[];

// The composer fields a keystroke changes. Everything else on the composer is compared.
const TYPED_COMPOSER_FIELDS = new Set(["text", "isEmpty"]);
const EMPTY: readonly unknown[] = [];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isThreadState(value: Record<string, unknown>): boolean {
  return "messages" in value && "composer" in value;
}

function isThreadListState(value: Record<string, unknown>): boolean {
  return "main" in value && "threadItems" in value;
}

function isComposerState(value: Record<string, unknown>): boolean {
  return "text" in value && "isEmpty" in value && "attachments" in value;
}

function pushComposer(out: Fingerprint, composer: unknown): void {
  if (!isRecord(composer)) {
    out.push(composer);
    return;
  }
  for (const key of Object.keys(composer)) {
    if (TYPED_COMPOSER_FIELDS.has(key)) continue;
    const value = composer[key];
    // The composer state is rebuilt per keystroke with a fresh `queue: []` literal.
    out.push(Array.isArray(value) && value.length === 0 ? EMPTY : value);
  }
}

// Thread and thread-list states embed the composer, so a keystroke gives them a new identity
// while every field but the composer keeps its own: compare those fields, not the container.
function pushState(out: Fingerprint, state: unknown): void {
  if (!isRecord(state)) {
    out.push(state);
    return;
  }
  if (isThreadState(state)) {
    for (const key of Object.keys(state)) {
      if (key === "composer") pushComposer(out, state.composer);
      else out.push(state[key]);
    }
    return;
  }
  if (isThreadListState(state)) {
    for (const key of Object.keys(state)) {
      if (key === "main") pushState(out, state.main);
      else out.push(state[key]);
    }
    return;
  }
  if (isComposerState(state)) {
    pushComposer(out, state);
    return;
  }
  out.push(state);
}

/** Every scope's state, with the thread composer's typed text left out. */
export function composerTextFingerprint(client: AssistantClient): Fingerprint {
  const out: Fingerprint = [];
  // Scopes are enumerable accessor functions on the client and its parents.
  for (const key in client) {
    if (key === "subscribe" || key === "on") continue;
    const scope = (client as unknown as Record<string, unknown>)[key];
    if (typeof scope !== "function") continue;
    let state: unknown;
    try {
      const methods = (scope as () => unknown)();
      const getState = isRecord(methods) ? methods.getState : undefined;
      if (typeof getState !== "function") continue;
      state = getState.call(methods);
    } catch {
      // A scope with no source here (none is mounted) has nothing a row could read.
      continue;
    }
    out.push(key);
    pushState(out, state);
  }
  return out;
}

export function sameFingerprint(a: Fingerprint, b: Fingerprint): boolean {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index += 1) {
    if (!Object.is(a[index], b[index])) return false;
  }
  return true;
}

/**
 * `parent` with a `subscribe` that skips notifications in which only the thread composer's text
 * changed. Everything else is inherited, so scopes, events and state reads are the parent's own.
 * The parent subscription is held only while a row is subscribed.
 */
export function createComposerTextGatedClient(
  parent: AssistantClient,
): AssistantClient {
  const listeners = new Set<Listener>();
  let release: (() => void) | null = null;
  let last: Fingerprint | null = null;

  // Fail open: a fingerprint that cannot be taken must never swallow a notification.
  const fingerprint = (): Fingerprint | null => {
    try {
      return composerTextFingerprint(parent);
    } catch (error) {
      console.error("composer text gate: fingerprint error", error);
      return null;
    }
  };

  const onParentNotify = () => {
    const next = fingerprint();
    const skip = last !== null && next !== null && sameFingerprint(last, next);
    last = next;
    if (skip) return;
    for (const listener of listeners) {
      try {
        listener();
      } catch (error) {
        console.error("composer text gate: subscriber callback error", error);
      }
    }
  };

  const subscribe = (listener: Listener) => {
    listeners.add(listener);
    if (release === null) {
      last = fingerprint();
      release = parent.subscribe(onParentNotify);
    }
    return () => {
      listeners.delete(listener);
      if (listeners.size === 0 && release !== null) {
        release();
        release = null;
        last = null;
      }
    };
  };

  const client = Object.create(parent) as AssistantClient;
  Object.defineProperty(client, "subscribe", {
    value: subscribe,
    enumerable: false,
    configurable: true,
    writable: true,
  });
  return client;
}
