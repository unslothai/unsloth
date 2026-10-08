// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Every assistant-ui store write re-runs every row's selectors (#12552), so each row subscribes
// through a client that drops what cannot change it: a thread-composer keystroke reaches no row; a
// same-length `thread.messages` change reaches rows from one before the first changed message (row
// selectors read their own, earlier, or the next message, never two ahead); anything else reaches all.
// Assumes scope methods change only with `thread.messages` (assistant-ui 0.12): recheck on upgrade.

import type { AssistantClient } from "@assistant-ui/react";

type Listener = () => void;

export interface RowFingerprint {
  rest: unknown[];
  messages: readonly unknown[] | null;
}

const TYPED_COMPOSER_FIELDS = new Set(["text", "isEmpty"]);
const EMPTY: readonly unknown[] = [];
const ROW_MESSAGES = Symbol("row messages");

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

function pushComposer(out: unknown[], composer: unknown): void {
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

// Thread and thread-list states embed the composer, so compare their fields, not the container.
function pushState(
  out: unknown[],
  state: unknown,
  rowMessages: readonly unknown[] | null,
): void {
  if (!isRecord(state)) {
    out.push(state);
    return;
  }
  if (isThreadState(state)) {
    for (const key of Object.keys(state)) {
      const value = state[key];
      if (key === "composer") pushComposer(out, value);
      else if (key === "messages" && value === rowMessages)
        out.push(ROW_MESSAGES);
      else out.push(value);
    }
    return;
  }
  if (isThreadListState(state)) {
    for (const key of Object.keys(state)) {
      if (key === "main") pushState(out, state.main, rowMessages);
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

function scopeState(client: AssistantClient, key: string): unknown {
  const scope = (client as unknown as Record<string, unknown>)[key];
  if (typeof scope !== "function") return undefined;
  const methods = (scope as () => unknown)();
  const getState = isRecord(methods) ? methods.getState : undefined;
  if (typeof getState !== "function") return undefined;
  return getState.call(methods);
}

export function rowFingerprint(client: AssistantClient): RowFingerprint {
  let messages: readonly unknown[] | null = null;
  try {
    const thread = scopeState(client, "thread");
    if (isRecord(thread) && Array.isArray(thread.messages))
      messages = thread.messages;
  } catch {
    // No thread scope: every change lands in `rest` and reaches every row.
  }
  const rest: unknown[] = [];
  for (const key in client) {
    if (key === "subscribe" || key === "on") continue;
    let state: unknown;
    try {
      state = scopeState(client, key);
    } catch {
      continue;
    }
    if (state === undefined) continue;
    rest.push(key);
    pushState(rest, state, messages);
  }
  return { rest, messages };
}

function sameValues(a: readonly unknown[], b: readonly unknown[]): boolean {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index += 1) {
    if (!Object.is(a[index], b[index])) return false;
  }
  return true;
}

export function rowsToNotify(
  last: RowFingerprint | null,
  next: RowFingerprint | null,
): "none" | "all" | number {
  if (last === null || next === null || !sameValues(last.rest, next.rest)) {
    return "all";
  }
  if (last.messages === next.messages) return "none";
  if (
    last.messages === null ||
    next.messages === null ||
    last.messages.length !== next.messages.length
  ) {
    return "all";
  }
  for (let index = 0; index < next.messages.length; index += 1) {
    if (last.messages[index] !== next.messages[index]) {
      return Math.max(0, index - 1);
    }
  }
  return "none";
}

export interface RowNotificationGate {
  row(index: number): AssistantClient;
}

function withSubscribe(
  parent: AssistantClient,
  subscribe: (listener: Listener) => () => void,
): AssistantClient {
  const client = Object.create(parent) as AssistantClient;
  Object.defineProperty(client, "subscribe", {
    value: subscribe,
    enumerable: false,
    configurable: true,
    writable: true,
  });
  return client;
}

export function createRowNotificationGate(
  parent: AssistantClient,
): RowNotificationGate {
  const entries = new Set<{ listener: Listener; index: number }>();
  const rows = new Map<number, AssistantClient>();
  let release: (() => void) | null = null;
  let last: RowFingerprint | null = null;

  // Fail open: an unreadable state reaches every row.
  const fingerprint = (): RowFingerprint | null => {
    try {
      return rowFingerprint(parent);
    } catch (error) {
      console.error("row notification gate: fingerprint error", error);
      return null;
    }
  };

  const onParentNotify = () => {
    const next = fingerprint();
    const from = rowsToNotify(last, next);
    last = next;
    if (from === "none") return;
    for (const entry of entries) {
      if (from !== "all" && entry.index < from) continue;
      try {
        entry.listener();
      } catch (error) {
        console.error(
          "row notification gate: subscriber callback error",
          error,
        );
      }
    }
  };

  const subscribe = (index: number, listener: Listener) => {
    const entry = { listener, index };
    entries.add(entry);
    if (release === null) {
      last = fingerprint();
      release = parent.subscribe(onParentNotify);
    }
    return () => {
      entries.delete(entry);
      if (entries.size === 0 && release !== null) {
        release();
        release = null;
        last = null;
      }
    };
  };

  return {
    row(index: number): AssistantClient {
      let client = rows.get(index);
      if (client === undefined) {
        client = withSubscribe(parent, (listener) =>
          subscribe(index, listener),
        );
        rows.set(index, client);
      }
      return client;
    },
  };
}
