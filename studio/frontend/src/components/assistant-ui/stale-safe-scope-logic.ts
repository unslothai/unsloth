// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure halves of stale-safe-scopes.tsx, kept free of React so node tests can import them.

import type {
  MessagePartState,
  ReasoningMessagePart,
  TextMessagePart,
} from "@assistant-ui/core";

type PartSelector = { index: number } | { toolCallId: string };

interface MessageLike<P> {
  getState(): { readonly parts: readonly unknown[] };
  part(selector: PartSelector): P;
}

interface ThreadLike<M> {
  getState(): { readonly messages: readonly unknown[] };
  message(query: { index: number }): M;
}

/**
 * One row's message lookup that keeps the last message (and per part index, the last part) it resolved
 * while the index is past the end; never resolved throws as upstream (assistant-ui#5127, #5948).
 */
export function createStaleSafeMessageResolver<P, M extends MessageLike<P>>(
  index: number,
): (thread: ThreadLike<M>) => M {
  let last: M | undefined;
  const lastParts = new Map<number, P>();
  const guarded = new WeakMap<M, M>();
  const guard = (message: M): M => {
    let proxy = guarded.get(message);
    if (proxy === undefined) {
      const part = (selector: PartSelector): P => {
        if (!("index" in selector)) return message.part(selector);
        const kept = lastParts.get(selector.index);
        if (
          kept !== undefined &&
          selector.index >= message.getState().parts.length
        ) {
          return kept;
        }
        const resolved = message.part(selector);
        lastParts.set(selector.index, resolved);
        return resolved;
      };
      // Not `new Proxy(message)`: the client is a Proxy reporting absent keys as non-configurable, which
      // trips the get invariant. Receiver stays `message`, since the client rebinds methods per receiver.
      proxy = new Proxy({} as M, {
        get: (_, prop) => (prop === "part" ? part : Reflect.get(message, prop)),
        has: (_, prop) => prop === "part" || Reflect.has(message, prop),
      });
      guarded.set(message, proxy);
    }
    return proxy;
  };
  return (thread) => {
    if (last === undefined || index < thread.getState().messages.length) {
      last = thread.message({ index });
    }
    return guard(last);
  };
}

export type TextPartState = MessagePartState &
  (TextMessagePart | ReasoningMessagePart);

export const EMPTY_TEXT_PART: TextPartState = Object.freeze({
  type: "text",
  text: "",
  status: Object.freeze({ type: "complete" }),
}) as TextPartState;

/** A text leaf whose index now points at another part type reads an empty text part, never throws. */
export function textPartOrEmpty(part: MessagePartState): TextPartState {
  return part.type === "text" || part.type === "reasoning"
    ? (part as TextPartState)
    : EMPTY_TEXT_PART;
}
