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
 * Resolves message `index` of the current thread for one row. While `index` is past the thread's end
 * it keeps the last message it resolved, and while a part index is past the message's parts it keeps
 * the last part it resolved there (assistant-ui#5127). Reads only land past the end in the window
 * between a thread switch and React unmounting the row. Never resolved: the lookup throws as upstream.
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
      // A fresh target, not `message`: assistant-ui's client is itself a Proxy whose
      // getOwnPropertyDescriptor reports absent keys as non-configurable, so the engine's invariant
      // check on any read through a Proxy over it throws. Receiver stays `message`: it rebinds per receiver.
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
