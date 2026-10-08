// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useAuiState } from "@assistant-ui/react";
import { useRef } from "react";

type MessageState = Parameters<Parameters<typeof useAuiState>[0]>[0]["message"];

function sameDeps(a: readonly unknown[], b: readonly unknown[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i += 1) {
    if (!Object.is(a[i], b[i])) return false;
  }
  return true;
}

/**
 * `useAuiState(({ message }) => select(message))`, re-running `select` only when the message's
 * state object or one of `deps` changed (#12552).
 *
 * Every store write runs every selector in the tree, and a write elsewhere (a composer keystroke,
 * another message streaming) hands this message back unchanged. A selector that walks the parts
 * paid that walk per write anyway; this pays it per revision of the message. `select` must read
 * nothing but `message` and `deps`. `equal` keeps the previous result when a recompute matches it,
 * for selectors returning a fresh array (the `useShallow` case).
 */
export function useMessageMemo<T>(
  select: (message: MessageState) => T,
  deps: readonly unknown[],
  equal?: (a: T, b: T) => boolean,
): T {
  const memo = useRef<{
    message: MessageState;
    deps: readonly unknown[];
    value: T;
  } | null>(null);
  return useAuiState(({ message }) => {
    const last = memo.current;
    if (
      last !== null &&
      last.message === message &&
      sameDeps(last.deps, deps)
    ) {
      return last.value;
    }
    const next = select(message);
    const value =
      last !== null && equal !== undefined && equal(last.value, next)
        ? last.value
        : next;
    memo.current = { message, deps, value };
    return value;
  });
}
