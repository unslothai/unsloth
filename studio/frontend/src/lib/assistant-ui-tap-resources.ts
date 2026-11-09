// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/*
 * @assistant-ui/tap's `tapResources` with a bail-out. Upstream re-renders every child whenever its
 * element list is rebuilt, and the thread rebuilds the list of message clients on every thread update,
 * so each streamed token re-rendered every message's client: O(messages) work per token, which is
 * what made a reply stream slower the longer the chat (tap reducer 98 -> 729 ms, GC 116 -> 589 ms from
 * 40 to 200 turns). Here a child re-renders only when it marked itself dirty, its element changed, or
 * a context value provided above it changed; otherwise its last committed output is reused, like
 * React.memo. Everything else follows upstream 0.5.10 (src/hooks/tap-resources.ts).
 *
 * Bookkeeping only moves at commit: a render that is thrown away leaves a child dirty and its cached
 * output untouched, so the next render re-renders it.
 */

import {
  tapCallback,
  tapConst,
  tapEffect,
  tapMemo,
  tapRef,
} from "../../node_modules/@assistant-ui/tap/dist/index.js";
import {
  commitResourceFiber,
  createResourceFiber,
  renderResourceFiber,
  unmountResourceFiber,
} from "../../node_modules/@assistant-ui/tap/dist/core/ResourceFiber.js";
import { withContextProvider as upstreamWithContextProvider } from "../../node_modules/@assistant-ui/tap/dist/core/context.js";
import { getCurrentResourceFiber } from "../../node_modules/@assistant-ui/tap/dist/core/helpers/execution-context.js";

type Element = { type: unknown; props: unknown; key?: string | number };
type Fiber = ReturnType<typeof createResourceFiber>;
type Rendered = ReturnType<typeof renderResourceFiber>;
type Slot = {
  fiber: Fiber;
  /** Rendered and not yet committed: a result, a replacement fiber, a removal, or nothing (reused). */
  next: Rendered | [Fiber, Rendered] | "delete" | null;
  /** Set by the child's own updates; cleared when a render of it commits. */
  dirty: boolean;
  /** The last committed render, re-committed if the child was unmounted under it (see the commit effect). */
  committed: Rendered | null;
};

/** Context values provided above the resource being rendered, outermost first. */
const provided: unknown[] = [];

export function withContextProvider<T, R>(context: Parameters<typeof upstreamWithContextProvider>[0], value: T, fn: () => R): R {
  provided.push(value);
  try {
    return upstreamWithContextProvider(context, value, fn) as R;
  } finally {
    provided.pop();
  }
}

const isElement = (value: unknown): value is Element =>
  typeof value === "object" && value !== null && "type" in value && "props" in value;

/** Props equal key by key; nested resource elements are equal when type, key and props are. */
export function propsEqual(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false;
  // A wrapper resource's props are themselves an element (tapClientLookup's ClientResourceWithKey).
  if (isElement(a) || isElement(b)) {
    return isElement(a) && isElement(b) && a.type === b.type && a.key === b.key && propsEqual(a.props, b.props);
  }
  if (Array.isArray(a) || Array.isArray(b)) return false;
  const ka = Object.keys(a);
  const kb = Object.keys(b);
  if (ka.length !== kb.length) return false;
  for (const key of ka) {
    if (!Object.prototype.hasOwnProperty.call(b, key)) return false;
    const x = (a as Record<string, unknown>)[key];
    const y = (b as Record<string, unknown>)[key];
    if (Object.is(x, y)) continue;
    if (isElement(x) && isElement(y) && x.type === y.type && x.key === y.key && propsEqual(x.props, y.props)) continue;
    return false;
  }
  return true;
}

/** Equal by identity, or one level deep: the root provides `{ clientRef, emit }` as a new literal per render. */
export const sameContextValue = (a: unknown, b: unknown): boolean => {
  if (Object.is(a, b)) return true;
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) return false;
  if (Array.isArray(a) || Array.isArray(b)) {
    return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
  }
  const ka = Object.keys(a);
  if (ka.length !== Object.keys(b).length) return false;
  return ka.every(
    (k) => Object.prototype.hasOwnProperty.call(b, k) && Object.is((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]),
  );
};

const sameValues = (a: readonly unknown[], b: readonly unknown[]): boolean =>
  a.length === b.length && a.every((value, i) => sameContextValue(value, b[i]));

/** The context each render saw, for its commit. */
const renderedContext = new WeakMap<object, unknown[]>();

export function tapResources(getElements: () => Element[], getElementsDeps?: readonly unknown[]): unknown[] {
  const versionRef = tapRef(0);
  const version = versionRef.current;
  const parentFiber = tapConst(getCurrentResourceFiber, []) as Fiber & { markDirty?: () => void };
  const slots = tapConst(() => new Map<string | number, Slot>(), []);
  const committedContext = tapRef<unknown[] | null>(null);
  const getElementsMemo = getElementsDeps
    ? // deps are the caller's, as upstream
      tapCallback(getElements, getElementsDeps)
    : getElements;

  const fiberFor = (slot: Slot | null, type: unknown): [Fiber, Slot] => {
    const owner: Slot = slot ?? { fiber: undefined as unknown as Fiber, next: null, dirty: false, committed: null };
    const fiber = createResourceFiber(type as never, parentFiber.root, () => {
      owner.dirty = true;
      versionRef.current++;
      parentFiber.markDirty?.();
    });
    return [fiber, owner];
  };

  const res = tapMemo(() => {
    void version;
    const elements = getElementsMemo();
    const context = provided.slice();
    const contextSame = committedContext.current !== null && sameValues(committedContext.current, context);
    const seen = new Set<string | number>();
    const results: unknown[] = [];
    let created = 0;
    for (let i = 0; i < elements.length; i++) {
      const element = elements[i];
      const key = element.key;
      if (key === undefined) throw new Error(`tapResources did not provide a key for array at index ${i}`);
      if (seen.has(key)) throw new Error(`Duplicate key ${key} in tapResources`);
      seen.add(key);
      const slot = slots.get(key);
      if (!slot) {
        const [fiber, owner] = fiberFor(null, element.type);
        owner.fiber = fiber;
        owner.next = renderResourceFiber(fiber, element.props);
        slots.set(key, owner);
        created++;
        results.push(owner.next.output);
      } else if (slot.fiber.type !== element.type) {
        const [fiber] = fiberFor(slot, element.type);
        const result = renderResourceFiber(fiber, element.props);
        slot.next = [fiber, result];
        results.push(result.output);
      } else if (contextSame && !slot.dirty && slot.committed !== null && propsEqual(slot.committed.props, element.props)) {
        slot.next = null;
        results.push(slot.committed.output);
      } else {
        const result = renderResourceFiber(slot.fiber, element.props);
        slot.next = result;
        results.push(result.output);
      }
    }
    if (slots.size > results.length - created) {
      for (const [key, slot] of slots) if (!seen.has(key)) slot.next = "delete";
    }
    renderedContext.set(results, context);
    return results;
  }, [getElementsMemo, version]);

  tapEffect(() => {
    return () => {
      for (const slot of slots.values()) unmountResourceFiber(slot.fiber);
    };
  }, []);

  tapEffect(() => {
    for (const [key, slot] of slots) {
      const next = slot.next;
      if (next === "delete") {
        if (slot.fiber.isMounted) unmountResourceFiber(slot.fiber);
        slots.delete(key);
        continue;
      }
      if (next === null) {
        // Reused, but tap's dev strict mode runs this fiber's effects, cleans them up (unmounting
        // every child) and runs them again on first mount: remount from the last commit, as upstream
        // re-commits its kept result.
        if (!slot.fiber.isMounted && slot.committed !== null) commitResourceFiber(slot.fiber, slot.committed);
        continue;
      }
      if (Array.isArray(next)) {
        unmountResourceFiber(slot.fiber);
        slot.fiber = next[0];
        commitResourceFiber(slot.fiber, next[1]);
        slot.committed = next[1];
      } else {
        commitResourceFiber(slot.fiber, next);
        slot.committed = next;
      }
      slot.dirty = false;
      slot.next = null;
    }
    committedContext.current = renderedContext.get(res) ?? null;
  }, [res]);

  return res;
}
