// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Fallback when no frame paints; hidden pages throttle this timer too, hence the arrival cap.
export const UNPAINTED_REOPEN_MS = 500;

/** assistant-ui drops output after an abort, so this bounds what Stop can discard. */
export const MAX_HELD_CHARS = 256;

export type StreamPublishGate = ((streamed: number) => boolean) & {
  /** Resolves when the gate reopens while text it refused is still unpublished; null when nothing is held. */
  heldUntilReopen(): Promise<void> | null;
  /** The held text went out through a publish the gate did not pace. */
  published(): void;
  /** Cancels the pending frame and timer; the gate stays usable and open. */
  dispose(): void;
};

/** One publish per frame, but publishes anyway once MAX_HELD_CHARS have arrived. */
export function createStreamPublishGate(): StreamPublishGate {
  let open = true;
  let publishedAt = 0;
  let held = false;
  let cancelCycle: (() => void) | null = null;
  let reopenWaiter: Promise<void> | null = null;
  let wakeWaiter: (() => void) | null = null;
  const canPublish = (streamed: number) => {
    if (!open && streamed - publishedAt < MAX_HELD_CHARS) {
      held = true;
      return false;
    }
    held = false;
    publishedAt = streamed;
    if (open) {
      open = false;
      // Per cycle, so the loser of the previous race cannot reopen this one.
      let reopened = false;
      // Assigned before reopen closes over them, avoiding the temporal dead zone.
      const handles: {
        frame?: number;
        timer?: ReturnType<typeof setTimeout>;
      } = {};
      const settle = () => {
        if (reopened) {
          return false;
        }
        reopened = true;
        if (handles.frame !== undefined) {
          cancelAnimationFrame(handles.frame);
        }
        if (handles.timer !== undefined) {
          clearTimeout(handles.timer);
        }
        open = true;
        cancelCycle = null;
        return true;
      };
      const reopen = () => {
        if (!settle()) {
          return;
        }
        const wake = wakeWaiter;
        reopenWaiter = null;
        wakeWaiter = null;
        wake?.();
      };
      cancelCycle = settle;
      handles.frame = requestAnimationFrame(reopen);
      handles.timer = setTimeout(reopen, UNPAINTED_REOPEN_MS);
    }
    return true;
  };
  return Object.assign(canPublish, {
    heldUntilReopen(): Promise<void> | null {
      if (!held) {
        return null;
      }
      if (open) {
        return Promise.resolve();
      }
      reopenWaiter ??= new Promise<void>((resolve) => {
        wakeWaiter = resolve;
      });
      return reopenWaiter;
    },
    published() {
      held = false;
    },
    dispose() {
      cancelCycle?.();
      // Nothing waits on a disposed gate; dropping the waiter lets its awaiters be collected.
      reopenWaiter = null;
      wakeWaiter = null;
      held = false;
    },
  });
}

/** Marks a gate reopening while text is held, so the caller can publish it without waiting for a chunk. */
export const TRAILING_PUBLISH: unique symbol = Symbol("trailing publish");

/**
 * Yields `source` unchanged, plus TRAILING_PUBLISH when the gate reopens while text is held and the
 * next item has not arrived yet. An item that is ready always wins, so arrivals keep their order.
 */
export async function* withTrailingPublish<T>(
  source: AsyncIterable<T>,
  gate: StreamPublishGate,
): AsyncGenerator<T | typeof TRAILING_PUBLISH, void, undefined> {
  const iterator = source[Symbol.asyncIterator]();
  let pending: Promise<IteratorResult<T>> | null = null;
  let exhausted = false;
  try {
    while (true) {
      pending ??= iterator.next();
      const reopened = gate.heldUntilReopen();
      const next = reopened
        ? await Promise.race([
            pending,
            reopened.then((): typeof TRAILING_PUBLISH => TRAILING_PUBLISH),
          ])
        : await pending;
      if (next === TRAILING_PUBLISH) {
        yield TRAILING_PUBLISH;
        continue;
      }
      pending = null;
      if (next.done) {
        exhausted = true;
        return;
      }
      yield next.value;
    }
  } catch (error) {
    // A rejected next() ends the source; there is nothing left to return.
    exhausted = true;
    throw error;
  } finally {
    gate.dispose();
    if (!exhausted) {
      if (pending) {
        // A read is in flight: return() would queue behind it, so do not hold the caller's exit on it.
        const inFlight = pending;
        void inFlight
          .then(
            () => iterator.return?.(),
            () => undefined,
          )
          .catch(() => undefined);
      } else {
        await iterator.return?.();
      }
    }
  }
}
