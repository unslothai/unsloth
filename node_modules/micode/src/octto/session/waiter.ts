// src/octto/session/waiter.ts
// Immutable waiter management for async response handling

export interface Waiters<K, T> {
  register: (key: K, callback: (data: T) => void) => () => void;
  notifyFirst: (key: K, data: T) => void;
  notifyAll: (key: K, data: T) => void;
  has: (key: K) => boolean;
  count: (key: K) => number;
  clear: (key: K) => void;
}

/**
 * Create a waiter registry for async response handling.
 * Each operation creates a new array rather than mutating in place.
 *
 * @typeParam K - Key type (e.g., string for question_id or session_id)
 * @typeParam T - Data type passed to waiter callbacks
 */
export function createWaiters<K, T>(): Waiters<K, T> {
  const waiters = new Map<K, Array<(data: T) => void>>();

  return {
    /**
     * Register a waiter callback for a key.
     * Returns a cleanup function to remove this specific waiter.
     */
    register(key: K, callback: (data: T) => void): () => void {
      // Create new array with callback appended (immutable)
      const current = waiters.get(key) || [];
      waiters.set(key, [...current, callback]);

      // Return cleanup function that removes this specific callback
      return () => {
        const callbacks = waiters.get(key);
        if (!callbacks) return;

        const idx = callbacks.indexOf(callback);
        if (idx >= 0) {
          // Create new array without this callback (immutable)
          const remaining = [...callbacks.slice(0, idx), ...callbacks.slice(idx + 1)];
          if (remaining.length === 0) {
            waiters.delete(key);
          } else {
            waiters.set(key, remaining);
          }
        }
      };
    },

    /**
     * Notify only the first waiter for a key and remove it.
     * Other waiters remain registered for subsequent notifications.
     */
    notifyFirst(key: K, data: T): void {
      const callbacks = waiters.get(key);
      if (!callbacks || callbacks.length === 0) return;

      const [first, ...rest] = callbacks;
      first(data);

      // Set new array without first element (immutable)
      if (rest.length === 0) {
        waiters.delete(key);
      } else {
        waiters.set(key, rest);
      }
    },

    /**
     * Notify all waiters for a key and remove them all.
     */
    notifyAll(key: K, data: T): void {
      const callbacks = waiters.get(key);
      if (!callbacks) return;

      try {
        for (const callback of callbacks) {
          try {
            callback(data);
          } catch (error) {
            console.error("Waiter notifyAll failed", error);
            break;
          }
        }
      } finally {
        waiters.delete(key);
      }
    },

    /**
     * Check if there are any waiters for a key.
     */
    has(key: K): boolean {
      const callbacks = waiters.get(key);
      return callbacks !== undefined && callbacks.length > 0;
    },

    /**
     * Get the number of waiters for a key.
     */
    count(key: K): number {
      return waiters.get(key)?.length ?? 0;
    },

    /**
     * Remove all waiters for a key without notifying them.
     */
    clear(key: K): void {
      waiters.delete(key);
    },
  };
}

/**
 * Result of waiting for a response
 */
export type WaitResult<T> = { ok: true; data: T } | { ok: false; reason: "timeout" };

/**
 * Wait for a response with timeout.
 * Registers a waiter and returns a promise that resolves when notified or times out.
 */
export function waitForResponse<K, T>(waiters: Waiters<K, T>, key: K, timeoutMs: number): Promise<WaitResult<T>> {
  return new Promise((resolve) => {
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let cleanup: (() => void) | undefined;

    cleanup = waiters.register(key, (data) => {
      if (timeoutId) clearTimeout(timeoutId);
      resolve({ ok: true, data });
    });

    timeoutId = setTimeout(() => {
      if (cleanup) cleanup();
      resolve({ ok: false, reason: "timeout" });
    }, timeoutMs);
  });
}
