// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Serialize writes that touch one or more keys. */
export class KeyedWriteQueue {
  private readonly tails = new Map<string, Promise<void>>();

  enqueue<T>(keys: Iterable<string>, write: () => Promise<T>): Promise<T> {
    const uniqueKeys = Array.from(new Set(keys));
    const predecessors = Array.from(
      new Set(
        uniqueKeys
          .map((key) => this.tails.get(key))
          .filter((tail): tail is Promise<void> => tail !== undefined),
      ),
    );
    const ready = Promise.all(predecessors);
    const result = ready.then(write);
    const tail = result.then(
      () => undefined,
      () => undefined,
    );

    for (const key of uniqueKeys) {
      this.tails.set(key, tail);
    }
    tail.then(() => {
      for (const key of uniqueKeys) {
        if (this.tails.get(key) === tail) {
          this.tails.delete(key);
        }
      }
    });
    return result;
  }

  get(key: string): Promise<void> | undefined {
    return this.tails.get(key);
  }

  has(key: string): boolean {
    return this.tails.has(key);
  }

  keys(): string[] {
    return Array.from(this.tails.keys());
  }
}
