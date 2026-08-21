// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Bounds legacy storage access and latches to the fallback after a failure. */
export class LegacyStoreGate {
  responds = true;
  timeoutMs: number;

  constructor(timeoutMs = 1_000) {
    this.timeoutMs = timeoutMs;
  }

  async read<T>(read: () => Promise<T>, fallback: T): Promise<T> {
    if (!this.responds) return fallback;
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      return await Promise.race([
        read(),
        new Promise<T>((resolve) => {
          timer = setTimeout(() => {
            this.responds = false;
            resolve(fallback);
          }, this.timeoutMs);
        }),
      ]);
    } catch {
      this.responds = false;
      return fallback;
    } finally {
      if (timer !== undefined) clearTimeout(timer);
    }
  }
}
