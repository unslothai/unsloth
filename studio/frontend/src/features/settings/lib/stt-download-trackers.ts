// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** One poller per model: STT engines run downloads concurrently. */
export class SttDownloadTrackers {
  private readonly running = new Map<string, () => void>();

  has(model: string): boolean {
    return this.running.has(model);
  }

  start(model: string, stop: () => void): void {
    this.stop(model);
    this.running.set(model, stop);
  }

  stop(model: string): void {
    const stop = this.running.get(model);
    this.running.delete(model);
    stop?.();
  }
}
