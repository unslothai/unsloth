// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type CompareRun<Model> = {
  readonly id: number;
  readonly controller: AbortController;
  loadingModel: Model | null;
  cleanup: Promise<void> | null;
};

/**
 * Owns the mutable lifecycle of one generalized compare submission.
 *
 * A stopped submission can unwind after a newer one starts. Keeping ownership
 * in one object gives every late callback the same identity check before it
 * clears shared loading/busy state.
 */
export class CompareRunOwnership<Model> {
  private nextId = 1;
  private activeRun: CompareRun<Model> | null = null;

  begin(): CompareRun<Model> {
    this.activeRun?.controller.abort();
    const run: CompareRun<Model> = {
      id: this.nextId,
      controller: new AbortController(),
      loadingModel: null,
      cleanup: null,
    };
    this.nextId += 1;
    this.activeRun = run;
    return run;
  }

  current(): CompareRun<Model> | null {
    return this.activeRun;
  }

  owns(run: CompareRun<Model>): boolean {
    return this.activeRun === run;
  }

  setLoadingModel(run: CompareRun<Model>, model: Model | null): boolean {
    if (!this.owns(run)) {
      return false;
    }
    run.loadingModel = model;
    return true;
  }

  setCleanup(run: CompareRun<Model>, cleanup: Promise<void>): boolean {
    if (!this.owns(run)) {
      return false;
    }
    run.cleanup = cleanup;
    return true;
  }

  cancelCurrent(): CompareRun<Model> | null {
    const run = this.activeRun;
    run?.controller.abort();
    return run;
  }

  release(run: CompareRun<Model>): boolean {
    if (!this.owns(run)) {
      return false;
    }
    this.activeRun = null;
    return true;
  }
}

export function throwIfCompareCancelled(signal: AbortSignal): void {
  if (signal.aborted) {
    throw new DOMException("Compare cancelled", "AbortError");
  }
}

export function isCompareCancellation(
  error: unknown,
  signal: AbortSignal,
): boolean {
  return (
    signal.aborted ||
    (error instanceof DOMException && error.name === "AbortError")
  );
}
