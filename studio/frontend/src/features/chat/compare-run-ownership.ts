// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type CompareRun<Model> = {
  readonly id: number;
  readonly controller: AbortController;
  loadingModel: Model | null;
  cleanup: Promise<void> | null;
};

/** One generalized compare send; a stopped send can unwind after a newer one begins, so late
 *  callbacks check ownership before touching shared state. */
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

type ComparePaneRunPreparer = (
  pane: "model1" | "model2",
  signal: AbortSignal,
) => Promise<void>;

export type ComparePaneModel = {
  name: string;
  isActive: () => boolean;
  isLoading: () => boolean;
  load: () => Promise<void>;
  cancelLoad: () => void;
};

const CANCEL_POLL_MS = 100;
let latestPaneLoad = 0;

export function isLoadingPick(
  pick: { id: string; ggufVariant?: string | null } | null | undefined,
  sel: { id: string; ggufVariant?: string | null },
): boolean {
  return (
    pick?.id === sel.id &&
    (pick.ggufVariant ?? null) === (sel.ggufVariant ?? null)
  );
}

export async function loadComparePaneModel(
  model: ComparePaneModel,
  signal: AbortSignal,
): Promise<void> {
  if (model.isActive()) return;
  signal.throwIfAborted();
  // Stopping cancels this pane's load, as stopping a shared Send does. The load is published only
  // after its preflight, so a stop before then watches for it until a newer pane load takes over.
  const thisLoad = ++latestPaneLoad;
  let settled = false;
  const cancel = () => {
    if (settled || latestPaneLoad !== thisLoad) return;
    if (model.isLoading()) model.cancelLoad();
    else setTimeout(cancel, CANCEL_POLL_MS);
  };
  signal.addEventListener("abort", cancel, { once: true });
  try {
    await model.load();
  } finally {
    settled = true;
    signal.removeEventListener("abort", cancel);
  }
  signal.throwIfAborted();
  if (!model.isActive()) {
    throw new Error(
      `${model.name} is not loaded, so this pane was not answered.`,
    );
  }
}

const comparePaneRunPreparers = new Map<string, ComparePaneRunPreparer>();

export function registerComparePaneRunPreparer(
  pairId: string,
  prepare: ComparePaneRunPreparer,
): () => void {
  comparePaneRunPreparers.set(pairId, prepare);
  return () => {
    if (comparePaneRunPreparers.get(pairId) === prepare) {
      comparePaneRunPreparers.delete(pairId);
    }
  };
}

// Waits out a load in flight first: until it lands, the store still names the outgoing model.
// The preparer is read after the wait, as the one registered before it holds the old model.
export async function prepareComparePaneRun(
  pairId: string | undefined,
  pane: string | undefined,
  signal: AbortSignal,
  waitForModelIdle: () => Promise<void>,
): Promise<void> {
  if (
    !pairId ||
    !comparePaneRunPreparers.has(pairId) ||
    (pane !== "model1" && pane !== "model2")
  ) {
    return;
  }
  await waitForModelIdle();
  await comparePaneRunPreparers.get(pairId)?.(pane, signal);
}
