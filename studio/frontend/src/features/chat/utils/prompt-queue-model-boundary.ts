import { chatModelLifecycleGate } from "./model-lifecycle-gate.ts";

export class PromptQueueModelBoundary {
  private generation = 0;

  capture(): number {
    return this.generation;
  }

  advance(): number {
    this.generation += 1;
    return this.generation;
  }
}

export const localPromptQueueModelBoundary = new PromptQueueModelBoundary();

export type PromptQueueModelStopItem = {
  usesLocalModel: boolean;
  dispatched: boolean;
};

export type LocalPromptQueueStopPlan = {
  cancelActiveItem: boolean;
  retainedItemIndexes: number[];
  /** Indexes into the original items of unsent local prompts to hold until Resume. */
  heldItemIndexes: number[];
};

/** A local model change ends the local reply in flight, so that item is cancelled and dropped as a
 *  Stop drops it. Unsent local prompts are held for Resume instead of discarded (#10428), and
 *  external-provider items are untouched since they never used the outgoing model. */
export function planLocalPromptQueueStop(
  items: readonly PromptQueueModelStopItem[],
  runIndex: number,
): LocalPromptQueueStopPlan {
  const activeIndex = Math.max(runIndex, 0);
  const activeItem = items[activeIndex];
  const cancelActiveItem = Boolean(
    activeItem?.usesLocalModel && activeItem.dispatched,
  );
  const retainedItemIndexes = items.flatMap((_, index) =>
    cancelActiveItem && index === activeIndex ? [] : [index],
  );
  const heldItemIndexes = items.flatMap((item, index) =>
    index >= activeIndex && item.usesLocalModel && !item.dispatched
      ? [index]
      : [],
  );
  return { cancelActiveItem, retainedItemIndexes, heldItemIndexes };
}

export function shouldAbortPendingQueueForModelBoundary({
  capturedGeneration,
  usesLocalModel,
}: {
  capturedGeneration: number;
  usesLocalModel: boolean;
}): boolean {
  // Preparation can still clear queues at the final model-switch boundary.
  return (
    usesLocalModel &&
    (!chatModelLifecycleGate.canQueue() ||
      capturedGeneration !== localPromptQueueModelBoundary.capture())
  );
}

export function shouldAbortPendingQueueForSettingsChange({
  capturedEpoch,
  currentEpoch,
  capturedTemporary,
  currentTemporary,
}: {
  capturedEpoch: number;
  currentEpoch: number;
  capturedTemporary: boolean;
  currentTemporary: boolean;
}): boolean {
  return (
    capturedEpoch !== currentEpoch || capturedTemporary !== currentTemporary
  );
}
