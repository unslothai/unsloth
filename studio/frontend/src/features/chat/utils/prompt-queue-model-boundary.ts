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
  activeItemRemoved: boolean;
  refreshTargetIdleWait: boolean;
  retainedItemIndexes: number[];
};

/** Only local items depend on the outgoing model; external follow-ups stay valid. */
export function planLocalPromptQueueStop(
  items: readonly PromptQueueModelStopItem[],
  runIndex: number,
): LocalPromptQueueStopPlan {
  const activeIndex = Math.max(runIndex, 0);
  const activeItem = items[activeIndex];
  const retainedItemIndexes = items.flatMap((item, index) =>
    index < activeIndex || !item.usesLocalModel ? [index] : [],
  );
  return {
    cancelActiveItem: Boolean(
      activeItem?.usesLocalModel && activeItem.dispatched,
    ),
    activeItemRemoved: Boolean(activeItem?.usesLocalModel),
    refreshTargetIdleWait: Boolean(
      runIndex < 0 &&
        activeItem?.usesLocalModel &&
        retainedItemIndexes.length > 0,
    ),
    retainedItemIndexes,
  };
}

export function shouldAbortPendingQueueForModelBoundary({
  capturedGeneration,
  usesLocalModel,
}: {
  capturedGeneration: number;
  usesLocalModel: boolean;
}): boolean {
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
