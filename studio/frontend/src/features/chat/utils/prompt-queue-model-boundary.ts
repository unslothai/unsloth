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
  stopEntireRun: boolean;
  activeItemRemoved: boolean;
  retainedItemIndexes: number[];
};

/**
 * Preserve external-provider work when the singleton local model changes.
 *
 * A dispatched local item owns the sequential run, so cancelling it invalidates
 * every follow-up. Otherwise only undispatched local items depend on the
 * outgoing model and external items can continue safely.
 */
export function planLocalPromptQueueStop(
  items: readonly PromptQueueModelStopItem[],
  runIndex: number,
): LocalPromptQueueStopPlan {
  const activeIndex = Math.max(runIndex, 0);
  const activeItem = items[activeIndex];
  if (activeItem?.usesLocalModel && activeItem.dispatched) {
    return {
      stopEntireRun: true,
      activeItemRemoved: true,
      retainedItemIndexes: [],
    };
  }

  const retainedItemIndexes = items.flatMap((item, index) =>
    index < activeIndex || !item.usesLocalModel ? [index] : [],
  );
  return {
    stopEntireRun: false,
    activeItemRemoved: Boolean(activeItem?.usesLocalModel),
    retainedItemIndexes,
  };
}

export function shouldAbortPendingQueueForModelBoundary({
  capturedGeneration,
  usesLocalModel,
  modelLoading,
}: {
  capturedGeneration: number;
  usesLocalModel: boolean;
  modelLoading: boolean;
}): boolean {
  return (
    usesLocalModel &&
    (modelLoading ||
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
