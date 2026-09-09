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

/** Preserve external-provider work when the singleton local model changes. Only local items depend
 *  on the outgoing model. A dispatched local item must be cancelled, but external follow-ups
 *  remain valid and can resume once the thread runtime becomes idle. */
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
