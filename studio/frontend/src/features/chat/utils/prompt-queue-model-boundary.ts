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
