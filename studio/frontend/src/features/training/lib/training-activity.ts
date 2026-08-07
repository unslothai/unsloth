


import {
  isTrainingStartPending,
  useTrainingRuntimeStore,
} from "../stores/training-runtime-store";

export function subscribeToTrainingActivity(
  publish: (active: boolean) => void,
): () => void {
  publish(isTrainingStartPending(useTrainingRuntimeStore.getState()));
  return useTrainingRuntimeStore.subscribe((state, previous) => {
    const active = isTrainingStartPending(state);
    if (active !== isTrainingStartPending(previous)) {
      publish(active);
    }
  });
}
