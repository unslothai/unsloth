// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
