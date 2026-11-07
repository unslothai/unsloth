// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import {
  LIVE_STATS_INTERVAL_MS,
  createTrailingThrottle,
  nextPassNumber,
} from "../lib/live-generation-stats";
import {
  type LiveGenerationProgress,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";

/** Subscribes outside React so a burst of frames costs one trailing render per interval. */
export function useThrottledLiveGeneration(
  threadKey: string,
): LiveGenerationProgress | undefined {
  return useThrottledLiveGenerationWithPass(threadKey).live;
}

/** Also numbers the passes, counted on every frame so a boundary inside one throttle window is kept. */
export function useThrottledLiveGenerationWithPass(
  threadKey: string,
): { live: LiveGenerationProgress | undefined; pass: number } {
  const read = () =>
    useChatRuntimeStore.getState().liveGenerationByThreadId[threadKey];
  // Tagged with its key so the render after a thread switch never shows the last thread's figures.
  const [shown, setShown] = useState(() => ({
    key: threadKey,
    live: read(),
    pass: read()?.prefill ? 1 : 0,
  }));
  useEffect(() => {
    let pass = read()?.prefill ? 1 : 0;
    const show = () =>
      setShown({ key: threadKey, live: read(), pass });
    const throttle = createTrailingThrottle(LIVE_STATS_INTERVAL_MS, show);
    show();
    const unsubscribe = useChatRuntimeStore.subscribe((state, previous) => {
      const next = state.liveGenerationByThreadId[threadKey];
      const before = previous.liveGenerationByThreadId[threadKey];
      if (next !== before) {
        pass = nextPassNumber(pass, before, next);
        throttle.schedule();
      }
    });
    return () => {
      unsubscribe();
      throttle.cancel();
    };
  }, [threadKey]);
  return shown.key === threadKey
    ? { live: shown.live, pass: shown.pass }
    : { live: read(), pass: read()?.prefill ? 1 : 0 };
}
