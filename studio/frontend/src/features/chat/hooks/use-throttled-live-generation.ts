// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useEffect, useState } from "react";
import {
  LIVE_STATS_INTERVAL_MS,
  createTrailingThrottle,
} from "../lib/live-generation-stats";
import {
  type LiveGenerationProgress,
  useChatRuntimeStore,
} from "../stores/chat-runtime-store";

/** Subscribes outside React so a burst of frames costs one trailing render per interval. */
export function useThrottledLiveGeneration(
  threadKey: string,
): LiveGenerationProgress | undefined {
  const read = () =>
    useChatRuntimeStore.getState().liveGenerationByThreadId[threadKey];
  // Tagged with its key so the render after a thread switch never shows the last thread's figures.
  const [shown, setShown] = useState(() => ({ key: threadKey, live: read() }));
  useEffect(() => {
    const show = () =>
      setShown({
        key: threadKey,
        live: useChatRuntimeStore.getState().liveGenerationByThreadId[
          threadKey
        ],
      });
    const throttle = createTrailingThrottle(LIVE_STATS_INTERVAL_MS, show);
    show();
    const unsubscribe = useChatRuntimeStore.subscribe((state, previous) => {
      if (
        state.liveGenerationByThreadId[threadKey] !==
        previous.liveGenerationByThreadId[threadKey]
      ) {
        throttle.schedule();
      }
    });
    return () => {
      unsubscribe();
      throttle.cancel();
    };
  }, [threadKey]);
  return shown.key === threadKey ? shown.live : read();
}
