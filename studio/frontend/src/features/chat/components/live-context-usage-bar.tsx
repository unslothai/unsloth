// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { type ComponentProps, type FC, useRef } from "react";
import { useThrottledLiveGeneration } from "../hooks/use-throttled-live-generation";
import {
  createPassSettledTracker,
  liveContextUsage,
} from "../lib/live-generation-stats";
import { useChatPreferencesStore } from "../stores/chat-preferences-store";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { ContextUsageBar } from "./context-usage-bar";

type BarProps = ComponentProps<typeof ContextUsageBar>;

/** Settled mode subscribes to no live figures, so the header never re-renders mid-stream. */
export const LiveContextUsageBar: FC<BarProps & { liveThreadKey: string }> = ({
  liveThreadKey,
  ...props
}) => {
  const realtime = useChatPreferencesStore(
    (state) => state.generationStatsUpdates === "realtime",
  );
  // Keyed so a chat switch mid-pass starts a fresh settled tracker for the new thread.
  return realtime ? (
    <RealtimeContextUsageBar key={liveThreadKey} liveThreadKey={liveThreadKey} {...props} />
  ) : (
    <ContextUsageBar {...props} />
  );
};

const RealtimeContextUsageBar: FC<BarProps & { liveThreadKey: string }> = ({
  liveThreadKey,
  ...props
}) => {
  const live = useThrottledLiveGeneration(liveThreadKey);
  const settled = useChatRuntimeStore((state) => state.contextUsage);
  const tracker = useRef<ReturnType<typeof createPassSettledTracker> | null>(
    null,
  );
  tracker.current ??= createPassSettledTracker();
  const usage = liveContextUsage(
    live,
    tracker.current(live?.prefill != null, live?.decode != null, settled),
  );
  if (!usage) return <ContextUsageBar {...props} />;
  return (
    <ContextUsageBar
      {...props}
      used={usage.used}
      promptTokens={usage.promptTokens}
      completionTokens={usage.completionTokens}
      cached={usage.cached}
      cacheWrites={undefined}
      estimated={false}
    />
  );
};
