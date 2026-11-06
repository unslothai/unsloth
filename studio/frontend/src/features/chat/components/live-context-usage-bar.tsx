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

/** The chat header's meter. Settled mode subscribes to no live figures, so the header never
 *  re-renders mid-stream; realtime counts the prompt and reply as they stream. */
export const LiveContextUsageBar: FC<BarProps & { liveThreadKey: string }> = ({
  liveThreadKey,
  ...props
}) => {
  const realtime = useChatPreferencesStore(
    (state) => state.generationStatsUpdates === "realtime",
  );
  return realtime ? (
    <RealtimeContextUsageBar liveThreadKey={liveThreadKey} {...props} />
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
    tracker.current(live?.decode != null, settled),
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
