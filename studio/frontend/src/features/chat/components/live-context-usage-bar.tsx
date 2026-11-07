// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import type { ComponentProps, FC } from "react";
import { useThrottledLiveGeneration } from "../hooks/use-throttled-live-generation";
import { liveContextUsage } from "../lib/live-generation-stats";
import { useChatPreferencesStore } from "../stores/chat-preferences-store";
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
  const usage = liveContextUsage(useThrottledLiveGeneration(liveThreadKey));
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
