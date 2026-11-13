// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  formatLiveDecode,
  type LiveDecodeProgress,
  useChatPreferencesStore,
  useChatRuntimeStore,
  useThrottledLiveGeneration,
} from "@/features/chat";
import { useLocale, useT } from "@/i18n";
import { useAuiState } from "@assistant-ui/react";
import type { FC } from "react";

/** Gated on the preferences first so a reply mounts no assistant-ui selector while both are off
 *  (composer-keystroke-subscription-budget). */
export const LiveGenerationLine: FC = () => {
  const details = useChatPreferencesStore((state) => state.showGenerationDetails);
  const realtime = useChatPreferencesStore(
    (state) => state.generationStatsUpdates === "realtime",
  );
  return details || realtime ? <RunningGate realtime={realtime} /> : null;
};

const RunningGate: FC<{ realtime: boolean }> = ({ realtime }) => {
  const running = useAuiState(({ message }) => message.status?.type === "running");
  if (!running) return null;
  return realtime ? <ThrottledDecodeFigures /> : <LiveDecodeFigures />;
};

// Keyed like the prefill indicator: a first turn reads "__default" until the thread persists.
const useLiveThreadKey = () =>
  useAuiState(({ threadListItem }) => threadListItem.remoteId) ?? "__default";

const LiveDecodeFigures: FC = () => {
  const threadKey = useLiveThreadKey();
  const decode = useChatRuntimeStore(
    (s) => s.liveGenerationByThreadId[threadKey]?.decode ?? null,
  );
  return <DecodeFigures decode={decode} />;
};

const ThrottledDecodeFigures: FC = () => {
  const decode = useThrottledLiveGeneration(useLiveThreadKey())?.decode ?? null;
  return <DecodeFigures decode={decode} />;
};

const DecodeFigures: FC<{ decode: LiveDecodeProgress | null }> = ({ decode }) => {
  const locale = useLocale();
  const t = useT();
  if (!decode) {
    return null;
  }
  return (
    <span
      data-slot="live-generation-line"
      className="aui-live-generation-line ml-[var(--icon-btn-inset)] mr-2 min-w-0 select-none font-mono text-ui-11p5 text-muted-foreground tabular-nums"
    >
      {formatLiveDecode(decode, t, locale)}
    </span>
  );
};
