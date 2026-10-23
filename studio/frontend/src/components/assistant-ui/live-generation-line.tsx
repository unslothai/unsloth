// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  type LiveDecodeProgress,
  useChatPreferencesStore,
  useChatRuntimeStore,
} from "@/features/chat";
import { useLocale } from "@/i18n";
import { useAuiState } from "@assistant-ui/react";
import type { FC } from "react";

/** Gated on the preference first so a reply mounts no assistant-ui selector while it is off
 *  (composer-keystroke-subscription-budget), like AssistantMessageTime. */
export const LiveGenerationLine: FC = () => {
  const enabled = useChatPreferencesStore((state) => state.showGenerationDetails);
  return enabled ? <RunningGate /> : null;
};

const RunningGate: FC = () => {
  const running = useAuiState(({ message }) => message.status?.type === "running");
  return running ? <LiveDecodeFigures /> : null;
};

const LiveDecodeFigures: FC = () => {
  const locale = useLocale();
  // Keyed like the prefill indicator: a first turn reads "__default" until the thread persists.
  const threadKey =
    useAuiState(({ threadListItem }) => threadListItem.remoteId) ?? "__default";
  const decode = useChatRuntimeStore(
    (s) => s.liveGenerationByThreadId[threadKey]?.decode ?? null,
  );
  if (!decode) {
    return null;
  }
  return (
    <span
      data-slot="live-generation-line"
      className="aui-live-generation-line ml-[var(--icon-btn-inset)] mr-2 shrink-0 select-none font-mono text-ui-11p5 text-muted-foreground tabular-nums"
    >
      {formatLiveDecode(decode, locale)}
    </span>
  );
};

function formatLiveDecode(decode: LiveDecodeProgress, locale?: string): string {
  const parts = [
    `${decode.tokens.toLocaleString(locale)} ${decode.tokens === 1 ? "token" : "tokens"}`,
  ];
  if (decode.tokPerS !== null) {
    parts.push(`${decode.tokPerS.toFixed(1)} tok/s`);
  }
  if (decode.draftN && decode.draftAccepted !== null) {
    const share = Math.round((decode.draftAccepted / decode.draftN) * 100);
    parts.push(`draft ${decode.draftAccepted}/${decode.draftN} accepted (${share}%)`);
  }
  return parts.join(" · ");
}
