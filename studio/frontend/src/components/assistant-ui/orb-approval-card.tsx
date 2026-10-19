// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import { decideToolCall } from "@/features/chat/tool-decision";
import { toolBrief } from "@/features/chat/voice/approval-brief";
import { alwaysAllowToolName } from "@/features/chat/voice/orb-widget";
import { describeToolCall } from "@/features/chat/voice/voice-loop-bridge";
import { useT } from "@/i18n";
import { type FC, useState } from "react";

// The orb's own surface for a tool call waiting on Allow / Deny: the same decision the chat
// card and the floating orb's popup make (decideToolCall), so whichever answers, all three
// clear. Always allow stays here, as on the chat card; only the tiny floating popup drops it.
export const OrbApprovalCard: FC<{ id: string; more: number }> = ({ id, more }) => {
  const t = useT();
  const disclosure = useChatRuntimeStore((s) => s.toolConfirmations[id]?.imageDisclosure);
  const [pending, setPending] = useState(false);
  // Always allow grants a tool by name for the chat: with the call not found, it is not offered.
  const alwaysName = disclosure ? null : alwaysAllowToolName(describeToolCall(id));
  const decide = (decision: "allow" | "deny", always = false) => {
    setPending(true);
    void decideToolCall(id, decision, always && alwaysName ? alwaysName : undefined).finally(() =>
      setPending(false),
    );
  };
  const brief = toolBrief(id);
  return (
    <div
      data-orb-approval
      // One row: the call on the left, the answers on the right; a narrow window wraps the
      // buttons under the text rather than overflowing.
      className="flex w-[min(calc(36rem*var(--ui-space-scale,1)),calc(100vw-2rem))] flex-wrap items-center justify-end gap-x-3 gap-y-2 rounded-2xl border border-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)] bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)] px-4 py-2.5"
    >
      <span className="flex min-w-[calc(12rem*var(--ui-space-scale,1))] flex-1 items-center gap-2">
        <span className="min-w-0 truncate text-ui-13 text-trace-foreground" title={brief}>
          {brief}
        </span>
        {more > 0 ? (
          <span className="shrink-0 text-ui-11 text-muted-foreground">{t("orbWidget.more", { n: String(more) })}</span>
        ) : null}
      </span>
      <span className="flex shrink-0 items-center gap-2">
        <Button size="xs" disabled={pending} onClick={() => decide("allow")}>
          {t("browser.agent.allow")}
        </Button>
        {alwaysName ? (
          <Button size="xs" variant="outline" disabled={pending} onClick={() => decide("allow", true)}>
            {t("browser.agentSettings.alwaysAllow")}
          </Button>
        ) : null}
        <Button size="xs" variant="destructive" disabled={pending} onClick={() => decide("deny")}>
          {t("browser.agent.deny")}
        </Button>
      </span>
    </div>
  );
};
