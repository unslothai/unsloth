// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { partsHaveNonEmptyText } from "@/components/assistant-ui/message-derived";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import {
  openSubagentPanelForDescription,
  useToolAwaitingApproval,
} from "@/features/chat";
import { useT } from "@/i18n";
import { stringifyToolResult } from "@/lib/strip-ansi";
import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { Robot01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { memo } from "react";
import { ScrollPane } from "./scroll-pane";
import { toolArgText } from "./tool-arg-text";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { useToolActivityOpen } from "./use-tool-activity-open";

function SubagentIcon({ className }: { className?: string }) {
  return <HugeiconsIcon icon={Robot01Icon} className={className} />;
}

/** The `spawn_subagent` call, labelled by its description, with a shortcut into the side panel. */
const SubagentToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
  toolCallId,
}) => {
  const t = useT();
  const description =
    toolArgText((args as { description?: unknown })?.description).trim() ||
    "subagent";
  const isRunning = status?.type === "running";
  const resultText = result == null ? "" : stringifyToolResult(result);
  const hasText = useAuiState(({ message }) =>
    partsHaveNonEmptyText(message.content),
  );
  const remoteThreadId = useAuiState(
    ({ threadListItem }) => threadListItem.remoteId,
  );
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const [open, setOpen] = useToolActivityOpen(isRunning, hasText);

  const showInPanel = () =>
    openSubagentPanelForDescription(remoteThreadId, description);

  return (
    <ToolFallbackRoot
      open={open}
      onOpenChange={setOpen}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        toolName={t("subagents.toolCardLabel", { description })}
        status={status}
        icon={SubagentIcon}
      />
      <ToolFallbackContent>
        <div className="flex flex-col gap-2">
          {isRunning ? (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Spinner className="size-3.5" />
              <span>{t("subagents.working")}</span>
            </div>
          ) : resultText ? (
            <ScrollPane
              className="rounded-lg bg-muted/50 p-2"
              scrollerClassName="max-h-64 overflow-auto whitespace-pre-wrap break-words text-xs"
            >
              {resultText}
            </ScrollPane>
          ) : null}
          <div>
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="h-7 px-2.5 text-ui-11"
              onClick={showInPanel}
            >
              {t("subagents.viewInPanel")}
            </Button>
          </div>
        </div>
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const SubagentToolUI = memo(
  SubagentToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
SubagentToolUI.displayName = "SubagentToolUI";
