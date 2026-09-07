// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { useToolAwaitingApproval } from "@/features/chat/tool-approval";
import { stringifyToolResult } from "@/lib/strip-ansi";
import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { BookOpenIcon } from "lucide-react";
import { memo } from "react";
import { toolArgText } from "./tool-arg-text";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { useToolActivityOpen } from "./use-tool-activity-open";

const ReadSkillToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
  toolCallId,
}) => {
  const name = toolArgText((args as { name?: unknown })?.name) || "skill";
  const resource =
    toolArgText((args as { resource?: unknown })?.resource) || "SKILL.md";
  const isRunning = status?.type === "running";
  const resultText = result == null ? "" : stringifyToolResult(result);
  const hasText = useAuiState(({ message }) =>
    message.content.some(
      (part) =>
        part.type === "text" &&
        "text" in part &&
        (part as { text: string }).text.length > 0,
    ),
  );
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const [open, setOpen] = useToolActivityOpen(isRunning, hasText);

  return (
    <ToolFallbackRoot
      open={open}
      onOpenChange={setOpen}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        toolName={isRunning ? `Reading ${name}…` : `Read ${name} · ${resource}`}
        status={status}
        icon={BookOpenIcon}
      />
      <ToolFallbackContent>
        {isRunning ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Spinner className="size-3.5" />
            <span>Reading {name}&hellip;</span>
          </div>
        ) : resultText ? (
          <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/50 p-2 text-xs">
            {resultText}
          </pre>
        ) : (
          <div className="text-sm text-muted-foreground">
            Loaded {resource}.
          </div>
        )}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const ReadSkillToolUI = memo(
  ReadSkillToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
ReadSkillToolUI.displayName = "ReadSkillToolUI";
