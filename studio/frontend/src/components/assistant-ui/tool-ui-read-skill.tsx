// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Spinner } from "@/components/ui/spinner";
import { useToolAwaitingApproval } from "@/features/chat/tool-approval";
import { stringifyToolResult } from "@/lib/strip-ansi";
import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { partsHaveNonEmptyText } from "@/components/assistant-ui/message-derived";
import { Scroll01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ComponentProps, memo } from "react";
import { toolArgText } from "./tool-arg-text";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { useToolActivityOpen } from "./use-tool-activity-open";
import { ScrollPane } from "./scroll-pane";

// `strokeWidth` is dropped: SVG types it `string | number`, HugeiconsIcon wants a number.
function SkillIcon({
  strokeWidth: _strokeWidth,
  ...props
}: ComponentProps<"svg">) {
  return <HugeiconsIcon icon={Scroll01Icon} {...props} />;
}

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
  const isPreload = (args as { _studio_skill_load?: unknown })?._studio_skill_load === true;
  const preloadLoaded = resultText.startsWith("Complete SKILL.md read");
  const label = isPreload
    ? isRunning
      ? `Loading ${name}…`
      : preloadLoaded
        ? `Loaded ${name} · ${resource}`
        : `Skill not loaded · ${name}`
    : isRunning
      ? `Reading ${name}…`
      : `Read ${name} · ${resource}`;
  const hasText = useAuiState(({ message }) =>
    partsHaveNonEmptyText(message.content),
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
        toolName={label}
        status={status}
        icon={SkillIcon}
      />
      <ToolFallbackContent>
        {isRunning ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Spinner className="size-3.5" />
            <span>Reading {name}&hellip;</span>
          </div>
        ) : resultText ? (
          <ScrollPane
            className="rounded bg-muted/50 p-2"
            scrollerClassName="max-h-64 overflow-auto whitespace-pre-wrap break-words text-xs"
          >
            {resultText}
          </ScrollPane>
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
