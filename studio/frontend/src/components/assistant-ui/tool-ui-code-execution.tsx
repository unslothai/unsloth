// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { FileGlyph } from "@/lib/file-icon";
import { stringifyToolResult } from "@/lib/strip-ansi";
import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { partsHaveNonEmptyText } from "@/components/assistant-ui/message-derived";
import { TerminalIcon } from "lucide-react";
import { Spinner } from "@/components/ui/spinner";
import { isToolCallRunning, toolArgText } from "./tool-arg-text";
import { type ComponentType, memo, useMemo } from "react";
import { useToolAwaitingApproval } from "@/features/chat";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { useToolActivityOpen } from "./use-tool-activity-open";
import { ScrollPane } from "./scroll-pane";
import { CopyBtn } from "./tool-code-cell";

/**
 * Backend collapses Anthropic's code_execution sub-tools into `arguments.kind`: bash
 * `{ command }` or text_editor `{ command: view|create|str_replace, path, ... }`; `result` is
 * preformatted text.
 */
interface CodeExecutionArgs {
  kind?: "bash" | "text_editor";
  // Straight off the wire: the model, not the schema, decides the JSON type.
  command?: unknown;
  path?: unknown;
}

const MAX_COMMAND_LABEL = 80;
const MAX_RESULT_DISPLAY = 10_000;

function truncateCommandLabel(text: string): string {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (normalized.length <= MAX_COMMAND_LABEL) {
    return normalized;
  }
  const head = Math.ceil((MAX_COMMAND_LABEL - 3) * 0.65);
  const tail = MAX_COMMAND_LABEL - head - 3;
  return `${normalized.slice(0, head)}...${normalized.slice(-tail)}`;
}

function truncateResult(text: string): string {
  return text.length <= MAX_RESULT_DISPLAY
    ? text
    : `${text.slice(0, MAX_RESULT_DISPLAY)}\n... (truncated)`;
}

export function CodeExecutionResultOutput({ result }: { result: unknown }) {
  const resultText = useMemo(
    () => (result == null ? "" : stringifyToolResult(result)),
    [result],
  );
  const displayedResult = useMemo(
    () => truncateResult(resultText),
    [resultText],
  );

  if (!resultText) {
    return null;
  }
  return (
    <div>
      <div className="flex justify-end">
        <CopyBtn text={resultText} />
      </div>
      <ScrollPane
        className="mt-1 rounded bg-muted/50 p-2"
        scrollerClassName="max-h-64 overflow-auto whitespace-pre-wrap break-words text-xs"
      >
        {displayedResult}
      </ScrollPane>
    </div>
  );
}



const CodeExecutionToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
  toolCallId,
}) => {
  const parsedArgs = (args as CodeExecutionArgs) ?? {};
  const kind = parsedArgs.kind ?? "bash";
  const command = toolArgText(parsedArgs.command);
  const path = toolArgText(parsedArgs.path);
  const isRunning = isToolCallRunning(status);

  const commandLabel = command ? truncateCommandLabel(command) : "";

  let runningLabel: string;
  let completedLabel: string;
  let Icon: ComponentType<{ className?: string }> = TerminalIcon;
  if (kind === "text_editor") {
    Icon = FileGlyph;
    if (command === "view") {
      runningLabel = path ? `Viewing ${path}…` : "Viewing file…";
      completedLabel = path ? `Viewed ${path}` : "Viewed file";
    } else if (command === "create") {
      runningLabel = path ? `Writing ${path}…` : "Writing file…";
      completedLabel = path ? `Wrote ${path}` : "Wrote file";
    } else if (command === "str_replace") {
      runningLabel = path ? `Editing ${path}…` : "Editing file…";
      completedLabel = path ? `Edited ${path}` : "Edited file";
    } else {
      runningLabel = "Running file operation…";
      completedLabel = "File operation";
    }
  } else {
    runningLabel = "Running command…";
    completedLabel = commandLabel ? `Ran \`${commandLabel}\`` : "Ran command";
  }

  // Collapse once prose resumes after the call (like WebSearchToolUI).
  const hasText = useAuiState(({ message }) =>
    partsHaveNonEmptyText(message.content),
  );
  // What is being approved lives inside the content, while Allow/Deny render outside it.
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const [open, setOpen] = useToolActivityOpen(isRunning, hasText);

  return (
    <ToolFallbackRoot
      open={open}
      onOpenChange={setOpen}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        toolName={isRunning ? runningLabel : completedLabel}
        status={status}
        icon={Icon}
      />
      <ToolFallbackContent>
        {isRunning ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Spinner className="size-3.5" />
            <span>{runningLabel}</span>
          </div>
        ) : (
          <CodeExecutionResultOutput result={result} />
        )}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const CodeExecutionToolUI = memo(
  CodeExecutionToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
CodeExecutionToolUI.displayName = "CodeExecutionToolUI";
