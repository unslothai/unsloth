// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Spinner } from "@/components/ui/spinner";
// eslint-disable-next-line no-restricted-imports -- the chat barrel imports this card
import { useToolAwaitingApproval } from "@/features/chat/tool-approval";
import { stringifyToolResult } from "@/lib/strip-ansi";
import type {
  ToolCallMessagePartComponent,
  ToolCallMessagePartStatus,
} from "@assistant-ui/react";
import { FileTextIcon } from "lucide-react";
import { memo, type ComponentProps } from "react";

import {
  editFileChangeLabel,
  editFileIncompleteTitle,
  editFileResultIsError,
  editFileResultWasDeclined,
  summarizeEditFileArgs,
} from "./edit-file-tool-summary";
import { toolArgText } from "./tool-arg-text";
import {
  ToolFallbackContent,
  ToolFallbackArgs,
  ToolFallbackError,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { ToolResultOutput } from "./tool-result-output";

function runningEditLabel(
  mode: "create" | "replace",
  awaitingApproval: boolean,
): string {
  if (awaitingApproval) {
    return "Waiting for approval...";
  }
  return mode === "create" ? "Creating file..." : "Applying replacements...";
}

function completedEditTitle(
  mode: "create" | "replace",
  path: string,
  isRunning: boolean,
  failed: boolean,
): string {
  if (failed) {
    return `Edit failed: ${path}`;
  }
  if (mode === "create") {
    return `${isRunning ? "Creating" : "Created"}: ${path}`;
  }
  return `${isRunning ? "Editing" : "Edited"}: ${path}`;
}

const EditFileToolUIImpl: ToolCallMessagePartComponent = ({
  toolCallId,
  args,
  result,
  status,
}) => {
  const summary = summarizeEditFileArgs(args);
  const path = toolArgText((args as { path?: unknown })?.path).trim() || "file";
  const changeLabel = editFileChangeLabel(summary);
  const isRunning = status?.type === "running";
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const output = result == null ? "" : stringifyToolResult(result);
  const incompleteTitle = editFileIncompleteTitle(status);
  const failed =
    !isRunning && (editFileResultIsError(output) || incompleteTitle !== null);
  const displayStatus: ToolCallMessagePartStatus | undefined =
    failed && !incompleteTitle
      ? { type: "incomplete", reason: "error" }
      : status;
  const title = incompleteTitle
    ? `${incompleteTitle}: ${path}`
    : !isRunning && editFileResultWasDeclined(output)
      ? `Edit declined: ${path}`
      : completedEditTitle(summary.mode, path, isRunning, failed);

  return (
    <ToolFallbackRoot
      defaultOpen={isRunning}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        toolName={title}
        status={displayStatus}
        icon={failed ? undefined : FileTextIcon}
      />
      <ToolFallbackContent>
        <ToolFallbackError status={displayStatus} />
        <div className="border-l-2 border-muted-foreground/20 pl-2">
          <div className="grid gap-1 text-xs">
            <div className="flex min-w-0 gap-2">
              <span className="shrink-0 font-medium text-muted-foreground">
                file
              </span>
              <code className="min-w-0 break-all text-foreground/85">
                {path}
              </code>
            </div>
            <div className="flex gap-2">
              <span className="font-medium text-muted-foreground">changes</span>
              <span className="text-foreground/85">{changeLabel}</span>
            </div>
            {summary.replaceAllCount > 0 ? (
              <p className="text-muted-foreground">
                {summary.replaceAllCount} replacement
                {summary.replaceAllCount === 1 ? "" : "s"} may match multiple
                locations.
              </p>
            ) : null}
            {awaitingApproval ? (
              <div className="mt-2 max-h-80 overflow-auto rounded border p-2">
                <ToolFallbackArgs argsText={toolArgText(args)} />
              </div>
            ) : null}
          </div>
          {isRunning ? (
            <div className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
              <Spinner className="size-3.5" />
              <span>{runningEditLabel(summary.mode, awaitingApproval)}</span>
            </div>
          ) : output ? (
            <div className="mt-2 border-t border-dashed pt-2">
              <ToolResultOutput text={output} />
            </div>
          ) : null}
        </div>
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

const MemoizedEditFileToolUI = memo(
  EditFileToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
MemoizedEditFileToolUI.displayName = "EditFileToolUI";

// The thread installs this card during chat-module initialization. A hoisted
// entry point avoids reading the memo binding while that import cycle is open.
export function EditFileToolUI(
  props: ComponentProps<typeof MemoizedEditFileToolUI>,
) {
  return <MemoizedEditFileToolUI {...props} />;
}
