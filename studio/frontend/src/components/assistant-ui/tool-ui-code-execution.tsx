// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { type ToolCallMessagePartComponent, useAuiState } from "@assistant-ui/react";
import { FileTextIcon, LoaderIcon, TerminalIcon } from "lucide-react";
import { memo, useEffect, useState } from "react";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";

/**
 * Renders the synthetic `_toolEvent` chunks emitted by
 * `_stream_anthropic` when Anthropic's `code_execution_20250825` tool
 * fires. The backend collapses Anthropic's two sub-tools
 * (`bash_code_execution`, `text_editor_code_execution`) into a single
 * `tool_name: "code_execution"`, with `arguments.kind` ("bash" or
 * "text_editor") and a per-kind argument shape:
 *
 *   kind=bash:        { command: "<shell command>" }
 *   kind=text_editor: { command: "view"|"create"|"str_replace", path, ... }
 *
 * The `result` payload is preformatted text:
 *   - bash: stdout, then "--- stderr ---" block + return_code if non-zero
 *   - text_editor view:        file contents verbatim
 *   - text_editor create:      "Created <path>" / "Updated <path>"
 *   - text_editor str_replace: unified-diff `lines` joined with "\n"
 *   - error:                   "Error: <error_code>"
 */
interface CodeExecutionArgs {
  kind?: "bash" | "text_editor";
  command?: string;
  path?: string;
}

const CodeExecutionToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
}) => {
  const parsedArgs = (args as CodeExecutionArgs) ?? {};
  const kind = parsedArgs.kind ?? "bash";
  const command = parsedArgs.command ?? "";
  const path = parsedArgs.path ?? "";
  const isRunning = status?.type === "running";

  let runningLabel: string;
  let completedLabel: string;
  let Icon = TerminalIcon;
  if (kind === "text_editor") {
    Icon = FileTextIcon;
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
    completedLabel = command ? `Ran \`${command}\`` : "Ran command";
  }

  // Collapse the card once the model has resumed streaming prose after
  // the tool call. Mirrors WebSearchToolUI's behavior so the tool-card
  // doesn't crowd the final answer once the run is done.
  const hasText = useAuiState(({ message }) =>
    message.content.some(
      (p) =>
        p.type === "text" &&
        "text" in p &&
        (p as { text: string }).text.length > 0,
    ),
  );
  const [open, setOpen] = useState(isRunning);
  useEffect(() => {
    if (isRunning) {
      setOpen(true);
    } else if (hasText) {
      setOpen(false);
    }
  }, [isRunning, hasText]);

  const resultText =
    typeof result === "string"
      ? result
      : result != null
        ? JSON.stringify(result, null, 2)
        : "";

  return (
    <ToolFallbackRoot open={open} onOpenChange={setOpen}>
      <ToolFallbackTrigger
        toolName={isRunning ? runningLabel : completedLabel}
        status={status}
        icon={Icon}
      />
      <ToolFallbackContent>
        {isRunning ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <LoaderIcon className="size-3.5 animate-spin" />
            <span>{runningLabel}</span>
          </div>
        ) : resultText ? (
          <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/50 p-2 text-xs">
            {resultText}
          </pre>
        ) : null}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const CodeExecutionToolUI = memo(
  CodeExecutionToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
CodeExecutionToolUI.displayName = "CodeExecutionToolUI";
