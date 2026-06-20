// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { CopyIcon, FileTextIcon, TerminalIcon } from "lucide-react";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { Spinner } from "@/components/ui/spinner";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";

/**
 * Renders synthetic `_toolEvent` chunks from `_stream_anthropic` for the
 * `code_execution_20250825` tool. The backend collapses Anthropic's two
 * sub-tools into `tool_name: "code_execution"` with `arguments.kind`:
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

const MAX_COMMAND_LABEL = 80;
const MAX_RESULT_DISPLAY = 10_000;
const COPY_RESET_MS = 2000;

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

function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (timer.current) {
        clearTimeout(timer.current);
      }
    };
  }, []);

  const copy = useCallback(async () => {
    if (await copyToClipboard(text)) {
      setCopied(true);
      if (timer.current) {
        clearTimeout(timer.current);
      }
      timer.current = setTimeout(() => setCopied(false), COPY_RESET_MS);
    }
  }, [text]);

  return (
    <button
      type="button"
      onClick={copy}
      className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      aria-label="Copy to clipboard"
    >
      {copied ? (
        <HugeiconsIcon icon={Tick02Icon} strokeWidth={2} className="size-3" />
      ) : (
        <CopyIcon className="size-3" />
      )}
      {copied ? "Copied" : "Copy"}
    </button>
  );
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

  const commandLabel = command ? truncateCommandLabel(command) : "";

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
    completedLabel = commandLabel ? `Ran \`${commandLabel}\`` : "Ran command";
  }

  // Collapse the card once the model resumes streaming prose after the tool
  // call (mirrors WebSearchToolUI) so it doesn't crowd the final answer.
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

  const resultText = useMemo(
    () =>
      typeof result === "string"
        ? result
        : result != null
          ? JSON.stringify(result, null, 2)
          : "",
    [result],
  );
  const displayedResult = useMemo(
    () => truncateResult(resultText),
    [resultText],
  );

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
            <Spinner className="size-3.5" />
            <span>{runningLabel}</span>
          </div>
        ) : resultText ? (
          <div>
            <div className="flex justify-end">
              <CopyBtn text={resultText} />
            </div>
            <pre className="mt-1 max-h-64 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/50 p-2 text-xs">
              {displayedResult}
            </pre>
          </div>
        ) : null}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const CodeExecutionToolUI = memo(
  CodeExecutionToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
CodeExecutionToolUI.displayName = "CodeExecutionToolUI";
