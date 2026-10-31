// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { useToolArgsStatus } from "@assistant-ui/react";
import { TerminalIcon } from "lucide-react";
import { Spinner } from "@/components/ui/spinner";
import { memo } from "react";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { isToolCallRunning, toolArgText } from "./tool-arg-text";
import { CopyBtn, ToolCodeCell } from "./tool-code-cell";
import { ToolLiveOutput } from "./tool-live-output";
import { ToolResultOutput } from "./tool-result-output";
import { SandboxFiles } from "./sandbox-files-view";
import { isSandboxToolResult, type SandboxFile } from "./sandbox-files";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";

import {
  preferSanitizedFullToolOutput,
  toolResultText,
  useToolAwaitingApproval,
  useToolOutputFor,
  useToolPaneScope,
} from "@/features/chat";

const TerminalToolUIImpl: ToolCallMessagePartComponent = ({
  toolCallId,
  args,
  result,
  status,
}) => {
  const command = toolArgText((args as { command?: unknown })?.command);
  const isRunning = isToolCallRunning(status);
  // Args still streaming = the model is WRITING the command, not running it yet.
  const { propStatus } = useToolArgsStatus();
  const isWritingCommand = isRunning && propStatus.command === "streaming";
  // Same test the adapter applies; a foreign result with only text would otherwise render alone.
  const structured = isSandboxToolResult(result)
    ? (result as unknown as { text: string; sessionId?: string; files?: SandboxFile[] })
    : null;
  const files = structured?.files ?? [];
  const sessionId = structured?.sessionId ?? "";
  const output =
    structured !== null
      ? toolResultText(structured.text)
      : result == null
        ? ""
        : toolResultText(result);

  // Prefer the fuller live stream over a truncated result; after a reload only the result remains.
  const paneScope = useToolPaneScope();
  const fullOutput = useToolOutputFor(
    useChatRuntimeStore((s) => s.toolFullOutput),
    paneScope,
    toolCallId,
  );
  // Compare sanitized text on both sides or the reconciliation appends a duplicate prefix.
  const displayOutput = preferSanitizedFullToolOutput(fullOutput, output);
  // The gate opens only once the call parsed, so pending approval means the command is written.
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const isWriting = isWritingCommand && !awaitingApproval;

  return (
    // awaitingApproval overrides the preference: the trigger shows only 60 characters.
    <ToolFallbackRoot
      defaultOpen={isRunning}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        toolName={command ? `$ ${command.slice(0, 60)}` : "Terminal"}
        status={status}
        icon={TerminalIcon}
      />
      <ToolFallbackContent>
        {command && (
          <ToolCodeCell
            label="command"
            code={command}
            language="bash"
            downloadName="command.sh"
            streaming={isWriting}
          />
        )}
        <div className="border-l-2 border-muted-foreground/20 pl-2">
          {isRunning ? (
            <>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Spinner className="size-3.5" />
                <span>
                  {awaitingApproval
                    ? "Waiting for approval…"
                    : isWriting
                      ? "Writing command…"
                      : "Running…"}
                </span>
              </div>
              <ToolLiveOutput toolCallId={toolCallId} />
            </>
          ) : displayOutput ? (
            <div>
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">output</span>
                <CopyBtn text={displayOutput} />
              </div>
              <ToolResultOutput text={displayOutput} />
            </div>
          ) : null}
        </div>
      </ToolFallbackContent>
      {/* Files stay outside even when the card is collapsed (#10425) */}
      <SandboxFiles className="ml-5" sessionId={sessionId} files={files} />
    </ToolFallbackRoot>
  );
};

export const TerminalToolUI = memo(
  TerminalToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
TerminalToolUI.displayName = "TerminalToolUI";
