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
import { CopyBtn, ToolCodeCell } from "./tool-code-cell";
import { ToolLiveOutput } from "./tool-live-output";
import { ToolResultOutput } from "./tool-result-output";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";

import { stringifyToolResult } from "@/lib/strip-ansi";
import {
  preferFullToolOutput,
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
  const command = (args as { command?: string })?.command ?? "";
  const isRunning = status?.type === "running";
  // Args still streaming = the model is WRITING the command, not running it yet.
  const { propStatus } = useToolArgsStatus();
  const isWritingCommand = isRunning && propStatus.command === "streaming";
  const output = result == null ? "" : stringifyToolResult(result);

  // Show the fuller live stream over a truncated result, keeping its exit
  // status. Session-transient: after a reload only the result remains.
  const paneScope = useToolPaneScope();
  const fullOutput = useToolOutputFor(
    useChatRuntimeStore((s) => s.toolFullOutput),
    paneScope,
    toolCallId,
  );
  const displayOutput = preferFullToolOutput(fullOutput, output);
  // The gate only opens once the call parsed, so a pending approval means the command is
  // written even while the args status still reads as streaming.
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const isWriting = isWritingCommand && !awaitingApproval;

  return (
    // Open mid-run so command and live output show, collapsed from history.
    <ToolFallbackRoot defaultOpen={isRunning}>
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
              {/* Live stdout streamed via tool_output SSE events. */}
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
    </ToolFallbackRoot>
  );
};

export const TerminalToolUI = memo(
  TerminalToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
TerminalToolUI.displayName = "TerminalToolUI";
