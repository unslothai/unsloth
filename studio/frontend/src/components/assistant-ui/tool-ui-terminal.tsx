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
import { toolArgText } from "./tool-arg-text";
import { ToolLiveOutput } from "./tool-live-output";
import { ToolResultOutput } from "./tool-result-output";
import { SandboxFiles } from "./sandbox-files-view";
import { isSandboxToolResult, type SandboxFile } from "./sandbox-files";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";

import { stringifyToolResult } from "@/lib/strip-ansi";
import {
  preferSanitizedFullToolOutput,
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
  const isRunning = status?.type === "running";
  // Args still streaming = the model is WRITING the command, not running it yet.
  const { propStatus } = useToolArgsStatus();
  const isWritingCommand = isRunning && propStatus.command === "streaming";
  // A command that wrote files arrives as the python tool's structured shape;
  // a plain string means it wrote none.
  // The same test the adapter applies: a foreign result that merely has text
  // would otherwise be rendered as that field alone.
  const structured = isSandboxToolResult(result)
    ? (result as unknown as { text: string; sessionId?: string; files?: SandboxFile[] })
    : null;
  const files = structured?.files ?? [];
  const sessionId = structured?.sessionId ?? "";
  const output =
    structured !== null
      ? stringifyToolResult(structured.text)
      : result == null
        ? ""
        : stringifyToolResult(result);

  // Show the fuller live stream over a truncated result, keeping its exit
  // status. Session-transient: after a reload only the result remains.
  const paneScope = useToolPaneScope();
  const fullOutput = useToolOutputFor(
    useChatRuntimeStore((s) => s.toolFullOutput),
    paneScope,
    toolCallId,
  );
  // Compare the same plain-text representation on both sides. Otherwise a raw
  // SGR-prefixed stream cannot match its cleaned truncated result and the
  // reconciliation helper appends a duplicate prefix.
  const displayOutput = preferSanitizedFullToolOutput(fullOutput, output);
  // The gate only opens once the call parsed, so a pending approval means the command is
  // written even while the args status still reads as streaming.
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const isWriting = isWritingCommand && !awaitingApproval;

  return (
    // Open mid-run so command and live output show, collapsed from history.
    // awaitingApproval pins it open past the collapse preference: the command
    // lives inside ToolFallbackContent, and Allow/Always allow/Deny render
    // outside the card, so a collapsed card would ask for a decision about a
    // command the user cannot read -- the trigger only carries its first 60
    // characters, untruncated and with no ellipsis to say so.
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
          {/* Files the command wrote; this card used to show nothing for them */}
          <SandboxFiles sessionId={sessionId} files={files} />
        </div>
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const TerminalToolUI = memo(
  TerminalToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
TerminalToolUI.displayName = "TerminalToolUI";
