// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { getAuthToken } from "@/features/auth/session";
import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { useToolArgsStatus } from "@assistant-ui/react";
import { CodeIcon } from "lucide-react";
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
import {
  preferFullToolOutput,
  useToolAwaitingApproval,
  useToolOutputFor,
  useToolPaneScope,
} from "@/features/chat";

interface StructuredResult {
  text: string;
  images: string[];
  sessionId: string;
}

function isStructuredResult(val: unknown): val is StructuredResult {
  return (
    typeof val === "object" &&
    val !== null &&
    "text" in val &&
    "images" in val &&
    "sessionId" in val
  );
}

const PythonToolUIImpl: ToolCallMessagePartComponent = ({
  toolCallId,
  args,
  result,
  status,
}) => {
  const code = (args as { code?: string })?.code ?? "";
  const firstLine = code.split("\n")[0]?.slice(0, 60) ?? "";
  const isRunning = status?.type === "running";
  // Args still streaming = the model is WRITING the code, not running it yet.
  const { propStatus } = useToolArgsStatus();
  const isWritingCode = isRunning && propStatus.code === "streaming";

  let output: string;
  let images: string[] = [];
  let sessionId = "";

  if (isStructuredResult(result)) {
    output = result.text;
    images = result.images;
    sessionId = result.sessionId;
  } else if (typeof result === "string") {
    output = result;
  } else if (result) {
    output = JSON.stringify(result, null, 2);
  } else {
    output = "";
  }

  // Show the fuller live stream over a truncated result, keeping its exit
  // status. Session-transient: after a reload only the result remains.
  const paneScope = useToolPaneScope();
  const fullOutput = useToolOutputFor(
    useChatRuntimeStore((s) => s.toolFullOutput),
    paneScope,
    toolCallId,
  );
  const displayOutput = preferFullToolOutput(fullOutput, output);

  const authToken = getAuthToken();
  // The gate only opens once the call parsed, so a pending approval means the script is
  // written even while the args status still reads as streaming.
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const isWriting = isWritingCode && !awaitingApproval;

  return (
    // Status, output and images collapse from history; the executed script
    // renders outside ToolFallbackContent so it stays visible on reopen
    // (#7165). Terminal keeps its command inside the collapsible -- a one-line
    // command is not the artifact a user comes back for, a script is.
    <ToolFallbackRoot defaultOpen={isRunning}>
      <ToolFallbackTrigger
        toolName={firstLine ? `Python: ${firstLine}` : "Python"}
        status={status}
        icon={CodeIcon}
      />
      {code && (
        <div className="mt-1 pl-5">
          <ToolCodeCell
            label="script"
            code={code}
            language="python"
            downloadName="script.py"
            streaming={isWriting}
          />
        </div>
      )}
      <ToolFallbackContent>
        <div className="border-l-2 border-muted-foreground/20 pl-2">
          {/* Output */}
          {isRunning ? (
            <>
              <div className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
                <Spinner className="size-3.5" />
                <span>
                  {awaitingApproval
                    ? "Waiting for approval…"
                    : isWriting
                      ? "Writing code…"
                      : "Running…"}
                </span>
              </div>
              {/* Live stdout streamed via tool_output SSE events. */}
              <ToolLiveOutput toolCallId={toolCallId} />
            </>
          ) : displayOutput ? (
            <div className="mt-2 border-t border-dashed pt-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">output</span>
                <CopyBtn text={displayOutput} />
              </div>
              <ToolResultOutput text={displayOutput} />
            </div>
          ) : null}

          {/* Images from Python tool execution */}
          {images.length > 0 && sessionId && (
            <div className="mt-2 flex flex-col gap-2">
              {images.map((filename) => (
                <img
                  key={filename}
                  src={`/api/inference/sandbox/${encodeURIComponent(sessionId)}/${encodeURIComponent(filename)}${authToken ? `?token=${encodeURIComponent(authToken)}` : ""}`}
                  alt={filename}
                  loading="lazy"
                  className="max-w-full rounded border border-border"
                />
              ))}
            </div>
          )}
        </div>
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const PythonToolUI = memo(
  PythonToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
PythonToolUI.displayName = "PythonToolUI";
