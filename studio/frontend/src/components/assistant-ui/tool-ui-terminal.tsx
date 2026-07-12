// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { useToolArgsStatus } from "@assistant-ui/react";
import { CopyIcon, TerminalIcon } from "lucide-react";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { Spinner } from "@/components/ui/spinner";
import { memo, useCallback, useEffect, useRef, useState } from "react";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { ToolLiveOutput } from "./tool-live-output";
import { ToolResultOutput } from "./tool-result-output";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";

const COPY_RESET_MS = 2000;

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

const TerminalToolUIImpl: ToolCallMessagePartComponent = ({
  toolCallId,
  args,
  result,
  status,
}) => {
  const command = (args as { command?: string })?.command ?? "";
  const isRunning = status?.type === "running";
  // Args still streaming = the model is WRITING the command (tool_args
  // events); execution has not started yet, so say so instead of "Running".
  const { propStatus } = useToolArgsStatus();
  const isWritingCommand = isRunning && propStatus.command === "streaming";
  const output =
    typeof result === "string"
      ? result
      : result
        ? JSON.stringify(result, null, 2)
        : "";

  // When the live stream captured more than the (model-visible, truncated)
  // final result, show the full stream instead. Session-transient: after a
  // reload only the truncated result remains.
  const fullOutput = useChatRuntimeStore(
    (s) => s.toolFullOutput[toolCallId] ?? "",
  );
  const displayOutput =
    fullOutput.length > output.length ? fullOutput : output;

  return (
    // Mounted mid-run (tool_start) the card opens so live output is visible;
    // mounted from history (not running) it stays collapsed as before.
    <ToolFallbackRoot defaultOpen={isRunning}>
      <ToolFallbackTrigger
        toolName={command ? `$ ${command.slice(0, 60)}` : "Terminal"}
        status={status}
        icon={TerminalIcon}
      />
      <ToolFallbackContent>
        <div className="border-l-2 border-muted-foreground/20 pl-2">
          {isRunning ? (
            <>
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Spinner className="size-3.5" />
                <span>{isWritingCommand ? "Writing command…" : "Running…"}</span>
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
