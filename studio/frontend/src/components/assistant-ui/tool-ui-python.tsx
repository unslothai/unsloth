// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { getAuthToken } from "@/features/auth/session";
import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { useToolArgsStatus } from "@assistant-ui/react";
import { code as codePlugin } from "@streamdown/code";
import { CodeIcon, CopyIcon, DownloadIcon } from "lucide-react";
import { Tick02Icon } from "@/lib/tick-icon";
import { HugeiconsIcon } from "@hugeicons/react";
import { Spinner } from "@/components/ui/spinner";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Streamdown } from "streamdown";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { ToolLiveOutput } from "./tool-live-output";
import { ToolResultOutput } from "./tool-result-output";
import { useChatRuntimeStore } from "@/features/chat/stores/chat-runtime-store";
import {
  preferFullToolOutput,
  toolOutputKey,
  useToolPaneScope,
} from "@/features/chat";

interface StructuredResult {
  text: string;
  images: string[];
  sessionId: string;
}

const MAX_DISPLAY = 10_000;
const COPY_RESET_MS = 2000;
const SHIKI_THEME = ["github-light", "github-dark"] as ["github-light", "github-dark"];

function truncate(text: string): string {
  return text.length <= MAX_DISPLAY
    ? text
    : `${text.slice(0, MAX_DISPLAY)}\n... (truncated)`;
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

/** Save the script as a .py file via a client-side Blob. */
function DownloadBtn({ code, name = "script.py" }: { code: string; name?: string }) {
  const download = useCallback(() => {
    if (typeof document === "undefined") {
      return;
    }
    try {
      const blob = new Blob([code], { type: "text/x-python" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = name;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      // Revoke next tick, after the click consumes the URL.
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch {
      // Best-effort: never break the transcript over a download.
    }
  }, [code, name]);

  return (
    <button
      type="button"
      onClick={download}
      className="inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
      aria-label="Download script"
    >
      <DownloadIcon className="size-3" />
      Download
    </button>
  );
}

/** Syntax-highlighted code via Streamdown + shiki; inherits parent container. */
function HighlightedCode({ code: source, language }: { code: string; language: string }) {
  const markdown = useMemo(
    () => `\`\`\`${language}\n${truncate(source)}\n\`\`\``,
    [source, language],
  );
  return (
    <div className="max-h-48 overflow-auto text-xs [&_pre]:!m-0 [&_pre]:!bg-transparent [&_pre]:!p-0 [&_pre]:!text-xs [&_[data-streamdown=code-block]]:!my-0 [&_[data-streamdown=code-block]]:!p-3 [&_[data-streamdown=code-block]]:!border-0">
      <Streamdown
        mode="static"
        plugins={{ code: codePlugin }}
        controls={{ code: false }}
        shikiTheme={SHIKI_THEME}
      >
        {markdown}
      </Streamdown>
    </div>
  );
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
  const fullOutput = useChatRuntimeStore(
    (s) => s.toolFullOutput[toolOutputKey(paneScope, toolCallId)] ?? "",
  );
  const displayOutput = preferFullToolOutput(fullOutput, output);

  const authToken = getAuthToken();

  return (
    // Status/output collapse from history; the script source renders outside
    // ToolFallbackContent so it stays visible on reopen (#7165).
    <ToolFallbackRoot defaultOpen={isRunning}>
      <ToolFallbackTrigger
        toolName={firstLine ? `Python: ${firstLine}` : "Python"}
        status={status}
        icon={CodeIcon}
      />
      {code && (
        <div className="mt-1 pl-5">
          <div className="border-l-2 border-muted-foreground/20 pl-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">
                script
              </span>
              <div className="flex items-center gap-1">
                <CopyBtn text={code} />
                <DownloadBtn code={code} />
              </div>
            </div>
            <HighlightedCode code={code} language="python" />
          </div>
        </div>
      )}
      <ToolFallbackContent>
        <div className="border-l-2 border-muted-foreground/20 pl-2">
          {/* Output */}
          {isRunning ? (
            <>
              <div className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
                <Spinner className="size-3.5" />
                <span>{isWritingCode ? "Writing code…" : "Running…"}</span>
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
