// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { Spinner } from "@/components/ui/spinner";
import { authFetch } from "@/features/auth";

import { SandboxFiles } from "./sandbox-files-view";
import { isSandboxFileList, type SandboxFile } from "./sandbox-files";
import {
  preferSanitizedFullToolOutput,
  useChatRuntimeStore,
  useChatPreferencesStore,
  useToolAwaitingApproval,
  useToolOutputFor,
  useToolPaneScope,
} from "@/features/chat";
import { stringifyToolResult } from "@/lib/strip-ansi";
import type { ToolCallMessagePartComponent } from "@assistant-ui/react";
import { useToolArgsStatus } from "@assistant-ui/react";
import { CodeIcon } from "lucide-react";
import { memo, useEffect, useRef, useState } from "react";
import { pythonToolImagePath } from "./python-tool-image-path";
import { CopyBtn, ToolCodeCell } from "./tool-code-cell";
import { toolArgText } from "./tool-arg-text";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";
import { ToolLiveOutput } from "./tool-live-output";
import { ToolResultOutput } from "./tool-result-output";

interface StructuredResult {
  text: string;
  images: string[];
  sessionId: string;
  files?: SandboxFile[];
}

function isStructuredResult(val: unknown): val is StructuredResult {
  if (typeof val !== "object" || val === null) return false;
  const v = val as { files?: unknown };
  return (
    "text" in val &&
    "images" in val &&
    "sessionId" in val &&
    // Persisted content can carry anything, and the card maps over this and
    // reads name off each entry.
    isSandboxFileList(v.files)
  );
}

function PythonToolImage({
  sessionId,
  filename,
}: {
  sessionId: string;
  filename: string;
}) {
  const imageKey = `${sessionId}\0${filename}`;
  const [image, setImage] = useState<{ key: string; url: string } | null>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const [nearViewport, setNearViewport] = useState(
    () => typeof IntersectionObserver === "undefined",
  );

  useEffect(() => {
    if (nearViewport) {
      return;
    }
    const element = imageRef.current;
    if (!element) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setNearViewport(true);
          observer.disconnect();
        }
      },
      { rootMargin: "200px" },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, [nearViewport]);

  useEffect(() => {
    if (!nearViewport) {
      return;
    }
    const controller = new AbortController();
    let objectUrl: string | null = null;

    const load = async () => {
      const response = await authFetch(
        pythonToolImagePath(sessionId, filename),
        {
          signal: controller.signal,
        },
      );
      if (!response.ok) {
        return;
      }

      const blob = await response.blob();
      if (controller.signal.aborted) {
        return;
      }

      objectUrl = URL.createObjectURL(blob);
      setImage({ key: imageKey, url: objectUrl });
    };
    load().catch(() => {
      // A failed or cancelled image stays as its accessible alt text.
    });

    return () => {
      controller.abort();
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [filename, imageKey, nearViewport, sessionId]);

  return (
    <img
      ref={imageRef}
      src={image?.key === imageKey ? image.url : undefined}
      alt={filename}
      loading="lazy"
      className="max-w-full rounded border border-border"
    />
  );
}

const PythonToolUIImpl: ToolCallMessagePartComponent = ({
  toolCallId,
  args,
  result,
  status,
}) => {
  const code = toolArgText((args as { code?: unknown })?.code);
  const firstLine = code.split("\n")[0]?.slice(0, 60) ?? "";
  const isRunning = status?.type === "running";
  // Args still streaming = the model is WRITING the code, not running it yet.
  const { propStatus } = useToolArgsStatus();
  const isWritingCode = isRunning && propStatus.code === "streaming";

  let output: string;
  let images: string[] = [];
  let files: SandboxFile[] = [];
  let sessionId = "";

  if (isStructuredResult(result)) {
    output = result.text;
    images = result.images;
    files = result.files ?? [];
    sessionId = result.sessionId;
  } else if (result != null) {
    output = stringifyToolResult(result);
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
  const displayOutput = preferSanitizedFullToolOutput(fullOutput, output);

  // The gate only opens once the call parsed, so a pending approval means the script is
  // written even while the args status still reads as streaming.
  const awaitingApproval = useToolAwaitingApproval(toolCallId);
  const isWriting = isWritingCode && !awaitingApproval;
  const collapseByDefault = useChatPreferencesStore(
    (state) => state.collapseToolActivityByDefault,
  );
  const scriptCell = code ? (
    <div className="mt-1 pl-5">
      <ToolCodeCell
        label="script"
        code={code}
        language="python"
        downloadName="script.py"
        streaming={isWriting}
      />
    </div>
  ) : null;

  return (
    // Status, output and images collapse from history; the executed script
    // renders outside ToolFallbackContent so it stays visible on reopen
    // (#7165). Terminal keeps its command inside the collapsible -- a one-line
    // command is not the artifact a user comes back for, a script is.
    //
    // That #7165 guarantee is structural only while collapseToolActivity is
    // off. With it on the script moves inside the collapsible below, so a
    // reopened chat shows the row collapsed and the script (and its download
    // button) behind one click. That is the trade the preference buys, and
    // awaitingApproval is the one case it does not get to make: a decision
    // about a script has to be taken with the script on screen.
    <ToolFallbackRoot
      defaultOpen={isRunning}
      awaitingApproval={awaitingApproval}
    >
      <ToolFallbackTrigger
        toolName={firstLine ? `Python: ${firstLine}` : "Python"}
        status={status}
        icon={CodeIcon}
      />
      {!collapseByDefault && scriptCell}
      <ToolFallbackContent>
        {collapseByDefault && scriptCell}
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
                <span className="text-xs font-medium text-muted-foreground">
                  output
                </span>
                <CopyBtn text={displayOutput} />
              </div>
              <ToolResultOutput text={displayOutput} />
            </div>
          ) : null}

          {/* Anything the script wrote, as a real download */}
          <SandboxFiles sessionId={sessionId} files={files} />

          {/* Images from Python tool execution */}
          {images.length > 0 && sessionId && (
            <div className="mt-2 flex flex-col gap-2">
              {images.map((filename) => (
                <PythonToolImage
                  key={filename}
                  sessionId={sessionId}
                  filename={filename}
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
