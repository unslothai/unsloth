// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { ArtifactCard } from "@/features/chat";
import {
  type ToolCallMessagePartComponent,
  useToolArgsStatus,
} from "@assistant-ui/react";
import { memo } from "react";

// Context7 assistant-ui docs: tool UIs can read streaming args via
// useToolArgsStatus, so render_html does not need to wait for tool completion.
type RenderHtmlArgs = Record<string, unknown> & {
  code?: string;
  title?: string;
};

function formatToolResult(result: unknown): string {
  if (typeof result === "string") return result;
  if (result == null) return "";
  try {
    return JSON.stringify(result, null, 2);
  } catch {
    return String(result);
  }
}

const RenderHtmlToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  result,
  status,
  toolCallId,
}) => {
  const { propStatus } = useToolArgsStatus<RenderHtmlArgs>();
  const parsedArgs = (args as RenderHtmlArgs) ?? {};
  const code = typeof parsedArgs.code === "string" ? parsedArgs.code : "";
  const hasCode = code.trim().length > 0;
  const title =
    typeof parsedArgs.title === "string" ? parsedArgs.title : "HTML artifact";
  const resultText = formatToolResult(result);
  const isRunning = status?.type === "running";
  const codeIsStreaming = propStatus.code === "streaming";

  if (hasCode) {
    return (
      <ArtifactCard
        code={code}
        title={title}
        source="tool"
        sourceToolCallId={toolCallId}
        autoOpen={true}
        isStreaming={isRunning || codeIsStreaming}
      />
    );
  }

  return (
    <div className="my-3 overflow-hidden rounded-xl border border-border/80 bg-background/80 px-3 py-2.5 shadow-sm shadow-black/5 dark:bg-muted/10 dark:shadow-black/20">
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold text-foreground">
            Generating artifact…
          </p>
          <p className="mt-0.5 truncate text-xs text-muted-foreground">
            {resultText || "Waiting for HTML source"}
          </p>
        </div>
        <span className="shrink-0 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
          Generating
        </span>
      </div>
      <div
        className="mt-2 h-1.5 overflow-hidden rounded-full bg-muted"
        aria-hidden={true}
      >
        <div className="h-full w-1/2 rounded-full bg-primary/25 shimmer motion-reduce:animate-none" />
      </div>
    </div>
  );
};

export const RenderHtmlToolUI = memo(
  RenderHtmlToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
RenderHtmlToolUI.displayName = "RenderHtmlToolUI";
