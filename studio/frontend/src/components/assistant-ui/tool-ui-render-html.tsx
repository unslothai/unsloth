// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { ArtifactCard } from "@/features/chat";
import {
  type ToolCallMessagePartComponent,
  useAuiState,
} from "@assistant-ui/react";
import { FileTextIcon, LoaderIcon } from "lucide-react";
import { memo, useEffect, useState } from "react";
import {
  ToolFallbackContent,
  ToolFallbackRoot,
  ToolFallbackTrigger,
} from "./tool-fallback";

interface RenderHtmlArgs {
  code?: string;
  title?: string;
}

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
  const parsedArgs = (args as RenderHtmlArgs) ?? {};
  const code = typeof parsedArgs.code === "string" ? parsedArgs.code : "";
  const hasCode = code.trim().length > 0;
  const title =
    typeof parsedArgs.title === "string" ? parsedArgs.title : "HTML artifact";
  const resultText = formatToolResult(result);
  const isRunning = status?.type === "running";
  const hasText = useAuiState(({ message }) =>
    message.content.some(
      (part) =>
        part.type === "text" &&
        "text" in part &&
        (part as { text: string }).text.length > 0,
    ),
  );
  const [open, setOpen] = useState(true);

  useEffect(() => {
    const nextOpen = isRunning ? true : !hasText;
    const timeoutId = window.setTimeout(() => setOpen(nextOpen), 0);
    return () => window.clearTimeout(timeoutId);
  }, [hasText, isRunning]);

  return (
    <ToolFallbackRoot open={open} onOpenChange={setOpen}>
      <ToolFallbackTrigger
        toolName={isRunning ? "Rendering HTML artifact…" : title}
        status={status}
        icon={FileTextIcon}
      />
      <ToolFallbackContent>
        {isRunning && !hasCode ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <LoaderIcon className="size-3.5 animate-spin" />
            <span>Waiting for artifact source…</span>
          </div>
        ) : hasCode ? (
          <ArtifactCard
            code={code}
            title={title}
            source="tool"
            sourceToolCallId={toolCallId}
            preview={false}
          />
        ) : (
          <p className="whitespace-pre-wrap text-sm text-muted-foreground">
            {resultText || "No HTML artifact source was provided."}
          </p>
        )}
      </ToolFallbackContent>
    </ToolFallbackRoot>
  );
};

export const RenderHtmlToolUI = memo(
  RenderHtmlToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
RenderHtmlToolUI.displayName = "RenderHtmlToolUI";
