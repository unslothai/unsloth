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

const RenderHtmlToolUIImpl: ToolCallMessagePartComponent = ({
  args,
  status,
  toolCallId,
}) => {
  const parsedArgs = (args as RenderHtmlArgs) ?? {};
  const code = typeof parsedArgs.code === "string" ? parsedArgs.code : "";
  const title =
    typeof parsedArgs.title === "string" ? parsedArgs.title : "HTML artifact";
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
    const nextOpen = isRunning || code ? true : hasText ? false : null;
    if (nextOpen == null) return;
    const timeoutId = window.setTimeout(() => setOpen(nextOpen), 0);
    return () => window.clearTimeout(timeoutId);
  }, [code, hasText, isRunning]);

  return (
    <ToolFallbackRoot open={open} onOpenChange={setOpen}>
      <ToolFallbackTrigger
        toolName={isRunning ? "Rendering HTML artifact…" : title}
        status={status}
        icon={FileTextIcon}
      />
      <ToolFallbackContent>
        {isRunning && !code ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <LoaderIcon className="size-3.5 animate-spin" />
            <span>Waiting for artifact source…</span>
          </div>
        ) : code ? (
          <ArtifactCard
            code={code}
            title={title}
            source="tool"
            sourceToolCallId={toolCallId}
            preview={false}
          />
        ) : (
          <p className="text-sm text-muted-foreground">
            No HTML artifact source was provided.
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
