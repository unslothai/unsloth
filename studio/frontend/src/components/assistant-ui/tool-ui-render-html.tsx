// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  ArtifactCard,
  useChatArtifactsStore,
  useSelectedChatArtifact,
} from "@/features/chat";
import { BrowserIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type ToolCallMessagePartComponent,
  useAuiState,
  useToolArgsStatus,
} from "@assistant-ui/react";
import { memo, useEffect } from "react";

// Context7 assistant-ui docs: tool UIs can read streaming args via
// useToolArgsStatus, so render_html does not need to wait for tool completion.
type RenderHtmlArgs = Record<string, unknown> & {
  code?: string;
  title?: string;
};

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
  const isRunning = status?.type === "running";
  const codeIsStreaming = propStatus.code === "streaming";
  const isGeneratingArtifact = isRunning || codeIsStreaming;

  // Surface the backend error when the tool call completed with invalid
  // args.  Backend success results start with "Rendered HTML artifact";
  // error results start with "Error:".
  const errorText =
    status?.type === "complete" &&
    typeof result === "string" &&
    result.startsWith("Error:")
      ? result
      : null;
  const messageId = useAuiState(({ message }) => message.id) ?? null;
  const selectedArtifact = useSelectedChatArtifact();
  const closeArtifactSurface = useChatArtifactsStore(
    (state) => state.closeArtifactSurface,
  );

  useEffect(() => {
    if (!errorText) return;
    if (!messageId || !toolCallId) return;
    if (selectedArtifact?.sourceToolCallId !== toolCallId) return;
    if (selectedArtifact?.sourceMessageId !== messageId) return;
    closeArtifactSurface();
  }, [
    closeArtifactSurface,
    errorText,
    messageId,
    selectedArtifact,
    toolCallId,
  ]);

  if (hasCode || (isGeneratingArtifact && !errorText)) {
    return (
      <ArtifactCard
        code={code}
        title={title}
        source="tool"
        sourceToolCallId={toolCallId}
        autoOpen={true}
        isStreaming={isGeneratingArtifact}
      />
    );
  }

  return (
    <div className="relative my-2 flex min-h-[52px] w-full max-w-md items-center overflow-hidden rounded-lg border border-border/70 bg-muted/15 px-3 py-2 text-left dark:bg-muted/10">
      <div className="relative z-10 flex min-w-0 flex-1 items-center gap-2.5">
        <HugeiconsIcon
          icon={BrowserIcon}
          strokeWidth={1.75}
          className="size-5 shrink-0 text-muted-foreground"
        />
        <span className="grid min-w-0 flex-1 gap-1">
          <span className="truncate text-sm font-medium leading-tight text-foreground">
            {errorText ? "Artifact error" : "Generating artifact"}
          </span>
          <span className="truncate text-[11px] leading-none text-muted-foreground">
            {errorText ?? "HTML artifact"}
          </span>
        </span>
        {errorText ? null : (
          <span className="shimmer shrink-0 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary motion-reduce:animate-none">
            Generating
          </span>
        )}
      </div>
    </div>
  );
};

export const RenderHtmlToolUI = memo(
  RenderHtmlToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
RenderHtmlToolUI.displayName = "RenderHtmlToolUI";
