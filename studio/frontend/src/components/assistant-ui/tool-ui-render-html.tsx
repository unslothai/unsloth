// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  ArtifactCard,
  useChatArtifactsStore,
  useSelectedChatArtifact,
} from "@/features/chat";
import {
  type ToolCallMessagePartComponent,
  useAuiState,
  useToolArgsStatus,
} from "@assistant-ui/react";
import { BrowserIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { memo, useEffect } from "react";

// Per Context7 assistant-ui docs: tool UIs read streaming args via
// useToolArgsStatus, so render_html need not wait for tool completion.
type RenderHtmlArgs = Record<string, unknown> & {
  code?: string;
  title?: string;
};

const RENDER_HTML_SESSION_STARTED_AT = Date.now();

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
    typeof parsedArgs.title === "string" ? parsedArgs.title : "HTML canvas";
  const isRunning = status?.type === "running";
  const codeIsStreaming = propStatus.code === "streaming";

  // Surface the backend error when the tool call completed with invalid args.
  // Success results start with "Rendered HTML canvas"; errors with "Error:".
  const errorText =
    status?.type === "complete" &&
    typeof result === "string" &&
    result.startsWith("Error:")
      ? result
      : null;
  const messageId = useAuiState(({ message }) => message.id) ?? null;
  const isMessageRunning = useAuiState(
    ({ message }) => message.status?.type === "running",
  );
  const messageCreatedAtMs = useAuiState(({ message }) =>
    message.createdAt instanceof Date ? message.createdAt.getTime() : null,
  );
  const isThreadRunning = useAuiState(({ thread }) => thread.isRunning);
  const isLiveGeneratingArtifact =
    isThreadRunning && isMessageRunning && (isRunning || codeIsStreaming);
  const isStaleGeneratingArtifact =
    !(isThreadRunning && isMessageRunning) && (isRunning || codeIsStreaming);
  const messageCreatedThisSession =
    messageCreatedAtMs != null &&
    messageCreatedAtMs >= RENDER_HTML_SESSION_STARTED_AT - 1000;
  const shouldAutoOpenArtifact =
    (isLiveGeneratingArtifact && (hasCode || isRunning || codeIsStreaming)) ||
    (hasCode && messageCreatedThisSession);
  const selectedArtifact = useSelectedChatArtifact();
  const closeArtifactSurface = useChatArtifactsStore(
    (state) => state.closeArtifactSurface,
  );

  useEffect(() => {
    if (!errorText) {
      return;
    }
    if (!(messageId && toolCallId)) {
      return;
    }
    if (selectedArtifact?.sourceToolCallId !== toolCallId) {
      return;
    }
    if (selectedArtifact?.sourceMessageId !== messageId) {
      return;
    }
    closeArtifactSurface();
  }, [
    closeArtifactSurface,
    errorText,
    messageId,
    selectedArtifact,
    toolCallId,
  ]);

  if (hasCode || (isLiveGeneratingArtifact && !errorText)) {
    return (
      <ArtifactCard
        code={code}
        title={title}
        source="tool"
        sourceToolCallId={toolCallId}
        autoOpen={!errorText && shouldAutoOpenArtifact}
        isStreaming={!errorText && isLiveGeneratingArtifact}
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
            {errorText
              ? "Canvas error"
              : isStaleGeneratingArtifact
                ? "Canvas interrupted"
                : "Canvas unavailable"}
          </span>
          <span className="truncate text-[0.6875rem] leading-none text-muted-foreground">
            {errorText ??
              (isStaleGeneratingArtifact
                ? "Refresh stopped this preview"
                : "HTML canvas")}
          </span>
        </span>
      </div>
    </div>
  );
};

export const RenderHtmlToolUI = memo(
  RenderHtmlToolUIImpl,
) as unknown as ToolCallMessagePartComponent;
RenderHtmlToolUI.displayName = "RenderHtmlToolUI";
