// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { cn } from "@/lib/utils";
import { LayoutTwoColumnIcon as Layout2ColumnIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useAuiState } from "@assistant-ui/react";
import { useLayoutEffect, useMemo } from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import {
  hasAutoOpenedArtifact,
  rememberAutoOpenedArtifact,
  useChatArtifactsStore,
} from "./store";
import {
  type ChatArtifact,
  type ChatArtifactSource,
  createChatArtifact,
} from "./types";

export function ArtifactCard({
  code,
  title,
  source,
  sourceToolCallId,
  sourceMessageId,
  className,
  autoOpen = false,
  isStreaming = false,
}: {
  code: string;
  title?: string | null;
  source: ChatArtifactSource;
  sourceToolCallId?: string | null;
  sourceMessageId?: string | null;
  className?: string;
  autoOpen?: boolean;
  isStreaming?: boolean;
}) {
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const messageIdFromContext = useAuiState(({ message }) => message.id);
  const threadIdFromContext = useAuiState(
    ({ threads }) => threads.mainThreadId,
  );
  const artifactThreadId = threadIdFromContext ?? activeThreadId ?? null;
  const openArtifact = useChatArtifactsStore((state) => state.openArtifact);
  const updateArtifact = useChatArtifactsStore((state) => state.updateArtifact);
  const selectedArtifactId = useChatArtifactsStore(
    (state) => state.selectedArtifactId,
  );
  const artifact = useMemo<ChatArtifact>(
    () =>
      createChatArtifact({
        code,
        title,
        source,
        sourceMessageId: sourceMessageId ?? messageIdFromContext ?? null,
        sourceToolCallId: sourceToolCallId ?? null,
        threadId: artifactThreadId,
        isStreaming,
      }),
    [
      artifactThreadId,
      code,
      isStreaming,
      messageIdFromContext,
      source,
      sourceMessageId,
      sourceToolCallId,
      title,
    ],
  );
  const surface = artifactThreadId ? "panel" : "overlay";

  useLayoutEffect(() => {
    if (!autoOpen) return;
    if (!hasAutoOpenedArtifact(artifact.id)) {
      rememberAutoOpenedArtifact(artifact.id);
      openArtifact(artifact, { surface });
      return;
    }
    if (selectedArtifactId === artifact.id) {
      updateArtifact(artifact);
    }
  }, [
    artifact,
    autoOpen,
    openArtifact,
    selectedArtifactId,
    surface,
    updateArtifact,
  ]);

  return (
    <button
      type="button"
      className={cn(
        "group/artifact-card relative my-2 flex min-h-[52px] w-full max-w-md cursor-pointer items-center overflow-hidden rounded-lg border border-border/70 bg-muted/15 px-3 py-2 text-left transition-colors hover:bg-muted/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
        "dark:bg-muted/10 dark:hover:bg-muted/20",
        isStreaming &&
          "border-border/80 bg-muted/20 dark:border-border/70 dark:bg-muted/15",
        className,
      )}
      onClick={() => openArtifact(artifact, { surface })}
      aria-label={`Open ${artifact.title}`}
    >
      {isStreaming ? (
        <span
          aria-hidden={true}
          className="artifact-card-shimmer pointer-events-none absolute inset-0 z-0 motion-reduce:hidden"
        />
      ) : null}
      <div className="relative z-10 flex min-w-0 flex-1 items-center gap-2.5">
        <HugeiconsIcon
          icon={Layout2ColumnIcon}
          strokeWidth={1.75}
          className="size-5 shrink-0 text-muted-foreground"
        />
        <span className="grid min-w-0 flex-1 gap-1">
          <span className="truncate text-sm font-medium leading-tight text-foreground">
            {artifact.title}
          </span>
          <span className="truncate text-[11px] leading-none text-muted-foreground">
            HTML artifact
          </span>
        </span>
        {isStreaming ? (
          <span className="shimmer shrink-0 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary motion-reduce:animate-none">
            Generating
          </span>
        ) : null}
      </div>
    </button>
  );
}
