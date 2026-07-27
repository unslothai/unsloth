// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { CodeToggleIcon } from "@/components/assistant-ui/code-toggle-icon";
import { cn } from "@/lib/utils";
import { useAuiState } from "@assistant-ui/react";
import { LayoutTwoColumnIcon as Layout2ColumnIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useLayoutEffect, useMemo, useRef } from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ArtifactViewMode } from "./html-frame";
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

const CARD_BASE =
  "group/artifact-card relative flex min-h-[52px] cursor-pointer items-center overflow-hidden rounded-lg border border-border/70 bg-muted/15 px-3 py-2 text-left transition-colors hover:bg-muted/25 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring dark:bg-muted/10 dark:hover:bg-muted/20";

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
  // Canvas mode collapses the raw code in place, so offer a Code button too.
  // Diffusion keeps its code inline, so it needs no Code button.
  const showCodeButton = useChatRuntimeStore(
    (state) => state.artifactsEnabled && !state.loadedIsDiffusion,
  );
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
  // Once per mount, so a view-change cleanup can't re-trigger a stale open.
  const autoOpenAttemptedRef = useRef(false);

  useLayoutEffect(() => {
    if (selectedArtifactId === artifact.id) {
      updateArtifact(artifact);
    }

    if (!autoOpen || autoOpenAttemptedRef.current) {
      return;
    }
    autoOpenAttemptedRef.current = true;
    if (hasAutoOpenedArtifact(artifact.id)) {
      return;
    }

    rememberAutoOpenedArtifact(artifact.id);
    openArtifact(artifact, { surface, view: "preview" });
  }, [
    artifact,
    autoOpen,
    openArtifact,
    selectedArtifactId,
    surface,
    updateArtifact,
  ]);

  const renderButton = (view: ArtifactViewMode) => {
    const isCode = view === "source";
    return (
      <button
        key={view}
        type="button"
        className={cn(
          CARD_BASE,
          showCodeButton ? "min-w-0 flex-1" : "w-full max-w-md",
          isStreaming &&
            "border-border/80 bg-muted/20 dark:border-border/70 dark:bg-muted/15",
        )}
        onClick={() => openArtifact(artifact, { surface, view })}
        aria-label={`Open ${artifact.title} ${isCode ? "code" : "preview"}`}
      >
        {isStreaming ? (
          <span
            aria-hidden={true}
            className="artifact-card-shimmer pointer-events-none absolute inset-0 z-0 motion-reduce:hidden"
          />
        ) : null}
        <div className="relative z-10 flex min-w-0 flex-1 items-center gap-2.5">
          {isCode ? (
            <CodeToggleIcon className="size-5 shrink-0 text-muted-foreground" />
          ) : (
            <HugeiconsIcon
              icon={Layout2ColumnIcon}
              strokeWidth={1.75}
              className="size-5 shrink-0 text-muted-foreground"
            />
          )}
          <span className="grid min-w-0 flex-1 gap-1">
            <span className="truncate text-sm font-medium leading-tight text-foreground">
              {isCode ? "HTML Code" : artifact.title}
            </span>
            <span className="truncate text-ui-11 leading-none text-muted-foreground">
              HTML canvas
            </span>
          </span>
          {isStreaming && !isCode ? (
            <span className="shimmer shrink-0 rounded-full bg-primary/10 px-2 py-0.5 text-ui-10 font-medium text-primary motion-reduce:animate-none">
              Generating
            </span>
          ) : null}
        </div>
      </button>
    );
  };

  if (!showCodeButton) {
    return <div className={cn("my-2", className)}>{renderButton("preview")}</div>;
  }

  return (
    <div className={cn("my-2 flex w-full max-w-xl gap-2", className)}>
      {renderButton("preview")}
      {renderButton("source")}
    </div>
  );
}
