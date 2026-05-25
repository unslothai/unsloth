// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import { useAuiState } from "@assistant-ui/react";
import { CheckIcon, CopyIcon, DownloadIcon } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { useChatArtifactsStore } from "./store";
import {
  type ChatArtifact,
  type ChatArtifactSource,
  createChatArtifact,
  getArtifactFilename,
} from "./types";

const COPY_RESET_MS = 2000;
const autoOpenedArtifactIds = new Set<string>();

function downloadTextFile(filename: string, text: string): void {
  const blob = new Blob([text], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}

function useCopiedState() {
  const [copied, setCopied] = useState(false);
  const resetTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    return () => {
      if (resetTimeoutRef.current) clearTimeout(resetTimeoutRef.current);
    };
  }, []);

  const showCopied = () => {
    setCopied(true);
    if (resetTimeoutRef.current) clearTimeout(resetTimeoutRef.current);
    resetTimeoutRef.current = setTimeout(() => {
      setCopied(false);
      resetTimeoutRef.current = null;
    }, COPY_RESET_MS);
  };

  return { copied, showCopied };
}

function artifactDisplayTitle(title: string): string {
  return /artifact/i.test(title) ? title : `${title} Artifact`;
}

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
  preview?: boolean;
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
  const { copied, showCopied } = useCopiedState();
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
  const filename = getArtifactFilename(artifact);
  const surface = artifactThreadId ? "panel" : "overlay";
  const lineCount = artifact.code.split("\n").length;
  const displayTitle = artifactDisplayTitle(artifact.title);

  useEffect(() => {
    if (!autoOpen) return;
    if (!autoOpenedArtifactIds.has(artifact.id)) {
      autoOpenedArtifactIds.add(artifact.id);
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
    <div
      className={cn(
        "my-3 cursor-pointer overflow-hidden rounded-xl border border-border/80 bg-background/80 shadow-sm shadow-black/5 transition-colors hover:bg-muted/30",
        "dark:bg-muted/10 dark:shadow-black/20 dark:hover:bg-muted/20",
        className,
      )}
      onClick={() => openArtifact(artifact, { surface })}
    >
      <div className="flex items-center gap-3 px-3 py-2.5">
        <button
          type="button"
          className="min-w-0 flex-1 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          onClick={(event) => {
            event.stopPropagation();
            openArtifact(artifact, { surface });
          }}
          aria-label={`Open ${displayTitle}`}
        >
          <div className="flex min-w-0 items-center gap-2">
            <p className="truncate text-sm font-semibold text-foreground">
              {displayTitle}
            </p>
            {isStreaming ? (
              <span className="shrink-0 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                Generating
              </span>
            ) : null}
          </div>
          <p className="mt-0.5 text-xs text-muted-foreground">
            HTML artifact ·{" "}
            {source === "tool" ? "tool call" : "fenced fallback"} · {lineCount}{" "}
            lines
          </p>
        </button>
        <div className="flex shrink-0 items-center gap-1">
          <button
            type="button"
            className="flex size-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            title="Copy HTML"
            aria-label="Copy artifact HTML"
            onClick={async (event) => {
              event.stopPropagation();
              if (await copyToClipboard(artifact.code)) showCopied();
            }}
          >
            {copied ? (
              <CheckIcon className="size-4" />
            ) : (
              <CopyIcon className="size-4" />
            )}
          </button>
          <button
            type="button"
            className="flex size-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            title="Download HTML"
            aria-label="Download artifact HTML"
            onClick={(event) => {
              event.stopPropagation();
              downloadTextFile(filename, artifact.code);
            }}
          >
            <DownloadIcon className="size-4" />
          </button>
        </div>
      </div>
    </div>
  );
}
