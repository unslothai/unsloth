// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import { useAuiState } from "@assistant-ui/react";
import {
  CheckIcon,
  CopyIcon,
  DownloadIcon,
  ExternalLinkIcon,
  FileTextIcon,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import { ArtifactHtmlFrame } from "./html-frame";
import { useChatArtifactsStore } from "./store";
import {
  type ChatArtifact,
  type ChatArtifactSource,
  createChatArtifact,
  getArtifactFilename,
} from "./types";

const COPY_RESET_MS = 2000;

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

export function ArtifactCard({
  code,
  title,
  source,
  sourceToolCallId,
  sourceMessageId,
  className,
  preview = true,
}: {
  code: string;
  title?: string | null;
  source: ChatArtifactSource;
  sourceToolCallId?: string | null;
  sourceMessageId?: string | null;
  className?: string;
  preview?: boolean;
}) {
  const activeThreadId = useChatRuntimeStore((state) => state.activeThreadId);
  const messageIdFromContext = useAuiState(({ message }) => message.id);
  const threadIdFromContext = useAuiState(
    ({ threads }) => threads.mainThreadId,
  );
  const artifactThreadId = threadIdFromContext ?? activeThreadId ?? null;
  const openArtifact = useChatArtifactsStore((state) => state.openArtifact);
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
      }),
    [
      artifactThreadId,
      code,
      messageIdFromContext,
      source,
      sourceMessageId,
      sourceToolCallId,
      title,
    ],
  );
  const filename = getArtifactFilename(artifact);
  const surface = artifactThreadId ? "panel" : "overlay";

  return (
    <div
      className={cn(
        "my-3 overflow-hidden rounded-xl border border-border bg-card/70 shadow-sm",
        className,
      )}
    >
      <div className="flex items-center gap-3 border-b border-border/70 px-3 py-2">
        <div className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
          <FileTextIcon className="size-4" />
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium text-foreground">
            {artifact.title}
          </p>
          <p className="text-xs text-muted-foreground">
            HTML artifact ·{" "}
            {source === "tool" ? "tool call" : "fenced fallback"}
          </p>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          <button
            type="button"
            className="flex size-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            title="Open artifact"
            aria-label="Open artifact"
            onClick={() => openArtifact(artifact, { surface })}
          >
            <ExternalLinkIcon className="size-4" />
          </button>
          <button
            type="button"
            className="flex size-8 items-center justify-center rounded-lg text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
            title="Copy HTML"
            aria-label="Copy artifact HTML"
            onClick={async () => {
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
            onClick={() => downloadTextFile(filename, artifact.code)}
          >
            <DownloadIcon className="size-4" />
          </button>
        </div>
      </div>
      {preview ? (
        <div className="max-h-[320px] overflow-auto bg-background">
          <ArtifactHtmlFrame code={artifact.code} title={artifact.title} />
        </div>
      ) : null}
    </div>
  );
}
