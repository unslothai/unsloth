// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

// Once an assistant message finishes, append one canvas card per fenced ```html
// block in its text. No render_html tool call, no extra message. Defers to the
// other paths to avoid duplicates: skips the message when a render_html tool
// already rendered it, and skips full documents the in-place collapse handles.

import { ArtifactCard, useChatRuntimeStore } from "@/features/chat";
import {
  extractHtmlFences,
  isRenderableRenderHtmlToolPart,
} from "@/features/chat/artifacts/html-fences";
import { useAuiState } from "@assistant-ui/react";
import { type FC, useMemo } from "react";

export const MessageHtmlArtifacts: FC = () => {
  // Skip while streaming; "!== running" also covers loaded historical messages.
  const isRunning = useAuiState(
    ({ message }) => message.status?.type === "running",
  );
  const hasRenderHtmlTool = useAuiState(({ message }) =>
    message.parts.some(isRenderableRenderHtmlToolPart),
  );
  // Visible assistant text only (no reasoning, tools, sources, or errors).
  const fullText = useAuiState(({ message }) =>
    message.content
      .filter((part) => part.type === "text" && "text" in part)
      .map((part) => (part as { text: string }).text)
      .join(""),
  );
  // Same collapse gate as markdown-text.tsx: these full docs already render as a
  // card in place, so skip them here. Diffusion is excluded, so its full docs get
  // a trailing card while the raw code stays visible.
  const collapsesFullDocs = useChatRuntimeStore(
    (state) => state.artifactsEnabled || state.collapseHtmlArtifacts,
  );

  const fences = useMemo(() => {
    if (isRunning || hasRenderHtmlTool) {
      return [];
    }
    return extractHtmlFences(fullText).filter(
      (fence) => !(fence.isFullDocument && collapsesFullDocs),
    );
  }, [isRunning, hasRenderHtmlTool, fullText, collapsesFullDocs]);

  if (fences.length === 0) {
    return null;
  }

  return (
    <div className="mt-2 flex flex-col gap-2">
      {fences.map((fence, i) => (
        <ArtifactCard
          key={fence.index}
          code={fence.source}
          title={i === 0 ? "HTML preview" : `HTML preview ${i + 1}`}
          source="fence"
        />
      ))}
    </div>
  );
};
