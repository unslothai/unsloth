// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

// Once an assistant message finishes, append one browser card per fenced ```html
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

// Cannot occur in chat text; keeps a fence from being stitched across non-text parts.
const PART_SEPARATOR = "\u0000";

export const MessageHtmlArtifacts: FC = () => {
  const isRunning = useAuiState(
    ({ message }) => message.status?.type === "running",
  );
  const hasRenderHtmlTool = useAuiState(({ message }) =>
    message.parts.some(isRenderableRenderHtmlToolPart),
  );
  const textBlob = useAuiState(({ message }) =>
    message.content
      .filter((part) => part.type === "text" && "text" in part)
      .map((part) => (part as { text: string }).text)
      .join(PART_SEPARATOR),
  );
  const collapseHtmlArtifacts = useChatRuntimeStore(
    (state) => state.collapseHtmlArtifacts,
  );
  const loadedIsDiffusion = useChatRuntimeStore(
    (state) => state.loadedIsDiffusion,
  );
  // Full docs already shown by the in-place collapse; excluded for diffusion,
  // which keeps its code inline instead. This picks which path renders a full
  // document, not whether fenced HTML renders at all.
  const collapsesFullDocs = collapseHtmlArtifacts && !loadedIsDiffusion;

  const fences = useMemo(() => {
    if (isRunning || hasRenderHtmlTool) {
      return [];
    }
    return textBlob
      .split(PART_SEPARATOR)
      .flatMap((part) => extractHtmlFences(part))
      .filter(
        (fence) =>
          !(fence.isFullDocument && fence.isPlainFence && collapsesFullDocs),
      );
  }, [isRunning, hasRenderHtmlTool, textBlob, collapsesFullDocs]);

  if (fences.length === 0) {
    return null;
  }

  return (
    <div className="mt-2 flex flex-col gap-2">
      {fences.map((fence, i) => (
        <ArtifactCard
          key={i}
          code={fence.source}
          title={i === 0 ? "HTML preview" : `HTML preview ${i + 1}`}
          source="fence"
        />
      ))}
    </div>
  );
};
