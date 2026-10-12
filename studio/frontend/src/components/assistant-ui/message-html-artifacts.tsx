// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

// Once an assistant message finishes, append one browser card per fenced ```html
// block in its text. No render_html tool call, no extra message. Defers to the
// other paths to avoid duplicates: skips the message when a render_html tool
// already rendered it, and skips full documents the in-place collapse handles.
// React components (```jsx / ```tsx) get a card after the HTML ones; their code
// stays in place, and nothing else renders them.

import {
  memoOnArray,
  partsHaveRenderableRenderHtmlTool,
} from "@/components/assistant-ui/message-derived";
import { ArtifactCard, useChatRuntimeStore } from "@/features/chat";
import {
  extractHtmlFences,
  extractReactFences,
} from "@/features/chat/artifacts/html-fences";
import { useAuiState } from "@assistant-ui/react";
import { type FC, useMemo } from "react";

// A char that cannot occur in chat text, used to keep text parts separate so a
// fence is never stitched across a non-text part (tool call, source, reasoning).
const PART_SEPARATOR = "\u0000";

const visibleTextBlob = memoOnArray(
  (content: ReadonlyArray<{ type: string; text?: unknown }>) =>
    content
      .filter((part) => part.type === "text" && "text" in part)
      .map((part) => (part as { text: string }).text)
      .join(PART_SEPARATOR),
);

export const MessageHtmlArtifacts: FC = () => {
  // Skip while streaming; "!== running" also covers loaded historical messages.
  const isRunning = useAuiState(
    ({ message }) => message.status?.type === "running",
  );
  const hasRenderHtmlTool = useAuiState(({ message }) =>
    partsHaveRenderableRenderHtmlTool(message.parts),
  );
  // Visible assistant text parts only (no reasoning, tools, sources, or errors),
  // kept separate so a fence stays within the part the user actually sees.
  const textBlob = useAuiState(({ message }) =>
    visibleTextBlob(message.content),
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
          // Skip only the plain fences the in-place collapse actually handles.
          !(fence.isFullDocument && fence.isPlainFence && collapsesFullDocs),
      );
  }, [isRunning, hasRenderHtmlTool, textBlob, collapsesFullDocs]);

  const reactFences = useMemo(
    () =>
      isRunning
        ? []
        : textBlob.split(PART_SEPARATOR).flatMap((part) => extractReactFences(part)),
    [isRunning, textBlob],
  );

  if (fences.length === 0 && reactFences.length === 0) {
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
      {reactFences.map((fence, i) => (
        <ArtifactCard
          key={`react-${i}`}
          kind="react"
          lang={fence.lang}
          code={fence.source}
          title={i === 0 ? "React preview" : `React preview ${i + 1}`}
          source="fence"
        />
      ))}
    </div>
  );
};
