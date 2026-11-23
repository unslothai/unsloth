// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { attachmentFileKind } from "@/features/chat";
import { RagDocumentTrigger } from "@/features/rag";
import { FileGlyph } from "@/lib/file-icon";
import { useMemo, useState } from "react";
import {
  type Citation,
  type Passage,
  documentKey,
  isPassage,
  orderedPassages,
  passageDisplayText,
  passageNeedsClamp,
} from "./citation-utils";

// Same ceiling as the document chips: a model-chosen top_k would otherwise mount every passage.
const PASSAGES_SHOWN = 20;

function PassageItem({
  passage,
  showFilename,
}: {
  passage: Passage;
  /** Off when the search drew on one document: its chip already names it. */
  showFilename: boolean;
}) {
  const { citation, number } = passage;
  const text = passageDisplayText(citation.text, citation.filename);
  const clampable = passageNeedsClamp(text);
  // Code and config read by their indentation and alignment, which a proportional font loses.
  const code = attachmentFileKind(citation.filename, undefined) === "code";
  const [expanded, setExpanded] = useState(false);
  const location = citation.page != null ? `p.${citation.page}` : null;

  return (
    <li
      data-slot="knowledge-base-passage"
      className="flex flex-col gap-1 rounded-md bg-muted/50 px-2.5 py-2"
    >
      <div className="flex min-w-0 items-center gap-2 text-xs">
        <span
          className="shrink-0 tabular-nums"
          title="Its number in the search result the model was given"
        >
          #{number}
        </span>
        {showFilename ? (
          <span className="flex min-w-0 items-center gap-1 font-medium text-foreground">
            <FileGlyph className="size-3 shrink-0" />
            <span className="truncate" title={citation.filename}>
              {citation.filename}
            </span>
          </span>
        ) : null}
        {citation.documentId ? (
          <RagDocumentTrigger
            documentId={citation.documentId}
            filename={citation.filename}
            page={citation.page ?? undefined}
          >
            <button
              type="button"
              className="shrink-0 underline-offset-2 hover:text-foreground hover:underline"
            >
              {location ? `Open at ${location}` : "Open document"}
            </button>
          </RagDocumentTrigger>
        ) : location ? (
          <span className="shrink-0">{location}</span>
        ) : null}
      </div>
      <p
        className={`whitespace-pre-wrap break-words text-xs leading-relaxed text-foreground/90 ${
          code ? "font-mono" : ""
        } ${clampable && !expanded ? "line-clamp-4" : ""}`}
      >
        {text}
      </p>
      {clampable ? (
        <button
          type="button"
          aria-expanded={expanded}
          onClick={() => setExpanded((v) => !v)}
          className="self-start text-xs underline-offset-2 hover:text-foreground hover:underline"
        >
          {expanded ? "Show less" : "Show more"}
        </button>
      ) : null}
    </li>
  );
}

export function KnowledgeBasePassages({
  citations,
}: {
  citations: readonly Citation[];
}) {
  const total = useMemo(() => citations.filter(isPassage).length, [citations]);
  const [showAll, setShowAll] = useState(false);
  const shown = useMemo(
    () =>
      orderedPassages(
        citations,
        showAll ? Number.POSITIVE_INFINITY : PASSAGES_SHOWN,
      ),
    [citations, showAll],
  );
  const showFilename = useMemo(
    () => new Set(citations.filter(isPassage).map(documentKey)).size > 1,
    [citations],
  );
  const shownCount = shown.length;

  return (
    <div data-slot="knowledge-base-passages" className="flex flex-col gap-1.5">
      <ol className="flex flex-col gap-1.5">
        {shown.map((passage) => (
          <PassageItem
            key={passage.citation.id}
            passage={passage}
            showFilename={showFilename}
          />
        ))}
      </ol>
      {shownCount < total ? (
        <button
          type="button"
          onClick={() => setShowAll(true)}
          className="self-start text-xs underline-offset-2 hover:text-foreground hover:underline"
        >
          Show {total - shownCount} more passage
          {total - shownCount === 1 ? "" : "s"}
        </button>
      ) : null}
    </div>
  );
}
