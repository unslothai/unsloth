// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { RagDocumentTrigger } from "@/features/rag";
import { FileGlyph } from "@/lib/file-icon";
import { useMemo, useState } from "react";
import {
  type Citation,
  type Passage,
  firstPassages,
  passageDisplayText,
  passageNeedsClamp,
  passagesPerDocument,
} from "./citation-utils";

// Same ceiling as the document chips: a model-chosen top_k would otherwise mount every passage.
const PASSAGES_SHOWN = 20;

function PassageItem({ passage }: { passage: Passage }) {
  const { citation, rank } = passage;
  const text = passageDisplayText(citation.text, citation.filename);
  const clampable = passageNeedsClamp(text);
  const [expanded, setExpanded] = useState(false);
  const location = citation.page != null ? `p.${citation.page}` : null;

  return (
    <li
      data-slot="knowledge-base-passage"
      className="flex flex-col gap-1 rounded-md bg-muted/50 px-2.5 py-2"
    >
      <div className="flex items-center gap-2 text-xs">
        {rank != null ? (
          <span className="tabular-nums" title="Position in the search results">
            #{rank}
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
              className="underline-offset-2 hover:text-foreground hover:underline"
            >
              {location ? `Open at ${location}` : "Open document"}
            </button>
          </RagDocumentTrigger>
        ) : location ? (
          <span>{location}</span>
        ) : null}
      </div>
      <p
        className={`whitespace-pre-wrap break-words text-xs leading-relaxed text-foreground/90 ${
          clampable && !expanded ? "line-clamp-4" : ""
        }`}
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

/** The retrieved passages, each under its document once, with its rank and where it opens. */
export function KnowledgeBasePassages({
  citations,
}: {
  citations: readonly Citation[];
}) {
  const groups = useMemo(() => passagesPerDocument(citations), [citations]);
  const total = groups.reduce((n, g) => n + g.passages.length, 0);
  const [showAll, setShowAll] = useState(false);

  const shown = showAll ? groups : firstPassages(groups, PASSAGES_SHOWN);
  const shownCount = shown.reduce((n, g) => n + g.passages.length, 0);

  return (
    <div data-slot="knowledge-base-passages" className="flex flex-col gap-3">
      {shown.map((group) => (
        <section key={group.key} className="flex flex-col gap-1.5">
          <div className="flex min-w-0 items-center gap-1.5 text-xs font-medium text-foreground">
            <FileGlyph className="size-3 shrink-0" />
            <span className="truncate" title={group.filename}>
              {group.filename}
            </span>
          </div>
          <ol className="flex flex-col gap-1.5">
            {group.passages.map((passage) => (
              <PassageItem key={passage.citation.id} passage={passage} />
            ))}
          </ol>
        </section>
      ))}
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
