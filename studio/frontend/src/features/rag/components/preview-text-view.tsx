// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { authFetch } from "@/features/auth";
import { DownloadIcon, ExternalLinkIcon, FileTextIcon } from "lucide-react";
import { type FC, type ReactNode, useCallback } from "react";
import type { PreviewTarget } from "../api/rag-api";
import { isInlineBlobAllowed } from "../stores/preview-store";

interface PreviewTextViewProps {
  target: PreviewTarget;
}

/** Fetch the original document bytes via authFetch so the bearer
 *  token rides in the Authorization header. `window.open(url)` and
 *  `<a download href=url>` cannot set custom headers, so handing
 *  them the raw `/file` URL gets a 401 (HTTPBearer-only backend —
 *  see D1.3). */
async function fetchOriginalBlob(target: PreviewTarget): Promise<Blob> {
  const response = await authFetch(
    `/api/rag/documents/${encodeURIComponent(target.documentId)}/file`,
  );
  if (!response.ok) {
    throw new Error(`Failed to fetch document (${response.status})`);
  }
  return response.blob();
}

function clickDownloadUrl(url: string, filename: string): void {
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.style.display = "none";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

async function downloadOriginal(target: PreviewTarget): Promise<void> {
  const blob = await fetchOriginalBlob(target);
  const url = URL.createObjectURL(blob);
  clickDownloadUrl(url, target.filename);
  // Defer revocation so the browser's download pipeline gets the bytes
  // before the URL goes away.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

async function openOriginalInNewTab(target: PreviewTarget): Promise<void> {
  // Defense in depth: refuse to create an inline blob URL for the
  // unsafe types even if a future caller forgets the gate. The
  // browser would render an html-blob as live HTML in the new tab,
  // which is the Risk #3 / contracts §2.3 trip.
  if (!isInlineBlobAllowed(target.mediaKind)) {
    throw new Error(
      `Inline open not allowed for mediaKind "${target.mediaKind}" — use Download instead.`,
    );
  }
  const blob = await fetchOriginalBlob(target);
  const url = URL.createObjectURL(blob);
  window.open(url, "_blank", "noopener,noreferrer");
  // Defer revocation so the new tab loads the bytes first.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** Extracted-text / snippet preview used for:
 *  - `text` mediaKind (txt/md) — the snippet is the only inline
 *    rendering we trust, and the original is one click away.
 *  - `docx`, `html`, `unknown` — the original is NEVER rendered
 *    inline from an object URL (Risk #3); we show the cited chunk
 *    text plus a safe download/open action.
 *
 *  When `chunk_id` was not supplied (document-row preview per
 *  contracts §1.3 + decision Q2), `snippet` is `null` and we show a
 *  metadata-only state instead of guessing a first chunk. */
interface MatchRange {
  start: number;
  end: number;
}

function findCharacterRange(
  snippet: string,
  target: PreviewTarget,
): MatchRange | null {
  const { pageCharStart, pageCharEnd } = target;
  if (pageCharStart !== null && pageCharEnd !== null) {
    const start = Math.max(0, pageCharStart);
    const end = Math.min(snippet.length, pageCharEnd);
    if (start < end) {
      return { start, end };
    }
  }
  return null;
}

function findLineRange(
  snippet: string,
  target: PreviewTarget,
): MatchRange | null {
  const { lineStart, lineEnd } = target;
  if (lineStart !== null) {
    const lines = snippet.split("\n");
    const startLineIndex = Math.max(0, lineStart - 1);
    const endLineIndex =
      lineEnd !== null
        ? Math.min(lines.length - 1, lineEnd - 1)
        : startLineIndex;

    let charOffset = 0;
    let startChar = -1;
    let endChar = -1;

    for (let i = 0; i < lines.length; i++) {
      if (i === startLineIndex) {
        startChar = charOffset;
      }
      charOffset += lines[i].length;
      if (i === endLineIndex) {
        endChar = charOffset;
        break;
      }
      charOffset += 1; // for '\n'
    }

    if (startChar !== -1 && endChar !== -1 && startChar < endChar) {
      return { start: startChar, end: endChar };
    }
  }
  return null;
}

function findDensestLineRange(snippet: string): MatchRange | null {
  const lines = snippet.split("\n");
  let bestLineIndex = -1;
  let maxAlphanumericCount = 0;
  for (let i = 0; i < lines.length; i++) {
    const alphanumericCount = lines[i].replace(/[^a-zA-Z0-9]/g, "").length;
    if (alphanumericCount > maxAlphanumericCount) {
      maxAlphanumericCount = alphanumericCount;
      bestLineIndex = i;
    }
  }
  if (bestLineIndex !== -1) {
    let charOffset = 0;
    for (let i = 0; i < bestLineIndex; i++) {
      charOffset += lines[i].length + 1;
    }
    return { start: charOffset, end: charOffset + lines[bestLineIndex].length };
  }

  return null;
}

function findFuzzyMatch(
  snippet: string,
  target: PreviewTarget,
): MatchRange | null {
  return (
    findCharacterRange(snippet, target) ??
    findLineRange(snippet, target) ??
    findDensestLineRange(snippet)
  );
}

const renderHighlightedSnippet = (
  snippet: string,
  target: PreviewTarget,
): ReactNode => {
  const match = findFuzzyMatch(snippet, target);
  if (!match) {
    return snippet;
  }
  const before = snippet.slice(0, match.start);
  const highlighted = snippet.slice(match.start, match.end);
  const after = snippet.slice(match.end);

  return (
    <>
      {before}
      <mark className="rounded bg-primary/20 px-0.5 text-foreground ring-1 ring-primary/60">
        {highlighted}
      </mark>
      {after}
    </>
  );
};

/** Extracted-text / snippet preview used for:
 *  - `text` mediaKind (txt/md) — the snippet is the only inline
 *    rendering we trust, and the original is one click away.
 *  - `docx`, `html`, `unknown` — the original is NEVER rendered
 *    inline from an object URL (Risk #3); we show the cited chunk
 *    text plus a safe download/open action.
 *
 *  When `chunk_id` was not supplied (document-row preview per
 *  contracts §1.3 + decision Q2), `snippet` is `null` and we show a
 *  metadata-only state instead of guessing a first chunk. */
export const PreviewTextView: FC<PreviewTextViewProps> = ({ target }) => {
  const snippet = target.snippet;
  const hasSnippet = snippet !== null && snippet.trim().length > 0;
  const hasLocator =
    target.lineStart !== null ||
    target.lineEnd !== null ||
    target.pageCharStart !== null ||
    target.pageCharEnd !== null;
  // "Open original" creates a blob: URL of the original bytes and
  // passes it to `window.open`. For `html` the new tab would render
  // it as live HTML — exactly the Risk #3 / contracts §2.3 trip
  // ("MUST refuse to create an inline object URL for mediaKind ==
  // 'html' | 'docx' | 'unknown'"). For those types the only safe
  // action is Download (backend already sets
  // Content-Disposition: attachment for those Content-Types). The
  // pdf/text/image allowlist is the same one the preview-store
  // uses to decide whether to fetch the blob at all (§5.4). */
  const canOpenInline = isInlineBlobAllowed(target.mediaKind);

  const handleDownload = useCallback(() => {
    downloadOriginal(target).catch(() => {
      // best-effort; the user can retry the action.
    });
  }, [target]);

  const handleOpenExternal = useCallback(() => {
    // Re-fetch through authFetch and hand the new tab a blob URL.
    // `window.open(rawApiUrl)` would send the request WITHOUT the
    // Authorization header (window.open can't set custom headers)
    // and the HTTPBearer-protected /file route would respond 401.
    // See D1.3 finding.
    openOriginalInNewTab(target).catch(() => {
      // best-effort; the user can retry.
    });
  }, [target]);

  return (
    <div className="flex h-full flex-col gap-3 overflow-hidden p-4">
      <div className="flex items-center gap-2 text-sm">
        <FileTextIcon
          className="size-4 shrink-0 text-muted-foreground"
          aria-hidden={true}
        />
        <span className="truncate font-medium" title={target.filename}>
          {target.filename}
        </span>
      </div>
      {target.targetPage != null ? (
        <p className="text-xs text-muted-foreground">
          Cited from page {target.targetPage}
        </p>
      ) : null}

      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-hidden rounded-md border border-border/60 bg-muted/30 p-3">
        {hasSnippet ? (
          <>
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground">
              {hasLocator ? "Highlighted source excerpt" : "Source excerpt"}
            </p>
            {hasLocator ? (
              <pre className="flex-1 overflow-auto whitespace-pre-wrap break-words text-xs leading-relaxed text-foreground/85">
                {renderHighlightedSnippet(snippet, target)}
              </pre>
            ) : (
              <pre className="flex-1 overflow-auto whitespace-pre-wrap break-words text-xs leading-relaxed text-foreground/85">
                {snippet}
              </pre>
            )}
          </>
        ) : (
          <p className="my-auto text-center text-xs text-muted-foreground">
            {canOpenInline
              ? "No source excerpt — open the original to view this document."
              : "No source excerpt — download the original to view this document."}
          </p>
        )}
      </div>

      <div className="flex gap-2">
        {canOpenInline ? (
          <Button
            variant="outline"
            size="sm"
            onClick={handleOpenExternal}
            className="flex-1"
          >
            <ExternalLinkIcon className="size-3.5" />
            Open original
          </Button>
        ) : null}
        <Button
          variant="outline"
          size="sm"
          onClick={handleDownload}
          className="flex-1"
        >
          <DownloadIcon className="size-3.5" />
          Download
        </Button>
      </div>
    </div>
  );
};
