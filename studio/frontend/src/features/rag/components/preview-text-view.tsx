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

/** Fetch the original document bytes via authFetch so the bearer token
 *  rides in the Authorization header. `window.open(url)` / `<a download>`
 *  can't set custom headers, so the raw `/file` URL gets a 401 on the
 *  HTTPBearer-only backend (see D1.3). */
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
  // Defer revocation so the download pipeline gets the bytes first.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

async function openOriginalInNewTab(target: PreviewTarget): Promise<void> {
  // Defense in depth: refuse an inline blob URL for the unsafe types
  // even if a caller forgets the gate. An html-blob would render as
  // live HTML in the new tab — the Risk #3 / contracts §2.3 trip.
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
 *  - `text` (txt/md) — snippet is the only inline rendering we trust;
 *    the original is one click away.
 *  - `docx` / `html` / `unknown` — original is NEVER rendered inline
 *    from an object URL (Risk #3); show the cited chunk text plus a
 *    safe download/open action.
 *
 *  When no `chunk_id` was supplied (document-row preview, contracts
 *  §1.3 + Q2), `snippet` is null and we show a metadata-only state
 *  instead of guessing a first chunk. */
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
 *  - `text` (txt/md) — snippet is the only inline rendering we trust;
 *    the original is one click away.
 *  - `docx` / `html` / `unknown` — original is NEVER rendered inline
 *    from an object URL (Risk #3); show the cited chunk text plus a
 *    safe download/open action.
 *
 *  When no `chunk_id` was supplied (document-row preview, contracts
 *  §1.3 + Q2), `snippet` is null and we show a metadata-only state
 *  instead of guessing a first chunk. */
export const PreviewTextView: FC<PreviewTextViewProps> = ({ target }) => {
  const snippet = target.snippet;
  const hasSnippet = snippet !== null && snippet.trim().length > 0;
  const hasLocator =
    target.lineStart !== null ||
    target.lineEnd !== null ||
    target.pageCharStart !== null ||
    target.pageCharEnd !== null;
  // "Open original" blobs the bytes and hands them to `window.open`. For
  // html/docx/unknown the new tab would render live HTML — the Risk #3 /
  // contracts §2.3 trip ("MUST refuse an inline object URL" for those
  // kinds). Download is the only safe action there (backend sends
  // Content-Disposition: attachment). The pdf/text/image allowlist is the
  // same one preview-store uses to gate the blob fetch (§5.4).
  const canOpenInline = isInlineBlobAllowed(target.mediaKind);

  const handleDownload = useCallback(() => {
    downloadOriginal(target).catch(() => {
      // best-effort; the user can retry the action.
    });
  }, [target]);

  const handleOpenExternal = useCallback(() => {
    // Re-fetch via authFetch and hand the new tab a blob URL.
    // `window.open(rawApiUrl)` would omit the Authorization header
    // (window.open can't set custom headers), so the HTTPBearer-protected
    // /file route would 401. See D1.3 finding.
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
