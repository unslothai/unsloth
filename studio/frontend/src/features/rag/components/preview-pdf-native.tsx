// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { ExternalLinkIcon } from "lucide-react";
import { type FC, useEffect, useMemo } from "react";
import type { PreviewTarget } from "../api/rag-api";
import { renderHighlightedSnippet } from "./preview-text-view";

interface PreviewPdfNativeViewProps {
  target: PreviewTarget;
  /** Signed file URL (preferred — supports HTTP Range) or a Blob of bytes. */
  file: string | Blob;
}

/** Lightweight PDF preview: render the document in the browser's built-in
 *  PDF viewer via an <iframe> jumped to the cited page (`#page=N`). A
 *  companion strip shows the cited excerpt highlighted, since the native
 *  viewer can't be overlaid with on-page region boxes. This deliberately
 *  avoids bundling a JS PDF renderer (react-pdf/pdfjs). The file response is
 *  served same-origin with `frame-ancestors` allowing this iframe (see
 *  routes/rag.py `_serve_document_file_row` + main.py `_is_frameable_path`). */
export const PreviewPdfNativeView: FC<PreviewPdfNativeViewProps> = ({
  target,
  file,
}) => {
  // A Blob needs an object URL; a signed URL string is used directly.
  const objectUrl = useMemo(
    () => (typeof file === "string" ? null : URL.createObjectURL(file)),
    [file],
  );
  useEffect(() => {
    return () => {
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [objectUrl]);

  const baseUrl = typeof file === "string" ? file : objectUrl;
  const page = target.targetPage ?? 1;
  // #page=N + view=FitH are honored by the Chrome/Firefox/Edge native PDF
  // viewers; the hash is never sent to the server.
  const src = baseUrl ? `${baseUrl}#page=${page}&view=FitH` : null;

  const snippet = target.snippet;
  const hasSnippet = snippet !== null && snippet.trim().length > 0;

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <div className="flex items-center justify-between gap-2 border-b border-border/60 px-3 py-2">
        <span className="truncate text-sm font-medium" title={target.filename}>
          {target.filename}
        </span>
        {target.targetPage != null ? (
          <span className="shrink-0 text-xs text-muted-foreground">
            Page {target.targetPage}
          </span>
        ) : null}
      </div>

      {src ? (
        <iframe
          src={src}
          title={`PDF preview: ${target.filename}`}
          className="min-h-0 w-full flex-1 border-0 bg-muted/20"
        />
      ) : (
        <div className="flex flex-1 items-center justify-center text-xs text-muted-foreground">
          Preview unavailable.
        </div>
      )}

      {hasSnippet ? (
        <details
          open={true}
          className="max-h-40 shrink-0 overflow-auto border-t border-border/60 bg-muted/30 px-3 py-2"
        >
          <summary className="cursor-pointer text-[10px] uppercase tracking-wider text-muted-foreground">
            Cited excerpt
          </summary>
          <pre className="mt-1.5 whitespace-pre-wrap break-words text-xs leading-relaxed text-foreground/85">
            {renderHighlightedSnippet(snippet, target)}
          </pre>
        </details>
      ) : null}

      {src ? (
        <div className="shrink-0 border-t border-border/60 px-3 py-2">
          <Button asChild={true} variant="outline" size="sm" className="w-full">
            <a href={src} target="_blank" rel="noopener noreferrer">
              <ExternalLinkIcon className="size-3.5" />
              Open in new tab
            </a>
          </Button>
        </div>
      ) : null}
    </div>
  );
};
