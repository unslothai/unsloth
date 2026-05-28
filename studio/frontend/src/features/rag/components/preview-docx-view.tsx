// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import DOMPurify from "dompurify";
import { renderAsync } from "docx-preview";
import { type FC, useEffect, useRef, useState } from "react";
import type { PreviewTarget } from "../api/rag-api";

/** Inline DOCX preview. docx-preview renders the .docx bytes to HTML; the
 *  source document is user-supplied, so we sanitize that HTML with
 *  DOMPurify before injecting it (Risk #3 — a malicious .docx must not
 *  execute script in the app origin). The raw bytes are never exposed as
 *  an object URL — only the sanitized render and the Download action. */
export const PreviewDocxView: FC<{ target: PreviewTarget; blob: Blob }> = ({
  blob,
}) => {
  const ref = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    // Render off-screen first so we can sanitize before the markup ever
    // touches the live DOM.
    const offscreen = document.createElement("div");
    void renderAsync(blob, offscreen, undefined, {
      inWrapper: true,
      ignoreLastRenderedPageBreak: true,
    })
      .then(() => {
        if (cancelled || !ref.current) return;
        // Keep <style> so docx-preview's scoped layout CSS survives the
        // sanitize pass; everything else uses DOMPurify defaults (drops
        // <script>, event handlers, javascript: URLs, etc.).
        const clean = DOMPurify.sanitize(offscreen.innerHTML, {
          ADD_TAGS: ["style"],
        });
        ref.current.innerHTML = clean;
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [blob]);

  if (error) {
    return (
      <div className="p-4 text-xs text-destructive">
        Could not render this document. Use Download to open the original.
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto bg-muted/30 p-4">
      <div ref={ref} className="docx-preview mx-auto" />
    </div>
  );
};
