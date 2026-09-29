// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type DocumentKind, DocumentView, documentKind, isMarkdown } from "@/components/file-viewer";
import { queueParse } from "@/components/file-viewer/parse-queue";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { type AttachmentFileKind, readAttachmentText } from "@/features/chat";
import { cn } from "@/lib/utils";
import { type FC, type ReactNode, useEffect, useLayoutEffect, useState } from "react";

// Larger files keep the icon: a visible card parses its whole file.
const MAX_PREVIEW_BYTES = 10 * 1024 * 1024;
const TEXT_PREVIEW_CHARS = 8 * 1024;
// Width the first page lays out at before it is scaled to the card.
const PAGE_WIDTH: Record<DocumentKind, number> = { pdf: 400, docx: 816, sheet: 640, slides: 640 };
const TEXT_WIDTH = 480;

export type AttachmentPreview =
  | { kind: "document"; document: DocumentKind }
  | { kind: "markdown" }
  | { kind: "text" };

export function attachmentPreview(
  file: File | undefined,
  kind: AttachmentFileKind,
): AttachmentPreview | null {
  if (!file || file.size === 0 || file.size > MAX_PREVIEW_BYTES) return null;
  // The resolved kind is MIME first; only preview when the extension agrees with it.
  const document = documentKind(file.name, file.type);
  if (document) return DOCUMENT_KINDS[document] === kind ? { kind: "document", document } : null;
  if (kind === "text" && isMarkdown(file.name, file.type)) return { kind: "markdown" };
  if (kind === "text" || kind === "code" || kind === "web") return { kind: "text" };
  return null;
}

const DOCUMENT_KINDS: Record<DocumentKind, AttachmentFileKind> = {
  pdf: "pdf",
  docx: "document",
  sheet: "spreadsheet",
  slides: "presentation",
};

/** Whether the element is on screen now; the strip clips offscreen cards. */
function useVisible(element: HTMLElement | null): boolean {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    if (!element) return;
    const observer = new IntersectionObserver((entries) => {
      const entry = entries[entries.length - 1];
      if (entry) setVisible(entry.isIntersecting);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [element]);
  return visible;
}

function useSize(element: HTMLElement | null): { width: number; height: number } {
  const [size, setSize] = useState({ width: 0, height: 0 });
  useLayoutEffect(() => {
    if (!element) return;
    const measure = () => setSize({ width: element.clientWidth, height: element.clientHeight });
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, [element]);
  return size;
}

// Same decoder as the attachment itself, so a declared charset reads correctly.
// Queued like Office parses: a charset-declaring file is read whole.
function useLeadingText(file: File, enabled: boolean): { text: string } | "failed" | null {
  const [state, setState] = useState<{ file: File; text: string | null } | null>(null);
  const done = state?.file === file;
  useEffect(() => {
    if (!enabled || done) return;
    let cancelled = false;
    queueParse(() => readAttachmentText(file, file.name, file.type), () => cancelled).then(
      (read) => !cancelled && read && setState({ file, text: read.text.slice(0, TEXT_PREVIEW_CHARS) }),
      () => !cancelled && setState({ file, text: null }),
    );
    return () => {
      cancelled = true;
    };
  }, [file, enabled, done]);
  if (state?.file !== file) return null;
  return state.text === null ? "failed" : { text: state.text };
}

/** First page of the file, scaled to fill the card. Display only: never focused or clicked. */
export const AttachmentCardPreview: FC<{
  file: File;
  preview: AttachmentPreview;
  /** Shown when the text cannot be decoded. */
  fallback: ReactNode;
}> = ({ file, preview, fallback }) => {
  const [frame, setFrame] = useState<HTMLDivElement | null>(null);
  const size = useSize(frame);
  // Reads start only while visible. Documents unmount offscreen; read text stays.
  const visible = useVisible(frame);
  const text = useLeadingText(file, visible && preview.kind !== "document");
  const pageWidth = preview.kind === "document" ? PAGE_WIDTH[preview.document] : TEXT_WIDTH;
  const scale = size.width / pageWidth;
  // PDF, Word and slide pages are white paper; sheets and text follow the theme.
  const paper = preview.kind === "document" && preview.document !== "sheet";

  if (text === "failed") return fallback;

  let body: ReactNode = null;
  if (preview.kind === "document") {
    body = visible ? (
      <DocumentView
        file={file}
        kind={preview.document}
        name={file.name}
        contentType={file.type}
        thumbnail={true}
      />
    ) : null;
  } else if (text !== null && preview.kind === "markdown") {
    body = (
      <MarkdownPreview
        markdown={text.text}
        className="max-h-none overflow-visible border-0 bg-transparent px-6 py-5 text-ui-15p5"
      />
    );
  } else if (text !== null) {
    body = (
      <pre className="whitespace-pre-wrap break-words px-6 py-5 font-mono text-[13px] leading-snug text-foreground">
        {text.text}
      </pre>
    );
  }

  return (
    <div
      ref={setFrame}
      aria-hidden={true}
      inert={true}
      className={cn(
        "attachment-card-preview pointer-events-none relative size-full select-none overflow-hidden",
        paper ? "bg-white" : "bg-background",
      )}
    >
      {scale > 0 && (
        <div
          className="attachment-card-preview-page absolute top-0 left-0 flex origin-top-left flex-col"
          style={{ width: pageWidth, height: size.height / scale, transform: `scale(${scale})` }}
        >
          {body}
        </div>
      )}
    </div>
  );
};
