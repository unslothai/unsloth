// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/* eslint-disable react-refresh/only-export-components */

import { useDocumentPreviewStore } from "@/features/rag/components/preview-store";
import type { PreviewFigure } from "@/features/rag/types/rag";
import { cn } from "@/lib/utils";
import { FileText } from "lucide-react";
import type { ReactElement } from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { ExtractedDocument, PendingDocumentAttachment } from "../types";
import {
  documentFigureImageDataUrl,
  documentVisualPayloads,
  formatDocumentTokens,
} from "../utils/document-extraction";
import {
  AttachmentChipBody,
  AttachmentChipButton,
  AttachmentChipRemoveButton,
  AttachmentChipTitle,
} from "./attachment-chip-primitives";

const QUERY_FRAGMENT_RE = /[?#]/;
const PATH_SEPARATOR_RE = /[\\/]/;

export function documentFileTypeLabel(filename: string): string {
  const cleanName = filename.split(QUERY_FRAGMENT_RE)[0] ?? filename;
  const baseName = cleanName.split(PATH_SEPARATOR_RE).pop() ?? cleanName;
  const extension = baseName.includes(".") ? baseName.split(".").pop() : "";

  if (!extension) {
    return "DOC";
  }
  return extension.slice(0, 8).toUpperCase();
}

function previewFiguresForDocument(doc: ExtractedDocument): PreviewFigure[] {
  return doc.figures.map((figure) => ({
    id: figure.id,
    page: figure.page,
    caption: figure.caption,
    imageDataUrl: documentFigureImageDataUrl(figure),
  }));
}

/**
 * Opens an extracted document in the shared RAG preview Sheet, rendering its
 * markdown body and any inline figures without a backend documentId.
 */
export function openExtractedDocumentPreview(input: {
  filename: string;
  document: ExtractedDocument;
}): void {
  const { filename, document: doc } = input;
  useDocumentPreviewStore.getState().openPreview({
    documentId: `extracted:${filename}`,
    filename: doc.filename || filename,
    mediaKind: "markdown",
    markdown: doc.markdown,
    figures: previewFiguresForDocument(doc),
  });
}

export function documentAttachmentSummary(
  doc: ExtractedDocument,
  maxVisualPayloads: number,
  sentImageIndexes: number[] | undefined,
): { fileType: string; subtitle: string } {
  const visualPayloadCount =
    sentImageIndexes?.length ??
    documentVisualPayloads(doc, maxVisualPayloads).length;
  const imageCount = doc.figures.length;
  const fileType = documentFileTypeLabel(doc.filename);
  const subtitle = [
    `${doc.page_count} page${doc.page_count === 1 ? "" : "s"}`,
    `${formatDocumentTokens(doc.tokens_est)} tokens`,
    `${imageCount} ref${imageCount === 1 ? "" : "s"}`,
    visualPayloadCount > 0
      ? `${visualPayloadCount} image${visualPayloadCount === 1 ? "" : "s"}`
      : "Text only",
  ].join(" · ");
  return { fileType, subtitle };
}

export interface DocAttachmentChipProps {
  attachment: PendingDocumentAttachment;
  onRemove?: () => void;
  className?: string;
  wrapperClassName?: string;
}

/**
 * Thin attachment chip for a ready document. Shows filename + a compact
 * summary (pages/tokens); clicking opens the shared RAG preview Sheet.
 * Reuses `attachment-chip-primitives` for layout and the RAG status-chip look.
 */
export function DocAttachmentChip({
  attachment,
  onRemove,
  className,
  wrapperClassName,
}: DocAttachmentChipProps): ReactElement {
  const maxVisualPayloads = useChatRuntimeStore(
    (s) => s.docExtract.maxVisualPayloads,
  );
  const { document: doc, filename } = attachment;
  const { fileType, subtitle } = documentAttachmentSummary(
    doc,
    maxVisualPayloads,
    attachment.sentImageIndexes,
  );

  return (
    <span className={cn("relative inline-flex max-w-full", wrapperClassName)}>
      <AttachmentChipButton
        className={cn(
          "aui-attachment-document-chip relative max-w-[min(20rem,calc(100vw-3rem))] items-center rounded-md border-border/70 bg-card text-card-foreground shadow-sm backdrop-blur-none dark:bg-card",
          onRemove ? "pr-9" : "pr-3",
          className,
        )}
        onClick={() => openExtractedDocumentPreview({ filename, document: doc })}
        aria-label={`Preview extracted markdown from ${filename}`}
      >
        <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-amber-500/15 text-amber-600 dark:text-amber-400">
          <FileText className="size-4" aria-hidden="true" />
        </span>
        <AttachmentChipBody className="gap-0">
          <span className="flex min-w-0 items-center gap-1.5">
            <AttachmentChipTitle className="text-xs" title={filename}>
              {filename}
            </AttachmentChipTitle>
            <span className="shrink-0 rounded-md border border-border/70 bg-background/80 px-1 py-0.5 text-[9px] font-semibold text-muted-foreground dark:bg-card/80">
              {fileType}
            </span>
          </span>
          <span
            className="truncate text-[11px] text-muted-foreground"
            title={subtitle}
          >
            {subtitle}
          </span>
        </AttachmentChipBody>
      </AttachmentChipButton>
      {onRemove ? (
        <AttachmentChipRemoveButton
          tooltip="Remove file"
          onClick={(event) => {
            event.preventDefault();
            event.stopPropagation();
            onRemove();
          }}
          aria-label={`Remove ${filename}`}
        />
      ) : null}
    </span>
  );
}
