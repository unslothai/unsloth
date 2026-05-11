// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/* eslint-disable react-refresh/only-export-components */

import { cn } from "@/lib/utils";
import { FileText } from "lucide-react";
import type { ReactElement } from "react";
import { useChatRuntimeStore } from "../stores/chat-runtime-store";
import type { PendingDocumentAttachment } from "../types";
import { documentVisualPayloads } from "../utils/document-extraction";
import {
  AttachmentChipBody,
  AttachmentChipButton,
  AttachmentChipRemoveButton,
  AttachmentChipTitle,
} from "./attachment-chip-primitives";
import {
  DocumentPreviewSheet,
  type DocumentSheetNavigation,
} from "./document-preview-panel";

const QUERY_FRAGMENT_RE = /[?#]/;
const PATH_SEPARATOR_RE = /[\\/]/;

export function formatDocumentTokens(tokens: number): string {
  if (tokens < 1000) {
    return `${tokens}`;
  }
  return `${(tokens / 1000).toFixed(1)}k`;
}

export function documentFileTypeLabel(filename: string): string {
  const cleanName = filename.split(QUERY_FRAGMENT_RE)[0] ?? filename;
  const baseName = cleanName.split(PATH_SEPARATOR_RE).pop() ?? cleanName;
  const extension = baseName.includes(".") ? baseName.split(".").pop() : "";

  if (!extension) {
    return "DOC";
  }
  return extension.slice(0, 8).toUpperCase();
}

export function documentAttachmentSummary(
  attachment: PendingDocumentAttachment,
  maxVisualPayloads: number,
): {
  fileType: string;
  subtitle: string;
  visualPayloads: ReturnType<typeof documentVisualPayloads>;
} {
  const { document: doc, filename } = attachment;
  const visualPayloads = documentVisualPayloads(doc, maxVisualPayloads);
  const visualPayloadCount =
    attachment.sentImageIndexes?.length ?? visualPayloads.length;
  const imageCount = doc.figures.length;
  const fileType = documentFileTypeLabel(filename);
  const subtitle = [
    `${doc.page_count} page${doc.page_count === 1 ? "" : "s"}`,
    `${formatDocumentTokens(doc.tokens_est)} tokens`,
    `${imageCount} ref${imageCount === 1 ? "" : "s"}`,
    visualPayloadCount > 0
      ? `${visualPayloadCount} image${visualPayloadCount === 1 ? "" : "s"}`
      : "Text only",
  ].join(" · ");

  return { fileType, subtitle, visualPayloads };
}

export interface DocAttachmentChipProps {
  attachment: PendingDocumentAttachment;
  contextWindow?: number;
  onRemove?: () => void;
  className?: string;
  wrapperClassName?: string;
  navigation?: DocumentSheetNavigation;
  previewOpen?: boolean;
  onPreviewOpenChange?: (open: boolean) => void;
}

export function DocAttachmentChip({
  attachment,
  onRemove,
  className,
  wrapperClassName,
  navigation,
  previewOpen,
  onPreviewOpenChange,
}: DocAttachmentChipProps): ReactElement {
  const maxVisualPayloads = useChatRuntimeStore(
    (s) => s.docExtract.maxVisualPayloads,
  );
  const { document: doc, filename, sizeBytes } = attachment;
  const { fileType, subtitle, visualPayloads } = documentAttachmentSummary(
    attachment,
    maxVisualPayloads,
  );
  const sentImageIndexes = new Set(
    attachment.sentImageIndexes ?? visualPayloads.map((payload) => payload.index),
  );

  const chip = (
    <AttachmentChipButton
      className={cn(
        "aui-attachment-document-chip relative max-w-[min(20rem,calc(100vw-3rem))] items-center rounded-md border-border/70 bg-card text-card-foreground shadow-sm backdrop-blur-none dark:bg-card",
        onRemove ? "pr-9" : "pr-3",
        className,
      )}
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
  );

  return (
    <span className={cn("relative inline-flex max-w-full", wrapperClassName)}>
      <DocumentPreviewSheet
        document={doc}
        filename={filename}
        sizeBytes={sizeBytes}
        extractedAt={attachment.extractedAt}
        sentImageIndexes={sentImageIndexes}
        navigation={navigation}
        open={previewOpen}
        onOpenChange={onPreviewOpenChange}
      >
        {chip}
      </DocumentPreviewSheet>
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
