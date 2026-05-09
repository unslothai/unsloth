// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

// Avatar removed — caused circular crop on image thumbnails
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  AttachmentChipBody,
  AttachmentChipButton,
  AttachmentChipProgress,
  AttachmentChipRemoveButton,
  AttachmentChipTitle,
  DocumentPreviewSheet,
  DocumentStack,
  attachmentChipTokens,
  documentFigureImageDataUrl,
  isDocumentAttachment,
  type DocumentPendingAttachment,
  type ExtractedDocument,
  type PendingDocumentAttachment as DocumentStackAttachment,
} from "@/features/chat";
import { cn } from "@/lib/utils";
import {
  AttachmentPrimitive,
  type CompleteAttachment,
  ComposerPrimitive,
  MessagePrimitive,
  type PendingAttachment as AuiPendingAttachment,
  useAui,
  useAuiState,
} from "@assistant-ui/react";
import { FileText, LoaderIcon, PlusIcon, XIcon } from "lucide-react";
import {
  type FC,
  type PropsWithChildren,
  useEffect,
  useId,
  useMemo,
  useState,
} from "react";
import { useShallow } from "zustand/shallow";

const useFileSrc = (file: File | undefined): string | undefined => {
  const objectUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : undefined),
    [file],
  );

  useEffect(() => {
    if (!objectUrl) return;
    return () => URL.revokeObjectURL(objectUrl);
  }, [objectUrl]);

  return objectUrl;
};

const useAttachmentSrc = (): string | undefined => {
  const { file, src } = useAuiState(
    useShallow(({ attachment }): { file?: File; src?: string } => {
      if (attachment.type === "image" && attachment.file) {
        return { file: attachment.file };
      }
      const src = attachment.content?.filter((c) => c.type === "image")[0]
        ?.image;
      if (!src) {
        return {};
      }
      return { src };
    }),
  );

  return useFileSrc(file) ?? src;
};

type DocumentAttachmentState = {
  id?: string;
  type: string;
  name: string;
  file?: File;
  content?: Array<{ type: string; image?: string }>;
  sizeBytes?: number;
  extractedAt?: number;
  truncated?: boolean;
  sentImageIndexes?: number[];
  errorCode?: string;
  errorMessage?: string;
  retryCount?: number;
  status: {
    type: "running" | "requires-action" | "incomplete" | "complete";
    progress?: number;
    reason?: string;
  };
  document?: ExtractedDocument;
};

type StackableAttachment = AuiPendingAttachment | CompleteAttachment;

type DocumentVisualAttachment = {
  content?: ReadonlyArray<{ type: string; image?: string }>;
  sentImageIndexes?: readonly number[];
};

function isDocumentAttachmentState(
  attachment: unknown,
): attachment is DocumentAttachmentState {
  return (
    typeof attachment === "object" &&
    attachment !== null &&
    "type" in attachment &&
    (attachment as { type?: unknown }).type === "document"
  );
}

function isReadyDocumentAttachment(
  attachment: DocumentAttachmentState,
): boolean {
  return (
    Boolean(attachment.document) &&
    attachment.status.type !== "running" &&
    attachment.status.type !== "incomplete"
  );
}

function documentStackItemFromAttachment(
  attachment: StackableAttachment,
): DocumentStackAttachment | null {
  if (!isDocumentAttachment(attachment) || !attachment.document) {
    return null;
  }

  const documentAttachment = attachment as DocumentPendingAttachment;
  const document = documentAttachment.document;
  if (!document) {
    return null;
  }

  const filename = document.filename || documentAttachment.name;
  const sentImageIndexes = sentImageIndexesForAttachment(
    documentAttachment,
    document,
  );

  return {
    id: documentAttachment.id,
    filename,
    sizeBytes: documentAttachment.sizeBytes ?? 0,
    document,
    extractedAt: documentAttachment.extractedAt ?? 0,
    truncated: documentAttachment.truncated ?? document.truncated,
    sentImageIndexes,
  };
}

function sentImageIndexesForAttachment(
  documentAttachment: DocumentVisualAttachment,
  document: ExtractedDocument,
): number[] {
  if (Array.isArray(documentAttachment.sentImageIndexes)) {
    return documentAttachment.sentImageIndexes.filter(
      (index) =>
        Number.isInteger(index) && index >= 0 && index < document.figures.length,
    );
  }

  const sentImageUrls = new Set(
    (documentAttachment.content ?? [])
      .flatMap((part) => {
        if (part.type !== "image" || !part.image) {
          return [];
        }
        return [part.image];
      }),
  );
  return document.figures
    .map((figure, index) => ({
      index,
      dataUrl: documentFigureImageDataUrl(figure),
    }))
    .filter(({ dataUrl }) => dataUrl !== null && sentImageUrls.has(dataUrl))
    .map(({ index }) => index);
}

function documentStackItemsFromAttachments(
  attachments: readonly StackableAttachment[] | undefined,
): DocumentStackAttachment[] {
  return (attachments ?? [])
    .map(documentStackItemFromAttachment)
    .filter((item): item is DocumentStackAttachment => item !== null);
}

function fileExtension(filename: string): string {
  const idx = filename.lastIndexOf(".");
  if (idx < 0 || idx === filename.length - 1) return "Document";
  return filename.slice(idx + 1).toUpperCase();
}

function formatTokens(tokens: number): string {
  if (tokens < 1000) return `${tokens}`;
  return `${(tokens / 1000).toFixed(1)}k`;
}

function buildDocSubtitle(
  doc: ExtractedDocument,
  visualPayloadCount: number,
): string {
  const imageCount = doc.figures.length;
  return [
    `${doc.page_count} page${doc.page_count === 1 ? "" : "s"}`,
    `${formatTokens(doc.tokens_est)} tokens`,
    imageCount > 0 ? `${imageCount} ref${imageCount === 1 ? "" : "s"}` : null,
    visualPayloadCount > 0
      ? `${visualPayloadCount} image${visualPayloadCount === 1 ? "" : "s"}`
      : "Text only",
  ]
    .filter((item): item is string => Boolean(item))
    .join(" · ");
}

type AttachmentPreviewProps = {
  src: string;
};

const AttachmentPreview: FC<AttachmentPreviewProps> = ({ src }) => {
  const [isLoaded, setIsLoaded] = useState(false);
  return (
    <img
      src={src}
      alt="Preview"
      className={cn(
        "block h-auto max-h-[80vh] w-auto max-w-full object-contain",
        isLoaded
          ? "aui-attachment-preview-image-loaded"
          : "aui-attachment-preview-image-loading invisible",
      )}
      onLoad={() => setIsLoaded(true)}
    />
  );
};

const AttachmentPreviewDialog: FC<PropsWithChildren> = ({ children }) => {
  const src = useAttachmentSrc();

  if (!src) {
    return children;
  }

  return (
    <Dialog>
      <DialogTrigger
        className="aui-attachment-preview-trigger cursor-pointer transition-colors hover:bg-accent/50"
        asChild={true}
      >
        {children}
      </DialogTrigger>
      <DialogContent className="aui-attachment-preview-dialog-content p-2 sm:max-w-3xl [&>button]:rounded-full [&>button]:bg-foreground/60 [&>button]:p-1 [&>button]:opacity-100 [&>button]:ring-0! [&_svg]:text-background [&>button]:hover:[&_svg]:text-destructive">
        <DialogTitle className="aui-sr-only sr-only">
          Image Attachment Preview
        </DialogTitle>
        <div className="aui-attachment-preview relative mx-auto flex max-h-[80dvh] w-full items-center justify-center overflow-hidden bg-background">
          <AttachmentPreview src={src} />
        </div>
      </DialogContent>
    </Dialog>
  );
};

const AttachmentThumb: FC = () => {
  const src = useAttachmentSrc();

  if (src) {
    return (
      <img
        src={src}
        alt="Attachment preview"
        className="h-full w-full object-cover"
      />
    );
  }

  return (
    <div className="flex h-full w-full items-center justify-center">
      <FileText className="size-6 text-muted-foreground" />
    </div>
  );
};

const AttachmentUI: FC = () => {
  const aui = useAui();
  const isComposer = aui.attachment.source === "composer";
  const rawAttachment = useAuiState(useShallow(({ attachment }) => attachment));
  const docAttachment: DocumentAttachmentState | null =
    isDocumentAttachmentState(rawAttachment)
      ? (rawAttachment as unknown as DocumentAttachmentState)
      : null;

  const isImage = useAuiState(({ attachment }) => attachment.type === "image");
  const typeLabel = useAuiState(({ attachment }) => {
    const type = attachment.type;
    switch (type) {
      case "image":
        return "Image";
      case "document":
        return "Document";
      case "file":
        return "File";
      default:
        throw new Error(`Unknown attachment type: ${type as string}`);
    }
  });
  // Suffix with a per-instance React id so attachments without a stable
  // `rawAttachment.id` (or that share a typeLabel like "image") still produce
  // a unique DOM id within a single composer.
  const reactInstanceId = useId().replace(/[^A-Za-z0-9_-]/g, "-");

  if (docAttachment !== null) {
    const doc = docAttachment.document;
    const running = docAttachment.status.type === "running";
    const failed = docAttachment.status.type === "incomplete";
    const truncated =
      (docAttachment as { truncated?: boolean }).truncated === true;
    const failedReason = failed
      ? (docAttachment.errorMessage ??
        docAttachment.status.reason ??
        "Extraction failed")
      : null;
    const sentImageIndexes = new Set(
      doc ? sentImageIndexesForAttachment(docAttachment, doc) : [],
    );
    const progressValue =
      typeof docAttachment.status.progress === "number" &&
      Number.isFinite(docAttachment.status.progress)
        ? Math.max(0, Math.min(100, docAttachment.status.progress * 100))
        : null;
    const progressLabel =
      progressValue === null
        ? "Reading document"
        : `${Math.round(progressValue)}% uploaded`;
    const ext = fileExtension(docAttachment.name);
    const visualPayloadCount = sentImageIndexes.size;
    const readyDetails = doc ? buildDocSubtitle(doc, visualPayloadCount) : ext;
    const subtitle = failed
      ? (failedReason ?? "Extraction failed")
      : running
        ? progressValue !== null
          ? `Reading… ${Math.round(progressValue)}%`
          : "Reading…"
        : truncated
          ? `${readyDetails} · Truncated`
          : readyDetails;
    const tileClass = failed
      ? "bg-destructive/10 text-destructive/90"
      : running
        ? "bg-muted/50 text-muted-foreground/80"
        : "bg-amber-500/10 text-amber-600 dark:text-amber-400/90";
    const chip = (
      <AttachmentChipButton
        className="aui-attachment-document-chip max-w-[min(20rem,calc(100vw-3rem))] items-center pr-9"
        aria-label={`${typeLabel} attachment ${docAttachment.name}`}
      >
        <span
          className={cn(
            "flex size-10 shrink-0 items-center justify-center rounded-md",
            tileClass,
          )}
        >
          {running ? (
            <LoaderIcon
              className="size-5 animate-spin motion-reduce:animate-none"
              aria-hidden="true"
            />
          ) : (
            <FileText className="size-5" aria-hidden="true" />
          )}
        </span>
        <AttachmentChipBody className="gap-0.5">
          <AttachmentChipTitle className="text-sm" title={docAttachment.name}>
            <AttachmentPrimitive.Name />
          </AttachmentChipTitle>
          <span
            className={cn(
              "truncate text-xs",
              failed ? "text-destructive" : "text-muted-foreground",
            )}
            title={subtitle}
          >
            {subtitle}
          </span>
          {running ? (
            <AttachmentChipProgress
              value={progressValue}
              label={progressLabel}
              className="mt-1"
            />
          ) : null}
        </AttachmentChipBody>
      </AttachmentChipButton>
    );

    return (
      <Tooltip>
        <AttachmentPrimitive.Root
          className="aui-attachment-root relative max-w-full"
          role={failed ? "alert" : undefined}
        >
          {doc ? (
            <DocumentPreviewSheet
              document={doc}
              filename={doc.filename || docAttachment.name}
              sizeBytes={docAttachment.sizeBytes}
              extractedAt={docAttachment.extractedAt}
              sentImageIndexes={sentImageIndexes}
            >
              <TooltipTrigger asChild={true}>{chip}</TooltipTrigger>
            </DocumentPreviewSheet>
          ) : (
            <AttachmentPreviewDialog>
              <TooltipTrigger asChild={true}>{chip}</TooltipTrigger>
            </AttachmentPreviewDialog>
          )}
          {isComposer && <AttachmentRemove />}
        </AttachmentPrimitive.Root>
        <TooltipContent side="top">
          <AttachmentPrimitive.Name />
        </TooltipContent>
      </Tooltip>
    );
  }

  const attachmentDomId = `attachment-tile-${String(
    (rawAttachment as { id?: string }).id ?? typeLabel,
  ).replace(/[^A-Za-z0-9_-]/g, "-")}-${reactInstanceId}`;

  return (
    <Tooltip>
      <AttachmentPrimitive.Root
        className={cn(
          "aui-attachment-root relative",
          isImage &&
            "aui-attachment-root-composer only:[&>.aui-attachment-tile]:size-16",
        )}
      >
        <AttachmentPreviewDialog>
          <TooltipTrigger asChild={true}>
            <button
              className={cn(
                attachmentChipTokens.tile,
                "aui-attachment-tile cursor-pointer transition-opacity hover:opacity-75",
                isComposer &&
                  "aui-attachment-tile-composer border-foreground/20",
              )}
              id={attachmentDomId}
              aria-label={`${typeLabel} attachment`}
              type="button"
            >
              <AttachmentThumb />
            </button>
          </TooltipTrigger>
        </AttachmentPreviewDialog>
        {isComposer && <AttachmentRemove />}
      </AttachmentPrimitive.Root>
      <TooltipContent side="top" className="tooltip-compact">
        <AttachmentPrimitive.Name />
      </TooltipContent>
    </Tooltip>
  );
};

const AttachmentUIWithoutReadyDocument: FC = () => {
  const rawAttachment = useAuiState(useShallow(({ attachment }) => attachment));

  if (
    isDocumentAttachmentState(rawAttachment) &&
    isReadyDocumentAttachment(rawAttachment)
  ) {
    return null;
  }

  return <AttachmentUI />;
};

const AttachmentRemove: FC = () => {
  return (
    <AttachmentPrimitive.Remove asChild={true}>
      <AttachmentChipRemoveButton
        tooltip="Remove file"
        className="aui-attachment-tile-remove"
      >
        <XIcon className="aui-attachment-remove-icon size-3 dark:stroke-[2.5px]" />
      </AttachmentChipRemoveButton>
    </AttachmentPrimitive.Remove>
  );
};

export const UserMessageAttachments: FC = () => {
  const attachments = useAuiState(({ message }) => message.attachments);
  const documentItems = useMemo(
    () => documentStackItemsFromAttachments(attachments),
    [attachments],
  );

  return (
    <div className="aui-user-message-attachments-end col-span-full col-start-1 row-start-1 flex w-full flex-row justify-end gap-2">
      <div className="flex max-w-full flex-row flex-wrap items-end justify-end gap-2">
        {documentItems.length > 0 ? (
          <DocumentStack items={documentItems} />
        ) : null}
        <MessagePrimitive.Attachments
          components={{ Attachment: AttachmentUIWithoutReadyDocument }}
        />
      </div>
    </div>
  );
};

export const ComposerAttachments: FC = () => {
  return (
    <div className="aui-composer-attachments mb-2 flex w-full flex-row items-end gap-2 overflow-x-auto px-1.5 pt-0.5 pb-1 empty:hidden">
      <ComposerPrimitive.Attachments components={{ Attachment: AttachmentUI }} />
    </div>
  );
};

export const ComposerAddAttachment: FC = () => {
  return (
    <ComposerPrimitive.AddAttachment asChild={true}>
      <TooltipIconButton
        tooltip="Add files"
        side="bottom"
        variant="ghost"
        size="icon"
        className="aui-composer-add-attachment size-8.5 rounded-full p-1 font-semibold text-xs hover:bg-muted-foreground/15 dark:border-muted-foreground/15 dark:hover:bg-muted-foreground/30"
        aria-label="Add files"
      >
        <PlusIcon className="aui-attachment-add-icon size-5 stroke-[1.5px]" />
      </TooltipIconButton>
    </ComposerPrimitive.AddAttachment>
  );
};
