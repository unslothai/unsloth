// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  ATTACHMENT_PAGE_SCALES,
  attachmentViewerMeta,
} from "@/components/assistant-ui/attachment-viewer-meta";
import type { AttachmentSource } from "@/components/assistant-ui/use-attachment-source";
import { DocumentView, documentKind } from "@/components/file-viewer";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { type MediaViewerActions, MediaViewer, ScaleMenu } from "@/components/media-viewer";
import { Spinner } from "@/components/ui/spinner";
import { attachmentBodyText, fetchChatAttachmentBlob, truncateAttachmentPreviewText } from "@/features/chat";
import { startLibraryChat } from "@/features/library";
import { useT } from "@/i18n";
import { MessageCircleIcon } from "@/lib/hugeicons-derived";
import { downloadFile } from "@/lib/native-files";
import { useAuiState } from "@assistant-ui/react";
import { useNavigate } from "@tanstack/react-router";
import { Slot } from "radix-ui";
import { toast } from "sonner";
import {
  type FC,
  type PropsWithChildren,
  type ReactNode,
  useEffect,
  useRef,
  useState,
} from "react";

const MARKDOWN_NAME = /\.(md|markdown|mdx)$/i;

/**
 * Opens an attachment in the Library's viewer, as a click on its tile, row or chip. The header
 * matches the Library's: name and subtitle, the page's own controls, "Chat about this" and a
 * download. Only a sent attachment offers the chat: an unsent one is already in the chat it would
 * open. `load` is read on click, so nothing is copied until the user asks for it.
 */
export const AttachmentViewer: FC<{
  trigger: ReactNode;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  source: Pick<AttachmentSource, "name" | "contentType">;
  meta: string;
  media: boolean;
  noun: "image" | "clip" | "file";
  redactFromReload: boolean;
  /** The attachment's bytes, for the download and the chat. Unset while they are not ready. */
  load?: () => Promise<Blob>;
  flush?: boolean;
  extra?: ReactNode;
  children: ReactNode;
}> = ({
  trigger,
  open,
  onOpenChange,
  source,
  meta,
  media,
  noun,
  redactFromReload,
  load,
  flush = true,
  extra,
  children,
}) => {
  const t = useT();
  const navigate = useNavigate();
  // Mounted on first open and kept for its close animation. A transcript draws one of these per
  // attachment, and the viewer's project menu subscribes to and refetches the project list.
  const [mounted, setMounted] = useState(open);
  if (open && !mounted) setMounted(true);
  const name = source.name || "attachment";
  // A sent original is fetched on click and can fail; say so rather than doing nothing.
  const actions: MediaViewerActions = {
    primary:
      load && !redactFromReload
        ? {
            label: t("library.menu.chatAboutThis"),
            icon: MessageCircleIcon,
            onClick: () =>
              void load()
                .then((blob) => {
                  const file = new File([blob], name, { type: source.contentType || blob.type });
                  onOpenChange(false);
                  startLibraryChat(navigate, { files: [file] });
                })
                .catch(() => toast.error(`Could not read ${name}`)),
          }
        : undefined,
    onDownload: load
      ? () =>
          void load()
            .then((blob) => downloadFile(blob, name, source.contentType || undefined))
            .catch(() => toast.error(t("library.toast.downloadFailed", { name })))
      : undefined,
  };
  return (
    <>
      <Slot.Root
        onClick={() => onOpenChange(true)}
        className="aui-attachment-preview-trigger cursor-pointer transition-colors hover:bg-accent/50"
      >
        {trigger}
      </Slot.Root>
      {mounted && (
        <MediaViewer
          open={open}
          onOpenChange={onOpenChange}
          title={source.name}
          meta={meta}
          media={media}
          noun={noun}
          flush={flush}
          redactFromReload={redactFromReload}
          extra={extra}
          actions={actions}
        >
          {open && children}
        </MediaViewer>
      )}
    </>
  );
};

/** `plain`: a sent document's stored text, shown when its original file is gone. `text` and `plain`
 *  are capped for rendering (`truncated`); `blob`, which a download saves, is whole. */
type Loaded = { blob?: Blob; text?: string; plain?: string; truncated?: boolean; error?: boolean };

/** Rendered markdown, or the document's pages, grid or slides, at `scale`. */
const DocumentBody: FC<{ name: string; contentType?: string; loaded: Loaded; scale: number }> = ({
  name,
  contentType,
  loaded,
  scale,
}) => {
  const t = useT();
  if (loaded.error) {
    return <p className="m-auto text-sm text-muted-foreground">{t("library.preview.cannotPreview")}</p>;
  }
  if (loaded.plain !== undefined) {
    return (
      <pre className="size-full overflow-auto whitespace-pre-wrap px-6 py-4 font-sans text-ui-14 select-text">
        {loaded.plain}
      </pre>
    );
  }
  if (loaded.text !== undefined) {
    return (
      <div className="size-full overflow-auto px-6">
        <div className="mx-auto max-w-3xl" style={{ zoom: scale }}>
          <MarkdownPreview
            markdown={loaded.text}
            defer={true}
            className="max-h-none select-text overflow-visible border-0 bg-transparent px-2 py-4 text-ui-15p5"
          />
        </div>
      </div>
    );
  }
  const kind = documentKind(name, contentType);
  if (!loaded.blob || !kind) return <Spinner className="m-auto size-6" />;
  return <DocumentView file={loaded.blob} kind={kind} name={name} scale={scale} />;
};

/** Opens a document attachment in the Library's viewer. `load` returns its bytes. */
const DocumentDialog: FC<
  PropsWithChildren<{
    source: AttachmentSource;
    load: () => Promise<Blob>;
    redactFromReload: boolean;
    /** A text response is the stored text: the server serves that when the original is gone. */
    textFallback?: boolean;
  }>
> = ({ children, source, load, redactFromReload, textFallback = false }) => {
  const [open, setOpen] = useState(false);
  const [scale, setScale] = useState(1);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const markdown = MARKDOWN_NAME.test(source.name);
  // Read on open, so a re-render (e.g. streaming) does not restart the load.
  const loadRef = useRef(load);
  useEffect(() => {
    loadRef.current = load;
  });

  useEffect(() => {
    if (!open || loaded) return;
    let cancelled = false;
    loadRef
      .current()
      .then(async (blob) => {
        let next: Loaded = { blob };
        if (markdown) {
          const { text, truncated } = truncateAttachmentPreviewText(await blob.text());
          next = { blob, text, truncated };
        } else if (textFallback && blob.type.startsWith("text/")) {
          const { text, truncated } = truncateAttachmentPreviewText(await blob.text());
          next = { plain: text, truncated };
        }
        if (!cancelled) setLoaded(next);
      })
      .catch(() => !cancelled && setLoaded({ error: true }));
    return () => {
      cancelled = true;
    };
  }, [open, loaded, markdown, textFallback]);

  const blob = loaded?.blob;
  return (
    <AttachmentViewer
      trigger={children}
      open={open}
      onOpenChange={setOpen}
      source={source}
      meta={attachmentViewerMeta(source, blob?.size, loaded?.truncated && "preview truncated")}
      media={false}
      noun="file"
      redactFromReload={redactFromReload}
      load={blob ? () => Promise.resolve(blob) : undefined}
      extra={
        <ScaleMenu
          value={scale}
          scales={ATTACHMENT_PAGE_SCALES}
          onChange={(value) => setScale(Number(value))}
        />
      }
    >
      <DocumentBody name={source.name} contentType={source.contentType} loaded={loaded ?? {}} scale={scale} />
    </AttachmentViewer>
  );
};

/** A sent document's kept original, fetched by message id. Sent attachments only. */
const SentOriginalDialog: FC<
  PropsWithChildren<{ source: AttachmentSource; redactFromReload: boolean }>
> = ({ children, source, redactFromReload }) => {
  const messageId = useAuiState(({ message }) => message.id);
  const attachmentId = useAuiState(({ attachment }) => attachment.id);
  return (
    <DocumentDialog
      source={source}
      load={() => fetchChatAttachmentBlob(messageId, attachmentId)}
      redactFromReload={redactFromReload}
      textFallback={true}
    >
      {children}
    </DocumentDialog>
  );
};

export const AttachmentDocumentDialog: FC<
  PropsWithChildren<{ source: AttachmentSource; redactFromReload: boolean }>
> = ({ children, source, redactFromReload }) => {
  const { file, text } = source;
  if (file) {
    return (
      <DocumentDialog source={source} load={() => Promise.resolve(file)} redactFromReload={redactFromReload}>
        {children}
      </DocumentDialog>
    );
  }
  // A sent CSV or note keeps the whole file as its text, inside the attachment's wrapper.
  if (!source.hasOriginal && text !== undefined) {
    return (
      <DocumentDialog
        source={source}
        load={() => Promise.resolve(new Blob([attachmentBodyText(text)]))}
        redactFromReload={redactFromReload}
      >
        {children}
      </DocumentDialog>
    );
  }
  return (
    <SentOriginalDialog source={source} redactFromReload={redactFromReload}>
      {children}
    </SentOriginalDialog>
  );
};
