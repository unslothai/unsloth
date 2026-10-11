// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import {
  ATTACHMENT_PAGE_SCALES,
  attachmentViewerMeta,
} from "@/components/assistant-ui/attachment-viewer-meta";
import type { AttachmentSource } from "@/components/assistant-ui/use-attachment-source";
import { AttachmentBrowserOpenContext } from "@/components/assistant-ui/attachment-browser-open-context";
import { filesOpenInBrowser } from "@/features/browser";
import { DocumentView, documentKind, isMarkdown } from "@/components/file-viewer";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import {
  type MediaViewerActions,
  type MediaViewerGallery,
  MediaViewer,
  ScaleMenu,
} from "@/components/media-viewer";
import { Spinner } from "@/components/ui/spinner";
import {
  attachmentBodyText,
  fetchChatAttachmentBlob,
  readAttachmentText,
  truncateAttachmentPreviewText,
} from "@/features/chat";
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
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

/** Opens an attachment in the Library's viewer; `load` is read on click, so nothing is copied until asked. */
export const AttachmentViewer: FC<{
  trigger: ReactNode;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  source: Pick<AttachmentSource, "name" | "contentType">;
  meta: string;
  media: boolean;
  noun: "image" | "video" | "clip" | "file";
  redactFromReload: boolean;
  load?: () => Promise<Blob>;
  saveAs?: { name: string; contentType: string };
  flush?: boolean;
  extra?: ReactNode;
  /** The Library's own actions on the attachment, for one it lists. */
  libraryActions?: Omit<MediaViewerActions, "primary" | "onDownload">;
  variant?: "card" | "lightbox";
  gallery?: MediaViewerGallery;
  itemKey?: string;
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
  saveAs,
  flush = true,
  extra,
  libraryActions,
  variant,
  gallery,
  itemKey,
  children,
}) => {
  const t = useT();
  const navigate = useNavigate();
  const openInBrowser = useContext(AttachmentBrowserOpenContext);
  // Mounted on first open: every mounted viewer's project menu refetches the project list.
  const [mounted, setMounted] = useState(open);
  if (open && !mounted) setMounted(true);
  const name = saveAs?.name ?? (source.name || "attachment");
  const contentType = saveAs?.contentType ?? source.contentType;
  const actions: MediaViewerActions = {
    primary:
      load && !redactFromReload
        ? {
            label: t("library.menu.chatAboutThis"),
            icon: MessageCircleIcon,
            onClick: () =>
              void load()
                .then((blob) => {
                  const file = new File([blob], name, { type: contentType || blob.type });
                  onOpenChange(false);
                  startLibraryChat(navigate, { files: [file] });
                })
                .catch(() => toast.error(`Could not read ${name}`)),
          }
        : undefined,
    onDownload: load
      ? () =>
          void load()
            .then((blob) => downloadFile(blob, name, contentType || undefined))
            .catch(() => toast.error(t("library.toast.downloadFailed", { name })))
      : undefined,
    ...libraryActions,
  };
  return (
    <>
      <Slot.Root
        onClick={() => {
          if (openInBrowser && filesOpenInBrowser()) {
            openInBrowser();
            return;
          }
          onOpenChange(true);
        }}
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
          variant={variant}
          gallery={gallery}
          itemKey={itemKey}
        >
          {open && children}
        </MediaViewer>
      )}
    </>
  );
};

type Loaded = { blob?: Blob; text?: string; plain?: string; truncated?: boolean; error?: boolean };

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
  return <DocumentView file={loaded.blob} kind={kind} name={name} contentType={contentType} scale={scale} />;
};

const DocumentDialog: FC<
  PropsWithChildren<{
    source: AttachmentSource;
    load: () => Promise<Blob>;
    redactFromReload: boolean;
    textFallback?: boolean;
  }>
> = ({ children, source, load, redactFromReload, textFallback = false }) => {
  const [open, setOpen] = useState(false);
  const [scale, setScale] = useState(1);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const markdown = isMarkdown(source.name, source.contentType ?? "");
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
          // A composer File decodes as its adapter read it (BOM, UTF-16); Blob.text() is UTF-8 only.
          const { text, truncated } =
            blob instanceof File
              ? await readAttachmentText(blob, source.name, source.contentType)
              : truncateAttachmentPreviewText(await blob.text());
          next = { blob, text, truncated };
        } else if (textFallback && blob.type.startsWith("text/")) {
          // A File decodes as the composer does (BOM, UTF-16); a code page it refuses still shows raw.
          const { text, truncated } =
            blob instanceof File
              ? await readAttachmentText(blob, source.name, blob.type).catch(async () =>
                  truncateAttachmentPreviewText(await blob.text()),
                )
              : truncateAttachmentPreviewText(await blob.text());
          next = { blob, plain: text, truncated };
        }
        if (!cancelled) setLoaded(next);
      })
      .catch(() => !cancelled && setLoaded({ error: true }));
    return () => {
      cancelled = true;
    };
  }, [open, loaded, markdown, textFallback, source.name, source.contentType]);

  const blob = loaded?.blob;
  return (
    <AttachmentViewer
      trigger={children}
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (!next) setLoaded(null);
      }}
      source={source}
      meta={attachmentViewerMeta(source, blob?.size, loaded?.truncated && "preview truncated")}
      media={false}
      noun="file"
      redactFromReload={redactFromReload}
      load={blob ? () => Promise.resolve(blob) : undefined}
      saveAs={
        loaded?.plain !== undefined
          ? { name: `${source.name.replace(/\.[^.]+$/, "")}.txt`, contentType: "text/plain" }
          : undefined
      }
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

/** A file outside any message, such as a chat with files source, read on open: pages, markdown or
 *  text (a text/* blob). */
export const LocalFileDialog: FC<PropsWithChildren<{ name: string; load: () => Promise<Blob> }>> = ({
  children,
  name,
  load,
}) => (
  <DocumentDialog
    source={{
      kind: "document",
      name,
      contentType: undefined,
      file: undefined,
      src: undefined,
      audio: undefined,
      video: undefined,
      text: undefined,
      hasOriginal: true,
    }}
    load={load}
    redactFromReload={false}
    textFallback={true}
  >
    {children}
  </DocumentDialog>
);

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
