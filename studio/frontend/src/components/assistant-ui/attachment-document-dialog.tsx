// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import type { AttachmentSource } from "@/components/assistant-ui/use-attachment-source";
import { DocumentView, documentKind } from "@/components/file-viewer";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { MediaViewer, ScaleMenu } from "@/components/media-viewer";
import { Spinner } from "@/components/ui/spinner";
import { fetchChatAttachmentBlob, parseAttachmentText } from "@/features/chat";
import { formatBytes } from "@/features/hub";
import { useT } from "@/i18n";
import { downloadFile } from "@/lib/native-files";
import { useAuiState } from "@assistant-ui/react";
import { Slot } from "radix-ui";
import { type FC, type PropsWithChildren, useEffect, useRef, useState } from "react";

const SCALES = [0.5, 0.75, 1, 1.25, 1.5, 2];
const MARKDOWN_NAME = /\.(md|markdown|mdx)$/i;

/** `plain`: a sent document's stored text, shown when its original file is gone. */
type Loaded = { blob?: Blob; text?: string; plain?: string; error?: boolean };

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
        const next: Loaded = markdown
          ? { blob, text: await blob.text() }
          : textFallback && blob.type.startsWith("text/")
            ? { plain: await blob.text() }
            : { blob };
        if (!cancelled) setLoaded(next);
      })
      .catch(() => !cancelled && setLoaded({ error: true }));
    return () => {
      cancelled = true;
    };
  }, [open, loaded, markdown, textFallback]);

  const blob = loaded?.blob;
  const meta = [source.name.split(".").pop()?.toUpperCase(), blob ? formatBytes(blob.size) : null]
    .filter(Boolean)
    .join(" · ");

  return (
    <>
      <Slot.Root
        onClick={() => setOpen(true)}
        className="aui-attachment-preview-trigger cursor-pointer transition-colors hover:bg-accent/50"
      >
        {children}
      </Slot.Root>
      <MediaViewer
        open={open}
        onOpenChange={setOpen}
        title={source.name}
        meta={meta}
        media={false}
        noun="file"
        flush={true}
        redactFromReload={redactFromReload}
        extra={
          <ScaleMenu value={scale} scales={SCALES} onChange={(value) => setScale(Number(value))} />
        }
        actions={{
          onDownload: blob
            ? () => void downloadFile(blob, source.name, source.contentType || undefined)
            : undefined,
        }}
      >
        {open && (
          <DocumentBody
            name={source.name}
            contentType={source.contentType}
            loaded={loaded ?? {}}
            scale={scale}
          />
        )}
      </MediaViewer>
    </>
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
        load={() => Promise.resolve(new Blob([parseAttachmentText(text).text]))}
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
