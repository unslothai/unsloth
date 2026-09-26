// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

// Every attachment opens in the Library's viewer, from the composer and from a sent message:
// images zoom as they do there, documents show their pages, grid or slides, source files their
// highlighted code, web pages render, and clips play. Nothing is read until a viewer opens.

import {
  AttachmentDocumentDialog,
  AttachmentViewer,
} from "@/components/assistant-ui/attachment-document-dialog";
import {
  ATTACHMENT_PAGE_SCALES,
  attachmentViewerMeta,
} from "@/components/assistant-ui/attachment-viewer-meta";
import { AudioPlayer } from "@/components/assistant-ui/audio-player";
import { CodeToggleIcon } from "@/components/assistant-ui/code-toggle-icon";
import {
  type AttachmentSource,
  useAttachmentSource,
} from "@/components/assistant-ui/use-attachment-source";
import { CodeSourceView } from "@/components/code-source-view";
import { ScaleMenu } from "@/components/media-viewer";
import { Spinner } from "@/components/ui/spinner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  type AttachmentText,
  ArtifactHtmlFrame,
  attachmentAudioSrc,
  attachmentTextLanguage,
  countAttachmentTextLines,
  parseAttachmentText,
  readAttachmentText,
  truncateAttachmentPreviewText,
} from "@/features/chat";
import { useT } from "@/i18n";
import { MAX_HIGHLIGHT_CHARS } from "@/lib/markdown-plugins";
import { cn } from "@/lib/utils";
import { PlayIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type FC,
  type PropsWithChildren,
  type ReactNode,
  useEffect,
  useMemo,
  useState,
} from "react";

type TextPreviewState =
  | { status: "loading" }
  | { status: "error" }
  | ({ status: "ready" } & AttachmentText);

const WEB_PAGE_NAME = /\.(html?|xhtml)$/i;

/** A URL's bytes: an object URL for a composer file, a data URL for a sent one. */
const fetchBlob = (src: string): Promise<Blob> => fetch(src).then((response) => response.blob());

/** Scales text while still filling the pane, as the Library's viewer does. */
const Zoomed: FC<{ scale: number; children: ReactNode }> = ({ scale, children }) => (
  <div className="size-full overflow-hidden">
    <div
      className="origin-top-left"
      style={{
        width: `${100 / scale}%`,
        height: `${100 / scale}%`,
        transform: `scale(${scale})`,
      }}
    >
      {children}
    </div>
  </div>
);

const AttachmentImageDialog: FC<
  PropsWithChildren<{ source: AttachmentSource; src: string; redactFromReload?: boolean }>
> = ({ children, source, src, redactFromReload = false }) => {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);
  return (
    <AttachmentViewer
      trigger={children}
      open={open}
      onOpenChange={setOpen}
      source={source}
      meta={attachmentViewerMeta(source, source.file?.size)}
      media={!failed}
      noun="image"
      redactFromReload={redactFromReload}
      load={() => (source.file ? Promise.resolve(source.file) : fetchBlob(src))}
    >
      {failed ? (
        <p className="m-auto text-sm text-muted-foreground">{t("library.preview.cannotPreview")}</p>
      ) : (
        <img
          src={src}
          alt={source.name || "Image attachment"}
          onError={() => setFailed(true)}
          className="size-full object-contain"
        />
      )}
    </AttachmentViewer>
  );
};

// Extraction only starts once the viewer has been opened: parsing every PDF or
// spreadsheet in a thread up front would stall the composer.
const useAttachmentTextPreview = (
  enabled: boolean,
  file: File | undefined,
  name: string,
  contentType: string | undefined,
  text: string | undefined,
): TextPreviewState => {
  const [fileState, setFileState] = useState<TextPreviewState>({
    status: "loading",
  });
  // Unwrapping runs on open too: a thread can hold several large sent
  // attachments, and their payloads are scanned in full to strip the wrapper.
  const sentState = useMemo(
    (): TextPreviewState =>
      enabled
        ? { status: "ready", ...parseAttachmentText(text ?? "") }
        : { status: "loading" },
    [enabled, text],
  );

  useEffect(() => {
    if (!(enabled && file)) {
      return;
    }
    let active = true;
    readAttachmentText(file, name, contentType)
      .then((result) => {
        if (active) {
          setFileState({ status: "ready", ...result });
        }
      })
      .catch(() => {
        if (active) {
          setFileState({ status: "error" });
        }
      });
    return () => {
      active = false;
    };
  }, [enabled, file, name, contentType]);

  return file ? fileState : sentState;
};

/** The Library's two view buttons for a page that has a source: its code, or rendered. */
const ViewButton: FC<{
  label: string;
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}> = ({ label, active, onClick, children }) => (
  <Tooltip>
    <TooltipTrigger asChild={true}>
      <button
        type="button"
        aria-label={label}
        aria-pressed={active}
        onClick={onClick}
        className={cn(
          "flex size-9 shrink-0 items-center justify-center rounded-full outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring",
          active && "bg-muted",
        )}
      >
        {children}
      </button>
    </TooltipTrigger>
    <TooltipContent>{label}</TooltipContent>
  </Tooltip>
);

const AttachmentTextDialog: FC<
  PropsWithChildren<{ source: AttachmentSource; redactFromReload?: boolean }>
> = ({ children, source, redactFromReload = false }) => {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [opened, setOpened] = useState(false);
  const [scale, setScale] = useState(1);
  const [showCode, setShowCode] = useState(false);
  const state = useAttachmentTextPreview(
    opened,
    source.file,
    source.name,
    source.contentType,
    source.text,
  );
  const ready = state.status === "ready" ? state : null;
  const preview = useMemo(
    () => (ready ? truncateAttachmentPreviewText(ready.text) : undefined),
    [ready],
  );
  const truncated = Boolean(preview?.truncated || ready?.truncated);
  // Highlighting stops at the transcript's ceiling, so a long file still opens without tokenizing.
  const language = useMemo(() => {
    if (!ready || !preview || preview.text.length > MAX_HIGHLIGHT_CHARS) return null;
    return attachmentTextLanguage(source.name, ready.label);
  }, [ready, preview, source.name]);
  // A page's own HTML renders, as the Library shows it; text pulled out of a document never does.
  const webPage = Boolean(ready && !ready.label && !truncated && WEB_PAGE_NAME.test(source.name));
  const meta = useMemo(() => {
    if (state.status === "error") return "This file could not be read";
    if (!ready || !preview) return "Reading file";
    // Counting the capped text, not ready.text: a sent attachment keeps its
    // full payload in memory and splitting all of it would stall the webview.
    const lines = countAttachmentTextLines(preview.text);
    return attachmentViewerMeta(
      source,
      source.file?.size,
      `${lines} ${lines === 1 ? "line" : "lines"}`,
      ready.label && `text extracted from ${ready.label}`,
      truncated && "preview truncated",
    );
  }, [state.status, ready, preview, source, truncated]);
  // The file itself, or a sent text file's whole text. Text pulled out of a PDF is not the file.
  const file = source.file;
  const load = file
    ? () => Promise.resolve<Blob>(file)
    : ready && !ready.label
      ? () => Promise.resolve(new Blob([ready.text], { type: source.contentType || "text/plain" }))
      : undefined;

  let body: ReactNode;
  if (!ready) {
    body =
      state.status === "error" ? (
        <p className="m-auto text-sm text-muted-foreground">This file could not be read.</p>
      ) : (
        <Spinner className="m-auto size-6" />
      );
  } else if (!preview?.text.trim()) {
    body = (
      <p className="m-auto text-sm text-muted-foreground">
        {truncated
          ? "No readable text in the part of this file the preview reads."
          : "No readable text in this file."}
      </p>
    );
  } else if (webPage && !showCode) {
    body = (
      <Zoomed scale={scale}>
        <ArtifactHtmlFrame code={preview.text} title={source.name} fill />
      </Zoomed>
    );
  } else {
    const code = truncated ? `${preview.text}\n\n…` : preview.text;
    body = (
      <Zoomed scale={scale}>
        {language || webPage ? (
          <CodeSourceView code={code} language={language ?? "html"} className="px-6 pb-6" />
        ) : (
          <pre className="size-full overflow-auto whitespace-pre-wrap break-words px-6 pb-6 font-mono text-sm leading-relaxed select-text">
            {code}
          </pre>
        )}
      </Zoomed>
    );
  }

  return (
    <AttachmentViewer
      trigger={children}
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (next) setOpened(true);
      }}
      source={source}
      meta={meta}
      media={false}
      noun="file"
      redactFromReload={redactFromReload}
      load={load}
      extra={
        <>
          <ScaleMenu
            value={scale}
            scales={ATTACHMENT_PAGE_SCALES}
            onChange={(value) => setScale(Number(value))}
          />
          {webPage && (
            <div className="mr-1 flex items-center gap-1">
              <ViewButton label={t("library.preview.code")} active={showCode} onClick={() => setShowCode(true)}>
                <CodeToggleIcon className="size-4.5" />
              </ViewButton>
              <ViewButton
                label={t("library.preview.preview")}
                active={!showCode}
                onClick={() => setShowCode(false)}
              >
                <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} className="size-5" />
              </ViewButton>
            </div>
          )}
        </>
      }
    >
      {body}
    </AttachmentViewer>
  );
};

/**
 * The player, and the only place a sent clip's data URL is built for playback.
 *
 * Joining the header onto the base64 payload copies up to MAX_AUDIO_SIZE of
 * it, and a transcript mounts `useAttachmentSource` once per tile and again
 * per dialog, so the join waits until the viewer renders its body, which it
 * only does once open.
 */
const AttachmentAudioBody: FC<{ source: AttachmentSource }> = ({ source }) => {
  const src = useMemo(() => {
    if (source.src) {
      return source.src;
    }
    return source.audio
      ? attachmentAudioSrc(source.audio, source.contentType, source.name)
      : undefined;
  }, [source.src, source.audio, source.contentType, source.name]);

  return src ? (
    <div className="m-auto w-full max-w-lg px-6">
      <AudioPlayer src={src} filename={source.name || "attachment.wav"} />
    </div>
  ) : null;
};

const AttachmentAudioDialog: FC<
  PropsWithChildren<{ source: AttachmentSource; redactFromReload?: boolean }>
> = ({ children, source, redactFromReload = false }) => {
  const [open, setOpen] = useState(false);
  const { file, src, audio } = source;
  return (
    <AttachmentViewer
      trigger={children}
      open={open}
      onOpenChange={setOpen}
      source={source}
      meta={attachmentViewerMeta(source, file?.size)}
      media={false}
      noun="clip"
      redactFromReload={redactFromReload}
      // Built on click, like the player's: a sent clip is only base64 until someone asks for it.
      load={() =>
        file
          ? Promise.resolve(file)
          : fetchBlob(src ?? attachmentAudioSrc(audio!, source.contentType, source.name))
      }
    >
      <AttachmentAudioBody source={source} />
    </AttachmentViewer>
  );
};

export const AttachmentPreviewDialog: FC<
  /** Composer attachments are local and unsent, so keep them out of the reload snapshot. */
  PropsWithChildren<{ redactFromReload?: boolean }>
> = ({ children, redactFromReload = false }) => {
  const source = useAttachmentSource();

  if (source.kind === "image") {
    return source.src ? (
      <AttachmentImageDialog source={source} src={source.src} redactFromReload={redactFromReload}>
        {children}
      </AttachmentImageDialog>
    ) : (
      children
    );
  }

  if (source.kind === "audio") {
    return source.src || source.audio ? (
      <AttachmentAudioDialog source={source} redactFromReload={redactFromReload}>
        {children}
      </AttachmentAudioDialog>
    ) : (
      children
    );
  }

  if (source.kind === "document") {
    return (
      <AttachmentDocumentDialog source={source} redactFromReload={redactFromReload}>
        {children}
      </AttachmentDocumentDialog>
    );
  }

  if (!(source.file || source.text)) {
    return children;
  }

  return (
    <AttachmentTextDialog source={source} redactFromReload={redactFromReload}>
      {children}
    </AttachmentTextDialog>
  );
};
