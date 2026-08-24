// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";

import { AudioPlayer } from "@/components/assistant-ui/audio-player";
import {
  type AttachmentSource,
  useAttachmentSource,
} from "@/components/assistant-ui/use-attachment-source";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import {
  type AttachmentText,
  attachmentAudioSrc,
  attachmentTextLanguage,
  countAttachmentTextLines,
  parseAttachmentText,
  readAttachmentText,
  truncateAttachmentPreviewText,
} from "@/features/chat";
import { formatBytes } from "@/features/hub";
import { MAX_HIGHLIGHT_CHARS, codeFence } from "@/lib/markdown-plugins";
import { cn } from "@/lib/utils";
import { code as codePlugin } from "@streamdown/code";
import {
  type FC,
  type PropsWithChildren,
  useEffect,
  useMemo,
  useState,
} from "react";
import { Streamdown } from "streamdown";

type TextPreviewState =
  | { status: "loading" }
  | { status: "error" }
  | ({ status: "ready" } & AttachmentText);

const SHIKI_THEME = ["github-light", "github-dark"] as [
  "github-light",
  "github-dark",
];

const AttachmentImage: FC<{ src: string }> = ({ src }) => {
  const [isLoaded, setIsLoaded] = useState(false);
  return (
    <img
      src={src}
      alt="Preview"
      className={cn(
        "block h-auto max-h-[90dvh] w-auto max-w-[92vw] object-contain",
        isLoaded
          ? "aui-attachment-preview-image-loaded"
          : "aui-attachment-preview-image-loading invisible",
      )}
      onLoad={() => setIsLoaded(true)}
    />
  );
};

const AttachmentPreviewTrigger: FC<PropsWithChildren> = ({ children }) => {
  return (
    <DialogTrigger
      className="aui-attachment-preview-trigger cursor-pointer transition-colors hover:bg-accent/50"
      asChild={true}
    >
      {children}
    </DialogTrigger>
  );
};

const AttachmentImageDialog: FC<
  PropsWithChildren<{ src: string; redactFromReload?: boolean }>
> = ({ children, src, redactFromReload = false }) => {
  return (
    <Dialog>
      <AttachmentPreviewTrigger>{children}</AttachmentPreviewTrigger>
      {/* Chrome-free lightbox: the image floats on the dimmed backdrop with
          no dialog panel, and the close button sits in the screen corner. */}
      <DialogContent
        data-reload-snapshot-sensitive={redactFromReload ? "" : undefined}
        overlayClassName="bg-black/70"
        className="aui-attachment-preview-dialog-content top-0 left-0 grid h-dvh w-screen max-h-none max-w-none translate-x-0 translate-y-0 place-items-center overflow-hidden rounded-none border-0 bg-transparent p-0 shadow-none ring-0 sm:max-w-none [&>button]:fixed [&>button]:top-4 [&>button]:right-4 [&>button]:z-20 [&>button]:size-9 [&>button]:rounded-full [&>button]:bg-transparent [&>button]:text-white [&>button]:opacity-100 [&>button]:ring-0! [&>button]:hover:bg-white/25 [&>button]:hover:text-white [&_svg]:text-white"
      >
        <DialogTitle className="aui-sr-only sr-only">
          Image Attachment Preview
        </DialogTitle>
        {/* Clicking the backdrop (anywhere off the image) closes the preview. */}
        <DialogClose asChild={true}>
          <div aria-hidden="true" className="absolute inset-0" />
        </DialogClose>
        <div className="aui-attachment-preview pointer-events-none relative z-10 flex items-center justify-center">
          <span className="pointer-events-auto">
            <AttachmentImage src={src} />
          </span>
        </div>
      </DialogContent>
    </Dialog>
  );
};

// Extraction only starts once the dialog has been opened: parsing every PDF or
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

/**
 * The preview body: a source file keeps its syntax colours, prose stays plain.
 *
 * Highlighting is skipped past MAX_HIGHLIGHT_CHARS, the same ceiling the
 * transcript uses, so a long file still opens without tokenizing on the way in.
 */
const AttachmentTextBody: FC<{ text: string; language: string | null }> = ({
  text,
  language,
}) => {
  const markdown = useMemo(() => {
    if (!language || text.length > MAX_HIGHLIGHT_CHARS) {
      return null;
    }
    const fence = codeFence(text);
    return `${fence}${language}\n${text}\n${fence}`;
  }, [text, language]);

  if (!markdown) {
    return (
      <pre className="whitespace-pre-wrap break-words px-4 py-3 font-mono text-xs leading-relaxed">
        {text}
      </pre>
    );
  }

  // streamdown draws its own card and language header, flattened here so the dialog panel is the only one
  return (
    <div className="[&_[data-streamdown=code-block-header]]:!hidden [&_[data-streamdown=code-block]]:!my-0 [&_[data-streamdown=code-block]]:!gap-0 [&_[data-streamdown=code-block]]:!rounded-none [&_[data-streamdown=code-block]]:!border-0 [&_[data-streamdown=code-block]]:!bg-transparent [&_[data-streamdown=code-block]]:!p-0 [&_[data-streamdown=code-block-body]]:!rounded-none [&_[data-streamdown=code-block-body]]:!border-0 [&_[data-streamdown=code-block-body]]:!bg-transparent [&_[data-streamdown=code-block-body]]:!p-0 [&_pre]:!m-0 [&_pre]:!bg-transparent [&_pre]:!px-4 [&_pre]:!py-3 [&_pre]:!text-xs [&_pre]:!leading-relaxed">
      <Streamdown
        mode="static"
        plugins={{ code: codePlugin }}
        controls={{ code: false }}
        shikiTheme={SHIKI_THEME}
      >
        {markdown}
      </Streamdown>
    </div>
  );
};

const AttachmentTextDialog: FC<
  PropsWithChildren<{ source: AttachmentSource; redactFromReload?: boolean }>
> = ({ children, source, redactFromReload = false }) => {
  const [opened, setOpened] = useState(false);
  const state = useAttachmentTextPreview(
    opened,
    source.file,
    source.name,
    source.contentType,
    source.text,
  );
  const preview = useMemo(
    () =>
      state.status === "ready"
        ? truncateAttachmentPreviewText(state.text)
        : undefined,
    [state],
  );
  const language = useMemo(
    () =>
      state.status === "ready"
        ? attachmentTextLanguage(source.name, state.label)
        : null,
    [state, source.name],
  );
  const meta = useMemo(() => {
    if (state.status === "error") {
      return "This file could not be read";
    }
    if (state.status !== "ready" || !preview) {
      return "Reading file";
    }
    // Counting the capped text, not state.text: a sent attachment keeps its
    // full payload in memory and splitting all of it would stall the webview.
    const lines = countAttachmentTextLines(preview.text);
    return [
      source.file ? formatBytes(source.file.size) : null,
      `${lines} ${lines === 1 ? "line" : "lines"}`,
      state.label ? `text extracted from ${state.label}` : null,
      preview.truncated || state.truncated ? "preview truncated" : null,
    ]
      .filter(Boolean)
      .join(" · ");
  }, [state, preview, source.file]);

  return (
    <Dialog
      onOpenChange={(isOpen) => {
        if (isOpen) {
          setOpened(true);
        }
      }}
    >
      <AttachmentPreviewTrigger>{children}</AttachmentPreviewTrigger>
      <DialogContent
        data-reload-snapshot-sensitive={redactFromReload ? "" : undefined}
        className="aui-attachment-text-dialog-content gap-4 sm:max-w-2xl"
      >
        <DialogHeader className="gap-1 pr-10">
          <DialogTitle className="truncate">
            {source.name || "Attachment"}
          </DialogTitle>
          <DialogDescription className="text-xs">{meta}</DialogDescription>
        </DialogHeader>
        <div className="overflow-hidden rounded-2xl border bg-muted/40">
          {state.status === "loading" ? (
            <div className="flex h-32 items-center justify-center">
              <Spinner className="size-5 text-muted-foreground" />
            </div>
          ) : preview?.text.trim() ? (
            <div className="overlay-scrollbar-gutter max-h-[60dvh] overflow-y-auto">
              <AttachmentTextBody text={preview.text} language={language} />
            </div>
          ) : (
            <div className="px-4 py-6 text-muted-foreground text-sm">
              {state.status === "error"
                ? "This file could not be read."
                : state.status === "ready" && state.truncated
                  ? "No readable text in the part of this file the preview reads."
                  : "No readable text in this file."}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

/**
 * The player, and the only place a sent clip's data URL is built.
 *
 * Joining the header onto the base64 payload copies up to MAX_AUDIO_SIZE of
 * it, and a transcript mounts `useAttachmentSource` once per tile and again
 * per dialog, so the join waits until DialogContent renders, which Radix only
 * does once the dialog is open.
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
    <AudioPlayer src={src} filename={source.name || "attachment.wav"} />
  ) : null;
};

const AttachmentAudioDialog: FC<
  PropsWithChildren<{ source: AttachmentSource; redactFromReload?: boolean }>
> = ({ children, source, redactFromReload = false }) => {
  return (
    <Dialog>
      <AttachmentPreviewTrigger>{children}</AttachmentPreviewTrigger>
      <DialogContent
        data-reload-snapshot-sensitive={redactFromReload ? "" : undefined}
        className="aui-attachment-audio-dialog-content gap-4 sm:max-w-lg"
      >
        <DialogHeader className="gap-1 pr-10">
          <DialogTitle className="truncate">
            {source.name || "Audio attachment"}
          </DialogTitle>
          <DialogDescription className="text-xs">
            {[
              source.file ? formatBytes(source.file.size) : null,
              source.contentType || "audio",
            ]
              .filter(Boolean)
              .join(" · ")}
          </DialogDescription>
        </DialogHeader>
        <AttachmentAudioBody source={source} />
      </DialogContent>
    </Dialog>
  );
};

export const AttachmentPreviewDialog: FC<
  /** Composer attachments are local and unsent, so keep them out of the reload snapshot. */
  PropsWithChildren<{ redactFromReload?: boolean }>
> = ({ children, redactFromReload = false }) => {
  const source = useAttachmentSource();

  if (source.kind === "image") {
    return source.src ? (
      <AttachmentImageDialog src={source.src} redactFromReload={redactFromReload}>
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

  if (!(source.file || source.text)) {
    return children;
  }

  return (
    <AttachmentTextDialog source={source} redactFromReload={redactFromReload}>
      {children}
    </AttachmentTextDialog>
  );
};
