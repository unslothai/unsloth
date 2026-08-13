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
  countAttachmentTextLines,
  parseAttachmentText,
  readAttachmentText,
  truncateAttachmentPreviewText,
} from "@/features/chat";
import { formatBytes } from "@/features/hub";
import { cn } from "@/lib/utils";
import {
  type FC,
  type PropsWithChildren,
  useEffect,
  useMemo,
  useState,
} from "react";

type TextPreviewState =
  | { status: "loading" }
  | { status: "error" }
  | ({ status: "ready" } & AttachmentText);

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

const AttachmentImageDialog: FC<PropsWithChildren<{ src: string }>> = ({
  children,
  src,
}) => {
  return (
    <Dialog>
      <AttachmentPreviewTrigger>{children}</AttachmentPreviewTrigger>
      {/* Chrome-free lightbox: the image floats on the dimmed backdrop with
          no dialog panel, and the close button sits in the screen corner. */}
      <DialogContent
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

const AttachmentTextDialog: FC<
  PropsWithChildren<{ source: AttachmentSource }>
> = ({ children, source }) => {
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
      <DialogContent className="aui-attachment-text-dialog-content gap-4 sm:max-w-2xl">
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
              <pre className="whitespace-pre-wrap break-words px-4 py-3 font-mono text-xs leading-relaxed">
                {preview.text}
              </pre>
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

const AttachmentAudioDialog: FC<
  PropsWithChildren<{ source: AttachmentSource; src: string }>
> = ({ children, source, src }) => {
  return (
    <Dialog>
      <AttachmentPreviewTrigger>{children}</AttachmentPreviewTrigger>
      <DialogContent className="aui-attachment-audio-dialog-content gap-4 sm:max-w-lg">
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
        <AudioPlayer src={src} filename={source.name || "attachment.wav"} />
      </DialogContent>
    </Dialog>
  );
};

export const AttachmentPreviewDialog: FC<PropsWithChildren> = ({
  children,
}) => {
  const source = useAttachmentSource();

  if (source.kind === "image") {
    return source.src ? (
      <AttachmentImageDialog src={source.src}>{children}</AttachmentImageDialog>
    ) : (
      children
    );
  }

  if (source.kind === "audio") {
    return source.src ? (
      <AttachmentAudioDialog source={source} src={source.src}>
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
    <AttachmentTextDialog source={source}>{children}</AttachmentTextDialog>
  );
};
