// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"use client";


import {
  AttachmentBrowserOpenProvider,
  AttachmentFileContextMenu,
} from "@/components/assistant-ui/attachment-browser-open";
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
  type AttachmentVideoPart,
  selectAttachmentSource,
} from "@/components/assistant-ui/attachment-selection";
import {
  type AttachmentSource,
  useAttachmentSource,
} from "@/components/assistant-ui/use-attachment-source";
import { CodeSourceView } from "@/components/code-source-view";
import { type MediaViewerActions, ScaleMenu } from "@/components/media-viewer";
import { Spinner } from "@/components/ui/spinner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import {
  type AttachmentText,
  ArtifactHtmlFrame,
  attachmentAudioSrc,
  attachmentBodyText,
  attachmentTextLanguage,
  countAttachmentTextLines,
  parseAttachmentText,
  readAttachmentText,
  truncateAttachmentPreviewText,
} from "@/features/chat";
import {
  ConfirmDeleteDialog,
  addLibraryItemToProject,
  uploadLibraryFiles,
  useLibraryFavorite,
  useLibraryFavoritesStore,
} from "@/features/library";
import { useT } from "@/i18n";
import { MAX_HIGHLIGHT_CHARS } from "@/lib/markdown-plugins";
import { asPng } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { useAui, useAuiState } from "@assistant-ui/react";
import { PlayIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  type FC,
  type PropsWithChildren,
  type ReactNode,
  isValidElement,
  useEffect,
  useMemo,
  useState,
} from "react";

type TextPreviewState =
  | { status: "loading" }
  | { status: "error" }
  | ({ status: "ready" } & AttachmentText);

const WEB_PAGE_NAME = /\.(html?|xhtml)$/i;

const fetchBlob = (src: string): Promise<Blob> => fetch(src).then((response) => response.blob());

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

type ImageActions = Omit<MediaViewerActions, "primary" | "onDownload">;

type GalleryImage = {
  id: string;
  name: string;
  contentType: string | undefined;
  file: File | undefined;
  image: string | undefined;
};

type AttachmentState = Parameters<typeof selectAttachmentSource>[0]["attachment"] & { id: string };

const galleryImagesOf = (attachments: readonly unknown[] | undefined): GalleryImage[] =>
  (attachments ?? []).flatMap((attachment) => {
    const state = attachment as AttachmentState;
    const source = selectAttachmentSource({ attachment: state });
    if (source.kind !== "image" || !(source.file || source.image)) return [];
    return [
      {
        id: state.id,
        name: source.name,
        contentType: source.contentType,
        file: source.file,
        image: source.image,
      },
    ];
  });

const useObjectUrl = (file: File | undefined): string | undefined => {
  const [url, setUrl] = useState<{ file: File; url: string } | null>(null);
  useEffect(() => {
    if (!file) return;
    const next = URL.createObjectURL(file);
    setUrl({ file, url: next });
    return () => URL.revokeObjectURL(next);
  }, [file]);
  return file && url?.file === file ? url.url : undefined;
};

const loadGalleryImage = (image: GalleryImage): Promise<Blob> =>
  image.file ? Promise.resolve(image.file) : fetchBlob(image.image ?? "");

const copyImage = (image: GalleryImage, t: ReturnType<typeof useT>): void => {
  if (typeof ClipboardItem === "undefined" || !navigator.clipboard?.write) {
    toast.error(t("imageViewer.copyFailed"));
    return;
  }
  // The item is made now, inside the click, and resolves later: Safari refuses a write made after.
  navigator.clipboard
    .write([new ClipboardItem({ "image/png": loadGalleryImage(image).then(asPng) })])
    .then(
      () => toast.success(t("imageViewer.copied")),
      () => toast.error(t("imageViewer.copyFailed")),
    );
};

 /** The full-window viewer for an image attachment, with arrows to the images beside it. */
const ImageGalleryDialog: FC<
  PropsWithChildren<{
    owner: GalleryImage;
    images: GalleryImage[];
    open: boolean;
    onOpenChange: (open: boolean) => void;
    shownId: string;
    onShow: (id: string) => void;
    redactFromReload: boolean;
    actionsFor: (image: GalleryImage) => ImageActions;
  }>
> = ({ children, owner, images, open, onOpenChange, shownId, onShow, redactFromReload, actionsFor }) => {
  const t = useT();
  const [failedId, setFailedId] = useState<string | null>(null);
  const index = images.findIndex((image) => image.id === shownId);
  const current = images[index] ?? owner;
  const objectUrl = useObjectUrl(open ? current.file : undefined);
  const src = current.file ? objectUrl : current.image;
  const failed = failedId === current.id;
  const previous = index > 0 ? images[index - 1] : undefined;
  const next = index >= 0 && index < images.length - 1 ? images[index + 1] : undefined;
  return (
    <AttachmentViewer
      trigger={children}
      open={open}
      onOpenChange={onOpenChange}
      source={current}
      meta={attachmentViewerMeta(current, current.file?.size)}
      media={!failed}
      noun="image"
      redactFromReload={redactFromReload}
      load={() => loadGalleryImage(current)}
      libraryActions={{
        copy: { label: t("imageViewer.copy"), onClick: () => copyImage(current, t) },
        ...actionsFor(current),
      }}
      variant="lightbox"
      itemKey={current.id}
      gallery={
        images.length > 1
          ? {
              onPrevious: previous ? () => onShow(previous.id) : undefined,
              onNext: next ? () => onShow(next.id) : undefined,
            }
          : undefined
      }
    >
      {failed ? (
        <p className="m-auto text-sm text-muted-foreground">{t("library.preview.cannotPreview")}</p>
      ) : src ? (
        <img
          key={current.id}
          src={src}
          alt={current.name || "Image attachment"}
          onError={() => setFailedId(current.id)}
          className="size-full object-contain shadow-[0_1px_10px_rgba(0,0,0,0.08)] dark:shadow-[0_1px_10px_rgba(0,0,0,0.3)]"
        />
      ) : (
        <Spinner className="m-auto size-6" />
      )}
    </AttachmentViewer>
  );
};

const useGalleryState = (ownerId: string) => {
  const [open, setOpen] = useState(false);
  const [shownId, setShownId] = useState(ownerId);
  const onOpenChange = (next: boolean) => {
    if (next) setShownId(ownerId);
    setOpen(next);
  };
  return { open, onOpenChange, shownId, onShow: setShownId };
};

const neighbourOf = (images: GalleryImage[], id: string): GalleryImage | undefined => {
  const index = images.findIndex((image) => image.id === id);
  return images[index + 1] ?? images[index - 1];
};

// Composer images saved to the Library, so a second star or project action reuses that copy.
const savedComposerImages = new WeakMap<File, Promise<string>>();

const saveComposerImage = (image: GalleryImage): Promise<string> => {
  const file = image.file;
  if (!file) return Promise.reject(new Error("This image has no file to save."));
  let saved = savedComposerImages.get(file);
  if (!saved) {
    saved = uploadLibraryFiles({ files: [file] }, null).then(([id]) => {
      if (!id) throw new Error("The Library didn't keep the file.");
      return id;
    });
    savedComposerImages.set(file, saved);
    saved.catch(() => savedComposerImages.delete(file));
  }
  return saved;
};

const ComposerImageDialog: FC<PropsWithChildren<{ source: AttachmentSource; src: string }>> = ({
  children,
  source,
  src,
}) => {
  const t = useT();
  const aui = useAui();
  const attachmentId = useAuiState(({ attachment }) => attachment.id);
  const attachments = useAuiState(({ composer }) => composer.attachments);
  const images = useMemo(() => galleryImagesOf(attachments), [attachments]);
  const owner = useMemo(
    () => ({
      id: attachmentId,
      name: source.name,
      contentType: source.contentType,
      file: source.file,
      image: src,
    }),
    [attachmentId, source.name, source.contentType, source.file, src],
  );
  const gallery = useGalleryState(attachmentId);
  const shown = images.find((image) => image.id === gallery.shownId) ?? owner;
  const [savedIds, setSavedIds] = useState<ReadonlyMap<File, string>>(new Map());
  const savedId = shown.file ? (savedIds.get(shown.file) ?? null) : null;
  const { favorite } = useLibraryFavorite(savedId, gallery.open);
  const save = (image: GalleryImage) =>
    saveComposerImage(image).then((id) => {
      const file = image.file;
      if (file) setSavedIds((current) => new Map(current).set(file, id));
      return id;
    });
  return (
    <ImageGalleryDialog
      owner={owner}
      images={images}
      redactFromReload={true}
      {...gallery}
      actionsFor={(image) => ({
        favorite: image.id === shown.id && favorite,
        onToggleFavorite: () =>
          void save(image)
            .then((id) =>
              useLibraryFavoritesStore
                .getState()
                .setFavorite(id, !useLibraryFavoritesStore.getState().ids.has(id)),
            )
            .catch((error: unknown) =>
              toast.error(t("library.toast.favoritesFailed"), {
                description: error instanceof Error ? error.message : undefined,
              }),
            ),
        onAddToProject: image.file
          ? (projectId) => save(image).then((id) => addLibraryItemToProject(id, projectId))
          : undefined,
        deleteLabel: t("imageViewer.removeFromMessage"),
        onDelete: () => {
          const neighbour = neighbourOf(images, image.id);
          if (image.id === attachmentId || !neighbour) gallery.onOpenChange(false);
          else gallery.onShow(neighbour.id);
          void aui.composer().attachment({ id: image.id }).remove();
        },
      })}
    >
      {children}
    </ImageGalleryDialog>
  );
};

// The Library's id for a sent attachment, as _attachment_id in backend/core/library.py quotes it.
const attachmentItemId = (messageId: string, attachmentId: string): string =>
  `attachment:${encodeURIComponent(messageId).replace(
    /[!'()*]/g,
    (char) => `%${char.charCodeAt(0).toString(16).toUpperCase()}`,
  )}:${attachmentId}`;

const SentImageDialog: FC<PropsWithChildren<{ source: AttachmentSource; src: string }>> = ({
  children,
  source,
  src,
}) => {
  const t = useT();
  const messageId = useAuiState(({ message }) => message.id);
  const attachmentId = useAuiState(({ attachment }) => attachment.id);
  const attachments = useAuiState(({ message }) => message.attachments);
  const images = useMemo(() => galleryImagesOf(attachments), [attachments]);
  const owner = useMemo(
    () => ({
      id: attachmentId,
      name: source.name,
      contentType: source.contentType,
      file: source.file,
      image: src,
    }),
    [attachmentId, source.name, source.contentType, source.file, src],
  );
  const gallery = useGalleryState(attachmentId);
  const [deleting, setDeleting] = useState<GalleryImage | null>(null);
  const shown = images.find((image) => image.id === gallery.shownId) ?? owner;
  const { favorite, toggleFavorite } = useLibraryFavorite(
    attachmentItemId(messageId, shown.id),
    gallery.open,
  );
  return (
    <>
      <ImageGalleryDialog
        owner={owner}
        images={images}
        redactFromReload={false}
        {...gallery}
        actionsFor={(image) => ({
          favorite,
          onToggleFavorite: toggleFavorite,
          onAddToProject: (projectId) =>
            addLibraryItemToProject(attachmentItemId(messageId, image.id), projectId),
          onDelete: () => setDeleting(image),
        })}
      >
        {children}
      </ImageGalleryDialog>
      <ConfirmDeleteDialog
        open={deleting !== null}
        title={t("library.dialog.deleteTitle", { name: deleting?.name || t("imageViewer.title") })}
        description={t("library.dialog.deleteAttachment")}
        confirmLabel={t("common.delete")}
        onOpenChange={(next) => !next && setDeleting(null)}
        onConfirm={() => {
          const image = deleting;
          setDeleting(null);
          if (!image) return;
          const neighbour = neighbourOf(images, image.id);
          if (image.id === attachmentId || !neighbour) gallery.onOpenChange(false);
          else gallery.onShow(neighbour.id);
          // Removing it from the Library also takes it out of this message.
          import("@/features/library/store")
            .then(({ removeLibraryItem }) => removeLibraryItem(attachmentItemId(messageId, image.id)))
            .catch((error: unknown) =>
              toast.error(t("library.toast.deleteFailed"), {
                description: error instanceof Error ? error.message : undefined,
              }),
            );
        }}
      />
    </>
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
  // Highlighting stops at the transcript's ceiling so a long file opens without tokenizing.
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
  // The preview caps a sent body, so the whole one is cut from the stored text on click.
  const { file, text: sentText } = source;
  const load = file
    ? () => Promise.resolve<Blob>(file)
    : ready && !ready.label && sentText !== undefined
      ? () =>
          Promise.resolve(
            new Blob([attachmentBodyText(sentText)], { type: source.contentType || "text/plain" }),
          )
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
      load={
        file
          ? () => Promise.resolve(file)
          : src || audio
            ? () => fetchBlob(src ?? attachmentAudioSrc(audio!, source.contentType, source.name))
            : undefined
      }
    >
      <AttachmentAudioBody source={source} />
    </AttachmentViewer>
  );
};

const attachmentVideoSrc = (video: AttachmentVideoPart): string =>
  video.data.startsWith("data:") ? video.data : `data:${video.mimeType};base64,${video.data}`;

const AttachmentVideoBody: FC<{ source: AttachmentSource; onError: () => void }> = ({
  source,
  onError,
}) => {
  const src = useMemo(
    () => source.src ?? (source.video ? attachmentVideoSrc(source.video) : undefined),
    [source.src, source.video],
  );
  return src ? (
    <video src={src} controls autoPlay onError={onError} className="size-full object-contain" />
  ) : null;
};

const AttachmentVideoDialog: FC<
  PropsWithChildren<{ source: AttachmentSource; redactFromReload?: boolean }>
> = ({ children, source, redactFromReload = false }) => {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [failed, setFailed] = useState(false);
  const { file, src, video } = source;
  return (
    <AttachmentViewer
      trigger={children}
      open={open}
      onOpenChange={setOpen}
      source={source}
      meta={attachmentViewerMeta(source, file?.size)}
      media={!failed}
      noun="video"
      redactFromReload={redactFromReload}
      load={
        file
          ? () => Promise.resolve(file)
          : src || video
            ? () => fetchBlob(src ?? attachmentVideoSrc(video!))
            : undefined
      }
    >
      {failed ? (
        <p className="m-auto text-sm text-muted-foreground">{t("library.preview.cannotPreview")}</p>
      ) : (
        <AttachmentVideoBody source={source} onError={() => setFailed(true)} />
      )}
    </AttachmentViewer>
  );
};

export const AttachmentPreviewDialog: FC<
  /** Composer attachments are local and unsent, so keep them out of the reload snapshot. */
  PropsWithChildren<{ redactFromReload?: boolean }>
> = ({ children, redactFromReload = false }) => {
  const source = useAttachmentSource();
  return (
    <AttachmentBrowserOpenProvider source={source}>
      <AttachmentPreviewBody source={source} redactFromReload={redactFromReload}>
        {isValidElement(children) ? (
          <AttachmentFileContextMenu source={source}>{children}</AttachmentFileContextMenu>
        ) : (
          children
        )}
      </AttachmentPreviewBody>
    </AttachmentBrowserOpenProvider>
  );
};

const AttachmentPreviewBody: FC<PropsWithChildren<{ source: AttachmentSource; redactFromReload: boolean }>> = ({
  children,
  source,
  redactFromReload,
}) => {
  if (source.kind === "image") {
    if (!source.src) return children;
    return redactFromReload ? (
      <ComposerImageDialog source={source} src={source.src}>
        {children}
      </ComposerImageDialog>
    ) : (
      <SentImageDialog source={source} src={source.src}>
        {children}
      </SentImageDialog>
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

  if (source.kind === "video") {
    return source.src || source.video ? (
      <AttachmentVideoDialog source={source} redactFromReload={redactFromReload}>
        {children}
      </AttachmentVideoDialog>
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
