// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CodeToggleIcon } from "@/components/assistant-ui/code-toggle-icon";
import { CodeSourceView } from "@/components/code-source-view";
import { Button } from "@/components/ui/button";
import { MediaViewer, ScaleMenu } from "@/components/media-viewer";
import { Spinner } from "@/components/ui/spinner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ArtifactHtmlFrame } from "@/features/chat";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { isTauri } from "@/lib/api-base";
import { MessageCircleIcon } from "@/lib/hugeicons-derived";
import { toast } from "@/lib/toast";
import { useBlocker, useNavigate } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import { PlayIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import {
  type LibraryItem,
  addLibraryItemToProject,
  errorMessage,
  fetchLibraryText,
  writeLibraryText,
} from "../api";
import {
  fileKind,
  hasImagePreview,
  isDeletable,
  isFileItem,
  isTextPreviewable,
  modelLabelKey,
} from "../file-kind";
import { formatCardTime, formatSize } from "../format";
import { type EmbeddedBody, hasOwnFile, itemVersion } from "../file-name";
import { useLibraryPreviewUrl } from "../hooks";
import { type NoteFormat, type NoteReadOnlyReason, encodeNote } from "../note-text";
import { canReveal, revealInFolder, useRevealLabel } from "../reveal";
import { KindIcon } from "./library-cards";
import { UnsavedChangesDialog } from "./library-dialogs";

const PAGE_SCALES = [0.5, 0.75, 1, 1.25, 1.5, 2];

const MAX_TEXT_PREVIEW_BYTES = 1024 * 1024;

type Body = "image" | "web" | "text" | "pdf" | "audio" | "video" | "model" | "none";

function bodyFor(item: LibraryItem): Body {
  if (item.model) return "model";
  if (hasImagePreview(item)) return "image";
  if (item.textOnly) return "text";
  const kind = fileKind(item);
  if (kind === "web") return "web";
  if (kind === "pdf") return isTauri || navigator.pdfViewerEnabled === false ? "none" : "pdf";
  if (kind === "audio" || kind === "video") return kind;
  return isTextPreviewable(item) ? "text" : "none";
}

function generatedOn(item: LibraryItem) {
  if (item.archived) return null;
  const [kind, ...rest] = item.id.split(":");
  const search = { item: rest.join(":") };
  if (kind === "image") return { label: "library.preview.viewInImages", to: "/images", search } as const;
  if (kind === "video") return { label: "library.preview.viewInVideo", to: "/video", search } as const;
  const speak = { ...search, task: "text-to-speech" } as const;
  if (kind === "audio") return { label: "library.preview.viewInAudio", to: "/audio", search: speak } as const;
  return null;
}

function isEditable(item: LibraryItem): boolean {
  return item.id.startsWith("upload:") && bodyFor(item) === "text";
}

interface LoadedText {
  key: string;
  itemId: string;
  text?: string;
  truncated?: boolean;
  format?: NoteFormat;
  readOnlyReason?: NoteReadOnlyReason | null;
  error?: string;
}

/**
 * The item's text, or null while it first loads. A newer version of the same item keeps showing
 * the last one until it arrives (`loadedKey` says which is on screen), so an editor is never
 * swapped for a spinner, and never loses focus, when its own save bumps the version.
 */
function useItemText(item: LibraryItem | null, enabled: boolean) {
  const key = item ? itemVersion(item) : "";
  const [state, setState] = useState<LoadedText | null>(null);
  if (!item && state) setState(null);
  useEffect(() => {
    if (!enabled || !item) return;
    let cancelled = false;
    const itemId = item.id;
    fetchLibraryText(item, MAX_TEXT_PREVIEW_BYTES).then(
      ({ text, truncated, format, readOnlyReason }) =>
        !cancelled && setState({ key, itemId, text, truncated, format, readOnlyReason }),
      (err: unknown) => !cancelled && setState({ key, itemId, error: errorMessage(err) }),
    );
    return () => {
      cancelled = true;
    };
    // `key` carries the item's identity and version; the object itself changes on every refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);
  const current = item && state?.itemId === item.id ? state : null;
  return {
    text: current?.text ?? null,
    truncated: current?.truncated ?? false,
    format: current?.format ?? null,
    readOnlyReason: current?.readOnlyReason ?? null,
    error: current?.error ?? null,
    loadedKey: current?.key ?? null,
  };
}

type ItemText = ReturnType<typeof useItemText>;

const READ_ONLY_REASONS: Record<NoteReadOnlyReason, TranslationKey> = {
  utf16: "library.preview.readOnlyUtf16",
  notUtf8: "library.preview.readOnlyNotUtf8",
};

function ModelDetails({ item }: { item: LibraryItem }) {
  const t = useT();
  const locale = useLocale();
  const unknown = t("library.preview.unknown");
  const rows: [string, string][] = [
    [t("library.preview.type"), t(modelLabelKey(item) ?? "library.modelKind.model")],
    [t("library.preview.baseModel"), item.model?.baseModel ?? unknown],
    [t("library.preview.size"), formatSize(item.sizeBytes, locale, t) || unknown],
    [t("library.preview.location"), item.model?.path ?? ""],
  ];
  return (
    <div className="m-auto flex w-full max-w-xl flex-col items-center gap-8">
      <KindIcon item={item} className="size-16" />
      <dl className="grid w-full grid-cols-[auto_1fr] gap-x-8 gap-y-3 text-sm">
        {rows.map(([label, value]) => (
          <div key={label} className="contents">
            <dt className="text-muted-foreground">{label}</dt>
            <dd className="select-text break-all text-foreground">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function TextPrefix({ text, className }: { text: string; className?: string }) {
  return (
    <pre
      className={cn(
        "size-full overflow-auto whitespace-pre-wrap break-words font-mono text-sm leading-relaxed",
        className,
      )}
    >
      {text}
    </pre>
  );
}

/** What shows when a file cannot be previewed here, by type or because it failed to load. */
function NoPreview({
  item,
  message,
  onDownload,
}: {
  item: LibraryItem;
  message: string;
  onDownload?: () => void;
}) {
  const t = useT();
  return (
    <div className="m-auto flex max-w-md flex-col items-center gap-3 text-center text-muted-foreground">
      <KindIcon item={item} className="size-16" />
      <p className="text-sm">{message}</p>
      {onDownload && (
        <Button variant="muted" size="sm" className="rounded-full px-4" onClick={onDownload}>
          {t("library.menu.download")}
        </Button>
      )}
    </div>
  );
}

function ViewButton({
  label,
  active,
  onClick,
  children,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
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
}

function PreviewBody({
  item,
  itemText,
  draft,
  onDraftChange,
  showCode,
  pageScale,
  mediaFailed,
  onMediaError,
  onDownload,
}: {
  item: LibraryItem;
  itemText: ItemText;
  draft: string | null;
  onDraftChange: (value: string) => void;
  showCode: boolean;
  pageScale: number;
  mediaFailed: boolean;
  onMediaError: () => void;
  onDownload?: () => void;
}) {
  const t = useT();
  const body = bodyFor(item);
  const embedded: EmbeddedBody | null =
    !mediaFailed && (body === "image" || body === "pdf" || body === "audio" || body === "video")
      ? body
      : null;
  const { url, error: urlError, retry } = useLibraryPreviewUrl(item, embedded);
  const handleMediaError = () => {
    if (!retry()) onMediaError();
  };
  const { text, truncated, readOnlyReason, error: textError } = itemText;
  const noPreview = (message: string) => (
    <NoPreview item={item} message={message} onDownload={onDownload} />
  );

  if (mediaFailed) return noPreview(t("library.preview.cannotPreview"));
  if (urlError) return noPreview(urlError);
  if (textError) return <p className="m-auto text-sm text-muted-foreground">{textError}</p>;
  if ((embedded && !url) || ((body === "text" || body === "web") && text === null)) {
    return <Spinner className="m-auto size-6" />;
  }
  const prefix = truncated ? `${text}\n\n…` : text!;
  switch (body) {
    case "model":
      return <ModelDetails item={item} />;
    case "image":
      return (
        <img src={url!} alt={item.name} onError={handleMediaError} className="size-full object-contain" />
      );
    case "pdf":
      return <iframe title={item.name} src={url!} className="size-full rounded-xl bg-white" />;
    case "audio":
      return (
        <audio src={url!} controls onError={handleMediaError} className="m-auto w-full max-w-lg" />
      );
    case "video":
      return (
        <video
          src={url!}
          controls
          autoPlay
          onError={handleMediaError}
          className="size-full object-contain"
        />
      );
    case "web":
      if (showCode) {
        return (
          <CodeSourceView
            code={truncated ? `${text!}\n…` : text!}
            language="html"
            className="rounded-xl"
          />
        );
      }
      if (truncated) return <TextPrefix text={prefix} />;
      return (
        <div className="size-full overflow-hidden rounded-xl">
          <div
            className="origin-top-left"
            style={{
              width: `${100 / pageScale}%`,
              height: `${100 / pageScale}%`,
              transform: `scale(${pageScale})`,
            }}
          >
            <ArtifactHtmlFrame code={text!} title={item.name} fill />
          </div>
        </div>
      );
    case "text":
      if (isEditable(item) && !truncated && !readOnlyReason) {
        return (
          <textarea
            value={draft ?? text!}
            onChange={(event) => onDraftChange(event.target.value)}
            spellCheck={false}
            placeholder={t("library.preview.startWriting")}
            className="size-full resize-none bg-transparent font-mono text-sm leading-relaxed outline-none"
          />
        );
      }
      // A note the editor would write back wrongly says why it cannot be edited.
      return isEditable(item) && readOnlyReason ? (
        <div className="flex size-full min-h-0 flex-col gap-3">
          <p className="text-[13px] text-muted-foreground">
            {t(READ_ONLY_REASONS[readOnlyReason])}
          </p>
          <TextPrefix text={prefix} className="min-h-0 flex-1" />
        </div>
      ) : (
        <TextPrefix text={prefix} />
      );
    default:
      return noPreview(t("library.preview.noPreview"));
  }
}

export function LibraryPreview({
  item,
  onOpenChange,
  onChat,
  onDownload,
  onOpenThread,
  onToggleFavorite,
  onDelete,
  onSaved,
}: {
  item: LibraryItem | null;
  onOpenChange: (open: boolean) => void;
  onChat: (item: LibraryItem) => void;
  onDownload: (item: LibraryItem) => void;
  onOpenThread: (threadId: string) => void;
  onToggleFavorite: (item: LibraryItem) => void;
  onDelete: (item: LibraryItem) => void;
  onSaved: () => void;
}) {
  const t = useT();
  const locale = useLocale();
  const body = item ? bodyFor(item) : "none";
  const itemText = useItemText(item, body === "text" || body === "web");
  const version = item && itemVersion(item);
  // Tagged with its item, so a draft never follows the preview to another file.
  // `savedAt` marks text already written: the item version it was saved over, shown until the
  // refreshed item's text has loaded, so the editor never falls back to the old text.
  const [edit, setEdit] = useState<{ itemId: string; text: string; savedAt?: number } | null>(
    null,
  );
  // The same draft, readable after an await: a save must not return while typing moved past it.
  const latestEdit = useRef(edit);
  const lastSaved = useRef<{ itemId: string; text: string } | null>(null);
  const current = item && edit?.itemId === item.id ? edit : null;
  if (
    current?.savedAt !== undefined &&
    item!.updatedAt !== current.savedAt &&
    itemText.loadedKey === version
  ) {
    setEdit(null);
  }
  const draft = current ? current.text : null;
  const unsaved = current !== null && current.savedAt === undefined;
  const setDraft = (text: string | null) => {
    const next = item && text !== null ? { itemId: item.id, text } : null;
    latestEdit.current = next;
    setEdit(next);
  };
  const [saving, setSaving] = useState(false);
  const [closeError, setCloseError] = useState<string | null>(null);
  const [brokenMedia, setBrokenMedia] = useState<string | null>(null);
  const mediaFailed = version !== null && brokenMedia === version;
  const [codeFor, setCodeFor] = useState<string | null>(null);
  const [zoom, setZoom] = useState<{ itemId: string; scale: number } | null>(null);
  if (item === null && (codeFor !== null || zoom !== null)) {
    setCodeFor(null);
    setZoom(null);
  }
  const showCode = item !== null && codeFor === item.id;
  const pageScale = item !== null && zoom?.itemId === item.id ? zoom.scale : 1;
  const revealLabel = useRevealLabel();
  const navigate = useNavigate();
  const origin = item ? generatedOn(item) : null;

  async function trySave(): Promise<string | null> {
    if (!item || draft === null || !unsaved) return null;
    const format = itemText.format ?? { encoding: "utf-8", bom: false, eol: "\n" };
    setSaving(true);
    try {
      let sent = draft;
      for (;;) {
        await writeLibraryText(item.id, encodeNote(sent, format), format.encoding);
        const latest = latestEdit.current;
        if (latest?.itemId !== item.id || latest.text === sent) break;
        sent = latest.text;
      }
      const savedAt = item.updatedAt;
      lastSaved.current = { itemId: item.id, text: sent };
      setEdit((latest) =>
        latest?.itemId === item.id && latest.text === sent ? { ...latest, savedAt } : latest,
      );
      onSaved();
      return null;
    } catch (error) {
      return errorMessage(error);
    } finally {
      setSaving(false);
    }
  }

  async function save(): Promise<boolean> {
    const error = await trySave();
    if (error !== null) toast.error(t("library.toast.saveFailed"), { description: error });
    return error === null;
  }

  async function saveThen(action: () => void) {
    if (await save()) action();
  }

  // Closing a note saves it, so an edit is never lost to a stray Escape. A save that fails asks
  // what to do instead, so the preview can always be closed.
  async function handleOpenChange(open: boolean) {
    if (!open) {
      const error = await trySave();
      if (error !== null) {
        setCloseError(error);
        return;
      }
    }
    onOpenChange(open);
  }

  // Back, or any link, closes the preview without handleOpenChange: save first, as closing does.
  // Refs, not `unsaved`: a discard or a save just made has not rendered yet when it navigates.
  useBlocker({
    shouldBlockFn: async () => {
      const latest = latestEdit.current;
      const saved = lastSaved.current;
      if (!latest || (saved?.itemId === latest.itemId && saved.text === latest.text)) return false;
      return !(await save());
    },
    enableBeforeUnload: () => unsaved,
  });

  function discardAndClose() {
    setCloseError(null);
    setDraft(null);
    onOpenChange(false);
  }

  const meta = item
    ? [
        t(
          modelLabelKey(item) ??
            (item.source === "generated" ? "library.toolbar.generated" : "library.toolbar.uploaded"),
        ),
        formatSize(item.sizeBytes, locale, t),
        formatCardTime(item.updatedAt, locale),
      ].filter(Boolean)
    : [];
  const download = item && isFileItem(item) ? () => void saveThen(() => onDownload(item)) : undefined;
  const media = (body === "image" || body === "video") && !mediaFailed;

  return (
    <MediaViewer
      open={item !== null}
      onOpenChange={(open) => void handleOpenChange(open)}
      title={item?.name ?? ""}
      meta={meta.join(" · ")}
      media={media}
      noun={media ? body : "file"}
      onKeyDown={(event) => {
        const saveKey = event.code === "KeyS" || event.key.toLowerCase() === "s";
        if ((event.metaKey || event.ctrlKey) && saveKey) {
          event.preventDefault();
          void save();
        }
      }}
      extra={
        <>
          {body === "web" && item && !showCode && (
            <ScaleMenu
              value={pageScale}
              scales={PAGE_SCALES}
              onChange={(value) => setZoom({ itemId: item.id, scale: Number(value) })}
            />
          )}
          {body === "web" && item && (
            <div className="mr-1 flex items-center gap-1">
              <ViewButton
                label={t("library.preview.code")}
                active={showCode}
                onClick={() => setCodeFor(item.id)}
              >
                <CodeToggleIcon className="size-4.5" />
              </ViewButton>
              <ViewButton
                label={t("library.preview.preview")}
                active={!showCode}
                onClick={() => setCodeFor(null)}
              >
                <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} className="size-5" />
              </ViewButton>
            </div>
          )}
          {unsaved && (
            <Button variant="dark" size="sm" className="mr-1" disabled={saving} onClick={() => void save()}>
              {t("common.save")}
            </Button>
          )}
        </>
      }
      actions={
        item
          ? {
              primary: {
                label: t(item.model ? "library.menu.chatWithModel" : "library.menu.chatAboutThis"),
                icon: MessageCircleIcon,
                disabled: saving,
                onClick: () => void saveThen(() => onChat(item)),
              },
              onDownload: download,
              viewOriginal: item.threadId
                ? {
                    label: t("library.preview.viewOriginalChat"),
                    onClick: () => void saveThen(() => onOpenThread(item.threadId!)),
                  }
                : origin
                  ? {
                      label: t(origin.label),
                      onClick: () => void navigate({ to: origin.to, search: origin.search }),
                    }
                  : undefined,
              reveal:
                revealLabel && canReveal(item)
                  ? {
                      label: revealLabel,
                      onClick: () => void saveThen(() => revealInFolder(item.id)),
                    }
                  : undefined,
              favorite: item.favorite,
              onToggleFavorite: () => onToggleFavorite(item),
              onAddToProject: hasOwnFile(item.id)
                ? async (projectId) => {
                    if (!(await save())) throw new Error(t("library.toast.saveNoteFirst"));
                    return addLibraryItemToProject(item.id, projectId);
                  }
                : undefined,
              onDelete: isDeletable(item) ? () => onDelete(item) : undefined,
            }
          : {}
      }
    >
      {item && (
        <PreviewBody
          item={item}
          itemText={itemText}
          draft={draft}
          onDraftChange={setDraft}
          showCode={showCode}
          pageScale={pageScale}
          mediaFailed={mediaFailed}
          onMediaError={() => setBrokenMedia(version)}
          onDownload={download}
        />
      )}
      {/* Portalled, so it sits over the preview whatever the body is. */}
      <UnsavedChangesDialog
        error={closeError}
        onRetry={() => {
          setCloseError(null);
          void handleOpenChange(false);
        }}
        onDiscard={discardAndClose}
        onKeepEditing={() => setCloseError(null)}
      />
    </MediaViewer>
  );
}
