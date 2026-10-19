// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CodeToggleIcon } from "@/components/assistant-ui/code-toggle-icon";
import { CodeSourceView } from "@/components/code-source-view";
import { DocumentView, documentKind, isMarkdown, sheetDelimiter } from "@/components/file-viewer";
import { MarkdownPreview } from "@/components/markdown/markdown-preview";
import { Button } from "@/components/ui/button";
import { MediaViewer, ScaleMenu } from "@/components/media-viewer";
import { Spinner } from "@/components/ui/spinner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { audioModelLabel } from "@/features/audio/audio-workspace-utils";
import { Waveform } from "@/features/audio/components/waveform";
import { formatSeconds } from "@/features/audio/components/waveform-peaks";
import { ArtifactHtmlFrame } from "@/features/chat";
import { type TranslationKey, useLocale, useT } from "@/i18n";
import { MessageCircleIcon } from "@/lib/hugeicons-derived";
import { toast } from "@/lib/toast";
import { useBlocker } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import { Download04Icon, PlayIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  type LibraryItem,
  addLibraryItemToProject,
  errorMessage,
  fetchLibraryText,
  writeLibraryText,
} from "../api";
import { audioDetail, audioWorkflow } from "../audio-items";
import { stopLibraryAudio } from "../audio-playback";
import {
  fileKind,
  hasImagePreview,
  isDeletable,
  isFileItem,
  isTextPreviewable,
  modelLabelKey,
} from "../file-kind";
import { formatCardTime, formatSize } from "../format";
import { type EmbeddedBody, fileExtension, hasOwnFile, itemVersion } from "../file-name";
import { useLibraryAudioPeaks, useLibraryDocument, useLibraryPreviewUrl } from "../hooks";
import { useLibraryOrigin } from "../origin";
import { type NoteFormat, type NoteReadOnlyReason, encodeNote } from "../note-text";
import { canReveal, revealInFolder, useRevealLabel } from "../reveal";
import { KindIcon } from "./library-cards";
import { UnsavedChangesDialog } from "./library-dialogs";

const PAGE_SCALES = [0.5, 0.75, 1, 1.25, 1.5, 2];

const MAX_TEXT_PREVIEW_BYTES = 1024 * 1024;

type Body =
  | "image"
  | "web"
  | "text"
  | "markdown"
  | "document"
  | "audio"
  | "video"
  | "model"
  | "none";

const CODE_LANGUAGES: Record<string, string> = {
  py: "python",
  js: "javascript",
  jsx: "jsx",
  ts: "typescript",
  tsx: "tsx",
  json: "json",
  sh: "bash",
  css: "css",
  sql: "sql",
  yaml: "yaml",
  yml: "yaml",
  xml: "xml",
};

function ownName(item: LibraryItem): string {
  return item.fileName ?? item.name;
}

function bodyFor(item: LibraryItem): Body {
  if (item.model) return "model";
  if (hasImagePreview(item)) return "image";
  if (item.textOnly) {
    if (isMarkdown(ownName(item), item.contentType)) return "markdown";
    return sheetDelimiter(ownName(item), item.contentType) ? "document" : "text";
  }
  if (documentKind(ownName(item), item.contentType)) return "document";
  const kind = fileKind(item);
  if (kind === "web") return "web";
  if (kind === "audio" || kind === "video") return kind;
  if (isMarkdown(ownName(item), item.contentType)) return "markdown";
  return isTextPreviewable(item) ? "text" : "none";
}

function hasSource(item: LibraryItem, body: Body): boolean {
  if (body === "web" || body === "markdown") return true;
  return body === "document" && sheetDelimiter(ownName(item), item.contentType) !== null;
}

function viewFor(item: LibraryItem, showCode: boolean): Body {
  const body = bodyFor(item);
  return showCode && body !== "web" && hasSource(item, body) ? "text" : body;
}

function isEditable(item: LibraryItem): boolean {
  const body = bodyFor(item);
  return item.id.startsWith("upload:") && (body === "text" || (body !== "web" && hasSource(item, body)));
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

const ZOOMABLE: ReadonlySet<Body> = new Set(["web", "document", "markdown", "text"]);

function Zoomed({ scale, children }: { scale: number; children: ReactNode }) {
  return (
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
}

function TextPrefix({ text, className }: { text: string; className?: string }) {
  return (
    <pre
      className={cn(
        "size-full overflow-auto whitespace-pre-wrap break-words px-6 pb-6 font-mono text-sm leading-relaxed",
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

function RunStrip({
  item,
  run,
  onOpenItem,
  onDownloadRun,
}: {
  item: LibraryItem;
  run: LibraryItem[];
  onOpenItem: (id: string) => void;
  onDownloadRun: () => void;
}) {
  const t = useT();
  return (
    <div className="flex flex-col gap-2">
      <p className="text-ui-12 text-muted-foreground">{t("library.audio.fromRun")}</p>
      <div className="flex flex-wrap items-center gap-1.5">
        {run.map((clip) => (
          <Button
            key={clip.id}
            variant={clip.id === item.id ? "secondary" : "ghost"}
            size="sm"
            aria-pressed={clip.id === item.id}
            className="rounded-full px-3"
            onClick={() => clip.id !== item.id && onOpenItem(clip.id)}
          >
            {audioDetail(clip) ?? clip.name}
          </Button>
        ))}
        <Button variant="outline" size="sm" className="ml-auto rounded-full px-3" onClick={onDownloadRun}>
          <HugeiconsIcon icon={Download04Icon} strokeWidth={1.75} className="size-4" />
          {t("library.audio.downloadRun")}
        </Button>
      </div>
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
  run,
  onOpenItem,
  onDownloadRun,
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
  run: LibraryItem[];
  onOpenItem: (id: string) => void;
  onDownloadRun: () => void;
}) {
  const t = useT();
  const body = viewFor(item, showCode);
  const embedded: EmbeddedBody | null =
    !mediaFailed && (body === "image" || body === "audio" || body === "video") ? body : null;
  const { url, error: urlError, retry } = useLibraryPreviewUrl(item, embedded);
  const doc = useLibraryDocument(item, body === "document");
  const peaks = useLibraryAudioPeaks(item, body === "audio" && !mediaFailed);
  const draftFile = useMemo(() => (draft === null ? null : new Blob([draft])), [draft]);
  const handleMediaError = () => {
    if (!retry()) onMediaError();
  };
  const { text, truncated, readOnlyReason, error: textError } = itemText;
  const noPreview = (message: string) => (
    <NoPreview item={item} message={message} onDownload={onDownload} />
  );

  if (mediaFailed) return noPreview(t("library.preview.cannotPreview"));
  if (urlError) return noPreview(urlError);
  if (doc.error) return noPreview(doc.error);
  if (textError) return <p className="m-auto text-sm text-muted-foreground">{textError}</p>;
  if (
    (embedded && !url) ||
    (body === "document" && !(draftFile ?? doc.file)) ||
    ((body === "text" || body === "web" || body === "markdown") && text === null)
  ) {
    return <Spinner className="m-auto size-6" />;
  }
  const prefix = truncated ? `${text}\n\n…` : text!;
  const language = !item.textOnly && CODE_LANGUAGES[fileExtension(ownName(item))];
  switch (body) {
    case "model":
      return <ModelDetails item={item} />;
    case "image":
      return (
        <img src={url!} alt={item.name} onError={handleMediaError} className="size-full object-contain" />
      );
    case "document":
      return (
        <DocumentView
          file={draftFile ?? doc.file!}
          kind={documentKind(ownName(item), item.contentType)!}
          name={ownName(item)}
          contentType={item.contentType}
          scale={pageScale}
        />
      );
    case "markdown":
      return (
        <Zoomed scale={pageScale}>
          <div className="size-full overflow-auto px-6">
            <MarkdownPreview
              markdown={draft ?? prefix}
              defer={true}
              className="mx-auto max-h-none max-w-3xl select-text overflow-visible border-0 bg-transparent px-2 py-4 text-ui-15p5"
            />
          </div>
        </Zoomed>
      );
    case "audio":
      return (
        <div className="m-auto flex w-full max-w-2xl flex-col gap-6 px-6">
          <Waveform
            peaks={peaks}
            durationS={item.audio?.durationS ?? null}
            src={url}
            label={item.name}
            onError={handleMediaError}
          />
          {run.length > 1 && (
            <RunStrip item={item} run={run} onOpenItem={onOpenItem} onDownloadRun={onDownloadRun} />
          )}
        </div>
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
          <Zoomed scale={pageScale}>
            <CodeSourceView
              code={truncated ? `${text!}\n…` : text!}
              language="html"
              className="px-6 pb-6"
            />
          </Zoomed>
        );
      }
      if (truncated) {
        return (
          <Zoomed scale={pageScale}>
            <TextPrefix text={prefix} />
          </Zoomed>
        );
      }
      return (
        <Zoomed scale={pageScale}>
          <ArtifactHtmlFrame code={text!} title={item.name} fill />
        </Zoomed>
      );
    case "text":
      if (isEditable(item) && !truncated && !readOnlyReason) {
        return (
          <Zoomed scale={pageScale}>
            <textarea
              value={draft ?? text!}
              onChange={(event) => onDraftChange(event.target.value)}
              spellCheck={false}
              placeholder={t("library.preview.startWriting")}
              className="size-full resize-none bg-transparent px-6 pb-6 font-mono text-sm leading-relaxed outline-none"
            />
          </Zoomed>
        );
      }
      // A note the editor would write back wrongly says why it cannot be edited.
      if (isEditable(item) && readOnlyReason) {
        return (
          <Zoomed scale={pageScale}>
            <div className="flex size-full min-h-0 flex-col gap-3">
              <p className="px-6 text-ui-13 text-muted-foreground">
                {t(READ_ONLY_REASONS[readOnlyReason])}
              </p>
              <TextPrefix text={prefix} className="min-h-0 flex-1" />
            </div>
          </Zoomed>
        );
      }
      return (
        <Zoomed scale={pageScale}>
          {language ? (
            <CodeSourceView code={prefix} language={language} className="px-6 pb-6" />
          ) : (
            <TextPrefix text={prefix} />
          )}
        </Zoomed>
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
  onToggleFavorite,
  onDelete,
  onSaved,
  run = [],
  onOpenItem,
  onDownloadRun,
}: {
  item: LibraryItem | null;
  onOpenChange: (open: boolean) => void;
  onChat: (item: LibraryItem) => void;
  onDownload: (item: LibraryItem) => void;
  onToggleFavorite: (item: LibraryItem) => void;
  onDelete: (item: LibraryItem) => void;
  onSaved: () => void;
  run?: LibraryItem[];
  onOpenItem: (id: string) => void;
  onDownloadRun: (items: LibraryItem[]) => void;
}) {
  const t = useT();
  const locale = useLocale();
  const [codeFor, setCodeFor] = useState<string | null>(null);
  const showCode = item !== null && codeFor === item.id;
  const body = item ? bodyFor(item) : "none";
  const view = item ? viewFor(item, showCode) : "none";
  const itemText = useItemText(item, view === "text" || view === "web" || view === "markdown");
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
  const [zoom, setZoom] = useState<{ itemId: string; scale: number } | null>(null);
  if (item === null && (codeFor !== null || zoom !== null)) {
    setCodeFor(null);
    setZoom(null);
  }
  const pageScale = item !== null && zoom?.itemId === item.id ? zoom.scale : 1;
  const revealLabel = useRevealLabel();
  const originOf = useLibraryOrigin();
  const origin = item ? originOf(item) : null;
  // The preview has its own player: a card left playing would talk over it.
  const open = item !== null;
  useEffect(() => {
    if (open) stopLibraryAudio();
  }, [open]);

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

  const workflow = item && audioWorkflow(item);
  const meta = item
    ? [
        item.threadId
          ? t("library.preview.fromChat")
          : workflow
            ? workflow.label
            : t(
                modelLabelKey(item) ??
                  (item.source === "generated" ? "library.toolbar.generated" : "library.toolbar.uploaded"),
              ),
        item.threadId ? item.threadTitle : null,
        ...(item.audio
          ? [
              audioDetail(item),
              item.audio.durationS ? formatSeconds(item.audio.durationS) : null,
              item.audio.model ? audioModelLabel(item.audio.model) : null,
            ]
          : []),
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
      flush={true}
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
          {item && ZOOMABLE.has(body) && (
            <ScaleMenu
              value={pageScale}
              scales={PAGE_SCALES}
              onChange={(value) => setZoom({ itemId: item.id, scale: Number(value) })}
            />
          )}
          {item && hasSource(item, body) && (
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
              viewOriginal: origin
                ? { label: t(origin.label), onClick: () => void saveThen(origin.open) }
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
          run={run}
          onOpenItem={onOpenItem}
          onDownloadRun={() => onDownloadRun(run)}
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
