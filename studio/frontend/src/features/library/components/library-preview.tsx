// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { CodeToggleIcon } from "@/components/assistant-ui/code-toggle-icon";
import { CodeSourceView } from "@/components/code-source-view";
import { Button } from "@/components/ui/button";
import { MediaViewer, ScaleMenu } from "@/components/media-viewer";
import { Spinner } from "@/components/ui/spinner";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ArtifactHtmlFrame } from "@/features/chat";
import { isTauri } from "@/lib/api-base";
import { MessageCircleIcon } from "@/lib/hugeicons-derived";
import { toast } from "@/lib/toast";
import { useNavigate } from "@tanstack/react-router";
import { cn } from "@/lib/utils";
import { PlayIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactNode, useEffect, useRef, useState } from "react";
import {
  type LibraryItem,
  addLibraryItemToProject,
  fetchLibraryTextPrefix,
  writeLibraryText,
} from "../api";
import {
  fileKind,
  hasImagePreview,
  isDeletable,
  isFileItem,
  isTextPreviewable,
  modelLabel,
} from "../file-kind";
import { formatCardTime, formatSize } from "../format";
import { useLibraryObjectUrl } from "../hooks";
import { canReveal, revealInFolder, useRevealLabel } from "../reveal";
import { KindIcon } from "./library-cards";

// Web pages zoom like a browser tab: the page reflows at the new size.
const PAGE_SCALES = [0.5, 0.75, 1, 1.25, 1.5, 2];

// Past this the preview shows a read-only prefix; the full file is a download away.
const MAX_TEXT_PREVIEW_BYTES = 1024 * 1024;

type Body = "image" | "web" | "text" | "pdf" | "audio" | "video" | "model" | "none";

function bodyFor(item: LibraryItem): Body {
  if (item.model) return "model";
  if (hasImagePreview(item)) return "image";
  if (item.textOnly) return "text";
  const kind = fileKind(item);
  if (kind === "web") return "web";
  // The desktop app's CSP allows no blob: frames, and some webviews (WebKitGTK) have no PDF viewer
  // at all; both get Download instead.
  if (kind === "pdf") return isTauri || navigator.pdfViewerEnabled === false ? "none" : "pdf";
  if (kind === "audio" || kind === "video") return kind;
  return isTextPreviewable(item) ? "text" : "none";
}

/** The page a generated file came from, which opens with it selected. None once archived: the
 *  page lists only its active shelf. */
function generatedOn(item: LibraryItem) {
  if (item.archived) return null;
  const [kind, ...rest] = item.id.split(":");
  const id = rest.join(":");
  if (kind === "image") return { label: "View in Images", to: "/images", search: { item: id } } as const;
  if (kind === "video") return { label: "View in Video", to: "/video", search: { item: id } } as const;
  if (kind === "audio") {
    // Generated clips list in Speak mode.
    return {
      label: "View in Audio",
      to: "/audio",
      search: { task: "text-to-speech", item: id },
    } as const;
  }
  return null;
}

/** Items with a file of their own; chat attachments live inside messages, fine-tunes are folders. */
function canAddToProject(item: LibraryItem): boolean {
  return /^(upload|image|video|audio|sandbox):/.test(item.id);
}

/** Library-owned text files can be edited in place; everything else is read-only. */
function isEditable(item: LibraryItem): boolean {
  return item.id.startsWith("upload:") && bodyFor(item) === "text";
}

/** The item's text, or null while it loads. Keyed by version so a stale result never shows. */
function useItemText(item: LibraryItem, enabled: boolean) {
  const key = `${item.id}@${item.updatedAt}`;
  const [state, setState] = useState<{
    key: string;
    text?: string;
    truncated?: boolean;
    error?: string;
  } | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    fetchLibraryTextPrefix(item, MAX_TEXT_PREVIEW_BYTES).then(
      ({ text, truncated }) => !cancelled && setState({ key, text, truncated }),
      (err: unknown) =>
        !cancelled && setState({ key, error: err instanceof Error ? err.message : String(err) }),
    );
    return () => {
      cancelled = true;
    };
    // `key` carries the item's identity and version; the object itself changes on every refresh.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);
  const current = state?.key === key ? state : null;
  return {
    text: current?.text ?? null,
    truncated: current?.truncated ?? false,
    error: current?.error ?? null,
  };
}

function ModelDetails({ item }: { item: LibraryItem }) {
  const rows: [string, string][] = [
    ["Type", modelLabel(item)],
    ["Base model", item.model?.baseModel ?? "Unknown"],
    ["Size", formatSize(item.sizeBytes) || "Unknown"],
    ["Location", item.model?.path ?? ""],
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

function TextPrefix({ text }: { text: string }) {
  return (
    <pre className="size-full overflow-auto whitespace-pre-wrap break-words font-mono text-sm leading-relaxed">
      {text}
    </pre>
  );
}

/** A round header button with a tooltip; `active` marks the view on screen. */
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
  draft,
  onDraftChange,
  showCode,
  pageScale,
}: {
  item: LibraryItem;
  draft: string | null;
  onDraftChange: (value: string) => void;
  /** Web pages: their source instead of the rendered page. */
  showCode: boolean;
  /** Web pages: the rendered page's zoom. */
  pageScale: number;
}) {
  const body = bodyFor(item);
  const needsUrl = body === "image" || body === "pdf" || body === "audio" || body === "video";
  const { url, error: urlError } = useLibraryObjectUrl(item, needsUrl);
  const { text, truncated, error: textError } = useItemText(
    item,
    body === "text" || body === "web",
  );
  const error = urlError ?? textError;

  if (error) {
    return <p className="m-auto text-sm text-muted-foreground">{error}</p>;
  }
  if ((needsUrl && !url) || ((body === "text" || body === "web") && text === null)) {
    return <Spinner className="m-auto size-6" />;
  }
  switch (body) {
    case "model":
      return <ModelDetails item={item} />;
    case "image":
      return <img src={url!} alt={item.name} className="size-full object-contain" />;
    case "pdf":
      return <iframe title={item.name} src={url!} className="size-full rounded-xl bg-white" />;
    case "audio":
      return <audio src={url!} controls className="m-auto w-full max-w-lg" />;
    case "video":
      return <video src={url!} controls autoPlay className="size-full object-contain" />;
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
      if (truncated) return <TextPrefix text={`${text!}\n\n…`} />;
      // The chat canvas frame: served by the backend under its own CSP, so it renders the same in
      // the browser and the desktop app, and honors the canvas network-access setting.
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
      return isEditable(item) && !truncated ? (
        <textarea
          value={draft ?? text!}
          onChange={(event) => onDraftChange(event.target.value)}
          spellCheck={false}
          placeholder="Start writing…"
          className="size-full resize-none bg-transparent font-mono text-sm leading-relaxed outline-none"
        />
      ) : (
        <TextPrefix text={truncated ? `${text!}\n\n…` : text!} />
      );
    default:
      return (
        <div className="m-auto flex flex-col items-center gap-3 text-muted-foreground">
          <KindIcon item={item} className="size-16" />
          <p className="text-sm">No preview for this file type.</p>
        </div>
      );
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
  // Tagged with its item, so a draft never follows the preview to another file.
  const [edit, setEdit] = useState<{ itemId: string; text: string } | null>(null);
  // The same draft, readable after an await: a save must not return while typing moved past it.
  const latestEdit = useRef(edit);
  const draft = item && edit?.itemId === item.id ? edit.text : null;
  const setDraft = (text: string | null) => {
    const next = item && text !== null ? { itemId: item.id, text } : null;
    latestEdit.current = next;
    setEdit(next);
  };
  const [saving, setSaving] = useState(false);
  // Tagged too, and cleared on close, so every file opens on its preview at 100%.
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

  async function save(): Promise<boolean> {
    if (!item || draft === null) return true;
    setSaving(true);
    try {
      // Typing during a save makes a newer draft; send that too before anything moves on.
      let sent = draft;
      for (;;) {
        await writeLibraryText(item.id, sent);
        const latest = latestEdit.current;
        if (latest?.itemId !== item.id || latest.text === sent) break;
        sent = latest.text;
      }
      if (latestEdit.current?.text === sent) latestEdit.current = null;
      setEdit((current) => (current?.itemId === item.id && current.text === sent ? null : current));
      onSaved();
      return true;
    } catch (error) {
      toast.error("Could not save", {
        description: error instanceof Error ? error.message : String(error),
      });
      return false;
    } finally {
      setSaving(false);
    }
  }

  // Leaving the preview any other way saves first too.
  async function saveThen(action: () => void) {
    if (await save()) action();
  }

  // Closing a note saves it, so an edit is never lost to a stray Escape.
  async function handleOpenChange(open: boolean) {
    if (!open && !(await save())) return;
    onOpenChange(open);
  }

  const meta = item
    ? [
        item.model ? modelLabel(item) : item.source === "generated" ? "Generated" : "Uploaded",
        formatSize(item.sizeBytes),
        formatCardTime(item.updatedAt),
      ].filter(Boolean)
    : [];
  const body = item ? bodyFor(item) : "none";
  const media = body === "image" || body === "video";

  return (
    <MediaViewer
      open={item !== null}
      onOpenChange={(open) => void handleOpenChange(open)}
      title={item?.name ?? ""}
      meta={meta.join(" · ")}
      media={media}
      noun={media ? body : "file"}
      onKeyDown={(event) => {
        if ((event.metaKey || event.ctrlKey) && event.key === "s") {
          event.preventDefault();
          void save();
        }
      }}
      extra={
        <>
          {body === "web" && item && !showCode && (
            <ScaleMenu
              label={`${Math.round(pageScale * 100)}%`}
              value={String(pageScale)}
              options={PAGE_SCALES.map((scale) => ({
                value: String(scale),
                label: `${Math.round(scale * 100)}%`,
              }))}
              onChange={(value) => setZoom({ itemId: item.id, scale: Number(value) })}
            />
          )}
          {body === "web" && item && (
            <div className="mr-1 flex items-center gap-1">
              <ViewButton label="Code" active={showCode} onClick={() => setCodeFor(item.id)}>
                <CodeToggleIcon className="size-4.5" />
              </ViewButton>
              <ViewButton label="Preview" active={!showCode} onClick={() => setCodeFor(null)}>
                <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} className="size-5" />
              </ViewButton>
            </div>
          )}
          {draft !== null && (
            <Button variant="dark" size="sm" className="mr-1" disabled={saving} onClick={() => void save()}>
              Save
            </Button>
          )}
        </>
      }
      actions={
        item
          ? {
              primary: {
                label: item.model ? "Chat with this model" : "Chat about this",
                icon: MessageCircleIcon,
                disabled: saving,
                onClick: () => void saveThen(() => onChat(item)),
              },
              onDownload: isFileItem(item) ? () => void saveThen(() => onDownload(item)) : undefined,
              viewOriginal: item.threadId
                ? {
                    label: "View original chat",
                    onClick: () => void saveThen(() => onOpenThread(item.threadId!)),
                  }
                : origin
                  ? {
                      label: origin.label,
                      onClick: () => void navigate({ to: origin.to, search: origin.search }),
                    }
                  : undefined,
              reveal:
                revealLabel && canReveal(item)
                  ? { label: revealLabel, onClick: () => revealInFolder(item.id) }
                  : undefined,
              favorite: item.favorite,
              onToggleFavorite: () => onToggleFavorite(item),
              onAddToProject: canAddToProject(item)
                ? async (projectId) => {
                    // The project gets the text on screen, not the last saved copy.
                    if (!(await save())) throw new Error("Save the note first.");
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
          draft={draft}
          onDraftChange={setDraft}
          showCode={showCode}
          pageScale={pageScale}
        />
      )}
    </MediaViewer>
  );
}
