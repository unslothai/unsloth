// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogTitle } from "@/components/ui/dialog";
import { Spinner } from "@/components/ui/spinner";
import { ArtifactHtmlFrame } from "@/features/chat";
import { isTauri } from "@/lib/api-base";
import { MessageCircleIcon } from "@/lib/hugeicons-derived";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import { type LibraryItem, fetchLibraryTextPrefix, writeLibraryText } from "../api";
import {
  fileKind,
  hasImagePreview,
  isFileItem,
  isTextPreviewable,
  modelLabel,
} from "../file-kind";
import { formatCardTime, formatSize } from "../format";
import { useLibraryObjectUrl } from "../hooks";
import { KindIcon } from "./library-cards";

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

function PreviewBody({
  item,
  draft,
  onDraftChange,
}: {
  item: LibraryItem;
  draft: string | null;
  onDraftChange: (value: string) => void;
}) {
  const body = bodyFor(item);
  const needsUrl = body === "image" || body === "pdf" || body === "audio" || body === "video";
  const url = useLibraryObjectUrl(item, needsUrl);
  const { text, truncated, error } = useItemText(item, body === "text" || body === "web");

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
      return <video src={url!} controls className="size-full object-contain" />;
    case "web":
      if (truncated) return <TextPrefix text={`${text!}\n\n…`} />;
      // The chat canvas frame: served by the backend under its own CSP, so it renders the same in
      // the browser and the desktop app, and honors the canvas network-access setting.
      return (
        <div className="size-full overflow-hidden rounded-xl">
          <ArtifactHtmlFrame code={text!} title={item.name} fill />
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
  onSaved,
}: {
  item: LibraryItem | null;
  onOpenChange: (open: boolean) => void;
  onChat: (item: LibraryItem) => void;
  onDownload: (item: LibraryItem) => void;
  onOpenThread: (threadId: string) => void;
  onSaved: () => void;
}) {
  // Tagged with its item, so a draft never follows the preview to another file.
  const [edit, setEdit] = useState<{ itemId: string; text: string } | null>(null);
  const draft = item && edit?.itemId === item.id ? edit.text : null;
  const setDraft = (text: string | null) =>
    setEdit(item && text !== null ? { itemId: item.id, text } : null);
  const [saving, setSaving] = useState(false);

  async function save(): Promise<boolean> {
    if (!item || draft === null) return true;
    setSaving(true);
    try {
      await writeLibraryText(item.id, draft);
      setDraft(null);
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

  const media = item !== null && (bodyFor(item) === "image" || bodyFor(item) === "video");
  const meta = item
    ? [
        item.model ? modelLabel(item) : item.source === "generated" ? "Generated" : "Uploaded",
        formatSize(item.sizeBytes),
        formatCardTime(item.updatedAt),
      ].filter(Boolean)
    : [];

  return (
    <Dialog open={item !== null} onOpenChange={(open) => void handleOpenChange(open)}>
      <DialogContent
        className={cn(
          "flex max-w-none flex-col gap-0 p-0 sm:max-w-none",
          // Images and videos open near full screen; everything else keeps a reading width.
          media ? "h-[calc(100dvh-2rem)] w-[calc(100vw-2rem)]" : "h-[min(88vh,960px)] w-[min(92vw,1200px)]",
        )}
        onKeyDown={(event) => {
          if ((event.metaKey || event.ctrlKey) && event.key === "s") {
            event.preventDefault();
            void save();
          }
        }}
      >
        {item && (
          <>
            <div className="flex items-center gap-4 border-b border-border/60 py-4 pl-6 pr-14">
              <div className="min-w-0 flex-1">
                <DialogTitle className="truncate text-[17px]">{item.name}</DialogTitle>
                <DialogDescription className="mt-0.5 truncate text-[13px]">
                  {meta.join(" · ")}
                  {item.threadId && item.threadTitle && (
                    <>
                      {" · "}
                      <button
                        type="button"
                        onClick={() => void saveThen(() => onOpenThread(item.threadId!))}
                        className="underline-offset-2 hover:text-foreground hover:underline"
                      >
                        {item.threadTitle}
                      </button>
                    </>
                  )}
                </DialogDescription>
              </div>
              {draft !== null && (
                <Button variant="dark" size="sm" disabled={saving} onClick={() => void save()}>
                  Save
                </Button>
              )}
              <Button variant="ghost" size="sm" disabled={saving} onClick={() => void saveThen(() => onChat(item))}>
                <HugeiconsIcon icon={MessageCircleIcon} strokeWidth={1.75} className="size-4" />
                {item.model ? "Chat with this model" : "Chat about this"}
              </Button>
              {isFileItem(item) && (
                <Button variant="ghost" size="sm" onClick={() => onDownload(item)}>
                  <HugeiconsIcon icon={Download01Icon} strokeWidth={1.75} className="size-4" />
                  Download
                </Button>
              )}
            </div>
            <div className={cn("flex min-h-0 flex-1", media ? "p-3" : "p-6")}>
              <PreviewBody item={item} draft={draft} onDraftChange={setDraft} />
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  );
}
