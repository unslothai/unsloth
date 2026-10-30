// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  Archive02Icon,
  Delete02Icon,
  Download01Icon,
  ClosedCaptionIcon,
  MoreVerticalIcon,
} from "@hugeicons/core-free-icons";
import { ChevronDownStandardIcon } from "@/lib/chevron-icons";
import {
  CLIPS_HEADER_LINK,
  CLIPS_LIST,
  ClipsCard,
  ClipsCardEmpty,
} from "./components/clips-card";
import { Copy01Icon } from "@/lib/copy-icon";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { downloadTranscript } from "./transcript-download";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { toast } from "@/lib/toast";
import { cn } from "@/lib/utils";
import { archiveTranscript, deleteTranscript, listTranscripts } from "./api";
import {
  type ClipSendHandlers,
  ClipSendToMenu,
} from "./components/clip-card";
import { getTranscript } from "./transcribe-api";
import {
  TRANSCRIPT_EXPORT_FORMATS,
  type TranscriptExportFormat,
} from "./transcript-export";
import {
  type TranscriptSource,
  detailsFrom,
  formatTimestamp,
} from "./transcript-model";
import type { TranscriptRecord } from "./transcript-stream";

export function TranscriptExportItems({
  timed,
  label,
  onExport,
}: {
  timed: boolean;
  label: (format: TranscriptExportFormat) => ReactNode;
  onExport: (format: TranscriptExportFormat) => void;
}) {
  return TRANSCRIPT_EXPORT_FORMATS.map((format) => {
    const blocked = !timed && (format === "srt" || format === "vtt");
    return (
      <DropdownMenuItem
        key={format}
        disabled={blocked}
        onClick={() => onExport(format)}
      >
        {label(format)}
        {blocked ? (
          <span className="ml-auto pl-3 text-ui-11p5 text-muted-foreground">
            Needs timestamps
          </span>
        ) : null}
      </DropdownMenuItem>
    );
  });
}

// The list carries counts only, so timed formats fetch the full record.
async function downloadRecord(
  record: TranscriptRecord,
  format: TranscriptExportFormat,
): Promise<void> {
  let full = record;
  if ((record.segment_count ?? 0) > 0) {
    try {
      full = await getTranscript(record.id);
    } catch (error) {
      toast.error(
        error instanceof Error ? error.message : "Could not load transcript.",
      );
      return;
    }
  }
  await downloadTranscript(format, {
    title: full.title,
    text: full.text,
    model: full.model,
    details: detailsFrom(full),
    names: full.speaker_names ?? {},
  });
}

function recordBadge(record: TranscriptRecord): string | null {
  const speakers = record.speakers?.length ?? 0;
  if (speakers > 0)
    return speakers === 1 ? "1 speaker" : `${speakers} speakers`;
  return (record.segment_count ?? 0) > 0 ? "Timestamps" : null;
}

export type TranscriptSendHandlers = (transcript: {
  text: string;
  source?: TranscriptSource | null;
  duration: number | null;
}) => ClipSendHandlers;

export function TranscriptGallery({
  active,
  currentId,
  latest,
  onSelect,
  onDelete,
  canSelect,
  autoSelect,
  sendHandlersFor,
  top,
  emptyText,
}: {
  /** The card's expanded top row: the open transcript, its progress, or an error. */
  top: ReactNode;
  /** Shown under the title before the first transcript. */
  emptyText: string;
  active: boolean;
  autoSelect: boolean;
  currentId: string | null;
  latest: TranscriptRecord | null;
  onSelect: (record: TranscriptRecord) => void;
  onDelete: (ids: string[] | null) => void;
  canSelect: () => boolean;
  sendHandlersFor: TranscriptSendHandlers;
}) {
  const [records, setRecords] = useState<TranscriptRecord[]>([]);
  const [archived, setArchived] = useState(false);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [confirmingClear, setConfirmingClear] = useState(false);
  const generation = useRef(0);
  const loadingRef = useRef(false);
  const currentIdRef = useRef(currentId);
  const selectionRef = useRef({ autoSelect, onSelect, onDelete });
  useLayoutEffect(() => {
    currentIdRef.current = currentId;
    selectionRef.current = { autoSelect, onSelect, onDelete };
  }, [currentId, autoSelect, onSelect, onDelete]);
  const refresh = useCallback(() => {
    const ticket = ++generation.current;
    loadingRef.current = true;
    return listTranscripts(archived)
      .then((page) => {
        if (ticket !== generation.current) return;
        setRecords(page.transcripts);
        setCursor(page.next_cursor);
        if (
          selectionRef.current.autoSelect &&
          !currentIdRef.current &&
          page.transcripts[0]
        ) {
          selectionRef.current.onSelect(page.transcripts[0]);
        }
      })
      .catch((error: unknown) => {
        if (ticket !== generation.current) return;
        // Rows and cursor belong to the view just left; keeping them mislabels and mispages.
        setRecords([]);
        setCursor(null);
        toast.error(
          error instanceof Error ? error.message : "Could not load transcripts.",
        );
      })
      .finally(() => {
        if (ticket === generation.current) {
          loadingRef.current = false;
          setLoading(false);
        }
      });
  }, [archived]);
  const refreshRef = useRef(refresh);
  useLayoutEffect(() => {
    refreshRef.current = refresh;
  }, [refresh]);
  // biome-ignore lint/correctness/useExhaustiveDependencies: `latest` is the trigger; a newly saved transcript refetches the list.
  useEffect(() => {
    if (active) void refresh();
    return () => {
      generation.current += 1;
    };
  }, [active, latest, refresh]);
  const loadMore = async () => {
    if (!cursor || loadingRef.current) return;
    const ticket = generation.current;
    loadingRef.current = true;
    setLoading(true);
    try {
      const page = await listTranscripts(archived, cursor);
      if (ticket !== generation.current) return;
      setRecords((current) => [
        ...current,
        ...page.transcripts.filter(
          (record) => !current.some((item) => item.id === record.id),
        ),
      ]);
      setCursor(page.next_cursor);
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : "Could not load more transcripts.",
      );
    } finally {
      if (ticket === generation.current) {
        loadingRef.current = false;
        setLoading(false);
      }
    }
  };
  const mutate = async (
    action: () => Promise<void>,
    removed: string[] | null,
  ) => {
    try {
      await action();
      if (removed === null || removed.includes(currentIdRef.current ?? ""))
        selectionRef.current.onDelete(removed);
      await refreshRef.current();
    } catch (error) {
      toast.error(
        error instanceof Error
          ? error.message
          : "Could not update transcript history.",
      );
    }
  };
  return (
    // `contents`: the card is the layout box; this wrapper only holds it and the confirm dialog.
    <div className="contents">
      <ClipsCard
        label="Transcripts"
        title={
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                className="-mx-1.5 inline-flex items-center gap-1 rounded-full px-1.5 py-0.5 transition-colors hover:bg-wash-5 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {archived ? "Archived transcripts" : "Transcripts"}
                <HugeiconsIcon
                  icon={ChevronDownStandardIcon}
                  className="size-3.5 text-muted-foreground"
                />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="start">
              <DropdownMenuItem onClick={() => setArchived(false)}>
                Transcripts
              </DropdownMenuItem>
              <DropdownMenuItem onClick={() => setArchived(true)}>
                Archived transcripts
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        }
        count={records.length}
        countMore={Boolean(cursor)}
        action={
          !archived && records.length > 0 ? (
            <Button
              type="button"
              variant="ghost"
              onClick={() => setConfirmingClear(true)}
              className={cn(
                CLIPS_HEADER_LINK,
                "h-auto px-0 py-0 font-normal hover:bg-transparent dark:hover:bg-transparent",
              )}
            >
              Clear all
            </Button>
          ) : null
        }
      >
        {top}
        {!top && !archived && !loading && records.length === 0 ? (
          <ClipsCardEmpty text={emptyText} />
        ) : null}
        {/* The open transcript is the top row, so the list leaves it out. */}
        {records.some((record) => !top || record.id !== currentId) ||
        (archived && !loading) ||
        cursor ? (
      <div
        className={cn(CLIPS_LIST, !top && "mt-0 border-t-0 pt-0")}
        onScroll={(event) => {
          const el = event.currentTarget;
          if (el.scrollTop + el.clientHeight >= el.scrollHeight - 40)
            void loadMore();
        }}
      >
        {records
          .filter((record) => !top || record.id !== currentId)
          .map((record) => (
          <div
            key={record.id}
            className="group flex items-center gap-1 rounded-md ps-3 pe-1 transition-colors hover:bg-wash-3"
          >
            {/* Same 40px column and title edge as the card's top row. */}
            <span
              aria-hidden="true"
              className="grid w-[calc(40px*var(--ui-space-scale,1))] shrink-0 place-items-center text-muted-foreground"
            >
              <HugeiconsIcon icon={ClosedCaptionIcon} className="size-3.5" />
            </span>
            <button
              type="button"
              className="flex min-w-0 flex-1 items-center gap-3 py-2.5 pr-3 ps-2 text-left"
              onClick={() => {
                if (canSelect()) onSelect(record);
              }}
            >
              <span className="grid min-w-0 flex-1 gap-0.5">
                <span className="truncate text-ui-13" title={record.title}>
                  {record.title}
                </span>
                <span className="truncate text-ui-11p5 text-muted-foreground">
                  {new Date(record.created_at).toLocaleDateString()}
                  {recordBadge(record) ? ` · ${recordBadge(record)}` : ""}
                </span>
              </span>
              {record.duration ? (
                <span className="shrink-0 font-mono text-ui-11p5 tabular-nums text-muted-foreground">
                  {formatTimestamp(record.duration)}
                </span>
              ) : null}
            </button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-5 text-muted-foreground/60 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100 any-pointer-coarse:opacity-100"
                  title="More"
                  aria-label="Transcript actions"
                >
                  <HugeiconsIcon icon={MoreVerticalIcon} className="size-3.5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <ClipSendToMenu handlers={sendHandlersFor(record)} />
                <TranscriptExportItems
                  timed={(record.segment_count ?? 0) > 0}
                  label={(format) => (
                    <>
                      <HugeiconsIcon icon={Download01Icon} />
                      Download .{format}
                    </>
                  )}
                  onExport={(format) => void downloadRecord(record, format)}
                />
                <DropdownMenuItem
                  onClick={() =>
                    void copyToClipboard(record.text).then((ok) =>
                      ok
                        ? toast.success("Transcript copied")
                        : toast.error("Could not copy transcript."),
                    )
                  }
                >
                  <HugeiconsIcon icon={Copy01Icon} />
                  Copy
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onClick={() =>
                    void mutate(
                      () => archiveTranscript(record.id, !record.archived),
                      [record.id],
                    )
                  }
                >
                  <HugeiconsIcon icon={Archive02Icon} />
                  {record.archived ? "Unarchive" : "Archive"}
                </DropdownMenuItem>
                <DropdownMenuItem
                  variant="destructive"
                  onClick={() =>
                    void mutate(() => deleteTranscript(record.id), [record.id])
                  }
                >
                  <HugeiconsIcon icon={Delete02Icon} />
                  Delete
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        ))}
        {archived && !loading && !records.length ? (
          <p className="px-3 py-1.5 text-ui-13 text-muted-foreground">
            No archived transcripts.
          </p>
        ) : null}
        {cursor && (
          <Button
            variant="ghost"
            size="sm"
            disabled={loading}
            onClick={() => void loadMore()}
          >
            {loading ? "Loading…" : "Load more"}
          </Button>
        )}
      </div>
        ) : null}
      </ClipsCard>
      <AlertDialog open={confirmingClear} onOpenChange={setConfirmingClear}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Clear transcript history</AlertDialogTitle>
            <AlertDialogDescription>
              Delete all transcripts in history? Archived transcripts will be
              kept. This cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => void mutate(() => deleteTranscript(), null)}
            >
              Clear all
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
