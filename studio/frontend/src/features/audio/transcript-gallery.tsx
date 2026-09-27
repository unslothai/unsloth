// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { HugeiconsIcon } from "@hugeicons/react";
import {
  Archive02Icon,
  Copy01Icon,
  Delete02Icon,
  Download01Icon,
  MoreVerticalIcon,
} from "@hugeicons/core-free-icons";
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
import type { TranscriptRecord } from "./transcript-stream";

export function TranscriptGallery({
  active,
  currentId,
  latest,
  onSelect,
  onDelete,
  canSelect,
  autoSelect,
}: {
  active: boolean;
  autoSelect: boolean;
  currentId: string | null;
  latest: TranscriptRecord | null;
  onSelect: (record: TranscriptRecord) => void;
  onDelete: (ids: string[] | null) => void;
  canSelect: () => boolean;
}) {
  const [records, setRecords] = useState<TranscriptRecord[]>([]);
  const [archived, setArchived] = useState(false);
  const [cursor, setCursor] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
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
        // Rows and cursor belong to the view we just left: keeping them renders History
        // entries under the Archived heading and pages the wrong cursor onto them.
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
    <div className="flex shrink-0 flex-col gap-2">
      <div className="flex items-center justify-between">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="px-0 text-ui-11p5 text-muted-foreground"
            >
              {archived ? "Archived transcripts" : "History"}
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            <DropdownMenuItem onClick={() => setArchived(false)}>
              History
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => setArchived(true)}>
              Archived transcripts
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
        {!archived && records.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              if (
                window.confirm(
                  "Delete all transcripts in history? Archived transcripts will be kept.",
                )
              )
                void mutate(() => deleteTranscript(), null);
            }}
          >
            Clear all
          </Button>
        )}
      </div>
      <div
        className="hover-scrollbar flex max-h-40 flex-col gap-1 overflow-y-auto"
        onScroll={(event) => {
          const el = event.currentTarget;
          if (el.scrollTop + el.clientHeight >= el.scrollHeight - 40)
            void loadMore();
        }}
      >
        {records.map((record) => (
          <div
            key={record.id}
            className={cn(
              "group flex items-center rounded-md pr-1 transition-colors hover:bg-muted",
              currentId === record.id && "bg-muted",
            )}
          >
            <button
              type="button"
              aria-current={currentId === record.id ? "true" : undefined}
              className="flex min-w-0 flex-1 items-center gap-2 px-2 py-1.5 text-left text-ui-13"
              onClick={() => {
                if (canSelect()) onSelect(record);
              }}
            >
              <span className="min-w-0 flex-1 truncate">{record.title}</span>
              <span className="shrink-0 text-ui-11p5 text-muted-foreground">
                {new Date(record.created_at).toLocaleDateString()}
              </span>
            </button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-5 text-muted-foreground/60 opacity-0 group-hover:opacity-100 focus-visible:opacity-100 data-[state=open]:opacity-100 any-pointer-coarse:opacity-100"
                  aria-label="Transcript actions"
                >
                  <HugeiconsIcon icon={MoreVerticalIcon} className="size-3.5" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem
                  onClick={() =>
                    void downloadTranscript(record.text, record.title)
                  }
                >
                  <HugeiconsIcon icon={Download01Icon} />
                  Download .txt
                </DropdownMenuItem>
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
        {!records.length && (
          <p className="px-2 py-1.5 text-ui-13 text-muted-foreground">
            {loading
              ? "Loading transcripts…"
              : archived
                ? "No archived transcripts."
                : "Completed transcripts are saved here."}
          </p>
        )}
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
    </div>
  );
}
