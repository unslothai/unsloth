// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect } from "react";
import { useRagStore } from "../stores/rag-store";

export function ThreadIndexList() {
  const threadIndexes = useRagStore((s) => s.threadIndexes);
  const loading = useRagStore((s) => s.threadIndexesLoading);
  const loadThreadIndexes = useRagStore((s) => s.loadThreadIndexes);
  const clearThreadIndex = useRagStore((s) => s.clearThreadIndex);

  useEffect(() => {
    void loadThreadIndexes();
  }, [loadThreadIndexes]);

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium">Thread documents</h3>
        <span className="text-xs text-muted-foreground">
          {loading
            ? "Loading…"
            : `${threadIndexes.length} thread${threadIndexes.length === 1 ? "" : "s"}`}
        </span>
      </div>
      <p className="text-xs text-muted-foreground">
        Documents attached directly to a chat thread. Deleting a thread (from
        the sidebar) also wipes its index.
      </p>
      <ScrollArea className="max-h-[160px]">
        <div className="flex flex-col gap-1 pr-2">
          {threadIndexes.length === 0 && !loading ? (
            <div className="rounded-md border border-dashed border-border/60 px-3 py-4 text-center text-xs text-muted-foreground">
              No threads have attached documents.
            </div>
          ) : null}
          {threadIndexes.map((t) => (
            <div
              key={t.thread_id}
              className="flex items-center justify-between gap-2 rounded-md border border-border/60 px-3 py-2"
            >
              <div className="flex min-w-0 flex-col">
                <span className="truncate text-sm font-medium">
                  {t.title ?? <em className="font-normal">Unsaved thread</em>}
                </span>
                <span className="text-xs text-muted-foreground">
                  {t.num_documents} document
                  {t.num_documents === 1 ? "" : "s"} · {t.num_chunks} chunks
                </span>
              </div>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Clear thread index"
                className="text-muted-foreground hover:text-destructive"
                onClick={() => {
                  if (
                    window.confirm(
                      `Delete all ${t.num_documents} document${t.num_documents === 1 ? "" : "s"} from this thread's index? This cannot be undone.`,
                    )
                  ) {
                    void clearThreadIndex(t.thread_id);
                  }
                }}
              >
                <HugeiconsIcon icon={Delete02Icon} size={14} />
              </Button>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
