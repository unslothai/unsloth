// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { useEffect, useState } from "react";
import { type NodeHistoryEntry, nodeHistory } from "../api";
import { useDocStore } from "../stores/doc-store";
import {
  historyPreview,
  sameOutputs,
  usePinsStore,
} from "../stores/pins-store";
import { useRuntimeStore } from "../stores/runtime-store";
import { MediaThumb } from "./media";

export function NodeHistory({
  nodeId,
  onView,
  canPin = true,
}: {
  nodeId: string;
  onView: (outputs: Record<string, unknown> | null) => void;
  canPin?: boolean;
}) {
  const flowId = useDocStore((s) => s.flowId);
  const runStatus = useRuntimeStore((s) => s.runStatus);
  const pinned = usePinsStore((s) => s.pins[nodeId]);
  const [open, setOpen] = useState(false);
  const [entries, setEntries] = useState<NodeHistoryEntry[] | null>(null);
  const [viewing, setViewing] = useState<number | null>(null);

  // biome-ignore lint/correctness/useExhaustiveDependencies: runStatus re-fetches after each run, which adds an entry
  useEffect(() => {
    if (!open || !flowId) return;
    let alive = true;
    void nodeHistory(flowId, nodeId)
      .then((res) => {
        if (!alive) return;
        setEntries(res.entries);
        setViewing(null);
        onView(null);
      })
      .catch(() => alive && setEntries([]));
    return () => {
      alive = false;
    };
  }, [open, flowId, nodeId, runStatus, onView]);

  // biome-ignore lint/correctness/useExhaustiveDependencies: a different node starts with nothing viewed
  useEffect(() => {
    setEntries(null);
    setViewing(null);
    onView(null);
  }, [nodeId, onView]);

  return (
    <section className="flex flex-col gap-2 border-border border-t px-4 py-3">
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground"
          data-testid="flows-history-toggle"
        >
          History {open ? "▾" : "▸"}
        </button>
        {pinned && (
          <span className="rounded bg-amber-500/15 px-1.5 py-0.5 text-[10.5px] text-amber-700 dark:text-amber-200">
            Pinned
          </span>
        )}
        {pinned && (
          <button
            type="button"
            onClick={() => usePinsStore.getState().setPin(nodeId, null)}
            className="ml-auto text-[11px] text-muted-foreground hover:text-foreground"
          >
            Unpin
          </button>
        )}
      </div>
      {open && entries && entries.length === 0 && (
        <p className="text-[11.5px] text-muted-foreground">
          No earlier outputs yet. Each run of this block adds one.
        </p>
      )}
      {open && entries && entries.length > 0 && (
        <div
          className="grid grid-cols-5 gap-1.5"
          data-testid="flows-history-strip"
        >
          {entries.map((entry, index) => {
            const preview = historyPreview(entry.outputs);
            const isPinned = !!pinned && sameOutputs(pinned, entry.outputs);
            return (
              <button
                key={entry.run_id}
                type="button"
                title={new Date(entry.updated_at).toLocaleString()}
                onClick={() => {
                  const next = viewing === index ? null : index;
                  setViewing(next);
                  onView(next === null ? null : entry.outputs);
                }}
                className={`relative aspect-square overflow-hidden rounded border text-left ${
                  isPinned
                    ? "border-amber-500"
                    : viewing === index
                      ? "border-ring-strong"
                      : "border-border hover:border-ring"
                }`}
              >
                {"media" in preview ? (
                  preview.media.kind === "image" ? (
                    <MediaThumb
                      media={preview.media}
                      className="h-full w-full object-cover"
                    />
                  ) : (
                    <span className="flex h-full items-center justify-center text-[10px] uppercase text-muted-foreground">
                      {preview.media.kind}
                    </span>
                  )
                ) : (
                  <span className="line-clamp-4 p-1 text-[9.5px] leading-tight text-foreground">
                    {preview.text}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
      {open && viewing !== null && entries?.[viewing] && !canPin && (
        <span className="text-[11px] text-muted-foreground">
          Blocks inside a For Each run every item, so they cannot be pinned.
        </span>
      )}
      {open && viewing !== null && entries?.[viewing] && canPin && (
        <div className="flex gap-2">
          <Button
            type="button"
            variant="outline"
            size="xs"
            onClick={() =>
              usePinsStore.getState().setPin(nodeId, entries[viewing].outputs)
            }
            data-testid="flows-history-pin"
          >
            Pin this output
          </Button>
          <span className="self-center text-[11px] text-muted-foreground">
            Pinned outputs feed the next blocks instead of running this one.
          </span>
        </div>
      )}
    </section>
  );
}
