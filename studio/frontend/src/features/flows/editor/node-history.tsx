// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CollapsibleSectionTriggerButton } from "@/features/recipe-studio/dialogs/shared/collapsible-section-trigger";
import { RECIPE_STUDIO_WARNING_BADGE_TONE } from "@/features/recipe-studio/utils/ui-tones";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import { type NodeHistoryEntry, nodeHistory } from "../api";
import { useDocStore } from "../stores/doc-store";
import {
  type HistoryKind,
  historyAgo,
  historyCard,
  sameOutputs,
  usePinsStore,
} from "../stores/pins-store";
import { useRuntimeStore } from "../stores/runtime-store";
import {
  AudioWave01Icon,
  CodeIcon,
  Database01Icon,
  HashtagIcon,
  Image01Icon,
  LeftToRightListBulletIcon,
  TextIcon,
  Video01Icon,
} from "./icon-data";
import { MediaThumb } from "./media";

const KIND_ICON: Record<HistoryKind, typeof TextIcon> = {
  image: Image01Icon,
  audio: AudioWave01Icon,
  video: Video01Icon,
  text: TextIcon,
  number: HashtagIcon,
  list: LeftToRightListBulletIcon,
  json: CodeIcon,
  dataset: Database01Icon,
};

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
    <section className="flex flex-col gap-2 border-border/60 border-t px-4 py-3">
      <CollapsibleSectionTriggerButton
        label="History"
        open={open}
        summary={pinned ? "Pinned" : undefined}
        onClick={() => setOpen((v) => !v)}
        data-testid="flows-history-toggle"
      />
      {pinned && (
        <div className="flex items-center gap-2">
          <Badge variant="outline" className={RECIPE_STUDIO_WARNING_BADGE_TONE}>
            Pinned
          </Badge>
          <Button
            type="button"
            variant="ghost"
            size="xs"
            onClick={() => usePinsStore.getState().setPin(nodeId, null)}
            className="ml-auto text-muted-foreground"
          >
            Unpin
          </Button>
        </div>
      )}
      {open && entries && entries.length === 0 && (
        <p className="text-xs text-muted-foreground">
          No earlier outputs yet. Each run of this block adds one.
        </p>
      )}
      {open && entries && entries.length > 0 && (
        <div
          className="grid grid-cols-3 gap-2"
          data-testid="flows-history-strip"
        >
          {entries.map((entry, index) => {
            const card = historyCard(entry.outputs);
            const ago = historyAgo(entry.updated_at);
            const isPinned = !!pinned && sameOutputs(pinned, entry.outputs);
            return (
              // Library's card in miniature: a picture fills it, anything else is name, type icon and detail.
              <button
                key={entry.run_id}
                type="button"
                title={`${card.title}\n${new Date(entry.updated_at).toLocaleString()}`}
                onClick={() => {
                  const next = viewing === index ? null : index;
                  setViewing(next);
                  onView(next === null ? null : entry.outputs);
                }}
                className={`relative aspect-square overflow-hidden rounded-xl text-left shadow-[0_1px_2px_rgba(0,0,0,0.04),0_4px_16px_rgba(0,0,0,0.06)] ring-offset-1 ring-offset-background transition dark:shadow-none ${
                  card.media?.kind === "image" ? "bg-muted" : "bg-card"
                } ${
                  isPinned
                    ? "ring-2 ring-amber-500"
                    : viewing === index
                      ? "ring-2 ring-emerald-500/80"
                      : "ring-1 ring-border/60 hover:ring-emerald-500/60"
                }`}
                data-kind={card.kind}
              >
                {card.media?.kind === "image" ? (
                  <>
                    <MediaThumb
                      media={card.media}
                      className="h-full w-full object-cover"
                    />
                    <span className="absolute bottom-1 left-1 rounded-full bg-black/55 px-1.5 py-0.5 text-[10px] leading-none text-white tabular-nums">
                      {ago}
                    </span>
                  </>
                ) : (
                  <span className="grid h-full grid-rows-[auto_1fr_auto] p-2">
                    <span className="line-clamp-2 break-words font-medium text-ui-11 leading-snug text-foreground">
                      {card.title}
                    </span>
                    <span className="flex items-center justify-center">
                      <HugeiconsIcon
                        icon={KIND_ICON[card.kind]}
                        strokeWidth={1.5}
                        className="size-6 text-muted-foreground"
                      />
                    </span>
                    <span className="truncate text-[10px] tabular-nums text-muted-foreground">
                      {card.detail} · {ago}
                    </span>
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
      {open && viewing !== null && entries?.[viewing] && !canPin && (
        <span className="text-xs text-muted-foreground">
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
          <span className="self-center text-xs text-muted-foreground">
            Pinned outputs feed the next blocks instead of running this one.
          </span>
        </div>
      )}
    </section>
  );
}
