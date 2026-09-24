// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Saved runs as Train's history cards: what was swept, on what, what won and by how much.

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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { formatRelativeTime, useLocale } from "@/i18n";
import { cn } from "@/lib/utils";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { type ReactElement, useMemo, useState } from "react";
import type { BenchRunSummary } from "../api/bench-runs-api";
import {
  type Family,
  SWEEP_TITLE,
  familyOf,
  ctxNote,
  inSentence,
  modelShort,
  ranToEnd,
} from "../lib/bench-math";
import { useBenchmarksStore } from "../stores/benchmarks-store";
import { useFamilyColors } from "./family-colors";

const ALL = "\u0000all";

const BADGE = {
  done: "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-400",
  partial: "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-400",
  running: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-400",
};

interface Scored {
  label: string;
  family: Family;
  tps: number;
  isBase: boolean;
}

/** Rows fastest first with the baseline kept apart, from the list's per-row means. */
function scoreRun(run: BenchRunSummary): {
  rows: Scored[];
  best: Scored | null;
  base: Scored | null;
  speedup: number | null;
} {
  const families = new Map(
    run.config.variants.map((v) => [v.label, familyOf(v.load)]),
  );
  const rows = Object.entries(run.rowMeans ?? {})
    .map(([label, tps]) => ({
      label,
      tps,
      family: families.get(label) ?? ("other" as Family),
      isBase: label === run.config.baseline,
    }))
    .sort((a, b) => b.tps - a.tps);
  const base = rows.find((r) => r.isBase) ?? null;
  const best = rows.find((r) => !r.isBase) ?? base;
  const speedup =
    best && base && best !== base && base.tps > 0 ? best.tps / base.tps : null;
  return { rows, best, base, speedup };
}

function ago(ms: number, locale: ReturnType<typeof useLocale>): string {
  const mins = Math.floor((Date.now() - ms) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return formatRelativeTime(locale, -mins, "minute");
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return formatRelativeTime(locale, -hrs, "hour");
  return formatRelativeTime(locale, -Math.floor(hrs / 24), "day");
}

/** Every row as a bar, fastest left, the baseline muted: the chart's shape at a glance. */
function MiniBars({ rows }: { rows: Scored[] }): ReactElement | null {
  const colors = useFamilyColors();
  if (rows.length < 2) return null;
  const top = Math.max(...rows.map((r) => r.tps));
  return (
    <span className="flex h-8 items-end gap-[2px]" aria-hidden={true}>
      {rows.slice(0, 24).map((r) => (
        <span
          key={r.label}
          className="min-w-[3px] flex-1 rounded-t-[2px]"
          style={{
            height: `${Math.max(8, (r.tps / top) * 100)}%`,
            background: r.isBase ? "var(--muted-foreground)" : colors[r.family],
            opacity: r.isBase ? 0.45 : 0.9,
          }}
        />
      ))}
    </span>
  );
}

export function HistoryGrid({
  onOpen,
}: {
  onOpen: (id: string) => void;
}): ReactElement {
  const runs = useBenchmarksStore((s) => s.runs);
  const runsLoaded = useBenchmarksStore((s) => s.runsLoaded);
  const live = useBenchmarksStore((s) => s.live);
  const selectedRunId = useBenchmarksStore((s) => s.selectedRunId);
  const deleteRun = useBenchmarksStore((s) => s.deleteRun);
  const locale = useLocale();
  const [model, setModel] = useState(ALL);
  const [sweep, setSweep] = useState(ALL);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const models = useMemo(
    () => [...new Set(runs.map((r) => modelShort(r.model)))],
    [runs],
  );
  const sweeps = useMemo(
    () => [...new Set(runs.map((r) => r.config.sweep))],
    [runs],
  );
  const shown = runs.filter(
    (r) =>
      (model === ALL || modelShort(r.model) === model) &&
      (sweep === ALL || r.config.sweep === sweep),
  );

  if (!runsLoaded) {
    return (
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3 4xl:grid-cols-4">
        {["a", "b", "c"].map((k) => (
          <div
            key={k}
            className="h-40 animate-pulse rounded-xl border bg-muted/30"
          />
        ))}
      </div>
    );
  }
  if (runs.length === 0) {
    return (
      <p className="py-16 text-center text-sm text-muted-foreground">
        No saved runs yet. Every benchmark is kept here as it runs.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex flex-wrap items-center gap-2">
        {models.length > 1 && (
          <Select value={model} onValueChange={setModel}>
            <SelectTrigger className="h-9 w-auto min-w-[calc(180px*var(--ui-space-scale,1))]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL}>All models</SelectItem>
              {models.map((m) => (
                <SelectItem key={m} value={m}>
                  {m}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        {sweeps.length > 1 && (
          <Select value={sweep} onValueChange={setSweep}>
            <SelectTrigger className="h-9 w-auto min-w-[calc(180px*var(--ui-space-scale,1))]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL}>All sweeps</SelectItem>
              {sweeps.map((s) => (
                <SelectItem key={s} value={s}>
                  {SWEEP_TITLE[s]}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
        <span className="ml-auto text-ui-12 tabular-nums text-muted-foreground">
          {shown.length === runs.length
            ? `${runs.length} run${runs.length === 1 ? "" : "s"}`
            : `${shown.length} of ${runs.length} runs`}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-3 4xl:grid-cols-4">
        {shown.map((run) => {
          const { rows, best, base, speedup } = scoreRun(run);
          const running = live?.run.id === run.id;
          const state = running
            ? "running"
            : run.finishedAt && ranToEnd(run)
              ? "done"
              : "partial";
          const rowsTotal = run.config.variants.length;
          return (
            <div
              key={run.id}
              className={cn(
                "elevated-card group relative bg-card transition-colors hover:bg-accent/30",
                run.id === selectedRunId &&
                  "ring-2 ring-[color-mix(in_oklab,var(--foreground)_calc(25%*var(--contrast-edge-gain,1)),transparent)]",
              )}
            >
              <button
                type="button"
                onClick={() => onOpen(run.id)}
                className="flex h-full w-full flex-col gap-3 rounded-[inherit] p-4 text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <span className="flex items-center justify-between gap-2 pr-6">
                  <span
                    className={cn(
                      "inline-flex items-center rounded-full px-2 py-0.5 text-ui-10 font-semibold",
                      BADGE[state],
                    )}
                  >
                    {state === "running"
                      ? "Running"
                      : state === "done"
                        ? "Complete"
                        : "Stopped early"}
                  </span>
                  <span
                    className="text-ui-10 text-muted-foreground"
                    title={new Date(run.createdAt).toLocaleString()}
                  >
                    {ago(run.createdAt, locale)}
                  </span>
                </span>
                <span className="block min-w-0">
                  <span className="block truncate text-sm font-medium">
                    {SWEEP_TITLE[run.config.sweep]}
                  </span>
                  <span
                    className="block truncate text-xs text-muted-foreground"
                    title={run.model}
                  >
                    {modelShort(run.model)}
                    {run.ggufVariant ? ` · ${run.ggufVariant}` : ""}
                  </span>
                  <span className="block truncate text-xs text-muted-foreground/80">
                    {[
                      run.kv && `KV ${run.kv}`,
                      ctxNote(run),
                      run.meta.gpu,
                    ]
                      .filter(Boolean)
                      .join(" · ")}
                  </span>
                </span>
                <MiniBars rows={rows} />
                <span className="flex flex-col gap-0.5 text-ui-11 text-muted-foreground">
                  {best ? (
                    <span className="flex items-baseline gap-1.5">
                      <span className="font-semibold tabular-nums text-foreground">
                        {best.tps.toFixed(1)} tok/s
                      </span>
                      <span className="truncate" title={best.label}>
                        {best.label}
                      </span>
                    </span>
                  ) : (
                    <span>No measured rows</span>
                  )}
                  <span className="flex flex-wrap gap-x-3 tabular-nums">
                    {speedup !== null && base && (
                      <span>
                        {speedup.toFixed(2)}× {inSentence(base.label)}
                      </span>
                    )}
                    <span>
                      {rows.length} of {rowsTotal} rows
                    </span>
                  </span>
                </span>
              </button>
              {!running && (
                <button
                  type="button"
                  className="absolute right-3 top-3 z-10 rounded-md p-1 text-muted-foreground/50 opacity-0 transition-opacity hover:bg-destructive/10 hover:text-destructive group-hover:opacity-100 focus-visible:opacity-100 pointer-coarse:opacity-100"
                  aria-label="Delete this run"
                  onClick={() => setDeleteTarget(run.id)}
                >
                  <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
                </button>
              )}
            </div>
          );
        })}
      </div>

      <AlertDialog
        open={deleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setDeleteTarget(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete this run?</AlertDialogTitle>
            <AlertDialogDescription>
              Its measurements are removed from studio.db. This can't be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                if (deleteTarget) void deleteRun(deleteTarget);
                setDeleteTarget(null);
              }}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
