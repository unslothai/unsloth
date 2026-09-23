// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { DownloadCancelledError, downloadFile } from "@/lib/native-files";
import { cn } from "@/lib/utils";
import { type ReactElement, type ReactNode, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import {
  type AggRow,
  type BenchRun,
  FAMILY_LABEL,
  type Family,
  SWEEP_TITLE,
  type VariantState,
  aggregate,
  depthSeries,
  familyOf,
  fmtMs,
  fmtPct,
  fmtRate,
  footerLines,
  headline,
  highlights,
  modelShort,
  rampingRows,
  runSeries,
  toCsv,
  toMarkdown,
} from "../lib/bench-math";
import { BenchChart, type ChartKind } from "./bench-chart";
import { svgToPng, svgToString } from "./chart-export";
import { useFamilyColors } from "./family-colors";

function Tile({ label, value, detail, accent }: { label: string; value: ReactNode; detail?: ReactNode; accent?: string }): ReactElement {
  return (
    <div className="relative flex min-w-0 flex-col gap-1 overflow-hidden rounded-2xl border border-border/60 bg-card px-4 py-3.5">
      {accent && <span className="absolute inset-y-0 left-0 w-1" style={{ background: accent }} aria-hidden={true} />}
      <span className="text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">{label}</span>
      <span className="truncate text-ui-25 font-semibold leading-tight tracking-[-0.02em] tabular-nums text-foreground">{value}</span>
      {detail && <span className="truncate text-ui-11 text-muted-foreground">{detail}</span>}
    </div>
  );
}

const STATE_STYLE: Record<VariantState, string> = {
  queued: "bg-muted text-muted-foreground",
  loading: "bg-sky-500/12 text-sky-700 dark:text-sky-300",
  running: "bg-primary/12 text-primary",
  done: "bg-emerald-500/12 text-emerald-700 dark:text-emerald-300",
  skipped: "bg-amber-500/14 text-amber-800 dark:text-amber-300",
  error: "bg-destructive/12 text-destructive",
  cancelled: "bg-muted text-muted-foreground/70",
};

function StatePill({ state }: { state: VariantState }): ReactElement {
  return (
    <span className={cn("inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-ui-10 font-semibold uppercase tracking-wide", STATE_STYLE[state])}>
      {(state === "loading" || state === "running") && <span className="size-1.5 animate-pulse rounded-full bg-current" aria-hidden={true} />}
      {state}
    </span>
  );
}

/** The queue while a sweep runs: one row per setting, with its live mean. */
export function LiveProgress({ run, progress, onStop }: { run: BenchRun; progress: string; onStop: () => void }): ReactElement {
  const colors = useFamilyColors();
  const perRow = run.config.warmup + run.config.repetitions;
  const total = run.config.variants.length * perRow;
  const finishedRows = run.outcomes.filter((o) => o.state === "skipped" || o.state === "error").length * perRow;
  const pct = total ? Math.min(100, ((run.results.length + finishedRows) / total) * 100) : 0;
  const rows = useMemo(() => new Map(aggregate(run.results, run.config.variants, null).map((r) => [r.label, r])), [run]);
  return (
    <section className="flex flex-col gap-4 rounded-2xl border border-border/60 bg-card p-5">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 flex-col">
          <span className="text-ui-15 font-semibold text-foreground">Running {SWEEP_TITLE[run.config.sweep].toLowerCase()}</span>
          <span className="truncate text-ui-12 text-muted-foreground">{progress}</span>
        </div>
        <Button variant="outline" size="sm" className="h-9 rounded-full" onClick={onStop}>
          Stop
        </Button>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-muted">
        <div className="h-full rounded-full bg-primary transition-[width] duration-500 ease-out" style={{ width: `${pct}%` }} />
      </div>
      <ul className="flex flex-col divide-y divide-border/40">
        {run.outcomes.map((o) => {
          const variant = run.config.variants.find((v) => v.label === o.label);
          const row = rows.get(o.label);
          return (
            <li key={o.label} className="flex items-center gap-3 py-2">
              <span className="size-2.5 shrink-0 rounded-[3px]" style={{ background: colors[variant ? familyOf(variant.load) : "other"] }} aria-hidden={true} />
              <span className="min-w-0 flex-1 truncate text-ui-13 text-foreground" title={o.reason}>
                {o.label}
                {o.reason && <span className="ml-2 text-ui-11 text-muted-foreground">{o.reason}</span>}
              </span>
              {row && <span className="text-ui-12 font-medium tabular-nums text-foreground">{fmtRate(row.mean)}</span>}
              <StatePill state={o.state} />
            </li>
          );
        })}
      </ul>
    </section>
  );
}

const CHART_KIND_LABEL: Record<ChartKind, string> = { bars: "Throughput", runs: "Run by run", depth: "By draft depth" };

function Legend({ families }: { families: Family[] }): ReactElement | null {
  const colors = useFamilyColors();
  if (families.length < 2) return null;
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-ui-11 text-muted-foreground">
      {families.map((f) => (
        <li key={f} className="flex items-center gap-1.5">
          <span className="size-2.5 rounded-[3px]" style={{ background: colors[f] }} aria-hidden={true} />
          {FAMILY_LABEL[f]}
        </li>
      ))}
    </ul>
  );
}

function ResultsTable({ rows }: { rows: AggRow[] }): ReactElement {
  const colors = useFamilyColors();
  return (
    <div className="overflow-x-auto rounded-2xl border border-border/60 bg-card">
      <table className="w-full text-ui-12">
        <thead>
          <tr className="text-left text-ui-10 uppercase tracking-wider text-muted-foreground [&>th]:px-4 [&>th]:py-2.5 [&>th]:font-medium">
            <th>Setting</th>
            <th className="text-right">Throughput</th>
            <th className="text-right">Range</th>
            <th className="text-right">vs baseline</th>
            <th className="text-right">First token</th>
            <th className="text-right">Load</th>
            <th className="text-right">Draft accept</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label} className="border-t border-border/40 tabular-nums [&>td]:px-4 [&>td]:py-2">
              <td className="flex items-center gap-2 text-foreground">
                <span className="size-2 shrink-0 rounded-[2px]" style={{ background: colors[r.family] }} aria-hidden={true} />
                <span className="truncate">{r.label}</span>
                {r.clientOnly && <span className="text-ui-10 text-muted-foreground">client-timed</span>}
              </td>
              <td className="text-right font-semibold text-foreground">{fmtRate(r.mean)}</td>
              <td className="text-right text-muted-foreground">{r.n > 1 ? `${r.min.toFixed(1)}–${r.max.toFixed(1)}` : "—"}</td>
              <td className={cn("text-right", r.pct !== null && r.pct > 0 ? "text-foreground" : "text-muted-foreground")}>{r.isBaseline ? "baseline" : fmtPct(r.pct) || "—"}</td>
              <td className="text-right text-muted-foreground">{fmtMs(r.ttftMs) || "—"}</td>
              <td className="text-right text-muted-foreground">{fmtMs(r.loadMs) || "—"}</td>
              <td className="text-right text-muted-foreground">{r.acceptRate === null ? "—" : `${Math.round(r.acceptRate * 100)}%`}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Note({ tone, title, children }: { tone: "warn" | "info"; title: string; children: ReactNode }): ReactElement {
  return (
    <div
      className={cn(
        "flex flex-col gap-1 rounded-xl border px-4 py-3 text-ui-12",
        tone === "warn" ? "border-amber-500/30 bg-amber-500/6" : "border-border/60 bg-muted/30",
      )}
    >
      <span className="font-medium text-foreground">{title}</span>
      <div className="leading-relaxed text-muted-foreground">{children}</div>
    </div>
  );
}

export function RunResults({ run }: { run: BenchRun }): ReactElement {
  const [kind, setKind] = useState<ChartKind>("bars");
  const svgRef = useRef<SVGSVGElement | null>(null);
  const rows = useMemo(() => aggregate(run.results, run.config.variants, run.config.baseline), [run]);
  const series = useMemo(() => runSeries(run.results, run.config.variants), [run]);
  const depth = useMemo(() => depthSeries(rows, run.config.variants), [rows, run]);
  const hl = highlights(rows);
  const ramping = rampingRows(rows);
  const skipped = run.outcomes.filter((o) => (o.state === "skipped" || o.state === "error") && o.reason);
  const kinds: ChartKind[] = depth.length ? ["bars", "runs", "depth"] : ["bars", "runs"];
  const families = [...new Set(rows.map((r) => r.family))];
  const colors = useFamilyColors();
  const title = `${SWEEP_TITLE[run.config.sweep]} · ${modelShort(run.model)}`;
  const footer = footerLines(run);
  const fileBase = `${run.config.sweep}-${modelShort(run.model) || "model"}-${new Date(run.createdAt).toISOString().slice(0, 10)}`.replace(/[^\w.-]+/g, "_");

  const save = async (content: string | Blob, name: string, mime: string) => {
    try {
      await downloadFile(content, name, mime);
    } catch (err) {
      if (!(err instanceof DownloadCancelledError)) toast.error("Could not save the file", { description: err instanceof Error ? err.message : String(err) });
    }
  };

  if (rows.length === 0) {
    return (
      <div className="flex flex-col gap-3">
        <Note tone="info" title="No measured runs in this sweep">
          Every setting was skipped or failed before it could run. The reasons are below.
        </Note>
        {skipped.map((o) => (
          <p key={o.label} className="text-ui-12 text-muted-foreground">
            <span className="text-foreground">{o.label}</span>: {o.reason}
          </p>
        ))}
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Tile
          label="Fastest"
          value={hl.best ? fmtRate(hl.best.mean) : "—"}
          detail={hl.best?.label}
          accent={hl.best ? colors[hl.best.family] : undefined}
        />
        <Tile
          label={hl.baseline ? `vs ${hl.baseline.label.toLowerCase()}` : "Speed-up"}
          value={hl.speedup ? `${hl.speedup.toFixed(hl.speedup >= 10 ? 0 : 2)}×` : "—"}
          detail={hl.baseline ? `baseline ${fmtRate(hl.baseline.mean)}` : "Pick a baseline row to compare against"}
        />
        <Tile
          label="Best draft acceptance"
          value={hl.bestAccept?.acceptRate != null ? `${Math.round(hl.bestAccept.acceptRate * 100)}%` : "—"}
          detail={hl.bestAccept?.label ?? "No speculative rows ran"}
          accent={hl.bestAccept ? colors[hl.bestAccept.family] : undefined}
        />
      </div>

      <section className="flex flex-col gap-3 rounded-2xl border border-border/60 bg-card p-3 sm:p-4">
        <div className="flex flex-wrap items-center gap-3 px-1">
          <div className="flex items-center gap-0.5 rounded-full bg-muted/60 p-0.5 text-ui-12">
            {kinds.map((k) => (
              <button
                key={k}
                type="button"
                onClick={() => setKind(k)}
                className={cn(
                  "rounded-full px-3 py-1 font-medium transition-colors",
                  kind === k ? "bg-background text-foreground shadow-sm ring-1 ring-border/50" : "text-muted-foreground hover:text-foreground",
                )}
              >
                {CHART_KIND_LABEL[k]}
              </button>
            ))}
          </div>
          <Legend families={families} />
          <div className="flex-1" />
          <div className="flex flex-wrap items-center gap-1.5">
            <Button
              variant="ghost"
              size="sm"
              className="h-8 rounded-full"
              onClick={async () => {
                if (await copyToClipboard(toMarkdown(run, rows))) toast.success("Copied as a markdown table");
              }}
            >
              Copy markdown
            </Button>
            <Button variant="ghost" size="sm" className="h-8 rounded-full" onClick={() => void save(toCsv(run), `${fileBase}.csv`, "text/csv")}>
              CSV
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 rounded-full"
              onClick={() => svgRef.current && void save(svgToString(svgRef.current), `${fileBase}.svg`, "image/svg+xml")}
            >
              SVG
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 rounded-full"
              onClick={async () => {
                if (!svgRef.current) return;
                try {
                  await save(await svgToPng(svgRef.current), `${fileBase}.png`, "image/png");
                } catch (err) {
                  toast.error("Could not render the PNG", { description: err instanceof Error ? err.message : String(err) });
                }
              }}
            >
              PNG
            </Button>
          </div>
        </div>
        <div className="overflow-hidden rounded-xl">
          <BenchChart kind={kind} rows={rows} series={series} depth={depth} title={title} subtitle={headline(rows)} footer={footer} svgRef={svgRef} />
        </div>
      </section>

      {ramping.length > 0 && (
        <Note tone="warn" title="Some settings kept speeding up from run to run">
          {ramping.map((r) => r.label).join(", ")} ran more than 1.5× faster by the end than on the first measured run. That is
          ngram drafting recognising text it has already seen. Real chats rarely repeat like that, so trust the first run (
          {ramping.map((r) => `${fmtRate(r.first ?? 0)}`).join(", ")}) more than the average.
          {!run.config.rotatePrompts && " Turning on Rotate prompts makes this much smaller."}
        </Note>
      )}

      {skipped.length > 0 && (
        <Note tone="info" title={`${skipped.length} setting${skipped.length === 1 ? "" : "s"} didn't run`}>
          <ul className="flex flex-col gap-0.5">
            {skipped.map((o) => (
              <li key={o.label}>
                <span className="text-foreground">{o.label}</span>: {o.reason}
              </li>
            ))}
          </ul>
        </Note>
      )}

      <ResultsTable rows={rows} />
    </div>
  );
}
