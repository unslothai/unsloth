// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SectionCard } from "@/components/section-card";
import {
  type SegmentedTabOption,
  SegmentedTabsList,
} from "@/components/segmented-tabs";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { DownloadCancelledError, downloadFile } from "@/lib/native-files";
import { cn } from "@/lib/utils";
import {
  type ReactElement,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  ArrowDown01Icon,
  ChartAverageIcon,
  Download04Icon,
  StopIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { toast } from "sonner";
import {
  type AggRow,
  type BenchRun,
  type VariantState,
  FAMILY_LABEL,
  type Family,
  SWEEP_TITLE,
  aggregate,
  depthSeries,
  familyOf,
  fmtMs,
  fmtPct,
  fmtRate,
  footerLines,
  headline,
  highlights,
  inSentence,
  modelShort,
  rampingRows,
  runSeries,
  toCsv,
  toMarkdown,
} from "../lib/bench-math";
import { BenchChart, type ChartKind, type PendingRow } from "./bench-chart";
import { svgToPng, svgToString } from "./chart-export";
import { useFamilyColors } from "./family-colors";

function Tile({
  label,
  value,
  detail,
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
}): ReactElement {
  return (
    <div className={cn(CARD, "flex min-w-0 flex-col gap-1.5 px-5 py-4")}>
      <span className="text-ui-11 font-medium tracking-nav text-muted-foreground">
        {label}
      </span>
      <span className="truncate font-heading text-ui-22 font-semibold leading-tight tracking-[-0.02em] tabular-nums text-foreground">
        {value}
      </span>
      {detail && (
        <span className="truncate text-ui-11 text-muted-foreground">
          {detail}
        </span>
      )}
    </div>
  );
}

const CHART_KIND_LABEL: Record<ChartKind, string> = {
  bars: "Throughput",
  runs: "Run by run",
  depth: "By draft depth",
};

function Legend({ families }: { families: Family[] }): ReactElement | null {
  const colors = useFamilyColors();
  if (families.length < 2) return null;
  return (
    <ul className="flex flex-wrap items-center gap-x-4 gap-y-1 text-ui-11 text-muted-foreground">
      {families.map((f) => (
        <li key={f} className="flex items-center gap-1.5">
          <span
            className="size-2.5 rounded-[3px]"
            style={{ background: colors[f] }}
            aria-hidden={true}
          />
          {FAMILY_LABEL[f]}
        </li>
      ))}
    </ul>
  );
}

function ResultsTable({
  rows,
  vram,
}: {
  rows: AggRow[];
  /** VRAM in use after each row's load, when it was read. */
  vram: Record<string, number>;
}): ReactElement {
  const colors = useFamilyColors();
  const showVram = Object.keys(vram).length > 0;
  return (
    <div className="overflow-x-auto rounded-xl">
      <table className="w-full text-ui-12">
        <thead>
          <tr className="text-left text-ui-10 uppercase tracking-wider text-muted-foreground [&>th]:px-3 [&>th]:py-2.5 [&>th]:font-medium">
            <th>Setting</th>
            <th className="text-right">Throughput</th>
            <th className="text-right">Range</th>
            <th className="text-right">vs baseline</th>
            <th className="text-right">First token</th>
            <th className="text-right">Load</th>
            {showVram && <th className="text-right">VRAM</th>}
            <th className="text-right">Draft accept</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr
              key={r.label}
              className="border-t border-border/40 tabular-nums [&>td]:whitespace-nowrap [&>td]:px-3 [&>td]:py-2"
            >
              <td className="flex items-center gap-2 text-foreground">
                <span
                  className="size-2 shrink-0 rounded-[2px]"
                  style={{ background: colors[r.family] }}
                  aria-hidden={true}
                />
                <span className="truncate">{r.label}</span>
                {r.clientOnly && (
                  <span className="text-ui-10 text-muted-foreground">
                    client-timed
                  </span>
                )}
              </td>
              <td className="text-right font-semibold text-foreground">
                {fmtRate(r.mean)}
              </td>
              <td className="text-right text-muted-foreground">
                {r.n > 1 ? `${r.min.toFixed(1)}–${r.max.toFixed(1)}` : "—"}
              </td>
              <td
                className={cn(
                  "text-right",
                  r.pct !== null && r.pct > 0
                    ? "text-foreground"
                    : "text-muted-foreground",
                )}
              >
                {r.isBaseline ? "baseline" : fmtPct(r.pct) || "—"}
              </td>
              <td className="text-right text-muted-foreground">
                {fmtMs(r.ttftMs) || "—"}
              </td>
              <td className="text-right text-muted-foreground">
                {fmtMs(r.loadMs) || "—"}
              </td>
              {showVram && (
                <td className="text-right text-muted-foreground">
                  {vram[r.label] !== undefined ? `${vram[r.label]} GB` : "—"}
                </td>
              )}
              <td className="text-right text-muted-foreground">
                {r.acceptRate === null
                  ? "—"
                  : `${Math.round(r.acceptRate * 100)}%`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

const CARD =
  "corner-squircle rounded-3xl bg-card ring-1 ring-[color-mix(in_oklab,var(--foreground)_calc(10%*var(--contrast-edge-gain,1)),transparent)]";

const TOP_ROWS = 8;

/** The fastest few plus the baseline: the chart answers "what wins" without a scroll. */
function visibleRows(rows: AggRow[], expanded: boolean): AggRow[] {
  if (expanded || rows.length <= TOP_ROWS + 1) return rows;
  const base = rows.find((r) => r.isBaseline);
  const rest = rows.filter((r) => !r.isBaseline).slice(0, TOP_ROWS);
  return base ? [...rest, base] : rest;
}

function NoteChip({
  tone,
  label,
  title,
  children,
}: {
  tone: "warn" | "info";
  label: string;
  title: string;
  children: ReactNode;
}): ReactElement {
  return (
    <Popover>
      <PopoverTrigger asChild={true}>
        <button
          type="button"
          className={cn(
            "inline-flex h-7 items-center gap-1.5 rounded-full px-2.5 text-ui-11p5 font-medium transition-colors",
            tone === "warn"
              ? "bg-amber-500/12 text-amber-800 hover:bg-amber-500/20 dark:text-amber-300"
              : "bg-muted/60 text-muted-foreground hover:text-foreground",
          )}
        >
          <span
            className={cn(
              "size-1.5 rounded-full",
              tone === "warn" ? "bg-amber-500" : "bg-muted-foreground/60",
            )}
            aria-hidden={true}
          />
          {label}
        </button>
      </PopoverTrigger>
      <PopoverContent align="end" className="w-96 text-ui-12">
        <p className="mb-1 font-medium text-foreground">{title}</p>
        <div className="leading-relaxed text-muted-foreground">{children}</div>
      </PopoverContent>
    </Popover>
  );
}

function Note({
  tone,
  title,
  children,
}: {
  tone: "warn" | "info";
  title: string;
  children: ReactNode;
}): ReactElement {
  return (
    <div
      className={cn(
        "flex flex-col gap-1 rounded-xl border px-4 py-3 text-ui-12",
        tone === "warn"
          ? "border-amber-500/30 bg-amber-500/6"
          : "border-border/60 bg-muted/30",
      )}
    >
      <span className="font-medium text-foreground">{title}</span>
      <div className="leading-relaxed text-muted-foreground">{children}</div>
    </div>
  );
}

const LIVE_STATUS: Partial<Record<VariantState, string>> = {
  queued: "Queued",
  loading: "Loading…",
  running: "Measuring…",
};

/** Its own component, so the once-a-second tick repaints one span and not the chart. */
function Elapsed({ since }: { since: number }): ReactElement {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, []);
  const sec = Math.max(0, Math.round((now - since) / 1000));
  return (
    <>
      {Math.floor(sec / 60)}:{String(sec % 60).padStart(2, "0")}
    </>
  );
}

export function RunResults({
  run,
  live = false,
  progress,
  onStop,
}: {
  run: BenchRun;
  live?: boolean;
  progress?: string;
  onStop?: () => void;
}): ReactElement {
  const [view, setView] = useState<ChartKind | "table">("bars");
  const [expanded, setExpanded] = useState(false);
  const exportRef = useRef<SVGSVGElement | null>(null);
  const spareRef = useRef<SVGSVGElement | null>(null);
  const rows = useMemo(
    () => aggregate(run.results, run.config.variants, run.config.baseline),
    [run],
  );
  const series = useMemo(
    () => runSeries(run.results, run.config.variants),
    [run],
  );
  const depth = useMemo(
    () => depthSeries(rows, run.config.variants),
    [rows, run],
  );
  const hl = highlights(rows);
  const ramping = rampingRows(rows);
  const skipped = run.outcomes.filter(
    (o) => (o.state === "skipped" || o.state === "error") && o.reason,
  );
  const kinds: ChartKind[] = depth.length
    ? ["bars", "runs", "depth"]
    : ["bars", "runs"];
  const families = [...new Set(rows.map((r) => r.family))];
  const title = `${SWEEP_TITLE[run.config.sweep]} · ${modelShort(run.model) || "reading the model"}`;
  const footer = footerLines(run);
  const fileBase =
    `${run.config.sweep}-${modelShort(run.model) || "model"}-${new Date(run.createdAt).toISOString().slice(0, 10)}`.replace(
      /[^\w.-]+/g,
      "_",
    );

  // In a live run the queue is the chart: rows still to come sit under the measured ones.
  const measured = new Set(rows.map((r) => r.label));
  const pending: PendingRow[] = live
    ? run.outcomes
        .filter((o) => LIVE_STATUS[o.state] && !measured.has(o.label))
        .map((o) => {
          const v = run.config.variants.find((x) => x.label === o.label);
          return {
            label: o.label,
            family: v ? familyOf(v.load) : "other",
            status: LIVE_STATUS[o.state] ?? "",
            active: o.state !== "queued",
          };
        })
    : [];
  const perRow = run.config.warmup + run.config.repetitions;
  const settled =
    run.outcomes.filter((o) => o.state === "skipped" || o.state === "error")
      .length * perRow;
  const fraction = run.config.variants.length
    ? Math.min(
        1,
        (run.results.length + settled) / (run.config.variants.length * perRow),
      )
    : 0;
  const current = run.outcomes.find(
    (o) => o.state === "loading" || o.state === "running",
  );
  const currentIndex = current
    ? run.outcomes.findIndex((o) => o.label === current.label)
    : -1;

  const save = async (content: string | Blob, name: string, mime: string) => {
    try {
      await downloadFile(content, name, mime);
    } catch (err) {
      if (!(err instanceof DownloadCancelledError))
        toast.error("Could not save the file", {
          description: err instanceof Error ? err.message : String(err),
        });
    }
  };

  if (rows.length === 0 && !live) {
    return (
      <Note tone="info" title="No measured runs in this sweep">
        <ul className="flex flex-col gap-0.5">
          {skipped.map((o) => (
            <li key={o.label}>
              <span className="text-foreground">{o.label}</span>: {o.reason}
            </li>
          ))}
        </ul>
      </Note>
    );
  }

  const views: (ChartKind | "table")[] = live ? kinds : [...kinds, "table"];
  const shownView = views.includes(view) ? view : "bars";
  const viewOptions = views.map((k) => ({
    value: k,
    label: k === "table" ? "Table" : CHART_KIND_LABEL[k],
  })) as unknown as readonly [
    SegmentedTabOption<ChartKind | "table">,
    SegmentedTabOption<ChartKind | "table">,
    ...SegmentedTabOption<ChartKind | "table">[],
  ];
  const isTable = shownView === "table";
  const chartKind: ChartKind = shownView === "table" ? "bars" : shownView;
  // Live rows keep queue order, so bars never jump as a faster one lands; the ranking waits for the end.
  const order = new Map(run.config.variants.map((v, i) => [v.label, i]));
  const top = live
    ? [...rows].sort(
        (a, b) => (order.get(a.label) ?? 0) - (order.get(b.label) ?? 0),
      )
    : visibleRows(rows, expanded);
  const hidden = rows.length - top.length;
  const folded = !live && (isTable || hidden > 0);

  return (
    <div className="flex flex-col gap-4">
      {live && (
        <SectionCard
          icon={<HugeiconsIcon icon={ChartAverageIcon} className="size-5" />}
          title={SWEEP_TITLE[run.config.sweep]}
          description={progress ?? "Starting"}
          className="shadow-border border border-border/60 bg-card/90 ring-0"
          headerAction={
            onStop && (
              <Button
                variant="destructive"
                size="sm"
                onClick={onStop}
                className="h-8 rounded-full px-3.5 text-xs shadow-sm"
              >
                <HugeiconsIcon icon={StopIcon} className="size-3" />
                Stop
              </Button>
            )
          }
        >
          <div className="flex flex-col gap-2">
            <div className="flex flex-wrap items-center gap-2">
              <span className="rounded-full bg-muted px-2.5 py-1 text-ui-10 font-semibold text-foreground">
                {current ? "Running" : "Finishing"}
              </span>
              {current && currentIndex >= 0 && (
                <span className="rounded-full border border-border/60 px-2.5 py-1 text-ui-10 font-medium text-foreground/80">
                  Row {currentIndex + 1} of {run.outcomes.length}
                </span>
              )}
              <span className="text-ui-10 tabular-nums text-muted-foreground">
                <Elapsed since={run.createdAt} /> elapsed
              </span>
            </div>
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>
                {run.results.length} of {run.config.variants.length * perRow}{" "}
                generations
              </span>
              <span>{Math.round(fraction * 100)}%</span>
            </div>
            <Progress
              value={fraction * 100}
              className="h-2 bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)]"
            />
          </div>
        </SectionCard>
      )}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Tile
          label="Fastest"
          value={hl.best ? fmtRate(hl.best.mean) : "—"}
          detail={hl.best?.label ?? "Waiting on the first row"}
        />
        <Tile
          label={
            hl.baseline ? `vs ${inSentence(hl.baseline.label)}` : "Speed-up"
          }
          value={
            hl.speedup
              ? `${hl.speedup.toFixed(hl.speedup >= 10 ? 0 : 2)}×`
              : "—"
          }
          detail={
            !hl.baseline
              ? "Star a row to compare against it"
              : hl.speedup !== null && hl.speedup < 1
                ? `nothing beat it · baseline ${fmtRate(hl.baseline.mean)}`
                : `baseline ${fmtRate(hl.baseline.mean)}`
          }
        />
        <Tile
          label="Best draft acceptance"
          value={
            hl.bestAccept?.acceptRate != null
              ? `${Math.round(hl.bestAccept.acceptRate * 100)}%`
              : "—"
          }
          detail={hl.bestAccept?.label ?? "No speculative rows yet"}
        />
      </div>

      <section className={cn(CARD, "flex flex-col gap-3 p-3 sm:p-4")}>
        <div className="flex flex-wrap items-center gap-3 px-1">
          <Tabs
            value={shownView}
            onValueChange={(v) => setView(v as ChartKind | "table")}
            className="contents"
          >
            <SegmentedTabsList
              value={shownView}
              options={viewOptions}
              ariaLabel="Chart view"
              size="compact"
            />
          </Tabs>
          <Legend families={families} />
          <div className="flex-1" />
          {ramping.length > 0 && (
            <NoteChip
              tone="warn"
              label={`${ramping.length} ramping`}
              title="Some settings kept speeding up from run to run"
            >
              {ramping.map((r) => r.label).join(", ")} ran more than 1.5× faster
              by the end than on the first measured run. That is ngram drafting
              recognising text it has already seen. Real chats rarely repeat
              like that, so trust the first run (
              {ramping.map((r) => fmtRate(r.first ?? 0)).join(", ")}) more than
              the average.
              {!run.config.rotatePrompts &&
                " Turning on Rotate prompts makes this much smaller."}
            </NoteChip>
          )}
          {skipped.length > 0 && (
            <NoteChip
              tone="info"
              label={`${skipped.length} skipped`}
              title={`${skipped.length} setting${skipped.length === 1 ? "" : "s"} did not run`}
            >
              <ul className="flex flex-col gap-0.5">
                {skipped.map((o) => (
                  <li key={o.label}>
                    <span className="text-foreground">{o.label}</span>:{" "}
                    {o.reason}
                  </li>
                ))}
              </ul>
            </NoteChip>
          )}
          {!live && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild={true}>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 gap-1.5 rounded-full"
                >
                  <HugeiconsIcon
                    icon={Download04Icon}
                    strokeWidth={1.75}
                    className="size-3.5"
                  />
                  Export
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem
                  onSelect={async () => {
                    if (await copyToClipboard(toMarkdown(run, rows)))
                      toast.success("Copied as a markdown table");
                  }}
                >
                  Copy as markdown
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={() =>
                    void save(toCsv(run), `${fileBase}.csv`, "text/csv")
                  }
                >
                  Save CSV
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={() =>
                    exportRef.current &&
                    void save(
                      svgToString(exportRef.current),
                      `${fileBase}.svg`,
                      "image/svg+xml",
                    )
                  }
                >
                  Save chart as SVG
                </DropdownMenuItem>
                <DropdownMenuItem
                  onSelect={async () => {
                    if (!exportRef.current) return;
                    try {
                      await save(
                        await svgToPng(exportRef.current),
                        `${fileBase}.png`,
                        "image/png",
                      );
                    } catch (err) {
                      toast.error("Could not render the PNG", {
                        description:
                          err instanceof Error ? err.message : String(err),
                      });
                    }
                  }}
                >
                  Save chart as PNG
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>

        {isTable ? (
          <ResultsTable
            rows={rows}
            vram={Object.fromEntries(
              run.outcomes.flatMap((o) =>
                typeof o.served?.vram_used_gb === "number"
                  ? [[o.label, o.served.vram_used_gb]]
                  : [],
              ),
            )}
          />
        ) : (
          <div className="overflow-hidden rounded-xl">
            <BenchChart
              kind={chartKind}
              rows={chartKind === "bars" ? top : rows}
              series={series}
              depth={depth}
              title={title}
              subtitle={headline(rows)}
              footer={footer}
              pending={chartKind === "bars" ? pending : []}
              svgRef={folded ? spareRef : exportRef}
            />
          </div>
        )}
        {!isTable &&
          !live &&
          chartKind === "bars" &&
          rows.length > TOP_ROWS + 1 && (
            <button
              type="button"
              onClick={() => setExpanded((v) => !v)}
              className="mx-auto flex items-center gap-1.5 rounded-full px-3 py-1 text-ui-12 text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
            >
              <HugeiconsIcon
                icon={ArrowDown01Icon}
                strokeWidth={1.75}
                className={cn(
                  "size-3.5 transition-transform",
                  expanded && "rotate-180",
                )}
              />
              {expanded
                ? `Show the top ${TOP_ROWS}`
                : `Show all ${rows.length} settings (${hidden} more)`}
            </button>
          )}
        {/* Exports always carry every row, whatever the screen is folded to. */}
        {folded && (
          <div className="hidden" aria-hidden={true}>
            <BenchChart
              kind={chartKind}
              rows={rows}
              series={series}
              depth={depth}
              title={title}
              subtitle={headline(rows)}
              footer={footer}
              svgRef={exportRef}
            />
          </div>
        )}
      </section>
    </div>
  );
}
