// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Compare grid controls and result. Lazily loaded by the Images page, so none of it is on the startup path. */

import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { InfoHint } from "@/components/ui/info-hint";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { downloadBlobStreaming, isDownloadCancelled } from "@/lib/native-files";
import { toast } from "@/lib/toast";
import { fetchGalleryBlob } from "../api";
import {
  availableAxes,
  MAX_COMPARE_CELLS,
  NOTHING,
  planCompare,
  type AxisInput,
  type CompareContext,
} from "./compare-axes";
import { createCompareRun, gridParameters, gridTitles, sliceCells, type CompareRun, type CompareRunState } from "./compare-run";
import { joinAxisList } from "./compare-values";
import { drawCellsInTurn, drawGrid, gridLayout, withPngText } from "./compose-grid";

export interface ComparePanelProps {
  context: CompareContext;
  busy: boolean;
  seedFieldEmpty: boolean;
  // The form's typed Variation seed, null when empty or the model has none.
  variationSeed: number | null;
  modelName: string | null;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  onRun: (run: CompareRun) => Promise<void>;
}

interface GridResult {
  title: string;
  url: string;
  blob: Blob;
  width: number;
  height: number;
}

const AXIS_NAMES = ["X", "Y", "Z"] as const;
const EMPTY_AXES: AxisInput[] = [
  { axisId: "steps", text: "" },
  { axisId: NOTHING, text: "" },
  { axisId: NOTHING, text: "" },
];

const HINT =
  "Generate the same prompt across up to three settings and lay the results out as a labelled grid " +
  "(AUTOMATIC1111's X/Y/Z plot). Values are comma separated; number axes also take ranges: 1-5, 10-30 (+10), " +
  "1-10 [4] (four evenly spaced). Z makes one grid per value. Prompt S/R swaps its first value in the prompt for " +
  "each value in turn. Every cell runs at batch size 1 with one shared seed and is saved to the gallery with its own recipe.";

async function composeGrid(
  run: CompareRunState,
  iz: number,
  title: string,
  options: { legend: boolean; margin: number },
  infotext: string,
): Promise<GridResult | null> {
  const plan = run.plan;
  const records = sliceCells(plan, run.results, iz);
  const first = records.find(Boolean);
  if (!first) return null;
  const canvas = document.createElement("canvas");
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const measure = (text: string, size: number) => {
    ctx.font = `${size}px ui-sans-serif, system-ui, sans-serif`;
    return ctx.measureText(text).width;
  };
  const layout = gridLayout(
    {
      cols: plan.axes[0].values.length,
      rows: plan.axes[1].values.length,
      cellWidth: first.width,
      cellHeight: first.height,
      colLabels: plan.axes[0].labels,
      rowLabels: plan.axes[1].labels,
      title,
      legend: options.legend,
      margin: options.margin,
    },
    measure,
  );
  canvas.width = layout.width;
  canvas.height = layout.height;
  // Placeholders and labels first, then one cell at a time decoded straight to its drawn size: a 64-cell grid of
  // full-size decodes held at once would be gigabytes of RGBA.
  drawGrid(ctx, layout, records.map(() => null));
  await drawCellsInTurn(
    records,
    async (r, i) => {
      const rect = layout.cells[i];
      return createImageBitmap(await fetchGalleryBlob(r.url), {
        resizeWidth: rect.w,
        resizeHeight: rect.h,
        resizeQuality: "high",
      });
    },
    (bitmap, i) => {
      const rect = layout.cells[i];
      ctx.drawImage(bitmap, rect.x, rect.y, rect.w, rect.h);
      bitmap.close();
    },
  );
  const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
  if (!blob) return null;
  const png = withPngText(new Uint8Array(await blob.arrayBuffer()), "parameters", infotext);
  const out = new Blob([Uint8Array.from(png).buffer], { type: "image/png" });
  return { title, blob: out, url: URL.createObjectURL(out), width: layout.width, height: layout.height };
}

export default function ComparePanel(props: ComparePanelProps) {
  const { context, busy } = props;
  const [enabled, setEnabled] = useState(false);
  const [axes, setAxes] = useState<AxisInput[]>(EMPTY_AXES);
  const [legend, setLegend] = useState(true);
  const [keepRandom, setKeepRandom] = useState(false);
  const [margin, setMargin] = useState(0);
  const [running, setRunning] = useState<CompareRunState | null>(null);
  const [progress, setProgress] = useState(0);
  const [grids, setGrids] = useState<GridResult[]>([]);
  const [open, setOpen] = useState(false);
  const mounted = useRef(true);
  const active = useRef(false);

  const defs = useMemo(() => availableAxes(context), [context]);
  const effective = axes.map((a) => (defs.some((d) => d.id === a.axisId) ? a : { axisId: NOTHING, text: "" }));
  // Cheap (at most 64 cells), so it runs every render; the real plan is drawn again on click for fresh seeds.
  const preview = planCompare(effective, context, { keepRandomSeeds: keepRandom, rng: () => 0 });

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);
  // Object URLs of replaced results are released; the last ones live as long as the page.
  useEffect(() => () => grids.forEach((g) => URL.revokeObjectURL(g.url)), [grids]);

  const setAxis = (i: number, next: Partial<AxisInput>) =>
    setAxes((prev) => prev.map((a, j) => (j === i ? { ...effective[j], ...next } : effective[j] ?? a)));
  const swap = (a: number, b: number) =>
    setAxes(() => {
      const next = [...effective];
      [next[a], next[b]] = [next[b], next[a]];
      return next;
    });

  const run = async () => {
    if (active.current) return;
    const result = planCompare(effective, context, { keepRandomSeeds: keepRandom });
    if (!result.ok) {
      toast.error(result.error);
      return;
    }
    const state = createCompareRun(result.plan, context, {
      keepRandomSeeds: keepRandom,
      seedFieldEmpty: props.seedFieldEmpty,
      variationSeed: props.variationSeed,
    });
    const tracked: CompareRunState = {
      ...state,
      onImages(i, images) {
        state.onImages(i, images);
        setProgress((n) => n + 1);
      },
    };
    active.current = true;
    setProgress(0);
    setRunning(tracked);
    // Busy until the grid is composed too: a second run composing alongside would multiply canvas memory and race
    // this one's result.
    try {
      await props.onRun(tracked);
      if (!mounted.current) return;
      const done = state.results.filter(Boolean).length;
      if (done === 0) return;
      const plan = result.plan;
      const infotext = gridParameters(plan, context, state.requests, {
        steps: props.steps,
        guidance: props.guidance,
        width: props.width,
        height: props.height,
        model: props.modelName,
        keepRandomSeeds: keepRandom,
      });
      try {
        // One Z slice at a time, for the same memory reason as the cells.
        const made: GridResult[] = [];
        const titles = gridTitles(plan);
        for (let iz = 0; iz < titles.length; iz++) {
          const grid = await composeGrid(state, iz, titles[iz], { legend, margin }, infotext);
          if (grid) made.push(grid);
        }
        if (!mounted.current) return;
        setGrids(made);
        if (made.length) setOpen(true);
        if (done < plan.cellCount) toast.info(`Grid made from ${done} of ${plan.cellCount} cells`);
      } catch (err) {
        toast.error(err instanceof Error ? `Could not build the grid: ${err.message}` : "Could not build the grid");
      }
    } finally {
      active.current = false;
      if (mounted.current) setRunning(null);
    }
  };

  const download = (g: GridResult, i: number) => {
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    void downloadBlobStreaming(g.blob, `compare-grid-${stamp}${grids.length > 1 ? `-${i + 1}` : ""}.png`).catch((err) => {
      // Cancelling the save dialog is not a failure; a native open / write / finalize error is.
      if (!isDownloadCancelled(err)) {
        toast.error(err instanceof Error ? `Could not save the grid: ${err.message}` : "Could not save the grid");
      }
    });
  };

  const count = preview.ok ? preview.plan.cellCount : preview.cellCount;
  const shape = preview.ok
    ? preview.plan.axes
        .filter((a) => a.def.id !== NOTHING)
        .map((a) => a.values.length)
        .join(" x ")
    : "";

  return (
    <div className="flex flex-col gap-1.5 pt-2">
      <div className="flex items-center gap-1">
        <label htmlFor="images-compare-toggle" className="text-xs font-medium text-muted-foreground">
          Compare grid
        </label>
        <InfoHint>{HINT}</InfoHint>
        <div className="ml-auto flex items-center gap-2">
          {grids.length > 0 && !running && (
            <Button type="button" variant="ghost" size="sm" className="h-6 px-2 text-xs" onClick={() => setOpen(true)}>
              Last grid
            </Button>
          )}
          <Switch id="images-compare-toggle" checked={enabled} onCheckedChange={setEnabled} aria-label="Compare grid" />
        </div>
      </div>
      {enabled && (
        <div className="space-y-2 rounded-lg border border-border bg-muted/30 p-2">
          {effective.map((axis, i) => {
            const def = defs.find((d) => d.id === axis.axisId);
            return (
              <div key={AXIS_NAMES[i]} className="flex items-center gap-1.5">
                <span className="w-3 shrink-0 text-xs font-medium text-muted-foreground">{AXIS_NAMES[i]}</span>
                <Select value={axis.axisId} onValueChange={(v) => setAxis(i, { axisId: v })}>
                  <SelectTrigger className="h-8 w-32 shrink-0 text-xs" aria-label={`${AXIS_NAMES[i]} type`}>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {defs.map((d) => (
                      <SelectItem key={d.id} value={d.id}>
                        {d.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Input
                  value={axis.axisId === NOTHING ? "" : axis.text}
                  disabled={axis.axisId === NOTHING}
                  placeholder={def?.placeholder}
                  aria-label={`${AXIS_NAMES[i]} values`}
                  onChange={(e) => setAxis(i, { text: e.target.value })}
                  className="h-8 min-w-0 flex-1 text-xs"
                />
                {def?.choices && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-8 shrink-0 px-2 text-xs"
                    title={`Fill every ${def.label.toLowerCase()}`}
                    onClick={() => setAxis(i, { text: joinAxisList(def.choices!.map((c) => c.label)) })}
                  >
                    All
                  </Button>
                )}
              </div>
            );
          })}
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs">
            <label className="flex items-center gap-1.5">
              <Switch size="sm" checked={legend} onCheckedChange={setLegend} />
              Draw legend
            </label>
            <label className="flex items-center gap-1.5" title="Every cell draws its own random seed when the Seed box is empty">
              <Switch size="sm" checked={keepRandom} onCheckedChange={setKeepRandom} />
              Keep -1 for seeds
            </label>
            <label className="flex items-center gap-1.5">
              Margin
              <Input
                type="number"
                min={0}
                max={200}
                step={2}
                value={margin}
                onChange={(e) => setMargin(Math.max(0, Math.min(200, Math.round(Number(e.target.value) || 0))))}
                className="h-7 w-16 text-xs"
                aria-label="Grid margin in pixels"
              />
            </label>
            <Button type="button" variant="ghost" size="sm" className="h-7 px-2 text-xs" onClick={() => swap(0, 1)}>
              Swap X/Y
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <p
              className={preview.ok ? "flex-1 text-xs text-muted-foreground" : "flex-1 text-xs text-destructive"}
              role={preview.ok ? undefined : "alert"}
            >
              {running
                ? `${progress} of ${running.cellCount} cells done`
                : preview.ok
                  ? `${shape} = ${count} cell${count === 1 ? "" : "s"} (limit ${MAX_COMPARE_CELLS})`
                  : preview.error}
            </p>
            <Button variant="dark" type="button" size="sm" disabled={busy || running !== null || !preview.ok} onClick={() => void run()}>
              {preview.ok ? `Generate grid (${count})` : "Generate grid"}
            </Button>
          </div>
        </div>
      )}
      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="corner-squircle dialog-soft-surface gap-4 sm:max-w-5xl">
          <DialogHeader>
            <DialogTitle>Compare grid</DialogTitle>
            <DialogDescription>
              Each cell is also in the gallery with its own recipe. The PNG carries the grid's parameters.
            </DialogDescription>
          </DialogHeader>
          <div className="max-h-[70dvh] space-y-4 overflow-auto">
            {grids.map((g, i) => (
              <figure key={g.url} className="space-y-2">
                <img src={g.url} alt={g.title || "Compare grid"} className="mx-auto max-w-full rounded-lg" />
                <figcaption className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
                  <span>
                    {g.title ? `${g.title}, ` : ""}
                    {g.width} x {g.height}
                  </span>
                  <Button type="button" size="sm" variant="secondary" onClick={() => download(g, i)}>
                    Download PNG
                  </Button>
                </figcaption>
              </figure>
            ))}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
