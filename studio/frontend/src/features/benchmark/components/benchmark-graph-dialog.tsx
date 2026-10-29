// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { BarChart } from "@/features/benchmarks/components/bar-chart";
import { niceMax } from "@/features/benchmarks/components/chart-math";
import { useT } from "@/i18n";
import {
  ChevronDownStandardIcon,
  ChevronUpStandardIcon,
} from "@/lib/chevron-icons";
import { DownloadCancelledError, downloadFile } from "@/lib/native-files";
import { cn } from "@/lib/utils";
import {
  AlertCircleIcon,
  ArrowDown01Icon,
  Download01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useMemo, useRef, useState } from "react";
import {
  BenchmarkRequestError,
  type BenchmarkRunMetric,
  type BenchmarkRunSummary,
  generateBenchmarkGraph,
} from "../api/benchmark-api";
import { useModelPalette } from "../lib/model-colors";
import {
  ACCURACY,
  comparisonGroups,
  fmtScore,
  modelsOf,
} from "../lib/score-bars";
import { ChartSaveMenu } from "./chart-save-menu";

function isDisplayMetric(m: { name: string }): boolean {
  const lower = m.name.toLowerCase().trim();
  const skip = new Set([
    "sample_len",
    "num_samples",
    "n_samples",
    "effective_samples",
    "alias",
    "name",
    "bootstrap_iters",
  ]);
  if (skip.has(lower) || lower.startsWith("sample_len")) {
    return false;
  }
  return true;
}

function getDefaultMetric(run: BenchmarkRunSummary): BenchmarkRunMetric | undefined {
  if (!run.default_metric) return undefined;
  return run.metrics.find((m) => m.name === run.default_metric);
}

function formatMetricValue(m: BenchmarkRunMetric): string {
  if (m.score >= 0 && m.score <= 1) {
    return (m.score * 100).toFixed(1);
  }
  if (Number.isInteger(m.score)) {
    return String(m.score);
  }
  return m.score.toFixed(m.score < 1 ? 4 : 1);
}

const fmtRaw = (v: number) =>
  Number.isInteger(v) ? String(v) : v.toFixed(Math.abs(v) < 1 ? 4 : 1);

function runLabel(run: BenchmarkRunSummary): string {
  const model = run.model.split("/").pop() ?? run.model;
  const short = model.length > 20 ? `${model.slice(0, 18)}…` : model;
  return `${run.task} — ${short}`;
}

interface GraphDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedRuns: BenchmarkRunSummary[];
}

export function GraphDialog({
  open,
  onOpenChange,
  selectedRuns,
}: GraphDialogProps) {
  const t = useT();
  const [orderedIds, setOrderedIds] = useState<string[]>(() =>
    selectedRuns.map((r) => r.id),
  );
  const [chartType, setChartType] = useState<
    "bar" | "line" | "grouped_bar" | "radar"
  >("bar");
  const [width, setWidth] = useState(10);
  const [height, setHeight] = useState(6);
  const [theme, setTheme] = useState("unsloth-dark");
  const [customThemeId, setCustomThemeId] = useState("");
  const [generating, setGenerating] = useState(false);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [styledOpen, setStyledOpen] = useState(false);
  const palette = useModelPalette();
  const exportRef = useRef<SVGSVGElement | null>(null);

  // Keep orderedIds in sync with selectedRuns
  useMemo(() => {
    setOrderedIds((prev) => {
      const selected = new Set(selectedRuns.map((r) => r.id));
      const existing = prev.filter((id) => selected.has(id));
      const newIds = selectedRuns
        .map((r) => r.id)
        .filter((id) => !existing.includes(id));
      return [...existing, ...newIds];
    });
  }, [selectedRuns]);

  const orderedRuns = useMemo(() => {
    const runMap = new Map(selectedRuns.map((r) => [r.id, r]));
    return orderedIds
      .map((id) => runMap.get(id))
      .filter((r): r is BenchmarkRunSummary => r != null);
  }, [orderedIds, selectedRuns]);

  // Collect available metrics, with "Accuracy" always first
  const availableMetrics = useMemo(() => {
    const metrics: { label: string; value: string }[] = [
      { label: t("benchmark.graph.accuracy"), value: ACCURACY },
    ];
    const seen = new Set<string>();
    for (const run of selectedRuns) {
      for (const m of run.metrics) {
        if (isDisplayMetric(m) && !seen.has(m.name)) {
          seen.add(m.name);
          metrics.push({ label: m.name, value: m.name });
        }
      }
    }
    return metrics;
  }, [selectedRuns, t]);

  const [pickedMetric, setSelectedMetric] = useState(ACCURACY);
  // The dialog stays mounted across selections; a metric the current runs lack
  // would chart every bar as 0, so fall back to accuracy.
  const selectedMetric = availableMetrics.some((m) => m.value === pickedMetric)
    ? pickedMetric
    : ACCURACY;

  const moveUp = useCallback((index: number) => {
    setOrderedIds((prev) => {
      if (index <= 0) {
        return prev;
      }
      const next = [...prev];
      [next[index - 1], next[index]] = [next[index], next[index - 1]];
      return next;
    });
  }, []);

  const moveDown = useCallback((index: number) => {
    setOrderedIds((prev) => {
      if (index >= prev.length - 1) {
        return prev;
      }
      const next = [...prev];
      [next[index], next[index + 1]] = [next[index + 1], next[index]];
      return next;
    });
  }, []);

  // Matplotlib, in any chart type and theme, for a figure outside the app.
  const handleStyledExport = useCallback(async () => {
    setGenerating(true);
    setGraphError(null);
    const resolvedTheme = theme === "custom" ? customThemeId : theme;
    try {
      const blob = await generateBenchmarkGraph({
        runIds: orderedIds,
        chartType,
        metric: selectedMetric,
        width,
        height,
        theme: resolvedTheme,
      });
      if (!blob.type.startsWith("image/")) {
        setGraphError("Unexpected response from server");
        return;
      }
      await downloadFile(blob, "benchmark-chart.png", "image/png");
    } catch (err) {
      if (err instanceof DownloadCancelledError) return;
      if (err instanceof BenchmarkRequestError) {
        setGraphError(err.message);
      } else {
        setGraphError(
          err instanceof Error ? err.message : "Chart generation failed",
        );
      }
    } finally {
      setGenerating(false);
    }
  }, [
    orderedIds,
    chartType,
    selectedMetric,
    width,
    height,
    theme,
    customThemeId,
  ]);

  const chart = useMemo(() => {
    const models = modelsOf(orderedRuns);
    const color = (m: string) =>
      palette[models.indexOf(m) % palette.length] ?? palette[0];
    const metricLabel =
      availableMetrics.find((m) => m.value === selectedMetric)?.label ??
      selectedMetric;
    const probe = comparisonGroups(orderedRuns, selectedMetric, color, fmtScore);
    const format = probe.percent ? fmtScore : fmtRaw;
    const { groups, percent, peak } = probe.percent
      ? probe
      : comparisonGroups(orderedRuns, selectedMetric, color, format);
    return {
      groups,
      max: percent ? 100 : niceMax(peak),
      format,
      axisLabel: percent ? `${metricLabel} (%)` : metricLabel,
      title: metricLabel,
    };
  }, [orderedRuns, selectedMetric, availableMetrics, palette]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-5xl">
        <DialogHeader>
          <DialogTitle>{t("benchmark.graph.title")}</DialogTitle>
          <DialogDescription>
            {t("benchmark.graph.description")}
          </DialogDescription>
        </DialogHeader>

        <div className="max-h-[calc(100vh-200px)] space-y-4 overflow-y-auto">
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-[minmax(0,1fr)_calc(200px*var(--ui-space-scale,1))]">
            {/* Reorderable runs list */}
            <div>
              <div className="mb-1.5 text-ui-11 font-medium tracking-nav text-muted-foreground">
                {t("benchmark.graph.orderLabel")}
              </div>
              <div className="max-h-36 space-y-0.5 overflow-y-auto rounded-xl bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)] p-1.5">
                {orderedRuns.map((run, index) => (
                  <div
                    key={run.id}
                    className="flex items-center gap-2 rounded-lg px-2 py-1 text-ui-12"
                  >
                    <span
                      className="size-2 shrink-0 rounded-[2px]"
                      style={{
                        background:
                          chart.groups
                            .flatMap((g) => g.rows)
                            .find((r) => r.key === run.id)?.color ??
                          "transparent",
                      }}
                      aria-hidden={true}
                    />
                    <span className="min-w-0 flex-1 truncate text-foreground">
                      {runLabel(run)}
                    </span>
                    <span className="shrink-0 tabular-nums text-muted-foreground">
                      {formatMetricValue(
                        getDefaultMetric(run) ?? { name: "", score: 0 },
                      )}
                      {(getDefaultMetric(run)?.score ?? 0) >= 0 &&
                      (getDefaultMetric(run)?.score ?? 0) <= 1
                        ? "%"
                        : ""}
                    </span>
                    <div className="flex shrink-0 flex-col">
                      <button
                        type="button"
                        onClick={() => moveUp(index)}
                        disabled={index === 0}
                        className="text-muted-foreground/60 hover:text-foreground disabled:opacity-30"
                      >
                        <HugeiconsIcon
                          icon={ChevronUpStandardIcon}
                          className="size-3"
                        />
                      </button>
                      <button
                        type="button"
                        onClick={() => moveDown(index)}
                        disabled={index === orderedRuns.length - 1}
                        className="text-muted-foreground/60 hover:text-foreground disabled:opacity-30"
                      >
                        <HugeiconsIcon
                          icon={ChevronDownStandardIcon}
                          className="size-3"
                        />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div>
              <div className="mb-1.5 text-ui-11 font-medium tracking-nav text-muted-foreground">
                {t("benchmark.graph.metric")}
              </div>
              <Select value={selectedMetric} onValueChange={setSelectedMetric}>
                <SelectTrigger size="sm" className="w-full text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {availableMetrics.map((m) => (
                    <SelectItem key={m.value} value={m.value}>
                      {m.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          {/* The in-app chart is the shared Benchmarks bar chart, so it reads like Config sweeps. */}
          <div className="overflow-hidden rounded-xl">
            <BarChart {...chart} bare={true} />
          </div>
          <div className="hidden" aria-hidden={true}>
            <BarChart {...chart} bare={false} svgRef={exportRef} />
          </div>

          <div className="flex flex-col gap-3">
            <button
              type="button"
              onClick={() => setStyledOpen((v) => !v)}
              aria-expanded={styledOpen}
              className="mr-auto flex items-center gap-1.5 rounded-full px-2 py-0.5 text-ui-11 text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
            >
              <HugeiconsIcon
                icon={ArrowDown01Icon}
                strokeWidth={1.75}
                className={cn(
                  "size-3.5 transition-transform",
                  styledOpen && "rotate-180",
                )}
              />
              {t("benchmark.graph.styledExport")}
            </button>
            {styledOpen && (
              <div className="flex flex-col gap-3 rounded-xl bg-[color-mix(in_oklab,var(--foreground)_calc(4%*var(--contrast-wash-gain,1)),transparent)] p-3">
                <p className="text-ui-11 text-muted-foreground">
                  {t("benchmark.graph.styledExportHint")}
                </p>
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                  <div>
                    <div className="mb-1 text-ui-11 font-medium tracking-nav text-muted-foreground">
                      {t("benchmark.graph.chartType")}
                    </div>
                    <Select
                      value={chartType}
                      onValueChange={(v) => setChartType(v as typeof chartType)}
                    >
                      <SelectTrigger size="sm" className="w-full text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="bar">
                          {t("benchmark.graph.bar")}
                        </SelectItem>
                        <SelectItem value="line">
                          {t("benchmark.graph.line")}
                        </SelectItem>
                        <SelectItem value="grouped_bar">
                          {t("benchmark.graph.groupedBar")}
                        </SelectItem>
                        <SelectItem value="radar">
                          {t("benchmark.graph.radar")}
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <div className="mb-1 text-ui-11 font-medium tracking-nav text-muted-foreground">
                      {t("benchmark.graph.theme")}
                    </div>
                    <Select value={theme} onValueChange={setTheme}>
                      <SelectTrigger size="sm" className="w-full text-xs">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="unsloth-dark">
                          {t("benchmark.graph.themeUnslothDark")}
                        </SelectItem>
                        <SelectItem value="dark">
                          {t("benchmark.graph.dark")}
                        </SelectItem>
                        <SelectItem value="light">
                          {t("benchmark.graph.light")}
                        </SelectItem>
                        <SelectItem value="ggplot">
                          {t("benchmark.graph.themeGgplot")}
                        </SelectItem>
                        <SelectItem value="fivethirtyeight">
                          {t("benchmark.graph.themeFiveThirtyEight")}
                        </SelectItem>
                        <SelectItem value="bmh">
                          {t("benchmark.graph.themeBmh")}
                        </SelectItem>
                        <SelectItem value="grayscale">
                          {t("benchmark.graph.themeGrayscale")}
                        </SelectItem>
                        <SelectItem value="seaborn-v0_8">
                          {t("benchmark.graph.themeSeaborn")}
                        </SelectItem>
                        <SelectItem value="seaborn-v0_8-darkgrid">
                          {t("benchmark.graph.themeSeabornDarkgrid")}
                        </SelectItem>
                        <SelectItem value="custom">
                          {t("benchmark.graph.themeCustom")}
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <div className="mb-1 text-ui-11 font-medium tracking-nav text-muted-foreground">
                      {t("benchmark.graph.width")}
                    </div>
                    <Input
                      type="number"
                      value={width}
                      onChange={(e) => setWidth(Number(e.target.value) || 10)}
                      min={4}
                      max={20}
                      className="h-8 text-xs"
                    />
                  </div>
                  <div>
                    <div className="mb-1 text-ui-11 font-medium tracking-nav text-muted-foreground">
                      {t("benchmark.graph.height")}
                    </div>
                    <Input
                      type="number"
                      value={height}
                      onChange={(e) => setHeight(Number(e.target.value) || 5)}
                      min={3}
                      max={10}
                      className="h-8 text-xs"
                    />
                  </div>
                </div>
                {theme === "custom" && (
                  <Input
                    placeholder={t("benchmark.graph.customThemePlaceholder")}
                    value={customThemeId}
                    onChange={(e) => setCustomThemeId(e.target.value)}
                    className="h-8 w-48 text-xs"
                  />
                )}
                {graphError && (
                  <div className="flex items-center gap-2 rounded-lg bg-destructive/10 px-3 py-2 text-ui-12 text-destructive">
                    <HugeiconsIcon
                      icon={AlertCircleIcon}
                      className="size-4 shrink-0"
                    />
                    {graphError}
                  </div>
                )}
                <Button
                  variant="outline"
                  size="sm"
                  className="ml-auto gap-1.5 rounded-full"
                  onClick={handleStyledExport}
                  disabled={
                    generating ||
                    orderedIds.length === 0 ||
                    (theme === "custom" && !customThemeId.trim())
                  }
                >
                  {generating ? (
                    <Spinner className="size-3.5" />
                  ) : (
                    <HugeiconsIcon icon={Download01Icon} className="size-3.5" />
                  )}
                  {generating
                    ? t("benchmark.graph.generating")
                    : t("benchmark.graph.exportStyled")}
                </Button>
              </div>
            )}
          </div>
        </div>

        <DialogFooter>
          <ChartSaveMenu svgRef={exportRef} fileBase="evals-comparison" />
          <Button
            variant="outline"
            size="sm"
            onClick={() => onOpenChange(false)}
          >
            {t("common.close")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
