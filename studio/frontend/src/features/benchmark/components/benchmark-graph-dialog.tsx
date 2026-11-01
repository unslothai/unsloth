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
import { useT } from "@/i18n";
import {
  ChevronDownStandardIcon,
  ChevronUpStandardIcon,
} from "@/lib/chevron-icons";
import { AlertCircleIcon, Download01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useMemo, useState } from "react";
import {
  BenchmarkRequestError,
  type BenchmarkRunMetric,
  type BenchmarkRunSummary,
  generateBenchmarkGraph,
} from "../api/benchmark-api";

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
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);

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
      { label: t("benchmark.graph.accuracy"), value: "__accuracy__" },
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

  const [pickedMetric, setSelectedMetric] = useState("__accuracy__");
  // The dialog stays mounted across selections; a metric the current runs lack
  // would chart every bar as 0, so fall back to accuracy.
  const selectedMetric = availableMetrics.some((m) => m.value === pickedMetric)
    ? pickedMetric
    : "__accuracy__";

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

  const handleGenerate = useCallback(async () => {
    setGenerating(true);
    setPreviewUrl(null);
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
      const url = URL.createObjectURL(blob);
      setPreviewUrl(url);
    } catch (err) {
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

  const handleDownload = useCallback(() => {
    if (!previewUrl) {
      return;
    }
    const a = document.createElement("a");
    a.href = previewUrl;
    a.download = "benchmark-chart.png";
    a.click();
  }, [previewUrl]);

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
          {/* Reorderable runs list */}
          <div>
            <div className="mb-1.5 text-xs font-medium text-muted-foreground/70">
              {t("benchmark.graph.orderLabel")}
            </div>
            <div className="max-h-36 space-y-1 overflow-y-auto rounded-lg border border-border/40 p-2">
              {orderedRuns.map((run, index) => (
                <div
                  key={run.id}
                  className="flex items-center gap-2 rounded-md bg-accent/30 px-2 py-1.5 text-xs"
                >
                  <span className="min-w-0 flex-1 truncate">
                    {runLabel(run)}
                  </span>
                  <span className="shrink-0 font-mono text-muted-foreground/60">
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
                      className="text-muted-foreground/40 hover:text-foreground disabled:opacity-30"
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
                      className="text-muted-foreground/40 hover:text-foreground disabled:opacity-30"
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

          {/* Chart settings */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div>
              <div className="mb-1 text-xs font-medium text-muted-foreground/70">
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
              <div className="mb-1 text-xs font-medium text-muted-foreground/70">
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

            <div>
              <div className="mb-1 text-xs font-medium text-muted-foreground/70">
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
              <div className="mb-1 text-xs font-medium text-muted-foreground/70">
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

          <div className="flex items-center gap-3">
            <div className="text-xs font-medium text-muted-foreground/70">
              {t("benchmark.graph.theme")}
            </div>
            <Select value={theme} onValueChange={setTheme}>
              <SelectTrigger size="sm" className="w-auto text-xs">
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
            {theme === "custom" && (
              <Input
                placeholder={t("benchmark.graph.customThemePlaceholder")}
                value={customThemeId}
                onChange={(e) => setCustomThemeId(e.target.value)}
                className="h-8 w-48 text-xs"
              />
            )}
          </div>

          {/* Preview */}
          {graphError && (
            <div className="flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
              <HugeiconsIcon
                icon={AlertCircleIcon}
                className="size-4 shrink-0"
              />
              {graphError}
            </div>
          )}

          {generating && (
            <div className="flex items-center justify-center rounded-lg border border-border/40 bg-accent/20 py-12">
              <Spinner className="size-5" />
              <span className="ml-2 text-xs text-muted-foreground/60">
                {t("benchmark.graph.generating")}
              </span>
            </div>
          )}

          {previewUrl && !generating && (
            <div className="overflow-hidden rounded-lg border border-border/40">
              <img src={previewUrl} alt="Benchmark chart" className="w-full" />
            </div>
          )}
        </div>

        <DialogFooter>
          {previewUrl && (
            <Button variant="outline" size="sm" onClick={handleDownload}>
              <HugeiconsIcon icon={Download01Icon} className="mr-1 size-3.5" />
              {t("benchmark.graph.download")}
            </Button>
          )}
          <Button
            size="sm"
            onClick={handleGenerate}
            disabled={
              generating ||
              orderedIds.length === 0 ||
              (theme === "custom" && !customThemeId.trim())
            }
          >
            {t("benchmark.graph.generate")}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
