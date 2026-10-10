// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Spinner } from "@/components/ui/spinner";
import { useDebouncedValue } from "@/hooks";
import { useT } from "@/i18n";
import { toast } from "@/lib/toast";
import {
  AlertCircleIcon,
  ArrowLeftIcon,
  ArrowUpDownIcon,
  CancelCircleIcon,
  CheckmarkCircle01Icon,
  Delete01Icon,
  RefreshIcon,
  Search01Icon,
  WorkHistoryIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  type BenchmarkRunDetail,
  type BenchmarkRunMetric,
  type BenchmarkRunSummary,
  type BenchmarkSampleResult,
  deleteBenchmarkRun,
  exportBenchmarkRuns,
  getBenchmarkRunDetail,
  listBenchmarkRuns,
} from "../api/benchmark-api";
import { GraphDialog } from "./benchmark-graph-dialog";

const NON_METRIC_PATTERNS = new Set([
  "sample_len",
  "num_samples",
  "n_samples",
  "effective_samples",
  "alias",
  "name",
  "bootstrap_iters",
]);

function isDisplayMetric(m: { name: string }): boolean {
  const lower = m.name.toLowerCase().trim();
  if (NON_METRIC_PATTERNS.has(lower)) {
    return false;
  }
  if (lower.startsWith("sample_len")) {
    return false;
  }
  return true;
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

function getDefaultScore(run: BenchmarkRunSummary): number | null {
  if (!run.default_metric) return null;
  const m = run.metrics.find((x) => x.name === run.default_metric);
  return m?.score ?? null;
}

function formatDate(iso: string): string {
  try {
    return new Date(iso).toLocaleDateString(undefined, {
      day: "numeric",
      month: "short",
      year: "numeric",
    });
  } catch {
    return iso;
  }
}

type SortOption =
  | "date_desc"
  | "date_asc"
  | "score_desc"
  | "score_asc"
  | "task_asc"
  | "model_asc";

function sortRuns(
  runs: BenchmarkRunSummary[],
  sort: SortOption,
): BenchmarkRunSummary[] {
  const sorted = [...runs];
  switch (sort) {
    case "date_desc":
      return sorted.sort(
        (a, b) =>
          new Date(b.created_at).getTime() - new Date(a.created_at).getTime(),
      );
    case "date_asc":
      return sorted.sort(
        (a, b) =>
          new Date(a.created_at).getTime() - new Date(b.created_at).getTime(),
      );
    case "score_desc":
      return sorted.sort((a, b) => {
        const sa = getDefaultScore(a) ?? -1;
        const sb = getDefaultScore(b) ?? -1;
        return sb - sa;
      });
    case "score_asc":
      return sorted.sort((a, b) => {
        const sa = getDefaultScore(a) ?? -1;
        const sb = getDefaultScore(b) ?? -1;
        return sa - sb;
      });
    case "task_asc":
      return sorted.sort((a, b) => a.task.localeCompare(b.task));
    case "model_asc":
      return sorted.sort((a, b) => a.model.localeCompare(b.model));
    default:
      return sorted;
  }
}

function scoreColor(score: number): string {
  if (score >= 0.75) return "text-emerald-600 dark:text-emerald-400";
  if (score >= 0.5) return "text-amber-600 dark:text-amber-400";
  return "text-red-600 dark:text-red-400";
}

function filterRuns(
  runs: BenchmarkRunSummary[],
  query: string,
): BenchmarkRunSummary[] {
  if (!query.trim()) {
    return runs;
  }
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
  return runs.filter((r) => {
    const haystack = `${r.task} ${r.model}`.toLowerCase();
    return tokens.every((token) => haystack.includes(token));
  });
}

function RunCard({
  run,
  onSelect,
  onDelete,
  selected,
  onToggleSelect,
}: {
  run: BenchmarkRunSummary;
  onSelect: () => void;
  onDelete: () => void;
  selected: boolean;
  onToggleSelect: () => void;
}) {
  const t = useT();
  const score = getDefaultScore(run);
  const accuracy = score != null ? (score * 100).toFixed(1) : null;

  return (
    <div className="group flex w-full items-center gap-4 rounded-lg border border-border/60 bg-card px-4 py-3 transition-colors hover:bg-accent/50">
      <div className="flex shrink-0 items-center">
        <Checkbox
          checked={selected}
          onCheckedChange={() => onToggleSelect()}
          aria-label={t("benchmark.history.selectRun")}
        />
      </div>
      <button
        type="button"
        onClick={onSelect}
        className="flex min-w-0 flex-1 items-center gap-4 text-left"
      >
        <div className="flex w-20 shrink-0 items-baseline justify-center">
          {accuracy != null ? (
            <span
              className={`text-lg font-bold tabular-nums ${scoreColor(score ?? 0)}`}
            >
              {accuracy}
              <span className="text-xs font-medium">%</span>
            </span>
          ) : (
            <span className="text-xs text-muted-foreground">--</span>
          )}
        </div>
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          <div className="flex items-center gap-2">
            <span className="truncate text-sm font-medium">{run.task}</span>
            <Badge variant="outline" className="shrink-0 text-ui-10">
              {t("benchmark.history.samples", { count: run.n_samples })}
            </Badge>
            {run.num_fewshot != null && run.num_fewshot > 0 && (
              <Badge variant="outline" className="shrink-0 text-ui-10">
                {t("benchmark.history.shotCount", { count: run.num_fewshot })}
              </Badge>
            )}
          </div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground/70">
            <span className="truncate">{run.model}</span>
            <span className="shrink-0">·</span>
            <span className="shrink-0 tabular-nums">
              {formatDate(run.created_at)}
            </span>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-1">
          {run.metrics
            .filter(isDisplayMetric)
            .slice(0, 3)
            .map((m) => (
              <Badge
                key={m.name}
                variant="secondary"
                className="text-ui-10 font-mono"
              >
                {m.name}: {formatMetricValue(m)}
                {m.score >= 0 && m.score <= 1 ? "%" : ""}
              </Badge>
            ))}
        </div>
      </button>
      <button
        type="button"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
        className="shrink-0 rounded p-1 text-muted-foreground/40 opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
        title={t("common.delete")}
      >
        <HugeiconsIcon icon={Delete01Icon} className="size-4" />
      </button>
    </div>
  );
}

function SampleRow({ sample }: { sample: BenchmarkSampleResult }) {
  const t = useT();
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border-b border-border/30 last:border-0">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-3 px-4 py-2.5 text-left text-xs transition-colors hover:bg-accent/30"
      >
        <span className="shrink-0">
          {sample.correct ? (
            <HugeiconsIcon
              icon={CheckmarkCircle01Icon}
              className="size-4 text-emerald-500"
            />
          ) : (
            <HugeiconsIcon
              icon={CancelCircleIcon}
              className="size-4 text-red-400"
            />
          )}
        </span>
        <span className="line-clamp-1 flex-1 font-medium">
          Q{sample.doc_id + 1}: {sample.question}
        </span>
        <span className="shrink-0 text-muted-foreground/60">
          {sample.response ?? t("benchmark.history.noResponse")}
        </span>
      </button>
      {expanded && (
        <div className="space-y-2 border-t border-border/20 bg-accent/10 px-4 py-3 font-mono text-xs leading-relaxed">
          <div>
            <span className="font-semibold text-foreground/70">
              {t("benchmark.history.question")}
            </span>
            <p className="mt-0.5 whitespace-pre-wrap text-foreground/80">
              {sample.question}
            </p>
          </div>
          <div>
            <span className="font-semibold text-emerald-600">
              {t("benchmark.history.expected")}
            </span>
            <p className="mt-0.5 whitespace-pre-wrap text-emerald-700">
              {sample.target}
            </p>
          </div>
          <div>
            <span className="font-semibold text-amber-600">
              {t("benchmark.history.got")}
            </span>
            <p className="mt-0.5 whitespace-pre-wrap text-amber-700">
              {sample.response ?? t("benchmark.history.noValidResponse")}
            </p>
          </div>
          {sample.raw_response && (
            <div>
              <span className="font-semibold text-muted-foreground">
                {t("benchmark.history.rawOutput")}
              </span>
              <p className="mt-0.5 whitespace-pre-wrap text-muted-foreground/70">
                {sample.raw_response}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RunDetailView({
  detail,
  loading,
  onBack,
}: {
  detail: BenchmarkRunDetail | null;
  loading: boolean;
  onBack: () => void;
}) {
  const t = useT();
  const accuracy =
    detail && detail.total_count > 0
      ? ((detail.correct_count / detail.total_count) * 100).toFixed(1)
      : null;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2">
        <Button variant="ghost" size="sm" onClick={onBack}>
          <HugeiconsIcon icon={ArrowLeftIcon} className="mr-1 size-4" />
          {t("benchmark.history.back")}
        </Button>
      </div>

      {loading && (
        <div className="flex items-center justify-center py-12">
          <Spinner className="size-5" />
        </div>
      )}

      {detail && (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-lg border border-border/60 bg-card p-3">
              <div className="text-ui-10 font-medium text-muted-foreground/60">
                {t("benchmark.history.task")}
              </div>
              <div className="mt-0.5 text-sm font-semibold">{detail.task}</div>
            </div>
            <div className="rounded-lg border border-border/60 bg-card p-3">
              <div className="text-ui-10 font-medium text-muted-foreground/60">
                {t("benchmark.history.accuracy")}
              </div>
              <div
                className={`mt-0.5 text-lg font-bold tabular-nums ${scoreColor(detail.correct_count / Math.max(detail.total_count, 1))}`}
              >
                {accuracy}%
                <span className="ml-1 text-xs font-normal text-muted-foreground/60">
                  ({detail.correct_count}/{detail.total_count})
                </span>
              </div>
            </div>
            <div className="rounded-lg border border-border/60 bg-card p-3">
              <div className="text-ui-10 font-medium text-muted-foreground/60">
                {t("benchmark.history.model")}
              </div>
              <div
                className="mt-0.5 truncate text-sm font-medium"
                title={detail.model}
              >
                {detail.model}
              </div>
            </div>
            <div className="rounded-lg border border-border/60 bg-card p-3">
              <div className="text-ui-10 font-medium text-muted-foreground/60">
                {t("benchmark.history.date")}
              </div>
              <div className="mt-0.5 text-sm tabular-nums">
                {formatDate(detail.created_at)}
              </div>
            </div>
          </div>

          {detail.metrics.filter(isDisplayMetric).length > 0 && (
            <div className="flex flex-wrap gap-2">
              {detail.metrics.filter(isDisplayMetric).map((m) => (
                <Badge
                  key={m.name}
                  variant="secondary"
                  className="gap-1 px-3 py-1.5 text-xs"
                >
                  <span className="text-muted-foreground/70">{m.name}:</span>
                  <span className={scoreColor(m.score)}>
                    {formatMetricValue(m)}
                    {m.score >= 0 && m.score <= 1 ? "%" : ""}
                  </span>
                  {m.stderr && m.stderr !== "N/A" && (
                    <span className="text-muted-foreground/50">
                      ±{m.stderr}
                    </span>
                  )}
                </Badge>
              ))}
            </div>
          )}

          <Progress
            value={
              detail.total_count > 0
                ? (detail.correct_count / detail.total_count) * 100
                : 0
            }
            className="h-2 bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)]"
            indicatorClassName={
              detail.correct_count / Math.max(detail.total_count, 1) >= 0.8
                ? "bg-green-400"
                : detail.correct_count / Math.max(detail.total_count, 1) >= 0.5
                  ? "bg-orange-400"
                  : "bg-red-400"
            }
          />

          <div>
            <h3 className="mb-2 text-sm font-medium text-foreground/80">
              {t("benchmark.history.perSampleResults", {
                count: detail.samples.length,
              })}
            </h3>
            <div className="max-h-[calc(500px*var(--ui-space-scale,1))] overflow-y-auto rounded-lg border border-border/40">
              {detail.samples.length === 0 ? (
                <div className="flex items-center justify-center py-8 text-xs text-muted-foreground/60">
                  {t("benchmark.history.noSampleData")}
                </div>
              ) : (
                detail.samples.map((s) => (
                  <SampleRow key={s.doc_id} sample={s} />
                ))
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function CompareColumn({ detail }: { detail: BenchmarkRunDetail }) {
  const t = useT();
  const accuracy =
    detail.total_count > 0
      ? ((detail.correct_count / detail.total_count) * 100).toFixed(1)
      : null;

  return (
    <div className="flex min-w-[calc(640px*var(--ui-space-scale,1))] flex-1 flex-col gap-4 rounded-lg border border-border/60 bg-card p-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded-lg border border-border/60 bg-card p-3">
          <div className="text-ui-10 font-medium text-muted-foreground/60">
            {t("benchmark.history.task")}
          </div>
          <div className="mt-0.5 text-sm font-semibold">{detail.task}</div>
        </div>
        <div className="rounded-lg border border-border/60 bg-card p-3">
          <div className="text-ui-10 font-medium text-muted-foreground/60">
            {t("benchmark.history.accuracy")}
          </div>
          <div
            className={`mt-0.5 text-lg font-bold tabular-nums ${scoreColor(detail.correct_count / Math.max(detail.total_count, 1))}`}
          >
            {accuracy}%
            <span className="ml-1 text-xs font-normal text-muted-foreground/60">
              ({detail.correct_count}/{detail.total_count})
            </span>
          </div>
        </div>
        <div className="rounded-lg border border-border/60 bg-card p-3">
          <div className="text-ui-10 font-medium text-muted-foreground/60">
            {t("benchmark.history.model")}
          </div>
          <div
            className="mt-0.5 truncate text-sm font-medium"
            title={detail.model}
          >
            {detail.model}
          </div>
        </div>
        <div className="rounded-lg border border-border/60 bg-card p-3">
          <div className="text-ui-10 font-medium text-muted-foreground/60">
            {t("benchmark.history.date")}
          </div>
          <div className="mt-0.5 text-sm tabular-nums">
            {formatDate(detail.created_at)}
          </div>
        </div>
      </div>

      {detail.metrics.filter(isDisplayMetric).length > 0 && (
        <div className="flex flex-wrap gap-2">
          {detail.metrics.filter(isDisplayMetric).map((m) => (
            <Badge
              key={m.name}
              variant="secondary"
              className="gap-1 px-3 py-1.5 text-xs"
            >
              <span className="text-muted-foreground/70">{m.name}:</span>
              <span className={scoreColor(m.score)}>
                {formatMetricValue(m)}
                {m.score >= 0 && m.score <= 1 ? "%" : ""}
              </span>
              {m.stderr && m.stderr !== "N/A" && (
                <span className="text-muted-foreground/50">±{m.stderr}</span>
              )}
            </Badge>
          ))}
        </div>
      )}

      <Progress
        value={
          detail.total_count > 0
            ? (detail.correct_count / detail.total_count) * 100
            : 0
        }
        className="h-2 bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)]"
        indicatorClassName={
          detail.correct_count / Math.max(detail.total_count, 1) >= 0.8
            ? "bg-green-400"
            : detail.correct_count / Math.max(detail.total_count, 1) >= 0.5
              ? "bg-orange-400"
              : "bg-red-400"
        }
      />

      <div>
        <h3 className="mb-2 text-sm font-medium text-foreground/80">
          {t("benchmark.history.perSampleResults", {
            count: detail.samples.length,
          })}
        </h3>
        <div className="max-h-[calc(500px*var(--ui-space-scale,1))] overflow-y-auto rounded-lg border border-border/40">
          {detail.samples.length === 0 ? (
            <div className="flex items-center justify-center py-8 text-xs text-muted-foreground/60">
              {t("benchmark.history.noSampleData")}
            </div>
          ) : (
            detail.samples.map((s) => <SampleRow key={s.doc_id} sample={s} />)
          )}
        </div>
      </div>
    </div>
  );
}

export function BenchmarkHistoryPanel() {
  const t = useT();
  const [runs, setRuns] = useState<BenchmarkRunSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<BenchmarkRunDetail | null>(
    null,
  );
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortBy, setSortBy] = useState<SortOption>("date_desc");
  const debouncedSearch = useDebouncedValue(searchQuery, 200);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [graphOpen, setGraphOpen] = useState(false);
  const [exporting, setExporting] = useState(false);

  const displayRuns = useMemo(() => {
    return sortRuns(filterRuns(runs, debouncedSearch), sortBy);
  }, [runs, debouncedSearch, sortBy]);

  const toggleSelect = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set());
  }, []);

  const selectedRunObjects = useMemo(() => {
    return runs.filter((r) => selectedIds.has(r.id));
  }, [runs, selectedIds]);

  const fetchRuns = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listBenchmarkRuns();
      setRuns(data.runs);
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : t("benchmark.history.failedToLoad"),
      );
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    fetchRuns();
  }, [fetchRuns]);

  const handleSelect = useCallback(
    async (runId: string) => {
      setLoadingDetail(true);
      setSelectedRun(null);
      try {
        const detail = await getBenchmarkRunDetail(runId);
        setSelectedRun(detail);
      } catch (err) {
        setError(
          err instanceof Error
            ? err.message
            : t("benchmark.history.failedToLoadDetail"),
        );
      } finally {
        setLoadingDetail(false);
      }
    },
    [t],
  );

  const handleDelete = useCallback(
    async (runId: string) => {
      try {
        await deleteBenchmarkRun(runId);
        setRuns((prev) => prev.filter((r) => r.id !== runId));
        if (selectedRun?.id === runId) {
          setSelectedRun(null);
        }
      } catch {
        // ignore
      }
      setDeleteTarget(null);
    },
    [selectedRun],
  );

  const handleBulkDelete = useCallback(async () => {
    const ids = Array.from(selectedIds);
    for (const id of ids) {
      try {
        await deleteBenchmarkRun(id);
      } catch {
        // ignore individual failures
      }
    }
    setRuns((prev) => prev.filter((r) => !selectedIds.has(r.id)));
    if (selectedRun && selectedIds.has(selectedRun.id)) {
      setSelectedRun(null);
    }
    setSelectedIds(new Set());
    setDeleteTarget(null);
  }, [selectedIds, selectedRun]);

  const detail = selectedRun;

  const handleCompare = useCallback(() => {
    if (selectedIds.size < 2) {
      return;
    }
    setCompareMode(true);
  }, [selectedIds]);

  const [compareMode, setCompareMode] = useState(false);
  const [compareDetails, setCompareDetails] = useState<BenchmarkRunDetail[]>(
    [],
  );
  const [loadingCompare, setLoadingCompare] = useState(false);

  const startCompare = useCallback(async () => {
    const ids = Array.from(selectedIds);
    setLoadingCompare(true);
    const details: BenchmarkRunDetail[] = [];
    for (const id of ids) {
      try {
        const d = await getBenchmarkRunDetail(id);
        details.push(d);
      } catch {
        // skip
      }
    }
    setCompareDetails(details);
    setLoadingCompare(false);
    setCompareMode(true);
    setSelectedIds(new Set());
  }, [selectedIds]);

  if (compareMode || loadingCompare) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setCompareMode(false);
              setCompareDetails([]);
            }}
          >
            <HugeiconsIcon icon={ArrowLeftIcon} className="mr-1 size-4" />
            {t("benchmark.history.back")}
          </Button>
        </div>

        {loadingCompare && (
          <div className="flex items-center justify-center py-12">
            <Spinner className="size-5" />
          </div>
        )}

        {!loadingCompare && compareDetails.length > 0 && (
          <div className="overflow-x-auto">
            <div className="flex gap-4" style={{ minWidth: "min-content" }}>
              {compareDetails.map((detail) => (
                <CompareColumn key={detail.id} detail={detail} />
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  if (selectedRun || loadingDetail) {
    return (
      <RunDetailView
        detail={detail}
        loading={loadingDetail}
        onBack={() => {
          setSelectedRun(null);
          setLoadingDetail(false);
        }}
      />
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <HugeiconsIcon
            icon={WorkHistoryIcon}
            className="size-4 text-muted-foreground"
          />
          {selectedIds.size > 0 ? (
            <span className="text-sm font-medium">
              {t("benchmark.history.selected", { count: selectedIds.size })}
            </span>
          ) : (
            <>
              <span className="text-sm font-medium">
                {t("benchmark.history.title")}
              </span>
              {!loading && (
                <span className="text-xs text-muted-foreground/60">
                  ({displayRuns.length}{" "}
                  {displayRuns.length !== 1
                    ? t("benchmark.history.runs")
                    : t("benchmark.history.run")}
                  )
                </span>
              )}
            </>
          )}
        </div>
        {selectedIds.size > 0 ? (
          <div className="flex items-center gap-1.5">
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-xs"
              onClick={() => setGraphOpen(true)}
            >
              {t("benchmark.history.graph")}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-xs"
              disabled={exporting}
              onClick={async () => {
                setExporting(true);
                try {
                  await exportBenchmarkRuns(Array.from(selectedIds));
                } catch (err) {
                  toast.error(
                    err instanceof Error ? err.message : t("benchmark.history.exportError"),
                  );
                } finally {
                  setExporting(false);
                }
              }}
            >
              {exporting ? (
                <Spinner className="mr-1 size-3" />
              ) : null}
              {t("benchmark.history.export")}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-xs"
              disabled={selectedIds.size < 2}
              onClick={startCompare}
            >
              {t("benchmark.history.compare")}
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-7 text-xs text-destructive hover:text-destructive"
              onClick={() => setDeleteTarget("bulk")}
            >
              {t("common.delete")}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs"
              onClick={clearSelection}
            >
              {t("benchmark.history.done")}
            </Button>
          </div>
        ) : (
          <Button
            variant="ghost"
            size="sm"
            onClick={fetchRuns}
            disabled={loading}
          >
            <HugeiconsIcon
              icon={RefreshIcon}
              className={`mr-1 size-3.5 ${loading ? "animate-spin" : ""}`}
            />
            {t("benchmark.history.refresh")}
          </Button>
        )}
      </div>

      {runs.length > 0 && (
        <div className="flex items-center gap-2">
          <div className="relative flex-1">
            <HugeiconsIcon
              icon={Search01Icon}
              className="pointer-events-none absolute left-3 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground/60"
            />
            <Input
              placeholder={t("benchmark.history.searchPlaceholder")}
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="h-8 pl-8 text-xs"
            />
          </div>
          <Select
            value={sortBy}
            onValueChange={(v) => setSortBy(v as SortOption)}
          >
            <SelectTrigger size="sm" className="w-auto gap-1.5 text-xs">
              <HugeiconsIcon icon={ArrowUpDownIcon} className="size-3.5" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="date_desc">
                {t("benchmark.history.sortDateDesc")}
              </SelectItem>
              <SelectItem value="date_asc">
                {t("benchmark.history.sortDateAsc")}
              </SelectItem>
              <SelectItem value="score_desc">
                {t("benchmark.history.sortScoreDesc")}
              </SelectItem>
              <SelectItem value="score_asc">
                {t("benchmark.history.sortScoreAsc")}
              </SelectItem>
              <SelectItem value="task_asc">
                {t("benchmark.history.sortTaskAsc")}
              </SelectItem>
              <SelectItem value="model_asc">
                {t("benchmark.history.sortModelAsc")}
              </SelectItem>
            </SelectContent>
          </Select>
        </div>
      )}

      {loading && runs.length === 0 && (
        <div className="flex items-center justify-center py-8">
          <Spinner className="size-5" />
        </div>
      )}

      {error && (
        <div className="flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/5 px-3 py-2 text-xs text-destructive">
          <HugeiconsIcon icon={AlertCircleIcon} className="size-4 shrink-0" />
          {error}
        </div>
      )}

      {!loading && runs.length === 0 && !error && (
        <div className="flex flex-col items-center justify-center py-8 text-xs text-muted-foreground/60">
          <HugeiconsIcon
            icon={WorkHistoryIcon}
            className="mb-2 size-8 text-muted-foreground/20"
          />
          {t("benchmark.history.noResults")}
        </div>
      )}

      {!loading && runs.length > 0 && displayRuns.length === 0 && (
        <div className="flex flex-col items-center justify-center py-8 text-xs text-muted-foreground/60">
          <HugeiconsIcon
            icon={Search01Icon}
            className="mb-2 size-8 text-muted-foreground/20"
          />
          {t("benchmark.history.noMatches")}
        </div>
      )}

      {displayRuns.length > 0 && (
        <div className="space-y-2">
          {displayRuns.map((run) => (
            <RunCard
              key={run.id}
              run={run}
              onSelect={() => {
                handleSelect(run.id);
              }}
              onDelete={() => setDeleteTarget(run.id)}
              selected={selectedIds.has(run.id)}
              onToggleSelect={() => toggleSelect(run.id)}
            />
          ))}
        </div>
      )}

      <AlertDialog
        open={deleteTarget !== null}
        onOpenChange={(o) => {
          if (!o) {
            setDeleteTarget(null);
          }
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>
              {deleteTarget === "bulk"
                ? t("benchmark.history.deleteBulkTitle")
                : t("benchmark.history.deleteTitle")}
            </AlertDialogTitle>
            <AlertDialogDescription>
              {deleteTarget === "bulk"
                ? t("benchmark.history.deleteBulkDescription", {
                    count: selectedIds.size,
                  })
                : t("benchmark.history.deleteDescription")}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("common.cancel")}</AlertDialogCancel>
            <AlertDialogAction
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              onClick={() => {
                if (deleteTarget === "bulk") {
                  handleBulkDelete();
                } else if (deleteTarget) {
                  handleDelete(deleteTarget);
                }
              }}
            >
              {t("common.delete")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <GraphDialog
        open={graphOpen}
        onOpenChange={setGraphOpen}
        selectedRuns={selectedRunObjects}
      />
    </div>
  );
}
