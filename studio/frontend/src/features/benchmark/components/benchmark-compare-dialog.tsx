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
import { Spinner } from "@/components/ui/spinner";
import { useT } from "@/i18n";
import {
  CancelCircleIcon,
  CheckmarkCircle01Icon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useEffect, useState } from "react";
import {
  type BenchmarkRunDetail,
  type BenchmarkRunMetric,
  type BenchmarkRunSummary,
  getBenchmarkRunDetail,
} from "../api/benchmark-api";

const ACCURACY_LIKE_RE = /(em|exact|strict|flexible|acc|f1|match|score)/i;

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

function isRatioScore(m: { name: string; score: number }): boolean {
  return ACCURACY_LIKE_RE.test(m.name) && m.score >= 0 && m.score <= 1;
}

function formatMetricValue(m: BenchmarkRunMetric): string {
  if (isRatioScore(m)) {
    return (m.score * 100).toFixed(1);
  }
  if (Number.isInteger(m.score)) {
    return String(m.score);
  }
  return m.score.toFixed(m.score < 1 ? 4 : 1);
}

function scoreColor(score: number): string {
  if (score >= 0.8) {
    return "text-emerald-500";
  }
  if (score >= 0.5) {
    return "amber-500";
  }
  return "text-red-500";
}

function formatDate(iso: string): string {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return iso;
  }
}

function pickPrimaryMetric(
  metrics: BenchmarkRunMetric[],
): BenchmarkRunMetric | undefined {
  return (
    metrics.find((m) => isDisplayMetric(m) && ACCURACY_LIKE_RE.test(m.name)) ??
    metrics.find(isDisplayMetric)
  );
}

interface CompareDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedRuns: BenchmarkRunSummary[];
}

export function CompareDialog({
  open,
  onOpenChange,
  selectedRuns,
}: CompareDialogProps) {
  const t = useT();
  const [details, setDetails] = useState<Map<string, BenchmarkRunDetail>>(
    new Map(),
  );
  const [loading, setLoading] = useState(false);
  const [loadingProgress, setLoadingProgress] = useState({
    loaded: 0,
    total: 0,
  });

  useEffect(() => {
    if (!open || selectedRuns.length === 0) {
      return;
    }

    let cancelled = false;
    const loadDetails = async () => {
      setLoading(true);
      setLoadingProgress({ loaded: 0, total: selectedRuns.length });
      const map = new Map<string, BenchmarkRunDetail>();

      for (let i = 0; i < selectedRuns.length; i++) {
        const run = selectedRuns[i];
        try {
          const detail = await getBenchmarkRunDetail(run.id);
          if (!cancelled) {
            map.set(run.id, detail);
            setLoadingProgress({ loaded: i + 1, total: selectedRuns.length });
          }
        } catch {
          // Skip failed loads
        }
      }

      if (!cancelled) {
        setDetails(map);
        setLoading(false);
      }
    };

    loadDetails();
    return () => {
      cancelled = true;
    };
  }, [open, selectedRuns]);

  const loadedDetails = selectedRuns
    .map((r) => details.get(r.id))
    .filter((d): d is BenchmarkRunDetail => d != null);

  // Collect all metric names across all runs
  const allMetricNames =
    loadedDetails.length > 0
      ? [
          ...new Set(
            loadedDetails.flatMap((d) =>
              d.metrics.filter(isDisplayMetric).map((m) => m.name),
            ),
          ),
        ]
      : [];

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl">
        <DialogHeader>
          <DialogTitle>{t("benchmark.compare.title")}</DialogTitle>
          <DialogDescription>
            {t("benchmark.compare.description", {
              count: selectedRuns.length,
            })}
          </DialogDescription>
        </DialogHeader>

        {loading && (
          <div className="flex items-center justify-center py-12">
            <Spinner className="size-5" />
            <span className="ml-2 text-xs text-muted-foreground/60">
              {t("benchmark.compare.loading", {
                loaded: loadingProgress.loaded,
                total: loadingProgress.total,
              })}
            </span>
          </div>
        )}

        {!loading && loadedDetails.length > 0 && (
          <div className="overflow-x-auto">
            <div
              className="grid gap-3"
              style={{
                gridTemplateColumns: `repeat(${loadedDetails.length}, minmax(200px, 1fr))`,
              }}
            >
              {loadedDetails.map((detail) => {
                const primary = pickPrimaryMetric(detail.metrics);
                const accuracy =
                  detail.total_count > 0
                    ? (
                        (detail.correct_count / detail.total_count) *
                        100
                      ).toFixed(1)
                    : null;

                return (
                  <div
                    key={detail.id}
                    className="flex flex-col gap-3 rounded-lg border border-border/60 bg-card p-3"
                  >
                    {/* Header */}
                    <div className="space-y-1">
                      <div className="text-sm font-semibold">{detail.task}</div>
                      <div
                        className="truncate text-xs text-muted-foreground/70"
                        title={detail.model}
                      >
                        {detail.model}
                      </div>
                      <div className="text-xs text-muted-foreground/50">
                        {formatDate(detail.created_at)}
                      </div>
                    </div>

                    {/* Primary score */}
                    {primary && (
                      <div className="text-center">
                        <span
                          className={`text-2xl font-bold tabular-nums ${scoreColor(primary.score)}`}
                        >
                          {formatMetricValue(primary)}
                          {isRatioScore(primary) && (
                            <span className="text-sm font-medium">%</span>
                          )}
                        </span>
                        {accuracy && (
                          <div className="text-xs text-muted-foreground/60">
                            ({detail.correct_count}/{detail.total_count})
                          </div>
                        )}
                      </div>
                    )}

                    {/* All metrics */}
                    <div className="space-y-1">
                      {detail.metrics.filter(isDisplayMetric).map((m) => (
                        <div
                          key={m.name}
                          className="flex items-center justify-between text-xs"
                        >
                          <span className="truncate text-muted-foreground/70">
                            {m.name}
                          </span>
                          <span
                            className={`shrink-0 pl-2 font-mono font-medium ${scoreColor(m.score)}`}
                          >
                            {formatMetricValue(m)}
                            {isRatioScore(m) ? "%" : ""}
                          </span>
                        </div>
                      ))}
                    </div>

                    {/* Sample results summary */}
                    {detail.samples.length > 0 && (
                      <div className="space-y-1 border-t border-border/30 pt-2">
                        <div className="text-ui-10 font-medium text-muted-foreground/60">
                          {t("benchmark.compare.sampleSummary")}
                        </div>
                        <div className="flex items-center gap-3 text-xs">
                          <span className="flex items-center gap-1 text-emerald-500">
                            <HugeiconsIcon
                              icon={CheckmarkCircle01Icon}
                              className="size-3"
                            />
                            {detail.correct_count}
                          </span>
                          <span className="flex items-center gap-1 text-red-400">
                            <HugeiconsIcon
                              icon={CancelCircleIcon}
                              className="size-3"
                            />
                            {detail.total_count - detail.correct_count}
                          </span>
                          <span className="text-muted-foreground/50">
                            {detail.total_count}{" "}
                            {t("benchmark.history.samples", {
                              count: detail.total_count,
                            })}
                          </span>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>

            {/* Cross-run metric comparison table */}
            {allMetricNames.length > 0 && loadedDetails.length > 1 && (
              <div className="mt-4">
                <div className="mb-2 text-xs font-medium text-muted-foreground/70">
                  {t("benchmark.compare.metricBreakdown")}
                </div>
                <div className="overflow-x-auto rounded-lg border border-border/40">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-border/30 bg-accent/20">
                        <th className="px-3 py-2 text-left font-medium text-muted-foreground/70">
                          {t("benchmark.compare.metric")}
                        </th>
                        {loadedDetails.map((d) => (
                          <th
                            key={d.id}
                            className="px-3 py-2 text-right font-medium text-muted-foreground/70"
                          >
                            {d.task}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {allMetricNames.map((metricName) => {
                        const scores = loadedDetails.map((d) => {
                          const m = d.metrics.find(
                            (x) => x.name === metricName && isDisplayMetric(x),
                          );
                          return m?.score ?? null;
                        });
                        const maxScore = Math.max(
                          ...scores.filter((s): s is number => s != null),
                        );

                        return (
                          <tr
                            key={metricName}
                            className="border-b border-border/20 last:border-0"
                          >
                            <td className="px-3 py-1.5 text-muted-foreground/70">
                              {metricName}
                            </td>
                            {scores.map((score, i) => {
                              const isBest =
                                score != null &&
                                score === maxScore &&
                                maxScore > 0;
                              return (
                                <td
                                  key={loadedDetails[i].id}
                                  className={`px-3 py-1.5 text-right font-mono ${
                                    isBest ? "font-bold text-emerald-500" : ""
                                  }`}
                                >
                                  {score != null
                                    ? `${(score * 100).toFixed(1)}%`
                                    : "—"}
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        <DialogFooter>
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
