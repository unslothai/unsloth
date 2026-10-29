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
import { BENCH_CARD } from "@/features/benchmarks/components/bench-ui";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
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
import { useModelPalette } from "../lib/model-colors";
import { modelName, modelsOf } from "../lib/score-bars";

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
  const palette = useModelPalette();
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

  const models = modelsOf(loadedDetails);
  const color = (model: string) =>
    palette[models.indexOf(modelName(model)) % palette.length] ?? palette[0];

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
                    className={cn(BENCH_CARD, "flex flex-col gap-3 px-5 py-4")}
                  >
                    {/* Header: the sweeps tile, label then the headline number */}
                    <div className="flex min-w-0 flex-col gap-1.5">
                      <span className="flex min-w-0 items-center gap-1.5 text-ui-11 font-medium tracking-nav text-muted-foreground">
                        <span
                          className="size-2 shrink-0 rounded-[2px]"
                          style={{ background: color(detail.model) }}
                          aria-hidden={true}
                        />
                        <span className="truncate">{detail.task}</span>
                      </span>
                      {primary && (
                        <span className="truncate font-heading text-ui-22 font-semibold leading-tight tracking-[-0.02em] tabular-nums text-foreground">
                          {formatMetricValue(primary)}
                          {isRatioScore(primary) && "%"}
                          {accuracy && (
                            <span className="ml-2 text-ui-11 font-normal tracking-normal text-muted-foreground">
                              {detail.correct_count}/{detail.total_count}
                            </span>
                          )}
                        </span>
                      )}
                      <span
                        className="truncate text-ui-11 text-muted-foreground"
                        title={detail.model}
                      >
                        {modelName(detail.model)} ·{" "}
                        {formatDate(detail.created_at)}
                      </span>
                    </div>

                    {/* All metrics */}
                    <div className="space-y-1 border-t border-border/40 pt-2">
                      {detail.metrics.filter(isDisplayMetric).map((m) => (
                        <div
                          key={m.name}
                          className="flex items-center justify-between text-ui-12"
                        >
                          <span className="truncate text-muted-foreground">
                            {m.name}
                          </span>
                          <span className="shrink-0 pl-2 font-medium tabular-nums text-foreground">
                            {formatMetricValue(m)}
                            {isRatioScore(m) ? "%" : ""}
                          </span>
                        </div>
                      ))}
                    </div>

                    {/* Sample results summary */}
                    {detail.samples.length > 0 && (
                      <div className="space-y-1 border-t border-border/40 pt-2">
                        <div className="text-ui-10 uppercase tracking-wider font-medium text-muted-foreground">
                          {t("benchmark.compare.sampleSummary")}
                        </div>
                        <div className="flex items-center gap-3 text-ui-12 tabular-nums">
                          <span className="flex items-center gap-1 text-foreground">
                            <HugeiconsIcon
                              icon={CheckmarkCircle01Icon}
                              className="size-3"
                            />
                            {detail.correct_count}
                          </span>
                          <span className="flex items-center gap-1 text-muted-foreground">
                            <HugeiconsIcon
                              icon={CancelCircleIcon}
                              className="size-3"
                            />
                            {detail.total_count - detail.correct_count}
                          </span>
                          <span className="text-muted-foreground">
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
              <section className={cn(BENCH_CARD, "mt-4 flex flex-col gap-2 p-3 sm:p-4")}>
                <span className="px-1 text-ui-13 font-medium text-foreground">
                  {t("benchmark.compare.metricBreakdown")}
                </span>
                <div className="overflow-x-auto rounded-xl">
                  <table className="w-full text-ui-12">
                    <thead>
                      <tr className="text-left text-ui-10 uppercase tracking-wider text-muted-foreground [&>th]:px-3 [&>th]:py-2.5 [&>th]:font-medium">
                        <th>{t("benchmark.compare.metric")}</th>
                        {loadedDetails.map((d) => (
                          <th key={d.id} className="text-right">
                            <span className="inline-flex items-center gap-1.5">
                              <span
                                className="size-2 shrink-0 rounded-[2px]"
                                style={{ background: color(d.model) }}
                                aria-hidden={true}
                              />
                              {d.task}
                            </span>
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
                            className="border-t border-border/40 tabular-nums [&>td]:whitespace-nowrap [&>td]:px-3 [&>td]:py-2"
                          >
                            <td className="text-foreground">
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
                                  className={cn(
                                    "text-right",
                                    isBest
                                      ? "font-semibold text-foreground"
                                      : "text-muted-foreground",
                                  )}
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
              </section>
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
