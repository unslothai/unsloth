// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Scores at a glance: the latest run of each model on each task, as bars on one 0-100 scale,
// so two models on the same benchmark read side by side. The full list stays in history.

import { BarChart } from "@/features/benchmarks/components/bar-chart";
import { BENCH_CARD } from "@/features/benchmarks/components/bench-ui";
import { StatTile } from "@/features/benchmarks/components/stat-tile";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import { type ReactElement, useEffect, useMemo, useRef, useState } from "react";
import {
  type BenchmarkRunSummary,
  listBenchmarkRuns,
} from "../api/benchmark-api";
import { useModelPalette } from "../lib/model-colors";
import { fmtScore, latestScores, scoreboardGroups } from "../lib/score-bars";
import { useBenchmarkRuntimeStore } from "../stores/benchmark-runtime-store";
import { ChartSaveMenu } from "./chart-save-menu";

const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

export function EvalScoreboard(): ReactElement | null {
  const t = useT();
  const palette = useModelPalette();
  const running = useBenchmarkRuntimeStore(
    (s) => s.phase === "running" || s.phase === "starting",
  );
  const [runs, setRuns] = useState<BenchmarkRunSummary[]>([]);
  const exportRef = useRef<SVGSVGElement | null>(null);

  // On mount (even mid-run, e.g. after a reload) and again when a run ends, which lands in the list.
  useEffect(() => {
    let cancelled = false;
    void listBenchmarkRuns()
      .then((d) => {
        if (!cancelled) setRuns(d.runs);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [running]);

  const { groups, models, labels, best } = useMemo(
    () => latestScores(runs, palette.length),
    [runs, palette.length],
  );
  const bars = useMemo(
    () =>
      scoreboardGroups(
        groups,
        (model) => palette[models.indexOf(model)] ?? palette[0],
      ),
    [groups, models, palette],
  );

  if (groups.length === 0) return null;
  const color = (model: string) => palette[models.indexOf(model)] ?? palette[0];
  const chart = {
    groups: bars,
    max: 100,
    format: fmtScore,
    axisLabel: t("benchmark.chart.scoreAxis"),
    title: t("benchmark.chart.scoresTitle"),
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatTile
          label="Best score"
          value={best ? pct(best.score) : "—"}
          detail={best ? `${best.run.task} · ${best.label}` : undefined}
        />
        <StatTile
          label="Benchmarks"
          value={groups.length}
          detail={groups.map((g) => g.task).join(", ")}
        />
        <StatTile
          label="Models"
          value={models.length}
          detail="Latest run of each shown"
        />
      </div>

      <section className={cn(BENCH_CARD, "flex flex-col gap-3 p-3 sm:p-4")}>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 px-1">
          <span className="text-ui-13 font-medium text-foreground">Scores</span>
          {models.length > 1 && (
            <ul className="flex min-w-0 flex-wrap items-center gap-x-4 gap-y-1 text-ui-11 text-muted-foreground">
              {models.map((m) => (
                <li key={m} className="flex min-w-0 items-center gap-1.5">
                  <span
                    className="size-2.5 shrink-0 rounded-[3px]"
                    style={{ background: color(m) }}
                    aria-hidden={true}
                  />
                  <span className="truncate" title={m}>
                    {labels.get(m) ?? m}
                  </span>
                </li>
              ))}
            </ul>
          )}
          <div className="ml-auto">
            <ChartSaveMenu svgRef={exportRef} fileBase="evals-scores" />
          </div>
        </div>
        <div className="overflow-hidden rounded-xl">
          <BarChart {...chart} bare={true} />
        </div>
        {/* Exports carry the title the screen leaves off. */}
        <div className="hidden" aria-hidden={true}>
          <BarChart {...chart} bare={false} svgRef={exportRef} />
        </div>
      </section>
    </div>
  );
}
