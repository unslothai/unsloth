// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Scores at a glance: the latest run of each model on each task, as bars on one 0-100 scale,
// so two models on the same benchmark read side by side. The full list stays in history.

import { BENCH_CARD } from "@/features/benchmarks/components/bench-ui";
import { useTheme } from "@/features/settings";
import { cn } from "@/lib/utils";
import {
  type ReactElement,
  type ReactNode,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  type BenchmarkRunSummary,
  listBenchmarkRuns,
} from "../api/benchmark-api";
import { useBenchmarkRuntimeStore } from "../stores/benchmark-runtime-store";

// The Benchmarks page's validated categorical order, minus its neutral. Assigned by first
// appearance and never cycled: an eighth model is left to the history list.
const MODEL_COLORS = {
  light: [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
  ],
  dark: [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
  ],
};

interface Bar {
  model: string;
  score: number;
  stderr: number | null;
  run: BenchmarkRunSummary;
}

function modelName(model: string): string {
  return model.split(/[\\/]/).filter(Boolean).pop() ?? model;
}

function score(
  run: BenchmarkRunSummary,
): { score: number; stderr: number | null } | null {
  const m = run.metrics.find((x) => x.name === run.default_metric);
  if (!m || m.score < 0 || m.score > 1) return null;
  const err = m.stderr != null ? Number.parseFloat(m.stderr) : Number.NaN;
  return { score: m.score, stderr: Number.isFinite(err) ? err : null };
}

const pct = (n: number) => `${(n * 100).toFixed(1)}%`;

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
    <div className={cn(BENCH_CARD, "flex min-w-0 flex-col gap-1.5 px-5 py-4")}>
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

export function EvalScoreboard(): ReactElement | null {
  const { resolved } = useTheme();
  const palette = MODEL_COLORS[resolved === "dark" ? "dark" : "light"];
  const phase = useBenchmarkRuntimeStore((s) => s.phase);
  const [runs, setRuns] = useState<BenchmarkRunSummary[]>([]);

  // A finished run lands in the list, so read it again then.
  useEffect(() => {
    if (phase === "running" || phase === "starting") return;
    let cancelled = false;
    void listBenchmarkRuns()
      .then((d) => {
        if (!cancelled) setRuns(d.runs);
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [phase]);

  const { groups, models, best } = useMemo(() => {
    const newestFirst = [...runs].sort((a, b) =>
      b.created_at.localeCompare(a.created_at),
    );
    const models: string[] = [];
    for (const r of [...newestFirst].reverse()) {
      const name = modelName(r.model);
      if (!models.includes(name)) models.push(name);
    }
    const shown = models.slice(0, palette.length);
    const byTask = new Map<string, Bar[]>();
    for (const run of newestFirst) {
      const s = score(run);
      const model = modelName(run.model);
      if (!s || !shown.includes(model)) continue;
      const bars = byTask.get(run.task) ?? [];
      if (bars.some((b) => b.model === model)) continue;
      bars.push({ model, ...s, run });
      byTask.set(run.task, bars);
    }
    const groups = [...byTask.entries()].map(([task, bars]) => ({
      task,
      bars: bars.sort((a, b) => b.score - a.score),
    }));
    const best = groups
      .flatMap((g) => g.bars)
      .sort((a, b) => b.score - a.score)[0];
    return { groups, models: shown, best };
  }, [runs, palette.length]);

  if (groups.length === 0) return null;
  const color = (model: string) => palette[models.indexOf(model)] ?? palette[0];

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Tile
          label="Best score"
          value={best ? pct(best.score) : "—"}
          detail={best ? `${best.run.task} · ${best.model}` : undefined}
        />
        <Tile
          label="Benchmarks"
          value={groups.length}
          detail={groups.map((g) => g.task).join(", ")}
        />
        <Tile
          label="Models"
          value={models.length}
          detail="Latest run of each shown"
        />
      </div>

      <section className={cn(BENCH_CARD, "flex flex-col gap-5 p-4 sm:p-5")}>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5">
          <span className="text-ui-13 font-medium text-foreground">Scores</span>
          <span className="flex min-w-0 flex-wrap gap-x-3 gap-y-1 text-ui-11 text-muted-foreground">
            {models.map((m) => (
              <span key={m} className="flex min-w-0 items-center gap-1.5">
                <span
                  className="size-2 shrink-0 rounded-full"
                  style={{ background: color(m) }}
                  aria-hidden={true}
                />
                <span className="truncate">{m}</span>
              </span>
            ))}
          </span>
        </div>

        {groups.map(({ task, bars }) => (
          <div key={task} className="flex flex-col gap-1.5">
            <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
              {task}
            </span>
            {bars.map((b) => (
              <div
                key={b.model}
                className="grid grid-cols-[minmax(0,calc(180px*var(--ui-space-scale,1)))_minmax(0,1fr)_calc(64px*var(--ui-space-scale,1))] items-center gap-3"
                title={`${b.run.model}\n${b.run.n_samples} samples${
                  b.run.num_fewshot ? ` · ${b.run.num_fewshot}-shot` : ""
                } · ${new Date(b.run.created_at).toLocaleString()}`}
              >
                <span className="truncate text-right text-ui-12 text-foreground/85">
                  {b.model}
                </span>
                <span className="relative h-3 rounded-full bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)]">
                  {[25, 50, 75].map((x) => (
                    <span
                      key={x}
                      className="absolute inset-y-0 w-px bg-background/60"
                      style={{ left: `${x}%` }}
                      aria-hidden={true}
                    />
                  ))}
                  <span
                    className="absolute inset-y-0 left-0 rounded-full transition-[width] duration-500 ease-out"
                    style={{
                      width: `${b.score * 100}%`,
                      background: color(b.model),
                    }}
                  />
                  {b.stderr != null && b.stderr > 0 && (
                    <span
                      className="absolute top-1/2 h-px -translate-y-1/2 opacity-70"
                      style={{
                        background: "var(--foreground)",
                        left: `${Math.max(0, b.score - b.stderr) * 100}%`,
                        width: `${(Math.min(1, b.score + b.stderr) - Math.max(0, b.score - b.stderr)) * 100}%`,
                      }}
                      aria-hidden={true}
                    />
                  )}
                </span>
                <span className="text-ui-12 font-semibold tabular-nums text-foreground">
                  {pct(b.score)}
                </span>
              </div>
            ))}
          </div>
        ))}
      </section>
    </div>
  );
}
