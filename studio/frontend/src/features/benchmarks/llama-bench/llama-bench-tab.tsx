// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// llama-bench: llama.cpp's own prompt-processing and generation benchmark on the picked
// GGUF, drawn with Config sweeps' setup column, tiles and live card.

import { SectionCard } from "@/components/section-card";
import { SegmentedTabsList } from "@/components/segmented-tabs";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs } from "@/components/ui/tabs";
import { useLocale } from "@/i18n";
import { copyToClipboard } from "@/lib/copy-to-clipboard";
import { cn } from "@/lib/utils";
import {
  ArrowDown01Icon,
  Copy01Icon,
  Delete02Icon,
  Rocket01Icon,
  SpeedTrain01Icon,
  StopIcon,
} from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { toast } from "sonner";
import {
  type ReactElement,
  type ReactNode,
  useEffect,
  useMemo,
  useState,
} from "react";
import { BENCH_CARD, RUN_BUTTON } from "../components/bench-ui";
import { CountInput, Field } from "../components/setup-panel";
import { ago } from "../lib/ago";
import { modelShort } from "../lib/bench-math";
import type {
  LlamaBenchConfig,
  LlamaBenchMeta,
  LlamaBenchRow,
} from "./llama-bench-api";
import { toLlamaBenchMarkdown } from "./llama-bench-markdown";
import { useLlamaBenchStore } from "./llama-bench-store";

const PROMPT_SIZES = [128, 512, 1024, 2048, 4096, 8192];
const GEN_SIZES = [32, 64, 128, 256, 512];
const DEPTHS = [0, 4096, 16384, 32768];
const FA_OPTIONS = [
  { value: "auto", label: "Auto" },
  { value: "on", label: "On" },
  { value: "off", label: "Off" },
] as const;

const tokens = (n: number) => (n >= 1024 ? `${n / 1024}K` : String(n));
const rate = (n: number) =>
  n >= 100 ? n.toFixed(0) : n >= 10 ? n.toFixed(1) : n.toFixed(2);

/** Toggle chips for a set of sizes; at least one stays on unless `allowNone`. */
function SizeChips({
  options,
  value,
  onChange,
  allowNone = false,
  format = tokens,
}: {
  options: number[];
  value: number[];
  onChange: (next: number[]) => void;
  allowNone?: boolean;
  format?: (n: number) => string;
}): ReactElement {
  return (
    <div className="flex flex-wrap gap-1.5">
      {options.map((n) => {
        const on = value.includes(n);
        return (
          <button
            key={n}
            type="button"
            aria-pressed={on}
            onClick={() => {
              const next = on ? value.filter((x) => x !== n) : [...value, n];
              if (next.length === 0 && !allowNone) return;
              onChange(next.sort((a, b) => a - b));
            }}
            className={cn(
              "h-7 rounded-full px-2.5 text-ui-12 font-medium tabular-nums transition-colors",
              on
                ? "bg-foreground text-background"
                : "bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)] text-muted-foreground hover:text-foreground",
            )}
          >
            {format(n)}
          </button>
        );
      })}
    </div>
  );
}

function testCount(c: LlamaBenchConfig): number {
  return (c.prompt_tokens.length + c.gen_tokens.length) * c.depths.length;
}

function LlamaBenchSetup({
  model,
  variant,
  loaded,
}: {
  model: string | null;
  variant: string | null;
  loaded: string | null;
}): ReactElement {
  const config = useLlamaBenchStore((s) => s.config);
  const setConfig = useLlamaBenchStore((s) => s.setConfig);
  const available = useLlamaBenchStore((s) => s.available);
  const phase = useLlamaBenchStore((s) => s.phase);
  const start = useLlamaBenchStore((s) => s.start);
  const busy = phase !== "idle";
  const count = testCount(config);

  return (
    <fieldset
      disabled={busy}
      className={cn(
        BENCH_CARD,
        "flex min-w-0 flex-col gap-6 px-5 pb-5 pt-4 disabled:opacity-60",
      )}
    >
      <span className="text-ui-11 font-medium tracking-nav text-muted-foreground">
        Setup
      </span>
      <Field
        label="Prompt sizes"
        hint="Prompt processing (pp): how fast the model reads a prompt this many tokens long."
      >
        <SizeChips
          options={PROMPT_SIZES}
          value={config.prompt_tokens}
          allowNone={config.gen_tokens.length > 0}
          onChange={(prompt_tokens) => setConfig({ prompt_tokens })}
        />
      </Field>
      <Field
        label="Generation sizes"
        hint="Text generation (tg): how fast the model writes this many tokens."
      >
        <SizeChips
          options={GEN_SIZES}
          value={config.gen_tokens}
          allowNone={config.prompt_tokens.length > 0}
          onChange={(gen_tokens) => setConfig({ gen_tokens })}
        />
      </Field>
      <Field
        label="Context depth"
        hint="Runs each test with this many tokens already in the KV cache, to see how speed falls off as a chat gets long."
      >
        <SizeChips
          options={DEPTHS}
          value={config.depths}
          onChange={(depths) => setConfig({ depths })}
          format={(n) => (n === 0 ? "Empty" : tokens(n))}
        />
      </Field>
      <div className="grid grid-cols-2 gap-2">
        <Field label="Repetitions" hint="Runs per test, averaged.">
          <CountInput
            value={config.repetitions}
            min={1}
            max={20}
            step={1}
            onCommit={(repetitions) => setConfig({ repetitions })}
          />
        </Field>
        <Field label="Flash attn">
          <Tabs
            value={config.flash_attn}
            onValueChange={(v) =>
              setConfig({ flash_attn: v as LlamaBenchConfig["flash_attn"] })
            }
            className="contents"
          >
            <SegmentedTabsList
              value={config.flash_attn}
              options={FA_OPTIONS}
              ariaLabel="Flash attention"
              size="compact"
              className="w-full"
            />
          </Tabs>
        </Field>
      </div>

      {available === false && (
        <p className="rounded-xl bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)] px-3 py-2.5 text-ui-12 leading-relaxed text-muted-foreground">
          This llama.cpp install doesn't include llama-bench. It comes with the
          next llama.cpp update: when Settings offers one, update and it shows
          up here.
        </p>
      )}
      <div className="flex flex-col gap-2">
        <Button
          size="lg"
          className={RUN_BUTTON}
          disabled={
            busy || available === false || !(model ?? loaded) || count === 0
          }
          onClick={() => void start(model, variant)}
        >
          <HugeiconsIcon
            icon={Rocket01Icon}
            strokeWidth={1.75}
            className="size-4"
          />
          {(model ?? loaded)
            ? `Run ${count} test${count === 1 ? "" : "s"}`
            : "Pick a model"}
        </Button>
        <p className="text-center text-ui-11 leading-relaxed text-muted-foreground">
          Chat's model is unloaded during the run and loaded back after.
        </p>
      </div>
    </fieldset>
  );
}

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

/** pp and tg differ by an order of magnitude, so each gets its own scale. */
function RateBars({
  title,
  rows,
}: {
  title: string;
  rows: LlamaBenchRow[];
}): ReactElement | null {
  if (rows.length === 0) return null;
  const max = Math.max(...rows.map((r) => r.avg_ts + (r.stddev_ts || 0)));
  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-ui-11 font-medium uppercase tracking-[0.05em] text-muted-foreground/70">
        {title}
      </span>
      {rows.map((r) => (
        <div
          key={r.test}
          className="grid grid-cols-[calc(96px*var(--ui-space-scale,1))_minmax(0,1fr)_calc(112px*var(--ui-space-scale,1))] items-center gap-3"
          title={`${r.test}: ${r.samples_ts.map(rate).join(", ")} tok/s`}
        >
          <span className="truncate text-right font-mono text-ui-12 text-foreground/85">
            {r.test}
          </span>
          <span className="relative h-3 rounded-full bg-[color-mix(in_oklab,var(--foreground)_calc(6%*var(--contrast-wash-gain,1)),transparent)]">
            <span
              className="absolute inset-y-0 left-0 rounded-full bg-primary transition-[width] duration-500 ease-out"
              style={{ width: `${(r.avg_ts / max) * 100}%` }}
            />
          </span>
          <span className="text-ui-12 tabular-nums text-muted-foreground">
            <span className="font-semibold text-foreground">
              {rate(r.avg_ts)}
            </span>{" "}
            ± {rate(r.stddev_ts || 0)} t/s
          </span>
        </div>
      ))}
    </div>
  );
}

function gbOf(bytes: number | null | undefined): string | null {
  return bytes ? `${(bytes / 1024 ** 3).toFixed(1)} GB` : null;
}

function Results({
  rows,
  meta,
  model,
  variant,
}: {
  rows: LlamaBenchRow[];
  meta: LlamaBenchMeta;
  model: string;
  variant: string | null;
}): ReactElement {
  const pp = rows.filter((r) => r.n_prompt > 0);
  const tg = rows.filter((r) => r.n_gen > 0);
  const best = (rs: LlamaBenchRow[]) =>
    rs.reduce<LlamaBenchRow | null>(
      (a, r) => (!a || r.avg_ts > a.avg_ts ? r : a),
      null,
    );
  const bestPp = best(pp);
  const bestTg = best(tg);
  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Tile
          label="Prompt processing"
          value={bestPp ? `${rate(bestPp.avg_ts)} t/s` : "—"}
          detail={bestPp ? `best, at ${bestPp.test}` : undefined}
        />
        <Tile
          label="Generation"
          value={bestTg ? `${rate(bestTg.avg_ts)} t/s` : "—"}
          detail={bestTg ? `best, at ${bestTg.test}` : undefined}
        />
        <Tile
          label="Model"
          value={variant ?? meta.model_type ?? modelShort(model)}
          detail={
            [modelShort(model), gbOf(meta.model_size)]
              .filter(Boolean)
              .join(" · ") || undefined
          }
        />
      </div>
      <section className={cn(BENCH_CARD, "flex flex-col gap-5 p-4 sm:p-5")}>
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
          <span className="text-ui-13 font-medium text-foreground">
            Throughput
          </span>
          <span className="min-w-0 truncate text-ui-11 text-muted-foreground">
            {[
              meta.backends,
              meta.gpu_info,
              meta.build_number ? `llama.cpp b${meta.build_number}` : null,
            ]
              .filter(Boolean)
              .join(" · ")}
          </span>
          <Button
            variant="ghost"
            size="sm"
            className="ml-auto h-7 rounded-full px-2.5 text-ui-12"
            onClick={async () => {
              if (await copyToClipboard(toLlamaBenchMarkdown(rows, meta)))
                toast.success("Copied as llama-bench's markdown table");
            }}
          >
            <HugeiconsIcon
              icon={Copy01Icon}
              strokeWidth={1.75}
              className="size-3.5"
            />
            Copy as markdown
          </Button>
        </div>
        <RateBars title="Prompt processing" rows={pp} />
        <RateBars title="Generation" rows={tg} />
      </section>
    </div>
  );
}

function LiveCard(): ReactElement | null {
  const job = useLlamaBenchStore((s) => s.job);
  const phase = useLlamaBenchStore((s) => s.phase);
  const cancel = useLlamaBenchStore((s) => s.cancel);
  const [showLog, setShowLog] = useState(false);
  const pct = job && job.total > 0 ? (job.done / job.total) * 100 : 0;
  const stage =
    phase === "loading"
      ? "Loading the model"
      : phase === "restoring"
        ? "Loading chat's model back"
        : job?.stage || "Starting llama-bench";
  return (
    <SectionCard
      icon={<HugeiconsIcon icon={SpeedTrain01Icon} className="size-5" />}
      title="llama-bench"
      description={stage}
      className="shadow-border border border-border/60 bg-card/90 ring-0"
      headerAction={
        phase === "running" ? (
          <Button
            variant="destructive"
            size="sm"
            onClick={() => void cancel()}
            className="h-8 rounded-full px-3.5 text-xs shadow-sm"
          >
            <HugeiconsIcon icon={StopIcon} className="size-3" />
            Stop
          </Button>
        ) : undefined
      }
    >
      <div className="flex flex-col gap-2">
        {job && phase === "running" && (
          <>
            <div className="flex justify-between text-xs tabular-nums text-muted-foreground">
              <span>
                {job.done} of {job.total} tests
              </span>
              <span>{Math.round(pct)}%</span>
            </div>
            <Progress
              value={pct}
              className="h-2 bg-[color-mix(in_oklab,var(--foreground)_calc(5%*var(--contrast-wash-gain,1)),transparent)]"
            />
          </>
        )}
        {job && job.log.length > 0 && (
          <>
            <button
              type="button"
              onClick={() => setShowLog((v) => !v)}
              className="mr-auto flex items-center gap-1.5 rounded-full px-2 py-0.5 text-ui-11 text-muted-foreground transition-colors hover:bg-muted/60 hover:text-foreground"
            >
              <HugeiconsIcon
                icon={ArrowDown01Icon}
                strokeWidth={1.75}
                className={cn(
                  "size-3.5 transition-transform",
                  showLog && "rotate-180",
                )}
              />
              {showLog ? "Hide log" : "Show log"}
            </button>
            {showLog && (
              <div className="max-h-48 overflow-y-auto rounded-xl bg-black/80 p-3 font-mono text-xs leading-relaxed text-white/80">
                {job.log.map((line, i) => (
                  <div key={i}>{line}</div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </SectionCard>
  );
}

function SavedRuns(): ReactElement | null {
  const runs = useLlamaBenchStore((s) => s.runs);
  const shownId = useLlamaBenchStore((s) => s.shownId);
  const show = useLlamaBenchStore((s) => s.show);
  const remove = useLlamaBenchStore((s) => s.remove);
  const locale = useLocale();
  if (runs.length < 2) return null;
  const current = shownId ?? runs[0]?.id;
  return (
    <section className={cn(BENCH_CARD, "flex flex-col gap-1 p-3 sm:p-4")}>
      <span className="px-2 pb-1 text-ui-13 font-medium text-foreground">
        Past runs
      </span>
      {runs.map((r) => {
        const tg = r.outcomes.find((x) => x.n_gen > 0);
        const pp = r.outcomes.find((x) => x.n_prompt > 0);
        return (
          <div
            key={r.id}
            className={cn(
              "group flex items-center gap-3 rounded-xl px-2 py-1.5 text-ui-12",
              r.id === current ? "bg-muted/60" : "hover:bg-muted/40",
            )}
          >
            <button
              type="button"
              onClick={() => show(r.id)}
              className="flex min-w-0 flex-1 items-center gap-3 text-left"
            >
              <span className="min-w-0 flex-1 truncate text-foreground">
                {modelShort(r.model)}
                {r.ggufVariant ? ` · ${r.ggufVariant}` : ""}
              </span>
              <span className="shrink-0 tabular-nums text-muted-foreground">
                {[
                  pp ? `${pp.test} ${rate(pp.avg_ts)}` : null,
                  tg ? `${tg.test} ${rate(tg.avg_ts)}` : null,
                ]
                  .filter(Boolean)
                  .join(" · ")}
              </span>
              <span className="w-20 shrink-0 text-right text-muted-foreground">
                {ago(r.createdAt, locale)}
              </span>
            </button>
            <button
              type="button"
              aria-label="Delete run"
              onClick={() => void remove(r.id)}
              className="shrink-0 text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
            >
              <HugeiconsIcon icon={Delete02Icon} className="size-3.5" />
            </button>
          </div>
        );
      })}
    </section>
  );
}

/** The llama-bench tab. `model`/`variant` are the header pick, or null for chat's model;
 * `loaded` is chat's model, so a null pick can still run. */
export function LlamaBenchTab({
  model,
  variant,
  loaded,
}: {
  model: string | null;
  variant: string | null;
  loaded: string | null;
}): ReactElement {
  const job = useLlamaBenchStore((s) => s.job);
  const phase = useLlamaBenchStore((s) => s.phase);
  const error = useLlamaBenchStore((s) => s.error);
  const runs = useLlamaBenchStore((s) => s.runs);
  const shownId = useLlamaBenchStore((s) => s.shownId);
  const refresh = useLlamaBenchStore((s) => s.refresh);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  // A run in flight shows its rows as they land; otherwise the picked saved run, or the newest.
  const shown = useMemo(() => {
    if (job && phase !== "idle")
      return job.rows.length
        ? {
            rows: job.rows,
            meta: job.meta,
            model: job.model,
            variant: job.ggufVariant,
          }
        : null;
    const run = runs.find((r) => r.id === shownId) ?? runs[0];
    return run
      ? {
          rows: run.outcomes,
          meta: run.meta,
          model: run.model,
          variant: run.ggufVariant,
        }
      : null;
  }, [job, phase, runs, shownId]);

  return (
    <div className="@container/llamabench">
      <div className="grid grid-cols-1 items-start gap-6 @3xl/llamabench:grid-cols-[calc(264px*var(--ui-space-scale,1))_minmax(0,1fr)]">
        <div className="@3xl/llamabench:sticky @3xl/llamabench:top-6">
          <LlamaBenchSetup model={model} variant={variant} loaded={loaded} />
        </div>
        <div className="flex min-w-0 flex-col gap-4">
          {error && (
            <p
              role="alert"
              className="rounded-xl bg-destructive/10 px-4 py-2.5 text-ui-12p5 text-destructive"
            >
              {error}
            </p>
          )}
          {phase !== "idle" && <LiveCard />}
          {shown ? (
            <Results {...shown} />
          ) : (
            phase === "idle" && (
              <div className="corner-squircle flex flex-col items-center justify-center gap-1 rounded-3xl bg-card px-6 py-16 text-center ring-1 ring-border/60">
                <span className="text-ui-13p5 font-medium text-foreground">
                  No llama-bench runs yet
                </span>
                <span className="text-ui-12 text-muted-foreground">
                  Pick sizes and press Run. llama.cpp's own benchmark measures
                  how fast this machine reads prompts and writes tokens.
                </span>
              </div>
            )
          )}
          <SavedRuns />
        </div>
      </div>
    </div>
  );
}
