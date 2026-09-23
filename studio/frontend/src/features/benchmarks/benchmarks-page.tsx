// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Benchmarks: sweep load settings on the loaded GGUF and chart what each one does to
// throughput, draft acceptance and load time on this machine.

import { Button } from "@/components/ui/button";
import { type InferenceStatusResponse, getInferenceStatus } from "@/features/chat";
import { cn } from "@/lib/utils";
import { Delete02Icon, PlayIcon, StopIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { Link } from "@tanstack/react-router";
import { type ReactElement, useEffect, useState } from "react";
import { RunResults, LiveProgress } from "./components/results-panel";
import { SetupPanel } from "./components/setup-panel";
import { SWEEP_TITLE, fmtTokens, modelShort } from "./lib/bench-math";
import { useBenchmarksStore } from "./stores/benchmarks-store";

function useLoadedModel(paused: boolean): { status: InferenceStatusResponse | null; checked: boolean } {
  const [status, setStatus] = useState<InferenceStatusResponse | null>(null);
  const [checked, setChecked] = useState(false);
  useEffect(() => {
    if (paused) return;
    const controller = new AbortController();
    const read = () =>
      getInferenceStatus(controller.signal)
        .then((s) => setStatus(s))
        .catch(() => undefined)
        .finally(() => setChecked(true));
    void read();
    const timer = window.setInterval(read, 5000);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, [paused]);
  return { status, checked };
}

function Fact({ label, value }: { label: string; value: string }): ReactElement {
  return (
    <div className="flex min-w-0 flex-col">
      <span className="text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">{label}</span>
      <span className="truncate text-ui-13 tabular-nums text-foreground">{value}</span>
    </div>
  );
}

function ModelStrip({ status }: { status: InferenceStatusResponse }): ReactElement {
  return (
    <section className="flex flex-wrap items-center gap-x-8 gap-y-3 rounded-2xl border border-border/60 bg-card px-5 py-3.5">
      <div className="flex min-w-0 items-center gap-2.5">
        <span className="size-2 shrink-0 rounded-full bg-emerald-500" aria-hidden={true} />
        <div className="flex min-w-0 flex-col">
          <span className="text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">Benchmarking</span>
          <span className="truncate text-ui-15 font-medium text-foreground">
            {modelShort(status.active_model)}
            {status.gguf_variant && <span className="ml-2 text-ui-12 font-normal text-muted-foreground">{status.gguf_variant}</span>}
          </span>
        </div>
      </div>
      {status.cache_type_kv && <Fact label="KV cache" value={status.cache_type_kv} />}
      {status.context_length ? <Fact label="Context" value={fmtTokens(status.context_length)} /> : null}
      <Fact label="Speculative" value={status.speculative_type ?? "auto"} />
      {status.spec_draft_n_max ? <Fact label="Draft tokens" value={String(status.spec_draft_n_max)} /> : null}
    </section>
  );
}

function EmptyModel({ reason }: { reason: "none" | "not-gguf" }): ReactElement {
  return (
    <section className="flex flex-col items-center gap-3 rounded-2xl border border-dashed border-border/80 px-6 py-14 text-center">
      <span className="text-ui-16 font-medium text-foreground">
        {reason === "none" ? "Load a GGUF model to benchmark it" : "Benchmarks run on GGUF models"}
      </span>
      <p className="max-w-md text-ui-13 leading-relaxed text-muted-foreground">
        {reason === "none"
          ? "A sweep reloads the model you have loaded with each setting in turn, so pick one in chat first, then come back."
          : "The loaded model runs on another backend. Load a GGUF build of it in chat to compare speculative decoding, KV cache and the rest."}
      </p>
      <Button asChild variant="outline" size="sm" className="h-9 rounded-full">
        <Link to="/chat">Go to chat</Link>
      </Button>
    </section>
  );
}

function History(): ReactElement | null {
  const history = useBenchmarksStore((s) => s.history);
  const selectedRunId = useBenchmarksStore((s) => s.selectedRunId);
  const selectRun = useBenchmarksStore((s) => s.selectRun);
  const deleteRun = useBenchmarksStore((s) => s.deleteRun);
  if (history.length === 0) return null;
  const current = selectedRunId ?? history[0].id;
  return (
    <section className="flex flex-col gap-2">
      <h2 className="text-ui-10 font-medium uppercase tracking-wider text-muted-foreground">Past runs</h2>
      <div className="flex gap-2 overflow-x-auto pb-1">
        {history.map((run) => (
          <div
            key={run.id}
            className={cn(
              "group flex shrink-0 items-center gap-1 rounded-xl border pl-3 pr-1 transition-colors",
              run.id === current ? "border-primary/40 bg-primary/6" : "border-border/60 hover:bg-muted/40",
            )}
          >
            <button type="button" onClick={() => selectRun(run.id)} className="flex flex-col items-start py-2 text-left">
              <span className="text-ui-12 font-medium text-foreground">{SWEEP_TITLE[run.config.sweep]}</span>
              <span className="text-ui-11 text-muted-foreground">
                {modelShort(run.model)} ·{" "}
                {new Date(run.createdAt).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" })}
              </span>
            </button>
            <button
              type="button"
              onClick={() => deleteRun(run.id)}
              className="rounded-md p-1 text-muted-foreground/0 transition-colors group-hover:text-muted-foreground hover:bg-muted hover:text-foreground"
              aria-label="Delete this run"
            >
              <HugeiconsIcon icon={Delete02Icon} strokeWidth={1.75} className="size-3.5" />
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}

export function BenchmarksPage(): ReactElement {
  const live = useBenchmarksStore((s) => s.live);
  const error = useBenchmarksStore((s) => s.error);
  const history = useBenchmarksStore((s) => s.history);
  const selectedRunId = useBenchmarksStore((s) => s.selectedRunId);
  const start = useBenchmarksStore((s) => s.start);
  const cancel = useBenchmarksStore((s) => s.cancel);
  const { status, checked } = useLoadedModel(Boolean(live));
  const shown = live ? null : (history.find((r) => r.id === selectedRunId) ?? history[0] ?? null);
  const ready = Boolean(status?.active_model) && status?.is_gguf !== false;
  const maxContext = status?.max_context_length ?? status?.native_context_length ?? null;

  return (
    <main className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-6 pb-12 pt-12 font-heading sm:px-10">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="flex min-w-0 flex-col gap-1">
          <h1 className="text-ui-30 font-semibold leading-[1.04] tracking-[-0.028em] text-foreground sm:text-ui-34">Benchmarks</h1>
          <p className="text-sm text-muted-foreground">Find the fastest settings for the model you have loaded, on this machine.</p>
        </div>
        {live ? (
          <Button variant="outline" onClick={cancel} className="h-10 gap-2 rounded-full px-5">
            <HugeiconsIcon icon={StopIcon} strokeWidth={1.75} className="size-4" />
            Stop
          </Button>
        ) : (
          <Button onClick={() => void start()} disabled={!ready} className="h-10 gap-2 rounded-full px-5">
            <HugeiconsIcon icon={PlayIcon} strokeWidth={1.75} className="size-4" />
            Run benchmark
          </Button>
        )}
      </header>

      {status && ready && !live && <ModelStrip status={status} />}
      {error && (
        <div className="rounded-xl border border-destructive/30 bg-destructive/6 px-4 py-3 text-ui-13 text-destructive" role="alert">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-[360px_minmax(0,1fr)]">
        <SetupPanel maxContext={maxContext} locked={Boolean(live)} />
        <div className="flex min-w-0 flex-col gap-6">
          {live && <LiveProgress run={live.run} progress={live.progress} onStop={cancel} />}
          {live && live.run.results.some((r) => !r.warmup) && <RunResults run={live.run} />}
          {!live && checked && !ready && !shown && <EmptyModel reason={status?.active_model ? "not-gguf" : "none"} />}
          {!live && shown && <RunResults run={shown} />}
          {!live && ready && !shown && (
            <section className="flex flex-col items-center gap-2 rounded-2xl border border-dashed border-border/80 px-6 py-16 text-center">
              <span className="text-ui-16 font-medium text-foreground">No runs yet</span>
              <p className="max-w-md text-ui-13 leading-relaxed text-muted-foreground">
                Pick a sweep, then press Run benchmark. Each setting reloads the model, so a full speculative decoding sweep takes a few
                minutes.
              </p>
            </section>
          )}
          {!live && <History />}
        </div>
      </div>
    </main>
  );
}
