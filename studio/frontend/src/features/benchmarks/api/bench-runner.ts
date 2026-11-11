// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import { authFetch } from "@/features/auth";
import {
  type InferenceStatusResponse,
  getInferenceStatus,
  loadModel,
  useChatRuntimeStore,
} from "@/features/chat";
import { gpuMemoryDisplay } from "@/hooks/gpu-memory-display";
import { resolveGpuVramUsedGb } from "@/hooks/gpu-vram";
import type { SystemInfoResponse } from "@/hooks/use-system";
import {
  type ChatSampling,
  chatBaseLoad,
  chatSampling,
  chatSettings,
} from "./chat-base";
import {
  type BenchConfig,
  type BenchRun,
  type RunMeta,
  type RunResult,
  type VariantOutcome,
  offloadDemand,
  promptsFor,
  servedMismatch,
  variantLoad,
} from "../lib/bench-math";

export interface RunnerEvents {
  onStart: (run: BenchRun) => void;
  onOutcome: (outcome: VariantOutcome) => void;
  onResult: (result: RunResult) => void;
  onProgress: (text: string) => void;
  onRestoreError?: (message: string) => void;
  ownsServer?: () => boolean;
  leaseHolder?: string;
}

export class BenchSetupError extends Error {}

function leaseHeaders(holder?: string): Record<string, string> {
  return holder ? { "X-Unsloth-Bench-Lease": holder } : {};
}

function servedSubset(st: InferenceStatusResponse): Record<string, unknown> {
  const keys = [
    "speculative_type",
    "spec_draft_n_max",
    "spec_drafter_kind",
    "spec_fallback_reason",
    "cache_type_kv",
    "context_length",
    "parallel_slots",
    "gguf_variant",
    "n_cpu_moe",
  ] as const;
  const out: Record<string, unknown> = {};
  for (const k of keys) {
    const v = (st as unknown as Record<string, unknown>)[k];
    if (v !== null && v !== undefined) out[k] = v;
  }
  return out;
}

async function getJson(
  path: string,
  signal: AbortSignal,
): Promise<Record<string, unknown> | null> {
  try {
    const res = await authFetch(path, { signal });
    return res.ok ? ((await res.json()) as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

async function readMeta(signal: AbortSignal): Promise<RunMeta> {
  const [llama, hw, health, sys] = await Promise.all([
    getJson("/api/llama/backend", signal),
    getJson("/api/system/hardware", signal),
    getJson("/api/health", signal),
    getJson("/api/system", signal),
  ]);
  const gpu = (hw?.gpu ?? null) as Record<string, unknown> | null;
  const versions = (hw?.versions ?? null) as Record<string, unknown> | null;
  const cpu = (sys?.cpu ?? null) as Record<string, unknown> | null;
  const memory = (sys?.memory ?? null) as Record<string, unknown> | null;
  const str = (v: unknown) => (typeof v === "string" && v ? v : null);
  const num = (v: unknown) =>
    typeof v === "number" && Number.isFinite(v) ? v : null;
  const runtime = str(versions?.rocm)
    ? `ROCm ${str(versions?.rocm)}`
    : str(versions?.cuda)
      ? `CUDA ${str(versions?.cuda)}`
      : null;
  return {
    gpu: str(gpu?.gpu_name),
    vramGb: num(gpu?.vram_total_gb),
    backend: str(llama?.backend),
    runtime,
    llamaTag: str(llama?.installed_tag),
    studioVersion: str(health?.studio_version) ?? str(health?.version),
    os: str(sys?.platform),
    cpuThreads: num(cpu?.logical_count) ?? num(sys?.cpu_count),
    ramGb: num(memory?.total_gb),
  };
}

interface Completion {
  ttftMs: number | null;
  wallMs: number;
  clientTokens: number;
  timings: Record<string, unknown>;
}

class RowLimitError extends Error {}

const LOAD_LIMIT_MS = 8 * 60_000;
const MIN_GEN_LIMIT_MS = 90_000;
const FLOOR_TPS = 2;
const FLOOR_FRACTION = 0.2;

async function readVramUsedGb(signal: AbortSignal): Promise<number | null> {
  try {
    const res = await authFetch("/api/system?refresh_memory=true", { signal });
    if (!res.ok) return null;
    const sys = (await res.json()) as SystemInfoResponse;
    const gpu = sys.inference_gpu?.available ? sys.inference_gpu : sys.gpu;
    const used = resolveGpuVramUsedGb(gpuMemoryDisplay(gpu).usageGpu);
    return used === null ? null : Math.round(used * 10) / 10;
  } catch {
    return null;
  }
}

function genLimitMs(maxTokens: number): number {
  return Math.max(MIN_GEN_LIMIT_MS, (maxTokens / 4) * 1000 + 30_000);
}

async function withLimit<T>(
  ms: number,
  signal: AbortSignal,
  what: string,
  fn: (s: AbortSignal) => Promise<T>,
): Promise<T> {
  const ctl = new AbortController();
  const onAbort = () => ctl.abort(signal.reason);
  signal.addEventListener("abort", onAbort, { once: true });
  let timedOut = false;
  const timer = window.setTimeout(() => {
    timedOut = true;
    ctl.abort(new DOMException(what, "TimeoutError"));
  }, ms);
  try {
    return await fn(ctl.signal);
  } catch (err) {
    if (timedOut && !signal.aborted)
      throw new RowLimitError(
        `${what} took over ${Math.round(ms / 60_000) || 1} min, so it was stopped`,
      );
    throw err;
  } finally {
    window.clearTimeout(timer);
    signal.removeEventListener("abort", onAbort);
  }
}

function looksOutOfMemory(message: string): boolean {
  return /out of memory|OOM|failed to allocate|cudaMalloc|hipMalloc|ErrorOutOfDeviceMemory|not enough (video )?memory/i.test(
    message,
  );
}

async function streamOnce(
  model: string,
  prompt: string,
  config: BenchConfig,
  sampling: ChatSampling,
  signal: AbortSignal,
  leaseHolder?: string,
): Promise<Completion> {
  const started = performance.now();
  const res = await authFetch("/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...leaseHeaders(leaseHolder),
    },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
      stream: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      stream_options: { include_usage: true },
      // biome-ignore lint/style/useNamingConvention: API schema
      max_tokens: config.maxTokens,
      ...sampling,
      seed: config.seed,
    }),
    signal,
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new Error(
      `Chat completion answered ${res.status}${text ? `: ${text.slice(0, 300)}` : ""}`,
    );
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let ttftMs: number | null = null;
  let clientTokens = 0;
  let usageTokens: number | null = null;
  let timings: Record<string, unknown> = {};
  let finished = false;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut = buffer.search(/\r?\n\r?\n/);
    while (cut >= 0) {
      const event = buffer.slice(0, cut);
      buffer = buffer.slice(cut + (buffer[cut] === "\r" ? 4 : 2));
      cut = buffer.search(/\r?\n\r?\n/);
      const data = event
        .split(/\r?\n/)
        .filter((l) => l.startsWith("data:"))
        .map((l) => l.slice(5).trim())
        .join("\n");
      if (data === "[DONE]") finished = true;
      if (!data || data === "[DONE]") continue;
      let chunk: Record<string, unknown>;
      try {
        chunk = JSON.parse(data) as Record<string, unknown>;
      } catch {
        continue;
      }
      const err = chunk.error as { message?: string } | undefined;
      if (err) throw new Error(err.message ?? "Stream error");
      if (chunk.timings && typeof chunk.timings === "object")
        timings = chunk.timings as Record<string, unknown>;
      const usage = chunk.usage as { completion_tokens?: unknown } | undefined;
      if (usage && typeof usage.completion_tokens === "number")
        usageTokens = usage.completion_tokens;
      const choices = Array.isArray(chunk.choices)
        ? (chunk.choices as Record<string, unknown>[])
        : [];
      for (const choice of choices) {
        if (choice.finish_reason) finished = true;
        const delta = (choice.delta ?? {}) as Record<string, unknown>;
        const piece = [
          delta.content,
          delta.reasoning_content,
          delta.reasoning,
        ].find((x) => typeof x === "string" && x.length > 0);
        if (piece) {
          clientTokens += 1;
          if (ttftMs === null) ttftMs = performance.now() - started;
        }
      }
    }
  }
  if (!finished) throw new Error("The stream closed before the reply finished");
  return {
    ttftMs,
    wallMs: performance.now() - started,
    clientTokens: usageTokens ?? clientTokens,
    timings,
  };
}

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function abortError(): DOMException {
  return new DOMException("Benchmark cancelled", "AbortError");
}

/**
 * Run every variant in order. Resolves with the finished run; a cancelled run resolves
 * too, with the rows it reached, so partial results are never thrown away.
 */
export async function runBenchmark(
  config: BenchConfig,
  events: RunnerEvents,
  signal: AbortSignal,
): Promise<BenchRun> {
  await useChatRuntimeStore.getState().hydratePersistedSettings();
  let status = await getInferenceStatus(signal);
  const original = status;
  const originalLoad = chatBaseLoad(original);
  const prompts = promptsFor(config.promptSet, config.customPrompt);
  if (prompts.length === 0)
    throw new BenchSetupError("The custom prompt is empty.");

  let run: BenchRun | null = null;
  let swapped = false;
  const setOutcome = (o: VariantOutcome) => {
    if (!run) return;
    run.outcomes = run.outcomes.map((x) => (x.label === o.label ? o : x));
    events.onOutcome(o);
  };

  let fastest = 0;
  let failedContext: number | null = null;
  let failedDemand: number | null = null;
  const relativeFloor = config.sweep !== "offload";
  const contextSweep = config.sweep === "context";
  try {
    if (
      config.tuneModel &&
      (config.tuneModel !== status.active_model ||
        (config.tuneVariant ?? null) !== (status.gguf_variant ?? null))
    ) {
      events.onProgress(`Loading ${config.tuneModel}`);
      swapped = true;
      await loadModel(
        {
          ...chatBaseLoad({
            ...status,
            active_model: config.tuneModel,
            gguf_variant: config.tuneVariant ?? null,
          }),
          gguf_variant: config.tuneVariant ?? null,
          force_reload: true,
        },
        { signal, runtime: "chat", headers: leaseHeaders(events.leaseHolder) },
      );
      status = await getInferenceStatus(signal);
    }
    if (!status.active_model)
      throw new BenchSetupError(
        "Load a GGUF model in chat first. Benchmarks measure the model that is loaded.",
      );
    if (status.is_gguf === false)
      throw new BenchSetupError(
        "Benchmarks run on GGUF models. The loaded model runs on another backend.",
      );

    const base = chatBaseLoad(status);
    const sampling = chatSampling(useChatRuntimeStore.getState());
    config = { ...config, temperature: sampling.temperature };
    run = {
      id: `run-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`,
      createdAt: Date.now(),
      finishedAt: null,
      model: status.active_model,
      ggufVariant: status.gguf_variant ?? null,
      kv: status.cache_type_kv ?? null,
      context: status.context_length ?? null,
      config,
      meta: await readMeta(signal),
      base: chatSettings(useChatRuntimeStore.getState()),
      outcomes: config.variants.map((v) => ({ label: v.label, state: "queued" })),
      results: [],
    };
    events.onStart(run);
    for (const variant of config.variants) {
      if (signal.aborted) throw abortError();
      const ctx = variant.load.max_seq_length;
      if (failedContext !== null && ctx !== undefined && ctx >= failedContext) {
        setOutcome({
          label: variant.label,
          state: "skipped",
          reason: `a smaller context already ran out of memory`,
        });
        continue;
      }
      const demand = offloadDemand(variant.load);
      if (failedDemand !== null && demand !== null && demand >= failedDemand) {
        setOutcome({
          label: variant.label,
          state: "skipped",
          reason: "a lighter placement already ran out of memory",
        });
        continue;
      }
      setOutcome({ label: variant.label, state: "loading" });
      events.onProgress(`Loading ${variant.label}`);
      const loadStarted = performance.now();
      const payload = {
        ...variantLoad(base, variant),
        load_request_id: loadRequestId(),
      };
      let modelId: string;
      try {
        const loaded = await withLimit(LOAD_LIMIT_MS, signal, "Loading", (s) =>
          loadModel(payload, {
            signal: s,
            runtime: "chat",
            headers: leaseHeaders(events.leaseHolder),
          }),
        );
        modelId = loaded.model;
      } catch (err) {
        // The backend keeps loading after the fetch is aborted and holds the load gate,
        // which would stall every later load and the restore.
        if (signal.aborted || err instanceof RowLimitError)
          await cancelLoad(payload, events.leaseHolder);
        if (signal.aborted) throw abortError();
        const message = err instanceof Error ? err.message : String(err);
        const oom = err instanceof RowLimitError || looksOutOfMemory(message);
        if (oom && ctx !== undefined && demand === null && contextSweep)
          failedContext = Math.min(failedContext ?? ctx, ctx);
        if (oom && demand !== null)
          failedDemand = Math.min(failedDemand ?? demand, demand);
        setOutcome({
          label: variant.label,
          state: "error",
          reason: oom ? `out of memory: ${message}` : message,
        });
        continue;
      }
      const loadMs = performance.now() - loadStarted;
      const served = await getInferenceStatus(signal);
      const mismatch = servedMismatch(variant, served);
      if (mismatch) {
        setOutcome({
          label: variant.label,
          state: "skipped",
          reason: mismatch,
          served: servedSubset(served),
        });
        continue;
      }
      const vram = await readVramUsedGb(signal);
      const servedInfo = {
        ...servedSubset(served),
        ...(vram !== null ? { vram_used_gb: vram } : {}),
      };
      setOutcome({
        label: variant.label,
        state: "running",
        served: servedInfo,
      });
      const total = config.warmup + config.repetitions;
      let rowFailure: string | null = null;
      let rowFailureLimited = false;
      // A soft speed-floor stop, not a real OOM: an offload row with experts in RAM runs
      // slow by design, so it must not skip the hungrier placements that would fit and run fast.
      let rowFloorStop = false;
      for (let rep = 0; rep < total; rep++) {
        if (signal.aborted) throw abortError();
        const warmup = rep < config.warmup;
        events.onProgress(
          warmup
            ? `${variant.label} · warm-up ${rep + 1}/${config.warmup}`
            : `${variant.label} · run ${rep + 1 - config.warmup}/${config.repetitions}`,
        );
        const promptIndex = config.rotatePrompts ? rep % prompts.length : 0;
        let c: Completion;
        try {
          c = await withLimit(
            genLimitMs(config.maxTokens),
            signal,
            "A generation",
            (s) =>
              streamOnce(
                modelId,
                prompts[promptIndex],
                config,
                sampling,
                s,
                events.leaseHolder,
              ),
          );
        } catch (err) {
          if (signal.aborted) throw abortError();
          rowFailure = err instanceof Error ? err.message : String(err);
          rowFailureLimited =
            err instanceof RowLimitError || looksOutOfMemory(rowFailure);
          break;
        }
        const t = c.timings;
        const genTokens = num(t.predicted_n);
        const decodeMs = c.ttftMs === null ? null : c.wallMs - c.ttftMs;
        const result: RunResult = {
          variant: variant.label,
          rep,
          warmup,
          promptIndex,
          tps: num(t.predicted_per_second),
          promptTps: num(t.prompt_per_second),
          promptTokens: num(t.prompt_n),
          genTokens,
          ttftMs: c.ttftMs,
          wallMs: c.wallMs,
          clientTps:
            decodeMs && decodeMs > 0
              ? ((genTokens ?? c.clientTokens) / decodeMs) * 1000
              : null,
          draftN: num(t.draft_n),
          draftAccepted: num(t.draft_n_accepted),
          loadMs: rep === 0 ? loadMs : null,
          at: Date.now(),
        };
        run.results.push(result);
        events.onResult(result);
        const rate = result.tps ?? result.clientTps ?? 0;
        if (
          rate > 0 &&
          (rate < FLOOR_TPS ||
            (!warmup &&
              relativeFloor &&
              fastest > 0 &&
              rate < fastest * FLOOR_FRACTION))
        ) {
          rowFailure = `ran at ${rate.toFixed(1)} tok/s, far below the other rows, which means it spilled out of VRAM. Stopped to keep the machine responsive`;
          rowFailureLimited = true;
          rowFloorStop = true;
          break;
        }
        if (!warmup) fastest = Math.max(fastest, rate);
      }
      if (rowFailure) {
        if (rowFailureLimited) {
          if (ctx !== undefined && demand === null && contextSweep)
            failedContext = Math.min(failedContext ?? ctx, ctx);
          // A slow offload row is not out of memory, so it must not skip the GPU-heavier rows.
          if (demand !== null && !rowFloorStop)
            failedDemand = Math.min(failedDemand ?? demand, demand);
        }
        setOutcome({
          label: variant.label,
          state: "error",
          reason: rowFailure,
          served: servedInfo,
        });
        continue;
      }
      setOutcome({
        label: variant.label,
        state: "done",
        served: servedInfo,
      });
    }
  } catch (err) {
    if (!signal.aborted) throw err;
    if (run)
      for (const o of run.outcomes) {
        if (
          o.state === "queued" ||
          o.state === "loading" ||
          o.state === "running"
        )
          setOutcome({ ...o, state: "cancelled" });
      }
  } finally {
    if (run) run.finishedAt = Date.now();
    // Restore whenever chat's model may have changed: a variant ran, or the initial swap did.
    if (
      config.restoreAfter &&
      original.active_model &&
      (run || swapped) &&
      events.ownsServer?.() !== false
    ) {
      events.onProgress("Restoring your original settings");
      await restore(original, originalLoad, events.leaseHolder).catch((err) =>
        events.onRestoreError?.(
          err instanceof Error ? err.message : String(err),
        ),
      );
    }
  }
  if (!run) throw abortError();
  return run;
}

// Not randomUUID: Studio on a LAN address is not a secure context.
function loadRequestId(): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (byte) =>
    byte.toString(16).padStart(2, "0"),
  ).join("");
}

async function cancelLoad(
  payload: { model_path?: string | null; load_request_id: string },
  leaseHolder?: string,
): Promise<void> {
  if (!payload.model_path) return;
  await authFetch("/api/inference/unload", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...leaseHeaders(leaseHolder),
    },
    body: JSON.stringify({
      // biome-ignore lint/style/useNamingConvention: API schema
      model_path: payload.model_path,
      // biome-ignore lint/style/useNamingConvention: API schema
      cancel_load_request_id: payload.load_request_id,
    }),
  }).catch(() => undefined);
}

async function restore(
  before: InferenceStatusResponse,
  base: ReturnType<typeof chatBaseLoad>,
  leaseHolder?: string,
): Promise<void> {
  await loadModel(
    {
      ...base,
      llama_extra_args:
        before.requested_llama_extra_args ?? base.llama_extra_args,
      force_reload: true,
    },
    { runtime: "chat", headers: leaseHeaders(leaseHolder) },
  );
}
