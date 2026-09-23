// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Runs a sweep against this Studio: per variant, reload the loaded GGUF with the
// variant's overrides, then stream timed completions and keep llama-server's own
// timings. Everything goes through the public load / status / chat routes.

import { authFetch } from "@/features/auth";
import { type InferenceStatusResponse, getInferenceStatus, loadModel } from "@/features/chat";

type LoadModelRequest = Parameters<typeof loadModel>[0];
import {
  type BenchConfig,
  type BenchRun,
  type RunMeta,
  type RunResult,
  type VariantOutcome,
  promptsFor,
  servedMismatch,
  userExtraArgs,
  variantLoad,
} from "../lib/bench-math";

export interface RunnerEvents {
  onOutcome: (outcome: VariantOutcome) => void;
  onResult: (result: RunResult) => void;
  onProgress: (text: string) => void;
}

export class BenchSetupError extends Error {}

/** The loaded model's own settings, which every variant starts from. */
function baseLoad(status: InferenceStatusResponse): LoadModelRequest {
  const base: LoadModelRequest = {
    model_path: status.active_model ?? "",
    gguf_variant: status.gguf_variant ?? null,
    hf_token: null,
    load_in_4bit: false,
    is_lora: false,
    max_seq_length: status.requested_context_length ?? 0,
    cache_type_kv: status.cache_type_kv ?? null,
    // Held as served, so a KV or context sweep doesn't also change the drafting mode.
    speculative_type: status.speculative_type ?? null,
    spec_draft_n_max: status.spec_draft_n_max ?? null,
    n_parallel: status.requested_parallel_slots ?? null,
    llama_extra_args: userExtraArgs(status.requested_llama_extra_args),
    disable_vision: status.vision_disabled_by_user ?? null,
    tensor_parallel: status.tensor_parallel ?? null,
  };
  if (status.gpu_memory_mode === "manual") {
    base.gpu_memory_mode = "manual";
    if (status.gpu_layers !== undefined) base.gpu_layers = status.gpu_layers;
    if (status.n_cpu_moe !== undefined) base.n_cpu_moe = status.n_cpu_moe;
    if (status.tensor_split) base.tensor_split = status.tensor_split;
  }
  if (status.requested_gpu_ids?.length) base.gpu_ids = status.requested_gpu_ids;
  return base;
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
  ] as const;
  const out: Record<string, unknown> = {};
  for (const k of keys) {
    const v = (st as unknown as Record<string, unknown>)[k];
    if (v !== null && v !== undefined) out[k] = v;
  }
  return out;
}

async function getJson(path: string, signal: AbortSignal): Promise<Record<string, unknown> | null> {
  try {
    const res = await authFetch(path, { signal });
    return res.ok ? ((await res.json()) as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

async function readMeta(signal: AbortSignal): Promise<RunMeta> {
  const [llama, hw, health] = await Promise.all([
    getJson("/api/llama/backend", signal),
    getJson("/api/system/hardware", signal),
    getJson("/api/health", signal),
  ]);
  const gpu = (hw?.gpu ?? null) as Record<string, unknown> | null;
  const str = (v: unknown) => (typeof v === "string" && v ? v : null);
  return {
    gpu: str(gpu?.gpu_name),
    vramGb: typeof gpu?.vram_total_gb === "number" ? gpu.vram_total_gb : null,
    backend: str(llama?.backend),
    llamaTag: str(llama?.installed_tag),
    studioVersion: str(health?.studio_version) ?? str(health?.version),
  };
}

interface Completion {
  ttftMs: number | null;
  wallMs: number;
  clientTokens: number;
  timings: Record<string, unknown>;
}

async function streamOnce(
  model: string,
  prompt: string,
  config: BenchConfig,
  signal: AbortSignal,
): Promise<Completion> {
  const started = performance.now();
  const res = await authFetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model,
      messages: [{ role: "user", content: prompt }],
      stream: true,
      // biome-ignore lint/style/useNamingConvention: API schema
      stream_options: { include_usage: true },
      // biome-ignore lint/style/useNamingConvention: API schema
      max_tokens: config.maxTokens,
      temperature: config.temperature,
      seed: config.seed,
    }),
    signal,
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    throw new Error(`Chat completion answered ${res.status}${text ? `: ${text.slice(0, 300)}` : ""}`);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let ttftMs: number | null = null;
  let clientTokens = 0;
  let timings: Record<string, unknown> = {};
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
      if (!data || data === "[DONE]") continue;
      let chunk: Record<string, unknown>;
      try {
        chunk = JSON.parse(data) as Record<string, unknown>;
      } catch {
        continue;
      }
      const err = chunk.error as { message?: string } | undefined;
      if (err) throw new Error(err.message ?? "Stream error");
      if (chunk.timings && typeof chunk.timings === "object") timings = chunk.timings as Record<string, unknown>;
      const choices = Array.isArray(chunk.choices) ? (chunk.choices as Record<string, unknown>[]) : [];
      for (const choice of choices) {
        const delta = (choice.delta ?? {}) as Record<string, unknown>;
        const piece = [delta.content, delta.reasoning_content, delta.reasoning].find((x) => typeof x === "string" && x.length > 0);
        if (piece) {
          clientTokens += 1;
          if (ttftMs === null) ttftMs = performance.now() - started;
        }
      }
    }
  }
  return { ttftMs, wallMs: performance.now() - started, clientTokens, timings };
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
  const status = await getInferenceStatus(signal);
  if (!status.active_model) throw new BenchSetupError("Load a GGUF model in chat first. Benchmarks measure the model that is loaded.");
  if (status.is_gguf === false) throw new BenchSetupError("Benchmarks run on GGUF models. The loaded model runs on another backend.");
  const prompts = promptsFor(config.promptSet, config.customPrompt);
  if (prompts.length === 0) throw new BenchSetupError("The custom prompt is empty.");

  const base = baseLoad(status);
  const run: BenchRun = {
    id: `run-${Date.now().toString(36)}`,
    createdAt: Date.now(),
    finishedAt: null,
    model: status.active_model,
    ggufVariant: status.gguf_variant ?? null,
    kv: status.cache_type_kv ?? null,
    context: status.context_length ?? null,
    config,
    meta: await readMeta(signal),
    outcomes: config.variants.map((v) => ({ label: v.label, state: "queued" })),
    results: [],
  };
  const setOutcome = (o: VariantOutcome) => {
    run.outcomes = run.outcomes.map((x) => (x.label === o.label ? o : x));
    events.onOutcome(o);
  };

  let promptCursor = 0;
  try {
    for (const variant of config.variants) {
      if (signal.aborted) throw abortError();
      setOutcome({ label: variant.label, state: "loading" });
      events.onProgress(`Loading ${variant.label}`);
      const loadStarted = performance.now();
      let modelId: string;
      try {
        const loaded = await loadModel(variantLoad(base, variant), { signal, runtime: "chat" });
        modelId = loaded.model;
      } catch (err) {
        if (signal.aborted) throw abortError();
        setOutcome({ label: variant.label, state: "error", reason: err instanceof Error ? err.message : String(err) });
        continue;
      }
      const loadMs = performance.now() - loadStarted;
      const served = await getInferenceStatus(signal);
      const mismatch = servedMismatch(variant, served);
      if (mismatch) {
        setOutcome({ label: variant.label, state: "skipped", reason: mismatch, served: servedSubset(served) });
        continue;
      }
      setOutcome({ label: variant.label, state: "running", served: servedSubset(served) });
      const total = config.warmup + config.repetitions;
      for (let rep = 0; rep < total; rep++) {
        if (signal.aborted) throw abortError();
        const warmup = rep < config.warmup;
        events.onProgress(
          warmup ? `${variant.label} · warm-up ${rep + 1}/${config.warmup}` : `${variant.label} · run ${rep + 1 - config.warmup}/${config.repetitions}`,
        );
        const promptIndex = config.rotatePrompts ? promptCursor++ % prompts.length : 0;
        const c = await streamOnce(modelId, prompts[promptIndex], config, signal);
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
          clientTps: decodeMs && decodeMs > 0 ? ((genTokens ?? c.clientTokens) / decodeMs) * 1000 : null,
          draftN: num(t.draft_n),
          draftAccepted: num(t.draft_n_accepted),
          loadMs: rep === 0 ? loadMs : null,
          at: Date.now(),
        };
        run.results.push(result);
        events.onResult(result);
      }
      setOutcome({ label: variant.label, state: "done", served: servedSubset(served) });
    }
  } catch (err) {
    if (!signal.aborted) throw err;
    for (const o of run.outcomes) {
      if (o.state === "queued" || o.state === "loading" || o.state === "running") setOutcome({ ...o, state: "cancelled" });
    }
  } finally {
    run.finishedAt = Date.now();
    if (config.restoreAfter) {
      events.onProgress("Restoring your original settings");
      await restore(status).catch(() => undefined);
    }
  }
  return run;
}

/** Put the model back the way the user had it before the sweep. */
async function restore(before: InferenceStatusResponse): Promise<void> {
  const base = baseLoad(before);
  await loadModel(
    {
      ...base,
      llama_extra_args: before.requested_llama_extra_args ?? null,
      speculative_type: before.speculative_type ?? null,
      spec_draft_n_max: before.spec_draft_n_max ?? null,
      force_reload: true,
    },
    { runtime: "chat" },
  );
}
