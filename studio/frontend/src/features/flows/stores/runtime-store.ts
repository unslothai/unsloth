// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { NodeStatus, RunEvent, RunStatus } from "../types";

export interface NodeRuntime {
  status: NodeStatus;
  progress: number | null;
  phase: string | null;
  preview: string | null;
  outputs: Record<string, unknown> | null;
  partial: Record<string, unknown> | null;
  partialStep: { step: number; total: number } | null;
  error: string | null;
  cached: boolean;
  text: string;
  startedAt: number | null;
  finishedAt: number | null;
}

export interface RunLogLine {
  seq: number;
  ts: number;
  nodeId: string | null;
  level: string;
  message: string;
}

export interface RuntimeState {
  runId: string | null;
  runStatus: RunStatus | null;
  runError: string | null;
  lastSeq: number;
  nodes: Record<string, NodeRuntime>;
  logs: RunLogLine[];
  startRun: (runId: string) => void;
  applyEvents: (events: RunEvent[]) => void;
  seedOutputs: (nodes: Record<string, Partial<NodeRuntime>>) => void;
  /** Drops what these cards last showed, so they read as not run. */
  forget: (ids: string[]) => void;
  clear: () => void;
}

const LOG_LIMIT = 500;
const TEXT_LIMIT = 20_000;

export const IDLE_RUNTIME: NodeRuntime = Object.freeze({
  status: "idle",
  progress: null,
  phase: null,
  preview: null,
  outputs: null,
  partial: null,
  partialStep: null,
  error: null,
  cached: false,
  text: "",
  startedAt: null,
  finishedAt: null,
}) as NodeRuntime;

interface PartialData {
  port?: string;
  value?: unknown;
  step?: number;
  total?: number;
}

function stepOf(data: PartialData): { step: number; total: number } | null {
  return typeof data.step === "number" && typeof data.total === "number"
    ? { step: data.step, total: data.total }
    : null;
}

function partialFields(
  stored: Record<string, PartialData> | undefined,
): Pick<NodeRuntime, "partial" | "partialStep"> {
  if (!stored || typeof stored !== "object")
    return { partial: null, partialStep: null };
  const partial: Record<string, unknown> = {};
  let partialStep: { step: number; total: number } | null = null;
  for (const [port, data] of Object.entries(stored)) {
    partial[port] = data?.value;
    partialStep = (data && stepOf(data)) ?? partialStep;
  }
  return { partial, partialStep };
}

const TERMINAL_RUN: Record<string, RunStatus> = {
  "run.completed": "completed",
  "run.failed": "failed",
  "run.cancelled": "cancelled",
  "run.interrupted": "interrupted",
  "run.cancelling": "cancelling",
  "run.started": "running",
  "run.queued": "queued",
};

export function reduceEvents(
  state: Pick<
    RuntimeState,
    "runId" | "runStatus" | "runError" | "lastSeq" | "nodes" | "logs"
  >,
  events: RunEvent[],
): Pick<RuntimeState, "runStatus" | "runError" | "lastSeq" | "nodes" | "logs"> {
  let nodes = state.nodes;
  let touched: Record<string, NodeRuntime> | null = null;
  let logs: RunLogLine[] | null = null;
  let { runStatus, runError, lastSeq } = state;

  const entry = (id: string): NodeRuntime => {
    touched ??= {};
    let current = touched[id];
    if (!current) {
      current = { ...(nodes[id] ?? IDLE_RUNTIME) };
      touched[id] = current;
    }
    return current;
  };

  for (const event of events) {
    if (state.runId && event.run_id && event.run_id !== state.runId) continue;
    if (event.type !== "snapshot" && event.seq <= lastSeq) continue;
    lastSeq = Math.max(lastSeq, event.seq);
    const data = event.data ?? {};
    const id = event.node_id ?? null;
    if (event.type === "snapshot") {
      const run = data.run as
        | { status?: RunStatus; error?: string | null }
        | undefined;
      if (run?.status) runStatus = run.status;
      runError = run?.error ?? null;
      const snapNodes = (data.nodes ?? {}) as Record<
        string,
        {
          status?: NodeStatus;
          outputs?: Record<string, unknown> | null;
          error?: string | null;
          progress?: Record<string, unknown> | number | null;
          partial?: Record<string, PartialData>;
        }
      >;
      const rebuilt: Record<string, NodeRuntime> = {};
      for (const [nodeId, snap] of Object.entries(snapNodes)) {
        const prev = nodes[nodeId] ?? IDLE_RUNTIME;
        rebuilt[nodeId] = {
          ...prev,
          status: snap.status ?? prev.status,
          outputs: snap.outputs ?? prev.outputs,
          ...partialFields(snap.partial),
          error: snap.error ?? null,
          progress: progressFraction(snap.progress),
          cached: snap.status === "cached",
        };
      }
      nodes = rebuilt;
      touched = null;
      continue;
    }
    if (event.type in TERMINAL_RUN) {
      runStatus = TERMINAL_RUN[event.type];
      if (event.type === "run.failed")
        runError = (data.error as string) ?? (data.message as string) ?? null;
      continue;
    }
    if (!id) continue;
    if (event.type === "node.status") {
      const node = entry(id);
      const status = data.status as NodeStatus;
      node.status = status;
      node.cached = status === "cached";
      if (status === "running" || status === "loading_model")
        node.startedAt ??= event.ts;
      if (status === "queued") {
        node.text = "";
        node.error = null;
        node.progress = null;
        node.preview = null;
        node.partial = null;
        node.partialStep = null;
        node.startedAt = null;
        node.finishedAt = null;
      }
      if (status === "failed")
        node.error =
          (data.error as string) ?? (data.message as string) ?? "Failed";
      if (
        status === "done" ||
        status === "cached" ||
        status === "failed" ||
        status === "skipped" ||
        status === "cancelled"
      ) {
        node.finishedAt = event.ts;
        node.progress =
          status === "done" || status === "cached" ? 1 : node.progress;
        node.phase = null;
        node.preview = null;
      }
      if (data.message && status !== "failed")
        node.phase = String(data.message);
    } else if (event.type === "node.progress") {
      const node = entry(id);
      node.progress = progressFraction(data) ?? node.progress;
      if (typeof data.phase === "string") node.phase = data.phase;
      if (typeof data.preview === "string") node.preview = data.preview;
    } else if (event.type === "node.output") {
      const node = entry(id);
      node.outputs = (data.outputs as Record<string, unknown>) ?? null;
      node.partial = null;
      node.partialStep = null;
      if (data.cached) node.cached = true;
    } else if (event.type === "node.partial") {
      const node = entry(id);
      const port = String(data.port ?? "");
      if (port && "value" in data) {
        node.partial = { ...(node.partial ?? {}), [port]: data.value };
        node.partialStep = stepOf(data as PartialData) ?? node.partialStep;
      }
    } else if (event.type === "node.text_delta") {
      const node = entry(id);
      node.text = (node.text + String(data.delta ?? "")).slice(-TEXT_LIMIT);
    } else if (event.type === "node.log") {
      logs ??= state.logs.slice();
      logs.push({
        seq: event.seq,
        ts: event.ts,
        nodeId: id,
        level: String(data.level ?? "info"),
        message: String(data.message ?? ""),
      });
    }
  }
  if (touched)
    nodes = { ...nodes, ...(touched as Record<string, NodeRuntime>) };
  return {
    runStatus,
    runError,
    lastSeq,
    nodes,
    logs: logs ? (logs as RunLogLine[]).slice(-LOG_LIMIT) : state.logs,
  };
}

export function progressFraction(data: unknown): number | null {
  if (typeof data === "number") return data;
  if (!data || typeof data !== "object") return null;
  const p = data as { fraction?: unknown; step?: unknown; total?: unknown };
  if (typeof p.fraction === "number") return p.fraction;
  if (typeof p.step === "number" && typeof p.total === "number" && p.total > 0)
    return p.step / p.total;
  return null;
}

export const useRuntimeStore = create<RuntimeState>((set, get) => ({
  runId: null,
  runStatus: null,
  runError: null,
  lastSeq: 0,
  nodes: {},
  logs: [],
  startRun: (runId) =>
    set((s) => ({
      runId,
      runStatus: "queued",
      runError: null,
      lastSeq: 0,
      logs: [],
      nodes: Object.fromEntries(
        Object.entries(s.nodes).map(([id, node]) => [
          id,
          {
            ...node,
            status: "idle" as NodeStatus,
            progress: null,
            error: null,
            text: "",
            partial: null,
            partialStep: null,
          },
        ]),
      ),
    })),
  applyEvents: (events) => {
    if (!events.length) return;
    set(reduceEvents(get(), events));
  },
  seedOutputs: (seed) =>
    set((s) => {
      const nodes = { ...s.nodes };
      for (const [id, partial] of Object.entries(seed)) {
        nodes[id] = { ...(nodes[id] ?? IDLE_RUNTIME), ...partial };
      }
      return { nodes };
    }),
  forget: (ids) =>
    set((s) => {
      if (!ids.some((id) => s.nodes[id])) return s;
      const nodes = { ...s.nodes };
      for (const id of ids) delete nodes[id];
      return { nodes };
    }),
  clear: () =>
    set({
      runId: null,
      runStatus: null,
      runError: null,
      lastSeq: 0,
      nodes: {},
      logs: [],
    }),
}));

export function createEventBatcher(apply: (events: RunEvent[]) => void): {
  push: (event: RunEvent) => void;
  flush: () => void;
  dispose: () => void;
} {
  let queue: RunEvent[] = [];
  let handle: number | null = null;
  const raf: (cb: () => void) => number =
    typeof requestAnimationFrame === "function"
      ? (cb) => requestAnimationFrame(cb)
      : (cb) => setTimeout(cb, 16) as unknown as number;
  const caf: (h: number) => void =
    typeof cancelAnimationFrame === "function"
      ? (h) => cancelAnimationFrame(h)
      : (h) => clearTimeout(h);
  const flush = () => {
    handle = null;
    if (!queue.length) return;
    const batch = queue;
    queue = [];
    apply(batch);
  };
  return {
    push: (event) => {
      queue.push(event);
      handle ??= raf(flush);
    },
    flush: () => {
      if (handle !== null) caf(handle);
      flush();
    },
    dispose: () => {
      if (handle !== null) caf(handle);
      handle = null;
      queue = [];
    },
  };
}
