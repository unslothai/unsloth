// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import {
  cancelRun,
  followRunEvents,
  latestOutputs,
  listRuns,
  startRun,
} from "../api";
import { documentFromState, useDocStore } from "../stores/doc-store";
import { createEventBatcher, useRuntimeStore } from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";

let follower: AbortController | null = null;

export function attachRun(runId: string, afterSeq = 0): void {
  follower?.abort();
  const controller = new AbortController();
  follower = controller;
  const batcher = createEventBatcher((events) =>
    useRuntimeStore.getState().applyEvents(events),
  );
  void followRunEvents(
    runId,
    afterSeq,
    (event) => batcher.push(event),
    controller.signal,
  )
    .catch(() => undefined)
    .finally(() => {
      batcher.flush();
      batcher.dispose();
      if (follower === controller) follower = null;
      const { runStatus } = useRuntimeStore.getState();
      if (!controller.signal.aborted && runStatus === "failed") {
        toast.error(useRuntimeStore.getState().runError ?? "The flow failed");
      }
    });
}

export function detachRun(): void {
  follower?.abort();
  follower = null;
}

export interface RunOptions {
  targetNodes?: string[];
  forceNodes?: string[];
  params?: Record<string, unknown>;
  bake?: boolean;
}

export async function runFlow(options: RunOptions = {}): Promise<void> {
  const docState = useDocStore.getState();
  const flowId = docState.flowId;
  if (!flowId) return;
  const runtime = useRuntimeStore.getState();
  if (
    runtime.runStatus === "running" ||
    runtime.runStatus === "queued" ||
    runtime.runStatus === "cancelling"
  ) {
    toast.message("A run is already in progress");
    return;
  }
  try {
    const { run_id } = await startRun(flowId, {
      doc: documentFromState(docState),
      params: options.params,
      target_nodes: options.targetNodes,
      force_nodes: options.forceNodes,
      bake: options.bake,
    });
    useRuntimeStore.getState().startRun(run_id);
    useUiStore.getState().setDrawer(true, "run");
    attachRun(run_id, 0);
  } catch (error) {
    toast.error(
      error instanceof Error ? error.message : "Could not start the run",
    );
  }
}

export async function stopFlow(): Promise<void> {
  const { runId } = useRuntimeStore.getState();
  if (!runId) return;
  try {
    await cancelRun(runId);
  } catch (error) {
    toast.error(
      error instanceof Error ? error.message : "Could not stop the run",
    );
  }
}

/** On open: show the latest outputs and re-attach to a run still in progress. */
export async function hydrateRuntime(flowId: string): Promise<void> {
  const runtime = useRuntimeStore.getState();
  runtime.clear();
  try {
    const [latest, runs] = await Promise.all([
      latestOutputs(flowId).catch(() => ({ nodes: {} })),
      listRuns(flowId).catch(() => ({ runs: [] })),
    ]);
    if (useDocStore.getState().flowId !== flowId) return;
    const seed: Record<
      string,
      {
        outputs: Record<string, unknown> | null;
        status: "done" | "cached" | "failed" | "idle";
        error: string | null;
      }
    > = {};
    for (const [id, state] of Object.entries(latest.nodes ?? {})) {
      seed[id] = {
        outputs: state.outputs ?? null,
        status:
          state.status === "failed"
            ? "failed"
            : state.outputs
              ? "done"
              : "idle",
        error: state.error ?? null,
      };
    }
    useRuntimeStore.getState().seedOutputs(seed);
    const active = runs.runs.find(
      (r) =>
        r.status === "running" ||
        r.status === "queued" ||
        r.status === "cancelling",
    );
    if (active) {
      useRuntimeStore.getState().startRun(active.id);
      attachRun(active.id, 0);
    }
  } catch {
    // Previews are a nicety; the editor works without them.
  }
}
