// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import {
  cancelRun,
  followRunEvents,
  latestOutputs,
  listRuns,
  planFlow,
  startRun,
} from "../api";
import { useCatalogStore } from "../stores/catalog-store";
import { documentFromState, useDocStore } from "../stores/doc-store";
import { pinsForRun } from "../stores/pins-store";
import { createEventBatcher, useRuntimeStore } from "../stores/runtime-store";
import { useUiStore } from "../stores/ui-store";
import { failedSteps, modelSignature, planCheck } from "../ux-rules";

let follower: AbortController | null = null;
// flowId -> the model picks of the last run that had nothing to download: the same picks skip the plan request.
const checkedModels = new Map<string, string>();
// Bumped when the editor closes: a download check still out then belongs to an editor that is gone.
let editorSession = 0;

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
        reportFailure();
      }
    });
}

/** Name the block that failed and why, with a way to jump to it; the run error when no block failed. */
function reportFailure(): void {
  const { graph } = useDocStore.getState();
  const runtime = useRuntimeStore.getState();
  const specs = useCatalogStore.getState().specs;
  const failed = failedSteps(
    graph.nodeOrder,
    graph.nodes,
    runtime.nodes,
    (type) => specs[type]?.title,
  );
  const first = failed[0];
  if (!first) {
    toast.error(runtime.runError ?? "The flow failed");
    return;
  }
  const more = failed.length > 1 ? ` (and ${failed.length - 1} more)` : "";
  toast.error(`${first.title} failed${more}`, {
    description: first.error,
    action: {
      label: "Open block",
      onClick: () => {
        useUiStore.getState().setSelection([first.id], []);
        useUiStore.getState().openInspector(first.id);
      },
    },
  });
}

export function detachRun(): void {
  follower?.abort();
  follower = null;
  // Leaving the editor is where models get deleted (Hub): the next visit asks the plan again.
  checkedModels.clear();
  editorSession += 1;
}

export interface RunOptions {
  targetNodes?: string[];
  forceNodes?: string[];
  params?: Record<string, unknown>;
  bake?: boolean;
  downloadsConfirmed?: boolean;
}

const PLAN_CHECK_TIMEOUT_MS = 4000;

/** True when the run waits for the user to agree to downloads; a slow or failed plan never blocks the run. */
async function heldForDownloads(
  flowId: string,
  options: RunOptions,
): Promise<boolean> {
  if (options.downloadsConfirmed) return false;
  const docState = useDocStore.getState();
  const signature = modelSignature(docState.graph.nodes);
  if (checkedModels.get(flowId) === signature) return false;
  const session = editorSession;
  const plan = await Promise.race([
    planFlow(flowId, options.params ?? {}, documentFromState(docState)),
    new Promise<null>((resolve) =>
      setTimeout(() => resolve(null), PLAN_CHECK_TIMEOUT_MS),
    ),
  ]).catch(() => null);
  const now = useDocStore.getState();
  const verdict = planCheck(
    plan ?? { models: [], errors: [{ message: "no plan" }] },
    { flowId, models: signature },
    { flowId: now.flowId, models: modelSignature(now.graph.nodes) },
  );
  // The editor closed, or the flow or its model picks changed meanwhile: this plan no longer describes the run.
  if (verdict === "stale" || session !== editorSession) return true;
  if (!plan) return false;
  if (verdict === "clear") checkedModels.set(flowId, signature);
  if (verdict !== "download") return false;
  useUiStore.getState().setPendingRun({ options: { ...options }, plan, models: signature });
  return true;
}

// Set before the start request is awaited: a double-click on Run must not start two runs.
let starting = false;

export async function runFlow(options: RunOptions = {}): Promise<void> {
  const docState = useDocStore.getState();
  const flowId = docState.flowId;
  if (!flowId) return;
  const runtime = useRuntimeStore.getState();
  if (
    starting ||
    runtime.runStatus === "running" ||
    runtime.runStatus === "queued" ||
    runtime.runStatus === "cancelling"
  ) {
    toast.message("A run is already in progress");
    return;
  }
  starting = true;
  try {
    if (await heldForDownloads(flowId, options)) return;
    const { run_id } = await startRun(flowId, {
      doc: documentFromState(docState),
      params: options.params,
      target_nodes: options.targetNodes,
      force_nodes: options.forceNodes,
      bake: options.bake,
      pinned: pinsForRun(docState.graph.nodeOrder),
    });
    useRuntimeStore.getState().startRun(run_id);
    useUiStore.getState().setDrawer(true, "run");
    attachRun(run_id, 0);
  } catch (error) {
    toast.error(
      error instanceof Error ? error.message : "Could not start the run",
    );
  } finally {
    starting = false;
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
