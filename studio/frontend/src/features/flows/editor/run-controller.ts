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
import { syncResidentModel } from "./resident-sync";

let follower: AbortController | null = null;
const checkedModels = new Map<string, string>();
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
      if (controller.signal.aborted) return;
      // A run can load a different model on the server.
      syncResidentModel();
      if (useRuntimeStore.getState().runStatus === "failed") reportFailure();
    });
}

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
  if (verdict === "stale" || session !== editorSession) return true;
  if (!plan) return false;
  if (verdict === "clear") checkedModels.set(flowId, signature);
  if (verdict !== "download") return false;
  useUiStore.getState().setPendingRun({ options: { ...options }, plan, models: signature });
  return true;
}

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
  }
}
