// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { toast } from "sonner";
import { fetchMediaBlob, mediaPath } from "../api";
import { getSpec } from "../stores/catalog-store";
import { useDocStore } from "../stores/doc-store";
import { useRuntimeStore } from "../stores/runtime-store";
import { outputDownload } from "./canvas-rules";
import { runFlow } from "./run-controller";

/** Run only what this node needs. */
export function runToNode(nodeId: string): Promise<void> {
  return runFlow({ targetNodes: [nodeId] });
}

/** Run this node again past the cache, with a new seed when it has one. */
export function rerollNode(nodeId: string): Promise<void> {
  const spec = getSpec(useDocStore.getState().graph.nodes[nodeId]?.type ?? "");
  if (spec?.params.some((p) => p.id === "seed" && p.kind === "seed")) {
    useDocStore
      .getState()
      .updateParams(
        nodeId,
        { seed: Math.floor(Math.random() * 2 ** 31) },
        `param:${nodeId}:seed`,
      );
  }
  return runFlow({ targetNodes: [nodeId], forceNodes: [nodeId] });
}

export function downloadBlob(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}

export async function downloadNodeOutput(nodeId: string): Promise<void> {
  const node = useDocStore.getState().graph.nodes[nodeId];
  const outputs = useRuntimeStore.getState().nodes[nodeId]?.outputs ?? null;
  const plan = outputDownload(
    outputs,
    node?.label || getSpec(node?.type ?? "")?.title || "output",
  );
  if (!plan) {
    toast("Run this block first; it has no output yet");
    return;
  }
  if (plan.kind === "json") {
    downloadBlob(
      new Blob([plan.text], { type: "application/json" }),
      plan.filename,
    );
    return;
  }
  const path = mediaPath(plan.media);
  if (!path) {
    toast.error("This output cannot be downloaded");
    return;
  }
  try {
    downloadBlob(await fetchMediaBlob(path), plan.filename);
  } catch (error) {
    toast.error(error instanceof Error ? error.message : "Download failed");
  }
}
