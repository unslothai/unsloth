// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { InternalNode } from "@xyflow/react";
import { useDocStore } from "../stores/doc-store";
import {
  isSetupId,
  setupLayoutNodes,
  withLaidOutSetup,
} from "../recipe/recipe-setup";
import type { LayoutRequest, LayoutResult } from "./layout-core";

let worker: Worker | null = null;
let nextId = 0;
const pending = new Map<
  number,
  { resolve: (r: LayoutResult) => void; reject: (e: Error) => void }
>();

function getWorker(): Worker {
  if (worker) return worker;
  worker = new Worker(new URL("./layout.worker.ts", import.meta.url), {
    type: "module",
  });
  worker.onmessage = (
    event: MessageEvent<{ id: number; result?: LayoutResult; error?: string }>,
  ) => {
    const entry = pending.get(event.data.id);
    if (!entry) return;
    pending.delete(event.data.id);
    if (event.data.error) entry.reject(new Error(event.data.error));
    else entry.resolve(event.data.result ?? {});
  };
  return worker;
}

export async function autoLayout(
  getInternalNode: (id: string) => InternalNode | undefined,
  direction: "LR" | "TB" = "LR",
): Promise<void> {
  const { graph } = useDocStore.getState();
  const request: LayoutRequest = {
    direction,
    nodes: graph.nodeOrder.map((id) => {
      const measured = getInternalNode(id)?.measured;
      return {
        id,
        width: measured?.width ?? 260,
        height: measured?.height ?? 140,
      };
    }),
    edges: graph.edgeOrder.map((id) => ({
      source: graph.edges[id].source,
      target: graph.edges[id].target,
    })),
  };
  // Setup cards take part as nodes, so the layout leaves room for them and saves where they went.
  const setup = setupLayoutNodes(graph, (id) => getInternalNode(id)?.measured);
  request.nodes.push(...setup.nodes);
  request.edges.push(...setup.edges);
  const id = ++nextId;
  const result = await new Promise<LayoutResult>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, request });
  });
  const nodes = Object.fromEntries(
    Object.entries(result).filter(([key]) => !isSetupId(key)),
  );
  useDocStore.getState().moveNodes(nodes);
  const groups = withLaidOutSetup(useDocStore.getState().graph, result);
  if (groups.length) useDocStore.getState().patchGraph({ groups });
}
