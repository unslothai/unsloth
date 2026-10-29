// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { DocGraph } from "../stores/doc-store";
import { useDocStore } from "../stores/doc-store";
import { useRuntimeStore } from "../stores/runtime-store";

const BUSY = new Set(["queued", "waiting", "loading_model", "running"]);

function inputsOf(graph: DocGraph): Map<string, string> {
  const wires = new Map<string, string[]>();
  for (const id of graph.edgeOrder) {
    const e = graph.edges[id];
    const list = wires.get(e.target) ?? [];
    list.push(`${e.source}:${e.source_port}>${e.target_port}`);
    wires.set(e.target, list);
  }
  return new Map([...wires].map(([id, list]) => [id, list.sort().join("|")]));
}

/** Cards whose incoming wires differ between two graphs. */
export function rewiredCards(before: DocGraph, after: DocGraph): string[] {
  const a = inputsOf(before);
  const b = inputsOf(after);
  const ids = new Set([...a.keys(), ...b.keys()]);
  return [...ids].filter((id) => a.get(id) !== b.get(id) && after.nodes[id]);
}

/** A card rewired after a run stops showing that run's output; one mid-run keeps it. */
export function watchRewiredOutputs(): () => void {
  return useDocStore.subscribe((s, p) => {
    if (s.flowId !== p.flowId || s.graph.edges === p.graph.edges) return;
    const runtime = useRuntimeStore.getState();
    const stale = rewiredCards(p.graph, s.graph).filter(
      (id) => !BUSY.has(runtime.nodes[id]?.status ?? "idle"),
    );
    if (stale.length) runtime.forget(stale);
  });
}
