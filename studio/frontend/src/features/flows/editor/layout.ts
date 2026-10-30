// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { InternalNode } from "@xyflow/react";
import { getSpec } from "../stores/catalog-store";
import { type DocGraph, NOTE_KEY, useDocStore } from "../stores/doc-store";
import { useUiStore } from "../stores/ui-store";
import { CARD_MIN_WIDTH } from "./card-size";
import { inHandle, outHandle } from "./connection-rules";
import type { LayoutRequest, LayoutResult } from "./layout-core";
import {
  type TidyContext,
  fittedCardSize,
  tidyPatch,
  tidyRequest,
} from "./tidy";

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

function runLayout(request: LayoutRequest): Promise<LayoutResult> {
  const id = ++nextId;
  return new Promise<LayoutResult>((resolve, reject) => {
    pending.set(id, { resolve, reject });
    getWorker().postMessage({ id, request });
  });
}

function rfElement(id: string): HTMLElement | null {
  return document.querySelector<HTMLElement>(
    `.react-flow__node[data-id="${CSS.escape(id)}"]`,
  );
}

/** How much of an element's content is cut off, the most of it or any descendant. */
function hiddenBelow(root: HTMLElement): number {
  let hidden = root.scrollHeight - root.clientHeight;
  for (const el of root.querySelectorAll<HTMLElement>("*"))
    if (el.clientHeight) hidden = Math.max(hidden, el.scrollHeight - el.clientHeight);
  return hidden;
}

/** Empty space at the bottom of a sized card's output area. */
function slackBelow(grow: HTMLElement): number {
  let bottom = 0;
  for (const child of grow.children)
    bottom = Math.max(
      bottom,
      (child as HTMLElement).offsetTop + (child as HTMLElement).offsetHeight,
    );
  return grow.clientHeight - (bottom - grow.offsetTop);
}

/**
 * Each card's size after fitting: one with output is sized to show it (up to a cap), an oversized
 * one shrinks, and a card with nothing to show goes back to its automatic size. A card in a group
 * never grows out of the frame.
 */
function fitCards(
  graph: DocGraph,
  measured: (id: string) => { width: number; height: number } | undefined,
): Record<string, [number, number] | null> {
  const out: Record<string, [number, number] | null> = {};
  for (const id of graph.nodeOrder) {
    const node = graph.nodes[id];
    const size = measured(id);
    const card = rfElement(id);
    // Zoomed far out a card draws no body, so there is nothing to measure.
    if (node.collapsed || !size || !card?.querySelector("[data-card-body]"))
      continue;
    const grow = card.querySelector<HTMLElement>("[data-card-grow]");
    if (!grow) {
      if (node.size) out[id] = null;
      continue;
    }
    const spec = getSpec(node.type);
    let fit = fittedCardSize(
      size,
      hiddenBelow(grow),
      node.size ? slackBelow(grow) : 0,
      spec ? CARD_MIN_WIDTH : 0,
    );
    if (!fit) continue;
    const frame = node.group ? graph.groups[node.group]?.rect : undefined;
    if (frame) {
      const pad = 12;
      fit = [
        Math.min(fit[0], frame[0] + frame[2] - node.pos[0] - pad),
        Math.min(fit[1], frame[1] + frame[3] - node.pos[1] - pad),
      ];
      if (fit[0] < size.width && fit[1] < size.height && !node.size) continue;
    }
    out[id] = fit;
  }
  return out;
}

/** Notes resized to their text. */
function fitNotes(graph: DocGraph): Record<string, number> {
  const out: Record<string, number> = {};
  for (const id of graph.noteOrder) {
    if (!graph.notes[id].text) continue;
    const body = rfElement(NOTE_KEY + id)?.querySelector<HTMLElement>(
      ".overflow-y-auto",
    );
    if (!body) continue;
    // The body fills the note, so measure what is in it (plus its py-2 padding and the border).
    let content = 16 + 4;
    for (const child of body.children) content += (child as HTMLElement).offsetHeight;
    const height = Math.min(480, Math.max(60, content));
    if (Math.abs(height - graph.notes[id].rect[3]) > 6) out[id] = height;
  }
  return out;
}

/** Tidy up: fit cards to their output, lay out cards and groups, then put notes and reroutes back. */
export async function autoLayout(
  getInternalNode: (id: string) => InternalNode | undefined,
  direction: "LR" | "TB" = "LR",
): Promise<void> {
  const { graph } = useDocStore.getState();
  const measured = (id: string) => {
    const m = getInternalNode(id)?.measured;
    return m?.width && m.height ? { width: m.width, height: m.height } : undefined;
  };
  const sizes = fitCards(graph, measured);
  const ctx: TidyContext = {
    direction,
    graph,
    wireStyle: useUiStore.getState().wireStyle,
    size: (id) => {
      const fit = sizes[id];
      return fit
        ? { width: fit[0], height: fit[1] }
        : (measured(id) ?? { width: 260, height: 140 });
    },
    portY: (id, port, side) => {
      const handle = getInternalNode(id)?.internals.handleBounds?.[
        side === "out" ? "source" : "target"
      ]?.find((h) => h.id === (side === "out" ? outHandle(port) : inHandle(port)));
      return handle ? handle.y + handle.height / 2 : undefined;
    },
  };
  const result = await runLayout(tidyRequest(ctx));
  const patch = tidyPatch(ctx, result, sizes);
  for (const [id, height] of Object.entries(fitNotes(graph))) {
    const rect = patch.notes[id] ?? graph.notes[id].rect;
    patch.notes[id] = [rect[0], rect[1], rect[2], height];
  }
  useDocStore.getState().applyTidy(patch);
}
