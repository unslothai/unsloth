// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { useReactFlow } from "@xyflow/react";
import { useEffect } from "react";
import {
  type GraphFragment,
  cloneFragment,
  graphFromDocument,
  newId,
  useDocStore,
} from "../stores/doc-store";
import { removeNodes } from "../recipe/recipe-actions";
import { useUiStore } from "../stores/ui-store";
import type { FlowEdge, FlowNode } from "../types";
import { addFileNodes } from "./file-drop";
import { runFlow } from "./run-controller";
import { flushSave } from "./use-autosave";

const CLIPBOARD_KIND = "unsloth-flow-fragment";
let clipboard: GraphFragment | null = null;
let pointer: { x: number; y: number } | null = null;

function isTyping(target: EventTarget | null): boolean {
  const el = target as HTMLElement | null;
  if (!el) return false;
  return (
    el.isContentEditable ||
    el.tagName === "INPUT" ||
    el.tagName === "TEXTAREA" ||
    el.tagName === "SELECT"
  );
}

export function copySelection(): GraphFragment | null {
  const { graph } = useDocStore.getState();
  const ids = useUiStore
    .getState()
    .selectedNodes.filter((id) => graph.nodes[id]);
  if (!ids.length) return null;
  const set = new Set(ids);
  const fragment: GraphFragment = {
    nodes: ids.map((id) => graph.nodes[id]),
    edges: graph.edgeOrder
      .map((id) => graph.edges[id])
      .filter((e) => set.has(e.source) && set.has(e.target)),
  };
  clipboard = fragment;
  void navigator.clipboard
    ?.writeText(JSON.stringify({ kind: CLIPBOARD_KIND, ...fragment }))
    .catch(() => undefined);
  return fragment;
}

export function placeFragment(
  fragment: GraphFragment,
  at: { x: number; y: number } | null,
): GraphFragment {
  const graph = graphFromDocument({
    nodes: fragment.nodes,
    edges: fragment.edges,
  } as Parameters<typeof graphFromDocument>[0]);
  const minX = Math.min(...fragment.nodes.map((n) => n.pos[0]));
  const minY = Math.min(...fragment.nodes.map((n) => n.pos[1]));
  const delta: [number, number] = at ? [at.x - minX, at.y - minY] : [40, 40];
  return cloneFragment(graph, graph.nodeOrder, delta);
}

function parseClipboardText(text: string): GraphFragment | null {
  try {
    const data = JSON.parse(text) as {
      kind?: string;
      nodes?: FlowNode[];
      edges?: FlowEdge[];
    };
    if (data.kind !== CLIPBOARD_KIND || !Array.isArray(data.nodes)) return null;
    return {
      nodes: data.nodes,
      edges: Array.isArray(data.edges) ? data.edges : [],
    };
  } catch {
    return null;
  }
}

async function clipboardImage(): Promise<File | null> {
  if (!navigator.clipboard?.read) return null;
  try {
    for (const item of await navigator.clipboard.read()) {
      const type = item.types.find((t) => t.startsWith("image/"));
      if (type) {
        const blob = await item.getType(type);
        return new File([blob], `pasted.${type.split("/")[1] || "png"}`, {
          type,
        });
      }
    }
  } catch {
    return null;
  }
  return null;
}

function centreOfCanvas(): { x: number; y: number } {
  const rect = document
    .querySelector('[data-testid="flows-canvas"]')
    ?.getBoundingClientRect();
  return rect
    ? { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 }
    : { x: window.innerWidth / 2, y: window.innerHeight / 2 };
}

export function useEditorShortcuts(active: boolean): void {
  const { fitView, screenToFlowPosition } = useReactFlow();
  useEffect(() => {
    if (!active) return;
    const onPointerMove = (event: PointerEvent) => {
      pointer = { x: event.clientX, y: event.clientY };
    };
    const paste = (fragment: GraphFragment) => {
      const at = pointer ? screenToFlowPosition(pointer) : null;
      const placed = placeFragment(fragment, at);
      useDocStore.getState().insertFragment(placed);
      useUiStore.getState().setSelection(
        placed.nodes.map((n) => n.id),
        [],
      );
    };
    const onKeyDown = (event: KeyboardEvent) => {
      const mod = event.metaKey || event.ctrlKey;
      const key = event.key.toLowerCase();
      if (key === "escape") {
        const ui = useUiStore.getState();
        if (ui.quickAdd) ui.setQuickAdd(null);
        else if (!isTyping(event.target)) ui.setSelection([], []);
        return;
      }
      if (mod && key === "s") {
        event.preventDefault();
        void flushSave();
        return;
      }
      if (mod && key === "enter") {
        event.preventDefault();
        const run = document.querySelector<HTMLButtonElement>(
          '[data-testid="flows-run"]',
        );
        if (run) run.click();
        else void runFlow();
        return;
      }
      if (isTyping(event.target)) return;
      const doc = useDocStore.getState();
      if (mod && key === "z" && !event.shiftKey) {
        event.preventDefault();
        doc.undo();
      } else if (mod && ((key === "z" && event.shiftKey) || key === "y")) {
        event.preventDefault();
        doc.redo();
      } else if (mod && key === "c") {
        if (copySelection()) event.preventDefault();
      } else if (mod && key === "x") {
        if (copySelection()) {
          event.preventDefault();
          removeNodes(useUiStore.getState().selectedNodes);
          useUiStore.getState().setSelection([], []);
        }
      } else if (mod && key === "v") {
        event.preventDefault();
        const local = clipboard;
        const pasteBlocks = () => {
          if (navigator.clipboard?.readText) {
            void navigator.clipboard
              .readText()
              .then((text) => {
                const fragment = parseClipboardText(text) ?? local;
                if (fragment) paste(fragment);
              })
              .catch(() => local && paste(local));
          } else if (local) {
            paste(local);
          }
        };
        void clipboardImage().then((file) => {
          if (!file) return pasteBlocks();
          const at = pointer ?? centreOfCanvas();
          void addFileNodes([file], screenToFlowPosition(at));
        });
      } else if (mod && key === "d") {
        event.preventDefault();
        const ids = useUiStore
          .getState()
          .selectedNodes.filter((id) => doc.graph.nodes[id]);
        if (!ids.length) return;
        const fragment = cloneFragment(doc.graph, ids, [40, 40]);
        doc.insertFragment(fragment);
        useUiStore.getState().setSelection(
          fragment.nodes.map((n) => n.id),
          [],
        );
      } else if (mod && key === "a") {
        event.preventDefault();
        useUiStore
          .getState()
          .setSelection([...doc.graph.nodeOrder], [...doc.graph.edgeOrder]);
      } else if (!mod && key === "f") {
        event.preventDefault();
        void fitView({ padding: 0.2, duration: 300, maxZoom: 1.1 });
      } else if (!mod && key === "g" && event.shiftKey) {
        event.preventDefault();
        groupSelection();
      }
    };
    window.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [active, fitView, screenToFlowPosition]);
}

export function groupSelection(
  rectFallback?: [number, number, number, number],
): void {
  const doc = useDocStore.getState();
  const ids = useUiStore
    .getState()
    .selectedNodes.filter((id) => doc.graph.nodes[id]);
  let rect: [number, number, number, number];
  if (ids.length) {
    const xs = ids.map((id) => doc.graph.nodes[id].pos[0]);
    const ys = ids.map((id) => doc.graph.nodes[id].pos[1]);
    const minX = Math.min(...xs) - 30;
    const minY = Math.min(...ys) - 50;
    rect = [
      minX,
      minY,
      Math.max(...xs) + 260 + 30 - minX,
      Math.max(...ys) + 200 - minY,
    ];
  } else {
    rect = rectFallback ?? [0, 0, 480, 320];
  }
  doc.groupNodes(
    { id: newId("g"), label: "Group", color: "#8b5cf6", rect },
    ids,
  );
}
