// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";
import type { NodeCatalog, NodeCategory, NodeSpec, PortSpec } from "../types";

export interface CatalogState {
  catalog: NodeCatalog | null;
  specs: Record<string, NodeSpec>;
  categories: Record<string, NodeCategory>;
  setCatalog: (catalog: NodeCatalog) => void;
}

export const useCatalogStore = create<CatalogState>((set) => ({
  catalog: null,
  specs: {},
  categories: {},
  setCatalog: (catalog) =>
    set({
      catalog,
      specs: Object.fromEntries(catalog.nodes.map((spec) => [spec.type, spec])),
      categories: Object.fromEntries(catalog.categories.map((c) => [c.id, c])),
    }),
}));

export function getSpec(type: string): NodeSpec | undefined {
  return useCatalogStore.getState().specs[type];
}

/** True when this Studio does not know the block, or the block comes from a newer revision of it. */
export function needsNewerStudio(node: { type: string; v?: number }): boolean {
  const spec = getSpec(node.type);
  return !spec || (node.v ?? 1) > spec.v;
}

export function portColor(type: string): string {
  return useCatalogStore.getState().catalog?.port_colors[type] ?? "#cbd5e1";
}

export function categoryColor(category: string): string {
  return useCatalogStore.getState().categories[category]?.color ?? "#64748b";
}

export function findPort(
  spec: NodeSpec | undefined,
  side: "in" | "out",
  portId: string,
): PortSpec | undefined {
  if (!spec) return undefined;
  return (side === "in" ? spec.inputs : spec.outputs).find(
    (p) => p.id === portId,
  );
}

/** Params with every spec default applied: what a freshly dropped node carries. */
export function defaultParams(spec: NodeSpec): Record<string, unknown> {
  const params: Record<string, unknown> = {};
  for (const param of spec.params) {
    params[param.id] = structuredCloneSafe(param.default);
  }
  return params;
}

function structuredCloneSafe<T>(value: T): T {
  if (value === null || typeof value !== "object") return value;
  return JSON.parse(JSON.stringify(value)) as T;
}
