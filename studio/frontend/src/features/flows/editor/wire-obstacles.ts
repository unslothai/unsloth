// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Notes are annotation: wires run under them, and moving or resizing one never reroutes. */
export function isWireObstacle(type: string | undefined): boolean {
  return type !== "flowNote";
}

type RoutingNode = {
  id: string;
  type?: string;
  position: { x: number; y: number };
  measured?: { width?: number; height?: number };
  hidden?: boolean;
  parentId?: string;
  data?: unknown;
};

const FIELDS = 8;

/**
 * A counter that moves only when something the router reads changes: a non-note node's box or
 * visibility, or the direction of a reroute, which a turn can flip without changing its box.
 * It runs on every canvas store update (each pan frame), so it compares in place and builds nothing.
 */
export function routingVersion(): (nodes: Iterable<RoutingNode>) => number {
  let version = 0;
  const seen: RoutingNode[] = [];
  const fields: unknown[] = [];
  return (nodes) => {
    let changed = false;
    let k = 0;
    for (const n of nodes) {
      if (!isWireObstacle(n.type)) continue;
      if (seen[k] !== n) {
        seen[k] = n;
        const at = k * FIELDS;
        const next = [
          n.id,
          n.position.x,
          n.position.y,
          n.measured?.width ?? 0,
          n.measured?.height ?? 0,
          n.hidden ? 1 : 0,
          n.parentId ?? "",
          (n.data as { dir?: string } | undefined)?.dir ?? "",
        ];
        for (let f = 0; f < FIELDS; f += 1) {
          if (fields[at + f] !== next[f]) {
            fields[at + f] = next[f];
            changed = true;
          }
        }
      }
      k += 1;
    }
    if (k !== seen.length) {
      seen.length = k;
      fields.length = k * FIELDS;
      changed = true;
    }
    if (changed) version += 1;
    return version;
  };
}
