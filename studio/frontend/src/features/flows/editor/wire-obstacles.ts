// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Notes are annotation: wires run under them, and moving or resizing one never reroutes. */
export function isWireObstacle(type: string | undefined): boolean {
  return type !== "flowNote";
}

/** Changes only when something the router reads changes: a non-note node's box or visibility. */
export function routingKey(
  nodes: readonly {
    id: string;
    type?: string;
    position: { x: number; y: number };
    measured?: { width?: number; height?: number };
    hidden?: boolean;
    parentId?: string;
  }[],
): string {
  let key = "";
  for (const n of nodes) {
    if (!isWireObstacle(n.type)) continue;
    key += `${n.id}:${n.position.x},${n.position.y},${n.measured?.width ?? 0},${n.measured?.height ?? 0},${n.hidden ? 1 : 0},${n.parentId ?? ""};`;
  }
  return key;
}
