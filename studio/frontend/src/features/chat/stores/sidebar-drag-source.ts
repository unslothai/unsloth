// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import type { SidebarDragItem } from "../lib/sidebar-drag.ts";

export type { SidebarDragItem as SidebarDragSource } from "../lib/sidebar-drag.ts";

// A dragover can fire in the same frame as its dragstart, before React state commits, so the
// dragged row is held here and read synchronously. State holds a copy for painting only.
let source: SidebarDragItem | null = null;

export function setSidebarDragSource(next: SidebarDragItem | null): void {
  source = next;
}

export function sidebarDragSource(): SidebarDragItem | null {
  return source;
}
export type { SidebarRowKind } from "../lib/sidebar-drag.ts";
