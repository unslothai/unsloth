// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Chats and folders drag in lists of their own kind: a folder dropped among chats has no slot. */
export type SidebarRowKind = "chat" | "project";

/** The row a sidebar drag is carrying: which list it came from, and the folder it was in. */
export interface SidebarDragSource {
  id: string;
  scope: string;
  kind: SidebarRowKind;
  projectId: string | null;
}

// The document has one drag at a time, and a dragover can fire in the same frame as the dragstart
// that began it. A handler reading React state there sees nothing yet, and a drop refused on that
// frame is a drag that did nothing, so the source is held here and read synchronously. The
// sidebar keeps the same row in state as well, for what has to be painted.
let source: SidebarDragSource | null = null;

export function setSidebarDragSource(next: SidebarDragSource | null): void {
  source = next;
}

export function sidebarDragSource(): SidebarDragSource | null {
  return source;
}
