// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Decides what a sidebar drop does. Pure: no DOM, no React, so the hover cue, the hint and the
// drop itself all read the same answer.

import {
  dropEdgeAt,
  insertIdAt,
  PINNED_ORDER_SCOPE,
  PINNED_PROJECT_ORDER_SCOPE,
  placeIdAt,
  PROJECT_ORDER_SCOPE,
  projectOrderScope,
  RECENTS_ORDER_SCOPE,
  type SidebarChatSort,
  type SidebarOrganizeBy,
} from "../stores/sidebar-organization-store.ts";

export type SidebarRowKind = "chat" | "project";
export type SidebarSection = "pinned" | "projects" | "recents";
export type DropEdge = "top" | "bottom";

/** The row being carried. */
export interface SidebarDragItem {
  kind: SidebarRowKind;
  id: string;
  /** Section it was picked up from. */
  section: SidebarSection;
  /** Order scope of the list it was picked up from. */
  scope: string;
  /** For a chat, the folder it is filed in. Null outside every folder, and for a folder. */
  projectId: string | null;
}

/** A spot a row can be dropped on: a row, a folder block, a section body or a section header. */
export interface SidebarDropZone {
  section: SidebarSection;
  /** The row under the pointer, if any, with the scope it is drawn in. */
  row?: { id: string; kind: SidebarRowKind; scope: string };
  /** The folder this spot belongs to: its row, a chat under it, or its empty line. */
  folderId?: string;
  /** For a chat under a folder: its place in the folder's block. */
  block?: { index: number; count: number };
  /** The section header. It stands for the top of its first row (given as `row`) for a drag
   *  of the same kind, so the gap above the first row is not dead space, and takes a drop
   *  while the section is collapsed. */
  header?: boolean;
}

/** The sidebar as it stands, as far as a drop needs to know. */
export interface SidebarDropContext {
  organizeBy: SidebarOrganizeBy;
  chatSort: SidebarChatSort;
  pinnedSort: SidebarChatSort;
  pinnedChatIds: ReadonlySet<string>;
  pinnedProjectIds: ReadonlySet<string>;
  /** Row ids per list, in drawn order. */
  orders: {
    pinnedChats: string[];
    pinnedProjects: string[];
    projects: string[];
    recents: string[];
    projectChats: (projectId: string) => string[];
  };
  /** Whether a reorder in a Priority or Last updated list is taken (switching it to Manual)
   *  or refused. */
  reorderSwitchesSort: boolean;
}

/** What the drop does, for the hint. */
export type SidebarDropAction =
  | { kind: "reorder" }
  | { kind: "pin" }
  | { kind: "unpin" }
  | { kind: "move"; projectId: string | null };

/** What to paint: a line on one edge of a row, or a ring around a whole target. Row keys are
 *  `scope:id`, since one chat can be drawn in two lists. */
export type SidebarDropCue =
  | { line: { rowKey: string; edge: DropEdge } }
  | { ring: string };

/** Everything the drop changes. */
export interface SidebarDropEffects {
  orders: Array<{ scope: string; ids: string[] }>;
  pinChat?: string;
  unpinChat?: string;
  pinProject?: string;
  unpinProject?: string;
  moveChat?: { chatId: string; projectId: string | null };
  /** The list to switch to Manual, or its own rule undoes the drop. */
  switchSort?: "chats" | "pinned";
}

export interface SidebarDropPlan {
  action: SidebarDropAction;
  cue: SidebarDropCue;
  effects: SidebarDropEffects;
}

export const rowKey = (scope: string, id: string): string => `${scope}:${id}`;
export const sectionRingKey = (section: SidebarSection): string =>
  `section:${section}`;
export const folderRingKey = (projectId: string): string =>
  `folder:${projectId}`;

const line = (scope: string, id: string, edge: DropEdge): SidebarDropCue => ({
  line: { rowKey: rowKey(scope, id), edge },
});
const ring = (key: string): SidebarDropCue => ({ ring: key });

export { dropEdgeAt };

/** The plan for dropping `drag` on `zone`, or null when the drop would do nothing. */
export function planSidebarDrop(
  drag: SidebarDragItem,
  zone: SidebarDropZone,
  edge: DropEdge,
  ctx: SidebarDropContext,
): SidebarDropPlan | null {
  if (zone.header) {
    const first = zone.row?.kind === drag.kind ? zone.row : undefined;
    zone = {
      section: zone.section,
      header: true,
      row: first,
      folderId: first?.kind === "project" ? first.id : undefined,
    };
    edge = "top";
  }
  return drag.kind === "project"
    ? planFolderDrop(drag, zone, edge, ctx)
    : planChatDrop(drag, zone, edge, ctx);
}

// Folders reorder within Projects or Pinned, and cross between them to pin or unpin. A folder's
// block (its row plus its chats) is one target, aimed at the folder.
function planFolderDrop(
  drag: SidebarDragItem,
  zone: SidebarDropZone,
  edge: DropEdge,
  ctx: SidebarDropContext,
): SidebarDropPlan | null {
  if (zone.section === "recents") return null;
  if (zone.folderId === drag.id) return null;
  const toPinned = zone.section === "pinned";
  const fromPinned = ctx.pinnedProjectIds.has(drag.id);
  const scope = toPinned ? PINNED_PROJECT_ORDER_SCOPE : PROJECT_ORDER_SCOPE;
  const ids = toPinned ? ctx.orders.pinnedProjects : ctx.orders.projects;
  let target: { id: string; edge: DropEdge } | null = null;
  if (zone.folderId) {
    target =
      zone.block && zone.row?.kind === "chat"
        ? { id: zone.folderId, edge: blockEdge(zone.block, edge) }
        : { id: zone.folderId, edge };
  }
  if (fromPinned === toPinned) {
    // Same list: only a folder is a slot to land against.
    if (!target) return null;
    const next = insertIdAt(ids, drag.id, target.id, target.edge);
    if (next === ids) return null;
    return {
      action: { kind: "reorder" },
      cue: line(scope, target.id, target.edge),
      effects: { orders: [{ scope, ids: next }] },
    };
  }
  // Crossing lists pins or unpins, landing against a folder or last.
  const next = placeIdAt(ids, drag.id, target?.id ?? null, target?.edge ?? "bottom");
  const cue = target
    ? line(scope, target.id, target.edge)
    : ring(sectionRingKey(zone.section));
  return toPinned
    ? {
        action: { kind: "pin" },
        cue,
        effects: { orders: [{ scope, ids: next }], pinProject: drag.id },
      }
    : {
        action: { kind: "unpin" },
        cue,
        effects: { orders: [{ scope, ids: next }], unpinProject: drag.id },
      };
}

/** Which end of a folder the pointer aims at from inside its block: the line flips once, at
 *  the block's middle. */
function blockEdge(block: { index: number; count: number }, edge: DropEdge): DropEdge {
  const at = block.index + (edge === "bottom" ? 1 : 0);
  return at * 2 >= block.count ? "bottom" : "top";
}

function planChatDrop(
  drag: SidebarDragItem,
  zone: SidebarDropZone,
  edge: DropEdge,
  ctx: SidebarDropContext,
): SidebarDropPlan | null {
  const pinned = ctx.pinnedChatIds.has(drag.id);
  // Picked up from the Pinned list itself, not from inside a pinned folder.
  const fromPinnedList = drag.scope === PINNED_ORDER_SCOPE;

  if (zone.folderId) {
    const folderId = zone.folderId;
    const sameFolder = drag.projectId === folderId;
    if (fromPinnedList) {
      // A pinned folder keeps the pin; a folder under Projects takes it away.
      if (zone.section === "pinned") {
        return sameFolder ? null : moveChat(drag, zone, edge, ctx, folderId, false);
      }
      if (!sameFolder) return moveChat(drag, zone, edge, ctx, folderId, true);
      const landing = landingIn(
        projectOrderScope(folderId),
        ctx.orders.projectChats(folderId),
        drag.id,
        zone,
        edge,
        ctx.chatSort,
      );
      return {
        action: { kind: "unpin" },
        cue: landing?.cue ?? ring(folderRingKey(folderId)),
        effects: {
          orders: landing ? [landing.order] : [],
          unpinChat: drag.id,
        },
      };
    }
    if (!sameFolder) return moveChat(drag, zone, edge, ctx, folderId, false);
    if (zone.row?.kind !== "chat") return null;
    return reorder(
      drag,
      projectOrderScope(folderId),
      ctx.orders.projectChats(folderId),
      zone.row.id,
      edge,
      "chats",
      ctx,
    );
  }

  if (zone.section === "pinned") {
    if (pinned) {
      if (zone.row?.kind !== "chat") return null;
      return reorder(
        drag,
        PINNED_ORDER_SCOPE,
        ctx.orders.pinnedChats,
        zone.row.id,
        edge,
        "pinned",
        ctx,
      );
    }
    // Pin in the slot it dropped on, or last.
    const target = zone.row?.kind === "chat" ? { id: zone.row.id, edge } : null;
    const next = placeIdAt(
      ctx.orders.pinnedChats,
      drag.id,
      target?.id ?? null,
      target?.edge ?? "bottom",
    );
    return {
      action: { kind: "pin" },
      cue: target
        ? line(PINNED_ORDER_SCOPE, target.id, target.edge)
        : ring(sectionRingKey("pinned")),
      effects: {
        orders: [{ scope: PINNED_ORDER_SCOPE, ids: next }],
        pinChat: drag.id,
        switchSort: ctx.pinnedSort === "manual" ? undefined : "pinned",
      },
    };
  }

  if (zone.section === "recents") {
    // Recents holds chats that are neither pinned nor filed, so landing here takes both away.
    // With folders off every chat is a Recents row, and only the pin goes.
    const filed = drag.projectId !== null && ctx.organizeBy === "project";
    if (pinned || filed) {
      const landing = landingIn(
        RECENTS_ORDER_SCOPE,
        ctx.orders.recents,
        drag.id,
        zone,
        edge,
        ctx.chatSort,
      );
      return {
        action: filed ? { kind: "move", projectId: null } : { kind: "unpin" },
        cue: landing?.cue ?? ring(sectionRingKey("recents")),
        effects: {
          orders: landing ? [landing.order] : [],
          unpinChat: pinned ? drag.id : undefined,
          moveChat: filed ? { chatId: drag.id, projectId: null } : undefined,
        },
      };
    }
    if (zone.row?.kind !== "chat") return null;
    return reorder(
      drag,
      RECENTS_ORDER_SCOPE,
      ctx.orders.recents,
      zone.row.id,
      edge,
      "chats",
      ctx,
    );
  }

  // Projects outside any folder has nothing to file a chat into.
  return null;
}

/** Files a chat into `projectId`, in a slot when the folder is on Manual order. */
function moveChat(
  drag: SidebarDragItem,
  zone: SidebarDropZone,
  edge: DropEdge,
  ctx: SidebarDropContext,
  projectId: string,
  unpin: boolean,
): SidebarDropPlan {
  const landing = landingIn(
    projectOrderScope(projectId),
    ctx.orders.projectChats(projectId),
    drag.id,
    zone,
    edge,
    ctx.chatSort,
  );
  return {
    action: { kind: "move", projectId },
    cue: landing?.cue ?? ring(folderRingKey(projectId)),
    effects: {
      orders: landing ? [landing.order] : [],
      moveChat: { chatId: drag.id, projectId },
      unpinChat: unpin ? drag.id : undefined,
    },
  };
}

/** Where a chat from another list lands here. Only a list on Manual order has slots; the
 *  caller lights the whole target otherwise. */
function landingIn(
  scope: string,
  ids: string[],
  chatId: string,
  zone: SidebarDropZone,
  edge: DropEdge,
  sort: SidebarChatSort,
): { cue: SidebarDropCue; order: { scope: string; ids: string[] } } | null {
  if (sort !== "manual" || zone.row?.kind !== "chat" || zone.row.scope !== scope) {
    return null;
  }
  if (zone.row.id === chatId) return null;
  return {
    cue: line(scope, zone.row.id, edge),
    order: { scope, ids: placeIdAt(ids, chatId, zone.row.id, edge) },
  };
}

/** Reorders a chat within its list. A sorted list switches to Manual, or refuses when the
 *  user turned that off. */
function reorder(
  drag: SidebarDragItem,
  scope: string,
  ids: string[],
  targetId: string,
  edge: DropEdge,
  sortKey: "chats" | "pinned",
  ctx: SidebarDropContext,
): SidebarDropPlan | null {
  const sort = sortKey === "pinned" ? ctx.pinnedSort : ctx.chatSort;
  if (sort !== "manual" && !ctx.reorderSwitchesSort) return null;
  const next = insertIdAt(ids, drag.id, targetId, edge);
  if (next === ids) return null;
  return {
    action: { kind: "reorder" },
    cue: line(scope, targetId, edge),
    effects: {
      orders: [{ scope, ids: next }],
      switchSort: sort === "manual" ? undefined : sortKey,
    },
  };
}

/** Changes exactly when the painted state does, so equal plans skip a re-render. */
export function planKey(plan: SidebarDropPlan | null): string {
  if (!plan) return "";
  const cue =
    "line" in plan.cue
      ? `line:${plan.cue.line.rowKey}:${plan.cue.line.edge}`
      : `ring:${plan.cue.ring}`;
  const action =
    plan.action.kind === "move"
      ? `move:${plan.action.projectId ?? ""}`
      : plan.action.kind;
  return `${action}|${cue}`;
}
