// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Decides what a sidebar drop does. Pure: no DOM, no React, so the hover cue, the hint and the
// drop itself all read the same answer.

import {
  dropEdgeAt,
  insertIdAt,
  PINNED_ORDER_SCOPE,
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
  /** The last row drawn in that folder's block, where a line landing below the block goes. */
  blockEnd?: { scope: string; id: string };
  /** The section header. It stands for the top of its first row (given as `row`), so the gap
   *  above the first row is not dead space, and takes a drop while the section is collapsed. */
  header?: boolean;
}

/** The sidebar as it stands, as far as a drop needs to know. */
export interface SidebarDropContext {
  organizeBy: SidebarOrganizeBy;
  chatSort: SidebarChatSort;
  pinnedSort: SidebarChatSort;
  pinnedChatIds: ReadonlySet<string>;
  pinnedProjectIds: ReadonlySet<string>;
  /** Row ids per list, in drawn order. Pinned is one list of folders and chats. */
  orders: {
    pinned: string[];
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

/** Where a chat from another list lands: its slot against a row of that list. Kept beside the
 *  order snapshot, so a move that finishes after the list was reordered can re-aim the slot
 *  instead of writing the snapshot over the newer order. */
export interface SidebarDropPlace {
  id: string;
  targetId: string;
  edge: DropEdge;
}

/** Everything the drop changes. */
export interface SidebarDropEffects {
  orders: Array<{ scope: string; ids: string[]; place?: SidebarDropPlace }>;
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

/** The row is already in the slot under the pointer: its own row, or the edge it sits on.
 *  Nothing is painted and a drop does nothing, but the spot still claims the drag, so the
 *  section around it does not answer with its last slot instead. */
export const STAY = "stay";
export type SidebarDropOutcome = SidebarDropPlan | typeof STAY | null;

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

/** The plan for dropping `drag` on `zone`, STAY when the row is already there, or null when
 *  this spot has no answer and the zone around it may. */
export function planSidebarDrop(
  drag: SidebarDragItem,
  zone: SidebarDropZone,
  edge: DropEdge,
  ctx: SidebarDropContext,
): SidebarDropOutcome {
  if (zone.header) {
    // Pinned holds both kinds in one order; the other sections hold one each.
    const first =
      zone.row && (zone.section === "pinned" || zone.row.kind === drag.kind)
        ? zone.row
        : undefined;
    zone = {
      section: zone.section,
      header: true,
      row: first,
      folderId: first?.kind === "project" ? first.id : undefined,
    };
    edge = "top";
  }
  if (zone.section === "pinned") return planPinnedDrop(drag, zone, edge, ctx);
  return drag.kind === "project"
    ? planFolderDrop(drag, zone, edge, ctx)
    : planChatDrop(drag, zone, edge, ctx);
}

/** The line for landing against a folder: above its row, or below the last row of its block. */
function folderLine(
  scope: string,
  folderId: string,
  edge: DropEdge,
  zone: SidebarDropZone,
): SidebarDropCue {
  if (edge === "bottom" && zone.blockEnd) {
    return line(zone.blockEnd.scope, zone.blockEnd.id, "bottom");
  }
  return line(scope, folderId, edge);
}

// Pinned is one list: folders and chats in the order they were dropped into. A row of either
// kind lands in the slot the line shows. A chat still files into a pinned folder by landing on
// that folder's chats, or on the lower half of its row.
function planPinnedDrop(
  drag: SidebarDragItem,
  zone: SidebarDropZone,
  edge: DropEdge,
  ctx: SidebarDropContext,
): SidebarDropOutcome {
  if (zone.folderId === drag.id) return STAY;
  if (
    drag.kind === "chat" &&
    zone.folderId &&
    (zone.row?.kind !== "project" || edge === "bottom")
  ) {
    return planChatDrop(drag, zone, edge, ctx);
  }
  const ids = ctx.orders.pinned;
  const inList =
    drag.kind === "project"
      ? ctx.pinnedProjectIds.has(drag.id)
      : ctx.pinnedChatIds.has(drag.id);
  let target: { id: string; edge: DropEdge } | null = null;
  if (zone.row) {
    if (zone.folderId && zone.row.kind === "chat" && zone.block) {
      // A folder over another folder's chats aims at that folder; the block's middle decides.
      target = { id: zone.folderId, edge: blockEdge(zone.block, edge) };
    } else {
      target = { id: zone.row.id, edge };
    }
  } else if (zone.folderId) {
    // A folder block's own rows (Show more, or an empty folder) are its tail: below it.
    target = { id: zone.folderId, edge: "bottom" };
  } else if (ids.length > 0) {
    // The section's own space: last.
    target = { id: ids[ids.length - 1], edge: "bottom" };
  }
  if (!target) return null;
  if (target.id === drag.id) return STAY;
  const resorts = ctx.pinnedSort !== "manual";
  if (inList && resorts && !ctx.reorderSwitchesSort) return null;
  const next = inList
    ? insertIdAt(ids, drag.id, target.id, target.edge)
    : placeIdAt(ids, drag.id, target.id, target.edge);
  if (next === ids) return STAY;
  const cue =
    zone.folderId && target.id === zone.folderId
      ? folderLine(PINNED_ORDER_SCOPE, target.id, target.edge, zone)
      : line(PINNED_ORDER_SCOPE, target.id, target.edge);
  const effects: SidebarDropEffects = {
    orders: [{ scope: PINNED_ORDER_SCOPE, ids: next }],
    switchSort: resorts ? "pinned" : undefined,
  };
  if (inList) return { action: { kind: "reorder" }, cue, effects };
  if (drag.kind === "project") effects.pinProject = drag.id;
  else effects.pinChat = drag.id;
  return { action: { kind: "pin" }, cue, effects };
}

// A folder over Projects: reorders among its folders, or comes back out of Pinned. A folder's
// block (its row plus its chats) is one target, aimed at the folder.
function planFolderDrop(
  drag: SidebarDragItem,
  zone: SidebarDropZone,
  edge: DropEdge,
  ctx: SidebarDropContext,
): SidebarDropOutcome {
  if (zone.section === "recents") return null;
  if (zone.folderId === drag.id) return STAY;
  const ids = ctx.orders.projects;
  let target: { id: string; edge: DropEdge } | null = null;
  if (zone.folderId) {
    target =
      zone.block && zone.row?.kind === "chat"
        ? { id: zone.folderId, edge: blockEdge(zone.block, edge) }
        : { id: zone.folderId, edge };
  }
  if (!ctx.pinnedProjectIds.has(drag.id)) {
    // Same list: only a folder is a slot to land against.
    if (!target) return null;
    const next = insertIdAt(ids, drag.id, target.id, target.edge);
    if (next === ids) return STAY;
    return {
      action: { kind: "reorder" },
      cue: folderLine(PROJECT_ORDER_SCOPE, target.id, target.edge, zone),
      effects: { orders: [{ scope: PROJECT_ORDER_SCOPE, ids: next }] },
    };
  }
  // Out of Pinned: unpinned, landing against a folder or last.
  const next = placeIdAt(ids, drag.id, target?.id ?? null, target?.edge ?? "bottom");
  return {
    action: { kind: "unpin" },
    cue: target
      ? folderLine(PROJECT_ORDER_SCOPE, target.id, target.edge, zone)
      : ring(sectionRingKey("projects")),
    effects: {
      orders: [{ scope: PROJECT_ORDER_SCOPE, ids: next }],
      unpinProject: drag.id,
    },
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
): SidebarDropOutcome {
  const pinned = ctx.pinnedChatIds.has(drag.id);
  // Picked up from the Pinned list itself, not from inside a pinned folder.
  const fromPinnedList = drag.scope === PINNED_ORDER_SCOPE;

  if (zone.folderId) {
    const folderId = zone.folderId;
    const sameFolder = drag.projectId === folderId;
    if (fromPinnedList) {
      // A pinned folder keeps the pin; a folder under Projects takes it away.
      if (zone.section === "pinned") {
        return sameFolder ? STAY : moveChat(drag, zone, edge, ctx, folderId, false);
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
    // Its own folder's row: already filed here.
    if (zone.row?.kind !== "chat") return STAY;
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
): { cue: SidebarDropCue; order: SidebarDropEffects["orders"][number] } | null {
  if (sort !== "manual" || zone.row?.kind !== "chat" || zone.row.scope !== scope) {
    return null;
  }
  if (zone.row.id === chatId) return null;
  return {
    cue: line(scope, zone.row.id, edge),
    order: {
      scope,
      ids: placeIdAt(ids, chatId, zone.row.id, edge),
      place: { id: chatId, targetId: zone.row.id, edge },
    },
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
): SidebarDropOutcome {
  const sort = sortKey === "pinned" ? ctx.pinnedSort : ctx.chatSort;
  if (sort !== "manual" && !ctx.reorderSwitchesSort) return null;
  const next = insertIdAt(ids, drag.id, targetId, edge);
  if (next === ids) return STAY;
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
