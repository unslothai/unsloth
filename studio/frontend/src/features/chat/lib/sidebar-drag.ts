// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Pure (no DOM/React) so the hover cue, hint and drop read the same answer.

import {
  customSectionIdOf,
  customSectionScope,
  dropEdgeAt,
  insertIdAt,
  PINNED_ORDER_SCOPE,
  placeIdAt,
  PROJECT_ORDER_SCOPE,
  projectOrderScope,
  RECENTS_ORDER_SCOPE,
  type SidebarChatSort,
  type SidebarProjectSort,
  type SidebarOrganizeBy,
} from "../stores/sidebar-organization-store.ts";

/** `page` is a pinned browser page: shown in Pinned or a custom section only. */
export type SidebarRowKind = "chat" | "project" | "page";
/** A built-in list, or a custom section keyed by its order scope (`section:<id>`). */
export type SidebarSection =
  | "pinned"
  | "projects"
  | "recents"
  | `section:${string}`;
export type DropEdge = "top" | "bottom";

export interface SidebarDragItem {
  kind: SidebarRowKind;
  id: string;
  section: SidebarSection;
  scope: string;
  /** For a chat, the folder it is filed in. Null outside every folder, and for a folder or page. */
  projectId: string | null;
}

export interface SidebarDropZone {
  section: SidebarSection;
  row?: { id: string; kind: SidebarRowKind; scope: string };
  folderId?: string;
  block?: { index: number; count: number };
  blockEnd?: { scope: string; id: string };
  /** Stands for the top of its first row and takes drops while collapsed. */
  header?: boolean;
}

export interface SidebarDropContext {
  organizeBy: SidebarOrganizeBy;
  chatSort: SidebarChatSort;
  pinnedSort: SidebarChatSort;
  projectSort: SidebarProjectSort;
  pinnedChatIds: ReadonlySet<string>;
  pinnedProjectIds: ReadonlySet<string>;
  sectionByChatId: Readonly<Record<string, string>>;
  sectionByProjectId: Readonly<Record<string, string>>;
  sectionByPageId: Readonly<Record<string, string>>;
  /** A custom section's chat sort. */
  sectionSort: (sectionId: string) => SidebarChatSort;
  orders: {
    pinned: string[];
    projects: string[];
    recents: string[];
    projectChats: (projectId: string) => string[];
    sections: (sectionId: string) => string[];
  };
}

export type SidebarDropAction =
  | { kind: "reorder" }
  | { kind: "pin" }
  | { kind: "unpin" }
  | { kind: "move"; projectId: string | null }
  | { kind: "section"; sectionId: string | null };

/** Row keys are `scope:id`, since one chat can be drawn in two lists. */
export type SidebarDropCue =
  | {
      line: {
        rowKey: string;
        edge: DropEdge;
        folderId?: string;
      };
    }
  | { ring: string };

/** Kept with the order snapshot so a late move can re-aim instead of overwriting newer order. */
export interface SidebarDropPlace {
  id: string;
  targetId: string;
  edge: DropEdge;
}

export interface SidebarDropEffects {
  orders: Array<{ scope: string; ids: string[]; place?: SidebarDropPlace }>;
  pinChat?: string;
  unpinChat?: string;
  pinProject?: string;
  unpinProject?: string;
  moveChat?: { chatId: string; projectId: string | null };
  fileInSection?: { kind: SidebarRowKind; id: string; sectionId: string | null };
  /** Switch this list to Manual, or its sort undoes the drop. */
  switchSort?: "chats" | "pinned" | "projects" | `section:${string}`;
}

export interface SidebarDropPlan {
  action: SidebarDropAction;
  cue: SidebarDropCue;
  effects: SidebarDropEffects;
}

/** Already in place: nothing painted, but the spot still claims the drag. */
export const STAY = "stay";
export type SidebarDropOutcome = SidebarDropPlan | typeof STAY | null;

/** A dedicated scope, not id: any string can be a real row id, but no real scope equals this. */
export const SIDEBAR_TAIL_SCOPE = "sidebar-tail";

export const rowKey = (scope: string, id: string): string => `${scope}:${id}`;
export const sectionRingKey = (section: SidebarSection): string =>
  `section:${section}`;

export function isCustomSection(
  section: SidebarSection,
): section is `section:${string}` {
  return customSectionIdOf(section) !== null;
}

function ownSection(
  kind: SidebarRowKind,
  id: string,
  ctx: SidebarDropContext,
): string | null {
  if (kind === "project") {
    if (ctx.pinnedProjectIds.has(id)) return null;
    return ctx.sectionByProjectId[id] ?? null;
  }
  if (kind === "page") return ctx.sectionByPageId[id] ?? null;
  if (ctx.pinnedChatIds.has(id)) return null;
  return ctx.sectionByChatId[id] ?? null;
}

/** Leaving to a non-section list unfiles the row; reorders and pins keep the section. */
function leavingSection(
  drag: SidebarDragItem,
  outcome: SidebarDropOutcome,
  ctx: SidebarDropContext,
): SidebarDropOutcome {
  if (!outcome || outcome === STAY) return outcome;
  if (outcome.action.kind === "reorder" || outcome.action.kind === "pin") return outcome;
  const filed =
    drag.kind === "project"
      ? ctx.sectionByProjectId[drag.id]
      : drag.kind === "page"
        ? ctx.sectionByPageId[drag.id]
        : ctx.sectionByChatId[drag.id];
  if (!filed || outcome.effects.fileInSection) return outcome;
  return {
    ...outcome,
    effects: {
      ...outcome.effects,
      fileInSection: { kind: drag.kind, id: drag.id, sectionId: null },
    },
  };
}
export const folderRingKey = (projectId: string): string =>
  `folder:${projectId}`;

const FOLDER_SCOPE_PREFIX = projectOrderScope("");
const line = (scope: string, id: string, edge: DropEdge): SidebarDropCue => ({
  line: {
    rowKey: rowKey(scope, id),
    edge,
    ...(scope.startsWith(FOLDER_SCOPE_PREFIX)
      ? { folderId: scope.slice(FOLDER_SCOPE_PREFIX.length) }
      : {}),
  },
});
const ring = (key: string): SidebarDropCue => ({ ring: key });

export { dropEdgeAt };

/** STAY when already there; null lets the surrounding zone answer. */
export function planSidebarDrop(
  drag: SidebarDragItem,
  zone: SidebarDropZone,
  edge: DropEdge,
  ctx: SidebarDropContext,
): SidebarDropOutcome {
  if (zone.header) {
    const first =
      zone.row &&
      (zone.section === "pinned" ||
        isCustomSection(zone.section) ||
        zone.row.kind === drag.kind)
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
  if (isCustomSection(zone.section)) return planSectionDrop(drag, zone, edge, ctx);
  if (zone.section === "pinned") {
    return leavingSection(drag, planPinnedDrop(drag, zone, edge, ctx), ctx);
  }
  // Pages can't go in Projects, Recents or a folder.
  if (drag.kind === "page") return null;
  return leavingSection(
    drag,
    drag.kind === "project"
      ? planFolderDrop(drag, zone, edge, ctx)
      : planChatDrop(drag, zone, edge, ctx),
    ctx,
  );
}

// A pinned row dropped into a section loses its pin, or Pinned keeps drawing it.
function planSectionDrop(
  drag: SidebarDragItem,
  zone: SidebarDropZone,
  edge: DropEdge,
  ctx: SidebarDropContext,
): SidebarDropOutcome {
  const sectionId = customSectionIdOf(zone.section);
  if (sectionId === null) return null;
  const scope = customSectionScope(sectionId);
  if (zone.folderId === drag.id) return STAY;
  if (
    drag.kind === "chat" &&
    zone.folderId &&
    (zone.row?.kind !== "project" || edge === "bottom")
  ) {
    return leavingSection(drag, planChatDrop(drag, zone, edge, ctx), ctx);
  }
  const ids = ctx.orders.sections(sectionId);
  const inList = ownSection(drag.kind, drag.id, ctx) === sectionId;
  let target: { id: string; edge: DropEdge } | null = null;
  if (zone.row) {
    target =
      zone.folderId && zone.row.kind === "chat" && zone.block
        ? { id: zone.folderId, edge: blockEdge(zone.block, edge) }
        : { id: zone.row.id, edge };
  } else if (zone.folderId) {
    target = { id: zone.folderId, edge: "bottom" };
  } else if (ids.length > 0) {
    target = { id: ids[ids.length - 1], edge: "bottom" };
  }
  if (target?.id === drag.id) return STAY;
  const switchSort = ctx.sectionSort(sectionId) === "manual" ? undefined : scope;
  const cue = target
    ? folderLine(scope, target.id, target.edge, zone)
    : ring(sectionRingKey(scope));
  if (inList) {
    if (!target) return null;
    const next = insertIdAt(ids, drag.id, target.id, target.edge);
    if (next === ids) return STAY;
    return {
      action: { kind: "reorder" },
      cue,
      effects: { orders: [{ scope, ids: next }], switchSort },
    };
  }
  const next = placeIdAt(ids, drag.id, target?.id ?? null, target?.edge ?? "bottom");
  const effects: SidebarDropEffects = {
    orders: [{ scope, ids: next }],
    fileInSection: { kind: drag.kind, id: drag.id, sectionId },
    switchSort,
  };
  if (drag.kind === "project" && ctx.pinnedProjectIds.has(drag.id)) {
    effects.unpinProject = drag.id;
  }
  if (drag.kind === "chat" && ctx.pinnedChatIds.has(drag.id)) {
    effects.unpinChat = drag.id;
  }
  return { action: { kind: "section", sectionId }, cue, effects };
}

function folderLine(
  scope: string,
  folderId: string,
  edge: DropEdge,
  zone: SidebarDropZone,
): SidebarDropCue {
  if (edge === "bottom" && zone.blockEnd) {
    return { line: { rowKey: rowKey(zone.blockEnd.scope, zone.blockEnd.id), edge: "bottom" } };
  }
  return line(scope, folderId, edge);
}

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
  // A page is in Pinned unless filed in a section.
  const inList =
    drag.kind === "project"
      ? ctx.pinnedProjectIds.has(drag.id)
      : drag.kind === "page"
        ? ownSection("page", drag.id, ctx) === null
        : ctx.pinnedChatIds.has(drag.id);
  let target: { id: string; edge: DropEdge } | null = null;
  if (zone.row) {
    if (zone.folderId && zone.row.kind === "chat" && zone.block) {
      target = { id: zone.folderId, edge: blockEdge(zone.block, edge) };
    } else {
      target = { id: zone.row.id, edge };
    }
  } else if (zone.folderId) {
    target = { id: zone.folderId, edge: "bottom" };
  } else if (ids.length > 0) {
    target = { id: ids[ids.length - 1], edge: "bottom" };
  }
  if (target?.id === drag.id) return STAY;
  const resorts = ctx.pinnedSort !== "manual";
  // No target: Pinned is empty (shown while its pages are filed elsewhere), so this is its first row.
  const next = !target
    ? [drag.id]
    : inList
      ? insertIdAt(ids, drag.id, target.id, target.edge)
      : placeIdAt(ids, drag.id, target.id, target.edge);
  if (next === ids) return STAY;
  // folderLine is the plain line unless the zone names a block end to land below, so the section's
  // own tail draws under the last row of the folder that ends it, not under that folder's title.
  const cue = target
    ? folderLine(PINNED_ORDER_SCOPE, target.id, target.edge, zone)
    : ring(sectionRingKey("pinned"));
  const effects: SidebarDropEffects = {
    orders: [{ scope: PINNED_ORDER_SCOPE, ids: next }],
    switchSort: resorts ? "pinned" : undefined,
  };
  if (inList) return { action: { kind: "reorder" }, cue, effects };
  if (drag.kind === "page") {
    // Back to Pinned means leaving its section.
    effects.fileInSection = { kind: "page", id: drag.id, sectionId: null };
    return { action: { kind: "pin" }, cue, effects };
  }
  if (drag.kind === "project") effects.pinProject = drag.id;
  else effects.pinChat = drag.id;
  return { action: { kind: "pin" }, cue, effects };
}

function planFolderDrop(
  drag: SidebarDragItem,
  zone: SidebarDropZone,
  edge: DropEdge,
  ctx: SidebarDropContext,
): SidebarDropOutcome {
  if (zone.section === "recents") return null;
  if (zone.folderId === drag.id) return STAY;
  const ids = ctx.orders.projects;
  const switchSort = ctx.projectSort === "manual" ? undefined : "projects";
  const pinned = ctx.pinnedProjectIds.has(drag.id);
  let target: { id: string; edge: DropEdge } | null = null;
  if (zone.folderId) {
    target =
      zone.block && zone.row?.kind === "chat"
        ? { id: zone.folderId, edge: blockEdge(zone.block, edge) }
        : { id: zone.folderId, edge };
  }
  if (!pinned && ownSection("project", drag.id, ctx) === null) {
    if (!target) return null;
    const next = insertIdAt(ids, drag.id, target.id, target.edge);
    if (next === ids) return STAY;
    return {
      action: { kind: "reorder" },
      cue: folderLine(PROJECT_ORDER_SCOPE, target.id, target.edge, zone),
      effects: { orders: [{ scope: PROJECT_ORDER_SCOPE, ids: next }], switchSort },
    };
  }
  const next = placeIdAt(ids, drag.id, target?.id ?? null, target?.edge ?? "bottom");
  return {
    action: pinned ? { kind: "unpin" } : { kind: "section", sectionId: null },
    cue: target
      ? folderLine(PROJECT_ORDER_SCOPE, target.id, target.edge, zone)
      : ring(sectionRingKey("projects")),
    effects: {
      orders: [{ scope: PROJECT_ORDER_SCOPE, ids: next }],
      unpinProject: pinned ? drag.id : undefined,
      switchSort,
    },
  };
}

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
  const fromPinnedList = drag.scope === PINNED_ORDER_SCOPE;
  const inSection = ownSection("chat", drag.id, ctx) !== null;

  if (zone.folderId) {
    const folderId = zone.folderId;
    const sameFolder = drag.projectId === folderId;
    if (fromPinnedList || inSection) {
      if (!sameFolder) {
        return moveChat(
          drag,
          zone,
          edge,
          ctx,
          folderId,
          fromPinnedList && zone.section !== "pinned",
        );
      }
      const landing = landingIn(
        projectOrderScope(folderId),
        ctx.orders.projectChats(folderId),
        drag.id,
        zone,
        edge,
        ctx.chatSort,
      );
      return {
        action: fromPinnedList ? { kind: "unpin" } : { kind: "section", sectionId: null },
        cue: landing?.cue ?? ring(folderRingKey(folderId)),
        effects: {
          orders: landing ? [landing.order] : [],
          unpinChat: fromPinnedList ? drag.id : undefined,
          switchSort: landing?.resorts ? "chats" : undefined,
        },
      };
    }
    if (!sameFolder) return moveChat(drag, zone, edge, ctx, folderId, false);
    const folderScope = projectOrderScope(folderId);
    if (zone.row?.kind !== "chat") {
      // Block tail takes a drop: Show more sits right under the last visible chat.
      if (zone.row || zone.blockEnd?.scope !== folderScope) return STAY;
      return reorder(
        drag,
        folderScope,
        ctx.orders.projectChats(folderId),
        zone.blockEnd.id,
        "bottom",
        "chats",
        ctx,
      );
    }
    return reorder(
      drag,
      folderScope,
      ctx.orders.projectChats(folderId),
      zone.row.id,
      edge,
      "chats",
      ctx,
    );
  }

  if (zone.section === "recents") {
    const filed = drag.projectId !== null && ctx.organizeBy === "project";
    if (pinned || filed || inSection) {
      const landing =
        landingIn(
          RECENTS_ORDER_SCOPE,
          ctx.orders.recents,
          drag.id,
          zone,
          edge,
          ctx.chatSort,
        ) ??
        lastIn(RECENTS_ORDER_SCOPE, ctx.orders.recents, drag.id, ctx.chatSort);
      return {
        action: filed
          ? { kind: "move", projectId: null }
          : pinned
            ? { kind: "unpin" }
            : { kind: "section", sectionId: null },
        cue: landing?.cue ?? ring(sectionRingKey("recents")),
        effects: {
          orders: landing ? [landing.order] : [],
          unpinChat: pinned ? drag.id : undefined,
          moveChat: filed ? { chatId: drag.id, projectId: null } : undefined,
          switchSort: landing?.resorts ? "chats" : undefined,
        },
      };
    }
    if (zone.row?.kind !== "chat") {
      // Space beside a row stays inert so it does not send the row to the end.
      const ids = ctx.orders.recents;
      const last = ids[ids.length - 1];
      if (zone.row || zone.blockEnd?.scope !== SIDEBAR_TAIL_SCOPE || last === undefined) {
        return null;
      }
      if (last === drag.id) return STAY;
      return reorder(drag, RECENTS_ORDER_SCOPE, ids, last, "bottom", "chats", ctx);
    }
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

  return null;
}

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
      switchSort: landing?.resorts ? "chats" : undefined,
    },
  };
}

/** A sorted list switches to Manual, or the sort moves the chat again. */
function landingIn(
  scope: string,
  ids: string[],
  chatId: string,
  zone: SidebarDropZone,
  edge: DropEdge,
  sort: SidebarChatSort,
): Landing | null {
  if (zone.row?.kind !== "chat" || zone.row.scope !== scope) return null;
  return slotAt(scope, ids, chatId, zone.row.id, edge, sort);
}

interface Landing {
  cue: SidebarDropCue;
  order: SidebarDropEffects["orders"][number];
  resorts: boolean;
}

function slotAt(
  scope: string,
  ids: string[],
  chatId: string,
  targetId: string,
  edge: DropEdge,
  sort: SidebarChatSort,
): Landing | null {
  if (targetId === chatId) return null;
  return {
    cue: line(scope, targetId, edge),
    order: {
      scope,
      ids: placeIdAt(ids, chatId, targetId, edge),
      place: { id: chatId, targetId, edge },
    },
    resorts: sort !== "manual",
  };
}

function lastIn(
  scope: string,
  ids: string[],
  chatId: string,
  sort: SidebarChatSort,
): Landing | null {
  const last = ids[ids.length - 1];
  return last === undefined ? null : slotAt(scope, ids, chatId, last, "bottom", sort);
}

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

/** A filing drop lights the folder even while a line shows the slot. */
export function litRingKey(plan: SidebarDropPlan | null): string | null {
  if (!plan) return null;
  if ("ring" in plan.cue) return plan.cue.ring;
  if (plan.cue.line.folderId !== undefined) return folderRingKey(plan.cue.line.folderId);
  if (plan.action.kind === "move" && plan.action.projectId) {
    return folderRingKey(plan.action.projectId);
  }
  return null;
}

/** `place` is ignored: it names the same slot from different neighbours. */
export function equivalentDrop(a: SidebarDropPlan, b: SidebarDropPlan): boolean {
  const landing = (plan: SidebarDropPlan) =>
    JSON.stringify({
      action: plan.action,
      orders: plan.effects.orders.map(({ scope, ids }) => ({ scope, ids })),
      pinChat: plan.effects.pinChat,
      unpinChat: plan.effects.unpinChat,
      pinProject: plan.effects.pinProject,
      unpinProject: plan.effects.unpinProject,
      moveChat: plan.effects.moveChat,
      fileInSection: plan.effects.fileInSection,
      switchSort: plan.effects.switchSort,
    });
  return landing(a) === landing(b);
}

export function planKey(plan: SidebarDropPlan | null): string {
  if (!plan) return "";
  const cue =
    "line" in plan.cue
      ? `line:${plan.cue.line.rowKey}:${plan.cue.line.edge}:${plan.cue.line.folderId ?? ""}`
      : `ring:${plan.cue.ring}`;
  const action =
    plan.action.kind === "move"
      ? `move:${plan.action.projectId ?? ""}`
      : plan.action.kind === "section"
        ? `section:${plan.action.sectionId ?? ""}`
        : plan.action.kind;
  return `${action}|${cue}`;
}
