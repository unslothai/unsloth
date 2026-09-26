// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Custom sidebar sections: user-named lists that hold chats and project folders. The store keeps
// them, the drop planner files rows in and out of them, and the sidebar draws them between Pinned
// and Projects with the same header menu every other list carries.

import assert from "node:assert/strict";
import test from "node:test";

import {
  planSidebarDrop,
  sectionRingKey,
  STAY,
  type SidebarDragItem,
  type SidebarDropContext,
  type SidebarDropOutcome,
  type SidebarDropPlan,
  type SidebarDropZone,
} from "../src/features/chat/lib/sidebar-drag.ts";
import {
  customSectionIdOf,
  customSectionScope,
  mergePersistedOrganization,
  normalizeSectionName,
  PINNED_ORDER_SCOPE,
  projectOrderScope,
  RECENTS_ORDER_SCOPE,
  PROJECT_ORDER_SCOPE,
  PINNED_SECTION_KEY,
  PROJECTS_SECTION_KEY,
  resolveSectionOrder,
  SIDEBAR_ORGANIZATION_STORAGE_KEY,
  useSidebarOrganizationStore,
} from "../src/features/chat/stores/sidebar-organization-store.ts";
import { readSrcAsync } from "./helpers/kit.ts";

const APP_SIDEBAR = await readSrcAsync("components/app-sidebar.tsx");
const EN = await readSrcAsync("i18n/locales/en.ts");

function plannedDrop(outcome: SidebarDropOutcome): SidebarDropPlan {
  assert.ok(
    outcome !== null && outcome !== STAY,
    `expected a drop plan, got ${String(outcome)}`,
  );
  return outcome;
}

function resetStore() {
  const initial = useSidebarOrganizationStore.getInitialState();
  useSidebarOrganizationStore.setState({
    customSections: initial.customSections,
    sectionByChatId: initial.sectionByChatId,
    sectionByProjectId: initial.sectionByProjectId,
    hiddenSections: initial.hiddenSections,
    sectionOrder: initial.sectionOrder,
    pendingNewChatSection: null,
    manualOrder: {},
  });
}

// ---------------------------------------------------------------------------------------------
// Store

test("a new sidebar has no custom sections and hides nothing", () => {
  const fresh = useSidebarOrganizationStore.getInitialState();
  assert.deepEqual(fresh.customSections, []);
  assert.deepEqual(fresh.sectionByChatId, {});
  assert.deepEqual(fresh.sectionByProjectId, {});
  assert.deepEqual(fresh.hiddenSections, []);
});

test("a section is created newest first, named, and manual by default", () => {
  resetStore();
  const store = useSidebarOrganizationStore.getState();
  const first = store.createCustomSection("  Research  ");
  const second = store.createCustomSection("Work\n  stuff");
  assert.ok(first && second && first !== second);
  const { customSections } = useSidebarOrganizationStore.getState();
  assert.deepEqual(
    customSections.map((section) => [section.name, section.sort]),
    [
      ["Work stuff", "manual"],
      ["Research", "manual"],
    ],
  );
});

test("a blank name makes no section and renames nothing", () => {
  resetStore();
  const store = useSidebarOrganizationStore.getState();
  assert.equal(store.createCustomSection("   "), null);
  const id = store.createCustomSection("Keep")!;
  store.renameCustomSection(id, " \t ");
  assert.equal(useSidebarOrganizationStore.getState().customSections[0].name, "Keep");
  store.renameCustomSection(id, "Kept");
  assert.equal(useSidebarOrganizationStore.getState().customSections[0].name, "Kept");
  assert.equal(normalizeSectionName("x".repeat(200)).length, 60);
});

test("rows file into a section, move between sections, and come back out", () => {
  resetStore();
  const store = useSidebarOrganizationStore.getState();
  const a = store.createCustomSection("A")!;
  const b = store.createCustomSection("B")!;
  store.setChatsSection(["c1", "c2"], a);
  store.setProjectsSection(["p1"], a);
  store.setChatsSection(["c2"], b);
  let state = useSidebarOrganizationStore.getState();
  assert.deepEqual(state.sectionByChatId, { c1: a, c2: b });
  assert.deepEqual(state.sectionByProjectId, { p1: a });
  store.setChatsSection(["c1"], null);
  state = useSidebarOrganizationStore.getState();
  assert.deepEqual(state.sectionByChatId, { c2: b });
  // Filing into a section that does not exist is refused rather than stranding the row.
  store.setChatsSection(["c3"], "missing");
  assert.equal(useSidebarOrganizationStore.getState().sectionByChatId.c3, undefined);
});

test("deleting a section returns its rows and forgets its order and visibility", () => {
  resetStore();
  const store = useSidebarOrganizationStore.getState();
  const gone = store.createCustomSection("Gone")!;
  const kept = store.createCustomSection("Kept")!;
  store.setChatsSection(["c1"], gone);
  store.setChatsSection(["c2"], kept);
  store.setProjectsSection(["p1"], gone);
  store.setManualOrder(customSectionScope(gone), ["c1", "p1"]);
  store.setSectionHidden(gone, true);
  store.deleteCustomSection(gone);
  const state = useSidebarOrganizationStore.getState();
  assert.deepEqual(state.customSections.map((section) => section.id), [kept]);
  assert.deepEqual(state.sectionByChatId, { c2: kept });
  assert.deepEqual(state.sectionByProjectId, {});
  assert.equal(state.manualOrder[customSectionScope(gone)], undefined);
  assert.ok(!state.hiddenSections.includes(gone));
});

test("a section moves by dragging alone, and new sections open among the user's own", () => {
  resetStore();
  const store = useSidebarOrganizationStore.getState();
  const c = store.createCustomSection("C")!;
  const b = store.createCustomSection("B")!;
  const a = store.createCustomSection("A")!;
  const order = () => {
    const state = useSidebarOrganizationStore.getState();
    return resolveSectionOrder(state.sectionOrder, state.customSections);
  };
  const P = PINNED_SECTION_KEY;
  const J = PROJECTS_SECTION_KEY;
  // Newest first, between Pinned and Projects.
  assert.deepEqual(order(), [P, a, b, c, J]);
  store.moveSection(c, P, "top");
  assert.deepEqual(order(), [c, P, a, b, J]);
  // The list every menu reads follows the drawn order.
  assert.deepEqual(
    useSidebarOrganizationStore.getState().customSections.map((section) => section.id),
    [c, a, b],
  );
  // No step-by-step move is left: the header drag is the one way to reorder.
  assert.equal("moveCustomSection" in useSidebarOrganizationStore.getState(), false);
  assert.doesNotMatch(APP_SIDEBAR, /\bmoveSectionUp\b|\bmoveSectionDown\b|\bmoveCustomSection\b/);
});

test("no sidebar list sorts by priority", () => {
  assert.doesNotMatch(APP_SIDEBAR, /"priority"|PINNED_SORT_OPTIONS/);
  assert.match(
    APP_SIDEBAR,
    /\}> = \[\n  \{ value: "updated", key: "shell\.organize\.lastUpdated" \},\n  \{ value: "manual", key: "shell\.organize\.manualOrder" \},\n\];/,
  );
  // A saved Priority falls back to each list's default.
  const merged = mergePersistedOrganization(
    {
      pinnedSort: "priority",
      chatSort: "priority",
      customSections: [{ id: "s", name: "S", sort: "priority" }],
    },
    useSidebarOrganizationStore.getInitialState(),
  );
  assert.equal(merged.pinnedSort, "manual");
  assert.equal(merged.chatSort, "updated");
  assert.equal(merged.customSections[0]?.sort, "manual");
});

test("a dragged section lands on the edge it was dropped against", () => {
  resetStore();
  const store = useSidebarOrganizationStore.getState();
  const s = store.createCustomSection("S")!;
  const P = PINNED_SECTION_KEY;
  const J = PROJECTS_SECTION_KEY;
  const order = () => {
    const state = useSidebarOrganizationStore.getState();
    return resolveSectionOrder(state.sectionOrder, state.customSections);
  };
  assert.deepEqual(order(), [P, s, J]);
  // Projects above Pinned, then Pinned under everything.
  store.moveSection(J, P, "top");
  assert.deepEqual(order(), [J, P, s]);
  store.moveSection(P, s, "bottom");
  assert.deepEqual(order(), [J, s, P]);
  // A drop that changes nothing leaves the state alone.
  const before = useSidebarOrganizationStore.getState();
  store.moveSection(s, P, "top");
  assert.equal(useSidebarOrganizationStore.getState(), before);
  // A new section still opens among the user's own; a deleted one leaves the order.
  const t2 = store.createCustomSection("T")!;
  assert.deepEqual(order(), [J, t2, s, P]);
  store.deleteCustomSection(s);
  assert.deepEqual(order(), [J, t2, P]);
});

test("a saved order is read over the sections that exist now", () => {
  const sections = [
    { id: "a", name: "A", sort: "manual" as const },
    { id: "b", name: "B", sort: "manual" as const },
  ];
  // Unknown keys and repeats go; missing ones take their default place.
  assert.deepEqual(resolveSectionOrder(["b", "gone", PROJECTS_SECTION_KEY, "b"], sections), [
    PINNED_SECTION_KEY,
    "b",
    "a",
    PROJECTS_SECTION_KEY,
  ]);
  assert.deepEqual(resolveSectionOrder([], []), [PINNED_SECTION_KEY, PROJECTS_SECTION_KEY]);
  // Recents has no place in it: it is always last.
  assert.ok(!resolveSectionOrder(["recents"], sections).includes("recents"));
  const merged = mergePersistedOrganization(
    { customSections: sections, sectionOrder: [PROJECTS_SECTION_KEY, "a", 7, "zzz"] },
    useSidebarOrganizationStore.getInitialState(),
  );
  assert.deepEqual(merged.sectionOrder, [PINNED_SECTION_KEY, "b", PROJECTS_SECTION_KEY, "a"]);
});

test("Show toggles hide Projects and custom sections independently, never Pinned", () => {
  resetStore();
  const store = useSidebarOrganizationStore.getState();
  const id = store.createCustomSection("S")!;
  store.setSectionHidden(id, true);
  store.setSectionHidden(id, true);
  assert.deepEqual(useSidebarOrganizationStore.getState().hiddenSections, [id]);
  store.setSectionHidden(PROJECTS_SECTION_KEY, true);
  store.setSectionHidden(id, false);
  assert.deepEqual(useSidebarOrganizationStore.getState().hiddenSections, [
    PROJECTS_SECTION_KEY,
  ]);
  // Pinned always shows: a "hide Pinned" saved by an earlier build is dropped on load.
  const merged = mergePersistedOrganization(
    { hiddenSections: ["pinned", PROJECTS_SECTION_KEY] },
    useSidebarOrganizationStore.getState(),
  );
  assert.deepEqual(merged.hiddenSections, [PROJECTS_SECTION_KEY]);
});

test("a saved payload keeps only well-formed sections and assignments to them", () => {
  const merged = mergePersistedOrganization(
    {
      customSections: [
        { id: "s1", name: " Reading ", sort: "updated" },
        { id: "s1", name: "Duplicate" },
        { id: "s2", name: "   " },
        { id: 7, name: "Bad id" },
        { id: "s3", name: "Loose", sort: "sideways" },
        null,
      ],
      sectionByChatId: { c1: "s1", c2: "s2", c3: 4 },
      sectionByProjectId: { p1: "s3", p2: "nope" },
      hiddenSections: ["pinned", "s3", "s3", "recents", "nope", 1],
    },
    useSidebarOrganizationStore.getInitialState(),
  );
  assert.deepEqual(merged.customSections, [
    { id: "s1", name: "Reading", sort: "updated" },
    { id: "s3", name: "Loose", sort: "manual" },
  ]);
  assert.deepEqual(merged.sectionByChatId, { c1: "s1" });
  assert.deepEqual(merged.sectionByProjectId, { p1: "s3" });
  // Pinned always shows, so an old "hide Pinned" is dropped with the rest.
  assert.deepEqual(merged.hiddenSections, ["s3"]);
  // Reset-all already clears the key everything above lives under.
  assert.equal(SIDEBAR_ORGANIZATION_STORAGE_KEY, "unsloth_sidebar_organization");
});

test("a section's scope cannot be mistaken for a built-in list", () => {
  assert.equal(customSectionIdOf(customSectionScope("abc")), "abc");
  for (const scope of [
    RECENTS_ORDER_SCOPE,
    PINNED_ORDER_SCOPE,
    PROJECT_ORDER_SCOPE,
    projectOrderScope("section:abc"),
  ]) {
    assert.equal(customSectionIdOf(scope), null, scope);
  }
});

// ---------------------------------------------------------------------------------------------
// Drop planner

// Section S holds folder "lab" and chat s1; chat p1 is pinned; r1 and r2 are Recents rows;
// folder "home" is under Projects with chat c3.
const S = "s";
const SCOPE = customSectionScope(S);

function context(overrides: Partial<SidebarDropContext> = {}): SidebarDropContext {
  return {
    organizeBy: "project",
    chatSort: "updated",
    pinnedSort: "manual",
    projectSort: "manual",
    pinnedChatIds: new Set(["p1"]),
    pinnedProjectIds: new Set(),
    sectionByChatId: { s1: S },
    sectionByProjectId: { lab: S },
    sectionSort: () => "manual",
    orders: {
      pinned: ["p1"],
      projects: ["home"],
      recents: ["r1", "r2"],
      projectChats: (projectId) => ({ home: ["c3"], lab: ["l1"] })[projectId] ?? [],
      sections: (sectionId) => (sectionId === S ? ["lab", "s1"] : []),
    },
    ...overrides,
  };
}

const chat = (
  id: string,
  section: SidebarDragItem["section"],
  scope: string,
  projectId: string | null = null,
): SidebarDragItem => ({ kind: "chat", id, section, scope, projectId });
const chatRow = (
  section: SidebarDropZone["section"],
  scope: string,
  id: string,
  folderId?: string,
): SidebarDropZone => ({ section, row: { id, kind: "chat", scope }, folderId });

test("a Recents chat dropped on a section is filed there, in the slot shown", () => {
  const plan = plannedDrop(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE),
      chatRow(SCOPE, SCOPE, "s1"),
      "top",
      context(),
    ),
  );
  assert.deepEqual(plan.action, { kind: "section", sectionId: S });
  assert.deepEqual(plan.effects.fileInSection, { kind: "chat", id: "r1", sectionId: S });
  assert.deepEqual(plan.effects.orders, [{ scope: SCOPE, ids: ["lab", "r1", "s1"] }]);
  assert.equal(plan.effects.unpinChat, undefined);
});

test("a pinned chat dropped on a section loses its pin", () => {
  const plan = plannedDrop(
    planSidebarDrop(
      chat("p1", "pinned", PINNED_ORDER_SCOPE),
      { section: SCOPE },
      "bottom",
      context(),
    ),
  );
  assert.equal(plan.effects.unpinChat, "p1");
  assert.deepEqual(plan.effects.fileInSection, { kind: "chat", id: "p1", sectionId: S });
  assert.deepEqual(plan.effects.orders[0].ids, ["lab", "s1", "p1"]);
});

test("an empty section takes a drop anywhere in it, and rings itself", () => {
  const empty = context({ sectionByChatId: {}, sectionByProjectId: {} });
  const plan = plannedDrop(
    planSidebarDrop(chat("r2", "recents", RECENTS_ORDER_SCOPE), { section: SCOPE }, "top", {
      ...empty,
      orders: { ...empty.orders, sections: () => [] },
    }),
  );
  assert.deepEqual(plan.cue, { ring: sectionRingKey(SCOPE) });
  assert.deepEqual(plan.effects.orders, [{ scope: SCOPE, ids: ["r2"] }]);
});

test("a folder dropped on a section is filed with its chats", () => {
  const plan = plannedDrop(
    planSidebarDrop(
      { kind: "project", id: "home", section: "projects", scope: PROJECT_ORDER_SCOPE, projectId: null },
      chatRow(SCOPE, SCOPE, "s1"),
      "bottom",
      context(),
    ),
  );
  assert.deepEqual(plan.effects.fileInSection, {
    kind: "project",
    id: "home",
    sectionId: S,
  });
  assert.deepEqual(plan.effects.orders[0].ids, ["lab", "s1", "home"]);
});

test("a row already in the section reorders, switching a sorted section to Manual", () => {
  const plan = plannedDrop(
    planSidebarDrop(
      chat("s1", SCOPE, SCOPE),
      { section: SCOPE, row: { id: "lab", kind: "project", scope: SCOPE }, folderId: "lab" },
      "top",
      context({ sectionSort: () => "updated" }),
    ),
  );
  assert.equal(plan.action.kind, "reorder");
  assert.equal(plan.effects.fileInSection, undefined);
  assert.deepEqual(plan.effects.orders, [{ scope: SCOPE, ids: ["s1", "lab"] }]);
  assert.equal(plan.effects.switchSort, SCOPE);
});

test("a section's chat dragged to Recents leaves the section", () => {
  const plan = plannedDrop(
    planSidebarDrop(
      chat("s1", SCOPE, SCOPE),
      chatRow("recents", RECENTS_ORDER_SCOPE, "r1"),
      "bottom",
      context(),
    ),
  );
  assert.deepEqual(plan.action, { kind: "section", sectionId: null });
  assert.deepEqual(plan.effects.fileInSection, { kind: "chat", id: "s1", sectionId: null });
  assert.deepEqual(plan.effects.orders[0].ids, ["r1", "s1", "r2"]);
});

test("a section's chat dragged into a folder moves there and leaves the section", () => {
  const plan = plannedDrop(
    planSidebarDrop(
      chat("s1", SCOPE, SCOPE),
      chatRow("projects", projectOrderScope("home"), "c3", "home"),
      "bottom",
      context(),
    ),
  );
  assert.deepEqual(plan.effects.moveChat, { chatId: "s1", projectId: "home" });
  assert.deepEqual(plan.effects.fileInSection, { kind: "chat", id: "s1", sectionId: null });
  assert.equal(plan.effects.unpinChat, undefined);
});

test("a section's chat dropped back on its own folder only leaves the section", () => {
  const plan = plannedDrop(
    planSidebarDrop(
      chat("s1", SCOPE, SCOPE, "home"),
      chatRow("projects", projectOrderScope("home"), "c3", "home"),
      "top",
      context(),
    ),
  );
  assert.deepEqual(plan.action, { kind: "section", sectionId: null });
  assert.equal(plan.effects.moveChat, undefined);
  assert.equal(plan.effects.unpinChat, undefined);
  assert.deepEqual(plan.effects.fileInSection, { kind: "chat", id: "s1", sectionId: null });
});

test("a section's folder dragged to Projects leaves the section", () => {
  const plan = plannedDrop(
    planSidebarDrop(
      { kind: "project", id: "lab", section: SCOPE, scope: SCOPE, projectId: null },
      { section: "projects", row: { id: "home", kind: "project", scope: PROJECT_ORDER_SCOPE }, folderId: "home" },
      "bottom",
      context(),
    ),
  );
  assert.deepEqual(plan.action, { kind: "section", sectionId: null });
  assert.equal(plan.effects.unpinProject, undefined);
  assert.deepEqual(plan.effects.fileInSection, { kind: "project", id: "lab", sectionId: null });
  assert.deepEqual(plan.effects.orders[0].ids, ["home", "lab"]);
});

test("a chat dropped on a section's folder files into the project, not the section", () => {
  const plan = plannedDrop(
    planSidebarDrop(
      chat("r1", "recents", RECENTS_ORDER_SCOPE),
      chatRow(SCOPE, projectOrderScope("lab"), "l1", "lab"),
      "bottom",
      context(),
    ),
  );
  assert.deepEqual(plan.effects.moveChat, { chatId: "r1", projectId: "lab" });
  assert.equal(plan.effects.fileInSection, undefined);
});

test("reordering Pinned keeps a pinned row's section for when it is unpinned", () => {
  const ctx = context({
    pinnedChatIds: new Set(["p1", "s1"]),
    orders: { ...context().orders, pinned: ["p1", "s1"] },
  });
  const plan = plannedDrop(
    planSidebarDrop(
      chat("s1", "pinned", PINNED_ORDER_SCOPE),
      chatRow("pinned", PINNED_ORDER_SCOPE, "p1"),
      "top",
      ctx,
    ),
  );
  assert.equal(plan.action.kind, "reorder");
  assert.equal(plan.effects.fileInSection, undefined);
});

// ---------------------------------------------------------------------------------------------
// Sidebar wiring

test("each list header's menu carries only what that list is about, as in ChatGPT", () => {
  const menu = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("function renderSidebarHeaderMenu("),
    APP_SIDEBAR.indexOf("function renderSectionSubmenu("),
  );
  const branch = (from: string, to: string) =>
    menu.slice(menu.indexOf(from), to ? menu.indexOf(to) : undefined);
  const pinned = branch('if (options.kind === "pinned")', '} else if (options.kind === "projects")');
  const projects = branch('} else if (options.kind === "projects")', '} else if (options.kind === "recents")');
  const recents = branch('} else if (options.kind === "recents")', '} else if (options.kind === "section")');
  const section = branch('} else if (options.kind === "section")', "return (\n      <NonModalDropdownMenu");
  // Pinned sorts its pins straight in the menu: no submenus, no Show, no New section.
  assert.match(pinned, /value=\{pinnedSort\}/);
  assert.doesNotMatch(pinned, /organizeSubmenu|renderSortSubmenu|setSectionHidden|mode: "create"/);
  // Projects groups the sidebar and sorts its chats; the section toggles live on Recents alone.
  assert.match(projects, /\{organizeSubmenu\}/);
  assert.match(projects, /shell\.organize\.sortChatsBy/);
  assert.doesNotMatch(projects, /sortProjectsBy/);
  // Every header menu is the chat row menu's size, submenus included.
  assert.match(menu, /className=\{cn\("unsloth-plus-menu sidebar-row-menu sidebar-menu"/);
  // It opens out to the right of the "...", not squeezed back inside the sidebar.
  assert.match(menu, /<NonModalDropdownMenu\n\s*side="bottom"\n\s*align="start"/);
  assert.doesNotMatch(menu, /className="unsloth-plus-menu w-/);
  assert.match(menu, /icon=\{PanelLeftIcon\}/);
  assert.match(APP_SIDEBAR, /value: "project", key: "shell\.organize\.byProject", icon: Folder01Icon/);
  assert.match(APP_SIDEBAR, /value: "list", key: "shell\.organize\.inOneList", icon: LeftToRightListBulletIcon/);
  assert.doesNotMatch(projects, /setSectionHidden|mode: "create"/);
  // Recents: organize, sort, then Show (Projects and custom sections only) and New section.
  assert.match(recents, /\{organizeSubmenu\}[\s\S]*shell\.organize\.sortChatsBy/);
  assert.match(recents, /t\("shell\.organize\.show"\)/);
  assert.match(recents, /setSectionHidden\(PROJECTS_SECTION_KEY, !on\)/);
  assert.match(recents, /setSectionHidden\(candidate\.id, !on\)/);
  assert.match(recents, /setSectionDialog\(\{ mode: "create" \}\)/);
  // Pinned always shows, so it has no toggle.
  assert.doesNotMatch(APP_SIDEBAR, /setSectionHidden\(PINNED_SECTION_KEY/);
  // A custom section's menu manages that section only.
  assert.match(section, /mode: "rename", section/);
  assert.match(section, /removeCustomSection\(section\)/);
  assert.doesNotMatch(section, /organizeSubmenu|setSectionHidden|mode: "create"/);
  // The four headers each say which menu they are.
  for (const kind of ["pinned", "projects", "recents", "section"]) {
    assert.match(APP_SIDEBAR, new RegExp(`renderSidebarHeaderMenu\\(\\{\\n[^}]*kind: "${kind}",`));
  }
  assert.ok(!APP_SIDEBAR.includes("includeOrganize"));
});

test("chat and folder menus, and bulk selections, can file into a section", () => {
  assert.match(APP_SIDEBAR, /target: \{ chatIds: \[item\.id\] \},/);
  assert.match(APP_SIDEBAR, /target: \{ projectIds: \[project\.id\] \},/);
  assert.match(APP_SIDEBAR, /renderBulkSectionSubmenu\(\n\s*"chat",/);
  assert.match(APP_SIDEBAR, /renderBulkSectionSubmenu\(\n\s*"project",/);
});

test("a drop's filing is committed with its slot", () => {
  const commit = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("function commitDrop("),
    APP_SIDEBAR.indexOf("function dropCueClass("),
  );
  assert.match(commit, /if \(filing\.kind === "chat"\) setChatsSection\(\[filing\.id\], filing\.sectionId\);/);
  assert.match(commit, /setCustomSectionSort\(sortedSection, "manual"\);/);
});

test("the section strings exist in English", () => {
  for (const key of [
    'createTitle: "New section"',
    'createDescription: "Group chats and projects however you like"',
    'namePlaceholder: "Section name"',
    'create: "Create section"',
  ]) {
    assert.ok(EN.includes(key), key);
  }
});

test("a section's header starts a chat that is filed there on its first send", () => {
  const section = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("function renderCustomSection("),
    APP_SIDEBAR.indexOf("function renderProjectsSection("),
  );
  assert.match(section, /onClick=\{\(\) => openNewChatInSection\(section\.id\)\}/);
  assert.match(section, /icon=\{PencilEdit02Icon\}/);
  // Marked only once the new chat is on screen, so the chat being left is never filed.
  assert.match(
    APP_SIDEBAR,
    /navigate\(\{ to: "\/chat", search: \{ new: nonce \} \}\)\.then\(\(\) =>\n\s*setPendingNewChatSection\(\{ sectionId, nonce \}\),/,
  );
  // Filed when the store gains an id while the address still names that new chat; leaving it
  // first drops the mark.
  const effect = APP_SIDEBAR.slice(APP_SIDEBAR.indexOf("if (!pendingNewChatSection) return;"));
  assert.match(effect, /search\.new === pendingNewChatSection\.nonce &&\n\s*!search\.thread/);
  assert.match(effect, /if \(!onNewChat\) \{\n\s*setPendingNewChatSection\(null\);/);
  assert.match(effect, /setChatsSection\(\[storeThreadId\], pendingNewChatSection\.sectionId\)/);
  // Not saved: a reload must not file some later chat.
  const store = useSidebarOrganizationStore.getInitialState();
  assert.equal(store.pendingNewChatSection, null);
});

test("sections above Recents drag by their headers, and Recents stays last", () => {
  // Drawn in the saved order; Recents is drawn after them and is no key of it.
  assert.match(
    APP_SIDEBAR,
    /\{orderedSectionKeys\.map\(\(key\) => \(\n\s*<Fragment key=\{key\}>\{renderOrderedSection\(key\)\}<\/Fragment>/,
  );
  const drag = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("function startSectionDrag("),
    APP_SIDEBAR.indexOf("function renderOrderedSection("),
  );
  // Only from the header, never from its buttons, and not by touch, which scrolls.
  assert.match(drag, /event\.pointerType === "touch"/);
  assert.match(drag, /closest\('\[data-sidebar="group-label"\]'\)/);
  assert.match(drag, /closest\("\.sidebar-header-action"\)/);
  // A click ends the drag on the header, which must not fold the section it moved.
  assert.match(drag, /window\.addEventListener\("click", swallow, \{ capture: true, once: true \}\)/);
  assert.match(drag, /if \(commit && landing\) moveSection\(key, landing\.target, landing\.edge\)/);
});

test("the sidebar and account menus share one flat surface and type; other menus keep theirs", async () => {
  const css = await readSrcAsync("index.css");
  // The shared menu surface, which the composer's menus use, is left as it was.
  assert.match(css, /\.dark \.unsloth-plus-menu\[data-slot\] \{\n\s*background-color: var\(--card\);/);
  assert.match(
    css,
    /\.unsloth-plus-menu\[data-slot\] \{[\s\S]*?box-shadow: 0 2px 8px -2px rgba\(0, 0, 0, 0\.16\);\n\s*\}/,
  );
  const tagged = ":is\\(\\.unsloth-plus-menu, \\.app-user-menu\\)\\.sidebar-menu\\[data-slot\\]";
  // No shadow, over the shared menu shadow's !important.
  assert.match(css, new RegExp(`${tagged} \\{\\n\\s*box-shadow: none !important;`));
  assert.match(
    css,
    new RegExp(
      `\\.dark ${tagged} \\{\\n\\s*background-color: color-mix\\(in srgb, var\\(--card\\), white 4%\\);\\n\\s*color: color-mix\\(in srgb, var\\(--foreground\\), white 45%\\);`,
    ),
  );
  // The system face at 14px, scaled, on sidebar rows and account rows alike.
  assert.match(css, new RegExp(`${tagged} \\{\\n\\s*font-family: ui-sans-serif,`));
  assert.match(
    css,
    /\.unsloth-plus-menu\.sidebar-row-menu\.sidebar-menu :is\([\s\S]*?\) \{\n\s*@apply py-2 text-ui-14;/,
  );
  assert.match(css, /\.app-user-menu\.sidebar-menu :is\([\s\S]*?\) \{\n\s*@apply text-ui-14;\n\s*font-weight: 400;/);
  // Every sidebar menu is marked, the account menu and its Help submenu included.
  assert.equal((APP_SIDEBAR.match(/"unsloth-plus-menu sidebar-row-menu sidebar-menu/g) ?? []).length, 14);
  assert.match(APP_SIDEBAR, /className="app-user-menu sidebar-menu menu-soft-surface-up/);
});

test("a custom section's menu edits it, acts on its chats, and removes it", () => {
  const menu = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf('} else if (options.kind === "section") {'),
    APP_SIDEBAR.indexOf("    return (\n      // Opens out to the right"),
  );
  const labels = [...menu.matchAll(/t\("(shell\.[\w.]+)"\)/g)].map((match) => match[1]);
  assert.deepEqual(labels, [
    "shell.sections.edit",
    "shell.sections.markAllRead",
    "shell.selection.archiveChats",
    "shell.sections.remove",
  ]);
  assert.equal((menu.match(/<DropdownMenuSeparator \/>/g) ?? []).length, 2);
  // Only the section's own chats, and each action is off when it has nothing to do.
  assert.match(menu, /row\.kind === "chat" \? \[row\.item\] : \[\]/);
  assert.match(menu, /disabled=\{!threadIds\.some\(\(id\) => unreadThreadIds\.has\(id\)\)\}/);
  assert.match(menu, /onSelect=\{\(\) => clearThreadsUnread\(threadIds\)\}/);
  assert.match(menu, /disabled=\{chats\.length === 0\} onSelect=\{\(\) => void archiveChatItems\(chats\)\}/);
  assert.match(menu, /icon=\{Settings02Icon\}[\s\S]*icon=\{Tick02Icon\}[\s\S]*icon=\{Archive03Icon\}[\s\S]*icon=\{Cancel01Icon\}/);
  assert.doesNotMatch(menu, /variant="destructive"|renderSortSubmenu/);
});

test("a section header drag re-renders the sidebar only when its landing changes", () => {
  const start = APP_SIDEBAR.indexOf("function startSectionDrag(");
  const body = APP_SIDEBAR.slice(start, APP_SIDEBAR.indexOf("function renderOrderedSection(", start));
  // Every pointer move redrawing the whole sidebar is what made the drag lag.
  assert.match(
    body,
    /if \(drawn && next\?\.target === landing\?\.target && next\?\.edge === landing\?\.edge\) return;\n\s*drawn = true;\n\s*landing = next;\n\s*setSectionDrag\(\{ key, landing \}\);/,
  );
});

test("undoing a removed section puts it back where it was drawn", () => {
  const start = APP_SIDEBAR.indexOf("function removeCustomSection(");
  const body = APP_SIDEBAR.slice(start, APP_SIDEBAR.indexOf("function renderSortSubmenu", start));
  assert.match(body, /const followers = drawnOrder\.slice\(drawnOrder\.indexOf\(section\.id\) \+ 1\);/);
  assert.match(body, /const follower = followers\.find\(\(key\) => sectionOrder\.includes\(key\)\);/);
  assert.match(body, /customSections,\n\s*sectionOrder,/);
});

test("the section name dialog keeps its mode while it closes", async () => {
  const dialog = await readSrcAsync("features/chat/components/section-name-dialog.tsx");
  assert.match(dialog, /if \(open && \(shown\.mode !== mode \|\| shown\.initialName !== initialName\)\)/);
  assert.match(dialog, /mode=\{shown\.mode\}\n\s*initialName=\{shown\.initialName\}/);
});
