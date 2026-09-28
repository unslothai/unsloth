// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  type ChatEntry,
  EMPTY_CHAT_FILTERS,
  NO_PROJECT,
  NO_SECTION,
  chatFiltersActive,
  chatTime,
  dateBucket,
  filterChats,
  groupChats,
  modelFacets,
  modelsByChat,
  projectStats,
  sectionStats,
  sortChats,
  sortSections,
  sortProjects,
  summarizeChatMessages,
  validateChatsSection,
} from "../src/features/library/chats/model.ts";
import { readSrc } from "./helpers/kit.ts";

const DAY = 86_400_000;
const now = new Date(2026, 8, 27, 12).getTime();

const chats: ChatEntry[] = [
  { id: "a", type: "single", title: "Café planning", createdAt: now - 40 * DAY, updatedAt: now - 1 * DAY, projectId: "p1" },
  { id: "b", type: "single", title: "bug triage", createdAt: now - 3 * DAY, updatedAt: now, isFork: true },
  { id: "c", type: "compare", title: "Model shootout", createdAt: now - 2 * DAY, updatedAt: now - 2 * DAY, projectId: "p2" },
  { id: "d", type: "single", title: "Archive me", createdAt: now - 90 * DAY, updatedAt: now - 60 * DAY },
];

const context = {
  pinned: new Set(["c"]),
  projectNames: new Map([
    ["p1", "Research"],
    ["p2", "Evals"],
  ]),
  models: new Map([
    ["a", ["unsloth/Qwen3-4B-GGUF"]],
    ["c", ["unsloth/Qwen3-4B-GGUF", "openai/gpt-5"]],
  ]),
  sectionOf: new Map([
    ["b", "s1"],
    ["d", "s1"],
  ]),
  sectionNames: new Map([["s1", "Weekend reading"]]),
};

const ids = (list: { id: string }[]) => list.map((entry) => entry.id);

test("search matches titles, project and section names and models, ignoring accents and case", () => {
  assert.deepEqual(ids(filterChats(chats, "cafe", EMPTY_CHAT_FILTERS, context)), ["a"]);
  assert.deepEqual(ids(filterChats(chats, "research", EMPTY_CHAT_FILTERS, context)), ["a"]);
  assert.deepEqual(ids(filterChats(chats, "gpt-5", EMPTY_CHAT_FILTERS, context)), ["c"]);
  assert.deepEqual(ids(filterChats(chats, "BUG tri", EMPTY_CHAT_FILTERS, context)), ["b"]);
  assert.deepEqual(ids(filterChats(chats, "weekend", EMPTY_CHAT_FILTERS, context)), ["b", "d"]);
});

test("flags narrow, projects, sections and models widen within themselves", () => {
  const flags = (...values: ("pinned" | "forks" | "compare")[]) => ({
    ...EMPTY_CHAT_FILTERS,
    flags: new Set(values),
  });
  assert.deepEqual(ids(filterChats(chats, "", flags("pinned"), context)), ["c"]);
  assert.deepEqual(ids(filterChats(chats, "", flags("forks"), context)), ["b"]);
  assert.deepEqual(ids(filterChats(chats, "", flags("forks", "compare"), context)), []);
  const projects = { ...EMPTY_CHAT_FILTERS, projects: new Set(["p1", NO_PROJECT]) };
  assert.deepEqual(ids(filterChats(chats, "", projects, context)), ["a", "b", "d"]);
  const sections = { ...EMPTY_CHAT_FILTERS, sections: new Set([NO_SECTION]) };
  assert.deepEqual(ids(filterChats(chats, "", sections, context)), ["a", "c"]);
  assert.equal(chatFiltersActive(sections), true);
  const models = { ...EMPTY_CHAT_FILTERS, models: new Set(["unsloth/Qwen3-4B-GGUF"]) };
  assert.deepEqual(ids(filterChats(chats, "", models, context)), ["a", "c"]);
  assert.equal(chatFiltersActive(EMPTY_CHAT_FILTERS), false);
  assert.equal(chatFiltersActive(models), true);
});

test("sorting honours key, direction and pinned-first", () => {
  const byName = sortChats(chats, { key: "name", desc: false }, context.pinned, false);
  assert.deepEqual(ids(byName), ["d", "b", "a", "c"]);
  const recent = sortChats(chats, { key: "updated", desc: true }, context.pinned, false);
  assert.deepEqual(ids(recent), ["b", "a", "c", "d"]);
  const pinnedFirst = sortChats(chats, { key: "updated", desc: true }, context.pinned, true);
  assert.deepEqual(ids(pinnedFirst), ["c", "b", "a", "d"]);
  const oldestCreated = sortChats(chats, { key: "created", desc: false }, context.pinned, false);
  assert.deepEqual(ids(oldestCreated), ["d", "a", "b", "c"]);
});

test("date buckets and groups keep time order even under a name sort", () => {
  assert.equal(dateBucket(now, now).kind, "today");
  assert.equal(dateBucket(now - DAY, now).kind, "yesterday");
  assert.equal(dateBucket(now - 3 * DAY, now).kind, "week");
  assert.equal(dateBucket(now - 20 * DAY, now).kind, "month");
  assert.deepEqual(dateBucket(now - 60 * DAY, now), { kind: "older", year: 2026, month: 6 });

  const byName = sortChats(chats, { key: "name", desc: false }, new Set(), false);
  const groups = groupChats(byName, "date", { now });
  assert.deepEqual(
    groups.map((group) => group.key),
    ["today", "yesterday", "week", "older:2026-6"],
  );
  const oldest = groupChats(byName, "date", { now, oldestFirst: true });
  assert.equal(oldest[0]?.key, "older:2026-6");
});

test("project groups put chats outside any project last", () => {
  const groups = groupChats(chats, "project", { now });
  assert.deepEqual(
    groups.map((group) => group.projectId),
    ["p1", "p2", null],
  );
  assert.deepEqual(ids(groups[2]?.items ?? []), ["b", "d"]);
  assert.deepEqual(groupChats([], "none"), []);
});

test("section groups put unfiled chats last, and pinned chats lead in their own group", () => {
  const bySection = groupChats(chats, "section", { now, sectionOf: context.sectionOf });
  assert.deepEqual(
    bySection.map((group) => [group.sectionId, ids(group.items)]),
    [
      ["s1", ["b", "d"]],
      [null, ["a", "c"]],
    ],
  );
  // Pinned "c" gets its own group instead of reordering "week".
  const byDate = groupChats(chats, "date", { now, pinned: context.pinned });
  assert.deepEqual(
    byDate.map((group) => group.key),
    ["pinned", "today", "yesterday", "older:2026-6"],
  );
  assert.deepEqual(ids(byDate[0]?.items ?? []), ["c"]);
  assert.equal(byDate[0]?.pinned, true);
  assert.deepEqual(groupChats(chats, "none", { pinned: context.pinned })[0]?.items.length, 4);
});

test("project stats count live and archived chats and track last activity", () => {
  const projects = [
    { id: "p1", name: "Research", createdAt: 1, updatedAt: 2 },
    { id: "p2", name: "evals", createdAt: 5, updatedAt: 3 },
    { id: "p3", name: "Empty", createdAt: 9, updatedAt: 4 },
  ];
  const archived: ChatEntry[] = [
    { id: "z", type: "single", title: "old", createdAt: 0, updatedAt: 0, projectId: "p1" },
  ];
  const stats = projectStats(projects, chats, archived);
  assert.deepEqual(stats.get("p1"), { chats: 1, archived: 1, lastActive: now - DAY });
  assert.deepEqual(stats.get("p3"), { chats: 0, archived: 0, lastActive: 4 });
  const byChats = sortProjects(projects, { key: "chats", desc: true }, stats, new Set());
  assert.deepEqual(ids(byChats), ["p2", "p1", "p3"]);
  const pinnedFirst = sortProjects(projects, { key: "name", desc: false }, stats, new Set(["p3"]));
  assert.deepEqual(ids(pinnedFirst), ["p3", "p2", "p1"]);
});

test("models key compare pairs by pair id and rank facets by use", () => {
  const models = modelsByChat([
    { id: "t1", modelId: "m1" },
    { id: "t2", pairId: "pair", modelId: "m1" },
    { id: "t3", pairId: "pair", modelId: "m2" },
    { id: "t4", modelId: null },
  ]);
  assert.deepEqual(models.get("pair"), ["m1", "m2"]);
  assert.equal(models.has("t4"), false);
  const facets = modelFacets(
    [
      { id: "t1", type: "single", title: "", createdAt: 0, updatedAt: 0 },
      { id: "pair", type: "compare", title: "", createdAt: 0, updatedAt: 0 },
    ],
    models,
  );
  assert.deepEqual(facets, [
    { model: "m1", count: 2 },
    { model: "m2", count: 1 },
  ]);
});

test("chats tab validates its view and sits between Folders and Images", () => {
  assert.equal(validateChatsSection("archived"), "archived");
  assert.equal(validateChatsSection("latest"), undefined);
  const search = readSrc("features/library/search.ts");
  const tabs = search.slice(search.indexOf("LIBRARY_TABS = ["), search.indexOf("] as const"));
  assert.match(tabs, /"folders",\s*\/\/[^\n]*\n\s*"chats",\s*"images",/);
  // One strip: no divider sets the tab apart.
  assert.doesNotMatch(readSrc("features/library/components/library-header.tsx"), /separated/);
});

test("a starred chat narrows by the Favorites flag and lists in the Favorites tab alone", () => {
  const favorites = { ...context, favorites: new Set(["a", "d"]) };
  const starred = { ...EMPTY_CHAT_FILTERS, flags: new Set(["favorite" as const]) };
  assert.deepEqual(ids(filterChats(chats, "", starred, favorites)), ["a", "d"]);
  assert.deepEqual(ids(filterChats(chats, "", starred, context)), []);
  const page = readSrc("features/library/library-page.tsx");
  // Only Favorites counts (and draws) starred chats.
  assert.match(page, /useFavoriteChatMatches\(query, tab === "favorites" && !folderId\)/);
  assert.match(page, /if \(favoriteChats > 0\) return renderFavorites\(\);/);
  // Listed among the files, not under a heading of their own.
  assert.match(page, /leading=\{entries\.rows\}/);
  assert.match(page, /\.\.\.entries\.cards,/);
});

test("section stats count filed chats and live projects, and sections sort by them", () => {
  const sections = [
    { id: "s1", name: "Weekend reading", sort: "manual" },
    { id: "s2", name: "archive", sort: "manual" },
  ];
  const stats = sectionStats(
    sections,
    chats,
    context.sectionOf,
    new Map([
      ["p1", "s2"],
      ["gone", "s2"],
    ]),
    new Set(["p1"]),
  );
  assert.deepEqual(stats.get("s1"), { chats: 2, projects: 0, lastActive: now });
  assert.deepEqual(stats.get("s2"), { chats: 0, projects: 1, lastActive: 0 });
  assert.deepEqual(ids(sortSections(sections, { key: "updated", desc: true }, stats)), ["s1", "s2"]);
  assert.deepEqual(ids(sortSections(sections, { key: "name", desc: false }, stats)), ["s2", "s1"]);
  // A section page has its own URL.
  assert.equal(validateChatsSection("sections"), "sections");
  const search = readSrc("features/library/search.ts");
  assert.match(search, /typeof search\.chatSection === "string" \? \{ chatSection: search\.chatSection \}/);
});

test("chats never enter the file listing that every other tab reads", () => {
  // File tabs read the library API; chat history must not feed it.
  for (const file of ["api.ts", "store.ts", "file-kind.ts", "filters.ts"]) {
    const source = readSrc(`features/library/${file}`);
    assert.doesNotMatch(source, /useChatSidebarItems|listStoredChatThreads|chats\/model/, file);
  }
  const page = readSrc("features/library/library-page.tsx");
  assert.match(page, /if \(tab === "chats" && !folderId\) \{\s*return <ChatsLibrary/);
});

test("the Chats tab opens on All, and a project opens its home in Chat", () => {
  assert.equal(validateChatsSection("all"), "all");
  const library = readSrc("features/library/chats/chats-library.tsx");
  assert.match(library, /search\.chatSection \? "chats" : \(search\.chatView \?\? "all"\)/);
  // All is the tab's bare URL; every other pill names itself.
  assert.match(library, /go\(entry === "all" \? \{\} : \{ chatView: entry \}\)/);
  // A project opens its home in Chat, not a Library page.
  assert.match(library, /void navigate\(\{ to: "\/chat", search: \{ project: id \} \}\)/);
  assert.doesNotMatch(readSrc("features/library/search.ts"), /project\?: string/);
  // Chat settings links to the Library instead of managing chats.
  assert.doesNotMatch(readSrc("features/settings/tabs/data-tab.tsx"), /manageChats/);
  assert.match(
    readSrc("features/settings/tabs/chat-tab.tsx"),
    /void navigate\(\{ to: "\/library", search: \{ show: "chats" \} \}\)/,
  );
  // All lists projects and sections among the chats, without headings.
  assert.match(library, /return chatListing\(true, true\);/);
  // One list: no Pinned or date headings in All.
  assert.match(library, /const ungrouped = embedded \|\| section === "all";/);
  assert.match(library, /pinned: pinnedFirst && !ungrouped \? pinned : undefined,/);
  // The chat menu opens the chat's folder; the recents menu no longer does.
  assert.match(readSrc("features/library/chats/chats-items.tsx"), /<OpenChatFolderItem item=\{chat\} \/>/);
});

test("contents count the shown branch, its images and HTML blocks", () => {
  const message = (
    id: string,
    parentId: string | null,
    role: string,
    createdAt: number,
    content: unknown[],
    attachments: unknown[] = [],
  ) => ({ id, parentId, role, createdAt, content, attachments });
  const summary = summarizeChatMessages([
    message("u1", null, "user", 1, [{ type: "text", text: "hi" }], [{ type: "image" }]),
    message("a1", "u1", "assistant", 2, [{ type: "text", text: "```html\n<p/>\n```\n```python\n```" }]),
    // An older sibling of a1, from a regenerate: not on the shown branch.
    message("a0", "u1", "assistant", 1.5, [{ type: "image", image: "x" }]),
    message("u2", "a1", "user", 3, [{ type: "image", image: "y" }]),
    message("s", null, "system", 0, []),
  ]);
  assert.deepEqual(summary, { messages: 3, images: 2, html: 1 });
  assert.deepEqual(summarizeChatMessages([]), { messages: 0, images: 0, html: 0 });
});

test("one date column picks created, last active or last modified", () => {
  const chat = { ...chats[0]!, createdAt: 1, updatedAt: 5, modifiedAt: 9 };
  assert.equal(chatTime(chat, "created"), 1);
  assert.equal(chatTime(chat, "updated"), 5);
  assert.equal(chatTime(chat, "modified"), 9);
  // A rename older than the last message leaves modified at the last message.
  assert.equal(chatTime({ ...chat, modifiedAt: 2 }, "modified"), 5);
  const older = { ...chat, id: "older", modifiedAt: 3 };
  const sorted = sortChats([older, chat], { key: "modified", desc: true }, new Set(), false);
  assert.deepEqual(ids(sorted), [chat.id, "older"]);
  const items = readSrc("features/library/chats/chats-items.tsx");
  assert.match(items, /export function DateHeader\(/);
  assert.doesNotMatch(items, /CREATED_COLUMN/);
});
