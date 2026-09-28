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
  mixEntries,
  modelFacets,
  modelsByChat,
  projectStats,
  sectionStats,
  sectionTime,
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
    { id: "t1", modelIds: ["m1"] },
    { id: "pair", modelIds: ["m1", "m2"] },
    { id: "t4" },
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
  assert.doesNotMatch(readSrc("features/library/components/library-header.tsx"), /separated/);
});

test("a starred chat narrows by the Favorites flag and lists in the Favorites tab alone", () => {
  const favorites = { ...context, favorites: new Set(["a", "d"]) };
  const starred = { ...EMPTY_CHAT_FILTERS, flags: new Set(["favorite" as const]) };
  assert.deepEqual(ids(filterChats(chats, "", starred, favorites)), ["a", "d"]);
  assert.deepEqual(ids(filterChats(chats, "", starred, context)), []);
  const page = readSrc("features/library/library-page.tsx");
  assert.match(page, /useFavoriteChatMatches\(query, tab === "favorites" && !folderId\)/);
  assert.match(page, /if \(favoriteChats > 0\) return renderFavorites\(\);/);
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
  assert.match(library, /go\(entry === "all" \? \{\} : \{ chatView: entry \}\)/);
  assert.match(library, /void navigate\(\{ to: "\/chat", search: \{ project: id \} \}\)/);
  assert.doesNotMatch(readSrc("features/library/search.ts"), /project\?: string/);
  assert.doesNotMatch(readSrc("features/settings/tabs/data-tab.tsx"), /manageChats/);
  assert.match(
    readSrc("features/settings/tabs/chat-tab.tsx"),
    /void navigate\(\{ to: "\/library", search: \{ show: "chats" \} \}\)/,
  );
  assert.match(library, /return chatListing\(true\);/);
  assert.match(library, /mixEntries\(visibleChats, visibleProjects, visibleSections, \{/);
  assert.match(library, /const ungrouped = embedded \|\| section === "all";/);
  assert.match(library, /pinned: pinnedFirst && !ungrouped \? pinned : undefined,/);
  assert.match(readSrc("features/library/chats/chats-items.tsx"), /<OpenChatFolderItem item=\{chat\} \/>/);
});

test("contents count messages on the shown branch", () => {
  const message = (id: string, parentId: string | null, role: string, createdAt: number) => ({
    id,
    parentId,
    role,
    createdAt,
  });
  const summary = summarizeChatMessages([
    message("u1", null, "user", 1),
    message("a1", "u1", "assistant", 2),
    // An older sibling of a1, from a regenerate: not on the shown branch.
    message("a0", "u1", "assistant", 1.5),
    message("u2", "a1", "user", 3),
    message("s", null, "system", 0),
  ]);
  assert.deepEqual(summary, { messages: 3 });
  assert.deepEqual(summarizeChatMessages([]), { messages: 0 });
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

test("Favorites draws starred chats, projects and sections as file cards, without a star", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  assert.match(items, /export function FavoriteSectionTile\(/);
  assert.match(items, /if \(!useChatsActions\(\)\.favoriteMarks\) return null;/);
  assert.match(readSrc("features/library/chats/favorites-store.ts"), /sectionIds: readIds\(saved\?\.sectionIds\)/);
  assert.match(readSrc("features/library/library-page.tsx"), /favoriteMarks=\{tab !== "favorites"\}/);
});

test("the Library chat menu marks a chat read or unread", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  assert.match(items, /unread \? ViewIcon : ViewOffSlashIcon/);
  assert.match(items, /"shell\.selection\.markRead"/);
});

test("the Library lists chat metadata and counts messages without reading them", () => {
  const library = readSrc("features/library/chats/chats-library.tsx");
  assert.match(library, /useChatSidebarItems\(\{ requireMessages: false \}\)/);
  assert.match(
    readSrc("features/library/chats/favorites.ts"),
    /useChatSidebarItems\(\{ enabled, requireMessages: false \}\)/,
  );
  const contents = readSrc("features/library/chats/contents.ts");
  assert.match(contents, /await countStoredChatMessages\(ids\)/);
  // A re-render must not cancel a read in flight and ask for the same batch again.
  assert.doesNotMatch(contents, /let cancelled/);
  assert.match(contents, /!pending\.has\(key\)/);
});

test("last modified is the server's, stamped by renames, moves and archiving", () => {
  const sidebar = readSrc("features/chat/hooks/use-chat-sidebar-items.ts");
  assert.match(sidebar, /\.\.\.\(t\.modifiedAt \? \{ modifiedAt: t\.modifiedAt \} : \{\}\)/);
  assert.doesNotMatch(readSrc("features/chat/index.ts"), /useChatModifiedStore/);
});

test("deleting clears stars and pins only once the chats are gone", () => {
  const library = readSrc("features/library/chats/chats-library.tsx");
  const start = library.indexOf("async function confirmDelete(");
  const block = library.slice(start, library.indexOf("const { project } = target;", start));
  const deleted = block.indexOf("await deleteChatItems(");
  assert.ok(deleted !== -1);
  assert.ok(block.indexOf("setFavoriteChats(ids, false)") > deleted);
  assert.ok(block.indexOf("setPinned(ids, false)") > deleted);
  assert.match(library, /if \(confirmDeleteChats\) setPendingDelete\(target\);/);
});

test("shift-click selects a range anchored on a chat id", () => {
  const library = readSrc("features/library/chats/chats-library.tsx");
  assert.match(library, /const selectionAnchor = useRef<string \| null>\(null\);/);
  assert.match(library, /range && anchor \? rangeBetween\(shownOrder, anchor, id\) : \[id\]/);
  const items = readSrc("features/library/chats/chats-items.tsx");
  assert.match(items, /toggleSelected\(chat\.id, event\.shiftKey\)/);
});

test("a chat exports per pane with Markdown, several chats combined or per chat", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  assert.match(items, /chatExportOptions\(\)\.map/);
  assert.match(items, /<BulkExportItems onExport=/);
  assert.match(readSrc("features/chat/components/bulk-export-items.tsx"), /COMBINED_EXPORT_FORMATS_LIST\.map/);
  assert.match(items, /disabled=\{!actions\.projectChatCounts\.get\(project\.id\)\}/);
  const library = readSrc("features/library/chats/chats-library.tsx");
  assert.match(library, /for \(const id of threadIds\) await exportConversationByFormat\(id, choice\.format\);/);
});

test("fork is off while the chat generates or another fork runs", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  assert.match(items, /disabled=\{!canForkChatRow\(chat\) \|\| generating \|\| forking\}/);
});

test("All sorts chats, projects and sections together, pinned first", () => {
  const project = { id: "p1", name: "Research", createdAt: now - 5 * DAY, updatedAt: now - 5 * DAY };
  const section = { id: "s1", name: "Weekend reading" };
  const options = {
    sort: { key: "updated", desc: true } as const,
    projectStats: new Map([["p1", { chats: 1, archived: 0, lastActive: now - 1.5 * DAY }]]),
    sectionStats: new Map([["s1", { chats: 2, projects: 0, lastActive: now - 0.5 * DAY }]]),
    pinned: new Set<string>(),
    pinnedProjects: new Set<string>(),
    pinnedFirst: true,
  };
  const key = (entries: { kind: string; item: { id: string } }[]) =>
    entries.map((entry) => `${entry.kind}:${entry.item.id}`);
  assert.deepEqual(key(mixEntries(chats, [project], [section], options)), [
    "chat:b",
    "section:s1",
    "chat:a",
    "project:p1",
    "chat:c",
    "chat:d",
  ]);
  const pinned = mixEntries(chats, [project], [section], {
    ...options,
    pinnedProjects: new Set(["p1"]),
  });
  assert.equal(key(pinned)[0], "project:p1");
  const byName = mixEntries(chats, [project], [section], {
    ...options,
    sort: { key: "name", desc: false },
  });
  assert.deepEqual(key(byName).slice(0, 3), ["chat:d", "chat:b", "chat:a"]);
});

test("an empty Library opens on Chats whatever the start tab", () => {
  const page = readSrc("features/library/library-page.tsx");
  assert.match(page, /if \(emptyOnLoad === null && loaded\) setEmptyOnLoad\(items\.length === 0 && folders\.length === 0\);/);
  assert.doesNotMatch(page, /settings\.startTab === "last" &&\n\s*tabVisible\("chats"\)/);
  assert.match(page, /!\(preferred === "favorites" && hasStarredChats\)/);
});

test("sections have created and last modified dates, and Last modified is the default", async () => {
  const stats = { chats: 1, projects: 0, lastActive: 50 };
  const section = { id: "s", name: "S", createdAt: 10, modifiedAt: 80 };
  assert.equal(sectionTime(section, stats, "created"), 10);
  assert.equal(sectionTime(section, stats, "updated"), 50);
  assert.equal(sectionTime(section, stats, "modified"), 80);
  // A newer chat counts as a change, as it does for chats; old sections have no created date.
  assert.equal(sectionTime({ id: "o", name: "O" }, stats, "modified"), 50);
  assert.equal(sectionTime({ id: "o", name: "O" }, stats, "created"), 0);
  const sorted = sortSections(
    [section, { id: "t", name: "T", modifiedAt: 90 }],
    { key: "modified", desc: true },
    new Map([["s", stats]]),
  );
  assert.deepEqual(ids(sorted), ["t", "s"]);
  const prefs = readSrc("features/library/chats/prefs-store.ts");
  assert.match(prefs, /dateField: "modified",/);
  assert.match(prefs, /sort: \{ key: "modified", desc: true \},/);
  assert.doesNotMatch(readSrc("features/library/chats/chats-items.tsx"), /field="updated"/);
});

test("chat, project and section cards are a little shorter than square, dated bottom left", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  const card = items.slice(items.indexOf("const CARD = cn("), items.indexOf("const ICON ="));
  assert.match(card, /aspect-\[8\/7\]/);
  assert.doesNotMatch(card, /overflow-hidden/);
  assert.match(card, /self-stretch/);
  assert.doesNotMatch(items, /cn\(CARD, "min-h-/);
  assert.doesNotMatch(items, /justify-between gap-2 text-ui-12 text-muted-foreground/);
});

test("chat, project and section tiles share one box and hover tint", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  assert.match(items, /const TILE =\s*"[^"]*bg-muted[^"]*group-hover\/chat:bg-primary\/10 group-hover\/chat:text-primary"/);
  assert.equal(items.split("group-hover/chat:bg-primary/10").length - 1, 1, "one tile style");
  assert.match(items, /<div className=\{cn\(TILE, className\)\}>/);
  assert.match(items, /<span className=\{TILE\}>/);
  assert.doesNotMatch(items, /group-hover\/chat:opacity-0/);
});

test("grid cards share a two-line title and a footer of contents over date", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  assert.match(items, /const CARD_TITLE =\s*"[^"]*\[overflow-wrap:anywhere\]/);
  assert.match(items, /const CARD_TITLE =\s*"[^"]*font-medium text-ui-14 /);
  const footer = items.slice(items.indexOf("function CardFooter("), items.indexOf("const CARD_TITLE"));
  assert.match(footer, /flex-col items-start/);
  assert.match(footer, /\{meta && <span className="w-full min-w-0 truncate">\{meta\}<\/span>\}\s*<span className="truncate">\{date\}<\/span>/);
  assert.doesNotMatch(footer, /·/);
  const cardOf = (start: string, end: string) =>
    items.slice(items.indexOf(start), items.indexOf(end));
  const cards = [
    cardOf("export function ChatCard(", "export function GroupHeading("),
    cardOf("export function ProjectCard(", "export function ProjectRow("),
    cardOf("export function SectionCard(", "export function SectionRow("),
  ];
  for (const card of cards) {
    assert.match(card, /className=\{CARD_TITLE\}/);
    assert.match(card, /<span className="line-clamp-2">/);
    assert.match(card, /<CardFooter/);
  }
  assert.doesNotMatch(cards[0], /modelLabel|model &&/);
  assert.match(cards[0], /<ChatBadges chat=\{chat\} marksOnly \/>/);
  assert.doesNotMatch(cards[1], /archived/);
  assert.match(cards[0], /<ChatLocation[^>]*plain/);
});

test("project cards in the grid leave out the instructions", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  const card = items.slice(items.indexOf("export function ProjectCard("), items.indexOf("export function ProjectRow("));
  assert.doesNotMatch(card, /instructions/);
});

test("library menus take the sidebar menu look, with a shadow, in dark mode", () => {
  const css = readSrc("index.css");
  const menu = css.slice(css.indexOf("/* Library menus in dark"), css.indexOf("/* Library header row"));
  assert.match(menu, /--library-menu-surface: color-mix\(in srgb, var\(--card\), white 7%\)/);
  assert.match(menu, /--accent: color-mix\(in srgb, var\(--library-menu-surface\), white 10%\)/);
  assert.match(menu, /--destructive: #ed716a/);
  assert.match(menu, /box-shadow: 0 4px 14px var\(--background\) !important/);
  assert.match(menu, /\[data-slot="dropdown-menu-separator"\] \{\s*background-color: color-mix\(in srgb, var\(--card\), white 16%\)/);
  const toolbar = readSrc("features/library/components/library-toolbar.tsx");
  assert.equal(toolbar.match(/className="library-menu /g)?.length, 3);
  const page = readSrc("features/library/library-page.tsx");
  assert.equal(page.match(/className="library-menu /g)?.length, 2);
});

test("the library header row casts a light shadow once stuck", () => {
  const css = readSrc("index.css");
  assert.match(
    css,
    /\.library-header-row\[data-stuck\] \{\s*box-shadow: 0 2px 6px -3px rgba\(0, 0, 0, 0\.1\);\s*clip-path: inset\(0 0 -40px 0\);/,
  );
  assert.match(
    css,
    /\.dark \.library-header-row\[data-stuck\] \{\s*box-shadow: 0 2px 10px -2px var\(--background\);/,
  );
  const header = readSrc("features/library/components/library-header.tsx");
  assert.match(header, /data-stuck=\{stuck \|\| undefined\}/);
  assert.match(header, /setStuck\(gap > /);
});

test("a favorite project tile uses the open project folder, not the file folder", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  const tile = items.slice(
    items.indexOf("export function FavoriteProjectTile("),
    items.indexOf("}", items.indexOf("menu={<ProjectMenu", items.indexOf("export function FavoriteProjectTile("))),
  );
  assert.match(tile, /icon=\{Folder02Icon\}/);
  assert.doesNotMatch(tile, /Folder01Icon/);
});

test("a project's home and the Projects page use the Library project menu", () => {
  const items = readSrc("features/chat/components/project-menu-items.tsx");
  // Library order; without New chat, the folder goes under Export.
  const order = [
    "onNewChat && (",
    "<OpenProjectFolderItem projectId={project.id} />",
    "onSelect={onEdit}",
    "togglePin(project.id)",
    "setFavorites([project.id], !favorite)",
    "shell.sections.moveTo",
    "<BulkExportItems",
    "settings.chat.importChats",
    "{!opening && <OpenProjectFolderItem",
    "onSelect={onDelete}",
  ];
  let at = -1;
  for (const marker of order) {
    const next = items.indexOf(marker);
    assert.ok(next > at, `${marker} in order`);
    at = next;
  }
  const page = readSrc("features/chat/chat-page.tsx");
  const homeMenu = page.slice(page.indexOf("<ProjectMenuItems"), page.indexOf("/>", page.indexOf("<ProjectMenuItems")));
  assert.match(homeMenu, /project=\{\{ id: projectId, name: projectName \}\}\s+chatCount=\{items\.length\}/);
  assert.doesNotMatch(homeMenu, /onNewChat/);
  assert.match(page, /<SectionNameDialog\s+open=\{active && creatingSection\}/);
  const projects = readSrc("features/chat/projects-page.tsx");
  assert.match(projects, /<ProjectMenuItems\s+project=\{project\}\s+onNewChat=/);
  assert.match(projects, /onNewChat=\{\(\) => newChatInProject\(project\.id\)\}/);
  // Filed the same way everywhere, through the shared hook.
  for (const source of [page, projects, readSrc("features/library/chats/chats-library.tsx"), items]) {
    assert.match(source, /useFileProjectInSection\(\)/);
  }
});

test("Edit and Pin read the same, with the sidebar's icon, in the project menus", () => {
  const menu = readSrc("features/chat/components/project-menu-items.tsx");
  assert.match(menu, /<Item icon=\{Settings02Icon\} onSelect=\{onEdit\}>\s*\{t\("library\.chats\.menu\.edit"\)\}/);
  assert.match(menu, /t\(pinned \? "settings\.data\.library\.unpin" : "settings\.data\.library\.pin"\)/);
  assert.doesNotMatch(menu, /Edit project|Pin project|Edit03Icon/);
  const items = readSrc("features/library/chats/chats-items.tsx");
  assert.match(items, /icon=\{Settings02Icon\}\s*label=\{t\("library\.chats\.menu\.edit"\)\}/);
  assert.match(readSrc("i18n/locales/en.ts"), /\n        edit: "Edit",/);
});

test("Import chats sits in the project and section menus", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  assert.match(items, /pickAndImportChats\(\{ projectId: project\.id, name: project\.name \}\)/);
  assert.match(items, /pickAndImportChats\(\{ projectId: null, sectionId: section\.id, name: section\.name \}\)/);
  const runner = readSrc("features/chat/utils/import-chats.ts");
  assert.match(runner, /onSaved: \(threadId\) => threadIds\.push\(threadId\)/);
  assert.match(runner, /setChatsSection\(\[\.\.\.new Set\(threadIds\)\], target\.sectionId\)/);
  // The Projects page imports through the same runner.
  assert.match(readSrc("features/chat/projects-page.tsx"), /await runChatImport\(source, \{/);
});

test("the empty space in a Projects page row opens its chats", () => {
  const page = readSrc("features/chat/projects-page.tsx");
  assert.match(page, /<span\s+aria-hidden="true"\s+onClick=\{\(\) => toggleProjectChats\(project\.id\)\}\s+className="-my-4 min-w-0 flex-1 cursor-pointer self-stretch"/);
  // After the chevron, before Updated.
  const chevron = page.indexOf("<ChevronDownIcon");
  const spacer = page.indexOf('className="-my-4 min-w-0 flex-1 cursor-pointer self-stretch"');
  const updated = page.indexOf('className="hidden w-40 shrink-0 text-sm text-muted-foreground sm:block"');
  assert.ok(chevron < spacer && spacer < updated);
});

test("filing a project in a section unpins it and shows the section", () => {
  const hook = readSrc("features/chat/hooks/use-file-project-in-section.ts");
  assert.match(hook, /if \(pins\.pinnedIds\.includes\(project\.id\)\) pins\.unpin\(project\.id\);/);
  assert.match(hook, /organization\.setSectionHidden\(sectionId, false\);/);
});

test("clearing the selection drops the shift-click anchor", () => {
  const library = readSrc("features/library/chats/chats-library.tsx");
  assert.match(
    library,
    /const setSelection = useCallback\(\(next: SetStateAction<Set<string>>\) => \{\n\s*if \(typeof next !== "function" && next\.size === 0\) selectionAnchor\.current = null;/,
  );
  // Every emptying write goes through the wrapper, never the raw state setter.
  assert.doesNotMatch(library, /setSelectionState\(new Set\(\)\)/);
});

test("a section's menu reads New chat, New project and Edit", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  const menu = items.slice(items.indexOf("export function SectionMenuItems("), items.indexOf("function sectionCountLabel("));
  assert.match(menu, /label=\{t\("library\.chats\.toolbar\.newChat"\)\}/);
  assert.match(menu, /icon=\{FolderAddIcon\}\s*label=\{t\("library\.chats\.toolbar\.newProject"\)\}\s*onSelect=\{\(\) => actions\.newProjectInSection\(section\.id\)\}/);
  assert.match(menu, /icon=\{Settings02Icon\}\s*label=\{t\("shell\.sections\.edit"\)\}/);
  assert.doesNotMatch(menu, /newChatInSection"\)|renameTitle/);
  // A project made there, or from the section page's New button, is filed in the section.
  const library = readSrc("features/library/chats/chats-library.tsx");
  assert.match(library, /if \(newProjectSection\) fileProjectInSection\(project, newProjectSection\);/);
  assert.match(library, /openSectionId \? newProjectInSection\(openSectionId\) : setCreatingProject\(true\)/);
});

test("a chat card's location has its project or section icon; menus leave out View chats", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  const plain = items.slice(items.indexOf("  if (plain) {"), items.indexOf('<span className="flex min-w-0 items-center gap-3">'));
  assert.match(plain, /icon=\{projectId \? Folder02Icon : LayerIcon\}/);
  assert.match(plain, /<span className="truncate">\{projectId \? projectName : section\?\.name\}<\/span>/);
  // No View chats anywhere: clicking a project or section already shows its chats.
  for (const file of [
    "features/library/chats/chats-items.tsx",
    "features/chat/components/project-menu-items.tsx",
    "features/chat/projects-page.tsx",
  ]) {
    assert.doesNotMatch(readSrc(file), /viewChats/, file);
  }
  assert.doesNotMatch(readSrc("features/chat/components/project-menu-items.tsx"), /onView/);
  assert.doesNotMatch(readSrc("i18n/locales/en.ts"), /viewChats/);
});

test("the Chats library draws every project with the open project folder", () => {
  for (const file of ["chats-library.tsx", "chats-toolbar.tsx", "chats-items.tsx"]) {
    assert.doesNotMatch(readSrc(`features/library/chats/${file}`), /Folder01Icon/, file);
  }
  assert.match(readSrc("features/library/chats/chats-library.tsx"), /projects: Folder02Icon,/);
});

test("a chat's menu has no Open chat, and its folder sits under Export", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  const menu = items.slice(items.indexOf("function ChatMenu("), items.indexOf("const TILE ="));
  assert.doesNotMatch(menu, /library\.chats\.menu\.open"/);
  const exportAt = menu.indexOf("<ExportSubmenu");
  const folderAt = menu.indexOf("<OpenChatFolderItem");
  assert.ok(exportAt !== -1 && folderAt > exportAt, "folder under Export");
  assert.ok(!menu.slice(exportAt, folderAt).includes("DropdownMenuSeparator"), "same group as Export");
  const projects = readSrc("features/chat/projects-page.tsx");
  assert.ok(projects.indexOf("<OpenChatFolderItem item={chat} />") > projects.indexOf("Export all chats…"));
});

test("the Library's New chat is a saved chat with an empty composer, like the sidebar's", () => {
  const library = readSrc("features/library/chats/chats-library.tsx");
  const newChatIn = library.slice(library.indexOf("const newChatIn = "), library.indexOf("async function run("));
  assert.match(newChatIn, /clearNewChatDraft\(\);/);
  assert.match(newChatIn, /runtime\.setIncognito\(false\);/);
});

test("a chat among Favorites files dates from its last edit, as the Chats view does", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  assert.match(items, /<FileColumns modified=\{chatTime\(chat, "modified"\)\} \/>/);
});

test("Reset all local preferences clears the Chats library preferences", () => {
  const general = readSrc("features/settings/tabs/general-tab.tsx");
  const keys = general.slice(general.indexOf("const PREFS_KEYS"), general.indexOf("];", general.indexOf("const PREFS_KEYS")));
  assert.match(keys, /LIBRARY_CHATS_PREFS_STORAGE_KEY/);
});

test("a New chat from a project's menu on the Projects page is a saved chat with an empty composer", () => {
  const page = readSrc("features/chat/projects-page.tsx");
  const body = page.slice(page.indexOf("function newChatInProject("), page.indexOf("function newChatInProject(") + 300);
  assert.match(body, /clearNewChatDraft\(\);/);
  assert.match(body, /setIncognito\(false\)/);
});

test("imported comparisons are filed under their pair id, the row the section lists", () => {
  const source = readSrc("features/chat/utils/chat-import.ts");
  assert.match(source, /options\.onSaved\?\.\(conversation\.thread\?\.pairId \?\? conversation\.threadId\)/);
  assert.doesNotMatch(source, /saved\(conversation\.threadId\)/);
});

test("a favorite chat card dates from its last edit", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  assert.doesNotMatch(items, /time=\{chat\.updatedAt\}/);
});

test("a favorite section card dates as its list row does", () => {
  const items = readSrc("features/library/chats/chats-items.tsx");
  const tile = items.slice(items.indexOf("export function FavoriteSectionTile("));
  assert.match(tile.slice(0, 800), /time=\{sectionTime\(section, stats, "modified"\)\}/);
});
