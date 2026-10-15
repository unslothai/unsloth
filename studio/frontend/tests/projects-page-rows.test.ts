// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSrc, readSrcAsync } from "./helpers/kit.ts";
import { formatWorkedFor } from "../src/lib/format-worked-for.ts";

const PAGE = await readSrcAsync("features/chat/projects-page.tsx");

test("a project row opens its chats in place", () => {
  assert.match(PAGE, /aria-label=\{chatsOpen \? "Hide chats" : "Show chats"\}/);
  assert.match(PAGE, /aria-expanded=\{chatsOpen\}/);
  assert.match(PAGE, /if \(!opening\) return;\n\s*const cached = projectChats\[projectId\];\n\s*const loaded = cached !== undefined && cached !== "error";\n\s*if \(loaded && !staleProjectIds\.has\(projectId\)\) return;\n[^\n]*\n\s*loadProjectChats\(projectId, loaded\);/);
  // A failed reload keeps loaded rows only if non-empty: an empty folder may have just gained a chat.
  assert.match(PAGE, /const rows = prev\[projectId\];\n\s*if \(silent && Array\.isArray\(rows\) && rows\.length > 0\) \{\n\s*setStale\(projectId, true\);\n\s*return prev;\n\s*\}\n\s*return \{ \.\.\.prev, \[projectId\]: "error" \};/);
  assert.match(PAGE, /setStale\(projectId, false\);\n\s*setProjectChats/);
  assert.match(PAGE, /\{staleProjectIds\.has\(project\.id\) && \(/);
  assert.match(PAGE, /Could not refresh chats\. Retry/);
  assert.match(PAGE, /Could not load chats\. Retry/);
  assert.ok(!PAGE.includes("[projectId]: [] }"), "a failed load is still cached as an empty list");
  assert.match(
    PAGE,
    /listStoredChatThreads\(\{ projectId, includeArchived: false \}\)/,
  );
  assert.match(PAGE, /window\.addEventListener\(CHAT_HISTORY_UPDATED_EVENT, refresh\);/);
  assert.match(PAGE, /timer = setTimeout\(\(\) => \{[\s\S]*?for \(const id of open\) loadProjectChats\(id, true\);\n\s*\}, PROJECT_CHATS_REFRESH_DEBOUNCE_MS\);/);
  assert.match(PAGE, /if \(!silent\) \{\n\s*setProjectChats\(\(prev\) => \(\{ \.\.\.prev, \[projectId\]: "loading" \}\)\);/);
  assert.match(PAGE, /for \(const \[id, seq\] of loadSeqRef\.current\) \{\n\s*if \(!open\.has\(id\)\) loadSeqRef\.current\.set\(id, seq \+ 1\);/);
  assert.match(PAGE, /if \(loadSeqRef\.current\.get\(projectId\) !== seq\) return;\n\s*setStale\(projectId, false\);\n\s*setProjectChats/);
  assert.match(PAGE, /groupThreads\(threads\)\.sort\(\n\s*\(a, b\) => b\.updatedAt - a\.updatedAt,/);
  assert.match(PAGE, /No chats<\/p>/);
  assert.match(PAGE, /onClick=\{\(\) => openChat\(chat, project\.id\)\}/);
  assert.match(PAGE, /search: \{ compare: item\.id, project: projectId \}/);
  assert.match(
    PAGE,
    /<button\n\s*type="button"\n\s*onClick=\{\(\) => openChat\(chat, project\.id\)\}/,
  );
});

test("the disclosure sits with the name it opens", () => {
  const name = PAGE.indexOf("{project.name}");
  const chevron = PAGE.indexOf("<ChevronDownIcon", name);
  assert.notEqual(name, -1, "the name moved");
  assert.notEqual(chevron, -1, "the disclosure moved");
  assert.match(
    PAGE,
    /<span className="flex min-w-0 flex-1 items-center gap-2">[\s\S]{0,1200}?<span className="min-w-0 truncate text-ui-15 font-semibold text-foreground">\n\s*\{project\.name\}/,
  );
  assert.match(
    PAGE,
    /<button\n\s*type="button"\n\s*onClick=\{\(\) => openProject\(project\.id\)\}/,
  );
  assert.ok(
    PAGE.indexOf(
      'className="hidden w-40 shrink-0 text-sm text-muted-foreground sm:block"',
    ) > chevron,
    "the arrow is still drawn after the Updated column",
  );
});

test("the row menu edits a project rather than only renaming it", () => {
  assert.match(PAGE, /import \{ EditProjectDialog \} from "\.\/components\/edit-project-dialog";/);
  assert.match(PAGE, /onEdit=\{\(\) => setEditing\(project\)\}/);
  assert.match(readSrc("features/chat/components/project-menu-items.tsx"), /<Item icon=\{Settings02Icon\} onSelect=\{onEdit\}>/);
  assert.match(PAGE, /<EditProjectDialog\n\s*project=\{editing\}/);
  assert.match(PAGE, /onDelete=\{\(project\) => openProjectDelete\(project\)\}/);
  assert.ok(!PAGE.includes("Rename project"));
  assert.ok(!PAGE.includes("renameChatProject"));
});

test("pinning a project takes one click, and says which way it goes", () => {
  assert.match(PAGE, /aria-label=\{pinned \? "Unpin project" : "Pin project"\}/);
  assert.match(PAGE, /togglePinProject\(project\.id\);/);
  assert.ok(
    !PAGE.includes('{pinned ? "Unpin" : "Pin"}'),
    "the row menu still carries a pin item",
  );
  assert.equal(
    (PAGE.match(/togglePinProject\(project\.id\)/g) ?? []).length,
    1,
    "pinning has more than one control on a row",
  );
});

// Hover-revealed actions are invisible on touch screens.
test("the pin and the menu are on show, the disclosure follows the cursor", () => {
  const start = PAGE.indexOf("{visibleProjects.map((project) => {");
  assert.notEqual(start, -1, "the row list moved");
  const row = PAGE.slice(start, PAGE.indexOf("<DropdownMenuContent", start));
  assert.ok(row.length > 0, "the row moved");
  assert.ok(!row.includes("opacity-0 transition"), "the menu is still hover-revealed");
  assert.ok(
    !row.includes('pinned ? "opacity-100" : "opacity-0"'),
    "the pin is still hover-revealed",
  );
  assert.equal(
    (row.match(/pointer-coarse:opacity-100/g) ?? []).length,
    1,
    "the disclosure is not the only control left gated on hover",
  );
  assert.match(row, /group-hover\/project-row:opacity-100 pointer-coarse:opacity-100/);
  assert.match(row, /chatsOpen \? "opacity-100" : "opacity-0",/);
});

test("the Updated column names the list's order and turns it around", () => {
  assert.ok(!PAGE.includes(">Modified<"), "the old column name is still rendered");
  assert.match(PAGE, /\n\s*Updated\n(?:\s*\{\/\*[^\n]*\*\/\}\n)?\s*<ArrowDownIcon/);
  assert.match(PAGE, /import \{ ArrowDownIcon, ChevronDownIcon, MoreHorizontalIcon \} from "lucide-react";/);
  assert.match(
    PAGE,
    /onClick=\{\(\) => setSortDir\(\(dir\) => \(dir === "desc" \? "asc" : "desc"\)\)\}/,
  );
  assert.match(PAGE, /sortDir === "asc" && "rotate-180",/);
  assert.match(
    PAGE,
    /sortDir === "asc" \? a\.updatedAt - b\.updatedAt : b\.updatedAt - a\.updatedAt,/,
  );
  assert.ok(!PAGE.includes("Sort by"), "the sort control is still in the header");
  assert.ok(!PAGE.includes("sortMode"), "the sort mode outlived its control");
  assert.match(PAGE, /<div className="relative min-w-0 flex-1 sm:max-w-md">/);
  assert.match(PAGE, /className="h-9 w-full rounded-full border-none bg-muted pl-10/);
  assert.match(
    PAGE,
    /<div className="flex w-full min-w-0 items-center justify-end gap-3 sm:w-auto sm:flex-1">/,
  );
  assert.match(PAGE, /useState<"desc" \| "asc">\("desc"\)/);
  assert.match(
    PAGE,
    /<span className="size-7 shrink-0" \/>\n\s*<span className="w-8 shrink-0" \/>\n\s*<\/div>/,
  );
});

test("a finished run says how long it worked, in units that read", () => {
  assert.equal(formatWorkedFor(0), "0s");
  assert.equal(formatWorkedFor(45), "45s");
  assert.equal(formatWorkedFor(60), "1m 0s");
  assert.equal(formatWorkedFor(216), "3m 36s");
  assert.equal(formatWorkedFor(3600), "1h 0m");
  assert.equal(formatWorkedFor(3840), "1h 4m");
  assert.equal(formatWorkedFor(-5), "0s");
});

test("the reasoning header says Worked for", async () => {
  const reasoning = await readSrcAsync("components/assistant-ui/reasoning.tsx");
  assert.match(reasoning, /Worked for \{formatWorkedFor\(duration \?\? 0\)\}/);
  assert.ok(!reasoning.includes("Thought for"), "the old label is still rendered");
});

test("a chat row carries its own actions, revealed by hovering it", async () => {
  const start = PAGE.indexOf("{chats.map((chat) => {");
  assert.notEqual(start, -1, "the chat list moved");
  const row = PAGE.slice(start, PAGE.indexOf("</DropdownMenu>", start));
  assert.match(row, /className="group\/chat-row /);
  assert.equal(
    (row.match(/group-hover\/chat-row:opacity-100/g) ?? []).length,
    2,
    "the pin and the menu are not both tied to the row's hover",
  );
  assert.match(row, /chatPinned \? "opacity-100" : "opacity-0",/);
  assert.match(row, /aria-label="Chat options"/);
  // role="button" makes nested pin and menu presentational for screen readers.
  assert.ok(!row.includes('role="button"'), "the row is a button again");
  assert.ok(!row.includes("tabIndex={0}"), "the row is focusable again");
  assert.ok(!row.includes("e.currentTarget"), "the row still fakes key handling");
  assert.match(
    row,
    /<button\n\s*type="button"\n\s*onClick=\{\(\) => openChat\(chat, project\.id\)\}\n\s*className="flex min-w-0 flex-1 cursor-pointer items-center gap-3/,
  );
  assert.match(
    row,
    /<span className="hidden w-40 shrink-0 sm:block">\n\s*\{formatUpdated\(chat\.updatedAt\)\}/,
  );
  assert.ok(
    PAGE.includes('className="hidden w-40 shrink-0 text-sm text-muted-foreground sm:block"'),
    "a project row still draws its date below sm",
  );
  for (const kept of [
    '<span className="shrink-0 sm:w-40">Updated</span>',
    'className="flex shrink-0 cursor-pointer items-center gap-1 text-left transition-colors hover:text-foreground sm:w-40"',
  ]) {
    assert.ok(PAGE.includes(kept), `the sort control does not survive below sm: ${kept}`);
  }
  assert.match(row, /<div className="relative flex w-8 shrink-0 items-center justify-end">/);
});

test("the chat menu writes through the shared chat helpers", () => {
  assert.match(PAGE, /await renameChatItem\(target, name\);\n\s*notifyChatHistoryUpdated\(\);/);
  assert.match(PAGE, /await archiveChatItem\(chat, activeThreadId\(\), \(\) => \{\}\);/);
  assert.match(
    PAGE,
    /await deleteChatItem\(chat, activeThreadId\(\), \(\) => \{\}, \{ deleteFiles \}\);/,
  );
  // Pins live apart from the chats, so a deleted one would leave its id stored for good.
  assert.match(
    PAGE,
    /await deleteChatItem\([\s\S]{0,240}?\n\s*unpinChat\(chat\.id\);/,
  );
  assert.match(PAGE, /const unpinChat = usePinnedChatsStore\(\(s\) => s\.unpin\);/);
  assert.match(
    PAGE,
    /confirmDeleteChats\n\s*\? openChatDelete\(chat\)\n\s*: void deleteChat\(chat, alwaysDeleteChatFiles\)/,
  );
  assert.match(PAGE, /<DialogTitle>Delete chat<\/DialogTitle>/);
  assert.match(PAGE, /<DialogTitle>Rename chat<\/DialogTitle>/);
  assert.match(PAGE, /chatExportOptions\(\)\.map\(\(\{ label, format \}\) => \(/);
  assert.match(PAGE, /void handleChatExport\(chat, format\);/);
  assert.match(PAGE, /Export all chats…/);
});

test("the chat menu carries the sidebar's items, without the move", () => {
  const start = PAGE.indexOf("{chats.map((chat) => {");
  const menu = PAGE.slice(
    PAGE.indexOf("<DropdownMenuContent", start),
    PAGE.indexOf("</DropdownMenuContent>", start),
  );
  for (const label of [
    "<span>Rename</span>",
    "<span>Archive</span>",
    "<span>Delete</span>",
    "<span>Export</span>",
    "Export all chats…",
  ]) {
    assert.ok(menu.includes(label), `${label} is missing from the chat menu`);
  }
  assert.match(menu, /chatUnread \? "shell\.selection\.markRead" : "shell\.selection\.markUnread"/);
  assert.match(menu, /<OpenChatFolderItem item=\{chat\} \/>/);
  assert.ok(!menu.includes("<span>Project</span>"));
  assert.ok(!menu.includes("moveChatToProject"));
});

test("both confirmations offer the files, and neither carries the last one's answer", () => {
  assert.match(PAGE, /const \[deleteFilesOnDelete, setDeleteFilesOnDelete\] = useState\(false\);/);
  assert.match(
    PAGE,
    /function openChatDelete\(chat: SidebarItem\) \{\n\s*setDeleteFilesOnDelete\(alwaysDeleteChatFiles\);\n\s*setDeletingChat\(chat\);/,
  );
  assert.match(
    PAGE,
    /function openProjectDelete\(project: ProjectRecord\) \{\n\s*setDeleteFilesOnDelete\(false\);\n\s*setDeleting\(project\);/,
  );
  for (const setter of ["setDeletingChat", "setDeleting"]) {
    const opens = (PAGE.match(new RegExp(`${setter}\\((?!null\\))`, "g")) ?? []).length;
    assert.equal(opens, 1, `${setter} is called outside its opener`);
  }
  assert.match(
    PAGE,
    /const deleteFiles = deleteFilesOnDelete;\n\s*setDeletingChat\(null\);\n\s*setDeleteFilesOnDelete\(false\);/,
  );
  assert.match(
    PAGE,
    /const deleteFiles = deleteFilesOnDelete;\n\s*setDeleting\(null\);\n\s*setDeleteFilesOnDelete\(false\);/,
  );
  assert.match(PAGE, /await deleteChatProject\(target\.id, \{ deleteFiles \}\);/);
  assert.match(PAGE, /id="projects-delete-chat-files"/);
  assert.match(
    PAGE,
    /id="projects-delete-project-files"[\s\S]{0,200}?deleting\?\.rootPath \?\?\n\s*"The project workspace folder will be removed from disk\."/,
  );
  assert.equal((PAGE.match(/deleteFilesOnDelete \? "Delete all" : "Delete"/g) ?? []).length, 2);
});

// role="button" makes children presentational, hiding the pin and menu controls.
test("a row is a container, and each of its controls is its own button", () => {
  const list = PAGE.slice(PAGE.indexOf("{visibleProjects.map((project) => {"));
  assert.ok(!list.includes('role="button"'), "a row still carries the button role");
  assert.ok(!list.includes("tabIndex={0}"), "a row is still focusable itself");
  assert.ok(!list.includes("e.currentTarget"), "a row still fakes key handling");
  for (const opener of [
    /<button\n\s*type="button"\n\s*onClick=\{\(\) => openProject\(project\.id\)\}\n\s*className="flex min-w-0 cursor-pointer items-center gap-3/,
    /<button\n\s*type="button"\n\s*onClick=\{\(\) => openChat\(chat, project\.id\)\}\n\s*className="flex min-w-0 flex-1 cursor-pointer items-center gap-3/,
  ]) {
    assert.match(PAGE, opener);
  }
  assert.match(
    PAGE,
    /className="group\/project-row relative flex items-center gap-3 rounded-xl px-5 py-4 text-left transition-colors duration-150 hover:bg-muted\/70 dark:hover:bg-\[rgb\(255_255_255_\/_calc\(0\.055\*var\(--contrast-wash-gain,1\)\)\)\]"/,
  );
  assert.match(
    PAGE,
    /className="group\/chat-row flex items-center gap-3 rounded-xl py-1\.5 pl-2 pr-5 text-left text-sm/,
  );
});
