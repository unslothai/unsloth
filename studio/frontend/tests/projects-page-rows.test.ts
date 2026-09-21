// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Projects page is a list of folders with nothing to say about what is in them: opening a
// project was the only way to find out, and pinning one meant going through a menu.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrcAsync } from "./helpers/kit.ts";
import { formatWorkedFor } from "../src/lib/format-worked-for.ts";

const PAGE = await readSrcAsync("features/chat/projects-page.tsx");

test("a project row opens its chats in place", () => {
  assert.match(PAGE, /aria-label=\{chatsOpen \? "Hide chats" : "Show chats"\}/);
  assert.match(PAGE, /aria-expanded=\{chatsOpen\}/);
  // Loaded the first time the row is opened, and again when chat history changes.
  // Only an open loads; a close after a failed load must not.
  assert.match(PAGE, /if \(!opening\) return;\n\s*const cached = projectChats\[projectId\];\n\s*const loaded = cached !== undefined && cached !== "error";\n\s*if \(loaded && !staleProjectIds\.has\(projectId\)\) return;\n[^\n]*\n\s*loadProjectChats\(projectId, loaded\);/);
  // A failed reload keeps loaded rows, when there are some: an empty folder may have just gained
  // the chat the reload was for. A pending or failed first load becomes a retryable error.
  assert.match(PAGE, /const rows = prev\[projectId\];\n\s*if \(silent && Array\.isArray\(rows\) && rows\.length > 0\) \{\n\s*setStale\(projectId, true\);\n\s*return prev;\n\s*\}\n\s*return \{ \.\.\.prev, \[projectId\]: "error" \};/);
  // Kept rows are marked stale: the row says so, offers a retry, and its next open asks again.
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
  // Debounced, since streaming fires the event per chunk, and open rows reload in place.
  assert.match(PAGE, /timer = setTimeout\(\(\) => \{[\s\S]*?for \(const id of open\) loadProjectChats\(id, true\);\n\s*\}, PROJECT_CHATS_REFRESH_DEBOUNCE_MS\);/);
  assert.match(PAGE, /if \(!silent\) \{\n\s*setProjectChats\(\(prev\) => \(\{ \.\.\.prev, \[projectId\]: "loading" \}\)\);/);
  // A closed project's pending load is invalidated with its cache entry.
  assert.match(PAGE, /for \(const \[id, seq\] of loadSeqRef\.current\) \{\n\s*if \(!open\.has\(id\)\) loadSeqRef\.current\.set\(id, seq \+ 1\);/);
  // A response a newer request overtook is dropped.
  assert.match(PAGE, /if \(loadSeqRef\.current\.get\(projectId\) !== seq\) return;\n\s*setStale\(projectId, false\);\n\s*setProjectChats/);
  // Grouped as the sidebar groups them, newest first, and a row with none says so.
  assert.match(PAGE, /groupThreads\(threads\)\.sort\(\n\s*\(a, b\) => b\.updatedAt - a\.updatedAt,/);
  assert.match(PAGE, /No chats<\/p>/);
  // Each one opens the chat it names; a comparison opens as one.
  assert.match(PAGE, /onClick=\{\(\) => openChat\(chat, project\.id\)\}/);
  assert.match(PAGE, /search: \{ compare: item\.id, project: projectId \}/);
  // A key on a control inside the row is that control's, not the row's.
  assert.match(PAGE, /if \(e\.target !== e\.currentTarget\) return;\n\s*if \(e\.key === "Enter" \|\| e\.key === " "\)/);
});

// Out by the Modified column the arrow read as another row action, and said nothing about which
// name it belonged to.
test("the disclosure sits with the name it opens", () => {
  const name = PAGE.indexOf("{project.name}");
  const chevron = PAGE.indexOf("<ChevronDownIcon", name);
  assert.notEqual(name, -1, "the name moved");
  assert.notEqual(chevron, -1, "the disclosure moved");
  // The two share a group that takes the row's free width, so the arrow follows the name's end
  // rather than the column's.
  assert.match(
    PAGE,
    /<span className="flex min-w-0 flex-1 items-center gap-2">\n\s*<span className="min-w-0 truncate text-ui-15 font-semibold text-foreground">\n\s*\{project\.name\}/,
  );
  assert.ok(
    PAGE.indexOf(
      'className="hidden w-40 shrink-0 text-sm text-muted-foreground sm:block"',
    ) > chevron,
    "the arrow is still drawn after the Updated column",
  );
});

// One project, two lists: the page edits it through the dialog the sidebar opens.
test("the row menu edits a project rather than only renaming it", () => {
  assert.match(PAGE, /import \{ EditProjectDialog \} from "\.\/components\/edit-project-dialog";/);
  assert.match(PAGE, /onSelect=\{\(\) => setEditing\(project\)\}/);
  assert.match(PAGE, /<span>Edit<\/span>/);
  assert.match(PAGE, /<EditProjectDialog\n\s*project=\{editing\}/);
  // Delete still routes to this page's own confirmation.
  assert.match(PAGE, /onDelete=\{\(project\) => openProjectDelete\(project\)\}/);
  // And the rename-only dialog it replaces is gone, with the call it wrote through.
  assert.ok(!PAGE.includes("Rename project"));
  assert.ok(!PAGE.includes("renameChatProject"));
});

test("pinning a project takes one click, and says which way it goes", () => {
  assert.match(PAGE, /aria-label=\{pinned \? "Unpin project" : "Pin project"\}/);
  assert.match(PAGE, /togglePinProject\(project\.id\);/);
  // And the menu does not repeat the button beside it.
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

// Hover-revealed actions are invisible on a touch screen, and a row whose pin and menu only
// appear under the cursor hides what it can do.
test("the pin and the menu are on show, the disclosure follows the cursor", () => {
  const start = PAGE.indexOf("{visibleProjects.map((project) => {");
  assert.notEqual(start, -1, "the row list moved");
  // Down to the row menu's contents, so only the row's own controls are counted.
  const row = PAGE.slice(start, PAGE.indexOf("<DropdownMenuContent", start));
  assert.ok(row.length > 0, "the row moved");
  // Neither the pin nor the menu is faded out any more.
  assert.ok(!row.includes("opacity-0 transition"), "the menu is still hover-revealed");
  assert.ok(
    !row.includes('pinned ? "opacity-100" : "opacity-0"'),
    "the pin is still hover-revealed",
  );
  // The disclosure still follows the cursor, and shows outright without one.
  assert.equal(
    (row.match(/pointer-coarse:opacity-100/g) ?? []).length,
    1,
    "the disclosure is not the only control left gated on hover",
  );
  assert.match(row, /group-hover\/project-row:opacity-100 pointer-coarse:opacity-100/);
  assert.match(row, /chatsOpen \? "opacity-100" : "opacity-0",/);
});

// "Modified" named something the row does not track, and the header sat past its own values,
// over the pin and the menu.
test("the Updated column names the list's order and turns it around", () => {
  assert.ok(!PAGE.includes(">Modified<"), "the old column name is still rendered");
  // An arrow, not a chevron: it points the way the column is sorted.
  assert.match(PAGE, /\n\s*Updated\n(?:\s*\{\/\*[^\n]*\*\/\}\n)?\s*<ArrowDownIcon/);
  assert.match(PAGE, /import \{ ArrowDownIcon, ChevronDownIcon, MoreHorizontalIcon \} from "lucide-react";/);
  // Clicking it turns the order around, and it is the only thing that orders the list.
  assert.match(
    PAGE,
    /onClick=\{\(\) => setSortDir\(\(dir\) => \(dir === "desc" \? "asc" : "desc"\)\)\}/,
  );
  assert.match(PAGE, /sortDir === "asc" && "rotate-180",/);
  assert.match(
    PAGE,
    /sortDir === "asc" \? a\.updatedAt - b\.updatedAt : b\.updatedAt - a\.updatedAt,/,
  );
  // The Sort by control it replaces is gone, with the mode it carried.
  assert.ok(!PAGE.includes("Sort by"), "the sort control is still in the header");
  assert.ok(!PAGE.includes("sortMode"), "the sort mode outlived its control");
  // Search takes the room it leaves, on a row of its own where there is not enough of it:
  // beside the heading the two buttons left the field narrower than its own 56px of padding.
  assert.match(PAGE, /<div className="relative min-w-0 flex-1 sm:max-w-md">/);
  assert.match(PAGE, /className="h-9 w-full rounded-full border-none bg-muted pl-10/);
  assert.match(
    PAGE,
    /<div className="flex w-full min-w-0 items-center justify-end gap-3 sm:w-auto sm:flex-1">/,
  );
  // Newest first to begin with, as a file list opens.
  assert.match(PAGE, /useState<"desc" \| "asc">\("desc"\)/);
  // The header's trailing spacers match the row's pin and menu, so the label sits over its
  // values instead of over them.
  assert.match(
    PAGE,
    /<span className="size-7 shrink-0" \/>\n\s*<span className="w-8 shrink-0" \/>\n\s*<\/div>/,
  );
});

// "Thought for 216 seconds" is neither what the model did nor a readable duration.
test("a finished run says how long it worked, in units that read", () => {
  assert.equal(formatWorkedFor(0), "0s");
  assert.equal(formatWorkedFor(45), "45s");
  assert.equal(formatWorkedFor(60), "1m 0s");
  assert.equal(formatWorkedFor(216), "3m 36s");
  assert.equal(formatWorkedFor(3600), "1h 0m");
  assert.equal(formatWorkedFor(3840), "1h 4m");
  // Never a negative, whatever the clock did.
  assert.equal(formatWorkedFor(-5), "0s");
});

test("the reasoning header says Worked for", async () => {
  const reasoning = await readSrcAsync("components/assistant-ui/reasoning.tsx");
  assert.match(reasoning, /Worked for \{formatWorkedFor\(duration \?\? 0\)\}/);
  assert.ok(!reasoning.includes("Thought for"), "the old label is still rendered");
});

// A chat under an open project could only be opened: everything else meant going to the chat
// first. ChatGPT puts the row's own actions on the right, under the cursor.
test("a chat row carries its own actions, revealed by hovering it", async () => {
  const start = PAGE.indexOf("{chats.map((chat) => {");
  assert.notEqual(start, -1, "the chat list moved");
  const row = PAGE.slice(start, PAGE.indexOf("</DropdownMenu>", start));
  // Its own hover group, so one chat's actions do not answer for its neighbours.
  assert.match(row, /className="group\/chat-row /);
  assert.equal(
    (row.match(/group-hover\/chat-row:opacity-100/g) ?? []).length,
    2,
    "the pin and the menu are not both tied to the row's hover",
  );
  assert.match(row, /chatPinned \? "opacity-100" : "opacity-0",/);
  assert.match(row, /aria-label="Chat options"/);
  // Nested buttons are invalid, so the row is a div that still opens on Enter and Space.
  assert.match(row, /role="button"\n\s*tabIndex=\{0\}/);
  assert.match(row, /if \(e\.target !== e\.currentTarget\) return;/);
  // The same columns as a project row, so the two line up.
  // Same column as the project row above it, and gone with it on a phone, where 160px of date
  // left the title no width and pushed the actions past the screen.
  assert.match(
    row,
    /<span className="hidden w-40 shrink-0 sm:block">\n\s*\{formatUpdated\(chat\.updatedAt\)\}/,
  );
  assert.ok(
    PAGE.includes('className="hidden w-40 shrink-0 text-sm text-muted-foreground sm:block"'),
    "a project row still draws its date below sm",
  );
  // The header keeps the control, shrunk to its label: it is the only way to turn the order
  // around, and hiding it left a phone with no way to sort at all.
  for (const kept of [
    '<span className="shrink-0 sm:w-40">Updated</span>',
    'className="flex shrink-0 cursor-pointer items-center gap-1 text-left transition-colors hover:text-foreground sm:w-40"',
  ]) {
    assert.ok(PAGE.includes(kept), `the sort control does not survive below sm: ${kept}`);
  }
  assert.match(row, /<div className="relative flex w-8 shrink-0 items-center justify-end">/);
});

// The actions are the sidebar's, so a chat behaves the same wherever it is acted on.
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
  // Deleting asks first only when the setting says so, as in the sidebar, and an unconfirmed
  // delete is the one that follows the preference on its own.
  assert.match(
    PAGE,
    /confirmDeleteChats\n\s*\? openChatDelete\(chat\)\n\s*: void deleteChat\(chat, alwaysDeleteChatFiles\)/,
  );
  assert.match(PAGE, /<DialogTitle>Delete chat<\/DialogTitle>/);
  assert.match(PAGE, /<DialogTitle>Rename chat<\/DialogTitle>/);
  // The sidebar's export list, so the same chat offers the same formats either way.
  assert.match(PAGE, /chatExportOptions\(\)\.map\(\(\{ label, format \}\) => \(/);
  assert.match(PAGE, /void handleChatExport\(chat, format\);/);
  assert.match(PAGE, /Export all chats…/);
});

// Everything the sidebar's chat menu offers, minus what a project's own list already answers:
// no "Project" submenu, since these chats are in the project being looked at.
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
  assert.match(menu, /chatUnread \? "Mark as read" : "Mark as unread"/);
  assert.match(menu, /<span>Open chat folder<\/span>/);
  // Desktop opens the folder; the browser says why it cannot.
  assert.match(menu, /isTauri \? \(/);
  assert.match(menu, /<OpenChatFolderUnavailableItem \/>/);
  // The chats listed here are already in this project.
  assert.ok(!menu.includes("<span>Project</span>"));
  assert.ok(!menu.includes("moveChatToProject"));
});

// "Always delete files" makes a delete destructive past the chat itself, and a project workspace
// is a folder on disk. Both confirmations have to say so and let this one delete be told not to.
test("both confirmations offer the files, and neither carries the last one's answer", () => {
  // One switch state, seeded per delete rather than left standing.
  assert.match(PAGE, /const \[deleteFilesOnDelete, setDeleteFilesOnDelete\] = useState\(false\);/);
  assert.match(
    PAGE,
    /function openChatDelete\(chat: SidebarItem\) \{\n\s*setDeleteFilesOnDelete\(alwaysDeleteChatFiles\);\n\s*setDeletingChat\(chat\);/,
  );
  // A project workspace is bigger than a chat's sandbox, so it asks from scratch, as the sidebar does.
  assert.match(
    PAGE,
    /function openProjectDelete\(project: ProjectRecord\) \{\n\s*setDeleteFilesOnDelete\(false\);\n\s*setDeleting\(project\);/,
  );
  // Nothing else opens either dialog: every other call closes it.
  for (const setter of ["setDeletingChat", "setDeleting"]) {
    const opens = (PAGE.match(new RegExp(`${setter}\\((?!null\\))`, "g")) ?? []).length;
    assert.equal(opens, 1, `${setter} is called outside its opener`);
  }
  // Each commit reads the switch, then clears it.
  assert.match(
    PAGE,
    /const deleteFiles = deleteFilesOnDelete;\n\s*setDeletingChat\(null\);\n\s*setDeleteFilesOnDelete\(false\);/,
  );
  assert.match(
    PAGE,
    /const deleteFiles = deleteFilesOnDelete;\n\s*setDeleting\(null\);\n\s*setDeleteFilesOnDelete\(false\);/,
  );
  assert.match(PAGE, /await deleteChatProject\(target\.id, \{ deleteFiles \}\);/);
  // And both dialogs render the switch, the project one naming the folder it would remove.
  assert.match(PAGE, /id="projects-delete-chat-files"/);
  assert.match(
    PAGE,
    /id="projects-delete-project-files"[\s\S]{0,200}?deleting\?\.rootPath \?\?\n\s*"The project workspace folder will be removed from disk\."/,
  );
  assert.equal((PAGE.match(/deleteFilesOnDelete \? "Delete all" : "Delete"/g) ?? []).length, 2);
});
