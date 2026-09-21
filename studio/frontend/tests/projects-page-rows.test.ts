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
    /<span className="flex min-w-0 flex-1 items-center gap-0\.5">\n\s*<span className="min-w-0 truncate text-ui-15 font-semibold text-foreground">\n\s*\{project\.name\}/,
  );
  assert.ok(
    PAGE.indexOf('className="w-40 shrink-0 text-sm text-muted-foreground"') > chevron,
    "the arrow is still drawn after the Modified column",
  );
});

// One project, two lists: the page edits it through the dialog the sidebar opens.
test("the row menu edits a project rather than only renaming it", () => {
  assert.match(PAGE, /import \{ EditProjectDialog \} from "\.\/components\/edit-project-dialog";/);
  assert.match(PAGE, /onSelect=\{\(\) => setEditing\(project\)\}/);
  assert.match(PAGE, /<span>Edit<\/span>/);
  assert.match(PAGE, /<EditProjectDialog\n\s*project=\{editing\}/);
  // Delete still routes to this page's own confirmation.
  assert.match(PAGE, /onDelete=\{\(project\) => setDeleting\(project\)\}/);
  // And the rename-only dialog it replaces is gone, with the call it wrote through.
  assert.ok(!PAGE.includes("Rename project"));
  assert.ok(!PAGE.includes("renameChatProject"));
});

test("pinning a project takes one click, and says which way it goes", () => {
  assert.match(PAGE, /aria-label=\{pinned \? "Unpin project" : "Pin project"\}/);
  assert.match(PAGE, /togglePinProject\(project\.id\);/);
  // A pinned row keeps its pin showing; the rest reveal on hover.
  assert.match(PAGE, /pinned \? "opacity-100" : "opacity-0",/);
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

// A touch screen has no hover to reveal a row's actions with, and the menu behind the kebab was
// no way in either, since the kebab is revealed the same way.
test("a row's actions are on show where there is no hover", () => {
  const start = PAGE.indexOf("{visibleProjects.map((project) => {");
  assert.notEqual(start, -1, "the row list moved");
  // Down to the row menu's contents, so only the row's own controls are counted.
  const row = PAGE.slice(start, PAGE.indexOf("<DropdownMenuContent", start));
  assert.ok(row.length > 0, "the row moved");
  assert.equal(
    (row.match(/pointer-coarse:opacity-100/g) ?? []).length,
    3,
    "the disclosure, the pin and the menu are not all revealed on a touch screen",
  );
  // Still hover-revealed with a mouse, so a quiet row stays quiet.
  assert.match(row, /group-hover\/project-row:opacity-100 pointer-coarse:opacity-100/);
});

// "Modified" named something the row does not track, and the header sat past its own values,
// over the pin and the menu.
test("the Updated column names the list's order and turns it around", () => {
  assert.ok(!PAGE.includes(">Modified<"), "the old column name is still rendered");
  assert.match(PAGE, /\n\s*Updated\n\s*<ChevronDownIcon/);
  // One click puts the list back on this column, the next turns it around.
  assert.match(
    PAGE,
    /if \(sortMode !== "activity"\) setSortMode\("activity"\);\n\s*else setSortDir\(\(dir\) => \(dir === "desc" \? "asc" : "desc"\)\);/,
  );
  assert.match(PAGE, /sortDir === "asc" && "rotate-180",/);
  // The arrow stands for this column alone, so sorting by name hides it rather than lying.
  assert.match(PAGE, /sortMode !== "activity" && "invisible",/);
  assert.match(
    PAGE,
    /return sortDir === "asc" \? a\.updatedAt - b\.updatedAt : b\.updatedAt - a\.updatedAt;/,
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
