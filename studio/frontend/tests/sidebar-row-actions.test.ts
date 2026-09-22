// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// What a sidebar row offers without opening its menu, and what the menu says when it is opened.
// A row action that only some rows carry, or a label that repeats the row it is on, is the kind
// of thing that reads as missing rather than as absent.

import assert from "node:assert/strict";
import test from "node:test";
import { readSrcAsync } from "./helpers/kit.ts";

const APP_SIDEBAR = await readSrcAsync("components/app-sidebar.tsx");

// The pin used to appear on a Recents row only once it was pinned, so the only way to pin one was
// through the menu, while the project rows beside it had the one-click affordance all along.
test("every chat row offers the pin without opening a menu", () => {
  assert.match(
    APP_SIDEBAR,
    /\{variant === "recent" && \(\n\s*<button\n\s*type="button"\n\s*onClick=\{\(e\) => \{\n\s*e\.stopPropagation\(\);\n\s*togglePinnedChat\(item\.id\);/,
  );
  // It says which way it goes, in its glyph and to a screen reader.
  assert.ok(
    !APP_SIDEBAR.includes('{variant === "recent" && isPinned && ('),
    "the Recents pin is still gated on the row already being pinned",
  );
  assert.equal(
    (
      APP_SIDEBAR.match(
        /aria-label=\{isPinned \? "Unpin chat" : "Pin chat"\}/g,
      ) ?? []
    ).length,
    2,
    "a pin action stopped naming the state it moves to",
  );
  assert.equal(
    (
      APP_SIDEBAR.match(
        /icon=\{isPinned \? PinOffIcon : PinIcon\} strokeWidth=\{1\.75\} className="size-icon"/g,
      ) ?? []
    ).length,
    3,
    "a pin glyph no longer follows the row's state",
  );
});

// Two actions overlay the right edge of every chat row now, so the room they need is the same on
// every one of them rather than something the pinned rows alone reserved.
test("a chat row reserves one gutter, whatever its state", () => {
  assert.match(
    APP_SIDEBAR,
    /showWorkSpinner \? "pr-16" : hasUnreadActivity \? "pr-7" : undefined,/,
  );
  assert.ok(
    !APP_SIDEBAR.includes("hasSecondaryRowAction"),
    "the gutter still branches on whether the row has a second action",
  );
  // One hover gutter for Recents and Pinned, one for the project rows, and one for the folders.
  assert.equal(
    (APP_SIDEBAR.match(/group-hover\/recent-item:pr-16/g) ?? []).length,
    2,
    "the Recents gutter branched again",
  );
});

// Renaming from the menu is two clicks and a read; the title is right there.
test("double-clicking a chat title renames it in place", () => {
  assert.match(
    APP_SIDEBAR,
    /onDoubleClick=\{\(event\) => \{\n\s*event\.preventDefault\(\);\n\s*event\.stopPropagation\(\);\n\s*openRenameChat\(item, true, list\?\.scope\);\n\s*\}\}/,
  );
});

// Marking one chat unread was only reachable by selecting it first and using the bulk menu, and
// the dot could never be taken off again: the item was disabled on the rows that carried one.
test("a chat row marks itself read or unread from its own menu", () => {
  assert.match(
    APP_SIDEBAR,
    /onSelect=\{\(\) =>\n\s*alreadyUnread\n\s*\? clearThreadsUnread\(threadIds\)\n\s*: markThreadsUnread\(threadIds, rowIdByThreadId\)\n\s*\}/,
  );
  assert.ok(
    !APP_SIDEBAR.includes("disabled={alreadyUnread}"),
    "the item is still disabled on a row that carries the dot",
  );
  // The same strings the bulk menu uses, so the two cannot drift apart.
  for (const key of ["markUnread", "markRead"]) {
    assert.equal(
      (APP_SIDEBAR.match(new RegExp(`t\\("shell\\.selection\\.${key}"\\)`, "g")) ?? [])
        .length,
      2,
      `the row and bulk menus no longer say the same thing for ${key}`,
    );
  }
  // Both go by the dot the row already draws.
  assert.match(
    APP_SIDEBAR,
    /const alreadyUnread = threadIds\.some\(\(threadId\) =>\n\s*unreadThreadIds\.has\(threadId\),\n\s*\);/,
  );
  // An eye that is open once the row is read, crossed out while it is not.
  assert.equal(
    (
      APP_SIDEBAR.match(
        /icon=\{(alreadyUnread|allSelectedUnread) \? ViewIcon : ViewOffSlashIcon\}/g,
      ) ?? []
    ).length,
    2,
    "a read or unread item stopped naming the state it moves to",
  );
});

// The bulk menu only ever added dots, so a selection of read rows had no way back.
test("a selection of unread rows can be marked read", () => {
  assert.match(
    APP_SIDEBAR,
    /const allSelectedUnread =\n\s*selectionCount > 0 &&\n\s*selectedChatItems\.every\(/,
  );
  assert.match(
    APP_SIDEBAR,
    /function markSelectedRead\(\) \{\n\s*const threadIds = selectedChatItems\.flatMap\(getSidebarItemThreadIds\);\n\s*clearSelection\(\);\n\s*clearThreadsUnread\(threadIds\);\n\s*\}/,
  );
});

// Two headers carrying the same pair of actions in opposite orders is the kind of thing the eye
// catches without being able to name it.
test("a section header puts its own action before the menu", () => {
  const projects = APP_SIDEBAR.indexOf('aria-label="New project"');
  const projectsMenu = APP_SIDEBAR.indexOf(
    't("shell.organize.organizeProjects")',
  );
  assert.ok(projects > 0 && projectsMenu > 0);
  assert.ok(projects < projectsMenu, "Projects still opens with its menu");
  const newChat = APP_SIDEBAR.indexOf(
    'aria-label={t("shell.navigation.newChat")}',
  );
  const recentsMenu = APP_SIDEBAR.indexOf('t("shell.organize.organizeChats")');
  assert.ok(newChat > 0 && recentsMenu > 0);
  assert.ok(newChat < recentsMenu, "Recents still opens with its menu");
});

// A menu opened from a chat row is already about that chat: "Pin chat" there only repeats what
// the pointer just pointed at.
test("the pin item says Pin, not what it is pinning", () => {
  assert.match(APP_SIDEBAR, /<span>\{isPinned \? "Unpin" : "Pin"\}<\/span>/);
  assert.match(
    APP_SIDEBAR,
    /<span>\{isProjectPinned \? "Unpin" : "Pin"\}<\/span>/,
  );
  for (const gone of [
    '<span>{isPinned ? "Unpin chat" : "Pin chat"}</span>',
    '<span>{isProjectPinned ? "Unpin project" : "Pin project"}</span>',
  ]) {
    assert.ok(!APP_SIDEBAR.includes(gone), `${gone} is still in a menu`);
  }
});

// The disclosure is the section's, so the section is what reveals it: hunting for a chevron means
// travelling to the header when the pointer is already in the list it collapses.
test("a section's chevron appears on hovering anywhere in it", () => {
  assert.equal(
    (APP_SIDEBAR.match(/group\/sb-section/g) ?? []).length,
    4,
    "a collapsible section is not its own hover group",
  );
  assert.equal(
    (APP_SIDEBAR.match(/group-hover\/sb-section:opacity-100/g) ?? []).length,
    4,
    "a section chevron still waits for its header to be hovered",
  );
  // Hovering the header itself still counts, and so does reaching it by keyboard.
  assert.match(
    APP_SIDEBAR,
    /group-hover\/sb-section:opacity-100 group-hover\/sb-collap:opacity-100 group-focus-visible\/sb-collap:opacity-100/,
  );
});

// "the card that created it" named nothing the user can point at.
test("the chat-folder hint names what to click instead", async () => {
  // The item is shared with the Projects page's chat rows, so the hint lives with it.
  const item = await readSrcAsync("features/chat/components/open-chat-folder-item.tsx");
  assert.ok(
    !item.includes("card that created it"),
    "the hint still sends the user to a card it never identifies",
  );
  assert.ok(
    item.includes("download a file from the tool result that wrote it"),
    "the hint no longer says where the files can be had",
  );
  // Carried twice on purpose: the tooltip is for the pointer, the title for everything else.
  assert.equal(
    (item.match(/Only the desktop app can open a chat('|&apos;)s files folder/g) ??
      []).length,
    2,
    "the hint is no longer stated for both the pointer and the screen reader",
  );
});
