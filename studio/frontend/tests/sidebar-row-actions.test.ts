// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import vm from "node:vm";
import ts from "typescript";
import * as liveThreadHead from "../src/features/chat/utils/live-thread-head.ts";
import { en } from "../src/i18n/locales/en.ts";
import { readSrcAsync } from "./helpers/kit.ts";

const APP_SIDEBAR = await readSrcAsync("components/app-sidebar.tsx");

test("row forks retain the visible branch across settings settlement", async () => {
  const source = await readSrcAsync("features/chat/components/chat-row-menu.ts");
  const javascript = ts.transpileModule(
    source.slice(source.indexOf("export async function forkChatRow("), source.indexOf("/** The sandbox sessions")),
    { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } },
  ).outputText;
  for (const open of [true, false]) {
    let visible = open;
    const unregister = liveThreadHead.registerLiveThreadView({
      threads: () => ({ getState: () => ({ mainThreadId: "main" }) }),
      threadListItem: () => ({ getState: () => ({ id: "main", remoteId: visible ? "source" : "other" }) }),
      thread: () => ({ getState: () => ({ messages: [{ id: "root" }, { id: "older-reply" }] }) }),
    });
    const calls: string[] = [];
    const context = {
      exports: {} as { forkChatRow: (item: { id: string }) => Promise<unknown> },
      crypto,
      ...liveThreadHead,
      forkChatThread: async (id: string, args: { messageId?: string }) => {
        calls.push("fork");
        assert.equal(id, "source");
        assert.equal(args.messageId, open ? "older-reply" : undefined);
      },
      settleThreadScopedSettingsForCopy: async () => {
        calls.push("settings");
        visible = false;
      },
    };
    try {
      vm.runInNewContext(javascript, context);
      await context.exports.forkChatRow({ id: "source" });
      assert.deepEqual(calls, ["settings", "fork"]);
    } finally {
      unregister();
    }
  }
});

test("every chat row offers the pin without opening a menu", () => {
  assert.match(
    APP_SIDEBAR,
    /\{variant === "recent" && \(\n\s*<button\n\s*type="button"\n\s*onClick=\{\(e\) => \{\n\s*e\.stopPropagation\(\);\n\s*togglePinnedChat\(item\.id\);/,
  );
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

test("a chat row reserves one gutter, whatever its state", () => {
  assert.match(
    APP_SIDEBAR,
    /showWorkSpinner \? "pr-16" : hasUnreadActivity \? "pr-7" : undefined,/,
  );
  assert.ok(
    !APP_SIDEBAR.includes("hasSecondaryRowAction"),
    "the gutter still branches on whether the row has a second action",
  );
  assert.equal(
    (APP_SIDEBAR.match(/group-hover\/recent-item:pr-16/g) ?? []).length,
    2,
    "the Recents gutter branched again",
  );
});

test("double-clicking a chat title renames it in place", () => {
  assert.match(
    APP_SIDEBAR,
    /onDoubleClick=\{\(event\) => \{\n\s*event\.preventDefault\(\);\n\s*event\.stopPropagation\(\);\n\s*openRenameChat\(item, true, list\?\.scope\);\n\s*\}\}/,
  );
});

test("a chat row marks itself read or unread from its own menu", () => {
  assert.match(
    APP_SIDEBAR,
    /onSelect=\{\(\) =>\n\s*alreadyUnread\n\s*\? clearThreadsUnread\(threadIds\)\n\s*: markThreadsUnread\(threadIds, rowIdByThreadId\)\n\s*\}/,
  );
  assert.ok(
    !APP_SIDEBAR.includes("disabled={alreadyUnread}"),
    "the item is still disabled on a row that carries the dot",
  );
  for (const key of ["markUnread", "markRead"]) {
    assert.equal(
      (APP_SIDEBAR.match(new RegExp(`t\\("shell\\.selection\\.${key}"\\)`, "g")) ?? [])
        .length,
      2,
      `the row and bulk menus no longer say the same thing for ${key}`,
    );
  }
  assert.match(
    APP_SIDEBAR,
    /const alreadyUnread = threadIds\.some\(\(threadId\) =>\n\s*unreadThreadIds\.has\(threadId\),\n\s*\);/,
  );
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

test("a section's chevron appears on hovering anywhere in it", () => {
  assert.equal(
    (APP_SIDEBAR.match(/group\/sb-section/g) ?? []).length,
    5,
    "a collapsible section is not its own hover group",
  );
  assert.equal(
    (APP_SIDEBAR.match(/group-hover\/sb-section:opacity-100/g) ?? []).length,
    5,
    "a section chevron still waits for its header to be hovered",
  );
  assert.match(
    APP_SIDEBAR,
    /group-hover\/sb-section:opacity-100 group-hover\/sb-collap:opacity-100 group-focus-visible\/sb-collap:opacity-100/,
  );
});

test("the chat-folder hint names what to click instead", async () => {
  const item = await readSrcAsync("features/chat/components/open-chat-folder-item.tsx");
  const hint = en.library.chats.folder.chatHint;
  assert.ok(
    !hint.includes("card that created it"),
    "the hint still sends the user to a card it never identifies",
  );
  assert.ok(
    hint.includes("download a file from the tool result that wrote it"),
    "the hint no longer says where the files can be had",
  );
  assert.match(item, /hintOverride \?\? t\("library\.chats\.folder\.chatHint"\)/);
  assert.match(item, /title=\{hint\}/);
  assert.match(item, /<TooltipContent[^>]*>\s*\{hint\}\s*<\/TooltipContent>/);
});

test("a chat row forks from its own menu", async () => {
  const ROW_MENU = await readSrcAsync(
    "features/chat/components/chat-row-menu.ts",
  );
  assert.match(
    APP_SIDEBAR,
    /t\("shell\.selection\.markUnread"\)\}\n\s*<\/span>\n\s*<\/P\.Item>\n\s*\{\/\*[^]*?\*\/\}\n\s*<P\.Separator \/>\n\s*<P\.Item\n\s*disabled=\{!canForkChatRow\(item\)/,
  );
  assert.match(
    APP_SIDEBAR,
    /<P\.Item\n\s*disabled=\{!canForkChatRow\(item\)[^]*?<span>Fork<\/span>\n\s*<\/P\.Item>\n\s*\{\/\* Projects and sections in one place[^]*?\*\/\}\n\s*<P\.Sub>/,
  );
  const rowMenu = APP_SIDEBAR.slice(
    APP_SIDEBAR.indexOf("function renderChatRowMenuItems("),
    APP_SIDEBAR.indexOf("function renderChatSidebarItem("),
  );
  assert.doesNotMatch(rowMenu, /<span>Export<\/span>|Export all chats/);
  // A comparison has two threads and no single tip to fork from.
  assert.match(ROW_MENU, /export function canForkChatRow[^]*?return item\.type === "single";/);
  assert.match(
    ROW_MENU,
    /await settleThreadScopedSettingsForCopy\(item\.id\);\n\s*try \{/,
  );
  assert.ok(!ROW_MENU.includes("messages[messages.length - 1]"));
  assert.match(
    ROW_MENU,
    /return await forkChatThread\(item\.id, \{\n\s*messageId,\n\s*newThreadId: crypto\.randomUUID\(\),/,
  );
  assert.match(
    APP_SIDEBAR,
    /setActiveThreadId\(result\.thread\.id\);\n\s*navigate\(\{ to: "\/chat", search: \{ thread: result\.thread\.id \} \}\);/,
  );
});

// A streaming chat has no settled tip, so forking would end mid-answer.
test("a row being generated into cannot be forked", async () => {
  const THREAD = await readSrcAsync("components/assistant-ui/thread.tsx");
  assert.match(
    APP_SIDEBAR,
    /disabled=\{!canForkChatRow\(item\) \|\| isGenerating \|\| forkInFlight\}/,
  );
  // Shares the thread Fork's guard; two flags would still post two forks.
  assert.match(APP_SIDEBAR, /const inFlight = useForkInFlight\.getState\(\);\n\s*if \(inFlight\.forking\) return;\n\s*inFlight\.setForking\(true\);/);
  assert.match(APP_SIDEBAR, /\} finally \{\n\s*inFlight\.setForking\(false\);/);
  assert.match(APP_SIDEBAR, /const forkInFlight = useForkInFlight\(\(s\) => s\.forking\);/);
  assert.ok(!/const useForkInFlight = create</.test(THREAD));
  assert.match(THREAD, /^\s*useForkInFlight,$/m);
  const STORE = await readSrcAsync("features/chat/utils/fork-in-flight.ts");
  assert.match(STORE, /export const useForkInFlight = create</);
});

test("closed-chat forks leave tip selection to the server", async () => {
  const ROW_MENU = await readSrcAsync(
    "features/chat/components/chat-row-menu.ts",
  );
  const ROUTE = await readSrcAsync(
    "../../backend/routes/chat_history.py",
  );
  assert.ok(!ROW_MENU.includes("getActiveGenerations"));
  assert.ok(!ROW_MENU.includes("listStoredChatMessages(item.id)"));
  const fork = ROUTE.slice(ROUTE.indexOf("def fork_thread("));
  assert.match(fork, /branch_message_id = payload\.messageId/);
  assert.match(fork, /except ChatForkActiveGenerationError/);
  // Its 409 is a refusal, not a failure.
  assert.match(ROW_MENU, /if \(message\.includes\("still generating"\)\) throw forkRefused\(\);/);
  assert.match(ROW_MENU, /\{ unslothForkRefused: true \}/);
  assert.match(APP_SIDEBAR, /\?\.unslothForkRefused\) \{\n\s*toast\.info\(/);
});
