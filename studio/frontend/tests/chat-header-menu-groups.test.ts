// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const MENU = readSrc("features/chat/components/chat-header-menu.tsx");

test("the chat menu splits into naming, marking, copying the chat, moving it out, its folder, and removal", () => {
  const body = MENU.slice(
    MENU.indexOf("  return (\n    <>\n      <Item icon={Edit03Icon}"),
  );
  const outer =
    body.slice(0, body.indexOf('{t("shell.sections.moveTo")}')) +
    body.slice(body.indexOf('{t("chatMenu.copy")}'));
  const marks = [
    'regenerateTitle")}',
    'DropdownMenuSeparator className="mx-3"',
    "togglePin",
    "toggleUnread",
    'DropdownMenuSeparator className="mx-3"',
    "chatMenu.compact",
    "library.chats.menu.fork",
    'DropdownMenuSeparator className="mx-3"',
    "chatMenu.copy",
    "common.export",
    'chatFolderSessionId(menu.item) ? <DropdownMenuSeparator className="mx-3" /> : null',
    "<OpenChatFolderItem",
    "<UploadToChatFolderItem",
    'DropdownMenuSeparator className="mx-3"',
    "settings.data.library.archive",
    "common.delete",
  ];
  let at = 0;
  for (const mark of marks) {
    const next = outer.indexOf(mark, at);
    assert.ok(next >= 0, `${mark} out of order`);
    at = next + mark.length;
  }
  assert.equal(
    outer.match(/<DropdownMenuSeparator className="mx-3" \/>/g)?.length,
    5,
  );
});

test("the folder items and their separator share one condition", () => {
  const folder = readSrc("features/chat/components/open-chat-folder-item.tsx");
  assert.equal(
    folder.match(/const sandboxSessionId = chatFolderSessionId\(item\);/g)
      ?.length,
    2,
  );
  assert.match(
    folder,
    /export function chatFolderSessionId\(item: SidebarItem\): string \| undefined \{/,
  );
});
