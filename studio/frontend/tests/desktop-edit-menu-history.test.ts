// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  editMenuKey,
  editsText,
  runEditMenuAction,
} from "../src/app/edit-menu.ts";
import { historyAction } from "../src/features/flows/editor/history-keys.ts";
import { readSrc, readText } from "./helpers/kit.ts";

const APP_MENU = readText("../../src-tauri/src/app_menu.rs");

test("the desktop Edit menu sends Undo and Redo to the renderer on their usual keys", () => {
  assert.match(APP_MENU, /\("Undo", "edit-undo", "CmdOrCtrl\+Z"\)/);
  assert.match(APP_MENU, /\("Redo", "edit-redo", "CmdOrCtrl\+Shift\+Z"\)/);
  assert.match(APP_MENU, /route_edit_history\(app, find\("Edit"\)\?\)\?;/);
  assert.match(
    readSrc("app/routes/__root.tsx"),
    /hasAppMenus \? listenEditMenu\(\)/,
  );
});

test("the replayed key is the one the Flows editor reads as undo or redo", () => {
  assert.equal(historyAction(editMenuKey("edit-undo") as never), "undo");
  assert.equal(historyAction(editMenuKey("edit-redo") as never), "redo");
});

test("text fields keep the native undo; anything else gets the key", () => {
  assert.equal(editsText({ tagName: "INPUT", type: "text" }), true);
  assert.equal(editsText({ tagName: "TEXTAREA" }), true);
  assert.equal(editsText({ tagName: "DIV", isContentEditable: true }), true);
  assert.equal(editsText({ tagName: "INPUT", type: "checkbox" }), false);
  assert.equal(
    editsText({ tagName: "INPUT", type: "text", readOnly: true }),
    false,
  );
  assert.equal(editsText({ tagName: "DIV" }), false);
  assert.equal(editsText(null), false);

  const commands: string[] = [];
  const sent: KeyboardEventInit[] = [];
  const target = {
    dispatchEvent: (e: { init: KeyboardEventInit }) => sent.push(e.init),
  };
  const doc = (active: unknown) =>
    ({
      activeElement: active,
      body: target,
      execCommand: (name: string) => commands.push(name),
    }) as unknown as Document;
  const makeKey = (init: KeyboardEventInit) => ({ init }) as unknown as Event;

  runEditMenuAction("edit-undo", doc({ tagName: "TEXTAREA" }), makeKey);
  runEditMenuAction(
    "edit-redo",
    doc({ tagName: "INPUT", type: "search" }),
    makeKey,
  );
  assert.deepEqual(commands, ["undo", "redo"]);
  assert.equal(sent.length, 0);

  runEditMenuAction("edit-undo", doc(null), makeKey);
  runEditMenuAction("edit-redo", doc({ tagName: "DIV", ...target }), makeKey);
  assert.deepEqual(
    sent.map((init) => [init.key, init.metaKey, init.shiftKey, init.bubbles]),
    [
      ["z", true, false, true],
      ["z", true, true, true],
    ],
  );
});
