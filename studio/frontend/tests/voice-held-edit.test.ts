// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createHeldEdit } from "../src/features/chat/voice/held-edit.ts";

test("a claimed fragment is not auto-sent until the user sends it", () => {
  let box = "could you tell me just like";
  const events: string[] = [];
  const captions: boolean[] = [];
  const edit = createHeldEdit({
    readBox: () => box,
    onChange: (owned) => captions.push(owned),
    trace: (event) => events.push(event),
  });
  assert.equal(edit.claim(box), true);
  assert.equal(edit.autoSend(), false);
  box = "could you tell me the weather in Paris";
  assert.equal(edit.autoSend(), false);
  edit.sent();
  assert.equal(edit.owned, false);
  assert.equal(edit.autoSend(), true);
  assert.deepEqual(events, ["hold.userEdit", "hold.userEditEnd"]);
  assert.deepEqual(captions, [true, false]);
});

test("a click with nothing held changes nothing", () => {
  const events: string[] = [];
  const edit = createHeldEdit({ readBox: () => "", trace: (event) => events.push(event) });
  assert.equal(edit.claim("   "), false);
  assert.equal(edit.owned, false);
  assert.equal(edit.autoSend(), true);
  assert.deepEqual(events, []);
});

test("a second click while editing is not a second claim", () => {
  const edit = createHeldEdit({ readBox: () => "x" });
  assert.equal(edit.claim("x"), true);
  assert.equal(edit.claim("x"), false);
  edit.reset();
  assert.equal(edit.owned, false);
});

test("an emptied edit hands the turn back and reports itself once, a sent one does not", () => {
  let box = "draft";
  let cleared = 0;
  const edit = createHeldEdit({ readBox: () => box, onCleared: () => (cleared += 1) });
  assert.equal(edit.claim("draft"), true);
  assert.equal(edit.autoSend(), false);
  box = "";
  assert.equal(edit.autoSend(), true);
  assert.equal(edit.owned, false);
  assert.equal(edit.autoSend(), true);
  assert.equal(cleared, 1);
  box = "again";
  edit.claim("again");
  edit.sent();
  box = "";
  edit.autoSend();
  assert.equal(cleared, 1);
});
