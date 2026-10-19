// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createHeldEdit } from "../src/features/chat/voice/held-edit.ts";

// The adapter's hold, reduced to what the click changes: a parked fragment is sent
// by itself when the flush timer fires, unless the user has clicked into the box.
function loop() {
  let box = "";
  const sent: string[] = [];
  const events: string[] = [];
  const captions: boolean[] = [];
  const edit = createHeldEdit({
    readBox: () => box,
    onChange: (owned) => captions.push(owned),
    trace: (event) => events.push(event),
  });
  let held: string | null = null;
  let timer: (() => void) | null = null;
  const send = (text: string) => {
    sent.push(text);
    box = "";
  };
  return {
    edit,
    sent,
    events,
    captions,
    get box() {
      return box;
    },
    hear(text: string, judgedUnfinished: boolean) {
      if (!edit.autoSend()) {
        box = box ? `${box} ${text}` : text;
        return;
      }
      const turn = held ? `${held} ${text}` : text;
      if (judgedUnfinished) {
        held = turn;
        box = turn;
        timer = () => {
          timer = null;
          if (held === null) return;
          const t = held;
          held = null;
          send(t);
        };
        return;
      }
      held = null;
      timer = null;
      send(turn);
    },
    click() {
      if (held !== null && edit.claim(held)) held = null;
    },
    fireTimer() {
      timer?.();
    },
    type(text: string) {
      box = text;
    },
    pressEnter() {
      const t = box;
      send(t);
      edit.sent();
    },
  };
}

test("an unclicked hold still sends itself when the timer fires", () => {
  const l = loop();
  l.hear("could you tell me just like", true);
  l.fireTimer();
  assert.deepEqual(l.sent, ["could you tell me just like"]);
});

test("a click during the hold stops the timer sending; Enter sends; the next turn auto-sends", () => {
  const l = loop();
  l.hear("could you tell me just like", true);
  l.click();
  assert.equal(l.edit.owned, true);
  l.fireTimer();
  assert.deepEqual(l.sent, []);
  assert.equal(l.box, "could you tell me just like");

  // Speaking again appends to the box and still waits.
  l.hear("the weather", false);
  assert.deepEqual(l.sent, []);
  assert.equal(l.box, "could you tell me just like the weather");

  l.type("could you tell me the weather in Paris");
  l.pressEnter();
  assert.deepEqual(l.sent, ["could you tell me the weather in Paris"]);
  assert.equal(l.edit.owned, false);

  l.hear("thanks", false);
  assert.deepEqual(l.sent, ["could you tell me the weather in Paris", "thanks"]);
  assert.deepEqual(l.events, ["hold.userEdit", "hold.userEditEnd"]);
  assert.deepEqual(l.captions, [true, false]);
});

test("clearing the box hands the turn back to the loop", () => {
  const l = loop();
  l.hear("so what about", true);
  l.click();
  l.type("");
  l.hear("never mind, hello", false);
  assert.deepEqual(l.sent, ["never mind, hello"]);
  assert.equal(l.edit.owned, false);
});

test("a click with nothing held changes nothing", () => {
  const l = loop();
  l.click();
  assert.equal(l.edit.owned, false);
  assert.equal(l.edit.claim("   "), false);
  l.hear("hello", false);
  assert.deepEqual(l.sent, ["hello"]);
  assert.deepEqual(l.events, []);
});

test("a second click while editing is not a second claim", () => {
  const edit = createHeldEdit({ readBox: () => "x" });
  assert.equal(edit.claim("x"), true);
  assert.equal(edit.claim("x"), false);
  edit.reset();
  assert.equal(edit.owned, false);
});
