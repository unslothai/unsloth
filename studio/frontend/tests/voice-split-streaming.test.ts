// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { splitStreaming } from "../src/features/chat/voice/speech-text.ts";

const reply =
  "Egypt gets only about 2.5 centimeters of rain a year? It's pretty wild, without the Nile it would be desert!";

test("a number cut at its decimal point is not a finished sentence", () => {
  const cut = reply.slice(0, reply.indexOf("2.") + 2);
  assert.deepEqual(splitStreaming(cut).complete, []);
});

test("every streamed prefix agrees with the finished reply's sentences", () => {
  const final = splitStreaming(reply).complete;
  for (let i = 1; i <= reply.length; i++) {
    const { complete } = splitStreaming(reply.slice(0, i));
    assert.deepEqual(complete, final.slice(0, complete.length), `prefix ${i}`);
  }
});
