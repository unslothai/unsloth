// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type TurnEnd = typeof import("../src/features/chat/voice/turn-end.ts");
const load = () =>
  loadWithStubs<TurnEnd>(
    new URL("../src/features/chat/voice/turn-end.ts", import.meta.url),
    {
      "@/features/auth": {
        authFetch: () => Promise.reject(new Error("unused")),
      },
    },
  );

test("a transcript ending on a question mark reads as a finished turn", async () => {
  const { readsAsQuestion } = await load();
  for (const text of [
    "When did the first moon landing happen, and what time was it?",
    "What's my name? ",
    'He said "really?"',
    "你叫什么名字？",
    "ما اسمي؟",
  ]) {
    assert.ok(readsAsQuestion(text), text);
  }
  for (const text of ["What I want is", "Is it?  And then", "Thanks.", ""]) {
    assert.ok(!readsAsQuestion(text), text);
  }
});
