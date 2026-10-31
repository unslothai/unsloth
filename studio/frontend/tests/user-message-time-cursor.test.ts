// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const TIME = readSrc("components/assistant-ui/user-message-time.tsx");
const TRIGGER_CLASS_RE = /className="(aui-user-message-time-trigger [^"]*)"/;

test("a sent prompt's time keeps the arrow cursor, since clicking it does nothing", () => {
  const classes = TRIGGER_CLASS_RE.exec(TIME)?.[1] ?? "";
  assert.ok(classes, "could not find the time trigger");
  // Important, so the pointer cursors setting does not bring the hand back.
  assert.match(classes, /(^| )cursor-default!( |$)/);
  assert.doesNotMatch(classes, /cursor-pointer/);
  assert.doesNotMatch(TIME, /onClick=/);
});
