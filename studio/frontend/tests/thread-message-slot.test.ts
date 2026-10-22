// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A render prop lets the same element object be returned, so React skips unchanged messages.
// Component choice must match the old `components` map fallback chain.

import assert from "node:assert/strict";
import test from "node:test";

import { createElement } from "react";

import {
  proplessSlot,
  rendersAsRow,
  threadMessageKind,
} from "../src/components/assistant-ui/thread-message-slot.ts";

test("editing wins over role, for every role", () => {
  assert.equal(threadMessageKind("user", true), "edit");
  assert.equal(threadMessageKind("assistant", true), "edit");
  assert.equal(threadMessageKind("system", true), "edit");
});

test("a message that is not being edited goes to its role's component", () => {
  assert.equal(threadMessageKind("user", false), "user");
  assert.equal(threadMessageKind("assistant", false), "assistant");
});

test("a system message that is not being edited renders nothing", () => {
  // The old map resolved system messages to assistant-ui's default, which renders null.
  assert.equal(threadMessageKind("system", false), "none");
});

test("user and assistant are the only roles that paint a row", () => {
  // storage/studio_db.py keeps the same role pair for fork dividers; update both together.
  const roles = ["user", "assistant", "system"] as const;
  assert.deepEqual(
    roles.filter((role) => rendersAsRow(role, false)),
    ["user", "assistant"],
  );
});

test("the slot hands back one shared element rather than a new one per render", () => {
  const Component = () => null;
  const slot = proplessSlot(Component);

  const first = slot();
  const second = slot();

  // Identity, not equality: React skips only the very same element object.
  assert.equal(first, second);
  assert.notEqual(first, createElement(Component));
});

test("the slot's element carries no props", () => {
  const Component = () => null;
  const element = proplessSlot(Component)();

  assert.equal(element.type, Component);
  // RenderChildrenWithAccessor only memoizes a PROPLESS element.
  assert.deepEqual(Object.keys(element.props as object), []);
});
