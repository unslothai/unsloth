// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The thread renders one message through a render prop rather than through assistant-ui's
// `components` map, so that the element the render prop returns can be the same object every
// time. That is what lets React skip a message subtree when the message COUNT changes, which is
// what deleting a message does to every remaining message in the thread.
//
// Two things have to hold for that, and neither is visible in the rendered output, so neither is
// caught by anything that only looks at the DOM: the component choice has to match the fallback
// chain the `components` map used to resolve through, and the element has to be shared.

import assert from "node:assert/strict";
import test from "node:test";

import { createElement } from "react";

import {
  proplessSlot,
  threadMessageKind,
} from "../src/components/assistant-ui/thread-message-slot.ts";

test("editing wins over role, for every role", () => {
  assert.equal(threadMessageKind("user", true), "edit");
  assert.equal(threadMessageKind("assistant", true), "edit");
  // A system message being edited resolved SystemEditComposer -> EditComposer in the map form,
  // and only EditComposer was ever supplied.
  assert.equal(threadMessageKind("system", true), "edit");
});

test("a message that is not being edited goes to its role's component", () => {
  assert.equal(threadMessageKind("user", false), "user");
  assert.equal(threadMessageKind("assistant", false), "assistant");
});

test("a system message that is not being edited renders nothing", () => {
  // SystemMessage -> Message -> assistant-ui's default, which returns null. Neither SystemMessage
  // nor Message was supplied, so a system message has never had a body in this thread, and
  // giving it one here would put an unstyled message into every thread that has a system prompt.
  assert.equal(threadMessageKind("system", false), "none");
});

test("the slot hands back one shared element rather than a new one per render", () => {
  const Component = () => null;
  const slot = proplessSlot(Component);

  const first = slot();
  const second = slot();

  // Identity, not equality: React skips reconciling a child whose element is the very object it
  // rendered last time. A fresh element with identical contents does not get that treatment, and
  // the message subtree under it is re-rendered instead.
  assert.equal(first, second);
  assert.notEqual(first, createElement(Component));
});

test("the slot's element carries no props", () => {
  const Component = () => null;
  const element = proplessSlot(Component)();

  assert.equal(element.type, Component);
  // assistant-ui's RenderChildrenWithAccessor only memoizes a PROPLESS element: it keys its memo
  // on the props object, which is freshly allocated per render as soon as there is one prop to
  // put in it. That is exactly how the `components={{...}}` form lost the bail-out.
  assert.deepEqual(Object.keys(element.props as object), []);
});
