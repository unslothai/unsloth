// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { isBlockedByActiveModal } from "../src/components/ui/tooltip-modal-layer.ts";

/** A trigger plus its ancestors, nearest first, by inline pointer-events. */
function trigger(own: string, ...ancestors: string[]): HTMLElement {
  let parent: HTMLElement | null = null;
  for (const pointerEvents of [...ancestors].reverse()) {
    const node = { style: { pointerEvents }, parentElement: parent };
    parent = node as unknown as HTMLElement;
  }
  return {
    style: { pointerEvents: own },
    parentElement: parent,
  } as unknown as HTMLElement;
}

test("a trigger under the modal is blocked by the body", () => {
  // sidebar button -> ... -> body(none)
  assert.equal(isBlockedByActiveModal(trigger("", "none")), true);
});

test("a trigger inside the active layer is not blocked", () => {
  // dialog content(auto) sits between the trigger and body(none)
  assert.equal(isBlockedByActiveModal(trigger("", "auto", "none")), false);
});

test("a trigger on a layer beneath the modal is blocked", () => {
  assert.equal(isBlockedByActiveModal(trigger("", "none", "none")), true);
});

test("no modal anywhere means nothing is blocked", () => {
  assert.equal(isBlockedByActiveModal(trigger("", "", "")), false);
});

test("the trigger's own pointer-events is not modal ownership", () => {
  // The MCP dropdown hint anchor is authored pointer-events-none so the row
  // stays clickable. It is still inside the dropdown's layer.
  assert.equal(isBlockedByActiveModal(trigger("none", "auto", "none")), false);
});

test("a detached element answers no rather than throwing", () => {
  assert.equal(
    isBlockedByActiveModal({ parentElement: null } as unknown as HTMLElement),
    false,
  );
});
