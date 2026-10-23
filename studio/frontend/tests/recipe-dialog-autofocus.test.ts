// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { focusRecipeDialogOnOpen } from "../src/features/recipe-studio/dialogs/shared/recipe-dialog-focus.ts";
import { readSrc } from "./helpers/kit.ts";

const ROUTES_THROUGH_HELPER =
  /onOpenAutoFocus=\{\(event\) =>\s*focusRecipeDialogOnOpen\(event, onOpenAutoFocus\)\s*\}/;
const PROP_TAKEN_OUT_OF_SPREAD = /children,\s*onOpenAutoFocus,\s*\.\.\.props/;

function autoFocusEvent() {
  const focusCalls: unknown[] = [];
  const event = {
    defaultPrevented: false,
    preventDefault() {
      event.defaultPrevented = true;
    },
    currentTarget: {
      focus(options?: unknown) {
        focusCalls.push(options);
      },
    } as unknown as EventTarget,
  };
  return { event, focusCalls };
}

test("a recipe dialog focuses itself on open instead of its first hint button", () => {
  const { event, focusCalls } = autoFocusEvent();

  focusRecipeDialogOnOpen(event);

  assert.equal(event.defaultPrevented, true);
  assert.deepEqual(focusCalls, [{ preventScroll: true }]);
});

test("a caller's own open-focus handler runs first and keeps its choice", () => {
  const { event, focusCalls } = autoFocusEvent();
  let callerRan = false;

  focusRecipeDialogOnOpen(event, (received) => {
    assert.equal(received, event);
    callerRan = true;
    received.preventDefault();
  });

  assert.equal(callerRan, true);
  assert.deepEqual(focusCalls, []);
});

test("a caller's handler that leaves the default still gets the dialog focused", () => {
  const { event, focusCalls } = autoFocusEvent();
  let callerRan = false;

  focusRecipeDialogOnOpen(event, () => {
    callerRan = true;
  });

  assert.equal(callerRan, true);
  assert.deepEqual(focusCalls, [{ preventScroll: true }]);
});

test("every recipe dialog routes its open focus through the helper", () => {
  const source = readSrc("features/recipe-studio/dialogs/shared/recipe-dialog-content.tsx");
  assert.match(source, ROUTES_THROUGH_HELPER);
  assert.match(source, PROP_TAKEN_OUT_OF_SPREAD);
});
