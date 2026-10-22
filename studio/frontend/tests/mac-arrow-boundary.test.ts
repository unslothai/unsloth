// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  type ArrowPress,
  type CaretField,
  arrowCannotMoveCaret,
  watchMacArrowBoundary,
} from "../src/lib/mac-arrow-boundary.ts";

const MAIN = await readFile(new URL("../src/main.tsx", import.meta.url), "utf8");

function press(key: string, extra: Partial<ArrowPress> = {}): ArrowPress {
  return { key, altKey: false, ctrlKey: false, metaKey: false, shiftKey: false, ...extra };
}

function caret(value: string, at: number, end = at): CaretField {
  return { value, selectionStart: at, selectionEnd: end };
}

test("left at the start and right at the end cannot move the caret", () => {
  assert.equal(arrowCannotMoveCaret(press("ArrowLeft"), caret("hello", 0)), true);
  assert.equal(arrowCannotMoveCaret(press("ArrowRight"), caret("hello", 5)), true);
});

test("arrows that move the caret are left alone", () => {
  assert.equal(arrowCannotMoveCaret(press("ArrowLeft"), caret("hello", 3)), false);
  assert.equal(arrowCannotMoveCaret(press("ArrowRight"), caret("hello", 0)), false);
  assert.equal(arrowCannotMoveCaret(press("ArrowLeft"), caret("hello", 5)), false);
});

test("an empty field is both ends at once", () => {
  for (const key of ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"]) {
    assert.equal(arrowCannotMoveCaret(press(key), caret("", 0)), true, key);
  }
});

test("up at the start and down at the end cannot move the caret either", () => {
  assert.equal(arrowCannotMoveCaret(press("ArrowUp"), caret("a\nb", 0)), true);
  assert.equal(arrowCannotMoveCaret(press("ArrowDown"), caret("a\nb", 3)), true);
  assert.equal(arrowCannotMoveCaret(press("ArrowUp"), caret("a\nb", 3)), false);
  assert.equal(arrowCannotMoveCaret(press("ArrowDown"), caret("a\nb", 0)), false);
});

test("a selection still collapses on an arrow", () => {
  assert.equal(arrowCannotMoveCaret(press("ArrowLeft"), caret("hello", 0, 5)), false);
  assert.equal(arrowCannotMoveCaret(press("ArrowRight"), caret("hello", 0, 5)), false);
});

test("modified arrows and IME composition are left to the browser", () => {
  for (const modifier of ["altKey", "ctrlKey", "metaKey", "shiftKey", "isComposing"] as const) {
    assert.equal(
      arrowCannotMoveCaret(press("ArrowLeft", { [modifier]: true }), caret("hello", 0)),
      false,
      modifier,
    );
  }
});

test("right to left fields swap the ends", () => {
  assert.equal(arrowCannotMoveCaret(press("ArrowLeft"), caret("שלום", 4), true), true);
  assert.equal(arrowCannotMoveCaret(press("ArrowRight"), caret("שלום", 0), true), true);
  assert.equal(arrowCannotMoveCaret(press("ArrowLeft"), caret("שלום", 0), true), false);
});

test("other keys and fields without a caret are untouched", () => {
  assert.equal(arrowCannotMoveCaret(press("Home"), caret("hello", 0)), false);
  assert.equal(
    arrowCannotMoveCaret(press("ArrowLeft"), { value: "", selectionStart: null, selectionEnd: null }),
    false,
  );
});

type Listener = (event: KeyboardEvent) => void;

function fakeWindow(direction = "ltr") {
  let listener: Listener | null = null;
  const win = {
    addEventListener: (type: string, fn: Listener) => {
      assert.equal(type, "keydown");
      listener = fn;
    },
    getComputedStyle: () => ({ direction }),
  } as unknown as Window;
  watchMacArrowBoundary(win);
  return (key: string, target: unknown, defaultPrevented = false) => {
    let prevented = defaultPrevented;
    const event = {
      key,
      altKey: false,
      ctrlKey: false,
      metaKey: false,
      shiftKey: false,
      isComposing: false,
      target,
      get defaultPrevented() {
        return prevented;
      },
      preventDefault: () => {
        prevented = true;
      },
    } as unknown as KeyboardEvent;
    listener?.(event);
    return prevented;
  };
}

function textarea(value: string, at: number) {
  return { tagName: "TEXTAREA", value, selectionStart: at, selectionEnd: at };
}

test("the guard marks only a stuck arrow in a text field as handled", () => {
  const send = fakeWindow();
  assert.equal(send("ArrowLeft", textarea("hello", 0)), true);
  assert.equal(send("ArrowRight", textarea("hello", 5)), true);
  assert.equal(send("ArrowLeft", textarea("hello", 2)), false);
  assert.equal(send("ArrowLeft", { tagName: "DIV" }), false);
  assert.equal(send("ArrowLeft", { tagName: "INPUT", value: "", selectionStart: null, selectionEnd: null }), false);
});

test("the guard reads the field's direction", () => {
  const send = fakeWindow("rtl");
  assert.equal(send("ArrowLeft", textarea("שלום", 4)), true);
  assert.equal(send("ArrowLeft", textarea("שלום", 0)), false);
});

test("an arrow the app already handled is not touched again", () => {
  const send = fakeWindow();
  assert.equal(send("ArrowLeft", textarea("hello", 0), true), true);
});

test("the guard is installed only in the macOS desktop app", () => {
  assert.match(MAIN, /if \(isTauri && \/Mac\/\.test\(navigator\.platform\)\) \{\s*watchMacArrowBoundary\(window\);/);
});
