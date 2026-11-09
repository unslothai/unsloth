// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  installDismissClickStreakGuard,
  shiftedSelection,
  wordBounds,
} from "../src/lib/dismiss-click-streak.ts";
import { readSrc } from "./helpers/kit.ts";

const INSTALL_ONCE =
  /useEffect\(\(\) => installDismissClickStreakGuard\(\), \[\]\);/;

type Listener = (event: Event) => void;

function fakeDoc() {
  const listeners = new Map<string, Listener[]>();
  let menuOpen = false;
  const doc = {
    addEventListener: (type: string, fn: Listener) =>
      listeners.set(type, [...(listeners.get(type) ?? []), fn]),
    removeEventListener: (type: string, fn: Listener) =>
      listeners.set(
        type,
        (listeners.get(type) ?? []).filter((f) => f !== fn),
      ),
    querySelector: () => (menuOpen ? {} : null),
  } as unknown as Document;
  const el = (inside: string[] = []) => ({
    closest: (selector: string) =>
      inside.some((s) => selector.includes(s)) ? {} : null,
  });
  const selected: number[] = [];
  const extended: [number, number][] = [];
  const select = (event: MouseEvent) => {
    selected.push(event.detail);
    return (x: number, y: number) => {
      extended.push([x, y]);
    };
  };
  const fire = (type: string, init: Record<string, unknown>) => {
    for (const fn of listeners.get(type) ?? []) fn(init as unknown as Event);
  };
  const press = (detail: number, target = el(), button = 0): boolean => {
    let prevented = false;
    const base = {
      target,
      button,
      detail,
      preventDefault: () => {
        prevented = true;
      },
    };
    for (const fn of listeners.get("pointerdown") ?? [])
      fn(base as unknown as Event);
    menuOpen = false;
    for (const fn of listeners.get("mousedown") ?? [])
      fn(base as unknown as Event);
    return prevented;
  };
  return {
    doc,
    select,
    selected,
    extended,
    fire,
    press,
    el,
    openMenu: () => {
      menuOpen = true;
    },
    count: () => [...listeners.values()].flat().length,
  };
}

test("after a menu-dismissing press, the rest of the run selects one step finer", () => {
  const f = fakeDoc();
  installDismissClickStreakGuard(f.doc, f.select);
  f.openMenu();
  assert.equal(f.press(1), false, "the dismissing press itself is untouched");
  for (const detail of [2, 3, 4, 5, 6]) assert.equal(f.press(detail), true);
  assert.deepEqual(f.selected, [2, 3, 4, 5, 6]);
});

test("a run that began by dismissing a menu still selects: caret, word, then paragraph", () => {
  assert.equal(shiftedSelection(2), "caret");
  assert.equal(shiftedSelection(3), "word");
  assert.equal(shiftedSelection(4), "paragraph");
  assert.equal(shiftedSelection(9), "paragraph");
});

test("the word under the pointer is found the same way on every platform", () => {
  const text = "Answer 12. Lorem ipsum";
  assert.deepEqual(wordBounds(text, 2), [0, 6]);
  assert.deepEqual(wordBounds(text, 0), [0, 6]);
  assert.deepEqual(wordBounds(text, 13), [11, 16]);
  assert.deepEqual(wordBounds(text, text.length), [17, 22]);
  assert.deepEqual(wordBounds("", 0), [0, 0]);
});

test("a drag on a shifted press extends the selection until the button is released", () => {
  const f = fakeDoc();
  installDismissClickStreakGuard(f.doc, f.select);
  f.openMenu();
  f.press(1);
  f.press(2);
  f.fire("mousemove", { buttons: 1, clientX: 40, clientY: 10 });
  f.fire("mousemove", { buttons: 1, clientX: 90, clientY: 30 });
  f.fire("mouseup", { button: 0 });
  f.fire("mousemove", { buttons: 0, clientX: 120, clientY: 30 });
  assert.deepEqual(f.extended, [
    [40, 10],
    [90, 30],
  ]);
});

test("pointer moves without a shifted press extend nothing", () => {
  const f = fakeDoc();
  installDismissClickStreakGuard(f.doc, f.select);
  f.press(1);
  f.press(2);
  f.fire("mousemove", { buttons: 1, clientX: 40, clientY: 10 });
  assert.deepEqual(f.extended, []);
});

test("double and triple clicks select as usual when no menu was dismissed", () => {
  const f = fakeDoc();
  installDismissClickStreakGuard(f.doc, f.select);
  assert.equal(f.press(1), false);
  assert.equal(f.press(2), false);
  assert.equal(f.press(3), false);
});

test("a new click run after a dismissal selects as usual", () => {
  const f = fakeDoc();
  installDismissClickStreakGuard(f.doc, f.select);
  f.openMenu();
  f.press(1);
  assert.equal(f.press(1), false);
  assert.equal(f.press(2), false);
});

test("presses inside the menu and in text fields keep native selection", () => {
  const f = fakeDoc();
  installDismissClickStreakGuard(f.doc, f.select);
  f.openMenu();
  f.press(1, f.el(["data-radix-popper-content-wrapper"]));
  assert.equal(f.press(2), false, "a press inside the menu dismisses nothing");

  f.openMenu();
  f.press(1);
  assert.equal(
    f.press(2, f.el(["textarea"])),
    false,
    "double-click in a field selects a word",
  );
});

test("the guard removes its listeners and is installed once for the app", () => {
  const f = fakeDoc();
  const stop = installDismissClickStreakGuard(f.doc, f.select);
  assert.equal(f.count(), 4);
  stop();
  assert.equal(f.count(), 0);
  assert.match(readSrc("app/provider.tsx"), INSTALL_ONCE);
});
