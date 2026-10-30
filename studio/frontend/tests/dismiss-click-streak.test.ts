// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installDismissClickStreakGuard } from "../src/lib/dismiss-click-streak.ts";
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
    press,
    el,
    openMenu: () => {
      menuOpen = true;
    },
    count: () => [...listeners.values()].flat().length,
  };
}

test("the clicks after a menu-dismissing press start no word or block selection", () => {
  const f = fakeDoc();
  installDismissClickStreakGuard(f.doc);
  f.openMenu();
  assert.equal(f.press(1), false, "the dismissing press itself is untouched");
  assert.equal(f.press(2), true);
  assert.equal(f.press(3), true);
});

test("double and triple clicks select as usual when no menu was dismissed", () => {
  const f = fakeDoc();
  installDismissClickStreakGuard(f.doc);
  assert.equal(f.press(1), false);
  assert.equal(f.press(2), false);
  assert.equal(f.press(3), false);
});

test("a new click run after a dismissal selects as usual", () => {
  const f = fakeDoc();
  installDismissClickStreakGuard(f.doc);
  f.openMenu();
  f.press(1);
  assert.equal(f.press(1), false);
  assert.equal(f.press(2), false);
});

test("presses inside the menu and in text fields keep native selection", () => {
  const f = fakeDoc();
  installDismissClickStreakGuard(f.doc);
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
  const stop = installDismissClickStreakGuard(f.doc);
  assert.equal(f.count(), 2);
  stop();
  assert.equal(f.count(), 0);
  assert.match(readSrc("app/provider.tsx"), INSTALL_ONCE);
});
