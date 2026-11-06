// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

type Timer = () => void;

function fakeDom() {
  const classes = new Set<string>(["light"]);
  const attrs = new Map<string, string>();
  const log: string[] = [];
  const timers: Timer[] = [];
  const root = {
    classList: {
      add: (c: string) => classes.add(c),
      remove: (c: string) => classes.delete(c),
      contains: (c: string) => classes.has(c),
      toggle: (c: string, on: boolean) => {
        if (on !== classes.has(c)) log.push(`${on ? "+" : "-"}${c}`);
        if (on) classes.add(c);
        else classes.delete(c);
      },
    },
    style: { colorScheme: "light" },
    getAttribute: (n: string) => attrs.get(n) ?? null,
    setAttribute: (n: string, v: string) => {
      log.push(`${n}=${v}`);
      attrs.set(n, v);
    },
    removeAttribute: (n: string) => {
      log.push(`-${n}`);
      attrs.delete(n);
    },
  };
  const g = globalThis as Record<string, unknown>;
  g.document = { documentElement: root, body: {} };
  g.localStorage = { getItem: () => null, setItem: () => {} };
  g.window = {
    localStorage: g.localStorage,
    matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
    addEventListener() {},
    removeEventListener() {},
    getComputedStyle: () => {
      log.push(`flush(switching=${classes.has("theme-switching")})`);
      return { color: "" };
    },
    setTimeout: (fn: Timer) => {
      timers.push(fn);
      return 0;
    },
  };
  return { classes, log, timers };
}

test("a light/dark flip commits the new colours with transitions off, then turns them back on", async () => {
  const dom = fakeDom();
  const store = await import("../src/features/settings/stores/theme-store.ts");

  store.setTheme("dark");
  assert.deepEqual(dom.log, ["+dark", "-light", "flush(switching=true)"]);
  assert.ok(dom.classes.has("theme-switching"), "transitions stay off until the queued task");
  dom.timers.splice(0).forEach((fn) => fn());
  assert.ok(!dom.classes.has("theme-switching"));

  dom.log.length = 0;
  store.setTheme("dark");
  assert.deepEqual(dom.log, [], "re-applying the same theme forces no style flush");

  store.setPalette("standard");
  assert.deepEqual(dom.log, [], "the standard palette is already the attribute-free default");
});

test("the switching rule outranks both reduced-motion blocks", () => {
  const css = readSrc("index.css");
  assert.match(
    css,
    /:root\.theme-switching \*,\s*:root\.theme-switching \*::before,\s*:root\.theme-switching \*::after \{\s*transition: none !important;\s*\}/,
    ":root plus a class (0,2,0) beats html.force-reduced-motion * and html:not(.force-motion) * (0,1,1), which give every element a 0.01ms `all` transition",
  );
});
