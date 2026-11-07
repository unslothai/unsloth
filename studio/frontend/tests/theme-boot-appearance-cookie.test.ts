// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

import { readAppearanceCookie } from "../src/features/settings/stores/theme-store.ts";

const read = (path: string) =>
  readFileSync(new URL(path, import.meta.url), "utf8");

const themeBoot = read("../public/theme-boot.js");
const backendSettings = read("../../backend/routes/settings.py");

type Boot = { classes: Set<string>; palette: string | null; scheme: string };

// Runs the real boot script against a stub document; null storage means localStorage throws.
function boot(
  storage: Record<string, string> | null,
  cookie: string,
  osDark = false,
): Boot {
  const classes = new Set<string>();
  const attrs = new Map<string, string>();
  const style: Record<string, unknown> = { setProperty() {} };
  const context = {
    localStorage:
      storage === null
        ? {
            getItem() {
              throw new Error("blocked");
            },
          }
        : { getItem: (key: string) => storage[key] ?? null },
    document: {
      cookie,
      documentElement: {
        classList: {
          toggle: (name: string, on: boolean) =>
            on ? classes.add(name) : classes.delete(name),
          add: (name: string) => classes.add(name),
        },
        setAttribute: (name: string, value: string) => attrs.set(name, value),
        style,
      },
    },
    matchMedia: () => ({ matches: osDark, addEventListener() {} }),
    navigator: { userAgent: "Windows" },
    devicePixelRatio: 1,
  };
  vm.runInNewContext(themeBoot, context);
  return {
    classes,
    palette: attrs.get("data-palette") ?? null,
    scheme: String(style.colorScheme),
  };
}

test("a browser without stored theme keys paints the account's cookie appearance", () => {
  const result = boot({}, "other=1; unsloth_appearance=dark.matcha; x=y");
  assert.ok(result.classes.has("dark"));
  assert.ok(!result.classes.has("light"));
  assert.equal(result.scheme, "dark");
  assert.equal(result.palette, "matcha");
});

test("stored localStorage keys still win over the cookie", () => {
  const result = boot(
    { theme: "light", palette: "peach" },
    "unsloth_appearance=dark.matcha",
  );
  assert.ok(result.classes.has("light"));
  assert.equal(result.palette, "peach");
});

test("each missing key falls back to the cookie on its own", () => {
  const result = boot({ theme: "light" }, "unsloth_appearance=dark.matcha");
  assert.ok(result.classes.has("light"));
  assert.equal(result.palette, "matcha");
});

test("blocked localStorage uses the cookie", () => {
  const result = boot(null, "unsloth_appearance=dark.wasabi");
  assert.ok(result.classes.has("dark"));
  assert.equal(result.palette, "wasabi");
});

test("no cookie keeps following the OS preference", () => {
  assert.ok(boot({}, "").classes.has("light"));
  assert.ok(boot({}, "", true).classes.has("dark"));
});

test("a system theme in the cookie follows the OS, and unknown values are ignored", () => {
  assert.ok(boot({}, "unsloth_appearance=system.standard", true).classes.has("dark"));
  const junk = boot({}, "unsloth_appearance=neon.not-a-palette");
  assert.ok(junk.classes.has("light"));
  assert.equal(junk.palette, null);
  // A different cookie whose name merely ends with ours is not read.
  assert.ok(boot({}, "xunsloth_appearance=dark.matcha").classes.has("light"));
});

test("the theme store parses the cookie the same way as the boot script", () => {
  for (const cookie of [
    "unsloth_appearance=dark.matcha",
    "a=b; unsloth_appearance=light.butterfly-pea; c=d",
    "xunsloth_appearance=dark.matcha",
    "unsloth_appearance=dark",
    "",
  ]) {
    const store = readAppearanceCookie(cookie);
    const result = boot({}, cookie);
    if (store === null) {
      assert.ok(result.classes.has("light"), cookie);
      assert.equal(result.palette, null, cookie);
    } else {
      assert.ok(result.classes.has(store.theme), cookie);
      assert.equal(result.palette, store.palette, cookie);
    }
  }
});

test("the cookie name matches the one the backend sets", () => {
  assert.match(backendSettings, /APPEARANCE_COOKIE = "unsloth_appearance"/);
  assert.match(themeBoot, /unsloth_appearance=/);
});
