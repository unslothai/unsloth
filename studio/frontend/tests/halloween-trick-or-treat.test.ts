// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const store = new Map<string, string>();
const storage = {
  getItem: (k: string) => store.get(k) ?? null,
  setItem: (k: string, v: string) => void store.set(k, String(v)),
  removeItem: (k: string) => void store.delete(k),
  clear: () => store.clear(),
  key: (i: number) => [...store.keys()][i] ?? null,
  get length() {
    return store.size;
  },
};
let paletteAttr: string | null = null;
Object.assign(globalThis, {
  localStorage: storage,
  window: {
    localStorage: storage,
    matchMedia: () => ({
      matches: false,
      addEventListener: () => undefined,
      removeEventListener: () => undefined,
    }),
    addEventListener: () => undefined,
    removeEventListener: () => undefined,
  },
  document: {
    documentElement: {
      setAttribute: (_: string, v: string) => {
        paletteAttr = v;
      },
      removeAttribute: () => {
        paletteAttr = null;
      },
      classList: { toggle: () => undefined },
      style: {},
    },
  },
});

const { getPalette, setPalette } = await import(
  "../src/features/settings/stores/theme-store.ts"
);
const {
  HALLOWEEN_STORAGE_KEY,
  isHalloweenSeason,
  pickHalloweenTheme,
  pickRandomHalloweenTheme,
  removeHalloween,
  hideTrickOrTreatRow,
  showTrickOrTreatRow,
  useHalloweenStore,
} = await import("../src/features/halloween/halloween-store.ts");
const { HALLOWEEN_THEME_IDS } = await import(
  "../src/features/settings/lib/color-themes.ts"
);

const read = (path: string) =>
  readFileSync(new URL(path, import.meta.url), "utf8");

test("the season is October in local time", () => {
  assert.equal(isHalloweenSeason(new Date(2026, 9, 1, 0, 0)), true);
  assert.equal(isHalloweenSeason(new Date(2026, 9, 31, 23, 59)), true);
  assert.equal(isHalloweenSeason(new Date(2026, 10, 1, 0, 0)), false);
  assert.equal(isHalloweenSeason(new Date(2026, 8, 30, 23, 59)), false);
});

test("a Halloween pick remembers the theme from before, and removing hands it back", () => {
  setPalette("matcha");
  pickHalloweenTheme("pumpkin");
  pickHalloweenTheme("alien");
  assert.equal(getPalette(), "alien");
  assert.equal(paletteAttr, "alien");
  assert.equal(useHalloweenStore.getState().restorePalette, "matcha");

  useHalloweenStore.getState().setSpookyGreetings(true);
  removeHalloween(new Date(2026, 9, 27));
  assert.equal(getPalette(), "matcha");
  const state = useHalloweenStore.getState();
  assert.equal(state.spookyGreetings, false);
  assert.equal(state.hiddenYear, 2026);
  assert.equal(state.restorePalette, null);
});

test("the sidebar X hides the row without touching the theme, and settings brings it back", () => {
  setPalette("standard");
  pickHalloweenTheme("gothic");
  useHalloweenStore.getState().setSpookyGreetings(true);
  hideTrickOrTreatRow(new Date(2026, 9, 27));
  assert.equal(useHalloweenStore.getState().hiddenYear, 2026);
  assert.equal(getPalette(), "gothic");
  assert.equal(useHalloweenStore.getState().spookyGreetings, true);
  showTrickOrTreatRow();
  assert.equal(useHalloweenStore.getState().hiddenYear, null);
});

test("removing keeps a theme picked outside Halloween", () => {
  setPalette("plum");
  removeHalloween(new Date(2026, 9, 27));
  assert.equal(getPalette(), "plum");
});

test("a random pick is always a different Halloween theme", () => {
  setPalette("standard");
  for (let i = 0; i < 20; i++) {
    const before = getPalette();
    const id = pickRandomHalloweenTheme();
    assert.ok(HALLOWEEN_THEME_IDS.includes(id));
    assert.notEqual(id, before);
    assert.equal(getPalette(), id);
  }
  assert.equal(useHalloweenStore.getState().restorePalette, "standard");
});

test("state survives a switch of account like the theme does", () => {
  const chrome = read("../src/lib/account-transition.ts").match(
    /ACCOUNT_CHROME_KEYS = new Set\(\[([\s\S]*?)\]\)/,
  )?.[1];
  assert.ok(chrome?.includes(`"${HALLOWEEN_STORAGE_KEY}"`));
  assert.ok(chrome?.includes(`"palette"`));
});

test("every spooky greeting has English text, and named lines keep {name}", () => {
  const source = read("../src/features/halloween/spooky-greetings.ts");
  const en = read("../src/i18n/locales/en.ts");
  const keys = [...source.matchAll(/"halloween\.greeting\.(\w+)"/g)].map(
    (m) => m[1],
  );
  assert.equal(keys.length, 7);
  for (const key of keys) {
    const line = en.match(new RegExp(`\\n\\s+${key}: "([^"]+)"`))?.[1];
    assert.ok(line, key);
    assert.equal(line.includes("{name}"), key.endsWith("Name"), key);
  }
});

test("the sidebar row and dialog are gated on the season", () => {
  const sidebar = read("../src/components/app-sidebar.tsx");
  assert.match(sidebar, /\{showTrickOrTreat && \(\s*<NavItem/);
  assert.match(sidebar, /const showTrickOrTreat = useTrickOrTreatVisible\(\);/);
  assert.match(sidebar, /useHalloweenSeasonEnd\(\);/);
  // Above More, with its own hide button.
  assert.ok(
    sidebar.indexOf('testId="nav-row-trick-or-treat"') <
      sidebar.indexOf("{/* Unpinned destinations, behind one row. */}"),
  );
  assert.match(sidebar, /hideTrickOrTreatRow\(\);/);
  const appearance = read("../src/features/settings/tabs/appearance-tab.tsx");
  assert.match(appearance, /<TrickOrTreatSettingsRows \/>/);
  const select = read(
    "../src/features/settings/components/color-theme-select.tsx",
  );
  assert.match(select, /\{isHalloweenSeason\(\) && \(/);
});

test("October 30 and 31 always say Happy Halloween", async () => {
  const { buildSpookyWelcome } = await import(
    "../src/features/halloween/spooky-greetings.ts"
  );
  for (const day of [30, 31]) {
    const at = new Date(2026, 9, day, 21, 0);
    assert.equal(
      buildSpookyWelcome("", at).key,
      "halloween.greeting.happyHalloween",
    );
    assert.deepEqual(buildSpookyWelcome("Ada", at), {
      key: "halloween.greeting.happyHalloweenName",
      name: "Ada",
      sloth: "large sloth yay.png",
    });
  }
  for (let i = 0; i < 40; i++) {
    const { key } = buildSpookyWelcome("", new Date(2026, 9, 29, 23, 59));
    assert.notEqual(key, "halloween.greeting.happyHalloween");
  }
});
