// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Opening or closing a dialog registers and drops a trigger per shortcut and fires several modal
// observer batches. Each re-ask walks the whole document, which on a long chat is the cost of the
// open or close itself, so the re-asks are batched into one pass per microtask that asks each
// selector once. Press-time checks stay uncached.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const {
  isSurfaceBackgrounded,
  isSurfaceInForeground,
  registerShortcutTrigger,
  subscribeTriggers,
  triggerShortcut,
} = await import("../src/features/settings/hooks/use-shortcut.ts");

const covered = { closest: () => ({}) };
const clear = { closest: () => null };

function fakeDocument(): { queries: string[]; matches: unknown[] } {
  const state = { queries: [] as string[], matches: [clear] as unknown[] };
  (globalThis as { document?: unknown }).document = {
    querySelectorAll: (selector: string) => {
      state.queries.push(selector);
      return state.matches;
    },
  };
  return state;
}

const settle = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

test("a burst of trigger changes re-asks each subscriber once, each selector once", async () => {
  const doc = globalThis.document;
  const dom = fakeDocument();
  const seen: boolean[][] = [[], []];
  const offs = seen.map((answers) =>
    subscribeTriggers(() => {
      answers.push(isSurfaceBackgrounded(".aui-composer-input"));
      answers.push(isSurfaceBackgrounded(".aui-composer-input"));
    }, false),
  );
  const drops: (() => void)[] = [];
  try {
    for (let i = 0; i < 6; i += 1) {
      drops.push(registerShortcutTrigger("toggleSidebar", { claims: () => true, run: () => {} }));
    }
    assert.deepEqual(seen, [[], []], "nobody is asked synchronously");
    await settle();
    assert.deepEqual(seen, [[false, false], [false, false]], "one pass, every subscriber");
    assert.deepEqual(dom.queries, [".aui-composer-input"], "the pass shares one query");

    // Outside a pass the answer is live: a press reads the DOM as it is now.
    dom.matches = [covered];
    assert.equal(isSurfaceBackgrounded(".aui-composer-input"), true);
    assert.equal(isSurfaceInForeground(".aui-composer-input"), false);
    assert.equal(dom.queries.length, 3);

    drops.splice(0).forEach((drop) => drop());
    await settle();
    assert.deepEqual(seen[0].slice(2), [true, true], "the next pass asks again");
    assert.equal(dom.queries.length, 4);
  } finally {
    drops.forEach((drop) => drop());
    offs.forEach((off) => off());
    (globalThis as { document?: unknown }).document = doc;
  }
});

test("a menu trigger still asks its claims at press time", () => {
  let claims = true;
  const runs: string[] = [];
  const off = registerShortcutTrigger("findInPage", {
    claims: () => claims,
    run: () => runs.push("find"),
  });
  try {
    assert.equal(triggerShortcut("findInPage"), true);
    claims = false;
    assert.equal(triggerShortcut("findInPage"), false, "not answered from a cached pass");
    assert.deepEqual(runs, ["find"]);
  } finally {
    off();
  }
});

test("availability reads go through the per-pass answers", () => {
  const hook = readSrc("features/settings/hooks/use-shortcut.ts");
  assert.match(hook, /queueMicrotask\(flushTriggerListeners\)/);
  assert.match(hook, /passAnswers = new Map\(\);\s*try \{\s*triggerListeners\.forEach/);
  assert.match(hook, /\} finally \{\s*passAnswers = null;\s*\}/);
  assert.match(hook, /answerOncePerPass\(`available\\0\$\{id\}`/);
  assert.match(hook, /answerOncePerPass\(`backgrounded\\0\$\{selector\}`/);
  assert.match(hook, /answerOncePerPass\(`foreground\\0\$\{selector\}`/);
});

test("the preload helper looks for the CSP nonce meta in <head> only", async () => {
  const { scopePreloadNonceLookup } = await import("../vite-plugin-preload-nonce.ts");
  for (const q of ["`", '"', "'"]) {
    assert.equal(
      scopePreloadNonceLookup(`let c=document.querySelector(${q}meta[property=csp-nonce]${q}),l=c?.nonce;`),
      `let c=document.head.querySelector(${q}meta[property=csp-nonce]${q}),l=c?.nonce;`,
    );
  }
  assert.equal(scopePreloadNonceLookup("document.querySelector(`meta`)"), "document.querySelector(`meta`)");
  assert.match(readSrc("../vite.config.ts"), /katexWoff2Only\(\), preloadNonceInHead\(\)\]/);
});
