// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const { createNavigationCoalescer } = await import(
  "../src/components/sidebar-navigation.ts"
);

// A fake router: `href` moves when a navigation starts, as `latestLocation` does.
function fakeRouter(start = "/chat") {
  let href = start;
  const calls: string[] = [];
  const settles: Array<(ok: boolean) => void> = [];
  const go = createNavigationCoalescer<string>({
    navigate: (to) => {
      calls.push(to);
      href = to;
      return new Promise((resolve, reject) => {
        settles.push((ok) => (ok ? resolve(undefined) : reject(new Error("x"))));
      });
    },
    currentHref: () => href,
    hrefOf: (to) => to,
  });
  const flush = () => new Promise((r) => setTimeout(r, 0));
  return { go, calls, settles, flush };
}

test("a click on the current page does not navigate", () => {
  const r = fakeRouter("/library");
  r.go("/library");
  assert.deepEqual(r.calls, []);
});

test("spam clicks during a load collapse to the last one", async () => {
  const r = fakeRouter();
  for (let i = 0; i < 10; i++) {
    for (const to of ["/library", "/hub", "/images"]) r.go(to);
  }
  r.go("/hub");
  assert.deepEqual(r.calls, ["/library"]);
  r.settles[0](true);
  await r.flush();
  assert.deepEqual(r.calls, ["/library", "/hub"]);
});

test("clicking the row that is still loading does not start a second load", async () => {
  const r = fakeRouter();
  r.go("/hub");
  r.go("/hub");
  r.settles[0](true);
  await r.flush();
  assert.deepEqual(r.calls, ["/hub"]);
});

test("a failed navigation still lets the next click through", async () => {
  const r = fakeRouter();
  r.go("/hub");
  r.settles[0](false);
  await r.flush();
  r.go("/library");
  assert.deepEqual(r.calls, ["/hub", "/library"]);
});

test("sidebar rows and chat rows navigate through the coalescer", async () => {
  const source = await readSrcAsync("components/app-sidebar.tsx");
  for (const to of ["/library", "/hub", "/images", "/studio", "/projects"]) {
    assert.match(source, new RegExp(`navigateFromRow\\(\\{ to: "${to}" \\}\\)`));
  }
  assert.match(
    source,
    /function openChatItem\(item: SidebarItem\) \{[\s\S]*?navigateFromRow\(\{\s*to: "\/chat",/,
  );
});
