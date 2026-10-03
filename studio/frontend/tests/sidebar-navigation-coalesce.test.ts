// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const { createNavigationCoalescer } =
  await import("../src/components/sidebar-navigation.ts");

type Nav = { to: string; replace?: boolean };

// A fake router: `href` moves when a navigation starts, as `latestLocation` does.
function fakeRouter(start = "/chat") {
  let href = start;
  const calls: string[] = [];
  const settles: Array<(ok: boolean) => void> = [];
  const go = createNavigationCoalescer<Nav>({
    navigate: (nav) => {
      calls.push(nav.replace ? `replace ${nav.to}` : nav.to);
      href = nav.to;
      return new Promise((resolve, reject) => {
        settles.push((ok) =>
          ok ? resolve(undefined) : reject(new Error("x")),
        );
      });
    },
    currentHref: () => href,
    hrefOf: (nav) => nav.to,
    asReplace: (nav) => ({ ...nav, replace: true }),
  });
  const click = (to: string) => go({ to });
  const flush = () => new Promise((r) => setTimeout(r, 0));
  return { click, calls, settles, flush };
}

test("a click on the current page does not navigate", () => {
  const r = fakeRouter("/library");
  r.click("/library");
  assert.deepEqual(r.calls, []);
});

test("clicking the row that is still loading does not start a second load", () => {
  const r = fakeRouter();
  r.click("/hub");
  r.click("/hub");
  assert.deepEqual(r.calls, ["/hub"]);
});

test("a click during a load goes out at once and replaces the unshown entry", () => {
  const r = fakeRouter();
  r.click("/hub");
  r.click("/library");
  r.click("/images");
  assert.deepEqual(r.calls, ["/hub", "replace /library", "replace /images"]);
});

test("the next click after a load settles pushes a new entry", async () => {
  const r = fakeRouter();
  r.click("/hub");
  r.settles[0](true);
  await r.flush();
  r.click("/library");
  assert.deepEqual(r.calls, ["/hub", "/library"]);
});

test("an older load settling does not end a newer one", async () => {
  const r = fakeRouter();
  r.click("/hub");
  r.click("/library");
  r.settles[0](true);
  await r.flush();
  r.click("/images");
  assert.deepEqual(r.calls, ["/hub", "replace /library", "replace /images"]);
});

test("a failed navigation does not turn the next click into a replace", async () => {
  const r = fakeRouter();
  r.click("/hub");
  r.settles[0](false);
  await r.flush();
  r.click("/library");
  assert.deepEqual(r.calls, ["/hub", "/library"]);
});

test("sidebar rows and chat rows navigate through the coalescer", async () => {
  const source = await readSrcAsync("components/app-sidebar.tsx");
  for (const to of ["/library", "/hub", "/images", "/studio", "/projects"]) {
    assert.match(
      source,
      new RegExp(`navigateFromRow\\(\\{ to: "${to}" \\}\\)`),
    );
  }
  assert.match(
    source,
    /function openChatItem\(item: SidebarItem\) \{[\s\S]*?navigateFromRow\(\{\s*to: "\/chat",/,
  );
  assert.match(
    source,
    /asReplace: \(options\) => \(\{ \.\.\.options, replace: true \}\)/,
  );
});
