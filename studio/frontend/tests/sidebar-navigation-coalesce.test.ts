// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const { createNavigationCoalescer } =
  await import("../src/components/sidebar-navigation.ts");

type Nav = { to: string; replace?: boolean };

// A fake router: `latest` moves when a navigation reaches history, as `latestLocation` does;
// `shown` when its load renders, as `resolvedLocation` does. With `blocked`, navigations wait
// on a blocker before touching history, as the Library note preview's save does.
function fakeRouter(start = "/chat", { blocked = false } = {}) {
  let latest = start;
  let shown = start;
  const calls: string[] = [];
  const pending: Array<() => void> = [];
  const go = createNavigationCoalescer<Nav>({
    navigate: (nav) => {
      calls.push(nav.replace ? `replace ${nav.to}` : nav.to);
      const commit = () => {
        latest = nav.to;
        pending.push(() => {
          shown = nav.to;
        });
      };
      if (!blocked) commit();
      return new Promise(() => {});
    },
    currentHref: () => latest,
    hrefOf: (nav) => nav.to,
    entryShown: () => shown === latest,
    asReplace: (nav) => ({ ...nav, replace: true }),
  });
  const click = (to: string) => go({ to });
  const render = () => pending.splice(0).forEach((f) => f());
  return { click, calls, render };
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

test("the next click after a load renders pushes a new entry", () => {
  const r = fakeRouter();
  r.click("/hub");
  r.render();
  r.click("/library");
  assert.deepEqual(r.calls, ["/hub", "/library"]);
});

test("clicks held by a blocker never replace the entry on screen", () => {
  const r = fakeRouter("/library", { blocked: true });
  r.click("/hub");
  r.click("/images");
  assert.deepEqual(r.calls, ["/hub", "/images"]);
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
  assert.match(source, /shown\.href === router\.latestLocation\.href/);
  assert.match(
    source,
    /asReplace: \(options\) => \(\{ \.\.\.options, replace: true \}\)/,
  );
});
