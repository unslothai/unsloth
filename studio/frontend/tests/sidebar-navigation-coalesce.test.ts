// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrcAsync } from "./helpers/kit.ts";

const { createNavigationCoalescer } =
  await import("../src/components/sidebar-navigation.ts");

type Nav = { to: string; replace?: boolean };
type Entry = { href: string; key: string };

// A fake router over a history stack: push and replace mint a fresh key, as TanStack history
// does; `render()` resolves the current entry, as the router's onResolved does. With
// `blocked`, navigations wait on a blocker before touching history (the Library note save).
function fakeRouter(start = "/chat", { blocked = false } = {}) {
  let n = 0;
  const entry = (href: string): Entry => ({ href, key: `k${n++}` });
  const stack: Entry[] = [entry(start)];
  let index = 0;
  const calls: string[] = [];
  const nav = createNavigationCoalescer<Nav>({
    navigate: (o) => {
      calls.push(o.replace ? `replace ${o.to}` : o.to);
      if (!blocked) {
        if (o.replace) stack[index] = entry(o.to);
        else {
          stack.splice(index + 1, Infinity, entry(o.to));
          index += 1;
        }
      }
      return new Promise(() => {});
    },
    currentHref: () => stack[index].href,
    hrefOf: (o) => o.to,
    currentEntry: () => stack[index].key,
    asReplace: (o) => ({ ...o, replace: true }),
  });
  return {
    click: (to: string) => nav.go({ to }),
    render: () => nav.resolved(),
    back: () => {
      index -= 1;
    },
    forward: () => {
      index += 1;
    },
    hrefs: () => stack.map((e) => e.href),
    calls,
  };
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
  assert.deepEqual(r.hrefs(), ["/chat", "/images"]);
});

test("the next click after a load renders pushes a new entry", () => {
  const r = fakeRouter();
  r.click("/hub");
  r.render();
  r.click("/library");
  assert.deepEqual(r.hrefs(), ["/chat", "/hub", "/library"]);
});

test("a click while Back is still loading keeps the entry Back landed on", () => {
  const r = fakeRouter();
  r.click("/hub");
  r.render();
  r.click("/library");
  r.render();
  r.back();
  r.click("/images");
  assert.deepEqual(r.hrefs(), ["/chat", "/hub", "/images"]);
});

test("Back onto the entry a click pushed, once rendered, is not replaced", () => {
  const r = fakeRouter();
  r.click("/hub");
  r.render();
  r.click("/library");
  r.back();
  r.click("/images");
  assert.deepEqual(r.calls, ["/hub", "/library", "/images"]);
});

test("Back off an unrendered sidebar entry, then Forward onto it, keeps it", () => {
  const r = fakeRouter();
  r.click("/hub");
  r.back();
  r.render();
  r.forward();
  r.click("/images");
  assert.deepEqual(r.hrefs(), ["/chat", "/hub", "/images"]);
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
  assert.match(
    source,
    /currentEntry: \(\) => router\.latestLocation\.state\.__TSR_key/,
  );
  assert.match(source, /router\.subscribe\("onResolved"/);
});
