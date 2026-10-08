// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { clearOfWindowChrome } from "../src/lib/window-chrome.ts";
import { readSrcAsync } from "./helpers/kit.ts";

test("popups stay below the desktop titlebar", () => {
  assert.deepEqual(clearOfWindowChrome(16, 34), {
    top: 50,
    right: 16,
    bottom: 16,
    left: 16,
  });
  assert.deepEqual(clearOfWindowChrome(undefined, 34), {
    top: 34,
    right: 0,
    bottom: 0,
    left: 0,
  });
  assert.deepEqual(clearOfWindowChrome({ top: 8, left: 4 }, 34), {
    top: 42,
    left: 4,
  });
});

test("a browser keeps the caller's padding untouched", () => {
  assert.equal(clearOfWindowChrome(16, 0), 16);
  assert.equal(clearOfWindowChrome(undefined, 0), undefined);
  const padding = { top: 8 };
  assert.equal(clearOfWindowChrome(padding, 0), padding);
});

test("every floating primitive clears the titlebar", async () => {
  const uses = {
    "dropdown-menu": 2,
    "context-menu": 2,
    menubar: 2,
    popover: 1,
    select: 1,
    "hover-card": 1,
    combobox: 1,
  };
  for (const [file, count] of Object.entries(uses)) {
    const source = await readSrcAsync(`components/ui/${file}.tsx`);
    assert.equal(
      source.match(/collisionPadding=\{useWindowChromeCollisionPadding\(/g)
        ?.length,
      count,
      file,
    );
  }
});

test("the message More menu clears the titlebar", async () => {
  const thread = await readSrcAsync("components/assistant-ui/thread.tsx");
  assert.match(
    thread,
    /const moreMenuCollisionPadding = useWindowChromeCollisionPadding\(undefined\);/,
  );
  assert.match(
    thread,
    /<ActionBarMorePrimitive\.Content[^>]*collisionPadding=\{moreMenuCollisionPadding\}/,
  );
  // Padding alone can't stop a flipped menu growing under the titlebar.
  assert.match(
    thread,
    /aui-action-bar-more-content[^"]*max-h-\(--radix-dropdown-menu-content-available-height\)[^"]*overflow-hidden/,
  );
});

test("the titlebar height is re-read whenever it can change", async () => {
  const provider = await readSrcAsync("app/provider.tsx");
  assert.equal(provider.match(/refreshWindowChromeTop\(\);/g)?.length, 2);
  assert.match(
    provider,
    /subscribeAppliedInterfaceZoom\(refreshWindowChromeTop\)/,
  );
});

test("popovers stop at the titlebar and scroll", async () => {
  const popover = await readSrcAsync("components/ui/popover.tsx");
  assert.match(
    popover,
    /max-h-\(--radix-popover-content-available-height\) overflow-y-auto/,
  );
});

test("dropdown submenus stop at the titlebar and scroll", async () => {
  const menu = await readSrcAsync("components/ui/dropdown-menu.tsx");
  const sub = menu.slice(menu.indexOf("function DropdownMenuSubContent"));
  assert.match(
    sub,
    /max-h-\(--radix-dropdown-menu-content-available-height\)[^"]*flex flex-col overflow-hidden/,
  );
  assert.match(
    sub,
    /data-slot="dropdown-menu-viewport"\s+className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto"\s*>\s*\{children\}/,
  );
});

test("context submenus stop at the titlebar and scroll", async () => {
  const menu = await readSrcAsync("components/ui/context-menu.tsx");
  const sub = menu.slice(menu.indexOf("function ContextMenuSubContent"));
  assert.match(
    sub,
    /max-h-\(--radix-context-menu-content-available-height\) flex flex-col overflow-hidden"/,
  );
  // An inner viewport scrolls, so the rounded surface keeps its corners.
  assert.match(
    sub,
    /data-slot="context-menu-viewport"\s*className="min-h-0 flex-1 overflow-x-hidden overflow-y-auto"/,
  );
});
