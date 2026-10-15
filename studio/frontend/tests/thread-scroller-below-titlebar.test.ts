// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("the thread scrolls below the custom titlebar, so its scrollbar is never painted over", () => {
  const provider = readSrc("app/provider.tsx");
  const chrome = provider.slice(provider.indexOf("const CUSTOM_CHROME_STYLE = {"));
  const titlebar = /"--studio-custom-titlebar-height": "(\d+px)",/.exec(chrome)?.[1];
  assert.ok(titlebar);
  assert.match(chrome, new RegExp(`"--studio-chat-scroller-top": "${titlebar}",`));
  assert.doesNotMatch(
    provider.slice(0, provider.indexOf("const CUSTOM_CHROME_STYLE = {")),
    /--studio-chat-scroller-top/,
    "the macOS titlebar is native and the web has none, so the scroller stays at the top there",
  );

  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(
    thread,
    /\[--thread-header-offset:calc\(var\(--studio-content-top-inset,0px\)-var\(--studio-chat-scroller-top,0px\)\+[^\]]*\)\] mt-\[var\(--studio-chat-scroller-top,0px\)\] pt-\[var\(--thread-header-offset\)\]/,
    "the content keeps its place: what the margin adds, the padding gives back",
  );

  const css = readSrc("index.css");
  const dock = css.slice(css.indexOf(".chat-full-view-dock {"), css.indexOf("}", css.indexOf(".chat-full-view-dock {")));
  assert.match(dock, /--studio-content-top-inset: 0px;/);
  assert.match(dock, /--studio-chat-scroller-top: 0px;/, "the docked thread sits below the page, not under the titlebar");
});
