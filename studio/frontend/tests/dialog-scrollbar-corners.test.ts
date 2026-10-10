// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("scroll-rounded insets the track in Chromium and WebKit and clips in Firefox", () => {
  const css = readSrc("index.css");
  const targets = String.raw`:is\(\.scroll-rounded, \.hub-readme-prose pre\)`;
  for (const radius of ["sm", "md", "lg", "xl", "2xl", "3xl", "4xl"]) {
    assert.ok(
      css.includes(`.scroll-rounded.rounded-${radius} { --scroll-radius: var(--radius-${radius}); }`),
      radius,
    );
  }
  assert.match(
    css,
    new RegExp(`${targets}::-webkit-scrollbar-track:vertical \\{\\s*margin-block: var\\(--scroll-radius, 0px\\);`),
  );
  assert.match(
    css,
    new RegExp(`${targets}::-webkit-scrollbar-track:horizontal \\{\\s*margin-inline: var\\(--scroll-radius, 0px\\);`),
  );
  assert.match(
    css,
    new RegExp(
      String.raw`@supports selector\(::-webkit-scrollbar\) \{\s*` +
        `${targets} \\{\\s*scrollbar-width: auto;\\s*scrollbar-color: auto;`,
    ),
  );
  assert.match(
    css,
    new RegExp(
      String.raw`@supports \(-moz-appearance: none\) \{\s*` +
        String.raw`:is\(\.scroll-rounded:not\(\.overflow-hidden\), \.hub-readme-prose pre\) \{\s*` +
        String.raw`clip-path: inset\(-1px round calc\(var\(--scroll-radius, 0px\) \+ 1px\)\);`,
    ),
  );
  assert.doesNotMatch(css, /data-overflow-watched|data-overflowing/);
  assert.ok(css.includes(String.raw`.scroll-rounded.max-sm\:rounded-none { --scroll-radius: 0px; }`));
});

test("dialogs and inline rounded scrollers use it", () => {
  for (const [file, needle] of [
    ["components/ui/dialog.tsx", "overflow-y-auto scroll-rounded rounded-4xl"],
    ["components/ui/alert-dialog.tsx", "overflow-y-auto scroll-rounded rounded-4xl"],
    ["features/settings/tabs/debugging-tab.tsx", "scroll-rounded rounded-xl"],
    ["features/api-monitor/api-monitor-page.tsx", "scroll-rounded rounded-lg"],
    ["features/export/components/export-run-panel.tsx", "scroll-rounded rounded-lg"],
    ["features/chat/components/research-activity-panel.tsx", "scroll-rounded rounded-xl"],
    ["features/rag/components/knowledge-base-dialog.tsx", "scroll-rounded rounded-md border"],
    ["components/tauri/log-details.tsx", "scroll-rounded rounded-lg"],
  ] as const) {
    assert.ok(readSrc(file).includes(needle), `${file}: ${needle}`);
  }
});

test("command dialogs keep their shadow in Firefox", () => {
  const command = readSrc("components/ui/command.tsx");
  assert.match(command, /rounded-4xl! max-sm:rounded-none! top-1\/3 translate-y-0 overflow-hidden p-0/);
  const search = readSrc("features/chat/components/chat-search-dialog.tsx");
  assert.match(search, /<CommandDialog[\s\S]*className="chat-search-surface /);
  assert.doesNotMatch(search, /className="chat-search-surface [^"]*overflow-(y-)?(auto|scroll)/);
});

test("recipe dialogs scroll an inner viewport, keeping their shadow", () => {
  const shared = readSrc("features/recipe-studio/dialogs/shared/recipe-dialog-content.tsx");
  assert.match(shared, /overlayClassName="bg-transparent"/);
  assert.match(shared, /"corner-squircle flex max-h-\[[^"]*\] flex-col overflow-hidden p-0 sm:max-w-2xl shadow-border"/);
  assert.match(shared, /"grid min-h-0 gap-6 overflow-y-auto overflow-x-hidden scroll-rounded rounded-4xl px-7 pt-8 pb-7"/);
  for (const file of ["config-dialog", "import-dialog", "processors-dialog", "preview-dialog"]) {
    const source = readSrc(`features/recipe-studio/dialogs/${file}.tsx`);
    assert.match(source, /<RecipeDialogContent\b/, file);
    assert.doesNotMatch(source, /<DialogContent\b/, file);
  }
});

test("shadowed menus scroll an inner viewport, not their rounded surface", () => {
  const contextMenu = readSrc("components/ui/context-menu.tsx");
  assert.equal(contextMenu.match(/data-slot="context-menu-viewport"/g)?.length, 2);
  assert.doesNotMatch(contextMenu, /rounded-2xl p-1 shadow-2xl[^"]*overflow-y-auto/);
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /aui-action-bar-more-content[^"]*flex flex-col overflow-hidden rounded-\[21px\]/);
  const preview = readSrc("features/recipe-studio/dialogs/preview-dialog.tsx");
  assert.match(preview, /overflow-hidden rounded-2xl border border-destructive\/30 bg-destructive\/5 py-3 shadow-border">\s*<div className="max-h-38 space-y-2 overflow-y-auto px-4 py-1">/);
  assert.doesNotMatch(preview, /scroll-rounded rounded-2xl border border-destructive/);
  const presets = readSrc("features/generation-presets/media-generation-preset-control.tsx");
  assert.match(presets, /gap-0 overflow-hidden rounded-xl border-border\/70 p-0 shadow-xl/);
  assert.match(presets, /max-h-48 min-h-0 overflow-y-auto overscroll-contain p-2/);
});
