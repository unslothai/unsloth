// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Popups dim what is behind them but never blur it, including the titlebar layer that repeats the backdrop.

import assert from "node:assert/strict";
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

const SRC = fileURLToPath(new URL("../src/", import.meta.url));
const read = (path: string) => readFileSync(join(SRC, path), "utf8");
const BLUR = /backdrop-blur(?!-(?:none|0)\b)|backdrop-filter:\s*blur/;

function sources(dir: string): string[] {
  return readdirSync(join(SRC, dir)).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(join(SRC, path)).isDirectory()) return sources(path);
    return /\.tsx?$/.test(name) ? [path] : [];
  });
}

test("the shared dialog, alert dialog and sheet overlays do not blur", () => {
  for (const file of ["components/ui/dialog.tsx", "components/ui/alert-dialog.tsx", "components/ui/sheet.tsx"]) {
    assert.doesNotMatch(read(file), /backdrop-blur/, file);
  }
});

test("no dialog passes a blurring overlay", () => {
  for (const file of sources(".")) {
    for (const match of read(file).matchAll(/overlayClassName=\{?["`]([^"`]*)["`]/g)) {
      assert.doesNotMatch(match[1], BLUR, `${file}: ${match[1]}`);
    }
  }
});

test("lightboxes, the browser panel and the titlebar backdrop do not blur", () => {
  assert.doesNotMatch(read("components/media-viewer.tsx"), /media-lightbox-overlay[^"]*backdrop-blur/);
  assert.match(
    read("components/assistant-ui/thread.tsx"),
    /className="pointer-events-auto absolute inset-0 bg-background\/65 dark:bg-background\/55"/,
  );
  assert.match(
    read("features/chat/chat-page.tsx"),
    /className="fixed inset-0 z-50 flex items-center justify-center bg-background\/80 sm:p-4"/,
  );
  const css = read("index.css");
  const titlebar = css.slice(css.indexOf('[data-slot="window-titlebar"]::after {'));
  assert.doesNotMatch(titlebar.slice(0, titlebar.indexOf("}")), /backdrop-filter/);
});

test("Settings dims a step darker, titlebar included", () => {
  assert.match(
    read("features/settings/settings-dialog.tsx"),
    /overlayClassName="bg-black\/40"\s*overlayTone="strong"/,
  );
  assert.match(read("components/ui/dialog.tsx"), /data-viewport-backdrop-tone=\{overlayTone\}/);
  assert.match(
    read("components/tauri/window-titlebar.tsx"),
    /data-viewport-backdrop-tone=\{backdropTone \?\? undefined\}/,
  );
  assert.match(
    read("index.css"),
    /\[data-slot="window-titlebar"\]\[data-viewport-backdrop-tone="strong"\]::after \{\s*background: rgb\(0 0 0 \/ 40%\);/,
  );
});

test("every popup surface carries the faint shadow without losing its ring", () => {
  const css = read("index.css");
  assert.match(css, /--popup-shadow: 0 2px 8px -2px rgb\(0 0 0 \/ 0\.16\);/);
  assert.match(css, /--popup-shadow: 0 2px 8px -2px rgb\(0 0 0 \/ 0\.7\);/);
  assert.match(
    css,
    /:is\(\[data-slot="dialog-content"\], \[data-slot="alert-dialog-content"\], \[data-slot="sheet-content"\]\) \{\s*box-shadow:\s*var\(--tw-ring-offset-shadow, 0 0 #0000\),\s*var\(--tw-ring-shadow, 0 0 #0000\),\s*var\(--popup-shadow\);/,
  );
});
