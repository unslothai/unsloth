// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("chat code is 12px at any width, and the body's text-sm does not override it", () => {
  const css = readSrc("index.css");
  assert.match(
    css,
    /\.aui-thread-root \[data-streamdown="code-block"\] \{\s*font-size: calc\(0\.75rem \* var\(--ui-font-scale, 1\)\);/,
  );
  assert.match(readSrc("components/assistant-ui/code-fence-defer.tsx"), /p-4 text-sm/);
  assert.match(
    css,
    /\.aui-thread-root \[data-streamdown="code-block-body"\] \{\s*font-size: inherit;\s*line-height: inherit;\s*\}/,
  );
  assert.match(readSrc("features/settings/stores/appearance-custom-store.ts"), /CODE_FONT_SIZE_RANGE = \{[^}]*default: 12 \}/);
});
