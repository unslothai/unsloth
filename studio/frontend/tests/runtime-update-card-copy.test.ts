// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Read from source: the node suite has no DOM.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const LLAMA = readSrc("components/llama-update-banner.tsx");
const EN = readSrc("i18n/locales/en.ts");

test("the single-update and backend titles name no runtime", () => {
  assert.match(EN, /newUpdate: "New runtime update"/);
  assert.match(EN, /newBackend: "New runtime backend"/);
  assert.match(LLAMA, /t\("runtimeUpdate\.newBackend"\)/);
  assert.match(LLAMA, /t\("runtimeUpdate\.newUpdate"\)/);
});

test("the summary leads with the runtime and its short release tag", () => {
  const summary = LLAMA.slice(
    LLAMA.indexOf("const releaseOf"),
    LLAMA.indexOf('.join(" · ")'),
  );
  assert.match(summary, /\$\{row\.component\} \$\{row\.latestTag\.replace\(/);
  assert.equal("b12128-mix-2f23ac9".replace(/^(b\d+)-.+$/, "$1"), "b12128");
  assert.equal("v0.4.0".replace(/^(b\d+)-.+$/, "$1"), "v0.4.0");
  assert.match(summary, /^\s*releases,$/m);
  assert.match(summary, /!bundled && row\.versionChanged/);
  assert.match(summary, /bundled \? null : restartNote/);
});
