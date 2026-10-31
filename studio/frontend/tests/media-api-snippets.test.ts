// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The Image and video API card's copyable examples must not run anything a model name carries.

import assert from "node:assert/strict";
import test from "node:test";
import { buildMediaApiSnippet } from "../src/features/settings/components/media-api-snippets.ts";

const HOSTILE = "$(calc)";

for (const kind of ["images", "video"] as const) {
  test(`${kind}: the PowerShell body is a literal here-string, so $(...) in a model id stays text`, () => {
    const snippet = buildMediaApiSnippet(kind, {
      base: "http://127.0.0.1:8888",
      apiKey: "sk-unsloth-YOUR_KEY",
      model: HOSTILE,
      lang: "curl",
      os: "windows",
    });
    assert.match(snippet, /\$body = @'\n/);
    assert.match(snippet, /\n'@\n/);
    assert.doesNotMatch(snippet, /@"/);
    assert.ok(snippet.includes(`"model": "${HOSTILE}"`));
  });

  test(`${kind}: the bash body stays inside single quotes`, () => {
    const snippet = buildMediaApiSnippet(kind, {
      base: "http://127.0.0.1:8888",
      apiKey: "sk-unsloth-YOUR_KEY",
      model: "it's",
      lang: "curl",
      os: "unix",
    });
    assert.ok(snippet.includes(`"model": "it'\\''s"`));
  });
}
