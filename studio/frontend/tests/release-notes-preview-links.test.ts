// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { releaseNotesPreview } = await import(
  "../src/lib/release-notes-preview.ts"
);

test("a web link in a bullet keeps its destination for the collapsed card", () => {
  const { items } = releaseNotesPreview(
    "- **Qwen3.8-Flash-Next** Dynamic V3.1 - 10-15% extra accuracy [Guide](https://unsloth.ai/docs/models/qwen3.8-next)\n- Glass theme improvements",
  );
  assert.deepEqual(items[0], {
    lead: "Qwen3.8-Flash-Next Dynamic V3.1 - 10-15% extra accuracy Guide",
    rest: "",
    links: [
      { text: "Guide", href: "https://unsloth.ai/docs/models/qwen3.8-next" },
    ],
  });
  assert.deepEqual(items[1]?.links, []);
});

test("links across a bullet's lines stay in order", () => {
  const { items } = releaseNotesPreview(
    "- Faster loads. See [**the docs**](http://unsloth.ai/docs)\n  and [#123](https://github.com/unslothai/unsloth/pull/123)",
  );
  assert.equal(items[0]?.rest, "See the docs and #123");
  assert.deepEqual(items[0]?.links, [
    { text: "the docs", href: "http://unsloth.ai/docs" },
    { text: "#123", href: "https://github.com/unslothai/unsloth/pull/123" },
  ]);
});

test("only plain http and https links become clickable", () => {
  const { items } = releaseNotesPreview(
    [
      "- Run [this](javascript:alert(1)) now",
      "- Mail [us](mailto:a@b.c) today",
      "- Read [notes](/docs/notes) here",
      '- Open [page](https://unsloth.ai "Title") later',
      "- Code `[x](https://unsloth.ai)` stays code",
      "- Escaped \\[y](https://unsloth.ai) stays text",
      "- Image ![logo](https://unsloth.ai/logo.png) hidden",
    ].join("\n"),
    10,
  );
  for (const item of items) {
    assert.deepEqual(item.links, [], item.lead);
  }
});
