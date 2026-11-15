// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();

const { releaseNotesPreview } = await import(
  "../src/lib/release-notes-preview.ts"
);

function linkedText(markdown: string, limit?: number) {
  return releaseNotesPreview(markdown, limit).items.map((item) => ({
    lead: item.lead,
    rest: item.rest,
    links: item.links.map((link) => ({
      part: link.part,
      text: item[link.part].slice(link.start, link.end),
      href: link.href,
    })),
  }));
}

test("a web link in a bullet keeps its destination for the collapsed card", () => {
  const items = linkedText(
    "- **Qwen3.8-Flash-Next** Dynamic V3.1 - 10-15% extra accuracy [Guide](https://unsloth.ai/docs/models/qwen3.8-next)\n- Glass theme improvements",
  );
  assert.deepEqual(items[0], {
    lead: "Qwen3.8-Flash-Next Dynamic V3.1 - 10-15% extra accuracy Guide",
    rest: "",
    links: [
      {
        part: "lead",
        text: "Guide",
        href: "https://unsloth.ai/docs/models/qwen3.8-next",
      },
    ],
  });
  assert.deepEqual(items[1]?.links, []);
});

test("links across a bullet's lines stay in order", () => {
  const [item] = linkedText(
    "- Faster loads. See [**the docs**](http://unsloth.ai/docs)\n  and [#123](https://github.com/unslothai/unsloth/pull/123)",
  );
  assert.equal(item?.rest, "See the docs and #123");
  assert.deepEqual(item?.links, [
    { part: "rest", text: "the docs", href: "http://unsloth.ai/docs" },
    {
      part: "rest",
      text: "#123",
      href: "https://github.com/unslothai/unsloth/pull/123",
    },
  ]);
});

test("the link is the linked words, not an earlier match of its text", () => {
  const spans = releaseNotesPreview(
    [
      "- Studio now runs everywhere. Read more [here](https://unsloth.ai/a)",
      "- Guide mode is faster: read the [guide](https://unsloth.ai/b)",
      "- [Docs](https://unsloth.ai/c) and more [Docs](https://unsloth.ai/d)",
    ].join("\n"),
  ).items.map((item) =>
    item.links.map(({ part, start, end }) => [part, start, end]),
  );
  assert.deepEqual(spans, [
    [["rest", 10, 14]],
    [["lead", 31, 36]],
    [
      ["lead", 0, 4],
      ["lead", 14, 18],
    ],
  ]);
});

test("links keep the summary text and drop when clipped", () => {
  const long = `- ${"word ".repeat(22)}and the [docs](https://unsloth.ai/d)`;
  const [item] = linkedText(long);
  assert.ok(item?.lead.endsWith("..."));
  assert.deepEqual(item?.links, []);

  const [emphasis] = linkedText(
    "- **[Bold link](https://unsloth.ai/x)** and _[ spaced ](https://unsloth.ai/y)_ here",
  );
  assert.equal(emphasis?.lead, "Bold link and _ spaced _ here");
  assert.deepEqual(
    emphasis?.links.map((l) => l.text),
    ["Bold link", "spaced"],
  );
});

test("destinations decode character references and keep their own link", () => {
  const [entity, hidden] = linkedText(
    [
      "- [Guide](https://example.com/?a=1&amp;b=2&#x26;c=3) now",
      '- <span title="[hidden](https://a.test)">Model</span> [Guide](https://b.test)',
    ].join("\n"),
  );
  assert.deepEqual(entity?.links, [
    { part: "lead", text: "Guide", href: "https://example.com/?a=1&b=2&c=3" },
  ]);
  assert.equal(hidden?.lead, "Model Guide");
  assert.deepEqual(hidden?.links, [
    { part: "lead", text: "Guide", href: "https://b.test" },
  ]);
});

test("a link the lead split cuts through stays clickable on both sides", () => {
  const [item] = linkedText(
    "- [Faster inference. Read the guide](https://example.com/guide) today",
  );
  assert.equal(item?.lead, "Faster inference.");
  assert.deepEqual(item?.links, [
    {
      part: "lead",
      text: "Faster inference.",
      href: "https://example.com/guide",
    },
    { part: "rest", text: "Read the guide", href: "https://example.com/guide" },
  ]);
});

test("prose fallback keeps its links", () => {
  const [intro, links] = linkedText(
    "Muse Glimmer is out.\n\n[Run Muse Glimmer](https://unsloth.ai/docs/run)\n[Fine-tune it](https://unsloth.ai/docs/train)",
  );
  assert.deepEqual(intro?.links, []);
  assert.deepEqual(links?.links, [
    {
      part: "lead",
      text: "Run Muse Glimmer",
      href: "https://unsloth.ai/docs/run",
    },
    {
      part: "lead",
      text: "Fine-tune it",
      href: "https://unsloth.ai/docs/train",
    },
  ]);
});

test("only plain http and https links become clickable", () => {
  const items = linkedText(
    [
      "- Run [this](javascript:alert(1)) now",
      "- Mail [us](mailto:a@b.c) today",
      "- Read [notes](/docs/notes) here",
      '- Open [page](https://unsloth.ai "Title") later',
      "- Code `[x](https://unsloth.ai)` stays code",
      "- Escaped \\[y](https://unsloth.ai) stays text",
      "- Image ![logo](https://unsloth.ai/logo.png) hidden",
      "- Escaped [dest](https://unsloth.ai/a\\_b) stays text",
      "- Backslash [path](https://unsloth.ai/docs\\alpha.md) stays text",
      "- Encoded [path](https://unsloth.ai/docs&#92;alpha.md) stays text",
    ].join("\n"),
    10,
  );
  for (const item of items) {
    assert.deepEqual(item.links, [], item.lead);
  }
});
