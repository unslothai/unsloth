// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { loadWithStubs } from "./helpers/module-stubs.ts";

type Module = {
  hasIncompleteLinkRepair: (source: string, repaired?: string) => boolean;
  repairStreamingMarkdown: (source: string) => string;
};

function harness() {
  let calls = 0;
  const module = loadWithStubs<Module>(
    new URL("../src/components/assistant-ui/streaming-render-schedule.ts", import.meta.url),
    {
      remend: {
        default: (source: string) => {
          calls += 1;
          const open = source.lastIndexOf("[");
          return open !== -1 && !source.slice(open).includes("]")
            ? `${source}](streamdown:incomplete-link)`
            : source;
        },
      },
      streamdown: {},
      "../../lib/parse-markdown-blocks.ts": { parseMarkdownIntoBlocks: (s: string) => [s] },
    },
  );
  return { module, remendCalls: () => calls };
}

test("a settled body pays for the remend pass once per distinct text", () => {
  const { module, remendCalls } = harness();
  assert.equal(module.hasIncompleteLinkRepair("See [example"), true);
  assert.equal(module.hasIncompleteLinkRepair("See [example"), true);
  assert.equal(module.hasIncompleteLinkRepair(["See ", "[example"].join("")), true);
  assert.equal(remendCalls(), 1);
  assert.equal(module.hasIncompleteLinkRepair("See [example](https://example.com)"), false);
  assert.equal(module.hasIncompleteLinkRepair("See [example](https://example.com)"), false);
  assert.equal(remendCalls(), 2);
});

test("a bracket-free body and the two-argument streaming form never touch remend or the memo", () => {
  const { module, remendCalls } = harness();
  assert.equal(module.hasIncompleteLinkRepair("plain prose"), false);
  assert.equal(module.hasIncompleteLinkRepair("tail [x", "tail [x](streamdown:incomplete-link)"), true);
  assert.equal(module.hasIncompleteLinkRepair("tail [x", "tail [x"), false);
  assert.equal(remendCalls(), 0);
  assert.equal(module.hasIncompleteLinkRepair("tail [x"), true);
  assert.equal(remendCalls(), 1);
});

test("repairStreamingMarkdown does not fill the memo with streaming prefixes", () => {
  const { module, remendCalls } = harness();
  module.repairStreamingMarkdown("See [exa");
  const before = remendCalls();
  module.hasIncompleteLinkRepair("See [exa");
  assert.equal(remendCalls(), before + 1);
});

test("the memo keeps at most 256 bodies, dropping the oldest unrefreshed one", () => {
  const { module, remendCalls } = harness();
  module.hasIncompleteLinkRepair("[first");
  for (let i = 0; i < 255; i += 1) module.hasIncompleteLinkRepair(`[filler ${i}`);
  module.hasIncompleteLinkRepair("[first");
  module.hasIncompleteLinkRepair("[one more");
  const before = remendCalls();
  module.hasIncompleteLinkRepair("[first");
  assert.equal(remendCalls(), before);
  module.hasIncompleteLinkRepair("[filler 0");
  assert.equal(remendCalls(), before + 1);
});

test("the memo holds at most 4 Mi characters of text and never retains an oversized body", () => {
  const { module, remendCalls } = harness();
  const big = `[${"x".repeat(3_000_000)}`;
  const other = `[${"y".repeat(2_000_000)}`;
  module.hasIncompleteLinkRepair(big);
  module.hasIncompleteLinkRepair(other);
  const before = remendCalls();
  module.hasIncompleteLinkRepair(other);
  assert.equal(remendCalls(), before);
  module.hasIncompleteLinkRepair(big);
  assert.equal(remendCalls(), before + 1);
  const huge = `[${"z".repeat(4_194_304)}`;
  module.hasIncompleteLinkRepair(huge);
  module.hasIncompleteLinkRepair(huge);
  assert.equal(remendCalls(), before + 3);
});
