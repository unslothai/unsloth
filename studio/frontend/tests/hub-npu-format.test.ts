// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  matchesFormat,
  npuLinkNeedsFormat,
} from "../src/features/hub/lib/format-filters.ts";
import { readSrcAsync } from "./helpers/kit.ts";

test("the NPU format holds no Hub repo or scanned folder: those models come from Lemonade", () => {
  for (const format of [
    true,
    false,
    "gguf",
    "safetensors",
    "checkpoint",
    "mlx",
    "adapter",
    "unknown",
    null,
  ] as const) {
    assert.equal(matchesFormat(format, "npu"), false, String(format));
  }
});

test("an ?npu= link switches the format filter to NPU only when the NPU list can show it", () => {
  const base = {
    npu: "qwen3-0.6b-FLM",
    formatFilter: "recommended",
    modelsMode: true,
    npuSupported: true,
  } as const;
  assert.equal(npuLinkNeedsFormat(base), true);
  assert.equal(npuLinkNeedsFormat({ ...base, formatFilter: "all" }), true);
  assert.equal(npuLinkNeedsFormat({ ...base, formatFilter: "npu" }), false);
  assert.equal(npuLinkNeedsFormat({ ...base, npu: null }), false);
  assert.equal(npuLinkNeedsFormat({ ...base, modelsMode: false }), false);
  assert.equal(npuLinkNeedsFormat({ ...base, npuSupported: false }), false);
});

test("the Hub switches to NPU on the link's tab before the clearing effect can drop ?npu", async () => {
  const page = await readSrcAsync("features/hub/hub-page.tsx");
  const effects = page.slice(page.indexOf("const npuLinkPending"));
  // setFormatFilter picks Discover or On Device from the current tab.
  assert.match(effects, /if \(npuLinkPending\) setFormatFilter\("npu"\);/);
  assert.match(effects, /if \(\s*!urlNpu \|\|\s*npuLinkPending \|\|/);
  // Closing the pane forgets the link, so the same link followed again switches again.
  assert.match(effects, /if \(!urlNpu\) setNpuLinkShown\(null\);/);
  // An NPU runtime still in setup keeps the link until the list can resolve it.
  assert.match(
    page,
    /const npuListing =\s*showNpuCatalog && \(!npuCatalog\?\.ready \|\| npuModels === null\);/,
  );
});
