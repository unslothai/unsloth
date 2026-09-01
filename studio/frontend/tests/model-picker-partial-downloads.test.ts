// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A cancelled multi-GB download used to be invisible in the picker: filtered out of the inventory
// before any list saw it, so the one screen that could delete it never showed it. The Hub always
// listed these -- marked, and opening their download rather than a load -- and the picker now
// matches. The rule that makes that safe is that a partial is listed, never loaded.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";

function read(path: string): string {
  return readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf-8");
}

const PICKERS = read(
  "../src/features/model-picker/components/model-selector/pickers.tsx",
);
const INVENTORY = read(
  "../src/features/model-picker/inventory/use-chat-picker-inventory.ts",
);
const CHAT_ADAPTER = read("../src/features/chat/api/chat-adapter.ts");
const MODELS_TABLE = read("../src/features/hub/catalog/models-table.tsx");

test("the picker inventory lists partial snapshots instead of dropping them", () => {
  // The filter that hid them. Live downloads still go: bytes are moving and the Downloads panel
  // owns that row until they stop.
  assert.ok(!INVENTORY.includes("isCompleteCachedRow"), "old filter is gone");
  assert.match(
    INVENTORY,
    /function isListableCachedRow\(row: CachedInventoryRow\): boolean \{\n\s*return !row\.liveDownload;\n\}/,
  );
  assert.ok(
    INVENTORY.includes("isListableCachedRow(row) &&"),
    "and both cached lists use it",
  );
  assert.equal(
    INVENTORY.split("isListableCachedRow(row) &&").length - 1,
    2,
    "gguf and non-gguf alike",
  );
});

test("the flag survives the mapping, or no row downstream could tell", () => {
  // toCachedGgufRepo / toCachedModelRepo are the only things the picker sees.
  assert.equal(
    INVENTORY.split("partial: row.partial,").length - 1,
    2,
    "carried onto both cached repo shapes",
  );
});

test("a partial is marked the way the Hub marks one", () => {
  const start = PICKERS.indexOf("function PartialBadge()");
  assert.ok(start > 0, "the picker has a partial mark");
  const badge = PICKERS.slice(start, PICKERS.indexOf("\n}", start));
  assert.match(badge, /size-\[5px\] rounded-full bg-status-warning/);
  assert.match(badge, /aria-label="Partial download"/);
  assert.ok(
    MODELS_TABLE.includes('aria-label="Partial download"') &&
      MODELS_TABLE.includes("bg-status-warning"),
    "and the Hub still uses that dot, so the two agree",
  );
});

test("complete and partial are alternatives, never both dots on one row", () => {
  // The bytes are there or they are not; two dots would say both.
  assert.equal(
    PICKERS.split("{partial ? <PartialBadge /> : null}").length - 1,
    2,
    "drawn in the aligned and unaligned branches alike",
  );
  assert.equal(
    PICKERS.split(
      "{downloaded && !partial && !loaded ? <DownloadedBadge /> : null}",
    ).length - 1,
    2,
    "and the on-device dot yields to it in both",
  );
});

test("selecting a partial opens its download instead of claiming the weights", () => {
  // isDownloaded is what the load path reads. Hard-coding true on these rows sent a torn snapshot
  // straight to a load that fails on the missing shards -- the reason they were hidden at all.
  assert.ok(
    PICKERS.includes("isDownloaded: !isPartial,"),
    "the pick reports what is actually on disk",
  );
  // Hub reaches the same answer from the same field.
  assert.ok(
    read("../src/features/hub/hub-page.tsx").includes(
      "isDownloaded: !row.partial",
    ),
  );
});

test("listing a partial never makes it auto-loadable", () => {
  // The picker lists them; the background pick must still refuse them. This guard is what makes
  // widening the inventory safe, so it is not free to drift.
  assert.match(
    CHAT_ADAPTER,
    /function isChattableCachedRepo\([\s\S]*?repo\.partial !== true/,
  );
  assert.ok(
    CHAT_ADAPTER.includes("row.partial !== true"),
    "and the local scan-folder rule agrees",
  );
});

test("a partial GGUF repo carries its own menu, not an empty gutter", () => {
  // A complete GGUF repo keeps delete on the quant rows inside the expander, so its own row only
  // reserves the gutter. A partial repo has no complete quant to hold those actions, so that
  // reservation left the torn bytes visible and unreachable: no delete, no reveal.
  const start = PICKERS.indexOf("const renderDownloadedGgufRow");
  const row = PICKERS.slice(start, PICKERS.indexOf("\n  };", start));
  assert.ok(row.includes("const isPartialRepo = c.partial === true;"));
  assert.match(
    row,
    /\{isPartialRepo \? \(\n\s*<span className=\{ROW_ACTIONS_PINNED_CLASS\}>\n\s*<ModelRowMenu/,
    "the partial branch draws real buttons",
  );
  assert.match(
    row,
    /\) : \(\n\s*<span aria-hidden="true" className=\{cn\(ROW_ACTIONS_CLASS, "h-6"\)\}/,
    "and a complete repo still only reserves the gutter",
  );
  // Reveal and delete are the two the row owes; resume stays per-quant in the expander.
  assert.ok(row.includes("cachePath={{ repoId: c.repo_id }}"), "reveal");
  assert.ok(row.includes('title: "Delete partial download?"'), "delete");
});

test("a partial row keeps its buttons on screen instead of hiding them behind hover", () => {
  // Every other row hides the gutter until hover because the row itself is the action: click it
  // and the model loads. A partial cannot be loaded, so the menu is its ONLY affordance -- hidden,
  // the row reads as a stalled download with no controls at all.
  assert.ok(
    PICKERS.includes(
      'const ROW_ACTIONS_PINNED_CLASS = cn(ROW_ACTIONS_CLASS, "opacity-100");',
    ),
    "the pinned variant exists and is built from the shared one",
  );
  // Both partial rows use it: the GGUF repo row and the cached non-GGUF row.
  const gguf = PICKERS.slice(PICKERS.indexOf("const renderDownloadedGgufRow"));
  assert.ok(
    gguf
      .slice(0, gguf.indexOf("\n  };"))
      .includes("<span className={ROW_ACTIONS_PINNED_CLASS}>"),
    "GGUF partial repo row",
  );
  assert.ok(
    PICKERS.includes(
      "isPartial ? ROW_ACTIONS_PINNED_CLASS : ROW_ACTIONS_CLASS",
    ),
    "non-GGUF row pins only when the snapshot is torn",
  );
  // The base class must keep hiding itself, or every row grows a permanent button strip.
  assert.match(
    PICKERS,
    /const ROW_ACTIONS_CLASS =\n\s*"[^"]*\bopacity-0\b/,
    "the default gutter still hides",
  );
});

test("the expander still lists torn quants, which is where resume lives", () => {
  // The repo row deletes the whole partial; resuming targets one quant, so it belongs to the
  // variant rows. If this filter ever drops partials there is no resume path left anywhere.
  const vis = read(
    "../src/features/model-picker/components/model-selector/variant-visibility.ts",
  );
  assert.ok(
    vis.includes("v.downloaded === true || v.partial === true"),
    "torn quants stay listed on device",
  );
});

test("the stale reason for hiding partials is gone from the picker", () => {
  // It justified a filter that no longer exists; left in place it reads as the current rule.
  assert.ok(
    !PICKERS.includes(
      "A partially-downloaded snapshot is not on-device: listing it as loadable errors",
    ),
  );
});
