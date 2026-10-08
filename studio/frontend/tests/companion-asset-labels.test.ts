// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Companion downloads say why they are needed; an on-disk quant shows only what Run still fetches.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("companion rows in the downloads panel say they are a one-time requirement", () => {
  const panel = readSrc(
    "features/hub/download-manager/download-manager-panel.tsx",
  );
  assert.match(panel, /Downloaded once, shared across compatible variants/);
  assert.match(
    panel,
    /: isRequiredAssetJob\(job\) && job\.checkpoint === false \? \(/,
  );
  assert.match(panel, /isRequiredAssetJob\(job\)\s*\? assetLabel\(/);
  assert.match(
    panel,
    /entry\.checkpoint !== false \? "Queued" : `Queued · \$\{REQUIRED_ASSET_NOTE\}`/,
  );
});

test("a downloaded quant with missing assets shows what Run still fetches", () => {
  const card = readSrc("features/hub/catalog/gguf-download-card.tsx");
  assert.match(card, /`\$\{companion\} more to run`/);
  assert.match(card, /downloaded=\{item\.downloaded\}/);
  assert.match(card, /downloaded=\{Boolean\(selected\.downloaded\)\}/);
  assert.match(
    card,
    /data-model-needs-required-assets=\{downloaded \|\| undefined\}/,
  );
});
